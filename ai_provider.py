"""Free/local and optional cloud answer providers.

The default provider is Ollama, which runs a model on the same computer and
does not require a paid API key. Cloud providers remain opt-in through the
``AI_PROVIDER`` setting. Verified local knowledge is deliberately checked
before any provider call.

FAQ matching and the college-specific detector below are conservative
heuristics, not a security boundary or a natural-language classifier. They
can miss a paraphrase or misclassify an unusual question. A missed college
question therefore gets the safe office/knowledge fallback, while a question
classified as general may still be sent to the selected model; the prompt
also tells that model not to invent college facts.
"""

import ipaddress
import json
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from settings import Settings
import semantic_retrieval


class AIProviderError(RuntimeError):
    """Raised when a configured answer provider cannot answer safely."""

    def __init__(self, message: str, *, code: str = "provider") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AnswerResult:
    """An answer plus privacy-safe routing metadata for callers and reports.

    ``source`` is one of ``verified_faq``, ``college_fallback``,
    ``offline_fallback``, ``provider``, or ``provider_fallback``.  It never
    contains the question text, and ``provider`` is a provider label rather
    than a credential.
    """

    text: str
    source: str
    provider: str | None = None


MAX_OUTPUT_TOKENS = 180
MAX_ANSWER_CHARS = 1600
MAX_RESPONSE_BYTES = 64 * 1024
MAX_CONTEXT_CHARS = 6000
MAX_CONTEXT_FACTS = 8
MAX_QUESTION_CHARS = 1200
RATE_LIMIT_WINDOW_SECONDS = 3600.0

_PLACEHOLDER_VALUES = {
    "n/a",
    "na",
    "none",
    "null",
    "pending",
    "tbd",
    "todo",
    "unknown",
    "not available",
    "not provided",
    "to be confirmed",
    "replace me",
    "your answer here",
}

_MATCH_STOP_WORDS = {
    "a",
    "about",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "good",
    "hello",
    "hey",
    "hi",
    "how",
    "i",
    "im",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "want",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "would",
    "you",
    "your",
}

_FUZZY_GENERIC_TERMS = {
    "college",
    "current",
    "department",
    "hod",
    "information",
    "listed",
    "mce",
    "official",
    "program",
    "programs",
}

# Conservative token normalisation for fuzzy scoring only.  This does not
# lower the required amount of evidence; it prevents trivial plural/synonym
# differences from hiding an otherwise specific match.
_CANONICAL_TOKEN_MAP = {
    "addresses": "address",
    "affiliated": "affiliation",
    "affiliation": "affiliation",
    "autonomous": "autonomy",
    "autonomy": "autonomy",
    "chairman": "head",
    "chairperson": "head",
    "course": "program",
    "courses": "program",
    "branch": "program",
    "branches": "program",
    "department": "department",
    "departments": "department",
    "email": "email",
    "emails": "email",
    "mail": "email",
    "facilities": "facility",
    "facility": "facility",
    "fees": "fee",
    "hostels": "hostel",
    "hours": "hour",
    "timing": "hour",
    "timings": "hour",
    "libraries": "library",
    "located": "location",
    "location": "location",
    "offer": "offer",
    "offered": "offer",
    "offering": "offer",
    "offers": "offer",
    "placement": "placement",
    "placements": "placement",
    "programme": "program",
    "programmes": "program",
    "programs": "program",
    "student": "student",
    "students": "student",
    "math": "mathematics",
    "maths": "mathematics",
    "mech": "mechanical",
    "civ": "civil",
    "elec": "electrical",
    "eee": "electrical",
    "chem": "chemistry",
    "phy": "physics",
    "phys": "physics",
    "auto": "automobile",
    "site": "website",
    "web": "website",
    "portal": "website",
    "call": "contact",
    "phone": "contact",
    "telephone": "contact",
    "mobile": "contact",
    "labs": "laboratory",
    "lab": "laboratory",
    "laboratories": "laboratory",
}

_GREETING_PREFIXES = (
    "good afternoon",
    "good evening",
    "good morning",
    "hello",
    "hey",
    "hi",
)

_GENERIC_MATCH_PHRASES = {
    "how",
    "how is",
    "tell me",
    "what",
    "what are",
    "what is",
    "where",
    "where is",
    "who",
    "who is",
}

_COLLEGE_SPECIFIC_PHRASES = (
    "college details",
    "college name",
    "college office",
    "official college",
    "campus location",
    "contact number",
    "contact details",
    "course fee",
    "course fees",
    "hostel fee",
    "hostel fees",
    "office hours",
    "office timing",
    "admission process",
    "admission deadline",
    "application deadline",
)

_COLLEGE_SPECIFIC_TERMS = {
    "academic",
    "academics",
    "admission",
    "admissions",
    "address",
    "attendance",
    "campus",
    "canteen",
    "college",
    "contact",
    "course",
    "courses",
    "branch",
    "branches",
    "department",
    "departments",
    "engineering",
    "exam",
    "exams",
    "faculty",
    "fee",
    "fees",
    "holiday",
    "holidays",
    "hostel",
    "hostels",
    "hod",
    "hours",
    "institute",
    "institution",
    "library",
    "location",
    "mce",
    "malnad",
    "hassan",
    "office",
    "placements",
    "placement",
    "principal",
    "program",
    "programmes",
    "programs",
    "programme",
    "registrar",
    "scholarship",
    "semester",
    "staff",
    "student",
    "students",
    "syllabus",
    "timings",
    "timetable",
    "transport",
    "tuition",
    "university",
    "facility",
    "facilities",
    "working",
    "vtu",
    "kcet",
    "comedk",
    "cet",
    "kea",
    "nba",
    "naac",
    "aicte",
    "autonomous",
    "autonomy",
    "affiliated",
    "affiliation",
    "intake",
    "seat",
    "seats",
    "website",
    "email",
    "phone",
    "sports",
    "gym",
    "lab",
    "labs",
    "laboratories",
    "civil",
    "mechanical",
    "electrical",
    "electronics",
    "csbs",
    "aiml",
    "vlsi",
    "mca",
    "phd",
    "automobile",
}

_NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

_KNOWN_PROVIDERS = {
    "ollama",
    "gemini",
    "openai",
    "groq",
    "nvidia",
    "compatible",
    "openai-compatible",
    "openai_compatible",
    "lmstudio",
    "offline",
    "none",
    "local-offline",
}


class _NoRedirectHandler(HTTPRedirectHandler):
    """Reject redirects so bearer/API-key headers cannot follow a new origin."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> Request:
        raise AIProviderError("provider redirect rejected", code="redirect")


# Keep a module-level ``urlopen`` name so tests can replace it with a local
# fake. In production this is the no-redirect opener, rather than urllib's
# default redirect-following opener.
urlopen = build_opener(_NoRedirectHandler).open


def _normalise_text(value: str) -> str:
    """Return a punctuation-insensitive form used only for local matching."""

    normalised = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalised, flags=re.UNICODE))


def _tokens(value: str) -> set[str]:
    return set(_normalise_text(value).split())


def _meaningful_tokens(value: str) -> set[str]:
    return {
        _CANONICAL_TOKEN_MAP.get(token, token)
        for token in _tokens(value)
        if len(token) > 1 and token not in _MATCH_STOP_WORDS
    }


def _is_placeholder_text(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        return True
    lowered = cleaned.casefold()
    if lowered in _PLACEHOLDER_VALUES:
        return True
    if re.match(
        r"^(?:todo|tbd|pending|replace|fill in|insert|n/?a|not available|not provided)\b",
        lowered,
    ):
        return True
    if (cleaned.startswith("[") and cleaned.endswith("]")) or (
        cleaned.startswith("<") and cleaned.endswith(">")
    ):
        return True
    return False


def _optional_phrases(item: dict[str, Any], field: str) -> list[str]:
    """Read optional aliases/keywords without trusting malformed values."""

    value = item.get(field, [])
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return []
    phrases: list[str] = []
    for candidate in values:
        if not isinstance(candidate, str) or _is_placeholder_text(candidate):
            continue
        normalised = _normalise_text(candidate)
        if normalised:
            phrases.append(normalised)
    return phrases


def filter_verified_knowledge(items: Any) -> list[dict[str, Any]]:
    """Return only explicitly verified, complete, non-placeholder FAQ facts.

    A fact must use the JSON boolean ``true`` in ``verified`` and contain
    non-empty string ``question``, ``answer``, and ``source`` fields. This
    intentionally rejects incomplete templates and unverified college data;
    callers may safely use the result as prompt context.
    """

    if isinstance(items, (str, bytes, dict)) or items is None:
        return []
    try:
        entries = iter(items)
    except TypeError:
        return []
    verified: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict) or item.get("verified") is not True:
            continue
        question = item.get("question")
        answer = item.get("answer")
        source = item.get("source")
        if any(_is_placeholder_text(value) for value in (question, answer, source)):
            continue
        clean_item = dict(item)
        clean_item["question"] = " ".join(question.split())
        clean_item["answer"] = " ".join(answer.split())
        clean_item["source"] = " ".join(source.split())
        verified.append(clean_item)
    return verified


def load_knowledge(path: Path) -> list[dict[str, Any]]:
    """Load only explicitly verified FAQ entries from a JSON knowledge file.

    Invalid files fail closed and return an empty list. The loader does not
    attempt to infer whether a fact is true from its wording or source; the
    explicit boolean and non-empty fields are required by design.
    """

    try:
        knowledge_path = Path(path)
        if not knowledge_path.is_file():
            return []
        with knowledge_path.open("r", encoding="utf-8") as knowledge_file:
            data = json.load(knowledge_file)
    except (OSError, TypeError, ValueError):
        print("Knowledge file could not be loaded; no verified local facts are available.")
        return []

    if isinstance(data, dict):
        data = data.get("facts", [])
    if not isinstance(data, list):
        print("Knowledge file has an invalid facts list; no verified local facts are available.")
        return []
    return filter_verified_knowledge(data)


def _validate_endpoint(url: str, *, allow_loopback_http: bool = True) -> str:
    """Validate an adapter endpoint without permitting credential-bearing URLs.

    Remote endpoints must use HTTPS. Plain HTTP is accepted only for explicit
    loopback hosts used by local Ollama/LM Studio installations. Redirects are
    separately rejected by the opener below. This is URL policy, not a full
    network-origin guarantee: DNS and a local proxy can still affect routing.
    """

    if not isinstance(url, str):
        raise AIProviderError("invalid provider endpoint", code="configuration")
    cleaned_url = url.strip()
    if not cleaned_url or any(ord(character) < 32 for character in cleaned_url):
        raise AIProviderError("invalid provider endpoint", code="configuration")
    if len(cleaned_url) > 2048:
        raise AIProviderError("invalid provider endpoint", code="configuration")
    try:
        parsed = urlsplit(cleaned_url)
        hostname = parsed.hostname
    except ValueError as error:
        raise AIProviderError("invalid provider endpoint", code="configuration") from error
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise AIProviderError("invalid provider endpoint", code="configuration")
    if parsed.username is not None or parsed.password is not None:
        raise AIProviderError("provider endpoint credentials are not allowed", code="configuration")
    if parsed.fragment or parsed.query:
        raise AIProviderError("provider endpoint parameters are not allowed", code="configuration")
    try:
        port = parsed.port
    except ValueError as error:
        raise AIProviderError("invalid provider endpoint", code="configuration") from error
    if port is not None and not 1 <= port <= 65535:
        raise AIProviderError("invalid provider endpoint", code="configuration")

    host = hostname.casefold().rstrip(".")
    is_loopback = host == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
    if parsed.scheme == "http" and not (allow_loopback_http and is_loopback):
        raise AIProviderError("remote provider endpoints must use HTTPS", code="configuration")
    return cleaned_url.rstrip("/")


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    """Send one bounded JSON request without following redirects.

    Error details and response bodies are intentionally not included in the
    raised message: callers print a status line, and provider payloads can
    contain private prompts, keys, or arbitrary remote text.
    """

    endpoint = _validate_endpoint(url)
    safe_headers: dict[str, str] = {"Content-Type": "application/json"}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise AIProviderError("invalid provider headers", code="configuration")
        if any(character in name + value for character in "\r\n"):
            raise AIProviderError("invalid provider headers", code="configuration")
        safe_headers[name] = value

    encoded_payload = None
    if payload is not None:
        try:
            encoded_payload = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise AIProviderError("invalid provider request", code="configuration") from error
    request = Request(
        endpoint,
        data=encoded_payload,
        headers=safe_headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except AIProviderError:
        raise
    except HTTPError as error:
        try:
            error.close()
        except Exception:
            pass
        raise AIProviderError("provider returned an HTTP error", code="http") from error
    except (URLError, TimeoutError, OSError):
        raise AIProviderError("provider request failed", code="network") from None
    except Exception:
        raise AIProviderError("provider request failed", code="network") from None

    if not isinstance(body, (bytes, bytearray)):
        raise AIProviderError("provider returned an invalid response", code="response")
    if len(body) > MAX_RESPONSE_BYTES:
        raise AIProviderError("provider response was too large", code="response")
    try:
        decoded = json.loads(bytes(body).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise AIProviderError("provider returned invalid JSON", code="response") from None
    if not isinstance(decoded, dict):
        raise AIProviderError("provider returned an unexpected response", code="response")
    return decoded


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    return _request_json("POST", url, payload, headers, timeout)


def _get_json(
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    return _request_json("GET", url, None, headers, timeout)


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    available: bool
    detail: str


class _RateLimiter:
    """Small in-process guard against accidental cloud-provider spend."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._events: deque[float] = deque()
        self._last_attempt: float | None = None

    def acquire(
        self,
        *,
        min_interval: float,
        max_per_hour: int,
        now: float | None = None,
    ) -> None:
        current = self._clock() if now is None else float(now)
        with self._lock:
            while (
                self._events
                and self._events[0] <= current - RATE_LIMIT_WINDOW_SECONDS
            ):
                self._events.popleft()
            if max_per_hour > 0 and len(self._events) >= max_per_hour:
                raise AIProviderError(
                    "cloud provider hourly rate limit reached", code="rate_limit"
                )
            if (
                self._last_attempt is not None
                and min_interval > 0
                and current - self._last_attempt < min_interval
            ):
                wait = min_interval - (current - self._last_attempt)
                raise AIProviderError(
                    f"cloud provider rate limit reached; retry in {wait:.0f}s",
                    code="rate_limit",
                )
            self._events.append(current)
            self._last_attempt = current

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._last_attempt = None


_CLOUD_RATE_LIMITER = _RateLimiter()


def reset_rate_limits() -> None:
    """Reset the process-wide cloud limiter (used by tests and diagnostics)."""

    _CLOUD_RATE_LIMITER.reset()


def _is_loopback_url(url: str) -> bool:
    try:
        hostname = urlsplit(str(url).strip()).hostname
    except ValueError:
        return False
    if not hostname:
        return False
    host = hostname.casefold().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _enforce_cloud_rate_limit(settings: Settings) -> None:
    min_interval = float(getattr(settings, "cloud_min_interval", 0.0) or 0.0)
    max_per_hour = int(getattr(settings, "cloud_max_calls_per_hour", 0) or 0)
    if min_interval <= 0 and max_per_hour <= 0:
        return
    _CLOUD_RATE_LIMITER.acquire(
        min_interval=max(0.0, min_interval),
        max_per_hour=max(0, max_per_hour),
    )


def _fact_phrases(item: dict[str, Any]) -> list[str]:
    return [
        _normalise_text(item["question"]),
        *_optional_phrases(item, "aliases"),
    ]


def _relevance_score(question: str, item: dict[str, Any]) -> int:
    """Score one verified fact for prompt retrieval, without failing closed."""

    normalised_question = _normalise_text(question)
    if not normalised_question:
        return 0
    phrases = _fact_phrases(item)
    if normalised_question in phrases:
        return 10000

    query_terms = _meaningful_tokens(question)
    if not query_terms:
        return 0
    searchable = " ".join(
        [*phrases, *_optional_phrases(item, "keywords")]
    )
    overlap = query_terms & _meaningful_tokens(searchable)
    score = len(overlap) * 10
    for keyword in _optional_phrases(item, "keywords"):
        if f" {keyword} " in f" {normalised_question} ":
            score += max(1, len(keyword.split())) * 3
    return score


def select_relevant_facts(
    question: str,
    facts: Iterable[dict[str, Any]],
    *,
    limit: int = MAX_CONTEXT_FACTS,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    """Return a bounded, question-relevant subset of verified facts.

    Sending every college fact to a language model wastes context and hides
    the useful entries behind a 6,000-character cut.  This retrieval is
    deliberately simple and deterministic: exact question/alias matches win,
    then token overlap against questions, aliases, and keywords.
    """

    verified = filter_verified_knowledge(facts)
    if not verified:
        return []
    if not isinstance(question, str) or not question.strip():
        candidates = [(0, index, item) for index, item in enumerate(verified)]
    else:
        candidates = []
        for index, item in enumerate(verified):
            score = _relevance_score(question, item)
            if score:
                candidates.append((score, index, item))
        candidates.sort(key=lambda entry: (-entry[0], entry[1]))

    selected: list[dict[str, Any]] = []
    used_chars = 0
    for _, _, item in candidates:
        if len(selected) >= max(1, int(limit)):
            break
        line_length = (
            len(item["question"])
            + len(item["answer"])
            + len(item["source"])
            + 40
        )
        if selected and used_chars + line_length > max_chars:
            continue
        selected.append(item)
        used_chars += line_length
    return selected


def _knowledge_context(
    facts: Iterable[dict[str, Any]], question: str = ""
) -> str:
    """Build bounded prompt context from the most relevant verified facts."""

    lines: list[str] = []
    for item in select_relevant_facts(question, facts):
        question_text = item["question"]
        answer = item["answer"]
        source = item["source"]
        lines.append(
            f"Question: {question_text}\nVerified answer: {answer}\nSource: {source}"
        )
    return "\n\n".join(lines)[:MAX_CONTEXT_CHARS]


_INQUIRY_PREFIXES = (
    "can you please tell me about",
    "could you please tell me about",
    "can you please tell me the",
    "could you please tell me the",
    "can you please tell me",
    "could you please tell me",
    "can you tell me about",
    "could you tell me about",
    "can you tell me the",
    "could you tell me the",
    "can you tell me",
    "could you tell me",
    "please tell me about",
    "please tell me the",
    "please tell me",
    "tell me about the",
    "tell me about",
    "tell me the",
    "tell me",
    "do you know about",
    "do you know",
    "i want to know about",
    "i want to know",
    "what about",
    "how about",
)


def _normalise_voice_transcription(text: str) -> str:
    """Normalise common voice transcription artifacts for better FAQ matching."""
    t = str(text).lower()
    t = t.replace("em cee", "mce")
    t = t.replace("m.c.e.", "mce")
    t = t.replace("who's", "who is")
    t = t.replace("whos", "who is")
    t = t.replace("what's", "what is")
    t = t.replace("whats", "what is")
    t = t.replace("where's", "where is")
    t = t.replace("wheres", "where is")
    t = t.replace("how's", "how is")
    t = t.replace("hows", "how is")
    # Collapse spaced single-letter abbreviations: e.g. "m c e" -> "mce", "h o d" -> "hod"
    t = re.sub(r"\b([a-z](?:\s+[a-z])+)\b", lambda m: m.group(1).replace(" ", ""), t)
    return t


def _query_variants(question: str) -> list[str]:
    """Return the question, voice-normalised variants, and prefix-stripped variants."""
    normalised = _normalise_text(question)
    variants = [normalised] if normalised else []
    voice_norm = _normalise_voice_transcription(normalised)
    if voice_norm and voice_norm not in variants:
        variants.append(voice_norm)

    # Greeting prefix stripping
    for greeting in _GREETING_PREFIXES:
        if normalised == greeting:
            break
        if normalised.startswith(greeting + " "):
            remainder = normalised[len(greeting) :].strip()
            if remainder and remainder not in variants:
                variants.append(remainder)
            voice_rem = _normalise_voice_transcription(remainder)
            if voice_rem and voice_rem not in variants:
                variants.append(voice_rem)
            break

    # Inquiry prefix stripping
    for base in list(variants):
        for prefix in _INQUIRY_PREFIXES:
            if base.startswith(prefix + " "):
                stripped = base[len(prefix) :].strip()
                if stripped and stripped not in variants:
                    variants.append(stripped)
                if stripped.startswith("the "):
                    stripped_the = stripped[4:].strip()
                    if stripped_the and stripped_the not in variants:
                        variants.append(stripped_the)
                break

    return variants


def _faq_score(query: str, item: dict[str, Any]) -> int:
    phrases = [
        _normalise_text(item["question"]),
        *_optional_phrases(item, "aliases"),
    ]
    if query in phrases:
        return 10000

    query_terms = _meaningful_tokens(query)
    if not query_terms:
        return 0
    keyword_phrases = _optional_phrases(item, "keywords")
    searchable = " ".join([*phrases, *keyword_phrases])
    item_terms = _meaningful_tokens(searchable)
    overlap = query_terms & item_terms
    specific_overlap = overlap - _FUZZY_GENERIC_TERMS
    if len(specific_overlap) < 2:
        return 0

    keyword_bonus = 0
    for keyword in keyword_phrases:
        keyword_terms = _meaningful_tokens(keyword)
        query_with_boundaries = f" {query} "
        if len(keyword_terms) >= 2 and f" {keyword} " in query_with_boundaries:
            keyword_bonus = max(keyword_bonus, len(keyword_terms) * 3)
    coverage = len(overlap) / len(query_terms)
    if coverage < 0.5 and keyword_bonus == 0:
        return 0
    return len(overlap) * 10 + keyword_bonus + int(coverage * 10)


def match_verified_fact(
    question: str, facts: Iterable[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return one unambiguous verified FAQ match, or ``None``.

    Exact questions/aliases are preferred. Fuzzy matching requires at least
    two meaningful overlapping terms and rejects ties, which avoids treating
    a generic ``what is`` as evidence for one of several facts. A greeting is
    only removed when it is a true prefix; a greeting entry cannot swallow a
    longer question.
    """

    if not isinstance(question, str):
        return None
    verified = filter_verified_knowledge(facts)
    variants = _query_variants(question[:MAX_QUESTION_CHARS])
    if not variants:
        return None

    exact_matches: list[dict[str, Any]] = []
    for item in verified:
        phrases = {
            _normalise_text(item["question"]),
            *_optional_phrases(item, "aliases"),
        }
        if any(
            variant in phrases
            and (
                variant not in _GENERIC_MATCH_PHRASES
                or variant in _GREETING_PREFIXES
            )
            for variant in variants
        ):
            exact_matches.append(item)
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None

    scored: list[tuple[int, dict[str, Any]]] = []
    for item in verified:
        score = max((_faq_score(variant, item) for variant in variants), default=0)
        if score:
            scored.append((score, item))
    if not scored:
        return None
    scored.sort(key=lambda result: result[0], reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def is_college_specific_question(question: str, college_name: str = "") -> bool:
    """Heuristically identify questions requiring college-specific evidence.

    This detector intentionally over-approximates common office/admission and
    campus terms. It cannot understand every paraphrase, so it is only one
    layer of protection; verified FAQ matching and the model prompt remain
    separate safeguards.
    """

    if not isinstance(question, str):
        return False
    normalised = _normalise_text(question[:MAX_QUESTION_CHARS])
    if not normalised:
        return False
    if any(phrase in normalised for phrase in _COLLEGE_SPECIFIC_PHRASES):
        return True
    if _tokens(normalised) & _COLLEGE_SPECIFIC_TERMS:
        return True
    configured_name = _meaningful_tokens(college_name)
    return bool(configured_name and len(configured_name & _tokens(normalised)) >= 2)


_COLLEGE_FALLBACK = (
    "I do not have a verified answer for that college-specific question. "
    "Please check the official college office or add a verified entry to "
    "college_knowledge.json."
)

_GENERAL_OFFLINE_FALLBACK = (
    "I do not have a verified local answer for that question. "
    "Please add a verified entry to college_knowledge.json or use the selected "
    "assistant model."
)


def _bounded_answer(answer: Any) -> str:
    if not isinstance(answer, str) or not answer.strip():
        raise AIProviderError("provider returned no answer", code="response")
    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    if not cleaned:
        raise AIProviderError("provider returned no answer", code="response")
    if len(cleaned) <= MAX_ANSWER_CHARS:
        return cleaned
    shortened = cleaned[: MAX_ANSWER_CHARS - 3].rsplit(" ", 1)[0].rstrip()
    return (shortened or cleaned[: MAX_ANSWER_CHARS - 3]).rstrip(".!? ") + "..."


def _offline_answer_result(
    question: str, settings: Settings, facts: list[dict[str, Any]]
) -> AnswerResult:
    """Answer from verified facts or a safe fallback, with route metadata."""

    matched = match_verified_fact(question, facts)
    if matched is not None:
        return AnswerResult(
            _bounded_answer(matched["answer"]),
            "verified_faq",
        )
    college_name = getattr(settings, "college_name", "")
    if is_college_specific_question(question, college_name):
        relevant = select_relevant_facts(question, facts)
        if relevant:
            return AnswerResult(
                _bounded_answer(relevant[0]["answer"]),
                "verified_faq",
            )
        return AnswerResult(_COLLEGE_FALLBACK, "college_fallback")
    return AnswerResult(_GENERAL_OFFLINE_FALLBACK, "offline_fallback")


def _offline_answer(question: str, settings: Settings, facts: list[dict[str, Any]]) -> str:
    """Answer verified local facts without network access.

    The matching is intentionally conservative and heuristic. Unknown
    college-specific questions receive the office fallback instead of being
    answered from model memory; unknown general questions receive a generic
    offline response.
    """

    return _offline_answer_result(question, settings, facts).text


def _system_prompt(
    question: str, settings: Settings, facts: list[dict[str, Any]], web_context: str = ""
) -> str:
    """Create a short, bounded prompt whose factual context is fail-closed."""

    context = _knowledge_context(facts, question)
    verified_context = context or "No relevant verified college facts were found."
    college_name = str(getattr(settings, "college_name", "the configured college"))[:200]
    return (
        f"You are a helpful voice assistant for the configured college ({college_name}). "
        "Answer in two or three short spoken sentences. Use verified context first. "
        "Never present model memory as a verified college fact. If context does not "
        "answer, answer only when confident, say the detail is unverified, and recommend "
        "an official source for current names, fees, dates, or policies. Do not claim "
        "to have searched the web unless a search tool was used. For general questions, "
        "answer helpfully and say when unsure. Do not "
        "use markdown, emojis, or long lists.\n\n"
        f"Relevant verified college context:\n{verified_context}\n\n"
        f"Web search results (unverified; prefer official sources):\n{web_context or 'No web results were available.'}"
    )


def _search_web_context(question: str, settings: Settings) -> str:
    """Fetch a few bounded search snippets; fail quickly when search is unavailable.

    This sends the question text to DuckDuckGo, so it is skipped entirely in
    offline mode and can be disabled with ``WEB_SEARCH=false``.
    """
    if not getattr(settings, "web_search", True):
        return ""
    try:
        from ddgs import DDGS

        results = DDGS(timeout=2).text(question[:MAX_QUESTION_CHARS], max_results=3)
    except Exception:
        return ""
    snippets = []
    for item in results or []:
        if isinstance(item, dict):
            title = str(item.get("title", ""))[:180]
            body = str(item.get("body", ""))[:500]
            url = str(item.get("href", ""))[:500]
            if body:
                snippets.append(f"{title}\n{body}\nSource: {url}")
    return "\n\n".join(snippets)[:2500]


def _provider_setting(settings: Settings, name: str, default: Any = "") -> Any:
    return getattr(settings, name, default)


def _ask_ollama(question: str, settings: Settings, system_prompt: str) -> str:
    """Ask a local Ollama server with a bounded generation request."""

    base_url = _validate_endpoint(str(_provider_setting(settings, "ollama_url", "")))
    model = str(_provider_setting(settings, "ollama_model", "")).strip()
    if not model:
        raise AIProviderError("Ollama model is not configured", code="configuration")
    response = _post_json(
        f"{base_url}/api/chat",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question[:MAX_QUESTION_CHARS]},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": MAX_OUTPUT_TOKENS},
        },
        {},
        float(_provider_setting(settings, "ai_timeout", 15.0)),
    )
    message = response.get("message")
    answer = message.get("content") if isinstance(message, dict) else None
    return _bounded_answer(answer)


def _ask_gemini(question: str, settings: Settings, system_prompt: str) -> str:
    """Ask Gemini with the API key in a header, never in the URL."""

    api_key = str(_provider_setting(settings, "gemini_api_key", "")).strip()
    if not api_key:
        raise AIProviderError("Gemini API key is not configured", code="configuration")
    if any(character in api_key for character in "\r\n"):
        raise AIProviderError("invalid Gemini API key", code="configuration")
    model = str(_provider_setting(settings, "gemini_model", "")).strip()
    if not model:
        raise AIProviderError("Gemini model is not configured", code="configuration")
    endpoint = _validate_endpoint(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model, safe='')}:generateContent",
        allow_loopback_http=False,
    )
    response = _post_json(
        endpoint,
        {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": question[:MAX_QUESTION_CHARS]}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        },
        {"x-goog-api-key": api_key},
        float(_provider_setting(settings, "ai_timeout", 15.0)),
    )
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        raise AIProviderError("Gemini returned no answer", code="response")
    content = candidates[0].get("content")
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list):
        raise AIProviderError("Gemini returned no answer", code="response")
    text_parts = [
        part.get("text")
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return _bounded_answer(" ".join(text_parts))


def _ask_openai_http(
    question: str,
    settings: Settings,
    system_prompt: str,
    api_key: str,
    model: str,
) -> str:
    response = _post_json(
        "https://api.openai.com/v1/responses",
        {
            "model": model,
            "instructions": system_prompt,
            "input": question[:MAX_QUESTION_CHARS],
            "temperature": 0.2,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        {"Authorization": f"Bearer {api_key}"},
        float(_provider_setting(settings, "ai_timeout", 15.0)),
    )
    direct_text = response.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return _bounded_answer(direct_text)
    output = response.get("output")
    if isinstance(output, list):
        text_parts = [
            part.get("text")
            for item in output
            if isinstance(item, dict)
            for part in (item.get("content") or [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return _bounded_answer(" ".join(text_parts))
    raise AIProviderError("OpenAI returned no answer", code="response")


def _ask_openai(question: str, settings: Settings, system_prompt: str) -> str:
    """Ask OpenAI with the optional SDK or its dependency-free HTTP API.

    The SDK path is retained when installed and is configured with no retries.
    Older SDK builds without the Responses API transparently use the HTTP
    path, so a stale virtual environment cannot fail at call time.  On a
    minimal install the standard-library path avoids a mandatory dependency
    while preserving the same selected provider.
    """

    api_key = str(_provider_setting(settings, "openai_api_key", "")).strip()
    if not api_key:
        raise AIProviderError("OpenAI API key is not configured", code="configuration")
    if any(character in api_key for character in "\r\n"):
        raise AIProviderError("invalid OpenAI API key", code="configuration")
    model = str(_provider_setting(settings, "openai_model", "")).strip()
    if not model:
        raise AIProviderError("OpenAI model is not configured", code="configuration")

    try:
        from openai import OpenAI
    except ImportError:
        return _ask_openai_http(question, settings, system_prompt, api_key, model)

    client = OpenAI(
        api_key=api_key,
        timeout=float(_provider_setting(settings, "ai_timeout", 15.0)),
        max_retries=0,
    )
    responses = getattr(client, "responses", None)
    if responses is None:
        return _ask_openai_http(question, settings, system_prompt, api_key, model)
    response = responses.create(
        model=model,
        instructions=system_prompt,
        input=question[:MAX_QUESTION_CHARS],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return _bounded_answer(getattr(response, "output_text", ""))


def _ask_groq(question: str, settings: Settings, system_prompt: str) -> str:
    """Ask Groq via its OpenAI-compatible endpoint.

    Groq provides an OpenAI-compatible API at https://api.groq.com/openai/v1
    """
    api_key = str(_provider_setting(settings, "groq_api_key", "")).strip()
    if not api_key:
        raise AIProviderError("Groq API key is not configured", code="configuration")
    if any(character in api_key for character in "\r\n"):
        raise AIProviderError("invalid Groq API key", code="configuration")
    model = str(_provider_setting(settings, "groq_model", "")).strip()
    if not model:
        raise AIProviderError("Groq model is not configured (set GROQ_MODEL)", code="configuration")

    response = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question[:MAX_QUESTION_CHARS]},
            ],
            "temperature": 0.2,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        },
        {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"espsthuthi2/{getattr(settings, 'project_version', '2.0.0')}",
        },
        float(_provider_setting(settings, "ai_timeout", 15.0)),
    )
    try:
        choice = response["choices"][0]
        message = choice["message"]
        return _bounded_answer(message.get("content", ""))
    except (AttributeError, KeyError, IndexError, TypeError):
        raise AIProviderError("Groq returned no answer", code="response") from None


def _extract_compatible_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            item.get("text")
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return " ".join(parts)
    return ""


def _ask_compatible(question: str, settings: Settings, system_prompt: str) -> str:
    """Ask an OpenAI-compatible local or HTTPS endpoint."""

    configured_url = str(_provider_setting(settings, "ai_base_url", "")).strip()
    if not configured_url:
        raise AIProviderError("compatible provider URL is not configured", code="configuration")
    base_url = _validate_endpoint(configured_url)
    endpoint = (
        base_url
        if base_url.endswith("/chat/completions")
        else f"{base_url}/chat/completions"
    )
    api_key = str(_provider_setting(settings, "ai_api_key", "")).strip()
    if any(character in api_key for character in "\r\n"):
        raise AIProviderError("invalid compatible provider API key", code="configuration")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    model = str(_provider_setting(settings, "ai_model", "")).strip()
    if not model:
        raise AIProviderError(
            "compatible provider model is not configured (set AI_MODEL)",
            code="configuration",
        )
    response = _post_json(
        endpoint,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question[:MAX_QUESTION_CHARS]},
            ],
            "temperature": 0.2,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        },
        headers,
        float(_provider_setting(settings, "ai_timeout", 15.0)),
    )
    try:
        choice = response["choices"][0]
        message = choice["message"]
        answer = _extract_compatible_content(message.get("content"))
    except (AttributeError, KeyError, IndexError, TypeError):
        raise AIProviderError("compatible provider returned no answer", code="response") from None
    return _bounded_answer(answer)


def nvidia_base_url(settings: Settings) -> str:
    """Return the validated NVIDIA NIM base URL.

    ``AI_BASE_URL`` may point at an OpenAI-compatible NVIDIA NIM deployment;
    when it is empty the hosted ``https://integrate.api.nvidia.com`` API is
    used.  Endpoint policy still applies, so plain HTTP remains loopback-only.
    """

    configured = str(_provider_setting(settings, "ai_base_url", "")).strip()
    return _validate_endpoint(configured or _NVIDIA_DEFAULT_BASE_URL)


def _ask_nvidia(question: str, settings: Settings, system_prompt: str) -> str:
    """Ask a hosted or self-hosted NVIDIA NIM model.

    NVIDIA NIM exposes an OpenAI-compatible ``/chat/completions`` endpoint at
    ``https://integrate.api.nvidia.com/v1``.  The key belongs in ``AI_API_KEY``
    and a build.nvidia.com model id (for example
    ``nvidia/nemotron-3.5-lightning-30b-a3b``) belongs in ``AI_MODEL``.
    """

    api_key = str(_provider_setting(settings, "ai_api_key", "")).strip()
    if not api_key:
        raise AIProviderError(
            "NVIDIA API key is not configured (set AI_API_KEY)",
            code="configuration",
        )
    if any(character in api_key for character in "\r\n"):
        raise AIProviderError("invalid NVIDIA API key", code="configuration")
    model = str(_provider_setting(settings, "ai_model", "")).strip()
    if not model:
        raise AIProviderError(
            "NVIDIA model is not configured (set AI_MODEL)",
            code="configuration",
        )

    base_url = nvidia_base_url(settings)
    endpoint = (
        base_url
        if base_url.endswith("/chat/completions")
        else f"{base_url}/chat/completions"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"espsthuthi2/{getattr(settings, 'project_version', '2.0.0')}",
    }
    timeout = float(_provider_setting(settings, "ai_timeout", 15.0))
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question[:MAX_QUESTION_CHARS]},
        ],
        "temperature": 0.2,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
        # Reasoning models (Nemotron family and similar) spend the completion
        # budget on an invisible reasoning channel first, which yields an empty
        # ``content`` for short kiosk answers. Disabling thinking keeps replies
        # bounded and fast. Templates that ignore the kwarg are unaffected.
        "chat_template_kwargs": {"thinking": False},
    }
    try:
        response = _post_json(endpoint, payload, headers, timeout)
    except AIProviderError as error:
        if error.code != "http":
            raise
        # One plain retry for gateways that reject the extra kwarg; timeouts
        # and network failures are never retried.
        plain = {
            name: value
            for name, value in payload.items()
            if name != "chat_template_kwargs"
        }
        response = _post_json(endpoint, plain, headers, timeout)
    try:
        choice = response["choices"][0]
        message = choice["message"]
        answer = _extract_compatible_content(message.get("content"))
    except (AttributeError, KeyError, IndexError, TypeError):
        raise AIProviderError("NVIDIA returned no answer", code="response") from None
    return _bounded_answer(answer)


def probe_provider(
    settings: Settings, *, network: bool = False
) -> ProviderStatus:
    """Return a secret-free provider readiness summary for ``--doctor``.

    Cloud credentials are checked locally.  No cloud request is made unless
    ``network=True``, and even then only a model-list request is sent.  Ollama
    on loopback is probed by default because it is local and free.
    """

    provider = str(getattr(settings, "ai_provider", "ollama") or "ollama").strip().lower()
    timeout = float(getattr(settings, "provider_probe_timeout", 3.0) or 3.0)

    if provider in {"offline", "none", "local-offline"}:
        return ProviderStatus(provider, True, "offline mode; no network provider used")
    if provider == "ollama":
        base_url = str(getattr(settings, "ollama_url", "")).strip()
        model = str(getattr(settings, "ollama_model", "")).strip()
        try:
            base_url = _validate_endpoint(base_url)
            if not model:
                return ProviderStatus(provider, False, "OLLAMA_MODEL is not configured")
            response = _get_json(f"{base_url}/api/tags", {}, timeout)
        except AIProviderError as error:
            return ProviderStatus(
                provider, False, f"Ollama is not reachable ({error})"
            )
        models = response.get("models")
        names = {
            str(entry.get("name") or entry.get("model") or "").strip()
            for entry in models or []
            if isinstance(entry, dict)
        }
        names.discard("")
        if not names:
            return ProviderStatus(
                provider, False, "Ollama is reachable but no models are installed"
            )
        if model not in names and f"{model}:latest" not in names:
            return ProviderStatus(
                provider,
                False,
                f"Ollama is reachable but model '{model}' is not pulled",
            )
        return ProviderStatus(
            provider, True, f"Ollama reachable with {len(names)} model(s)"
        )

    if provider == "gemini":
        if not str(getattr(settings, "gemini_api_key", "")).strip():
            return ProviderStatus(provider, False, "GEMINI_API_KEY is not configured")
        if not str(getattr(settings, "gemini_model", "")).strip():
            return ProviderStatus(provider, False, "GEMINI_MODEL is not configured")
        return ProviderStatus(
            provider, True, "credentials configured; network not tested"
        )

    if provider == "openai":
        if not str(getattr(settings, "openai_api_key", "")).strip():
            return ProviderStatus(provider, False, "OPENAI_API_KEY is not configured")
        if not str(getattr(settings, "openai_model", "")).strip():
            return ProviderStatus(provider, False, "OPENAI_MODEL is not configured")
        return ProviderStatus(
            provider, True, "credentials configured; network not tested"
        )

    if provider == "groq":
        if not str(getattr(settings, "groq_api_key", "")).strip():
            return ProviderStatus(provider, False, "GROQ_API_KEY is not configured")
        if not str(getattr(settings, "groq_model", "")).strip():
            return ProviderStatus(provider, False, "GROQ_MODEL is not configured")
        return ProviderStatus(
            provider, True, "credentials configured; network not tested"
        )

    if provider == "nvidia":
        api_key = str(getattr(settings, "ai_api_key", "")).strip()
        model = str(getattr(settings, "ai_model", "")).strip()
        try:
            base_url = nvidia_base_url(settings)
        except AIProviderError as error:
            return ProviderStatus(provider, False, str(error))
        if not api_key:
            return ProviderStatus(
                provider, False, "NVIDIA API key is not configured (set AI_API_KEY)"
            )
        if not model:
            return ProviderStatus(
                provider, False, "AI_MODEL is required for the NVIDIA provider"
            )
        if not network:
            return ProviderStatus(
                provider, True, "endpoint, key, and model configured; network not tested"
            )
        try:
            response = _get_json(
                f"{base_url}/models", {"Authorization": f"Bearer {api_key}"}, timeout
            )
        except AIProviderError as error:
            return ProviderStatus(provider, False, f"endpoint not reachable ({error})")
        entries = response.get("data")
        models = {
            str(entry.get("id") or "").strip()
            for entry in entries or []
            if isinstance(entry, dict)
        }
        models.discard("")
        if models and model not in models:
            return ProviderStatus(
                provider,
                False,
                f"endpoint is reachable but model '{model}' is unavailable",
            )
        return ProviderStatus(provider, True, "endpoint reachable")

    if provider in {"compatible", "openai-compatible", "openai_compatible", "lmstudio"}:
        base_url = str(getattr(settings, "ai_base_url", "")).strip()
        model = str(getattr(settings, "ai_model", "")).strip()
        try:
            base_url = _validate_endpoint(base_url)
        except AIProviderError as error:
            return ProviderStatus(provider, False, str(error))
        if not model:
            return ProviderStatus(
                provider, False, "AI_MODEL is required for a compatible provider"
            )
        if not network:
            return ProviderStatus(
                provider, True, "endpoint and model configured; network not tested"
            )
        endpoint = (
            base_url[: -len("/chat/completions")]
            if base_url.endswith("/chat/completions")
            else base_url
        )
        api_key = str(getattr(settings, "ai_api_key", "")).strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            response = _get_json(f"{endpoint}/models", headers, timeout)
        except AIProviderError as error:
            return ProviderStatus(provider, False, f"endpoint not reachable ({error})")
        entries = response.get("data")
        models = {
            str(entry.get("id") or "").strip()
            for entry in entries or []
            if isinstance(entry, dict)
        }
        models.discard("")
        if models and model not in models:
            return ProviderStatus(
                provider, False, f"endpoint is reachable but model '{model}' is unavailable"
            )
        return ProviderStatus(provider, True, "endpoint reachable")

    return ProviderStatus(provider, False, "unknown AI provider")


def _provider_needs_cloud_limit(provider: str, settings: Settings) -> bool:
    if provider in {"gemini", "openai", "groq", "nvidia"}:
        return True
    if provider in {"compatible", "openai-compatible", "openai_compatible", "lmstudio"}:
        return not _is_loopback_url(str(getattr(settings, "ai_base_url", "")))
    return False


def _error_category(error: BaseException) -> str:
    if isinstance(error, AIProviderError):
        return (
            error.code
            if error.code
            in {
                "configuration",
                "http",
                "network",
                "provider",
                "rate_limit",
                "redirect",
                "response",
                "timeout",
            }
            else "provider"
        )
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (HTTPError, URLError, OSError)):
        return "network"
    return "provider"


def _provider_label(provider: str) -> str:
    return provider if provider in _KNOWN_PROVIDERS else "configured provider"


def answer_question_result(
    question: str, settings: Settings, facts: list[dict[str, Any]]
) -> AnswerResult:
    """Return an answer plus a privacy-safe source category.

    Verified local facts are answered before provider selection, so a known
    fact never causes network traffic. Unmatched questions use the selected
    provider when one is configured; offline mode stays local. Routing is heuristic and cannot guarantee
    that every paraphrase is classified correctly; provider instructions add
    a second anti-invention layer. A provider failure never auto-switches to a
    different provider and produces only a bounded, safe offline answer.
    """

    question_text = question.strip() if isinstance(question, str) else ""
    verified_facts = filter_verified_knowledge(facts)
    if (
        not question_text
        or match_verified_fact(question_text, verified_facts) is not None
        or not _meaningful_tokens(question_text)
    ):
        return _offline_answer_result(question_text, settings, verified_facts)

    provider = str(getattr(settings, "ai_provider", "ollama") or "ollama").strip().lower()
    is_offline_provider = provider in {"offline", "none", "local-offline"}

    college_name = str(getattr(settings, "college_name", ""))
    is_college = is_college_specific_question(question_text, college_name)
    # Use the separate weighted retriever only for confident, unambiguous
    # college FAQ matches. Weak/near-tied matches must continue to the model
    # and web search instead of being presented as verified answers.
    semantic_match = (
        semantic_retrieval.is_unambiguous(question_text, verified_facts)
        if is_college
        else None
    )
    if semantic_match is not None:
        return AnswerResult(_bounded_answer(semantic_match["answer"]), "verified_faq")

    if is_offline_provider:
        return _offline_answer_result(question_text, settings, verified_facts)

    web_context = _search_web_context(question_text, settings)
    prompt = _system_prompt(question_text, settings, verified_facts, web_context)
    try:
        if _provider_needs_cloud_limit(provider, settings):
            _enforce_cloud_rate_limit(settings)
        if provider == "ollama":
            text = _ask_ollama(question_text, settings, prompt)
        elif provider == "gemini":
            text = _ask_gemini(question_text, settings, prompt)
        elif provider == "openai":
            text = _ask_openai(question_text, settings, prompt)
        elif provider == "groq":
            text = _ask_groq(question_text, settings, prompt)
        elif provider == "nvidia":
            text = _ask_nvidia(question_text, settings, prompt)
        elif provider in {
            "compatible",
            "openai-compatible",
            "openai_compatible",
            "lmstudio",
        }:
            text = _ask_compatible(question_text, settings, prompt)
        else:
            raise AIProviderError("unknown provider", code="configuration")
        return AnswerResult(_bounded_answer(text), "provider", provider)
    except Exception as error:
        print(
            f"AI provider '{_provider_label(provider)}' unavailable "
            f"({_error_category(error)}); using safe offline answer."
        )
        fallback = _offline_answer_result(question_text, settings, verified_facts)
        return AnswerResult(fallback.text, "provider_fallback", provider)


def answer_question(question: str, settings: Settings, facts: list[dict[str, Any]]) -> str:
    """Return a safe answer using verified FAQ, the selected provider, or fallback."""

    return answer_question_result(question, settings, facts).text
