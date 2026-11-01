"""Dependency-free semantic retrieval for the verified college FAQ base.

``ai_provider.py`` ships a conservative lexical matcher: it must not answer
the wrong fact, so it fails closed on anything ambiguous. That is the correct
safety posture, but it also means common rephrasings and small typos fall
through to the configured assistant even when the answer is sitting in the
local knowledge file.

This module adds a second, complementary retrieval layer that is still
dependency-free (standard library only) so it runs in CI and on any machine
without TensorFlow, NumPy, or a network call. It scores each candidate fact
against a question using:

- exact question/alias match (maximum confidence),
- token overlap weighted by inverse document frequency (IDF), so rare,
  specific words like ``hod`` or ``principal`` count more than ``college``
  or ``mce``,
- a small typo-tolerant prefix token match for short tokens,
- coverage of the query's own meaningful tokens.

Nothing here decides truth. It only produces a confidence score and a ranked
ordering; the caller (``ai_provider.py``) keeps the final veto over whether a
score is strong *and* unambiguous enough to answer as ``verified_faq``, or
whether the question must fall through to the model / web search.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from math import log
from typing import Any, Iterable

# Generic terms are common across many facts, so a hit on only these words is
# weak evidence that a particular fact was intended.
_GENERIC_TERMS = {
    "college",
    "current",
    "department",
    "departments",
    "hod",
    "information",
    "listed",
    "mce",
    "malnad",
    "official",
    "program",
    "programs",
    "programme",
    "programmes",
    "university",
    "student",
    "students",
    "campus",
    "office",
    "contact",
    "details",
}

_STOP_WORDS = {
    "a", "about", "am", "an", "and", "are", "as", "at", "be", "can", "could",
    "did", "do", "does", "for", "from", "get", "give", "good", "hello", "hey",
    "hi", "how", "i", "im", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "please", "tell", "that", "the", "this", "to", "want", "we",
    "what", "when", "where", "which", "who", "why", "will", "would", "you",
    "your",
}

# Plural / synonym normalisation so "hostels" and "hostel" are one token.
_CANONICAL = {
    "addresses": "address",
    "branches": "branch",
    "courses": "course",
    "departments": "department",
    "emails": "email",
    "facilities": "facility",
    "fees": "fee",
    "hostels": "hostel",
    "hours": "hour",
    "labs": "lab",
    "laboratories": "lab",
    "libraries": "library",
    "placements": "placement",
    "programs": "program",
    "programmes": "program",
    "students": "student",
    "timings": "timing",
    "maths": "math",
    "phys": "physics",
}


def normalize_text(value: Any) -> str:
    """Return a punctuation-insensitive, case- and accent-folded string."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _canonical_token(token: str) -> str:
    return _CANONICAL.get(token, token)


def tokens(value: Any) -> list[str]:
    """Return the normalised, stop-word-filtered tokens of ``value``."""
    return [
        _canonical_token(token)
        for token in normalize_text(value).split()
        if token not in _STOP_WORDS and len(token) > 1
    ]


def _prefix_match(a: str, b: str, minimum: int = 4) -> bool:
    """True when one short token is a strong prefix of the other.

    This catches small typos and stem variation (``principal`` vs
    ``principle`` does *not* match; the prefixes differ at the 6th char).
    """
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < minimum:
        return False
    return long.startswith(short)


def _idf_table(facts: Iterable[dict[str, Any]]) -> Counter:
    """Term -> document frequency across the fact base (for IDF weighting)."""
    doc_freq: Counter = Counter()
    total = 0
    for item in facts:
        if not isinstance(item, dict):
            continue
        total += 1
        seen = set()
        for token in tokens(_fact_text(item)):
            if token not in seen:
                doc_freq[token] += 1
                seen.add(token)
    idf: Counter = Counter()
    for token, freq in doc_freq.items():
        idf[token] = log((total + 1) / (freq + 1)) + 1.0
    return idf


def _fact_text(item: dict[str, Any]) -> str:
    phrases = [item.get("question", "")]
    aliases = item.get("aliases") or []
    keywords = item.get("keywords") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    if isinstance(keywords, str):
        keywords = [keywords]
    return " ".join(
        [str(phrases), *(str(x) for x in aliases), *(str(x) for x in keywords)]
    )


def _exact_match(question: str, item: dict[str, Any]) -> bool:
    q = normalize_text(question)
    phrases = [normalize_text(item.get("question", ""))]
    for field in ("aliases", "keywords"):
        value = item.get(field) or []
        parts = [value] if isinstance(value, str) else value
        phrases.extend(normalize_text(x) for x in parts)
    return q in phrases


def score_fact(question: str, item: dict[str, Any], idf: Counter) -> float:
    """Score one fact against a question on a roughly 0..1 scale.

    Returns a float in ``[0, 1]``. Exact question/alias matches score ``1.0``.
    Otherwise token overlap (weighted by IDF) plus a typo-tolerant prefix
    bonus is normalised by query coverage, so a fact only scores high when it
    matches a meaningful share of the question and the matched words are
    specific rather than generic.
    """
    if not isinstance(item, dict):
        return 0.0
    if _exact_match(question, item):
        return 1.0

    query = tokens(question)
    if not query:
        return 0.0
    fact = _fact_text(item)
    fact_tokens = set(tokens(fact))
    if not fact_tokens:
        return 0.0

    weighted = 0.0
    for qtoken in query:
        if qtoken in fact_tokens:
            weighted += idf.get(qtoken, 1.0)
            continue
        # Typo / stem tolerance only for meaningful (non-generic) tokens.
        if qtoken not in _GENERIC_TERMS:
            for ftoken in fact_tokens:
                if _prefix_match(qtoken, ftoken):
                    weighted += idf.get(ftoken, 1.0) * 0.7
                    break

    if weighted <= 0.0:
        return 0.0

    coverage = len([t for t in query if t not in _GENERIC_TERMS]) / len(query)
    # A query made only of broad college terms cannot identify one FAQ.
    if coverage <= 0.0:
        return 0.0
    # Normalise against the maximum achievable weight, capped for stability.
    max_weight = sum(idf.get(t, 1.0) for t in query) or 1.0
    return min(1.0, 0.6 * (weighted / max_weight) + 0.4 * coverage)


def rank_facts(
    question: str,
    facts: Iterable[dict[str, Any]],
    top_n: int = 8,
) -> list[tuple[float, dict[str, Any]]]:
    """Return ``(confidence, fact)`` pairs, highest confidence first.

    ``facts`` should already be the verified subset produced by
    ``ai_provider.filter_verified_knowledge``.
    """
    verified = [f for f in facts if isinstance(f, dict)]
    if not verified:
        return []
    idf = _idf_table(verified)
    scored = [(score_fact(question, f, idf), f) for f in verified]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[: max(1, int(top_n))]


def best_fact_confidence(
    question: str, facts: Iterable[dict[str, Any]]
) -> tuple[float, dict[str, Any] | None]:
    """Highest confidence and its fact, or ``(0.0, None)`` when empty."""
    ranked = rank_facts(question, facts, top_n=2)
    if not ranked:
        return 0.0, None
    return ranked[0][0], ranked[0][1]


def is_unambiguous(
    question: str, facts: Iterable[dict[str, Any]], threshold: float = 0.68
) -> dict[str, Any] | None:
    """Return the fact to answer as verified, or ``None`` when ambiguous.

    A fact qualifies only when its confidence is at least ``threshold`` *and*
    it is clearly ahead of the runner-up (no near-tie). This mirrors the
    fail-closed spirit of the lexical matcher while recovering more of the
    paraphrases and typo'd questions that the strict matcher deliberately
    leaves unanswered.
    """
    ranked = rank_facts(question, facts, top_n=3)
    if not ranked:
        return None
    best_score, best_fact = ranked[0]
    if best_score < threshold:
        return None
    if len(ranked) > 1:
        second_score = ranked[1][0]
        if best_score - second_score < 0.12:
            return None
    return best_fact
