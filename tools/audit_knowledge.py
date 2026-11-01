"""Validate the curated college knowledge base before a demo or release.

Run from the project root:

    python -B -m tools.audit_knowledge
    python -B -m tools.audit_knowledge --knowledge data/college_knowledge.json

The audit is deliberately dependency-free.  It checks structure, provenance,
uniqueness, and a small set of paraphrase probes so a curation mistake cannot
silently make common questions stop matching.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:  # Support `python tools/audit_knowledge.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_provider import filter_verified_knowledge, match_verified_fact  # noqa: E402

ALLOWED_SOURCE_TYPES = {
    "user_attachment",
    "assistant_help",
    "answer_policy",
    "official_website",
    "project_document",
}
REQUIRED_PROVENANCE = {"reviewed_on", "reviewed_by", "method", "limitations"}
MIN_ALIASES = 2
MAX_ALIAS_CHARS = 160

# These mirror the failure modes found during review: paraphrases that common
# visitors use but that do not share enough literal tokens with the FAQ text.
PROBES: tuple[tuple[str, str], ...] = (
    ("Who is the principal?", "Who is the current principal of MCE?"),
    ("What is the college address?", "What is the address of MCE?"),
    ("Does the college have a hostel?", "Does MCE have hostels?"),
    ("Is the college autonomous?", "Is MCE an autonomous institution?"),
    ("What courses are offered?", "What programs does MCE offer?"),
    ("Where is the college located?", "Where is Malnad College of Engineering located?"),
    ("Who is the HOD of CSE?", "Who is the HOD of Computer Science and Engineering at MCE?"),
    ("Who is the HOD of ECE?", "Who is the HOD of Electronics and Communication Engineering at MCE?"),
    ("Does MCE have a library?", "Does MCE have a library?"),
    ("Does MCE have labs?", "Does MCE have laboratories?"),
    ("How many students study at MCE?", "How many students are there in MCE?"),
    ("Does MCE offer a PhD?", "Does MCE offer Ph.D. programs?"),
    ("Is MCE NBA accredited?", "Is MCE accredited by NBA?"),
    ("Which companies recruit at MCE?", "Which companies are mentioned by MCE in relation to campus recruitment?"),
    ("Why choose MCE?", "What makes MCE suitable for engineering education?"),
)


def normalise(value: str) -> str:
    """Match the matcher's normalisation so uniqueness checks are meaningful."""

    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _is_blank(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def _is_placeholder(value: str) -> bool:
    lowered = " ".join(value.split()).casefold()
    return bool(
        re.match(
            r"^(?:todo|tbd|pending|replace|fill in|insert|n/?a|not available|not provided)\b",
            lowered,
        )
        or (value.strip().startswith("[") and value.strip().endswith("]"))
        or (value.strip().startswith("<") and value.strip().endswith(">"))
    )


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def _check_top_level(data: Any, audit: Audit) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        audit.error("top level must be a JSON object")
        return []
    schema = data.get("schema_version")
    if not isinstance(schema, int) or schema < 2:
        audit.error(f"schema_version must be an integer >= 2, got {schema!r}")

    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        audit.error("missing top-level provenance object")
    else:
        missing = REQUIRED_PROVENANCE - set(provenance)
        if missing:
            audit.error(
                "provenance is missing: " + ", ".join(sorted(missing))
            )
        reviewed_on = _parse_date(provenance.get("reviewed_on"))
        if reviewed_on is None:
            audit.error("provenance.reviewed_on must be an ISO date")
        elif reviewed_on > date.today():
            audit.error("provenance.reviewed_on is in the future")

    facts = data.get("facts")
    if not isinstance(facts, list) or not facts:
        audit.error("facts must be a non-empty list")
        return []

    templates = data.get("college_fact_templates", [])
    if not isinstance(templates, list):
        audit.error("college_fact_templates must be a list")
        templates = []
    for index, template in enumerate(templates):
        if not isinstance(template, dict):
            audit.error(f"template {index} is not an object")
            continue
        if template.get("verified") is not False:
            audit.error(f"template {index} must stay unverified")
        if _is_blank(template.get("question")):
            audit.error(f"template {index} needs a question")
    return facts


def _check_fact_shape(
    index: int,
    fact: Any,
    today: date,
    aliases: Counter[str],
    questions: Counter[str],
    audit: Audit,
) -> dict[str, Any] | None:
    label = f"fact {index}"
    if not isinstance(fact, dict):
        audit.error(f"{label} is not an object")
        return None

    question = fact.get("question")
    answer = fact.get("answer")
    source = fact.get("source")
    if _is_blank(question):
        audit.error(f"{label} has no question")
        return None
    label = f"fact {index} ({str(question)[:70]})"
    if fact.get("verified") is not True:
        audit.error(f"{label} must set verified to the boolean true")
    for field_name, value in (("answer", answer), ("source", source)):
        if _is_blank(value):
            audit.error(f"{label} has an empty {field_name}")
        elif _is_placeholder(str(value)):
            audit.error(f"{label} has placeholder {field_name}")
    if not _is_blank(str(answer)) and len(str(answer)) > 1200:
        audit.warn(f"{label} answer is longer than 1200 characters")

    source_type = fact.get("source_type")
    if source_type not in ALLOWED_SOURCE_TYPES:
        audit.error(
            f"{label} source_type must be one of "
            + ", ".join(sorted(ALLOWED_SOURCE_TYPES))
        )
    verified_on = _parse_date(fact.get("verified_on"))
    if verified_on is None:
        audit.error(f"{label} needs an ISO verified_on date")
    elif verified_on > today:
        audit.error(f"{label} verified_on is in the future")
    if _is_blank(fact.get("reviewed_by")):
        audit.error(f"{label} needs reviewed_by")

    question_key = normalise(str(question))
    if question_key:
        questions[question_key] += 1
    else:
        audit.error(f"{label} question normalises to an empty string")

    alias_list = _as_list(fact.get("aliases"))
    if not alias_list:
        audit.warn(f"{label} has no aliases")
    elif len(alias_list) < MIN_ALIASES:
        audit.warn(f"{label} has only one alias")
    for alias in alias_list:
        if not isinstance(alias, str) or not alias.strip():
            audit.error(f"{label} has a non-string or empty alias")
            continue
        alias = " ".join(alias.split())
        if len(alias) > MAX_ALIAS_CHARS:
            audit.error(f"{label} alias is longer than {MAX_ALIAS_CHARS} characters")
        key = normalise(alias)
        if not key:
            audit.error(f"{label} alias {alias!r} normalises to an empty string")
        elif key == question_key:
            audit.error(f"{label} repeats its own question as an alias: {alias!r}")
        else:
            aliases[key] += 1

    keywords = fact.get("keywords", [])
    if not isinstance(keywords, list):
        audit.error(f"{label} keywords must be a list")
    return fact


def _check_loaded_facts(
    facts: list[dict[str, Any]], audit: Audit
) -> list[dict[str, Any]]:
    loaded = filter_verified_knowledge(facts)
    if len(loaded) != len(facts):
        audit.error(
            f"loader accepts {len(loaded)} of {len(facts)} facts; "
            "one or more entries are incomplete or use invalid booleans"
        )
    return loaded


def _check_probes(loaded: list[dict[str, Any]], audit: Audit) -> None:
    if not loaded:
        return
    passed = 0
    for question, expected in PROBES:
        matched = match_verified_fact(question, loaded)
        if matched is None:
            audit.error(f"probe did not match: {question!r}")
            continue
        if matched.get("question") != expected:
            audit.error(
                f"probe {question!r} matched {matched.get('question')!r}, "
                f"expected {expected!r}"
            )
            continue
        passed += 1
    audit.note(f"paraphrase probes: {passed}/{len(PROBES)} passed")


def run_audit(path: Path) -> Audit:
    audit = Audit()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        audit.error(f"could not read {path}: {error}")
        return audit

    facts = _check_top_level(data, audit)
    today = date.today()
    aliases: Counter[str] = Counter()
    questions: Counter[str] = Counter()
    checked: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        checked_fact = _check_fact_shape(
            index, fact, today, aliases, questions, audit
        )
        if checked_fact is not None:
            checked.append(checked_fact)

    for key, count in questions.items():
        if count > 1:
            audit.error(f"duplicate question after normalisation: {key!r} ({count}x)")
    for key, count in aliases.items():
        if count > 1:
            audit.error(f"alias is used by more than one fact: {key!r} ({count}x)")

    loaded = _check_loaded_facts(checked, audit)
    _check_probes(loaded, audit)
    audit.note(f"facts: {len(facts)}")
    audit.note(f"unique questions: {len(questions)}")
    audit.note(f"unique aliases: {len(aliases)}")
    audit.note(f"verified facts accepted by loader: {len(loaded)}")
    return audit


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the college knowledge base")
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("data/college_knowledge.json"),
        help="path to the knowledge JSON file",
    )
    args = parser.parse_args(argv)

    audit = run_audit(args.knowledge)
    print(f"Knowledge audit: {args.knowledge}")
    for note in audit.notes:
        print(f"  {note}")
    for warning in audit.warnings:
        print(f"  WARN  {warning}")
    for error in audit.errors:
        print(f"  ERROR {error}")

    if audit.errors:
        print(f"\nFAILED: {len(audit.errors)} error(s), {len(audit.warnings)} warning(s).")
        return 1
    print(f"\nPASSED with {len(audit.warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
