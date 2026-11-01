"""Regression tests for the curated college knowledge base.

These tests protect the data file itself: every curated question and alias must
match its own fact, common paraphrases must resolve to the intended fact, and
provenance must stay complete.  They are dependency-free so they can run in CI
without TensorFlow, OpenCV, or audio packages.
"""

import json
import re
import unicodedata
import unittest
from pathlib import Path

from ai_provider import load_knowledge, match_verified_fact

PROJECT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT / "data" / "college_knowledge.json"

PARAPHRASES = {
    "Who is the principal?": "Who is the current principal of MCE?",
    "What is the college address?": "What is the address of MCE?",
    "Does the college have a hostel?": "Does MCE have hostels?",
    "Is the college autonomous?": "Is MCE an autonomous institution?",
    "What courses are offered?": "What programs does MCE offer?",
    "Where is the college located?": "Where is Malnad College of Engineering located?",
    "What is the college website?": "What is the official website of MCE?",
    "Who is the HOD of CSE?": "Who is the HOD of Computer Science and Engineering at MCE?",
    "Who is the HOD of ISE?": "Who is the HOD of Information Science and Engineering at MCE?",
    "Who is the HOD of CSBS?": "Who is the HOD of Computer Science and Business Systems at MCE?",
    "Who is the HOD of ECE?": "Who is the HOD of Electronics and Communication Engineering at MCE?",
    "How many students study at MCE?": "How many students are there in MCE?",
    "Does MCE have labs?": "Does MCE have laboratories?",
    "Which companies recruit at MCE?": (
        "Which companies are mentioned by MCE in relation to campus recruitment?"
    ),
    "Why choose MCE?": "What makes MCE suitable for engineering education?",
    "Does MCE offer a PhD?": "Does MCE offer Ph.D. programs?",
    "Is MCE NBA accredited?": "Is MCE accredited by NBA?",
    "Does MCE have VTU research centres?": "Does MCE have VTU-recognized research centers?",
}


def normalise(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


class KnowledgeRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        cls.facts = load_knowledge(KNOWLEDGE_PATH)

    def test_every_curated_fact_loads(self):
        self.assertGreaterEqual(len(self.facts), 100)
        self.assertEqual(len(self.facts), len(self.raw["facts"]))
        self.assertTrue(all(item["verified"] is True for item in self.raw["facts"]))
        self.assertEqual(self.raw["schema_version"], 2)

    def test_every_fact_has_aliases_and_provenance(self):
        for item in self.facts:
            label = item["question"]
            with self.subTest(question=label):
                aliases = item.get("aliases")
                self.assertIsInstance(aliases, list)
                self.assertGreaterEqual(len(aliases), 2)
                for field in ("source", "source_type", "verified_on", "reviewed_by"):
                    self.assertTrue(
                        isinstance(item.get(field), str) and item[field].strip(),
                        f"{label} is missing {field}",
                    )
                self.assertIsInstance(item.get("keywords"), list)

    def test_questions_and_aliases_are_unique(self):
        owners = {}
        for item in self.facts:
            for phrase in [item["question"], *item["aliases"]]:
                key = normalise(phrase)
                with self.subTest(phrase=phrase):
                    self.assertTrue(key)
                    self.assertNotIn(
                        key,
                        owners,
                        f"{phrase!r} is also curated for {owners.get(key)!r}",
                    )
                owners[key] = item["question"]

    def test_every_question_and_alias_matches_its_own_fact(self):
        for item in self.facts:
            for phrase in [item["question"], *item["aliases"]]:
                with self.subTest(phrase=phrase):
                    matched = match_verified_fact(phrase, self.facts)
                    self.assertIsNotNone(matched, f"no match for {phrase!r}")
                    self.assertEqual(matched["question"], item["question"])

    def test_common_paraphrases_resolve_to_the_expected_fact(self):
        for phrase, expected in PARAPHRASES.items():
            with self.subTest(phrase=phrase):
                matched = match_verified_fact(phrase, self.facts)
                self.assertIsNotNone(matched, f"no match for {phrase!r}")
                self.assertEqual(matched["question"], expected)

    def test_curated_names_and_answers_are_corrected(self):
        answers = " ".join(item["answer"] for item in self.facts)
        self.assertIn("Dr. Ananda Babu J", answers)
        self.assertIn("Dr. Naveen Kumar C M", answers)
        self.assertNotIn("Dr. Anand Babu J", answers)
        self.assertNotIn("Dr. C. M. Naveen Kumar", answers)
        self.assertNotIn("Dr. Naveen Kumar C.M.", answers)

    def test_hod_answers_use_the_curated_names(self):
        by_question = {item["question"]: item["answer"] for item in self.facts}
        self.assertIn(
            "Dr. Ananda Babu J",
            by_question["Who is the HOD of Computer Science and Engineering at MCE?"],
        )
        self.assertIn(
            "Dr. Ananda Babu J",
            by_question["Who is the HOD of Information Science and Engineering at MCE?"],
        )
        self.assertIn(
            "Dr. Naveen Kumar C M",
            by_question[
                "Who is the HOD of Computer Science and Business Systems at MCE?"
            ],
        )

    def test_unverified_templates_are_not_loadable_facts(self):
        templates = self.raw["college_fact_templates"]
        self.assertTrue(templates)
        self.assertTrue(all(item["verified"] is False for item in templates))
        template_questions = {normalise(item["question"]) for item in templates}
        fact_questions = {normalise(item["question"]) for item in self.facts}
        self.assertFalse(template_questions & fact_questions)


if __name__ == "__main__":
    unittest.main()
