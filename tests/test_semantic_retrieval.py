"""Tests for the dependency-free semantic retrieval layer."""

import unittest

import semantic_retrieval as sr


def fact(question, answer="answer", **extra):
    value = {"verified": True, "question": question, "answer": answer, "source": "x"}
    value.update(extra)
    return value


class NormalizationTests(unittest.TestCase):
    def test_normalize_folds_case_punctuation_and_accents(self):
        self.assertEqual(sr.normalize_text("  MCE's  HØD!  "), "mce s hød")

    def test_tokens_drop_stop_words_and_single_chars(self):
        self.assertEqual(sr.tokens("Who is the HOD of CSE at MCE?"), ["hod", "cse", "mce"])

    def test_plural_and_synonym_tokens_are_canonicalised(self):
        self.assertIn("hostel", sr.tokens("does the college have hostels and libraries"))


class ScoringTests(unittest.TestCase):
    def test_exact_question_scores_one(self):
        item = fact("Who is the current principal of MCE?")
        idf = sr._idf_table([item])
        self.assertEqual(sr.score_fact("Who is the current principal of MCE?", item, idf), 1.0)

    def test_exact_alias_scores_one(self):
        item = fact("Who is the HOD of CSE?", aliases=["Who heads computer science?"])
        idf = sr._idf_table([item])
        self.assertEqual(sr.score_fact("Who heads computer science?", item, idf), 1.0)

    def test_paraphrase_scores_higher_for_the_correct_fact(self):
        principal = fact("Who is the current principal of MCE?", keywords=["principal", "MCE"])
        library = fact("Does MCE have a library?", keywords=["library"])
        idf = sr._idf_table([principal, library])
        p = sr.score_fact("who runs the mce campus as principal", principal, idf)
        low = sr.score_fact("who runs the mce campus as principal", library, idf)
        self.assertGreater(p, low)

    def test_generic_overlap_only_scores_low(self):
        a = fact("What is the college address?", answer="addr")
        b = fact("Does MCE have a hostel?", answer="hostel")
        idf = sr._idf_table([a, b])
        score = sr.score_fact("tell me about the college", a, idf)
        self.assertLess(score, 0.5)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.facts = [
            fact("Who is the current principal of MCE?", aliases=["Who is the principal?"], keywords=["principal"]),
            fact("Does MCE have hostels?", aliases=["Is there a hostel?"], keywords=["hostel"]),
            fact("Where is Malnad College of Engineering located?", keywords=["location", "address"]),
        ]

    def test_rank_returns_best_first(self):
        ranked = sr.rank_facts("who is the principal", self.facts, top_n=1)
        self.assertEqual(ranked[0][1]["question"], "Who is the current principal of MCE?")

    def test_unambiguous_returns_none_on_a_tie(self):
        # Two facts with identical keyword overlap must not be guessed.
        a = fact("Does MCE have a library?", keywords=["library"])
        b = fact("Does MCE have a digital library?", keywords=["library"])
        self.assertIsNone(sr.is_unambiguous("does it have a library", [a, b]))

    def test_unambiguous_returns_the_clear_winner(self):
        self.assertIsNotNone(sr.is_unambiguous("where is the college located", self.facts))


class EmptyInputTests(unittest.TestCase):
    def test_empty_question_returns_none(self):
        self.assertEqual(sr.best_fact_confidence("", []), (0.0, None))
        self.assertEqual(sr.best_fact_confidence("   ", []), (0.0, None))

    def test_no_facts_returns_none(self):
        self.assertEqual(sr.best_fact_confidence("who is principal", []), (0.0, None))


if __name__ == "__main__":
    unittest.main()
