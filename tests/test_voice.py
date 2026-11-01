import contextlib
import io
import types
import unittest

from voice_service import (
    _debug_print,
    answer_spoken_question,
    apply_wake_word,
    is_greeting,
    whisper_language,
)


class GreetingTests(unittest.TestCase):
    def test_short_greetings_are_detected(self):
        for text in (
            "hi",
            "Hello!",
            "good morning",
            "hello everyone",
            "namaskara",
            "HELLO!!!",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_greeting(text))

    def test_greeting_prefixed_question_is_not_a_greeting(self):
        for text in (
            "Hello, what is the library?",
            "Hi, can you explain photosynthesis?",
            "Good morning, where is the principal's office?",
            "",
            "hire",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_greeting(text))

    def test_non_string_input_is_not_a_greeting(self):
        self.assertFalse(is_greeting(None))


class WakeWordTests(unittest.TestCase):
    def test_no_wake_word_accepts_every_transcript(self):
        result = apply_wake_word("What is the library?", "")
        self.assertTrue(result.matched)
        self.assertEqual(result.question, "What is the library?")

    def test_wake_word_must_be_a_leading_word(self):
        result = apply_wake_word("Hey assistant, what is the library?", "hey assistant")
        self.assertTrue(result.matched)
        self.assertEqual(result.question, "what is the library")

    def test_wake_word_is_case_insensitive(self):
        result = apply_wake_word("MCE ASSISTANT tell me the address", "mce assistant")
        self.assertTrue(result.matched)
        self.assertEqual(result.question, "tell me the address")

    def test_wake_word_elsewhere_or_too_similar_is_rejected(self):
        self.assertFalse(
            apply_wake_word("Please hey assistant help", "hey assistant").matched
        )
        self.assertFalse(
            apply_wake_word("hey assistants what is the library", "hey assistant").matched
        )
        self.assertFalse(apply_wake_word("hello", "hey assistant").matched)

    def test_wake_word_alone_yields_an_empty_accepted_question(self):
        result = apply_wake_word("Hey assistant!", "hey assistant")
        self.assertTrue(result.matched)
        self.assertEqual(result.question, "")

    def test_whitespace_is_normalised(self):
        result = apply_wake_word("  hey   assistant   hello  ", "hey assistant")
        self.assertEqual(result.question, "hello")


class LanguageTests(unittest.TestCase):
    def test_whisper_language_uses_the_base_code(self):
        self.assertEqual(whisper_language("en-IN"), "en")
        self.assertEqual(whisper_language("en_US"), "en")
        self.assertEqual(whisper_language("kn"), "kn")
        self.assertIsNone(whisper_language(""))

    def test_greeting_answers_are_local(self):
        called = []

        def fake_answer(question, settings, facts):
            called.append(question)
            return "provider answer"

        reply = answer_spoken_question(
            "Hello!", object(), [], answer_func=fake_answer
        )
        self.assertIn("How can I help", reply)
        self.assertEqual(called, [])

    def test_question_is_delegated_to_the_answer_function(self):
        called = []

        def fake_answer(question, settings, facts):
            called.append(question)
            return "provider answer"

        reply = answer_spoken_question(
            "  What is the library?  ", object(), [{"question": "x"}], answer_func=fake_answer
        )
        self.assertEqual(reply, "provider answer")
        self.assertEqual(called, ["What is the library?"])

    def test_empty_question_returns_empty_without_calling_provider(self):
        called = []
        reply = answer_spoken_question(
            "   ", object(), [], answer_func=lambda *args: called.append(args) or "x"
        )
        self.assertEqual(reply, "")
        self.assertEqual(called, [])


class VoiceDebugPrintTests(unittest.TestCase):
    def test_debug_print_is_silent_by_default_and_verbose_when_enabled(self):
        off = types.SimpleNamespace(voice_debug=False)
        on = types.SimpleNamespace(voice_debug=True)
        missing = types.SimpleNamespace()

        for settings in (off, missing):
            with self.subTest(voice_debug=getattr(settings, "voice_debug", "absent")):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    _debug_print("Heard question: private", settings)
                self.assertEqual(output.getvalue(), "")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _debug_print("Heard question: private", on)
        self.assertEqual(output.getvalue(), "Heard question: private\n")


if __name__ == "__main__":
    unittest.main()
