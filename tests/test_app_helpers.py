import argparse
import contextlib
import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app import (
    _apply_overrides,
    _display_text,
    _wrap_text,
    run_chat,
    warm_up_models,
)
from settings import Settings


def make_settings() -> Settings:
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ, {}, clear=True):
            return Settings.from_env(Path(directory), load_env=False)


def make_args(**overrides) -> argparse.Namespace:
    values = {
        "no_voice": False,
        "camera_index": None,
        "ai_provider": None,
        "stt_provider": None,
        "wake_word": None,
        "offline": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class OverrideTests(unittest.TestCase):
    def test_offline_disables_cloud_answers_and_google_stt(self):
        settings = make_settings()
        args = make_args(
            offline=True, ai_provider="gemini", stt_provider="google"
        )
        updated, notes = _apply_overrides(settings, args)
        self.assertEqual(updated.ai_provider, "offline")
        self.assertEqual(updated.stt_provider, "none")
        self.assertTrue(any("Google" in note for note in notes))
        self.assertTrue(any("Offline mode" in note for note in notes))

    def test_offline_keeps_local_whisper(self):
        settings = make_settings()
        args = make_args(offline=True, stt_provider="faster-whisper")
        updated, _ = _apply_overrides(settings, args)
        self.assertEqual(updated.ai_provider, "offline")
        self.assertEqual(updated.stt_provider, "whisper")

    def test_no_voice_and_runtime_overrides(self):
        settings = make_settings()
        args = make_args(
            no_voice=True,
            camera_index=2,
            ai_provider="  OLLAMA  ",
            stt_provider="GOOGLE_WEB",
            wake_word="  hey assistant  ",
        )
        updated, notes = _apply_overrides(settings, args)
        self.assertFalse(updated.enable_voice)
        self.assertEqual(updated.camera_index, 2)
        self.assertEqual(updated.ai_provider, "ollama")
        self.assertEqual(updated.stt_provider, "google")
        self.assertEqual(updated.wake_word, "hey assistant")
        self.assertTrue(any("disabled" in note for note in notes))

    def test_empty_wake_word_disables_it(self):
        settings = make_settings()
        updated, _ = _apply_overrides(settings, make_args(wake_word="   "))
        self.assertEqual(updated.wake_word, "")

    def test_original_settings_are_not_mutated(self):
        settings = make_settings()
        before = settings.ai_provider
        _apply_overrides(settings, make_args(offline=True))
        self.assertEqual(settings.ai_provider, before)


class HeadlessChatTests(unittest.TestCase):
    def test_chat_answers_and_stops_on_quit(self):
        settings = make_settings()
        facts = [
            {
                "verified": True,
                "question": "What is the library?",
                "answer": "The library is on campus.",
                "source": "test",
            }
        ]
        replies = iter(["What is the library?", "/quit"])
        output = []
        code = run_chat(
            settings,
            facts,
            ask=lambda _prompt: next(replies),
            output=output.append,
        )
        self.assertEqual(code, 0)
        self.assertTrue(output[0].startswith("Text chat started."))
        self.assertTrue(
            any(
                "Assistant [verified_faq]: The library is on campus." in line
                for line in output
            )
        )
        self.assertEqual(output[-1], "Text chat stopped.")

    def test_chat_answers_a_greeting_without_a_provider(self):
        settings = make_settings()
        replies = iter(["hello", "/quit"])
        output = []
        code = run_chat(
            settings,
            [],
            ask=lambda _prompt: next(replies),
            output=output.append,
        )
        self.assertEqual(code, 0)
        self.assertTrue(
            any(
                "Assistant [greeting]: Hello! How can I help you?" in line
                for line in output
            )
        )

    def test_chat_handles_end_of_input(self):
        settings = make_settings()
        output = []
        code = run_chat(
            settings,
            [],
            ask=lambda _prompt: (_ for _ in ()).throw(EOFError()),
            output=output.append,
        )
        self.assertEqual(code, 0)
        self.assertEqual(output[-1], "Text chat stopped.")


class ModelWarmupTests(unittest.TestCase):
    def test_warmup_builds_face_model_and_skips_whisper_when_disabled(self):
        calls = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = warm_up_models(
                make_settings(),
                deepface_builder=lambda: calls.append("face"),
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["face"])
        self.assertIn("Face model ready", output.getvalue())
        self.assertIn("Whisper model skipped", output.getvalue())

    def test_warmup_builds_both_models_when_whisper_is_enabled(self):
        settings = replace(make_settings(), stt_provider="whisper")
        calls = []
        with contextlib.redirect_stdout(io.StringIO()):
            code = warm_up_models(
                settings,
                deepface_builder=lambda: calls.append("face"),
                whisper_builder=lambda: calls.append("whisper"),
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["face", "whisper"])

    def test_warmup_failure_is_reported_without_leaking_details(self):
        def fail():
            raise RuntimeError("network unavailable")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = warm_up_models(make_settings(), deepface_builder=fail)
        self.assertEqual(code, 1)
        self.assertIn("Face model warm-up failed", output.getvalue())


class DisplayTextTests(unittest.TestCase):
    def test_non_ascii_text_is_replaced_for_opencv(self):
        self.assertEqual(_display_text("café"), "caf?")

    def test_wrap_text_handles_empty_and_short_values(self):
        self.assertEqual(_wrap_text(""), [])
        self.assertEqual(_wrap_text("   "), [])
        self.assertEqual(_wrap_text("Hello"), ["Hello"])

    def test_wrap_text_limits_lines_and_adds_ellipsis(self):
        lines = _wrap_text("word " * 60, width=20, max_lines=3)
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[-1].endswith("..."))
        for line in lines:
            self.assertLessEqual(len(line), 20)

    def test_wrap_text_enforces_a_minimum_width(self):
        # A width below 12 is raised to 12, so "one two three" must wrap.
        lines = _wrap_text("one two three four", width=1, max_lines=2)
        self.assertEqual(lines, ["one two", "three four"])


if __name__ == "__main__":
    unittest.main()
