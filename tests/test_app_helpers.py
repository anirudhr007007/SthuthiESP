import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from app import (
    _apply_overrides,
    _display_text,
    _ollama_ping,
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
                ollama_pinger=lambda _settings: calls.append("ollama"),
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["face", "ollama"])
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
                ollama_pinger=lambda _settings: calls.append("ollama"),
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["face", "whisper", "ollama"])

    def test_warmup_failure_is_reported_without_leaking_details(self):
        def fail():
            raise RuntimeError("network unavailable")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = warm_up_models(
                make_settings(),
                deepface_builder=fail,
                ollama_pinger=lambda _settings: None,
            )
        self.assertEqual(code, 1)
        self.assertIn("Face model warm-up failed", output.getvalue())


class OllamaWarmupTests(unittest.TestCase):
    def test_ollama_pinger_runs_for_ollama_provider(self):
        # Settings defaults AI_PROVIDER to ollama.
        calls = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = warm_up_models(
                make_settings(),
                deepface_builder=lambda: calls.append("face"),
                ollama_pinger=lambda _settings: calls.append("ping"),
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["face", "ping"])
        self.assertIn("Ollama model ready:", output.getvalue())
        self.assertIn("(kept warm for 1 hour)", output.getvalue())
        self.assertIn("Model warm-up complete", output.getvalue())

    def test_ollama_pinger_is_skipped_for_other_providers(self):
        calls = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = warm_up_models(
                replace(make_settings(), ai_provider="offline"),
                deepface_builder=lambda: calls.append("face"),
                ollama_pinger=lambda _settings: calls.append("ping"),
            )
        self.assertEqual(code, 0)
        self.assertEqual(calls, ["face"])
        self.assertIn(
            "Ollama warm-up skipped: AI_PROVIDER is not ollama.",
            output.getvalue(),
        )

    def test_ollama_pinger_failure_does_not_fail_the_run(self):
        def fail(_settings):
            raise RuntimeError("Ollama is not reachable")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = warm_up_models(
                make_settings(),
                deepface_builder=lambda: None,
                ollama_pinger=fail,
            )
        self.assertEqual(code, 0)
        self.assertIn("Ollama warm-up skipped:", output.getvalue())
        self.assertIn("Model warm-up complete", output.getvalue())

    def test_ollama_ping_sends_minimal_chat_request(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"{}"

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        settings = replace(
            make_settings(),
            ollama_url="http://127.0.0.1:11434/",
            ai_timeout=1.0,
        )
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _ollama_ping(settings)
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(captured["timeout"], 10.0)
        self.assertEqual(
            captured["payload"],
            {
                "model": settings.ollama_model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "think": False,
                "keep_alive": "1h",
                "options": {"num_predict": 1},
            },
        )

    def test_ollama_ping_failure_is_wrapped_into_a_skip_message(self):
        settings = replace(
            make_settings(),
            ollama_url="http://127.0.0.1:1",  # unreachable loopback port
        )
        output = io.StringIO()
        with patch(
            "urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ), contextlib.redirect_stdout(output):
            code = warm_up_models(settings, deepface_builder=lambda: None)
        self.assertEqual(code, 0)
        self.assertIn("Ollama warm-up skipped:", output.getvalue())
        self.assertIn("connection refused", output.getvalue())
        self.assertIn("Model warm-up complete", output.getvalue())

    def test_real_ollama_ping_raises_for_a_dead_url(self):
        settings = replace(
            make_settings(),
            ollama_url="http://127.0.0.1:1",  # unreachable loopback port
        )
        with patch(
            "urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaises(URLError):
                _ollama_ping(settings)


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
