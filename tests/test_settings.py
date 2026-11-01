import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from settings import Settings, load_env_file, normalize_stt_provider


class EnvFileTests(unittest.TestCase):
    def test_parser_handles_quotes_comments_exports_and_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "export NEW=\"hello world\"",
                        "QUOTED='single value'",
                        "INLINE=value # trailing comment",
                        "EMPTY=",
                        "BROKEN=\"unterminated",
                        "NO_EQUALS",
                        "1BAD=x",
                        "EXISTING=from-file",
                    ]
                ),
                encoding="utf-8",
            )
            environ = {"EXISTING": "from-environment"}
            applied = load_env_file(path, environ)

        self.assertEqual(applied, 4)
        self.assertEqual(environ["NEW"], "hello world")
        self.assertEqual(environ["QUOTED"], "single value")
        self.assertEqual(environ["INLINE"], "value")
        self.assertEqual(environ["EMPTY"], "")
        self.assertEqual(environ["EXISTING"], "from-environment")
        self.assertNotIn("BROKEN", environ)
        self.assertNotIn("NO_EQUALS", environ)
        self.assertNotIn("1BAD", environ)

    def test_missing_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_env_file(Path(directory) / "missing.env", {}), 0)


class SettingsTests(unittest.TestCase):
    def test_defaults_are_privacy_safe_and_project_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(Path(directory), load_env=False)

        project = Path(directory).resolve()
        self.assertEqual(settings.training_path, project / "data" / "training_images")
        self.assertEqual(
            settings.knowledge_path, project / "data" / "college_knowledge.json"
        )
        self.assertEqual(
            settings.embedding_cache_path, project / ".cache" / "embeddings.json"
        )
        self.assertEqual(settings.stt_provider, "none")
        self.assertTrue(settings.enable_voice)
        self.assertEqual(settings.similarity_threshold, 0.68)
        self.assertEqual(settings.match_margin, 0.08)
        self.assertEqual(settings.face_model, "VGG-Face")
        self.assertFalse(settings.wake_word)

    def test_description_shows_camera_and_recognition_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(Path(directory), load_env=False)
        description = settings.describe()

        self.assertEqual(description["camera"], "index 0, 1280x720 @ 30 fps")
        self.assertIn("threshold 0.68", description["recognition"])
        self.assertIn("margin 0.08", description["recognition"])
        self.assertIn("interval 0.35s", description["recognition"])
        self.assertIn("confirmations 3", description["recognition"])

    def test_dotenv_loads_but_real_environment_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / ".env").write_text(
                "COLLEGE_NAME=From file\nSTT_PROVIDER=google\nSIMILARITY_THRESHOLD=0.7\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ, {"COLLEGE_NAME": "From environment"}, clear=True
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    settings = Settings.from_env(project)

        self.assertEqual(settings.college_name, "From environment")
        self.assertEqual(settings.stt_provider, "google")
        self.assertEqual(settings.similarity_threshold, 0.7)
        self.assertIn("Loaded", output.getvalue())

    def test_stt_aliases_are_normalised(self):
        cases = {
            "": "none",
            "off": "none",
            "disabled": "none",
            "NONE": "none",
            "Google-Cloud": "google",
            "GOOGLE_WEB": "google",
            "faster-whisper": "whisper",
            "local_whisper": "whisper",
            "something-else": "something-else",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_stt_provider(value), expected)

    def test_web_search_defaults_off_and_environment_reenables_it(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"WEB_SEARCH": "true"}, clear=True):
                settings = Settings.from_env(Path(directory), load_env=False)
        self.assertTrue(settings.web_search)

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                defaults = Settings.from_env(Path(directory), load_env=False)
        self.assertFalse(defaults.web_search)

    def test_out_of_range_thresholds_are_clamped(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"SIMILARITY_THRESHOLD": "1.7", "MATCH_MARGIN": "-0.4"},
                clear=True,
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    settings = Settings.from_env(Path(directory), load_env=False)

        self.assertEqual(settings.similarity_threshold, 1.0)
        self.assertEqual(settings.match_margin, 0.0)
        self.assertIn("clamping", output.getvalue())

    def test_describe_does_not_reveal_api_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "openai-secret-value",
                    "GEMINI_API_KEY": "gemini-secret-value",
                    "AI_API_KEY": "compatible-secret-value",
                },
                clear=True,
            ):
                settings = Settings.from_env(Path(directory), load_env=False)

        description = " ".join(settings.describe().values())
        self.assertNotIn("openai-secret-value", description)
        self.assertNotIn("gemini-secret-value", description)
        self.assertNotIn("compatible-secret-value", description)
        self.assertIn("microphone off", description)

    def test_invalid_numbers_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {"CAMERA_INDEX": "front", "AI_TIMEOUT": "soon"},
                clear=True,
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    settings = Settings.from_env(Path(directory), load_env=False)

        self.assertEqual(settings.camera_index, 0)
        self.assertEqual(settings.ai_timeout, 15.0)
        self.assertIn("Invalid CAMERA_INDEX", output.getvalue())


if __name__ == "__main__":
    unittest.main()
