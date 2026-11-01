import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import doctor
from ai_provider import load_knowledge
from enrollment import save_embedding_cache

PROJECT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT / "data" / "college_knowledge.json"


def fact(question="What is the library?", answer="There is a library.", **extra):
    value = {
        "verified": True,
        "question": question,
        "answer": answer,
        "source": "test source",
        "source_type": "user_attachment",
        "verified_on": "2026-10-25",
        "reviewed_by": "test reviewer",
        "aliases": [f"{question} alias one", f"{question} alias two"],
    }
    value.update(extra)
    return value


def make_settings(root: Path):
    return types.SimpleNamespace(
        project_path=root,
        knowledge_path=root / "knowledge.json",
        training_path=root / "training",
    )


class KnowledgeCheckTests(unittest.TestCase):
    def test_project_knowledge_has_no_failures_or_warnings(self):
        facts = load_knowledge(KNOWLEDGE_PATH)
        results = doctor._check_knowledge(facts)
        self.assertTrue(results)
        problems = [
            result for result in results if result.level in {"FAIL", "WARN"}
        ]
        self.assertEqual(problems, [])
        names = {result.name for result in results}
        self.assertIn("FAQ ambiguity", names)
        self.assertIn("Knowledge provenance", names)

    def test_duplicate_alias_is_reported_as_ambiguity(self):
        results = doctor._check_knowledge(
            [
                fact(question="Question one", aliases=["shared phrase"]),
                fact(question="Question two", aliases=["shared phrase"]),
            ]
        )
        ambiguity = [result for result in results if result.name == "FAQ ambiguity"]
        self.assertEqual(len(ambiguity), 1)
        self.assertEqual(ambiguity[0].level, "FAIL")

    def test_duplicate_option_entries_without_aliases_still_pass_shape(self):
        results = doctor._check_knowledge(
            [fact(question="Question one"), fact(question="Question two")]
        )
        self.assertFalse([result for result in results if result.level == "FAIL"])

    def test_no_verified_facts_is_a_failure(self):
        results = doctor._check_knowledge([])
        usability = [result for result in results if result.name == "Knowledge usability"]
        self.assertEqual(len(usability), 1)
        self.assertEqual(usability[0].level, "FAIL")

    def test_missing_provenance_is_a_warning(self):
        results = doctor._check_knowledge(
            [fact(verified_on="", reviewed_by="", source_type="")]
        )
        provenance = [
            result for result in results if result.name == "Knowledge provenance"
        ]
        self.assertEqual(len(provenance), 1)
        self.assertEqual(provenance[0].level, "WARN")


class FileAndTrainingCheckTests(unittest.TestCase):
    def test_missing_data_files_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            results = doctor._check_files(settings)
        levels = {result.name: result.level for result in results}
        self.assertEqual(levels["Knowledge file"], "FAIL")
        self.assertEqual(levels["Training folder"], "FAIL")
        self.assertEqual(levels[".env"], "SKIP")

    def test_present_files_are_reported_without_secret_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            settings.knowledge_path.write_text("{}", encoding="utf-8")
            settings.training_path.mkdir()
            (root / ".env").write_text(
                "OPENAI_API_KEY=super-secret-value\n", encoding="utf-8"
            )
            results = doctor._check_files(settings)

        levels = {result.name: result.level for result in results}
        self.assertEqual(levels["Knowledge file"], "OK")
        self.assertEqual(levels["Training folder"], "OK")
        self.assertEqual(levels[".env"], "OK")
        details = " ".join(result.detail for result in results)
        self.assertNotIn("super-secret-value", details)

    def test_shallow_training_set_warns_about_depth_and_consent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            for index in range(2):
                person = settings.training_path / "Alice"
                person.mkdir(parents=True, exist_ok=True)
                (person / f"{index}.jpg").write_bytes(b"image")
            results = doctor._check_training(settings)

        by_name = {result.name: result for result in results}
        self.assertEqual(by_name["Enrollment images"].level, "OK")
        self.assertEqual(by_name["Enrollment depth"].level, "WARN")
        self.assertEqual(by_name["Consent/retention note"].level, "WARN")

    def test_deep_training_set_with_readme_has_no_depth_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            for person_name in ("Alice", "Bob"):
                person = settings.training_path / person_name
                person.mkdir(parents=True, exist_ok=True)
                for index in range(5):
                    (person / f"{index}.jpg").write_bytes(b"image")
            (settings.training_path / "README.md").write_text(
                "# Consent\n", encoding="utf-8"
            )
            results = doctor._check_training(settings)

        by_name = {result.name: result for result in results}
        self.assertEqual(by_name["Enrollment images"].level, "OK")
        self.assertNotIn("Enrollment depth", by_name)
        self.assertNotIn("Consent/retention note", by_name)

    def test_empty_training_folder_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            results = doctor._check_training(make_settings(Path(directory)))
        self.assertEqual(results[0].name, "Enrollment images")
        self.assertEqual(results[0].level, "FAIL")


class ModelCacheCheckTests(unittest.TestCase):
    def test_missing_project_local_models_are_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = types.SimpleNamespace(
                project_path=Path(directory),
                face_model="VGG-Face",
                stt_provider="whisper",
                whisper_model="small",
            )
            with patch.dict(os.environ, {}, clear=True):
                results = doctor._check_model_caches(settings)

        by_name = {result.name: result for result in results}
        self.assertEqual(by_name["Runtime caches"].level, "OK")
        self.assertEqual(by_name["DeepFace weights"].level, "WARN")
        self.assertEqual(by_name["Whisper model"].level, "WARN")

    def test_cached_project_local_models_are_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = (
                root
                / ".runtime"
                / "deepface"
                / ".deepface"
                / "weights"
                / "vgg_face_weights.h5"
            )
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"weights")
            whisper = (
                root
                / ".runtime"
                / "huggingface"
                / "hub"
                / "models--Systran--faster-whisper-small"
            )
            whisper.mkdir(parents=True)
            settings = types.SimpleNamespace(
                project_path=root,
                face_model="VGG-Face",
                stt_provider="whisper",
                whisper_model="small",
            )
            with patch.dict(os.environ, {}, clear=True):
                results = doctor._check_model_caches(settings)

        by_name = {result.name: result for result in results}
        self.assertEqual(by_name["DeepFace weights"].level, "OK")
        self.assertEqual(by_name["Whisper model"].level, "OK")


class EmbeddingCacheCheckTests(unittest.TestCase):
    def test_cache_coverage_warns_when_images_were_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "training" / "Alice"
            person.mkdir(parents=True)
            (person / "a.jpg").write_bytes(b"a")
            (person / "b.jpg").write_bytes(b"b")
            cache_path = root / "cache" / "embeddings.json"
            save_embedding_cache(
                cache_path,
                model="VGG-Face",
                detector="opencv",
                entries={"a" * 64: [1.0, 0.0]},
            )
            settings = types.SimpleNamespace(
                training_path=root / "training",
                embedding_cache_path=cache_path,
                use_embedding_cache=True,
                face_model="VGG-Face",
                face_detector="opencv",
            )
            results = doctor._check_embedding_cache(settings)

        self.assertEqual(results[0].level, "WARN")
        self.assertIn("1 of 2", results[0].detail)

    def test_cache_coverage_is_ok_when_every_image_is_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "training" / "Alice"
            person.mkdir(parents=True)
            (person / "a.jpg").write_bytes(b"a")
            cache_path = root / "cache" / "embeddings.json"
            save_embedding_cache(
                cache_path,
                model="VGG-Face",
                detector="opencv",
                entries={"a" * 64: [1.0, 0.0]},
            )
            settings = types.SimpleNamespace(
                training_path=root / "training",
                embedding_cache_path=cache_path,
                use_embedding_cache=True,
                face_model="VGG-Face",
                face_detector="opencv",
            )
            results = doctor._check_embedding_cache(settings)

        self.assertEqual(results[0].level, "OK")

    def test_cache_coverage_is_skipped_when_cache_is_disabled(self):
        settings = types.SimpleNamespace(
            training_path=Path("missing"),
            embedding_cache_path=Path("missing.json"),
            use_embedding_cache=False,
        )
        results = doctor._check_embedding_cache(settings)
        self.assertEqual(results[0].level, "SKIP")


class OpenCVGuiCheckTests(unittest.TestCase):
    def test_backend_parser_distinguishes_gui_and_headless(self):
        self.assertEqual(
            doctor.opencv_gui_backend("  GUI:                           WIN32UI\n"),
            "WIN32UI",
        )
        self.assertIsNone(doctor.opencv_gui_backend("  GUI:                           NONE\n"))
        self.assertIsNone(doctor.opencv_gui_backend("no gui line"))

    def test_headless_build_is_a_failure(self):
        with patch(
            "doctor._opencv_build_information",
            return_value="  GUI:                           NONE\n",
        ):
            result = doctor._check_opencv_gui()
        self.assertEqual(result.level, "FAIL")
        self.assertIn("headless", result.detail)

    def test_gui_build_is_reported_ok(self):
        with patch(
            "doctor._opencv_build_information",
            return_value="  GUI:                           WIN32UI\n",
        ):
            result = doctor._check_opencv_gui()
        self.assertEqual(result.level, "OK")
        self.assertIn("WIN32UI", result.detail)


class PythonCheckTests(unittest.TestCase):
    def test_current_interpreter_is_checked(self):
        results = doctor._check_python()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Python")
        self.assertIn(results[0].level, {"OK", "WARN"})
        self.assertIn("Python", results[0].detail)


if __name__ == "__main__":
    unittest.main()
