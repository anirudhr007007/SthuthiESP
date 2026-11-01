import contextlib
import hashlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from enrollment import (
    dataset_summary,
    iter_person_images,
    load_embedding_cache,
    load_known_embeddings,
    save_embedding_cache,
    sha256_file,
)


def make_settings(root: Path, *, use_cache: bool = True):
    return types.SimpleNamespace(
        training_path=root / "training",
        embedding_cache_path=root / "cache" / "embeddings.json",
        face_model="fake-model",
        face_detector="fake-detector",
        use_embedding_cache=use_cache,
    )


def write_image(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class FakeDeepFace:
    def __init__(
        self, vectors=None, *, fail=(), multiple=(), empty=(), dominant=()
    ):
        self.vectors = vectors or {}
        self.fail = set(fail)
        self.multiple = set(multiple)
        self.empty = set(empty)
        self.dominant = set(dominant)
        self.calls = 0

    def represent(self, path, **_kwargs):
        self.calls += 1
        path = Path(path)
        key = f"{path.parent.name}/{path.name}"
        if key in self.fail:
            raise RuntimeError("detector failed")
        if key in self.multiple:
            return [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        if key in self.empty:
            return []
        if key in self.dominant:
            return [
                {
                    "embedding": [1.0, 0.0],
                    "facial_area": {"x": 0, "y": 0, "w": 500, "h": 500},
                },
                {
                    "embedding": [0.0, 1.0],
                    "facial_area": {"x": 0, "y": 0, "w": 50, "h": 50},
                },
            ]
        vector = self.vectors.get(key, [1.0, 0.0])
        return [{"embedding": vector, "facial_area": {"x": 1, "y": 1, "w": 50, "h": 50}}]


class DiscoveryTests(unittest.TestCase):
    def test_discovers_nested_images_and_ignores_private_or_noise_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "training"
            write_image(root / "Alice" / "front.jpg", b"a1")
            write_image(root / "Alice" / "angles" / "left.PNG", b"a2")
            write_image(root / "Alice" / "notes.txt", b"not an image")
            write_image(root / "Alice" / ".hidden" / "secret.jpg", b"hidden")
            write_image(root / "Bob" / "front.jpeg", b"b1")
            write_image(root / ".private" / "front.jpg", b"private")
            write_image(root / "venv" / "front.jpg", b"venv")
            write_image(root / "tests" / "front.jpg", b"tests")
            write_image(root / "Nobody" / "readme.md", b"text")

            people = iter_person_images(root)

        self.assertEqual([person.name for person in people], ["Alice", "Bob"])
        self.assertEqual(len(people[0].images), 2)
        self.assertTrue(all(person.images for person in people))

    def test_summary_handles_missing_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = dataset_summary(Path(directory) / "missing")
        self.assertEqual(summary.people, 0)
        self.assertEqual(summary.images, 0)

    def test_summary_reports_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "training"
            write_image(root / "Alice" / "a.jpg", b"a")
            write_image(root / "Alice" / "b.jpg", b"b")
            write_image(root / "Bob" / "c.jpg", b"c")
            summary = dataset_summary(root)
        self.assertEqual(summary.people, 2)
        self.assertEqual(summary.images, 3)
        self.assertEqual(summary.min_images, 1)
        self.assertEqual(summary.max_images, 2)

    def test_sha256_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"content")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"content").hexdigest())


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache" / "embeddings.json"
            key = stable_key("alice")
            self.assertTrue(
                save_embedding_cache(
                    path,
                    model="model-a",
                    detector="opencv",
                    entries={key: [0.1, 0.2, 0.3]},
                )
            )
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            loaded = load_embedding_cache(
                path, model="model-a", detector="opencv"
            )
            self.assertEqual(loaded[key], (0.1, 0.2, 0.3))
            self.assertEqual(
                load_embedding_cache(path, model="model-b", detector="opencv"), {}
            )
            self.assertEqual(
                load_embedding_cache(path, model="model-a", detector="other"), {}
            )

    def test_corrupt_and_invalid_entries_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(
                load_embedding_cache(path, model="m", detector="d"), {}
            )

            payload = {
                "schema": 1,
                "model": "m",
                "detector": "d",
                "align": True,
                "images": {
                    "a" * 64: {"embedding": [1.0, 2.0]},
                    "b" * 64: {"embedding": "not a vector"},
                    "c" * 64: {"embedding": [1.0, float("nan")]},
                    "short": {"embedding": [1.0]},
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_embedding_cache(path, model="m", detector="d")
        self.assertEqual(loaded, {"a" * 64: (1.0, 2.0)})

    def test_save_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.json"
            key = stable_key("alice")
            save_embedding_cache(
                path, model="m", detector="d", entries={key: [1.0]}
            )
            save_embedding_cache(
                path, model="m", detector="d", entries={key: [2.0]}
            )
            loaded = load_embedding_cache(path, model="m", detector="d")
        self.assertEqual(loaded[key], (2.0,))


class EnrollmentTests(unittest.TestCase):
    def _dataset(self, directory: Path) -> Path:
        root = directory / "training"
        write_image(root / "Alice" / "a1.jpg", b"alice-1")
        write_image(root / "Alice" / "a2.jpg", b"alice-2")
        write_image(root / "Bob" / "b1.jpg", b"bob-1")
        return root

    def test_builds_embeddings_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            settings = make_settings(root)
            fake = FakeDeepFace(
                {"Alice/a1.jpg": [1.0, 0.0], "Alice/a2.jpg": [0.9, 0.1], "Bob/b1.jpg": [0.0, 1.0]}
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                first = load_known_embeddings(settings, deepface=fake)
            self.assertEqual(fake.calls, 3)
            self.assertEqual(len(first["Alice"]), 2)
            self.assertEqual(len(first["Bob"]), 1)
            self.assertTrue(settings.embedding_cache_path.is_file())

            exploding = FakeDeepFace()
            exploding.represent = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("cache should be used")
            )
            with contextlib.redirect_stdout(io.StringIO()):
                second = load_known_embeddings(settings, deepface=exploding)
        self.assertEqual(second, first)

    def test_duplicate_image_is_only_enrolled_for_one_person(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            # Bob has a duplicate of Alice's first image plus one unique image.
            write_image(root / "training" / "Bob" / "copy.jpg", b"alice-1")
            write_image(root / "training" / "Bob" / "b2.jpg", b"bob-2")
            settings = make_settings(root)
            fake = FakeDeepFace()
            with contextlib.redirect_stdout(io.StringIO()):
                embeddings = load_known_embeddings(settings, deepface=fake)
        self.assertEqual(len(embeddings["Alice"]), 2)
        self.assertEqual(len(embeddings["Bob"]), 2)

    def test_images_without_exactly_one_face_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            settings = make_settings(root)
            fake = FakeDeepFace(
                empty={"Alice/a1.jpg"},
                multiple={"Alice/a2.jpg"},
                fail={"Bob/b1.jpg"},
            )
            with contextlib.redirect_stdout(io.StringIO()):
                embeddings = load_known_embeddings(settings, deepface=fake)
        self.assertEqual(embeddings, {})

    def test_dominant_face_with_tiny_background_detection_is_enrolled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            settings = make_settings(root)
            fake = FakeDeepFace(dominant={"Alice/a1.jpg"})

            def dimensions(path):
                return (1000, 1000) if Path(path).name == "a1.jpg" else None

            with patch("enrollment._image_dimensions", side_effect=dimensions):
                with contextlib.redirect_stdout(io.StringIO()):
                    embeddings = load_known_embeddings(settings, deepface=fake)

        self.assertEqual(len(embeddings["Alice"]), 2)
        self.assertEqual(len(embeddings["Bob"]), 1)

    def test_cache_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            settings = make_settings(root, use_cache=False)
            with contextlib.redirect_stdout(io.StringIO()):
                embeddings = load_known_embeddings(settings, deepface=FakeDeepFace())
        self.assertTrue(embeddings)
        self.assertFalse(settings.embedding_cache_path.exists())

    def test_missing_training_folder_raises_runtime_error(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            with self.assertRaises(RuntimeError):
                load_known_embeddings(settings, deepface=FakeDeepFace())


if __name__ == "__main__":
    unittest.main()
