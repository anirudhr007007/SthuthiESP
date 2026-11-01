import tempfile
import unittest
from pathlib import Path

from runtime_env import (
    configure_runtime_environment,
    deepface_weights_dir,
    runtime_environment,
    runtime_root,
    whisper_cache_dir,
)


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_cache_paths_are_under_the_project_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = runtime_environment(root, {})
            runtime = (root / ".runtime").resolve()

        self.assertEqual(Path(values["DEEPFACE_HOME"]), runtime / "deepface")
        self.assertEqual(Path(values["KERAS_HOME"]), runtime / "keras")
        self.assertEqual(Path(values["HF_HOME"]), runtime / "huggingface")
        self.assertEqual(
            Path(values["HUGGINGFACE_HUB_CACHE"]), runtime / "huggingface" / "hub"
        )
        self.assertEqual(Path(values["TMP"]), runtime / "tmp")
        self.assertEqual(Path(values["PIP_CACHE_DIR"]), runtime / "pip-cache")
        self.assertEqual(
            deepface_weights_dir(root, {}),
            runtime / "deepface" / ".deepface" / "weights",
        )
        self.assertEqual(
            whisper_cache_dir(root, {}), runtime / "huggingface" / "hub"
        )

    def test_absolute_runtime_override_is_honoured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            alternate = root / "portable-runtime"
            values = runtime_environment(
                root, {"ESSPTHUTHI_RUNTIME_DIR": str(alternate)}
            )

        self.assertEqual(
            runtime_root(root, {"ESSPTHUTHI_RUNTIME_DIR": str(alternate)}),
            alternate.resolve(),
        )
        self.assertEqual(Path(values["DEEPFACE_HOME"]), alternate.resolve() / "deepface")

    def test_out_of_tree_runtime_override_falls_back_to_the_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "elsewhere-runtime"
            environ = {"ESSPTHUTHI_RUNTIME_DIR": str(outside)}

            self.assertEqual(runtime_root(root, environ), root.resolve() / ".runtime")
            self.assertEqual(
                Path(runtime_environment(root, environ)["HF_HOME"]),
                root.resolve() / ".runtime" / "huggingface",
            )

    def test_runtime_root_cannot_be_the_project_root_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            for configured in (".", str(root)):
                with self.subTest(configured=configured):
                    environ = {"ESSPTHUTHI_RUNTIME_DIR": configured}
                    self.assertEqual(
                        runtime_root(root, environ), root.resolve() / ".runtime"
                    )

    def test_configure_replaces_global_cache_values_and_creates_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environ = {
                "DEEPFACE_HOME": str(root / "outside" / "deepface"),
                "HF_HOME": str(root / "outside" / "huggingface"),
            }
            configure_runtime_environment(root, environ)
            runtime = (root / ".runtime").resolve()
            self.assertEqual(Path(environ["DEEPFACE_HOME"]), runtime / "deepface")
            self.assertEqual(Path(environ["HF_HOME"]), runtime / "huggingface")
            self.assertTrue((runtime / "deepface").is_dir())
            self.assertTrue((runtime / "huggingface" / "hub").is_dir())
            self.assertTrue((runtime / "tmp").is_dir())


if __name__ == "__main__":
    unittest.main()
