"""Release metadata and repository hygiene checks.

These tests run without third-party packages and catch drift between the
version file, pyproject metadata, pinned requirements, safe example config,
and documentation links.
"""

import re
import tomllib
import unittest
from pathlib import Path

import version

PROJECT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pyproject = tomllib.loads(
            (PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        )

    def test_version_matches_pyproject(self):
        self.assertEqual(self.pyproject["project"]["version"], version.__version__)

    def test_python_target_is_tensorflow_compatible(self):
        self.assertEqual(self.pyproject["project"]["requires-python"], ">=3.11,<3.13")

    def test_direct_dependencies_are_pinned(self):
        project = self.pyproject["project"]
        groups = [project["dependencies"], *project.get("optional-dependencies", {}).values()]
        for group in groups:
            for requirement in group:
                with self.subTest(requirement=requirement):
                    self.assertIn("==", requirement)

    def test_requirement_files_pin_every_direct_package(self):
        for path in sorted(PROJECT.glob("requirements*.txt")):
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.split("#", 1)[0].strip()
                if not line or line.startswith("-r "):
                    continue
                with self.subTest(path=path.name, line=line):
                    self.assertIn("==", line)

    def test_example_env_has_safe_defaults_and_no_filled_secrets(self):
        text = (PROJECT / ".env.example").read_text(encoding="utf-8")
        values = {}
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()

        self.assertEqual(values["STT_PROVIDER"], "none")
        self.assertEqual(values["AI_PROVIDER"], "ollama")
        self.assertEqual(values["ESSPTHUTHI_RUNTIME_DIR"], ".runtime")
        for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "AI_API_KEY"):
            with self.subTest(key=key):
                self.assertEqual(values[key], "")

    def test_gitignore_covers_secrets_and_biometrics(self):
        text = (PROJECT / ".gitignore").read_text(encoding="utf-8")
        for pattern in (
            ".env",
            ".cache/",
            ".runtime/",
            "data/training_images/**",
            "__pycache__/",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, text)

    def test_voice_service_never_logs_transcript_content(self):
        text = (PROJECT / "voice_service.py").read_text(encoding="utf-8")
        for forbidden in (
            'print(f"Question:',
            'print(f"Answer:',
            "Ignored without wake word",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_readme_relative_links_exist(self):
        text = (PROJECT / "README.md").read_text(encoding="utf-8")
        links = re.findall(r"\]\(([^)\s]+)\)", text)
        for link in links:
            if link.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = link.split("#", 1)[0]
            if not target:
                continue
            with self.subTest(link=link):
                self.assertTrue(
                    (PROJECT / target).exists(),
                    f"README link target does not exist: {target}",
                )

    def test_required_documentation_exists(self):
        for name in (
            "README.md",
            "CHANGELOG.md",
            "LICENSE",
            "docs/architecture.md",
            "docs/configuration.md",
            "docs/knowledge-base.md",
            "docs/operations.md",
            "docs/privacy.md",
            "docs/testing.md",
            "docs/model-card.md",
            "data/training_images/README.md",
            ".github/workflows/ci.yml",
            "setup_windows.ps1",
            "cleanup_runtime.ps1",
        ):
            with self.subTest(name=name):
                self.assertTrue((PROJECT / name).is_file(), f"missing {name}")


if __name__ == "__main__":
    unittest.main()
