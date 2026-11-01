"""Keep model, Hugging Face, Keras, and temporary caches inside the project.

The application is self-contained by design: installing dependencies must not
modify global Python packages, and first-run model downloads must not land in
the user profile.  ``configure_runtime_environment()`` is called before any
heavy import so DeepFace, Keras, faster-whisper, and their transitive
dependencies resolve their caches under ``.runtime/``.

Explicit values are used instead of ``setdefault`` so a pre-existing global
cache setting cannot silently move downloaded models outside the checkout.
Operators can choose a different root with ``ESSPTHUTHI_RUNTIME_DIR``, but it
must resolve inside the project; out-of-tree values fall back to ``.runtime``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

PROJECT_PATH = Path(__file__).resolve().parent
DEFAULT_RUNTIME_DIR = ".runtime"


def runtime_root(
    project_path: Path | str = PROJECT_PATH,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Return the project-local runtime root without requiring it to exist.

    A configured root must stay inside the checkout.  This preserves the
    guarantee that installing or running the app does not scatter caches into a
    user profile or another drive.  Out-of-tree values fall back to the default
    ``.runtime`` directory.
    """

    source = os.environ if environ is None else environ
    project = Path(project_path).resolve()
    fallback = project / DEFAULT_RUNTIME_DIR
    configured = source.get("ESSPTHUTHI_RUNTIME_DIR", "").strip()
    if not configured:
        return fallback
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = project / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(project)
    except (OSError, RuntimeError, ValueError):
        return fallback
    if resolved == project:
        return fallback
    return resolved


def runtime_environment(
    project_path: Path | str = PROJECT_PATH,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the project-local cache environment without mutating it."""

    root = runtime_root(project_path, environ)
    return {
        "DEEPFACE_HOME": str(root / "deepface"),
        "HF_HOME": str(root / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(root / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(root / "huggingface" / "transformers"),
        "KERAS_HOME": str(root / "keras"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "MPLCONFIGDIR": str(root / "matplotlib"),
        "NUMBA_CACHE_DIR": str(root / "numba"),
        "TORCH_HOME": str(root / "torch"),
        "PIP_CACHE_DIR": str(root / "pip-cache"),
        "TMP": str(root / "tmp"),
        "TEMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
    }


def configure_runtime_environment(
    project_path: Path | str = PROJECT_PATH,
    environ: MutableMapping[str, str] | None = None,
    *,
    create: bool = True,
) -> dict[str, str]:
    """Point every supported cache at ``.runtime`` and create the folders."""

    values = runtime_environment(project_path, environ)
    target = os.environ if environ is None else environ
    target.update(values)
    if create:
        for location in set(values.values()):
            Path(location).mkdir(parents=True, exist_ok=True)
    return values


def deepface_weights_dir(
    project_path: Path | str = PROJECT_PATH,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Return the DeepFace weight directory used by this project.

    DeepFace appends ``.deepface/weights`` to its ``DEEPFACE_HOME`` value, so
    the resulting path is nested one level below our cache root.
    """

    return (
        Path(runtime_environment(project_path, environ)["DEEPFACE_HOME"])
        / ".deepface"
        / "weights"
    )


def whisper_cache_dir(
    project_path: Path | str = PROJECT_PATH,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Return the Hugging Face hub directory used by faster-whisper."""

    return Path(runtime_environment(project_path, environ)["HUGGINGFACE_HUB_CACHE"])
