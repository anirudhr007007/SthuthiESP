"""Pre-flight diagnostics for camera, voice, data, and answer providers."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import platform
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_provider import filter_verified_knowledge, load_knowledge, probe_provider
from enrollment import dataset_summary, load_embedding_cache
from runtime_env import deepface_weights_dir, runtime_root, whisper_cache_dir
from settings import Settings

_DEEPFACE_WEIGHT_FILES = {
    "vgg-face": "vgg_face_weights.h5",
    "facenet": "facenet_weights.h5",
    "facenet512": "facenet512_weights.h5",
    "arcface": "arcface_weights.h5",
    "sface": "sface_weights.h5",
    "ghostfacenet": "ghostfacenet_weights.h5",
    "dlib": "dlib_face_recognition_resnet_model_v1.dat",
    "openface": "openface_weights.h5",
    "deepid": "deepid_weights.h5",
}


@dataclass(frozen=True)
class CheckResult:
    level: str  # OK, WARN, FAIL, SKIP
    name: str
    detail: str


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def opencv_gui_backend(build_information: str) -> str | None:
    """Return the OpenCV GUI backend name, or ``None`` for headless builds."""

    for raw_line in str(build_information).splitlines():
        line = raw_line.strip()
        if not line.upper().startswith("GUI:"):
            continue
        backend = line.partition(":")[2].strip()
        return backend if backend and backend.upper() != "NONE" else None
    return None


def _opencv_build_information() -> str:
    import cv2

    return str(cv2.getBuildInformation())


def _check_opencv_gui() -> CheckResult:
    try:
        build_information = _opencv_build_information()
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        return CheckResult("SKIP", "OpenCV GUI", f"could not inspect cv2: {error}")
    backend = opencv_gui_backend(build_information)
    if backend is None:
        return CheckResult(
            "FAIL",
            "OpenCV GUI",
            "headless build; the camera window cannot open. Reinstall the pinned "
            "opencv-python package (see docs/operations.md).",
        )
    return CheckResult("OK", "OpenCV GUI", f"{backend} backend available")


def _check_python() -> list[CheckResult]:
    version = sys.version_info
    detail = f"{platform.python_implementation()} {version.major}.{version.minor}.{version.micro}"
    if version < (3, 11):
        return [CheckResult("FAIL", "Python", f"{detail}; Python 3.11 or newer is required")]
    if version >= (3, 13):
        return [
            CheckResult(
                "WARN",
                "Python",
                f"{detail}; TensorFlow/DeepFace support is best on Python 3.11 or 3.12",
            )
        ]
    return [CheckResult("OK", "Python", detail)]


def _check_files(settings: Settings) -> list[CheckResult]:
    results: list[CheckResult] = []
    if settings.knowledge_path.is_file():
        results.append(CheckResult("OK", "Knowledge file", str(settings.knowledge_path)))
    else:
        results.append(
            CheckResult("FAIL", "Knowledge file", f"missing: {settings.knowledge_path}")
        )
    if settings.training_path.is_dir():
        results.append(CheckResult("OK", "Training folder", str(settings.training_path)))
    else:
        results.append(
            CheckResult("FAIL", "Training folder", f"missing: {settings.training_path}")
        )
    env_path = settings.project_path / ".env"
    if env_path.is_file():
        results.append(
            CheckResult(
                "OK",
                ".env",
                f"present at {env_path} (values are not printed; keep this file private)",
            )
        )
    else:
        results.append(
            CheckResult(
                "SKIP",
                ".env",
                "not present; using environment variables and defaults",
            )
        )
    return results


def _check_training(settings: Settings) -> list[CheckResult]:
    summary = dataset_summary(settings.training_path)
    if summary.people == 0:
        return [
            CheckResult(
                "FAIL",
                "Enrollment images",
                "no person folders with images were found",
            )
        ]
    detail = (
        f"{summary.people} people, {summary.images} images "
        f"({summary.min_images}-{summary.max_images} per person)"
    )
    results = [CheckResult("OK", "Enrollment images", detail)]
    if summary.min_images < 5:
        results.append(
            CheckResult(
                "WARN",
                "Enrollment depth",
                "fewer than 5 images for at least one person; 5-10 varied images "
                "per person improves recognition",
            )
        )
    readme = settings.training_path / "README.md"
    if not readme.is_file():
        results.append(
            CheckResult(
                "WARN",
                "Consent/retention note",
                f"add {readme.name} describing consent and deletion policy",
            )
        )
    return results


def _normalise_question(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _check_knowledge(facts: list[dict[str, Any]]) -> list[CheckResult]:
    verified = filter_verified_knowledge(facts)
    results = [
        CheckResult("OK", "Verified facts", f"{len(verified)} usable verified facts")
    ]
    alias_count = sum(len(item.get("aliases", [])) for item in verified)
    results.append(CheckResult("OK", "FAQ aliases", f"{alias_count} aliases"))

    phrases: Counter[str] = Counter()
    for item in verified:
        for phrase in [item.get("question", ""), *(item.get("aliases") or [])]:
            if isinstance(phrase, str) and phrase.strip():
                phrases[_normalise_question(phrase)] += 1
    duplicates = [phrase for phrase, count in phrases.items() if count > 1]
    if duplicates:
        results.append(
            CheckResult(
                "FAIL",
                "FAQ ambiguity",
                f"{len(duplicates)} question/alias phrase(s) map to more than one fact",
            )
        )
    else:
        results.append(
            CheckResult("OK", "FAQ ambiguity", "all questions and aliases are unique")
        )

    missing_metadata = [
        item.get("question", "?")
        for item in verified
        if not item.get("verified_on")
        or not item.get("reviewed_by")
        or not item.get("source_type")
    ]
    if missing_metadata:
        results.append(
            CheckResult(
                "WARN",
                "Knowledge provenance",
                f"{len(missing_metadata)} fact(s) lack verified_on/reviewed_by/source_type",
            )
        )
    else:
        results.append(
            CheckResult(
                "OK",
                "Knowledge provenance",
                "all facts carry a source type, review date, and reviewer",
            )
        )
    if not verified:
        results.append(
            CheckResult("FAIL", "Knowledge usability", "no explicit verified facts")
        )
    return results


def _check_modules(settings: Settings) -> list[CheckResult]:
    results: list[CheckResult] = []
    required = {
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "deepface": "DeepFace",
    }
    for module, label in required.items():
        available = _module_available(module)
        results.append(
            CheckResult(
                "OK" if available else "FAIL",
                label,
                "installed" if available else f"missing module '{module}'",
            )
        )
    if _module_available("cv2"):
        results.append(_check_opencv_gui())
    if _module_available("tensorflow") and _module_available("tf_keras"):
        results.append(CheckResult("OK", "TensorFlow/tf-keras", "both installed"))
    elif _module_available("tensorflow"):
        results.append(
            CheckResult(
                "WARN",
                "TensorFlow/tf-keras",
                "tensorflow is installed but tf-keras is missing; DeepFace may fail on Keras 3",
            )
        )
    else:
        results.append(
            CheckResult(
                "WARN",
                "TensorFlow/tf-keras",
                "not installed; DeepFace needs a supported TensorFlow runtime",
            )
        )

    if settings.enable_voice and settings.stt_provider != "none":
        for module, label in (
            ("speech_recognition", "SpeechRecognition"),
            ("pyaudio", "PyAudio"),
        ):
            available = _module_available(module)
            results.append(
                CheckResult(
                    "OK" if available else "FAIL",
                    label,
                    "installed" if available else f"missing module '{module}'",
                )
            )
    else:
        results.append(
            CheckResult("SKIP", "Microphone modules", "voice input is disabled")
        )

    if settings.enable_voice:
        if os.name == "nt":
            available = _module_available("win32com")
            results.append(
                CheckResult(
                    "OK" if available else "WARN",
                    "Speech output",
                    "Windows SAPI available" if available else "pywin32 is missing",
                )
            )
        else:
            available = _module_available("pyttsx3")
            results.append(
                CheckResult(
                    "OK" if available else "WARN",
                    "Speech output",
                    "pyttsx3 installed" if available else "pyttsx3 is missing",
                )
            )
    else:
        results.append(CheckResult("SKIP", "Speech output", "voice is disabled"))

    if settings.stt_provider == "whisper":
        available = _module_available("faster_whisper")
        results.append(
            CheckResult(
                "OK" if available else "FAIL",
                "Local Whisper",
                "faster-whisper installed"
                if available
                else "faster-whisper is required for STT_PROVIDER=whisper",
            )
        )
    if settings.ai_provider == "openai":
        available = _module_available("openai")
        results.append(
            CheckResult(
                "OK" if available else "WARN",
                "OpenAI SDK",
                "installed; HTTP fallback also exists"
                if available
                else "not installed; the built-in HTTP path will be used",
            )
        )
    return results


def _check_model_caches(settings: Settings) -> list[CheckResult]:
    """Report whether first-run model assets are already project-local."""

    results: list[CheckResult] = []
    runtime_path = runtime_root(settings.project_path)
    results.append(CheckResult("OK", "Runtime caches", str(runtime_path)))

    model_name = str(getattr(settings, "face_model", "VGG-Face")).strip()
    weight_name = _DEEPFACE_WEIGHT_FILES.get(model_name.casefold())
    weights_dir = deepface_weights_dir(settings.project_path)
    if weight_name is None:
        results.append(
            CheckResult(
                "SKIP",
                "DeepFace weights",
                f"no expected filename known for {model_name}; cache: {weights_dir}",
            )
        )
    else:
        weight_path = weights_dir / weight_name
        try:
            available = weight_path.is_file() and weight_path.stat().st_size > 0
        except OSError:
            available = False
        if available:
            size_mb = weight_path.stat().st_size / (1024 * 1024)
            results.append(
                CheckResult(
                    "OK",
                    "DeepFace weights",
                    f"{model_name} cached locally ({size_mb:.0f} MB)",
                )
            )
        else:
            results.append(
                CheckResult(
                    "WARN",
                    "DeepFace weights",
                    f"{model_name} is not cached; first use downloads about 580 MB. "
                    f"Expected: {weight_path}",
                )
            )

    if str(getattr(settings, "stt_provider", "none")) == "whisper":
        whisper_model = str(getattr(settings, "whisper_model", "small")).strip() or "small"
        hub = whisper_cache_dir(settings.project_path)
        pattern = f"models--Systran--faster-whisper-{whisper_model}"
        cached = (hub / pattern).is_dir()
        if cached:
            results.append(
                CheckResult(
                    "OK",
                    "Whisper model",
                    f"{whisper_model} cached locally in {hub}",
                )
            )
        else:
            results.append(
                CheckResult(
                    "WARN",
                    "Whisper model",
                    f"{whisper_model} is not cached; first use downloads it from "
                    f"Hugging Face. Expected: {hub / pattern}",
                )
            )
    else:
        results.append(
            CheckResult("SKIP", "Whisper model", "STT_PROVIDER is not whisper")
        )
    return results


def _check_embedding_cache(settings: Settings) -> list[CheckResult]:
    """Report whether the local embedding cache covers every enrollment image."""

    if not bool(getattr(settings, "use_embedding_cache", True)):
        return [
            CheckResult(
                "SKIP",
                "Embedding cache",
                "USE_EMBEDDING_CACHE is disabled; embeddings rebuild every run",
            )
        ]

    path = Path(getattr(settings, "embedding_cache_path", ""))
    summary = dataset_summary(Path(settings.training_path))
    if not path.is_file():
        return [
            CheckResult(
                "WARN",
                "Embedding cache",
                f"not built yet; the first run creates {path}",
            )
        ]

    with contextlib.redirect_stdout(io.StringIO()):
        cached = load_embedding_cache(
            path,
            model=str(getattr(settings, "face_model", "VGG-Face")),
            detector=str(getattr(settings, "face_detector", "opencv")),
            align=True,
        )
    if not cached:
        return [
            CheckResult(
                "WARN",
                "Embedding cache",
                f"empty or built by a different model/detector; rebuild at {path}",
            )
        ]
    if summary.images and len(cached) < summary.images:
        return [
            CheckResult(
                "WARN",
                "Embedding cache",
                f"{len(cached)} of {summary.images} enrollment images have cached "
                "embeddings; run --rebuild-embeddings and review skipped images",
            )
        ]
    return [
        CheckResult(
            "OK",
            "Embedding cache",
            f"{len(cached)} embeddings cached locally",
        )
    ]


def _probe_camera(settings: Settings) -> CheckResult:
    if not _module_available("cv2"):
        return CheckResult("SKIP", "Camera", "OpenCV is not installed")
    try:
        import cv2
    except Exception as error:
        return CheckResult("SKIP", "Camera", f"OpenCV could not be imported: {error}")

    indices: list[int] = []
    for index in (settings.camera_index, 0, 1, 2):
        if index not in indices:
            indices.append(index)
    for index in indices:
        capture = None
        try:
            capture = (
                cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if os.name == "nt"
                else cv2.VideoCapture(index)
            )
            if not capture.isOpened():
                continue
            for property_name, value in (
                ("CAP_PROP_FRAME_WIDTH", settings.camera_width),
                ("CAP_PROP_FRAME_HEIGHT", settings.camera_height),
                ("CAP_PROP_BUFFERSIZE", settings.camera_buffer_size),
            ):
                if value and hasattr(cv2, property_name):
                    try:
                        capture.set(getattr(cv2, property_name), value)
                    except Exception:
                        pass
            for _ in range(settings.camera_read_retries):
                success, frame = capture.read()
                if success and frame is not None:
                    height, width = frame.shape[:2]
                    return CheckResult(
                        "OK", "Camera", f"index {index} delivered a {width}x{height} frame"
                    )
            return CheckResult(
                "WARN", "Camera", f"index {index} opened but delivered no frame"
            )
        except Exception as error:
            return CheckResult("WARN", "Camera", f"camera probe failed: {error}")
        finally:
            if capture is not None:
                capture.release()
    return CheckResult("FAIL", "Camera", "no camera index could be opened")


def _check_microphone(settings: Settings) -> CheckResult:
    if not settings.enable_voice or settings.stt_provider == "none":
        return CheckResult("SKIP", "Microphone", "voice input is disabled")
    if not _module_available("speech_recognition"):
        return CheckResult("SKIP", "Microphone", "SpeechRecognition is not installed")
    try:
        import speech_recognition as sr
    except Exception as error:
        return CheckResult("SKIP", "Microphone", f"could not import: {error}")
    try:
        names = sr.Microphone.list_microphone_names()
    except Exception as error:
        return CheckResult("WARN", "Microphone", f"device list unavailable: {error}")
    if not names:
        return CheckResult("FAIL", "Microphone", "no input devices were reported")
    if settings.stt_device_index is not None and not (
        0 <= settings.stt_device_index < len(names)
    ):
        return CheckResult(
            "FAIL",
            "Microphone",
            f"STT_DEVICE_INDEX={settings.stt_device_index} is out of range "
            f"(0-{len(names) - 1})",
        )
    if settings.stt_device_index is not None:
        selected = names[settings.stt_device_index]
        return CheckResult(
            "OK",
            "Microphone",
            f"{len(names)} device(s); index {settings.stt_device_index}: {selected}",
        )
    return CheckResult(
        "OK",
        "Microphone",
        f"{len(names)} device(s); SpeechRecognition will use the system default input",
    )


def _check_provider(settings: Settings) -> CheckResult:
    try:
        status = probe_provider(settings, network=False)
    except Exception as error:
        return CheckResult("WARN", "Answer provider", f"probe failed: {error}")
    level = "OK" if status.available else "WARN"
    return CheckResult(level, "Answer provider", f"{status.provider}: {status.detail}")


def _check_ollama_model(settings: Settings) -> CheckResult:
    """Verify that Ollama serves the configured model (local request only)."""

    provider = str(getattr(settings, "ai_provider", "") or "").strip().lower()
    if provider != "ollama":
        return CheckResult("SKIP", "Ollama model", "AI_PROVIDER is not ollama")
    model = str(getattr(settings, "ollama_model", "") or "").strip()
    try:
        status = probe_provider(settings, network=False)
    except Exception as error:
        return CheckResult("WARN", "Ollama model", f"probe failed: {error}")
    if status.available:
        return CheckResult(
            "OK",
            "Ollama model",
            f"model '{model}' is served by Ollama ({status.detail})",
        )
    return CheckResult(
        "FAIL",
        "Ollama model",
        f"{status.detail}; run `ollama serve` and `ollama pull {model}` "
        "before the demo",
    )


def _check_web_search(settings: Settings) -> CheckResult:
    """Verify the optional web-search dependency without a network request."""

    if not bool(getattr(settings, "web_search", False)):
        return CheckResult("SKIP", "Web search", "WEB_SEARCH is not enabled")
    if _module_available("ddgs"):
        return CheckResult("OK", "Web search", "ddgs installed")
    return CheckResult(
        "WARN",
        "Web search",
        "WEB_SEARCH=true but the ddgs package is missing; unmatched questions "
        "will answer without snippets. Install requirements-cloud.txt or set "
        "WEB_SEARCH=false.",
    )


def run_doctor(settings: Settings, facts: list[dict[str, Any]] | None = None) -> int:
    """Print diagnostics and return nonzero only for blocking failures."""

    if facts is None:
        facts = load_knowledge(settings.knowledge_path)

    results: list[CheckResult] = []
    results.extend(_check_python())
    results.extend(_check_files(settings))
    results.extend(_check_training(settings))
    results.extend(_check_knowledge(facts))
    results.extend(_check_modules(settings))
    results.extend(_check_model_caches(settings))
    results.extend(_check_embedding_cache(settings))
    results.append(_probe_camera(settings))
    results.append(_check_microphone(settings))
    results.append(_check_provider(settings))
    results.append(_check_ollama_model(settings))
    results.append(_check_web_search(settings))

    print("\nDiagnostics")
    print("----------")
    for result in results:
        print(f"[{result.level:4}] {result.name}: {result.detail}")

    failures = sum(1 for result in results if result.level == "FAIL")
    warnings = sum(1 for result in results if result.level == "WARN")
    print(
        f"\nSummary: {len(results)} checks, {failures} failure(s), "
        f"{warnings} warning(s)."
    )
    if failures:
        print("Fix the failures above before running the camera demo.")
    elif warnings:
        print("Core checks passed; review the warnings for a smoother demo.")
    else:
        print("All checks passed.")
    return 1 if failures else 0
