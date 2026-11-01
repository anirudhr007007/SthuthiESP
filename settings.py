"""Runtime configuration for the face assistant.

Configuration is environment based so that API keys and machine-specific
paths do not have to be stored in source control.  A local ``.env`` file is
loaded when present (and is git-ignored); real environment variables always
take precedence over values from that file.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping

_STT_ALIASES = {
    "": "none",
    "off": "none",
    "disabled": "none",
    "disable": "none",
    "none": "none",
    "google": "google",
    "google-cloud": "google",
    "google_cloud": "google",
    "google-web": "google",
    "google_web": "google",
    "google speech": "google",
    "google-speech": "google",
    "whisper": "whisper",
    "faster-whisper": "whisper",
    "faster_whisper": "whisper",
    "local-whisper": "whisper",
    "local_whisper": "whisper",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using {default}.")
        return default


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using {default}.")
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using {default}.")
        return default


def _env_text(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_unit_float(
    name: str, default: float, *, minimum: float = 0.0, maximum: float = 1.0
) -> float:
    value = _env_float(name, default)
    if value < minimum or value > maximum:
        print(f"{name}={value!r} is outside {minimum}-{maximum}; clamping it.")
        return min(maximum, max(minimum, value))
    return value


def _project_relative(project_path: Path, name: str, default: str) -> Path:
    configured = Path(os.getenv(name, default)).expanduser()
    return configured if configured.is_absolute() else project_path / configured


def load_env_file(
    path: Path, environ: MutableMapping[str, str] | None = None
) -> int:
    """Load simple ``KEY=value`` lines from ``path`` without overriding env.

    The parser deliberately supports only the subset needed by this project:
    comments, optional ``export`` prefixes, single/double quotes, and inline
    comments after whitespace.  Malformed lines are ignored rather than
    guessing at their meaning.  Returns the number of values applied.
    """

    target = os.environ if environ is None else environ
    env_path = Path(path)
    try:
        text = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return 0

    applied = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if value.startswith(("'", '"')):
            try:
                parsed = shlex.split(value, comments=False, posix=True)
            except ValueError:
                continue
            value = parsed[0] if parsed else ""
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if key not in target:
            target[key] = value
            applied += 1
    return applied


def normalize_stt_provider(value: str) -> str:
    cleaned = str(value).strip().lower()
    return _STT_ALIASES.get(cleaned, cleaned)


_normalise_stt_provider = normalize_stt_provider


@dataclass(frozen=True)
class Settings:
    project_path: Path
    training_path: Path
    knowledge_path: Path
    embedding_cache_path: Path
    use_embedding_cache: bool
    college_name: str

    # Face recognition
    face_model: str
    face_detector: str
    similarity_threshold: float
    match_margin: float
    recognition_interval: float
    max_frame_width: int
    identity_confirmations: int
    recognition_result_max_age: float

    # Camera
    camera_index: int
    camera_width: int
    camera_height: int
    camera_fps: int
    camera_buffer_size: int
    camera_read_retries: int
    camera_max_read_failures: int

    # Voice input and presence behaviour
    enable_voice: bool
    stt_provider: str
    stt_language: str
    stt_device_index: int | None
    stt_energy_threshold: int | None
    stt_pause_threshold: float
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    listen_timeout: float
    phrase_time_limit: int
    wake_word: str
    echo_guard: bool
    echo_similarity_threshold: float
    echo_idle_grace: float
    echo_recent_window: float
    voice_debug: bool

    # Speech output
    speech_output: str
    speech_engine: str
    speech_voice: str
    speech_rate: int
    speech_volume: float
    speech_queue_size: int
    startup_speech: bool

    # Greeting / presence
    greeting_min_gap: float
    presence_timeout: float
    visitor_welcome_cooldown: float

    # Answer provider
    ai_provider: str
    ai_timeout: float
    ollama_url: str
    ollama_model: str
    gemini_model: str
    openai_model: str
    openai_api_key: str
    gemini_api_key: str
    groq_model: str
    groq_api_key: str
    ai_base_url: str
    ai_api_key: str
    ai_model: str
    cloud_min_interval: float
    cloud_max_calls_per_hour: int
    provider_probe_timeout: float
    web_search: bool

    @classmethod
    def from_env(
        cls, project_path: Path, *, load_env: bool = True
    ) -> "Settings":
        project_path = Path(project_path).resolve()
        if load_env:
            applied = load_env_file(project_path / ".env")
            if applied:
                print(f"Loaded {applied} setting(s) from .env.")

        return cls(
            project_path=project_path,
            training_path=_project_relative(
                project_path, "TRAINING_PATH", "data/training_images"
            ),
            knowledge_path=_project_relative(
                project_path, "KNOWLEDGE_PATH", "data/college_knowledge.json"
            ),
            embedding_cache_path=_project_relative(
                project_path, "EMBEDDING_CACHE_PATH", ".cache/embeddings.json"
            ),
            use_embedding_cache=_env_bool("USE_EMBEDDING_CACHE", True),
            college_name=os.getenv(
                "COLLEGE_NAME", "Malnad College of Engineering, Hassan"
            ).strip(),
            face_model=_env_text("FACE_MODEL", "VGG-Face"),
            face_detector=_env_text("FACE_DETECTOR", "opencv"),
            similarity_threshold=_env_unit_float("SIMILARITY_THRESHOLD", 0.68),
            match_margin=_env_unit_float("MATCH_MARGIN", 0.08),
            recognition_interval=max(
                0.1, _env_float("RECOGNITION_INTERVAL", 0.35)
            ),
            max_frame_width=max(0, _env_int("MAX_FRAME_WIDTH", 640)),
            identity_confirmations=max(1, _env_int("IDENTITY_CONFIRMATIONS", 3)),
            recognition_result_max_age=max(
                0.2, _env_float("RECOGNITION_RESULT_MAX_AGE", 1.5)
            ),
            camera_index=_env_int("CAMERA_INDEX", 0),
            camera_width=max(0, _env_int("CAMERA_WIDTH", 1280)),
            camera_height=max(0, _env_int("CAMERA_HEIGHT", 720)),
            camera_fps=max(0, _env_int("CAMERA_FPS", 30)),
            camera_buffer_size=max(0, _env_int("CAMERA_BUFFER_SIZE", 1)),
            camera_read_retries=max(1, _env_int("CAMERA_READ_RETRIES", 5)),
            camera_max_read_failures=max(
                1, _env_int("CAMERA_MAX_READ_FAILURES", 10)
            ),
            enable_voice=_env_bool("ENABLE_VOICE", True),
            stt_provider=normalize_stt_provider(
                os.getenv("STT_PROVIDER", "none")
            ),
            stt_language=_env_text("STT_LANGUAGE", "en-IN") or "en-IN",
            stt_device_index=_env_optional_int("STT_DEVICE_INDEX"),
            stt_energy_threshold=_env_optional_int("STT_ENERGY_THRESHOLD"),
            stt_pause_threshold=max(0.2, _env_float("STT_PAUSE_THRESHOLD", 0.8)),
            whisper_model=_env_text("WHISPER_MODEL", "small") or "small",
            whisper_device=_env_text("WHISPER_DEVICE", "cpu") or "cpu",
            whisper_compute_type=(
                _env_text("WHISPER_COMPUTE_TYPE", "int8") or "int8"
            ),
            listen_timeout=max(0.5, _env_float("LISTEN_TIMEOUT", 1.0)),
            phrase_time_limit=max(1, _env_int("PHRASE_TIME_LIMIT", 8)),
            wake_word=_env_text("WAKE_WORD", ""),
            echo_guard=_env_bool("ECHO_GUARD", True),
            echo_similarity_threshold=min(
                0.99,
                max(0.3, _env_float("ECHO_SIMILARITY_THRESHOLD", 0.72)),
            ),
            echo_idle_grace=max(0.0, _env_float("ECHO_IDLE_GRACE", 0.6)),
            echo_recent_window=max(2.0, _env_float("ECHO_RECENT_WINDOW", 20.0)),
            voice_debug=_env_bool("VOICE_DEBUG", False),
            speech_output=_env_text("SPEECH_OUTPUT", ""),
            speech_engine=_env_text("SPEECH_ENGINE", "sapi").lower() or "sapi",
            speech_voice=_env_text("SPEECH_VOICE", ""),
            speech_rate=min(320, max(80, _env_int("SPEECH_RATE", 165))),
            speech_volume=min(1.0, max(0.0, _env_float("SPEECH_VOLUME", 1.0))),
            speech_queue_size=max(1, _env_int("SPEECH_QUEUE_SIZE", 25)),
            startup_speech=_env_bool("STARTUP_SPEECH", True),
            greeting_min_gap=max(0.0, _env_float("GREETING_MIN_GAP", 2.0)),
            presence_timeout=max(
                1.0, _env_float("PRESENCE_TIMEOUT", 8.0)
            ),
            visitor_welcome_cooldown=max(
                0.0, _env_float("VISITOR_WELCOME_COOLDOWN", 20.0)
            ),
            ai_provider=_env_text("AI_PROVIDER", "ollama").lower() or "ollama",
            ai_timeout=max(2.0, _env_float("AI_TIMEOUT", 15.0)),
            ollama_url=(
                _env_text("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
            ),
            ollama_model=_env_text("OLLAMA_MODEL", "llama3.2:3b"),
            gemini_model=_env_text("GEMINI_MODEL", "gemini-2.5-flash"),
            openai_model=_env_text("OPENAI_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            groq_model=_env_text("GROQ_MODEL", "llama-3.3-70b-versatile"),
            groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
            ai_base_url=os.getenv("AI_BASE_URL", "").strip().rstrip("/"),
            ai_api_key=os.getenv("AI_API_KEY", "").strip(),
            ai_model=os.getenv("AI_MODEL", "").strip(),
            cloud_min_interval=max(
                0.0, _env_float("CLOUD_MIN_INTERVAL", 3.0)
            ),
            cloud_max_calls_per_hour=max(
                0, _env_int("CLOUD_MAX_CALLS_PER_HOUR", 60)
            ),
            provider_probe_timeout=max(
                0.5, _env_float("PROVIDER_PROBE_TIMEOUT", 3.0)
            ),
            web_search=_env_bool("WEB_SEARCH", True),
        )

    def describe(self) -> dict[str, str]:
        """Return a safe, non-secret summary for startup logs and doctors."""

        if not self.enable_voice:
            voice = "disabled"
        elif self.stt_provider == "none":
            voice = "speech output only (microphone off)"
        elif self.wake_word:
            voice = f"{self.stt_provider} (wake word enabled)"
        else:
            voice = self.stt_provider
        return {
            "college": self.college_name,
            "training": str(self.training_path),
            "knowledge": str(self.knowledge_path),
            "face_model": f"{self.face_model}/{self.face_detector}",
            "recognition": (
                f"threshold {self.similarity_threshold:.2f}, "
                f"margin {self.match_margin:.2f}, "
                f"interval {self.recognition_interval:.2f}s, "
                f"confirmations {self.identity_confirmations}"
            ),
            "camera": (
                f"index {self.camera_index}, "
                f"{self.camera_width}x{self.camera_height} @ {self.camera_fps} fps"
            ),
            "voice": voice,
            "echo_guard": "on" if self.echo_guard else "off",
            "web_search": "on" if self.web_search else "off",
            "ai_provider": self.ai_provider,
            "embedding_cache": (
                str(self.embedding_cache_path)
                if self.use_embedding_cache
                else "disabled"
            ),
        }
