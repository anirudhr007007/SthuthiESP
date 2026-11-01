"""College face-recognition and voice assistant.

Run on a normal Windows or Linux computer with a camera, speakers, and an
optional microphone.  Face recognition and verified FAQ answers stay local.
Cloud answer providers and Google speech recognition are opt-in; the default
``STT_PROVIDER`` is ``none``.  Use ``--offline`` for a guaranteed no-cloud
camera demo and ``--doctor`` before an event.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from ai_provider import answer_question_result, load_knowledge
from enrollment import load_known_embeddings
from face_utils import (
    MatchResult,
    StableIdentityTracker,
    face_box,
    identify,
    is_plausible_face_area,
    normalize,
)
from greeting import GreetingController
from recognition_worker import LatestFrameWorker
from runtime_env import configure_runtime_environment
from settings import Settings, normalize_stt_provider
from speech_service import SpeechService
from ui_state import AssistantState
from version import __version__
from voice_service import is_greeting, listen_for_questions

PROJECT_PATH = Path(__file__).resolve().parent


def recognize_frame(
    frame: Any,
    deepface: Any,
    known_embeddings: dict[str, list[tuple[float, ...]]],
    settings: Settings,
) -> list[tuple[dict[str, Any], MatchResult]]:
    """Detect and identify faces in one camera frame.

    Video frames use ``enforce_detection=False`` because an empty room is a
    normal condition, not an error.  DeepFace's synthetic full-frame fallback
    is filtered out so it cannot be enrolled as a person.
    """

    representations = deepface.represent(
        frame,
        model_name=settings.face_model,
        detector_backend=settings.face_detector,
        align=True,
        enforce_detection=False,
    )
    if isinstance(representations, dict):
        representations = [representations]
    if not isinstance(representations, (list, tuple)):
        return []

    frame_height, frame_width = frame.shape[:2]
    results: list[tuple[dict[str, Any], MatchResult]] = []
    for face in representations:
        if not isinstance(face, dict):
            continue
        area = face.get("facial_area")
        if not isinstance(area, dict) or not is_plausible_face_area(
            area, frame_width, frame_height
        ):
            continue
        confidence = face.get("confidence")
        if confidence is not None:
            try:
                if float(confidence) <= 0:
                    continue
            except (TypeError, ValueError, OverflowError):
                pass
        embedding = normalize(face.get("embedding", []))
        if not embedding or not any(value != 0 for value in embedding):
            continue
        result = identify(
            embedding,
            known_embeddings,
            settings.similarity_threshold,
            settings.match_margin,
        )
        results.append((face, result))
    return results


def _configure_capture(capture: Any, cv2: Any, settings: Settings) -> None:
    for property_name, value in (
        ("CAP_PROP_FRAME_WIDTH", settings.camera_width),
        ("CAP_PROP_FRAME_HEIGHT", settings.camera_height),
        ("CAP_PROP_FPS", settings.camera_fps),
        ("CAP_PROP_BUFFERSIZE", settings.camera_buffer_size),
    ):
        if not value:
            continue
        property_id = getattr(cv2, property_name, None)
        if property_id is None:
            continue
        try:
            capture.set(property_id, value)
        except Exception:
            pass


def open_camera(cv2: Any, settings: Settings) -> Any | None:
    indices: list[int] = []
    for index in (settings.camera_index, 0, 1, 2):
        if index not in indices:
            indices.append(index)

    for index in indices:
        capture = (
            cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if os.name == "nt"
            else cv2.VideoCapture(index)
        )
        if not capture.isOpened():
            capture.release()
            print(f"Camera index {index} unavailable; trying the next camera.")
            continue
        _configure_capture(capture, cv2, settings)
        for _ in range(settings.camera_read_retries):
            success, frame = capture.read()
            if success and frame is not None:
                height, width = frame.shape[:2]
                print(f"Camera ready: index {index} ({width}x{height})")
                return capture
            time.sleep(0.1)
        capture.release()
        print(f"Camera index {index} opened but delivered no frame.")
    return None


def _display_text(value: str) -> str:
    """Keep OpenCV's Hershey font limited to characters it can render."""

    return str(value).encode("ascii", "replace").decode("ascii")


def _wrap_text(value: str, width: int = 78, max_lines: int = 3) -> list[str]:
    words = _display_text(value).split()
    if not words:
        return []
    width = max(12, int(width))
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
        else:
            current = candidate
    if len(lines) < max_lines and current:
        lines.append(current)
    if len(lines) == max_lines:
        consumed = " ".join(lines)
        if len(_display_text(value)) > len(consumed):
            last = lines[-1]
            lines[-1] = (last[: max(0, width - 3)].rstrip() + "...") if last else "..."
    return lines[:max_lines]


def draw_status(
    frame: Any,
    cv2: Any,
    activity: str,
    answer: str,
    *,
    has_faces: bool,
) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 36), (18, 18, 18), -1)
    status = f"Status: {activity}"
    if not has_faces:
        status += " | No face detected"
    color = (0, 210, 0) if activity.lower().startswith("listen") else (0, 200, 255)
    cv2.putText(
        frame,
        _display_text(status)[:70],
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
    cv2.putText(
        frame,
        "Q: quit",
        (max(10, width - 105), 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    lines = _wrap_text(answer, max(30, width // 9), 3)
    if lines:
        line_height = 22
        box_height = line_height * len(lines) + 12
        top = max(38, height - box_height - 8)
        cv2.rectangle(frame, (0, top), (width, height), (18, 18, 18), -1)
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (10, top + 20 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
            )


def run_camera(
    settings: Settings,
    known_embeddings: dict[str, list[tuple[float, ...]]],
    speech: SpeechService,
    stop_event: threading.Event,
    state: AssistantState,
) -> None:
    try:
        import cv2
        from deepface import DeepFace
    except ImportError as error:
        raise RuntimeError(
            "Camera dependencies are missing. Install requirements.txt first."
        ) from error

    capture = open_camera(cv2, settings)
    if capture is None:
        stop_event.set()
        raise RuntimeError(
            "No camera could be opened. Check camera permissions and CAMERA_INDEX."
        )

    recognition_worker = LatestFrameWorker(
        lambda image: recognize_frame(image, DeepFace, known_embeddings, settings)
    )
    recognition_worker.start()

    cached_faces: list[tuple[dict[str, Any], MatchResult]] = []
    cached_faces_at = 0.0
    confirmed_names: set[str] = set()
    last_submission = 0.0
    last_worker_error = ""
    stale_reported = False
    read_failures = 0
    identity_tracker = StableIdentityTracker(
        confirmations=settings.identity_confirmations
    )
    greeting = GreetingController(
        settings.college_name,
        presence_timeout=settings.presence_timeout,
        visitor_welcome_cooldown=settings.visitor_welcome_cooldown,
        min_greeting_gap=settings.greeting_min_gap,
    )

    try:
        print("Camera started. Press Q or Esc to stop.")
        if settings.startup_speech and speech.enabled:
            speech.speak("The college assistant is ready.")
        while not stop_event.is_set():
            success, frame = capture.read()
            if not success or frame is None:
                read_failures += 1
                if read_failures >= settings.camera_max_read_failures:
                    print("Camera stopped delivering frames.")
                    break
                time.sleep(0.05)
                continue
            read_failures = 0

            if settings.max_frame_width and frame.shape[1] > settings.max_frame_width:
                scale = settings.max_frame_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (settings.max_frame_width, int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            now = time.monotonic()
            if now - last_submission >= settings.recognition_interval:
                # Inference is deliberately off the camera/UI loop.  The
                # worker keeps only the newest frame under load.
                recognition_worker.submit(frame.copy())
                last_submission = now

            batch = recognition_worker.poll()
            if batch is not None:
                if batch.error:
                    cached_faces = []
                    cached_faces_at = 0.0
                    confirmed_names = identity_tracker.update([])
                    if batch.error != last_worker_error:
                        print(f"Face recognition status: {batch.error}")
                        last_worker_error = batch.error
                    stale_reported = False
                elif now - batch.completed_at <= settings.recognition_result_max_age:
                    last_worker_error = ""
                    cached_faces = batch.results or []
                    cached_faces_at = batch.completed_at
                    confirmed_names = identity_tracker.update(
                        [match for _, match in cached_faces]
                    )
                    stale_reported = False
                else:
                    cached_faces = []
                    cached_faces_at = 0.0
                    confirmed_names = identity_tracker.update([])
                    if not stale_reported:
                        print("Discarded a stale face-recognition result.")
                        stale_reported = True

            if (
                cached_faces
                and cached_faces_at
                and now - cached_faces_at > settings.recognition_result_max_age
            ):
                cached_faces = []
                cached_faces_at = 0.0
                confirmed_names = identity_tracker.update([])
                if not stale_reported:
                    print("Face-recognition results expired.")
                    stale_reported = True

            has_unknown = any(not match.accepted for _, match in cached_faces)
            for action in greeting.update(
                confirmed_names,
                has_unknown,
                now=now,
            ):
                speech.speak(action.message)

            for face, match in cached_faces:
                area = face_box(face["facial_area"])
                if area is None:
                    continue
                x, y, width, height = area
                color = (0, 200, 0) if match.accepted else (0, 165, 255)
                label = (
                    f"{match.name} ({match.score:.2f})"
                    if match.score >= 0
                    else match.name
                )
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    frame,
                    _display_text(label),
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

            activity, answer = state.snapshot()
            if speech.is_speaking:
                activity = "Speaking"
            draw_status(
                frame,
                cv2,
                activity,
                answer,
                has_faces=bool(cached_faces),
            )
            cv2.imshow("College Face Assistant", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                stop_event.set()
                break
    finally:
        recognition_worker.stop(timeout=2)
        capture.release()
        stop_event.set()
        state.set_activity("Stopped")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Camera stopped.")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the college face-recognition assistant"
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="disable the microphone and speech output",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        help="override CAMERA_INDEX for this run",
    )
    parser.add_argument(
        "--ai-provider",
        help="override AI_PROVIDER (ollama, offline, gemini, openai, groq, nvidia, compatible)",
    )
    parser.add_argument(
        "--stt-provider",
        help="override STT_PROVIDER (none, whisper, or google)",
    )
    parser.add_argument(
        "--wake-word",
        help="require this leading phrase before answering (empty disables)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "disable the cloud answer provider and Google STT; local Whisper "
            "voice input remains available if configured"
        ),
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="answer one question locally, print it, and exit (no camera)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="start a text-only headless chat (no camera, microphone, or speech)",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="check dependencies, camera, microphone, data, and provider configuration",
    )
    parser.add_argument(
        "--warmup-models",
        action="store_true",
        help=(
            "download/load the configured face model and local Whisper model, "
            "then exit (no camera)"
        ),
    )
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="ignore the local embedding cache and recompute enrollment",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the project version and exit",
    )
    return parser.parse_args()


def _apply_overrides(
    settings: Settings, args: argparse.Namespace
) -> tuple[Settings, list[str]]:
    notes: list[str] = []
    if args.no_voice:
        settings = replace(settings, enable_voice=False)
        notes.append("Voice input and speech output are disabled for this run.")
    if args.camera_index is not None:
        settings = replace(settings, camera_index=args.camera_index)
    if args.ai_provider:
        settings = replace(settings, ai_provider=args.ai_provider.strip().lower())
    if args.stt_provider is not None:
        settings = replace(
            settings,
            stt_provider=normalize_stt_provider(args.stt_provider),
        )
    if args.wake_word is not None:
        settings = replace(settings, wake_word=args.wake_word.strip())
    if args.offline:
        settings = replace(settings, ai_provider="offline")
        if settings.stt_provider == "google":
            settings = replace(settings, stt_provider="none")
            notes.append(
                "Offline mode disabled Google speech recognition. "
                "Use STT_PROVIDER=whisper for local voice input."
            )
        notes.append(
            "Offline mode: answers come from verified local facts only; "
            "no cloud answer provider is used. First-run DeepFace or local "
            "Whisper model downloads are not blocked, so pre-download them "
            "before an offline demo."
        )
    return settings, notes


def warm_up_models(
    settings: Settings,
    *,
    deepface_builder: Callable[[], Any] | None = None,
    whisper_builder: Callable[[], Any] | None = None,
) -> int:
    """Load the configured face and speech models so caches are warm.

    This deliberately has no camera or microphone dependency.  The face model
    is built by DeepFace itself, which downloads official weights on the first
    run and reuses the project-local cache afterwards.
    """

    try:
        if deepface_builder is None:
            from deepface import DeepFace

            def build_deepface_model() -> Any:
                return DeepFace.build_model(settings.face_model)

            deepface_builder = build_deepface_model
        deepface_builder()
        print(f"Face model ready: {settings.face_model}")
    except Exception as error:
        print(f"Face model warm-up failed: {error}")
        return 1

    if settings.stt_provider == "whisper":
        try:
            if whisper_builder is None:
                from faster_whisper import WhisperModel

                def build_whisper_model() -> Any:
                    return WhisperModel(
                        settings.whisper_model,
                        device=settings.whisper_device,
                        compute_type=settings.whisper_compute_type,
                    )

                whisper_builder = build_whisper_model
            whisper_builder()
            print(f"Whisper model ready: {settings.whisper_model}")
        except Exception as error:
            print(f"Whisper model warm-up failed: {error}")
            return 1
    else:
        print("Whisper model skipped: STT_PROVIDER is not whisper.")

    print("Model warm-up complete. All caches stay under .runtime.")
    return 0


def run_chat(
    settings: Settings,
    facts: list[dict[str, Any]],
    *,
    ask: Any = input,
    output: Any = print,
) -> int:
    """Run a text-only chat without importing the camera or speech stack."""

    output("Text chat started. Type /quit, quit, or exit to stop.")
    while True:
        try:
            raw_question = ask("You: ")
        except (EOFError, KeyboardInterrupt):
            output("")
            break
        question = " ".join(str(raw_question).split()).strip()
        if not question:
            continue
        if question.casefold() in {"/quit", "quit", "exit"}:
            break
        if is_greeting(question):
            output("Assistant [greeting]: Hello! How can I help you?")
            continue
        result = answer_question_result(question, settings, facts)
        output(f"Assistant [{result.source}]: {result.text}")
    output("Text chat stopped.")
    return 0


def _print_startup_summary(settings: Settings) -> None:
    print(f"College face assistant {__version__}")
    for name, value in settings.describe().items():
        print(f"  {name}: {value}")
    if not settings.enable_voice:
        print("Speech output is disabled for this run.")
    elif settings.stt_provider == "none":
        print("Microphone input is off (STT_PROVIDER=none); speech output is on.")
    if settings.stt_provider == "google":
        print(
            "Privacy: STT_PROVIDER=google sends captured microphone audio to "
            "Google. Use whisper or none to keep audio local."
        )
    if settings.ai_provider not in {"offline", "none", "local-offline"}:
        print(
            "Privacy: verified FAQ answers stay local, but general questions "
            f"may be sent to the '{settings.ai_provider}' provider."
        )


def main() -> int:
    args = _parse_arguments()
    if args.version:
        print(f"espsthuthi2 {__version__}")
        return 0

    settings = Settings.from_env(PROJECT_PATH)
    configure_runtime_environment(PROJECT_PATH)
    settings, notes = _apply_overrides(settings, args)
    for note in notes:
        print(note)

    facts = load_knowledge(settings.knowledge_path)
    if args.doctor:
        from doctor import run_doctor

        return run_doctor(settings, facts)

    selected_mode = (
        int(args.ask is not None)
        + int(bool(args.chat))
        + int(bool(args.warmup_models))
    )
    if selected_mode > 1:
        print("Use only one of --ask, --chat, or --warmup-models.")
        return 2
    if args.warmup_models:
        return warm_up_models(settings)
    if args.ask is not None:
        question = args.ask.strip()
        if not question:
            print("--ask requires a non-empty question.")
            return 2
        if is_greeting(question):
            print("Hello! How can I help you?")
            return 0
        print(answer_question_result(question, settings, facts).text)
        return 0
    if args.chat:
        return run_chat(settings, facts)

    _print_startup_summary(settings)
    if not facts:
        print(
            "Warning: no verified knowledge facts loaded from "
            f"{settings.knowledge_path}."
        )

    try:
        known_embeddings = load_known_embeddings(
            settings, force_rebuild=args.rebuild_embeddings
        )
    except RuntimeError as error:
        print(f"Startup error: {error}")
        return 1

    speech = SpeechService(
        settings.enable_voice,
        settings.speech_output,
        voice_name=settings.speech_voice,
        rate=settings.speech_rate,
        volume=settings.speech_volume,
        max_queue=settings.speech_queue_size,
        idle_grace=settings.echo_idle_grace,
        recent_window=settings.echo_recent_window,
    )
    state = AssistantState("Starting")
    stop_event = threading.Event()
    listener: threading.Thread | None = None

    if settings.enable_voice:
        speech.start()
    if settings.enable_voice and settings.stt_provider != "none":
        state.set_activity("Listening")
        listener = threading.Thread(
            target=listen_for_questions,
            args=(settings, facts, speech, stop_event, state),
            name="voice-listener",
            daemon=True,
        )
        listener.start()
    elif settings.enable_voice:
        state.set_activity("Camera + speech")
    else:
        state.set_activity("Camera only")

    try:
        run_camera(settings, known_embeddings, speech, stop_event, state)
    except RuntimeError as error:
        print(f"Runtime error: {error}")
        return 1
    finally:
        stop_event.set()
        if listener is not None:
            listener.join(timeout=2)
        speech.stop()
        state.set_activity("Stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
