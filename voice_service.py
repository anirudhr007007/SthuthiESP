"""Microphone input, wake-word handling, and echo protection.

Voice input is opt-in.  The default ``STT_PROVIDER`` is ``none``; choosing
Google sends audio to Google, while choosing Whisper keeps it local.  The
listener never transcribes audio that may contain the assistant's own
text-to-speech output, and it skips transcripts that closely match recently
spoken answers.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ai_provider import answer_question
from settings import Settings

# A persistent failure (for example a corrupted local model raising on every
# transcription) must not log forever; the camera keeps running meanwhile.
_VOICE_MAX_CONSECUTIVE_ERRORS = 5

_GREETINGS = {
    "hi",
    "hii",
    "hiii",
    "hello",
    "hey",
    "hey there",
    "hi there",
    "hello there",
    "hi everyone",
    "hello everyone",
    "good day",
    "good morning",
    "good afternoon",
    "good evening",
    "namaskara",
    "namaste",
}


def is_greeting(question: str) -> bool:
    """Return true only for a short greeting, not a greeting-prefixed question."""

    cleaned = str(question).lower().strip()
    cleaned = re.sub(r"[^a-z\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)
    return cleaned in _GREETINGS


@dataclass(frozen=True)
class WakeWordResult:
    matched: bool
    question: str


def apply_wake_word(transcript: str, wake_word: str) -> WakeWordResult:
    """Strip a configured leading wake word without accepting similar words.

    Matching is word based, so ``hey assistant`` does not match
    ``hey assistants``.  When no wake word is configured every transcript is
    accepted unchanged.
    """

    text = " ".join(str(transcript).split()).strip()
    configured = " ".join(str(wake_word).split()).strip()
    if not configured:
        return WakeWordResult(True, text)

    words = re.findall(r"[A-Za-z0-9']+", text)
    wake_words = re.findall(r"[A-Za-z0-9']+", configured)
    if not wake_words or len(words) < len(wake_words):
        return WakeWordResult(False, text)
    if [
        word.casefold() for word in words[: len(wake_words)]
    ] != [word.casefold() for word in wake_words]:
        return WakeWordResult(False, text)
    remainder = " ".join(words[len(wake_words) :]).strip()
    return WakeWordResult(True, remainder)


def whisper_language(language: str) -> str | None:
    """Map a BCP-47 style language (``en-IN``) to Whisper's base code."""

    cleaned = str(language).strip().replace("_", "-")
    if not cleaned:
        return None
    return cleaned.split("-", 1)[0].lower() or None


def answer_spoken_question(
    question: str,
    settings: Settings,
    facts: Iterable[dict[str, Any]],
    answer_func: Callable[[str, Settings, Iterable[dict[str, Any]]], str] = answer_question,
) -> str:
    """Answer one spoken question while keeping greetings local and cheap."""

    cleaned = " ".join(str(question).split()).strip()
    if not cleaned:
        return ""
    if is_greeting(cleaned):
        return "Hello! How can I help you?"
    return answer_func(cleaned, settings, facts)


def _whisper_transcriber(settings: Settings) -> Callable[[Any], str]:
    """Create an optional local faster-whisper transcriber."""

    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "STT_PROVIDER=whisper requires the optional faster-whisper package. "
            "Install requirements-offline-voice.txt."
        ) from error

    print(f"Loading local Whisper model: {settings.whisper_model}")
    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    language = whisper_language(settings.stt_language)

    def transcribe(audio: Any) -> str:
        raw_audio = audio.get_raw_data(convert_rate=16000, convert_width=2)
        samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            samples,
            beam_size=1,
            vad_filter=True,
            language=language,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe


def _google_transcriber(recognizer: Any, settings: Settings) -> Callable[[Any], str]:
    """Create the opt-in Google recognizer callable.

    Google receives the captured audio; callers must announce this to visitors.
    ``--offline`` forces ``STT_PROVIDER`` back to ``none`` before this runs.
    """

    def transcribe(audio: Any) -> str:
        return recognizer.recognize_google(
            audio, language=settings.stt_language
        ).strip()

    return transcribe


def _set_activity(state: Any, text: str) -> None:
    if state is not None and hasattr(state, "set_activity"):
        state.set_activity(text)


def _set_answer(state: Any, text: str) -> None:
    if state is not None and hasattr(state, "set_answer"):
        state.set_answer(text)


def _debug_print(message: str, settings: Any) -> None:
    """Print transcript/answer content only when VOICE_DEBUG is enabled.

    Visitor speech can be private, so the default listener output stays
    content-free.  Set ``VOICE_DEBUG=true`` in ``.env`` while tuning the
    microphone, wake word, or echo guard.
    """

    if getattr(settings, "voice_debug", False):
        print(message)


def _open_microphone(sr: Any, settings: Settings) -> Any:
    try:
        if settings.stt_device_index is None:
            return sr.Microphone()
        return sr.Microphone(device_index=settings.stt_device_index)
    except Exception as error:
        try:
            names = sr.Microphone.list_microphone_names()
        except Exception:
            names = []
        print(f"Microphone unavailable: {error}")
        if names:
            print("Available microphone indexes:")
            for index, name in enumerate(names):
                print(f"  {index}: {name}")
        raise


def listen_for_questions(
    settings: Settings,
    facts: list[dict[str, Any]],
    speech: Any,
    stop_event: threading.Event,
    state: Any = None,
) -> None:
    """Listen, transcribe, answer, and speak until ``stop_event`` is set."""

    if not settings.enable_voice:
        print("Voice input disabled.")
        return
    if settings.stt_provider == "none":
        print("Voice input disabled (STT_PROVIDER=none); speech output remains available.")
        return

    try:
        import speech_recognition as sr
    except ImportError as error:
        print(
            "Voice input unavailable: install SpeechRecognition and PyAudio "
            f"({error})."
        )
        return

    try:
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = settings.stt_pause_threshold
        recognizer.dynamic_energy_threshold = (
            settings.stt_energy_threshold is None
        )
        if settings.stt_energy_threshold is not None:
            recognizer.energy_threshold = settings.stt_energy_threshold
        microphone = _open_microphone(sr, settings)
    except Exception as error:
        print(f"Microphone setup failed: {error}")
        return

    # Do not calibrate against the startup announcement or an answer.
    speech.wait_until_idle(settings.echo_idle_grace, timeout=20)
    try:
        with microphone as source:
            print("Calibrating microphone noise level...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except Exception as error:
        print(f"Microphone unavailable: {error}")
        return

    if settings.stt_provider == "whisper":
        try:
            transcribe = _whisper_transcriber(settings)
        except RuntimeError as error:
            print(error)
            return
        print("Voice input uses local Whisper; audio stays on this computer.")
    elif settings.stt_provider == "google":
        transcribe = _google_transcriber(recognizer, settings)
        print(
            "Voice input uses Google's online recognizer; captured audio leaves "
            "this computer. Use STT_PROVIDER=whisper or none for privacy."
        )
    else:
        print(f"Unknown STT_PROVIDER={settings.stt_provider!r}; voice input disabled.")
        return

    if settings.wake_word:
        print(f"Wake word required: {settings.wake_word!r}")
    else:
        print("No wake word configured; the microphone is always listening.")
    print(f"Voice assistant ready using {settings.stt_provider} speech recognition.")
    _set_activity(state, "Listening")

    last_network_error = 0.0
    consecutive_errors = 0
    while not stop_event.is_set():
        try:
            if speech.is_busy:
                # Mute the recognizer while TTS is active so it cannot hear
                # (or upload) the assistant's own answer.
                speech.wait_until_idle(
                    settings.echo_idle_grace,
                    timeout=settings.listen_timeout + settings.phrase_time_limit,
                )
                continue

            _set_activity(state, "Listening")
            activity_token = speech.activity_token
            with microphone as source:
                audio = recognizer.listen(
                    source,
                    timeout=settings.listen_timeout,
                    phrase_time_limit=settings.phrase_time_limit,
                )
            if stop_event.is_set():
                break
            if speech.activity_token != activity_token or speech.is_busy:
                # TTS started while this capture was in progress.  Discard the
                # audio before transcription, which also avoids uploading it.
                continue

            question = transcribe(audio)
            if not question:
                continue
            question = " ".join(question.split()).strip()
            consecutive_errors = 0
            if settings.echo_guard and speech.looks_like_echo(
                question,
                threshold=settings.echo_similarity_threshold,
                window=settings.echo_recent_window,
            ):
                print("Ignoring speech that matches the assistant's own recent answer.")
                continue

            wake_result = apply_wake_word(question, settings.wake_word)
            if not wake_result.matched:
                if settings.wake_word:
                    # Never echo the rejected transcript.  It may contain a
                    # visitor's private speech and is not needed for debugging.
                    print("Ignored speech without the configured wake word.")
                continue
            question = wake_result.question
            if not question:
                if settings.wake_word:
                    print("Wake word heard.")
                    speech.speak("Yes, I am listening.")
                continue

            _debug_print(f"Heard question: {question!r}", settings)
            _set_activity(state, "Thinking")
            reply = answer_spoken_question(question, settings, facts)
            if not reply:
                continue
            _debug_print(f"Assistant answering: {reply}", settings)
            _set_answer(state, reply)
            _set_activity(state, "Answering")
            speech.speak(reply)
            _set_activity(state, "Listening")
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            continue
        except sr.RequestError as error:
            now = time.monotonic()
            print(f"Speech recognition network error: {error}")
            if now - last_network_error > 30:
                speech.speak(
                    "Speech recognition is unavailable. You can still use the camera."
                )
                last_network_error = now
            _set_activity(state, "Listening")
        except OSError as error:
            consecutive_errors += 1
            print(f"Microphone error: {error}")
            _set_activity(state, "Microphone error")
            if consecutive_errors >= _VOICE_MAX_CONSECUTIVE_ERRORS:
                print("Voice listener stopped after repeated errors; camera continues.")
                _set_activity(state, "Voice unavailable")
                break
            if stop_event.wait(2):
                break
        except Exception as error:
            consecutive_errors += 1
            print(f"Voice assistant error ({consecutive_errors}): {error}")
            if consecutive_errors >= _VOICE_MAX_CONSECUTIVE_ERRORS:
                print("Voice listener stopped after repeated errors; camera continues.")
                _set_activity(state, "Voice unavailable")
                break
            if stop_event.wait(0.5):
                break
