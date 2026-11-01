"""Cross-platform, bounded text-to-speech with echo-guard metadata."""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque
from difflib import SequenceMatcher
from typing import Callable, Iterable, Protocol


class SpeechBackend(Protocol):
    def speak(self, message: str) -> None: ...

    def close(self) -> None: ...


def _normalise_for_matching(value: str) -> str:
    text = " ".join(str(value).casefold().split())
    return re.sub(r"[^a-z0-9\u00c0-\uffff ]+", " ", text).strip()


def text_similarity(left: str, right: str) -> float:
    """Return a conservative similarity score for echo detection.

    The score combines character-sequence similarity, token Jaccard overlap,
    and containment.  It is only used against recently spoken assistant text,
    never as a general-purpose question matcher.
    """

    first = _normalise_for_matching(left)
    second = _normalise_for_matching(right)
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0

    sequence = SequenceMatcher(None, first, second, autojunk=False).ratio()
    first_tokens = set(first.split())
    second_tokens = set(second.split())
    union = first_tokens | second_tokens
    jaccard = len(first_tokens & second_tokens) / len(union) if union else 0.0
    containment = 0.0
    if first in second or second in first:
        containment = min(len(first), len(second)) / max(len(first), len(second))
    return max(sequence, jaccard, containment)


def looks_like_echo(
    transcript: str,
    recent_spoken: Iterable[str],
    threshold: float = 0.72,
) -> bool:
    """Return true when ``transcript`` closely matches recent assistant speech."""

    if not isinstance(transcript, str) or not transcript.strip():
        return False
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        value = 0.72
    value = min(0.99, max(0.3, value))
    return any(
        text_similarity(transcript, spoken) >= value
        for spoken in recent_spoken
        if isinstance(spoken, str) and spoken.strip()
    )


def _sapi_rate(words_per_minute: int) -> int:
    """Map a words-per-minute rate onto SAPI's -10..10 scale."""

    return max(-10, min(10, round((int(words_per_minute) - 165) / 20)))


class _WindowsSapiBackend:
    def __init__(
        self,
        output_name: str,
        voice_name: str,
        rate: int,
        volume: float,
    ) -> None:
        import pythoncom
        from win32com.client import Dispatch

        self._pythoncom = pythoncom
        self._com_initialized = False
        pythoncom.CoInitialize()
        self._com_initialized = True
        try:
            engine = Dispatch("SAPI.SpVoice")
            engine.Rate = _sapi_rate(rate)
            engine.Volume = int(round(max(0.0, min(1.0, volume)) * 100))
            if output_name:
                outputs = engine.GetAudioOutputs()
                selected = next(
                    (
                        outputs.Item(index)
                        for index in range(outputs.Count)
                        if output_name.casefold()
                        in outputs.Item(index).GetDescription().casefold()
                    ),
                    None,
                )
                if selected is not None:
                    engine.AudioOutput = selected
                else:
                    print(
                        f"Speech output '{output_name}' was not found; "
                        "using the Windows default."
                    )
            if voice_name:
                voices = engine.GetVoices()
                selected_voice = next(
                    (
                        voices.Item(index)
                        for index in range(voices.Count)
                        if voice_name.casefold()
                        in voices.Item(index).GetDescription().casefold()
                    ),
                    None,
                )
                if selected_voice is not None:
                    engine.Voice = selected_voice
                else:
                    print(
                        f"Speech voice '{voice_name}' was not found; "
                        "using the Windows default."
                    )
            audio_output = engine.AudioOutput
            if audio_output is None:
                raise RuntimeError(
                    "Windows SAPI has no active audio output device."
                )
            print(f"Speech output: {audio_output.GetDescription()}")
            self._engine = engine
        except Exception:
            pythoncom.CoUninitialize()
            self._com_initialized = False
            raise

    def speak(self, message: str) -> None:
        self._engine.Speak(message)

    def close(self) -> None:
        self._engine = None
        if self._com_initialized:
            self._pythoncom.CoUninitialize()
            self._com_initialized = False


class _Pyttsx3Backend:
    def __init__(
        self,
        output_name: str,
        voice_name: str,
        rate: int,
        volume: float,
    ) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", int(rate))
        engine.setProperty("volume", max(0.0, min(1.0, volume)))
        if output_name:
            print(
                "SPEECH_OUTPUT is only supported by the Windows SAPI backend; "
                "using the system default output."
            )
        if voice_name:
            voices = engine.getProperty("voices") or []
            selected = next(
                (
                    voice
                    for voice in voices
                    if voice_name.casefold()
                    in f"{voice.name} {voice.id}".casefold()
                ),
                None,
            )
            if selected is not None:
                engine.setProperty("voice", selected.id)
            else:
                print(
                    f"Speech voice '{voice_name}' was not found; "
                    "using the system default."
                )
        self._engine = engine

    def speak(self, message: str) -> None:
        self._engine.say(message)
        self._engine.runAndWait()

    def close(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            pass
        self._engine = None


def _default_backend_factory(
    output_name: str,
    voice_name: str,
    rate: int,
    volume: float,
) -> Callable[[], SpeechBackend]:
    def create() -> SpeechBackend:
        if os.name == "nt":
            return _WindowsSapiBackend(output_name, voice_name, rate, volume)
        return _Pyttsx3Backend(output_name, voice_name, rate, volume)

    return create


class SpeechService:
    """Bounded speech queue with explicit busy/speaking state.

    There is exactly one worker thread, so the microphone listener can mute
    itself while any answer is queued or being spoken.  When the queue is
    full the oldest *queued* response is discarded and the newest request is
    kept; the previous implementation claimed to do this but actually dropped
    the newest request.
    """

    def __init__(
        self,
        enabled: bool = True,
        output_name: str = "",
        *,
        voice_name: str = "",
        rate: int = 165,
        volume: float = 1.0,
        max_queue: int = 25,
        idle_grace: float = 0.6,
        recent_window: float = 20.0,
        backend_factory: Callable[[], SpeechBackend] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_name = output_name
        self.voice_name = voice_name
        self.rate = int(rate)
        self.volume = max(0.0, min(1.0, float(volume)))
        self.idle_grace = max(0.0, float(idle_grace))
        self.recent_window = max(2.0, float(recent_window))
        self._max_queue = max(1, int(max_queue))
        self._backend_factory = backend_factory or _default_backend_factory(
            self.output_name, self.voice_name, self.rate, self.volume
        )

        self._condition = threading.Condition()
        self._messages: deque[str] = deque()
        self._history: deque[tuple[float, str]] = deque(maxlen=50)
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._failed = False
        self._speaking = False
        self._current_message = ""
        self._last_spoken_at = 0.0
        self._activity_token = 0

    @property
    def available(self) -> bool:
        with self._condition:
            return self.enabled and not self._failed and not self._stopping

    @property
    def is_speaking(self) -> bool:
        with self._condition:
            return self._speaking

    @property
    def is_busy(self) -> bool:
        with self._condition:
            return self._speaking or bool(self._messages)

    @property
    def queue_size(self) -> int:
        with self._condition:
            return len(self._messages)

    @property
    def activity_token(self) -> int:
        """Monotonic token that changes whenever speech activity changes."""

        with self._condition:
            return self._activity_token

    def start(self) -> bool:
        if not self.enabled:
            return False
        with self._condition:
            if self._started:
                return not self._failed and not self._stopping
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="speech-worker",
                daemon=True,
            )
            self._thread.start()
            return True

    def speak(self, message: str) -> bool:
        if not self.enabled or not isinstance(message, str) or not message.strip():
            return False
        self.start()
        with self._condition:
            if self._failed or self._stopping:
                return False
            if len(self._messages) >= self._max_queue:
                self._messages.popleft()
                print(
                    "Speech queue is full; dropping the oldest queued response."
                )
            self._messages.append(message)
            self._activity_token += 1
            self._condition.notify_all()
            return True

    def wait_until_idle(
        self, grace: float | None = None, timeout: float | None = None
    ) -> bool:
        """Wait until speech has been idle for ``grace`` seconds.

        Returns ``False`` only when ``timeout`` expires first.  A failed
        backend is treated as idle so the microphone listener cannot hang.
        """

        grace_value = self.idle_grace if grace is None else max(0.0, float(grace))
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                if self._failed:
                    return True
                now = time.monotonic()
                idle = not self._speaking and not self._messages
                if idle:
                    remaining_grace = (
                        self._last_spoken_at + grace_value - now
                    )
                    if remaining_grace <= 0:
                        return True
                    wait_for = remaining_grace
                else:
                    wait_for = 0.1
                if deadline is not None:
                    remaining = deadline - now
                    if remaining <= 0:
                        return False
                    wait_for = min(wait_for, remaining)
                self._condition.wait(max(0.01, wait_for))

    def recent_texts(
        self, window: float | None = None, now: float | None = None
    ) -> list[str]:
        window_value = self.recent_window if window is None else max(0.0, float(window))
        current = time.monotonic() if now is None else float(now)
        cutoff = current - window_value
        with self._condition:
            return [text for spoken_at, text in self._history if spoken_at >= cutoff]

    def looks_like_echo(
        self,
        transcript: str,
        *,
        threshold: float = 0.72,
        window: float | None = None,
    ) -> bool:
        return looks_like_echo(
            transcript,
            self.recent_texts(window=window),
            threshold=threshold,
        )

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._messages.clear()
            self._activity_token += 1
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            self._failed = True
            self._messages.clear()
            self._speaking = False
            self._current_message = ""
            self._activity_token += 1
            self._condition.notify_all()
        print(f"Speech output unavailable: {error}")

    def _run(self) -> None:
        try:
            backend = self._backend_factory()
        except Exception as error:
            self._fail(error)
            return

        try:
            while True:
                with self._condition:
                    while not self._messages and not self._stopping:
                        self._condition.wait()
                    if self._stopping and not self._messages:
                        return
                    message = self._messages.popleft()
                    self._speaking = True
                    self._current_message = message
                    self._activity_token += 1

                try:
                    try:
                        print(f"Speaking: {message}")
                    except (UnicodeEncodeError, OSError):
                        safe_text = message.encode("ascii", "replace").decode("ascii")
                        print(f"Speaking: {safe_text}")
                    backend.speak(message)
                except Exception as error:
                    print(f"Speech output error: {error}")
                finally:
                    with self._condition:
                        self._speaking = False
                        self._current_message = ""
                        self._last_spoken_at = time.monotonic()
                        self._history.append((self._last_spoken_at, message))
                        self._activity_token += 1
                        self._condition.notify_all()
        finally:
            try:
                backend.close()
            except Exception:
                pass
            with self._condition:
                self._speaking = False
                self._current_message = ""
                self._activity_token += 1
                self._condition.notify_all()
