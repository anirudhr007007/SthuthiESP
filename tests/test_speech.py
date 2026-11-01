import contextlib
import io
import queue
import sys
import threading
import time
import types
import unittest
from unittest import mock

from speech_service import (
    SpeechService,
    _KokoroBackend,
    _offer_piece,
    _PiperBackend,
    _sapi_rate,
    looks_like_echo,
    text_similarity,
)


class RecordingBackend:
    def __init__(self, *, block: bool = False) -> None:
        self.messages: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.closed = False

    def speak(self, message: str) -> None:
        self.started.set()
        if self.block:
            self.release.wait(3)
        self.messages.append(message)

    def close(self) -> None:
        self.closed = True


class SilentBackend:
    """Backend whose speak() does nothing; the service must keep working."""

    def speak(self, message: str) -> None:
        pass

    def close(self) -> None:
        pass


class EchoDetectionTests(unittest.TestCase):
    def test_identical_text_is_a_perfect_match(self):
        self.assertEqual(text_similarity("MCE has a library.", "MCE has a library!"), 1.0)

    def test_echo_detection_accepts_a_recent_answer_and_rejects_a_new_question(self):
        recent = ["MCE has a library and learning-resource facilities."]
        self.assertTrue(
            looks_like_echo(
                "MCE has a library and learning resource facilities",
                recent,
                threshold=0.72,
            )
        )
        self.assertFalse(
            looks_like_echo("What is photosynthesis?", recent, threshold=0.72)
        )
        self.assertFalse(looks_like_echo("", recent))

    def test_threshold_is_clamped_to_a_safe_range(self):
        recent = ["The library is open"]
        self.assertFalse(looks_like_echo("library", recent, threshold=5.0))
        self.assertTrue(looks_like_echo("library", recent, threshold=0.0))

    def test_sapi_rate_mapping_is_bounded(self):
        self.assertEqual(_sapi_rate(165), 0)
        self.assertEqual(_sapi_rate(80), -4)
        self.assertEqual(_sapi_rate(400), 10)
        # SAPI accepts -10..10; extreme requests must clamp, not overflow.
        self.assertEqual(_sapi_rate(10_000), 10)
        self.assertEqual(_sapi_rate(-10_000), -10)


class SpeechServiceTests(unittest.TestCase):
    def test_speaks_queued_messages_and_records_history(self):
        backend = RecordingBackend()
        service = SpeechService(
            enabled=True,
            backend_factory=lambda: backend,
            idle_grace=0.0,
            recent_window=60.0,
        )
        try:
            self.assertTrue(service.speak("Hello visitor."))
            self.assertTrue(service.wait_until_idle(timeout=2.0))
            self.assertEqual(backend.messages, ["Hello visitor."])
            self.assertEqual(service.queue_size, 0)
            self.assertFalse(service.is_speaking)
            self.assertFalse(service.is_busy)
            self.assertIn("Hello visitor.", service.recent_texts())
            self.assertTrue(
                service.looks_like_echo("hello visitor", threshold=0.8)
            )
        finally:
            service.stop()
        self.assertTrue(backend.closed)

    def test_full_queue_drops_the_oldest_queued_message(self):
        backend = RecordingBackend(block=True)
        service = SpeechService(
            enabled=True,
            backend_factory=lambda: backend,
            max_queue=2,
            idle_grace=0.0,
        )
        try:
            self.assertTrue(service.speak("first"))
            self.assertTrue(backend.started.wait(2.0))
            self.assertTrue(service.speak("second"))
            self.assertTrue(service.speak("third"))
            self.assertTrue(service.speak("fourth"))
            self.assertEqual(service.queue_size, 2)
            backend.release.set()
            self.assertTrue(service.wait_until_idle(timeout=3.0))
            self.assertEqual(backend.messages, ["first", "third", "fourth"])
        finally:
            backend.release.set()
            service.stop()

    def test_activity_token_changes_when_speech_is_queued_and_finishes(self):
        backend = RecordingBackend()
        service = SpeechService(
            enabled=True,
            backend_factory=lambda: backend,
            idle_grace=0.0,
        )
        try:
            before = service.activity_token
            service.speak("One")
            self.assertGreater(service.activity_token, before)
            self.assertTrue(service.wait_until_idle(timeout=2.0))
            self.assertGreater(service.activity_token, before)
        finally:
            service.stop()

    def test_backend_failure_marks_service_unavailable_without_hanging(self):
        def failing_factory():
            raise RuntimeError("no audio device")

        service = SpeechService(
            enabled=True,
            backend_factory=failing_factory,
            idle_grace=0.0,
        )
        try:
            service.speak("This cannot be spoken.")
            self.assertTrue(service.wait_until_idle(timeout=2.0))
            self.assertFalse(service.available)
            self.assertFalse(service.speak("Another message"))
        finally:
            service.stop()

    def test_service_keeps_working_when_backend_speak_does_nothing(self):
        backend = SilentBackend()
        service = SpeechService(
            enabled=True,
            backend_factory=lambda: backend,
            idle_grace=0.0,
        )
        try:
            self.assertTrue(service.speak("Nothing plays here."))
            self.assertTrue(service.wait_until_idle(timeout=2.0))
            self.assertTrue(service.available)
            self.assertTrue(service.speak("Still accepting messages."))
            self.assertTrue(service.wait_until_idle(timeout=2.0))
            self.assertEqual(service.queue_size, 0)
        finally:
            service.stop()

    def test_disabled_service_does_not_queue_or_start(self):
        service = SpeechService(enabled=False)
        self.assertFalse(service.speak("Hello"))
        self.assertFalse(service.start())
        self.assertFalse(service.available)
        self.assertTrue(service.wait_until_idle(timeout=0.01))
        service.stop()

    def test_wait_until_idle_times_out_while_backend_is_blocked(self):
        backend = RecordingBackend(block=True)
        service = SpeechService(
            enabled=True,
            backend_factory=lambda: backend,
            idle_grace=0.0,
        )
        try:
            service.speak("Blocked")
            self.assertTrue(backend.started.wait(2.0))
            started = time.monotonic()
            self.assertFalse(service.wait_until_idle(timeout=0.05))
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            backend.release.set()
            service.stop()


class _FailingKokoro:
    def create_stream(self, *args, **kwargs):
        raise RuntimeError("stream exploded")


class _FailingPiperVoice:
    def synthesize(self, chunk, config):
        raise RuntimeError("synthesis exploded")


def _fake_audio_modules(*, playback_fails: bool):
    """Build fake numpy/soundfile/winsound modules for producer tests.

    The backends import these inside speak(), so swapping them via
    sys.modules keeps the tests offline (no real audio or numpy needed).
    Returns (modules, list of played sound buffers).
    """

    played: list[bytes] = []

    winsound = types.ModuleType("winsound")
    winsound.SND_MEMORY = 2

    def play_sound(sound, flags):
        played.append(sound)
        if playback_fails:
            raise RuntimeError("audio device is wedged")

    winsound.PlaySound = play_sound

    numpy = types.ModuleType("numpy")
    numpy.concatenate = lambda parts: parts[0]

    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda buffer, data, sample_rate, **kwargs: buffer.write(
        b"fake-wav-bytes"
    )
    return {"numpy": numpy, "soundfile": soundfile, "winsound": winsound}, played


class SentenceChunkTests(unittest.TestCase):
    def test_abbreviations_do_not_split(self):
        self.assertEqual(
            _PiperBackend._sentence_chunks(
                "Dr. Smith met Mr. Lee at 9 a.m. and No. 42 left."
            ),
            ["Dr. Smith met Mr. Lee at 9 a.m. and No. 42 left."],
        )

    def test_single_letter_periods_do_not_split(self):
        self.assertEqual(
            _PiperBackend._sentence_chunks("A. B. C. continued anyway."),
            ["A. B. C. continued anyway."],
        )

    def test_normal_sentences_split(self):
        self.assertEqual(
            _PiperBackend._sentence_chunks("First. Second."),
            ["First.", "Second."],
        )
        self.assertEqual(
            _PiperBackend._sentence_chunks("What now? Go fast!"),
            ["What now?", "Go fast!"],
        )

    def test_empty_and_whitespace_input_return_no_chunks(self):
        self.assertEqual(_PiperBackend._sentence_chunks(""), [])
        self.assertEqual(_PiperBackend._sentence_chunks("   \n\t "), [])


class OfferPieceTests(unittest.TestCase):
    def test_aborted_event_stops_producing_and_logs_once(self):
        pieces: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        aborted = threading.Event()
        aborted.set()
        stop_logged = [False]
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            first = _offer_piece(pieces, aborted, stop_logged, b"chunk", timeout=0.05)
            second = _offer_piece(pieces, aborted, stop_logged, b"chunk", timeout=0.05)
        self.assertFalse(first)
        self.assertFalse(second)
        self.assertTrue(stop_logged[0])
        self.assertEqual(captured.getvalue().count("stops producing"), 1)

    def test_full_queue_timeout_stops_producing(self):
        pieces: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
        pieces.put(b"stuck")
        stop_logged = [False]
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            offered = _offer_piece(
                pieces, threading.Event(), stop_logged, b"chunk", timeout=0.05
            )
        self.assertFalse(offered)
        self.assertTrue(stop_logged[0])
        self.assertIn("stops producing", captured.getvalue())

    def test_successful_offer_keeps_producing(self):
        pieces: queue.Queue[bytes | None] = queue.Queue()
        stop_logged = [False]
        self.assertTrue(
            _offer_piece(pieces, threading.Event(), stop_logged, b"chunk", timeout=0.05)
        )
        self.assertFalse(stop_logged[0])
        self.assertEqual(pieces.get_nowait(), b"chunk")


class KokoroProducerTests(unittest.TestCase):
    def _backend(self, kokoro):
        backend = _KokoroBackend.__new__(_KokoroBackend)
        backend._kokoro = kokoro
        backend._voice = "af_heart"
        backend._speed = 1.0
        backend._aborted = threading.Event()
        return backend

    def test_generation_failure_is_logged_once_and_nothing_plays(self):
        backend = self._backend(_FailingKokoro())
        with mock.patch("winsound.PlaySound") as play:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                backend.speak("Hello. World.")
        play.assert_not_called()
        output = captured.getvalue()
        self.assertEqual(output.count("Kokoro speech generation failed"), 1)
        self.assertIn("RuntimeError: stream exploded", output)

    def test_playback_error_breaks_loop_and_sets_aborted(self):
        class _PlaceholderVoice:
            sample_rate = 24_000

            @staticmethod
            def create_stream(text, voice, speed):
                async def stream():
                    yield [0.0, 0.01, -0.01], _PlaceholderVoice.sample_rate

                return stream()

        backend = self._backend(_PlaceholderVoice())
        fakes, played = _fake_audio_modules(playback_fails=True)
        with mock.patch.dict(sys.modules, fakes):
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                backend.speak("Hello.")
        self.assertTrue(backend._aborted.is_set())
        self.assertEqual(len(played), 1)
        self.assertIn("Audio playback error:", captured.getvalue())


class PiperProducerTests(unittest.TestCase):
    def _backend(self, voice):
        backend = _PiperBackend.__new__(_PiperBackend)
        backend._voice = voice
        backend._config = None
        backend._aborted = threading.Event()
        return backend

    def test_generation_failure_is_logged_once_and_nothing_plays(self):
        backend = self._backend(_FailingPiperVoice())
        with mock.patch("winsound.PlaySound") as play:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                backend.speak("Hello. World.")
        play.assert_not_called()
        output = captured.getvalue()
        self.assertEqual(output.count("Piper speech generation failed"), 1)
        self.assertIn("RuntimeError: synthesis exploded", output)

    def test_playback_error_breaks_loop_and_sets_aborted(self):
        class _SingleChunkVoice:
            def synthesize(self, chunk, config):
                # One valid chunk per sentence; renderers skip empty output.
                yield types.SimpleNamespace(
                    audio_float_array=[0.0, 0.5, -0.5],
                    sample_rate=22_050,
                )

        backend = self._backend(_SingleChunkVoice())
        fakes, played = _fake_audio_modules(playback_fails=True)
        with mock.patch.dict(sys.modules, fakes):
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                backend.speak("Hello.")
        self.assertTrue(backend._aborted.is_set())
        self.assertEqual(len(played), 1)
        self.assertIn("Audio playback error:", captured.getvalue())


class DefaultBackendFactoryTests(unittest.TestCase):
    def test_broken_kokoro_and_piper_fall_back_to_sapi_with_winsound_hint(self):
        import speech_service

        class _BrokenBackend:
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError("No module named 'winsound'")

        class _FakeSapi:
            def __init__(self, output_name, voice_name, rate, volume):
                self.output_name = output_name

        fake_os = types.SimpleNamespace(name="nt")
        factory = speech_service._default_backend_factory("", "", 165, 1.0, "kokoro")
        with mock.patch.object(speech_service, "_KokoroBackend", _BrokenBackend):
            with mock.patch.object(speech_service, "_PiperBackend", _BrokenBackend):
                with mock.patch.object(speech_service, "_WindowsSapiBackend", _FakeSapi):
                    with mock.patch.object(speech_service, "os", fake_os):
                        with mock.patch.object(
                            speech_service, "_winsound_hinted", set()
                        ):
                            with contextlib.redirect_stdout(io.StringIO()) as captured:
                                backend = factory()
        self.assertIsInstance(backend, _FakeSapi)
        output = captured.getvalue()
        self.assertIn("Kokoro unavailable (ModuleNotFoundError);", output)
        self.assertIn("Piper unavailable (ModuleNotFoundError);", output)
        self.assertEqual(
            output.count("uses winsound, which is Windows-only in this build"), 2
        )
        self.assertEqual(output.count("note: Kokoro audio playback"), 1)
        self.assertEqual(output.count("note: Piper audio playback"), 1)


if __name__ == "__main__":
    unittest.main()
