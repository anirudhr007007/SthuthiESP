import threading
import time
import unittest

from speech_service import (
    SpeechService,
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


if __name__ == "__main__":
    unittest.main()
