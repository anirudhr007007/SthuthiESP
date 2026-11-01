import math
import threading
import time
import unittest

from face_utils import (
    MatchResult,
    StableIdentityTracker,
    cosine_similarity,
    face_box,
    identify,
    is_plausible_face_area,
    normalize,
    select_enrollment_face,
)
from recognition_worker import LatestFrameWorker


class FaceUtilityTests(unittest.TestCase):
    def test_invalid_embeddings_fail_closed_even_at_zero_threshold(self) -> None:
        known = {"Alice": [[1.0, 0.0]]}

        self.assertEqual(normalize([]), ())
        self.assertEqual(normalize([0.0, 0.0]), (0.0, 0.0))
        self.assertEqual(normalize([math.nan, 1.0]), (0.0, 0.0))
        self.assertEqual(cosine_similarity([], [1.0]), 0.0)
        self.assertEqual(cosine_similarity([math.inf], [1.0]), 0.0)
        self.assertEqual(cosine_similarity([1.0], [1.0, 0.0]), 0.0)

        invalid_queries = ([0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0], [])
        for query in invalid_queries:
            with self.subTest(query=query):
                result = identify(query, known, threshold=0.0, margin=0.0)
                self.assertFalse(result.accepted)
                self.assertEqual(result.name, "Unknown")
                self.assertTrue(math.isfinite(result.score))
                self.assertTrue(math.isfinite(result.runner_up_score))

        for invalid_known in (
            {"Alice": [[0.0, 0.0]]},
            {"Alice": [[math.nan, 1.0]]},
            {"Alice": [[1.0, 0.0, 0.0]]},
            {},
        ):
            with self.subTest(invalid_known=invalid_known):
                result = identify([1.0, 0.0], invalid_known, 0.0, 0.0)
                self.assertFalse(result.accepted)
                self.assertEqual(result.name, "Unknown")

    def test_ambiguous_match_is_rejected(self) -> None:
        result = identify(
            [1.0, 0.0],
            {"Alice": [[1.0, 0.0]], "Bob": [[1.0, 0.0]]},
            threshold=0.0,
            margin=0.01,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.name, "Unknown")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.runner_up_score, 1.0)

    def test_face_box_rejects_invalid_dimensions(self) -> None:
        self.assertEqual(face_box({"x": 1, "y": 2, "w": 3, "h": 4}), (1, 2, 3, 4))
        for width, height in ((0, 4), (-1, 4), (3, 0), (3, -1), (math.nan, 4)):
            with self.subTest(width=width, height=height):
                self.assertIsNone(face_box({"x": 1, "y": 2, "w": width, "h": height}))

    def test_plausible_face_area_rejects_tiny_and_full_frame_boxes(self) -> None:
        self.assertTrue(
            is_plausible_face_area({"x": 10, "y": 10, "w": 120, "h": 120}, 640, 480)
        )
        # DeepFace may return the whole frame when no face was detected.
        self.assertFalse(
            is_plausible_face_area({"x": 0, "y": 0, "w": 640, "h": 480}, 640, 480)
        )
        self.assertFalse(
            is_plausible_face_area({"x": 0, "y": 0, "w": 23, "h": 200}, 640, 480)
        )
        self.assertFalse(
            is_plausible_face_area({"x": 0, "y": 0, "w": 700, "h": 500}, 640, 480)
        )
        self.assertFalse(is_plausible_face_area({}, 640, 480))

    def test_plausible_face_area_tolerates_missing_frame_size(self) -> None:
        self.assertTrue(
            is_plausible_face_area({"x": 1, "y": 1, "w": 50, "h": 50}, 0, 0)
        )
        self.assertTrue(
            is_plausible_face_area({"x": 1, "y": 1, "w": 50, "h": 50}, "bad", "size")
        )

    def test_enrollment_accepts_one_dominant_face_and_ignores_tiny_false_positives(self) -> None:
        primary = {"embedding": [1.0], "facial_area": {"x": 0, "y": 0, "w": 532, "h": 532}}
        tiny = {"embedding": [0.0], "facial_area": {"x": 0, "y": 0, "w": 63, "h": 63}}

        self.assertIs(
            select_enrollment_face([primary], 900, 1600),
            primary,
        )
        self.assertIs(
            select_enrollment_face([primary, tiny], 900, 1600),
            primary,
        )

        detections = [
            {"facial_area": {"x": 0, "y": 0, "w": 62, "h": 62}},
            {"facial_area": {"x": 0, "y": 0, "w": 269, "h": 269}},
            {"facial_area": {"x": 0, "y": 0, "w": 108, "h": 108}},
            {"facial_area": {"x": 0, "y": 0, "w": 90, "h": 90}},
        ]
        self.assertIs(
            select_enrollment_face(detections, 1067, 1217),
            detections[1],
        )

    def test_enrollment_rejects_two_plausible_faces_or_a_too_small_primary(self) -> None:
        first = {"facial_area": {"x": 0, "y": 0, "w": 300, "h": 300}}
        second = {"facial_area": {"x": 400, "y": 0, "w": 300, "h": 300}}
        background = {"facial_area": {"x": 0, "y": 0, "w": 120, "h": 120}}
        small = {"facial_area": {"x": 0, "y": 0, "w": 100, "h": 100}}

        self.assertIsNone(select_enrollment_face([first, second], 1000, 1000))
        self.assertIsNone(select_enrollment_face([first, background], 1000, 1000))
        self.assertIsNone(select_enrollment_face([small], 1000, 1000))
        self.assertIsNone(select_enrollment_face([first], 0, 1000))
        self.assertIsNone(select_enrollment_face(["not a detection"], 1000, 1000))

    def test_enrollment_rejects_two_detections_without_face_areas(self) -> None:
        self.assertIsNone(
            select_enrollment_face(
                [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}],
                1000,
                1000,
            )
        )

    def test_tracker_confirms_once_per_batch_and_resets(self) -> None:
        accepted = MatchResult("Alice", 0.9, 0.1, True)
        ambiguous = MatchResult("Unknown", 0.9, 0.89, False)
        tracker = StableIdentityTracker(confirmations=3)

        self.assertEqual(tracker.update([accepted, accepted]), set())
        self.assertEqual(tracker.update([accepted, accepted]), set())
        self.assertEqual(tracker.update([accepted]), {"Alice"})
        self.assertEqual(tracker.update([ambiguous]), set())
        self.assertEqual(tracker.update([accepted]), set())


class LatestFrameWorkerTests(unittest.TestCase):
    @staticmethod
    def _wait_for(event: threading.Event, timeout: float = 2.0) -> None:
        if not event.wait(timeout):
            raise AssertionError("timed out waiting for worker")

    def test_latest_pending_frame_and_serial_callback(self) -> None:
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        release_second = threading.Event()
        seen: list[str] = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def callback(frame: str) -> str:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                seen.append(frame)
            if frame == "first":
                first_started.set()
                release_first.wait(2)
            else:
                second_started.set()
                release_second.wait(2)
            with lock:
                active -= 1
            return frame

        worker = LatestFrameWorker(callback)
        worker.start()
        self.assertTrue(worker.submit("first"))
        self._wait_for(first_started)
        self.assertTrue(worker.submit("second"))
        self.assertTrue(worker.submit("latest"))
        release_first.set()
        self._wait_for(second_started)
        self.assertEqual(seen, ["first", "latest"])
        self.assertEqual(max_active, 1)

        first_batch = worker.poll()
        self.assertIsNotNone(first_batch)
        self.assertEqual(first_batch.results, "first")
        release_second.set()
        deadline = time.monotonic() + 2
        latest_batch = None
        while latest_batch is None and time.monotonic() < deadline:
            latest_batch = worker.poll()
            if latest_batch is None:
                time.sleep(0.005)
        self.assertIsNotNone(latest_batch)
        self.assertEqual(latest_batch.results, "latest")
        self.assertLessEqual(latest_batch.captured_at, latest_batch.completed_at)
        worker.stop()
        self.assertEqual(worker.state, worker.STOPPED)

    def test_errors_are_returned_once_and_stop_disables_submission(self) -> None:
        completed = threading.Event()

        def callback(_frame: object) -> object:
            completed.set()
            raise ValueError("no faces")

        worker = LatestFrameWorker(callback)
        worker.start()
        self.assertTrue(worker.submit(object()))
        self._wait_for(completed)

        deadline = time.monotonic() + 2
        batch = None
        while batch is None and time.monotonic() < deadline:
            batch = worker.poll()
            if batch is None:
                time.sleep(0.005)
        self.assertIsNotNone(batch)
        self.assertIsNone(batch.results)
        self.assertEqual(batch.error, "no faces")
        self.assertIsNone(worker.poll())

        worker.stop()
        self.assertFalse(worker.submit(object()))
        with self.assertRaises(RuntimeError):
            worker.start()

    def test_stop_discards_pending_frame(self) -> None:
        started = threading.Event()
        release = threading.Event()
        seen: list[int] = []

        def callback(frame: int) -> int:
            seen.append(frame)
            started.set()
            release.wait(2)
            return frame

        worker = LatestFrameWorker(callback)
        worker.start()
        self.assertTrue(worker.submit(1))
        self._wait_for(started)
        self.assertTrue(worker.submit(2))
        worker.stop(timeout=0.01)
        self.assertEqual(worker.state, worker.STOPPING)
        release.set()

        deadline = time.monotonic() + 2
        while worker.state != worker.STOPPED and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(worker.state, worker.STOPPED)
        self.assertEqual(seen, [1])


if __name__ == "__main__":
    unittest.main()
