"""Dependency-free latest-frame execution support for face recognition."""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RecognitionBatch:
    """One completed callback invocation and its timing/error metadata."""

    results: Any
    captured_at: float
    completed_at: float
    error: str | None = None


class LatestFrameWorker:
    """Run one callback at a time while retaining only the newest frame."""

    NEW = "new"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"

    _EMPTY = object()

    def __init__(self, callback: Callable[[Any], Any]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")

        self._callback = callback
        self._condition = threading.Condition()
        self._state = self.NEW
        self._thread: threading.Thread | None = None
        self._pending_frame: object = self._EMPTY
        self._pending_captured_at: float | None = None
        self._latest_result: RecognitionBatch | None = None

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    def start(self) -> None:
        """Start the daemon worker; a stopped worker cannot be restarted."""

        with self._condition:
            if self._state == self.RUNNING:
                return
            if self._state != self.NEW:
                raise RuntimeError("worker cannot be started after stopping")

            self._state = self.RUNNING
            thread = threading.Thread(
                target=self._run,
                name="face-recognition-worker",
                daemon=True,
            )
            self._thread = thread
            # Start while holding the condition so a concurrent stop cannot
            # attempt to join a thread that has not been started yet.
            thread.start()

    def submit(self, frame: Any) -> bool:
        """Queue ``frame`` as the newest pending input, if still running."""

        with self._condition:
            if self._state != self.RUNNING:
                return False
            self._pending_frame = frame
            self._pending_captured_at = time.monotonic()
            self._condition.notify()
            return True

    def poll(self) -> RecognitionBatch | None:
        """Return and remove the newest completed result without waiting."""

        with self._condition:
            batch = self._latest_result
            self._latest_result = None
            return batch

    def stop(self, timeout: float = 2) -> None:
        """Stop accepting work and wait up to ``timeout`` seconds."""

        with self._condition:
            if self._state == self.NEW:
                self._discard_pending_locked()
                self._state = self.STOPPED
                return

            if self._state == self.RUNNING:
                self._state = self.STOPPING
            self._discard_pending_locked()
            self._condition.notify_all()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _discard_pending_locked(self) -> None:
        self._pending_frame = self._EMPTY
        self._pending_captured_at = None

    def _run(self) -> None:
        while True:
            with self._condition:
                while (
                    self._state == self.RUNNING
                    and self._pending_frame is self._EMPTY
                ):
                    self._condition.wait()

                if self._state != self.RUNNING:
                    self._discard_pending_locked()
                    self._state = self.STOPPED
                    self._condition.notify_all()
                    return

                frame = self._pending_frame
                captured_at = self._pending_captured_at
                self._discard_pending_locked()

            # ``captured_at`` is set whenever a RUNNING submission is queued.
            assert captured_at is not None
            try:
                results = self._callback(frame)
            except Exception as exc:
                results = None
                error = str(exc) or exc.__class__.__name__
            else:
                error = None

            batch = RecognitionBatch(
                results=results,
                captured_at=captured_at,
                completed_at=time.monotonic(),
                error=error,
            )

            with self._condition:
                self._latest_result = batch
                if self._state != self.RUNNING:
                    self._discard_pending_locked()
                    self._state = self.STOPPED
                    self._condition.notify_all()
                    return
