"""Thread-safe, presentation-neutral assistant status."""

from __future__ import annotations

import threading


class AssistantState:
    """Small status board shared by the camera and voice threads."""

    def __init__(self, activity: str = "Starting", answer: str = "") -> None:
        self._lock = threading.Lock()
        self._activity = activity
        self._answer = answer

    def set_activity(self, activity: str) -> None:
        cleaned = " ".join(str(activity).split())[:80]
        with self._lock:
            self._activity = cleaned

    def set_answer(self, answer: str) -> None:
        cleaned = " ".join(str(answer).split())[:600]
        with self._lock:
            self._answer = cleaned

    def snapshot(self) -> tuple[str, str]:
        with self._lock:
            return self._activity, self._answer

    def reset(self) -> None:
        with self._lock:
            self._activity = "Starting"
            self._answer = ""
