"""Presence-aware greeting decisions for the camera loop.

The controller is deliberately independent of OpenCV and text-to-speech so
its behaviour (one greeting per visit, visitor welcome cooldowns, and
absence-based resets) can be unit tested deterministically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable


@dataclass(frozen=True)
class GreetingAction:
    """One utterance the camera loop should enqueue."""

    kind: str  # "known" or "visitor"
    name: str
    message: str


class GreetingController:
    def __init__(
        self,
        college_name: str,
        *,
        presence_timeout: float = 8.0,
        visitor_welcome_cooldown: float = 20.0,
        min_greeting_gap: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.college_name = " ".join(str(college_name).split()) or "the college"
        self.presence_timeout = max(1.0, float(presence_timeout))
        self.visitor_welcome_cooldown = max(0.0, float(visitor_welcome_cooldown))
        self.min_greeting_gap = max(0.0, float(min_greeting_gap))
        self._clock = clock
        self._wall_clock = wall_clock

        self._last_seen: dict[str, float] = {}
        self._greeted_this_visit: set[str] = set()
        self._unknown_last_seen: float | None = None
        self._unknown_welcomed = False
        self._last_welcome_at: float | None = None
        self._last_greeting_at: float | None = None

    def update(
        self,
        confirmed_names: Iterable[str],
        has_unknown: bool,
        *,
        now: float | None = None,
        wall_time: datetime | None = None,
    ) -> list[GreetingAction]:
        """Return zero or one greeting for the current camera frame.

        ``confirmed_names`` must contain only identities confirmed by the
        stability tracker.  ``has_unknown`` is true when at least one detected
        face was not accepted as a known person.
        """

        current = self._clock() if now is None else float(now)
        wall = self._wall_clock() if wall_time is None else wall_time
        self._expire(current)

        names: set[str] = set()
        for name in confirmed_names:
            if not isinstance(name, str):
                continue
            cleaned = name.strip()
            if cleaned and cleaned.casefold() != "unknown":
                names.add(cleaned)
        for name in names:
            self._last_seen[name] = current

        if has_unknown:
            self._unknown_last_seen = current

        if self._can_greet(current):
            candidates = sorted(names - self._greeted_this_visit)
            if candidates:
                name = candidates[0]
                self._greeted_this_visit.add(name)
                self._last_greeting_at = current
                return [
                    GreetingAction(
                        kind="known",
                        name=name,
                        message=(
                            f"Hi {name}, the time is "
                            f"{wall.strftime('%I:%M %p').lstrip('0')}"
                        ),
                    )
                ]

        if (
            has_unknown
            and not self._unknown_welcomed
            and self._can_greet(current)
            and (
                self._last_welcome_at is None
                or current - self._last_welcome_at
                >= self.visitor_welcome_cooldown
            )
        ):
            self._unknown_welcomed = True
            self._last_welcome_at = current
            self._last_greeting_at = current
            return [
                GreetingAction(
                    kind="visitor",
                    name="",
                    message=f"Welcome to {self.college_name}.",
                )
            ]
        return []

    def _can_greet(self, now: float) -> bool:
        return (
            self._last_greeting_at is None
            or now - self._last_greeting_at >= self.min_greeting_gap
        )

    def _expire(self, now: float) -> None:
        for name, last_seen in tuple(self._last_seen.items()):
            if now - last_seen > self.presence_timeout:
                self._last_seen.pop(name, None)
                self._greeted_this_visit.discard(name)
        if (
            self._unknown_last_seen is not None
            and now - self._unknown_last_seen > self.presence_timeout
        ):
            self._unknown_last_seen = None
            self._unknown_welcomed = False

    def reset(self) -> None:
        self._last_seen.clear()
        self._greeted_this_visit.clear()
        self._unknown_last_seen = None
        self._unknown_welcomed = False
        self._last_welcome_at = None
        self._last_greeting_at = None
