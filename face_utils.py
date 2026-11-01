"""Small, dependency-free helpers for comparing face embeddings."""

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MatchResult:
    name: str
    score: float
    runner_up_score: float
    accepted: bool


def _finite_vector(values: Sequence[float] | Iterable[float]) -> tuple[float, ...] | None:
    """Return a finite, non-empty vector, or ``None`` for invalid input."""

    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return None
    if not vector or any(not math.isfinite(value) for value in vector):
        return None
    return vector


def _unknown_result() -> MatchResult:
    return MatchResult("Unknown", -1.0, -1.0, False)


def normalize(values: Iterable[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return ()
    if not vector or any(not math.isfinite(value) for value in vector):
        return tuple(0.0 for _ in vector)

    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        return tuple(0.0 for _ in vector)
    normalized = tuple(value / norm for value in vector)
    if any(not math.isfinite(value) for value in normalized):
        return tuple(0.0 for _ in vector)
    return normalized


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    first_vector = _finite_vector(first)
    second_vector = _finite_vector(second)
    if (
        first_vector is None
        or second_vector is None
        or len(first_vector) != len(second_vector)
    ):
        return 0.0

    first_norm = math.hypot(*first_vector)
    second_norm = math.hypot(*second_vector)
    if (
        not math.isfinite(first_norm)
        or not math.isfinite(second_norm)
        or first_norm == 0
        or second_norm == 0
    ):
        return 0.0

    score = math.fsum(
        (left / first_norm) * (right / second_norm)
        for left, right in zip(first_vector, second_vector, strict=False)
    )
    return score if math.isfinite(score) else 0.0


def identify(
    embedding: Sequence[float],
    known_embeddings: Mapping[str, Sequence[Sequence[float]]],
    threshold: float,
    margin: float,
) -> MatchResult:
    query = _finite_vector(embedding)
    if query is None or not any(value != 0 for value in query):
        return _unknown_result()

    try:
        threshold_value = float(threshold)
        margin_value = float(margin)
    except (TypeError, ValueError, OverflowError):
        return _unknown_result()
    if not math.isfinite(threshold_value) or not math.isfinite(margin_value):
        return _unknown_result()

    scores: list[tuple[float, str]] = []
    for person, embeddings in known_embeddings.items():
        try:
            candidate_embeddings = tuple(embeddings)
        except TypeError:
            continue

        person_scores = []
        for known in candidate_embeddings:
            known_vector = _finite_vector(known)
            if (
                known_vector is None
                or len(known_vector) != len(query)
                or not any(value != 0 for value in known_vector)
            ):
                continue
            score = cosine_similarity(query, known_vector)
            if math.isfinite(score):
                person_scores.append(score)
        if person_scores:
            scores.append((max(person_scores), person))

    if not scores:
        return _unknown_result()

    scores.sort(reverse=True)
    best_score, best_name = scores[0]
    runner_up_score = scores[1][0] if len(scores) > 1 else -1.0
    score_gap = best_score - runner_up_score
    accepted = (
        math.isfinite(best_score)
        and math.isfinite(runner_up_score)
        and math.isfinite(score_gap)
        and best_score >= threshold_value
        and score_gap >= margin_value
    )
    return MatchResult(
        best_name if accepted else "Unknown",
        best_score,
        runner_up_score,
        accepted,
    )


def face_box(area: Mapping[str, object]) -> tuple[int, int, int, int] | None:
    """Convert DeepFace's face area to safe OpenCV integer coordinates."""

    try:
        values = tuple(float(area[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        any(not math.isfinite(value) for value in values)
        or values[2] <= 0
        or values[3] <= 0
    ):
        return None

    box = tuple(int(round(value)) for value in values)
    if box[2] <= 0 or box[3] <= 0:
        return None
    return box  # type: ignore[return-value]


def _area_fraction(
    area: Mapping[str, object], frame_width: int, frame_height: int
) -> float | None:
    box = face_box(area)
    if box is None:
        return None
    try:
        width = int(frame_width)
        height = int(frame_height)
    except (TypeError, ValueError, OverflowError):
        return None
    if width <= 0 or height <= 0:
        return None
    fraction = (box[2] * box[3]) / (width * height)
    return fraction if math.isfinite(fraction) else None


def select_enrollment_face(
    representations: Sequence[Mapping[str, object]],
    frame_width: int,
    frame_height: int,
    *,
    min_face_fraction: float = 0.02,
    max_background_fraction: float = 0.01,
) -> Mapping[str, object] | None:
    """Select exactly one dominant face from DeepFace enrollment results.

    OpenCV's Haar detector occasionally reports tiny false positives in the
    background.  Rejecting every image with more than one detection needlessly
    discards a usable enrollment photo in that case.  This helper accepts an
    image only when exactly one detection covers at least
    ``min_face_fraction`` of the frame and every other detection is below
    ``max_background_fraction``.  Two plausibly sized faces still fail, so
    group photos are not enrolled as one person.
    """

    if not isinstance(representations, (list, tuple)):
        return None
    try:
        primary_min = float(min_face_fraction)
        background_max = float(max_background_fraction)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(primary_min)
        or not math.isfinite(background_max)
        or primary_min <= 0
        or background_max < 0
        or background_max >= primary_min
    ):
        return None

    primary: tuple[float, Mapping[str, object]] | None = None
    for representation in representations:
        if not isinstance(representation, Mapping):
            return None
        area = representation.get("facial_area")
        if not isinstance(area, Mapping):
            return None
        if not is_plausible_face_area(area, frame_width, frame_height):
            continue
        fraction = _area_fraction(area, frame_width, frame_height)
        if fraction is None:
            return None
        if fraction >= primary_min:
            if primary is not None:
                return None
            primary = (fraction, representation)
        elif fraction >= background_max:
            return None

    return primary[1] if primary is not None else None


def is_plausible_face_area(
    area: Mapping[str, object],
    frame_width: int,
    frame_height: int,
) -> bool:
    """Reject tiny areas and DeepFace's full-frame no-face placeholder.

    ``enforce_detection=False`` is correct for video frames, but some DeepFace
    detector backends return the whole image as a synthetic face when nothing
    was detected.  That placeholder must never become a known identity.
    """

    box = face_box(area)
    if box is None:
        return False
    _, _, width, height = box
    if width < 24 or height < 24:
        return False
    try:
        frame_width = int(frame_width)
        frame_height = int(frame_height)
    except (TypeError, ValueError, OverflowError):
        return True
    if frame_width <= 0 or frame_height <= 0:
        return True
    if width > frame_width * 1.05 or height > frame_height * 1.05:
        return False
    # A real, detector-produced face box never covers essentially the entire
    # frame.  DeepFace's no-face fallback does exactly that.
    if width >= frame_width * 0.98 and height >= frame_height * 0.98:
        return False
    return True


class StableIdentityTracker:
    """Confirm accepted identity names across consecutive inference batches.

    This is a name-level confirmation helper.  It does not associate a name
    with a particular face or provide liveness/security guarantees.
    """

    def __init__(self, confirmations: int = 3) -> None:
        if isinstance(confirmations, bool) or not isinstance(confirmations, int):
            raise TypeError("confirmations must be an integer")
        if confirmations < 1:
            raise ValueError("confirmations must be at least 1")
        self.confirmations = confirmations
        self._counts: dict[str, int] = {}

    def update(self, results: list[MatchResult]) -> set[str]:
        active: set[str] = set()
        for result in results:
            if not isinstance(result, MatchResult):
                continue
            try:
                score = float(result.score)
                runner_up_score = float(result.runner_up_score)
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                not result.accepted
                or not isinstance(result.name, str)
                or not result.name.strip()
                or result.name.strip().lower() == "unknown"
                or not math.isfinite(score)
                or not math.isfinite(runner_up_score)
            ):
                continue
            active.add(result.name)

        for name in tuple(self._counts):
            if name not in active:
                del self._counts[name]
        for name in active:
            self._counts[name] = self._counts.get(name, 0) + 1

        return {
            name for name in active if self._counts[name] >= self.confirmations
        }
