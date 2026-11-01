"""Estimate face-recognition similarity thresholds from enrollment images.

Run from the project root after installing the camera/ML requirements:

    python -B -m tools.calibrate_threshold
    python -B -m tools.calibrate_threshold --rebuild

The tool computes genuine (same-person) and impostor (different-person)
cosine scores, then reports the current threshold, equal-error threshold, and
a small recommended range.  It never edits configuration automatically.

You need at least two usable images for at least two people.  With the
single-image-per-person demo set it exits with a clear message instead of
pretending the result is statistically meaningful.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in {None, ""}:  # Support `python tools/calibrate_threshold.py`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enrollment import load_known_embeddings  # noqa: E402
from face_utils import cosine_similarity  # noqa: E402
from runtime_env import configure_runtime_environment  # noqa: E402
from settings import Settings  # noqa: E402

PROJECT_PATH = Path(__file__).resolve().parents[1]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _pair_scores(
    embeddings: dict[str, list[tuple[float, ...]]]
) -> tuple[list[float], list[float]]:
    genuine: list[float] = []
    impostor: list[float] = []

    for person_embeddings in embeddings.values():
        for left, right in combinations(person_embeddings, 2):
            genuine.append(cosine_similarity(left, right))

    people = list(embeddings)
    for left_index, left_name in enumerate(people):
        for right_name in people[left_index + 1 :]:
            for left in embeddings[left_name]:
                for right in embeddings[right_name]:
                    impostor.append(cosine_similarity(left, right))
    return genuine, impostor


def _rates_at(
    genuine: Sequence[float], impostor: Sequence[float], threshold: float
) -> tuple[float, float]:
    """Return false-accept and false-reject rates at one threshold."""

    false_accept = (
        sum(score >= threshold for score in impostor) / len(impostor)
        if impostor
        else 0.0
    )
    false_reject = (
        sum(score < threshold for score in genuine) / len(genuine)
        if genuine
        else 1.0
    )
    return false_accept, false_reject


def _equal_error_threshold(
    genuine: Sequence[float], impostor: Sequence[float]
) -> tuple[float, float, float]:
    """Return (threshold, false_accept_rate, false_reject_rate)."""

    best: tuple[float, float, float, float] | None = None
    steps = 200
    for step in range(steps + 1):
        threshold = step / steps
        false_accept, false_reject = _rates_at(genuine, impostor, threshold)
        gap = abs(false_accept - false_reject)
        if best is None or gap < best[0]:
            best = (gap, threshold, false_accept, false_reject)
    assert best is not None
    return best[1], best[2], best[3]


def _threshold_table(
    genuine: Sequence[float], impostor: Sequence[float]
) -> list[dict[str, float]]:
    """Return an interpretable FAR/FRR sweep for report appendices."""

    table: list[dict[str, float]] = []
    for index in range(11):
        threshold = 0.40 + index * 0.05
        false_accept, false_reject = _rates_at(genuine, impostor, threshold)
        table.append(
            {
                "threshold": round(threshold, 2),
                "false_accept_rate": round(false_accept, 3),
                "false_reject_rate": round(false_reject, 3),
            }
        )
    return table


def _describe(values: Sequence[float]) -> str:
    if not values:
        return "none"
    return (
        f"n={len(values)} min={min(values):.3f} "
        f"p10={_percentile(values, 0.10):.3f} "
        f"median={statistics.median(values):.3f} "
        f"p90={_percentile(values, 0.90):.3f} max={max(values):.3f}"
    )


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate similarity thresholds from enrollment images"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore the embedding cache and re-extract every image",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a machine-readable summary",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings.from_env(PROJECT_PATH)
    configure_runtime_environment(PROJECT_PATH)
    try:
        embeddings = load_known_embeddings(settings, force_rebuild=args.rebuild)
    except RuntimeError as error:
        print(f"Cannot calibrate: {error}")
        return 2

    counts = {name: len(items) for name, items in embeddings.items()}
    print("Enrollment embeddings:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    if len(embeddings) < 2:
        print("Calibration needs at least two people with usable images.")
        return 2
    if not any(count >= 2 for count in counts.values()):
        print(
            "Calibration needs at least one person with two or more images. "
            "Add varied photos before tuning thresholds."
        )
        return 2

    genuine, impostor = _pair_scores(embeddings)
    if len(genuine) < 3 or len(impostor) < 3:
        print(
            "Not enough genuine/impostor pairs for a stable estimate "
            f"(genuine={len(genuine)}, impostor={len(impostor)})."
        )
        return 2

    threshold, far, frr = _equal_error_threshold(genuine, impostor)
    genuine_min = min(genuine)
    impostor_max = max(impostor)
    separable = genuine_min > impostor_max
    if separable:
        midpoint = (genuine_min + impostor_max) / 2
        recommended_low = max(0.0, midpoint - 0.02)
        recommended_high = min(1.0, midpoint + 0.02)
    else:
        recommended_low = max(0.0, threshold - 0.02)
        recommended_high = min(1.0, threshold + 0.02)

    table = _threshold_table(genuine, impostor)
    summary = {
        "evaluation_set": "enrollment_images_resubstitution",
        "people": len(embeddings),
        "images": sum(counts.values()),
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "configured_threshold": settings.similarity_threshold,
        "configured_margin": settings.match_margin,
        "equal_error_threshold": round(threshold, 3),
        "false_accept_rate": round(far, 3),
        "false_reject_rate": round(frr, 3),
        "genuine_min": round(genuine_min, 3),
        "impostor_max": round(impostor_max, 3),
        "separable": separable,
        "recommended_range": [round(recommended_low, 3), round(recommended_high, 3)],
        "threshold_table": table,
        "caveat": (
            "Resubstitution estimate on enrollment images; use held-out images "
            "for report-quality FAR/FRR."
        ),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"\nGenuine scores:  {_describe(genuine)}")
    print(f"Impostor scores: {_describe(impostor)}")
    print(f"\nConfigured threshold: {settings.similarity_threshold:.3f}")
    print(f"Configured margin:    {settings.match_margin:.3f}")
    print(
        f"Equal-error threshold: {threshold:.3f} "
        f"(FAR={far:.1%}, FRR={frr:.1%})"
    )
    print("\nThreshold sweep:")
    print("  threshold    FAR      FRR")
    for row in table:
        print(
            f"  {row['threshold']:.2f}       "
            f"{row['false_accept_rate']:.1%}    "
            f"{row['false_reject_rate']:.1%}"
        )
    print(
        "Suggested SIMILARITY_THRESHOLD range: "
        f"{recommended_low:.2f}-{recommended_high:.2f}"
    )
    if separable:
        print(
            "Genuine and impostor scores are separable "
            f"(min genuine {genuine_min:.3f} > max impostor {impostor_max:.3f})."
        )
    else:
        print(
            "Genuine and impostor scores overlap "
            f"(min genuine {genuine_min:.3f}, max impostor {impostor_max:.3f}); "
            "add more varied images and re-check."
        )
    if settings.match_margin < 0.05:
        print(
            "Warning: MATCH_MARGIN below 0.05 increases the chance of "
            "confusing similar people."
        )
    print(
        "\nThis is a resubstitution estimate from your enrollment set, not a "
        "held-out evaluation or a guarantee. Validate with separate images of "
        "the actual people and lighting."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
