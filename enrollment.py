"""Face enrollment, dataset checks, and a private embedding cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from face_utils import normalize, select_enrollment_face

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORED_DIRECTORIES = {
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    "tests",
    "node_modules",
}
CACHE_SCHEMA = 1


@dataclass(frozen=True)
class PersonImages:
    name: str
    images: tuple[Path, ...]


@dataclass(frozen=True)
class DatasetSummary:
    people: int
    images: int
    min_images: int
    max_images: int


def iter_person_images(training_path: Path) -> list[PersonImages]:
    """Return usable person folders and their image files.

    Person folders are direct children of ``training_path``; images may be
    nested one or more levels below a person folder.  Hidden folders, Python
    caches, virtual environments, and test assets are ignored.
    """

    root = Path(training_path)
    if not root.is_dir():
        raise RuntimeError(f"Training folder does not exist: {root}")

    people: list[PersonImages] = []
    for person_path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if (
            not person_path.is_dir()
            or person_path.name.startswith(".")
            or person_path.name in IGNORED_DIRECTORIES
        ):
            continue
        images: list[Path] = []
        for candidate in sorted(
            person_path.rglob("*"), key=lambda item: str(item).casefold()
        ):
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_parts = candidate.relative_to(person_path).parts
            if any(part.startswith(".") for part in relative_parts):
                continue
            images.append(candidate)
        if images:
            people.append(PersonImages(person_path.name, tuple(images)))
    return people


def dataset_summary(training_path: Path) -> DatasetSummary:
    try:
        people = iter_person_images(training_path)
    except RuntimeError:
        return DatasetSummary(0, 0, 0, 0)
    counts = [len(person.images) for person in people]
    return DatasetSummary(
        people=len(people),
        images=sum(counts),
        min_images=min(counts) if counts else 0,
        max_images=max(counts) if counts else 0,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    """Return (width, height) for an image, or ``None`` when unreadable."""

    try:
        import cv2
    except ImportError:
        return None
    try:
        image = cv2.imread(str(path))
    except Exception:
        return None
    if image is None or getattr(image, "ndim", 0) < 2:
        return None
    try:
        return int(image.shape[1]), int(image.shape[0])
    except (IndexError, TypeError, ValueError):
        return None


def _valid_embedding(value: Any) -> tuple[float, ...] | None:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        return None
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not numbers or any(not math.isfinite(item) for item in numbers):
        return None
    return numbers


def load_embedding_cache(
    path: Path,
    *,
    model: str,
    detector: str,
    align: bool = True,
) -> dict[str, tuple[float, ...]]:
    """Load a cache only when it was produced by the same face pipeline."""

    cache_path = Path(path)
    if not cache_path.is_file():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("Embedding cache could not be read; rebuilding it.")
        return {}
    if not isinstance(payload, dict):
        return {}
    if (
        payload.get("schema") != CACHE_SCHEMA
        or payload.get("model") != model
        or payload.get("detector") != detector
        or bool(payload.get("align", True)) != bool(align)
    ):
        print("Embedding cache does not match the selected face model; rebuilding it.")
        return {}
    images = payload.get("images")
    if not isinstance(images, dict):
        return {}

    cache: dict[str, tuple[float, ...]] = {}
    for key, entry in images.items():
        if not isinstance(key, str) or len(key) != 64:
            continue
        value = entry.get("embedding") if isinstance(entry, dict) else entry
        embedding = _valid_embedding(value)
        if embedding is not None:
            cache[key] = embedding
    return cache


def save_embedding_cache(
    path: Path,
    *,
    model: str,
    detector: str,
    align: bool = True,
    entries: Mapping[str, Sequence[float]],
) -> bool:
    """Atomically write the cache with best-effort owner-only permissions."""

    cache_path = Path(path)
    payload = {
        "schema": CACHE_SCHEMA,
        "model": model,
        "detector": detector,
        "align": bool(align),
        "images": {
            key: {"embedding": [float(value) for value in embedding]}
            for key, embedding in entries.items()
        },
    }
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, cache_path)
        try:
            os.chmod(cache_path, 0o600)
        except OSError:
            pass
        return True
    except (OSError, TypeError, ValueError) as error:
        print(f"Embedding cache could not be saved: {error}")
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def load_known_embeddings(
    settings: Any,
    *,
    deepface: Any = None,
    force_rebuild: bool = False,
) -> dict[str, list[tuple[float, ...]]]:
    """Build embeddings for every valid image, using a verified local cache.

    Enrollment still requires exactly one detectable face per image.  Camera
    frames deliberately use a different, lenient detection path.
    """

    if deepface is None:
        try:
            from deepface import DeepFace
        except ImportError as error:
            raise RuntimeError(
                "DeepFace is not installed. Install requirements.txt first."
            ) from error
        deepface = DeepFace

    training_path = Path(settings.training_path)
    people = iter_person_images(training_path)
    print(f"Loading known faces from {training_path} ...")

    cache_enabled = bool(getattr(settings, "use_embedding_cache", True))
    cache_path = Path(settings.embedding_cache_path)
    cache: dict[str, tuple[float, ...]] = {}
    if cache_enabled and not force_rebuild:
        cache = load_embedding_cache(
            cache_path,
            model=settings.face_model,
            detector=settings.face_detector,
            align=True,
        )

    used_cache_entries: dict[str, tuple[float, ...]] = {}
    known_embeddings: dict[str, list[tuple[float, ...]]] = {}
    image_owners: dict[str, str] = {}
    cached_count = 0
    computed_count = 0

    for person in people:
        embeddings: list[tuple[float, ...]] = []
        for image_path in person.images:
            try:
                key = sha256_file(image_path)
            except OSError as error:
                print(f"Skipped {person.name}/{image_path.name}: {error}")
                continue

            duplicate_owner = image_owners.get(key)
            if duplicate_owner is not None:
                print(
                    f"Skipped {person.name}/{image_path.name}: identical image "
                    f"already used for {duplicate_owner}."
                )
                continue
            embedding = cache.get(key)
            if embedding is not None:
                image_owners[key] = person.name
                embeddings.append(embedding)
                used_cache_entries[key] = embedding
                cached_count += 1
                print(f"Loaded {image_path.name} for {person.name} (cache)")
                continue

            try:
                representations = deepface.represent(
                    str(image_path),
                    model_name=settings.face_model,
                    detector_backend=settings.face_detector,
                    align=True,
                    enforce_detection=True,
                )
            except Exception as error:
                print(f"Skipped {person.name}/{image_path.name}: {error}")
                continue
            if isinstance(representations, dict):
                representations = [representations]
            if not isinstance(representations, (list, tuple)) or not representations:
                print(
                    f"Skipped {person.name}/{image_path.name}: "
                    "the image must contain exactly one dominant face."
                )
                continue
            dimensions = _image_dimensions(image_path)
            if dimensions is None:
                selected = representations[0] if len(representations) == 1 else None
            else:
                selected = select_enrollment_face(
                    representations, dimensions[0], dimensions[1]
                )
            if selected is None:
                print(
                    f"Skipped {person.name}/{image_path.name}: "
                    "the image must contain exactly one dominant face."
                )
                continue
            if not isinstance(selected, Mapping):
                print(f"Skipped {person.name}/{image_path.name}: invalid detection.")
                continue
            vector = normalize(selected.get("embedding", []))
            if not vector or not any(value != 0 for value in vector):
                print(f"Skipped {person.name}/{image_path.name}: invalid embedding.")
                continue
            image_owners[key] = person.name
            embeddings.append(vector)
            used_cache_entries[key] = vector
            computed_count += 1
            print(f"Loaded {image_path.name} for {person.name}")

        if embeddings:
            known_embeddings[person.name] = embeddings

    if cache_enabled and used_cache_entries:
        save_embedding_cache(
            cache_path,
            model=settings.face_model,
            detector=settings.face_detector,
            align=True,
            entries=used_cache_entries,
        )

    print(
        f"Loaded {len(known_embeddings)} people "
        f"({computed_count} new, {cached_count} cached)."
    )
    missing = [person.name for person in people if person.name not in known_embeddings]
    if missing:
        print("Warning: no usable images for: " + ", ".join(sorted(missing)))
    if not known_embeddings:
        print("Warning: no usable training faces were found.")
    return known_embeddings
