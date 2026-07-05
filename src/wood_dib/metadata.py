"""Metadata construction for raw IC4SD-Wood-Eucalyptus image folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable

from .constants import CLASS_TO_INDEX, IMAGE_EXTENSIONS
from .utils import (
    infer_light_condition,
    infer_specimen_key,
    is_hidden_or_system,
    is_image_file,
    normalize_class_name,
    parse_group_id,
    sha256_file,
    specimen_group_id,
)


@dataclass(frozen=True)
class ManifestBuildResult:
    metadata: pd.DataFrame
    skipped_files: pd.DataFrame
    unreadable_images: pd.DataFrame
    class_notes: list[dict[str, str]]
    phash_backend: str


def _load_phash_dependency():
    try:
        import imagehash
    except ModuleNotFoundError:
        return None
    return imagehash


def _fallback_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """Compute a deterministic pHash using only Pillow and NumPy."""
    img_size = hash_size * highfreq_factor
    pixels = np.asarray(
        image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    n = pixels.shape[0]
    x = np.arange(n)
    u = np.arange(n)[:, None]
    dct_matrix = np.cos(((2 * x + 1) * u * np.pi) / (2 * n))
    dct_matrix[0, :] *= 1 / np.sqrt(2)
    dct_matrix *= np.sqrt(2 / n)
    dct = dct_matrix @ pixels @ dct_matrix.T
    low_freq = dct[:hash_size, :hash_size]
    median = np.median(low_freq[1:, 1:])
    bits = (low_freq > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:0{hash_size * hash_size // 4}x}"


def compute_phash_value(image: Image.Image, imagehash_module: object | None) -> str:
    if imagehash_module is not None:
        return str(imagehash_module.phash(image.convert("RGB")))
    return _fallback_phash(image)


def scan_raw_dataset(
    raw_root: Path,
    compute_sha256: bool = True,
    compute_phash: bool = False,
) -> ManifestBuildResult:
    raw_root = raw_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset root does not exist or is not a directory: {raw_root}")

    imagehash = _load_phash_dependency() if compute_phash else None
    phash_backend = "not_computed"
    if compute_phash and imagehash is None:
        print("[WARN] imagehash is not installed; using built-in PIL/NumPy pHash fallback.", flush=True)
        phash_backend = "pil_numpy_fallback"
    elif compute_phash:
        phash_backend = "imagehash.phash"

    class_dirs = [p for p in sorted(raw_root.iterdir()) if p.is_dir() and not is_hidden_or_system(p)]
    if not class_dirs:
        raise FileNotFoundError(f"No top-level class folders found under: {raw_root}")

    rows: list[dict] = []
    skipped_rows: list[dict] = []
    unreadable_rows: list[dict] = []
    class_notes: list[dict[str, str]] = []

    for class_dir in class_dirs:
        try:
            class_name, note = normalize_class_name(class_dir.name)
        except ValueError as exc:
            skipped_rows.append(
                {
                    "path": str(class_dir),
                    "reason": str(exc),
                    "kind": "top_level_folder",
                }
            )
            continue

        if note:
            class_notes.append({"top_level_folder": class_dir.name, "class_name": class_name, "note": note})

        all_files = sorted(p for p in class_dir.rglob("*") if p.is_file())
        for path in tqdm(all_files, desc=f"Scanning {class_dir.name}", leave=False):
            if is_hidden_or_system(path):
                skipped_rows.append({"path": str(path), "reason": "hidden/system file", "kind": "file"})
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                skipped_rows.append({"path": str(path), "reason": "unsupported extension", "kind": "file"})
                continue
            if not is_image_file(path):
                skipped_rows.append({"path": str(path), "reason": "not a valid image file", "kind": "file"})
                continue

            relative_path = path.relative_to(raw_root)
            warning_notes: list[str] = []
            width = None
            height = None
            phash = ""
            try:
                with Image.open(path) as image:
                    image = ImageOps.exif_transpose(image)
                    width, height = image.size
                    if compute_phash:
                        phash = compute_phash_value(image, imagehash)
            except Exception as exc:
                unreadable_rows.append(
                    {
                        "path": str(path),
                        "relative_path": relative_path.as_posix(),
                        "class_name": class_name,
                        "error": repr(exc),
                    }
                )
                continue

            file_hash = ""
            if compute_sha256:
                try:
                    file_hash = sha256_file(path)
                except Exception as exc:
                    warning_notes.append(f"sha256_failed={exc!r}")

            group_id = specimen_group_id(relative_path)

            rows.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "raw_path": str(path.resolve()),
                    "original_filename": path.name,
                    "top_level_folder": class_dir.name,
                    "class_name": class_name,
                    "class_index": CLASS_TO_INDEX[class_name],
                    "width": int(width),
                    "height": int(height),
                    "file_extension": path.suffix.lower(),
                    "sha256": file_hash,
                    "phash": phash,
                    "group_id": group_id,
                    "specimen_key": infer_specimen_key(group_id),
                    "light_condition": infer_light_condition(relative_path),
                    # Kept for backwards compatibility with older release scripts.
                    "parsed_group_id": group_id,
                    "legacy_parsed_group_id": parse_group_id(relative_path, class_name),
                    "notes": "; ".join(warning_notes),
                }
            )

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise FileNotFoundError(f"No readable image files found under: {raw_root}")

    metadata = metadata.sort_values(["class_index", "relative_path"]).reset_index(drop=True)
    metadata.insert(0, "image_id", [f"IC4SD_EUC_{idx:06d}" for idx in range(1, len(metadata) + 1)])

    return ManifestBuildResult(
        metadata=metadata,
        skipped_files=pd.DataFrame(skipped_rows),
        unreadable_images=pd.DataFrame(unreadable_rows),
        class_notes=class_notes,
        phash_backend=phash_backend,
    )


def summarize_manifest(metadata: pd.DataFrame, skipped_count: int, unreadable_count: int, raw_root: Path) -> dict:
    class_counts = metadata.groupby("class_name").size().sort_index().to_dict()
    specimen_counts = metadata.groupby("class_name")["group_id"].nunique().sort_index().to_dict()
    extension_counts = (
        metadata.groupby(["class_name", "file_extension"])
        .size()
        .reset_index(name="n_images")
        .to_dict("records")
    )
    light_counts = (
        metadata.groupby(["class_name", "light_condition"])
        .size()
        .reset_index(name="n_images")
        .to_dict("records")
        if "light_condition" in metadata.columns
        else []
    )
    size_summary = {
        "width": {
            "min": int(metadata["width"].min()),
            "max": int(metadata["width"].max()),
            "mean": float(metadata["width"].mean()),
        },
        "height": {
            "min": int(metadata["height"].min()),
            "max": int(metadata["height"].max()),
            "mean": float(metadata["height"].mean()),
        },
    }
    return {
        "dataset_name": "IC4SD-Wood-Eucalyptus",
        "raw_root": str(raw_root.expanduser().resolve()),
        "total_images": int(len(metadata)),
        "n_classes": int(metadata["class_name"].nunique()),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "specimen_counts": {str(k): int(v) for k, v in specimen_counts.items()},
        "file_extension_counts": extension_counts,
        "light_condition_counts": light_counts,
        "skipped_files": int(skipped_count),
        "unreadable_images": int(unreadable_count),
        "image_size_summary": size_summary,
    }
