#!/usr/bin/env python3
"""Create a public release dataset folder that matches a split manifest.

This script is useful when the original raw archive contains extra images that
were excluded from the strict technical-validation split. It copies only the
images referenced by a validated split manifest, preserving the nested raw
structure under normalized top-level class folders.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from wood_dib.constants import SPLIT_NAMES
from wood_dib.splitting import load_metadata, validate_split_completeness
from wood_dib.utils import label_map_payload, save_json, timestamp_utc


CLASS_FOLDER_NAMES = {
    "Eucalyptus camaldulensis": "Eucalyptus_camaldulensis",
    "Eucalyptus cladocalyx": "Eucalyptus_cladocalyx",
    "Eucalyptus deglupta": "Eucalyptus_deglupta",
    "Eucalyptus diversicolor": "Eucalyptus_diversicolor",
    "Eucalyptus grandis": "Eucalyptus_grandis",
    "Eucalyptus microcorys": "Eucalyptus_microcorys",
    "Eucalyptus saligna": "Eucalyptus_saligna",
    "Syzygium hemisphericum": "Syzygium_hemisphericum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the strict-split images into a clean public dataset release folder."
    )
    parser.add_argument("--source-raw-root", type=Path, required=True, help="Original full raw archive root.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Metadata built from the source raw root.")
    parser.add_argument("--split-csv", type=Path, required=True, help="Validated split manifest to define included images.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset folder, e.g. .../public_dib_code/dataset.")
    parser.add_argument(
        "--folder-name-style",
        choices=["underscore", "canonical"],
        default="underscore",
        help="Top-level class folder naming style in output raw/. Default: underscore.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Use copy for Google Drive release folders; symlink is useful only locally.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output-root before creating the release dataset.")
    return parser.parse_args()


def class_folder_name(class_name: str, style: str) -> str:
    if style == "canonical":
        return class_name
    return CLASS_FOLDER_NAMES.get(class_name, class_name.replace(" ", "_"))


def relative_inside_class(relative_path: str) -> Path:
    parts = Path(str(relative_path)).parts
    if len(parts) <= 1:
        return Path(parts[-1])
    return Path(*parts[1:])


def resolve_source_path(source_raw_root: Path, row: pd.Series) -> Path:
    candidate = source_raw_root / str(row["relative_path"])
    if candidate.exists():
        return candidate
    if "raw_path" in row and pd.notna(row["raw_path"]):
        raw_candidate = Path(str(row["raw_path"]))
        if raw_candidate.exists():
            return raw_candidate
    raise FileNotFoundError(
        f"Cannot resolve source image for image_id={row['image_id']} relative_path={row['relative_path']}"
    )


def main() -> None:
    args = parse_args()
    source_raw_root = args.source_raw_root.expanduser().resolve()
    if not source_raw_root.is_dir():
        raise FileNotFoundError(f"Missing source raw root: {source_raw_root}")

    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    release_raw = args.output_root / "raw"
    release_splits = args.output_root / "splits"
    release_metadata = args.output_root / "metadata"
    for directory in [release_raw, release_splits, release_metadata]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_csv)
    split_df = pd.read_csv(args.split_csv)
    validate_split_completeness(split_df)

    frame = split_df.merge(
        metadata,
        on=["image_id", "relative_path", "class_name", "class_index"],
        how="left",
        suffixes=("", "_metadata"),
    )
    if frame["raw_path"].isna().any() if "raw_path" in frame.columns else False:
        missing = frame.loc[frame["raw_path"].isna(), "image_id"].head(10).tolist()
        raise ValueError(f"Split rows missing from metadata. Examples: {missing}")

    copied_rows = []
    seen_destinations: set[Path] = set()
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Copying release images"):
        src = resolve_source_path(source_raw_root, row)
        class_dir = class_folder_name(str(row["class_name"]), args.folder_name_style)
        dst_rel = Path(class_dir) / relative_inside_class(str(row["relative_path"]))
        dst = release_raw / dst_rel
        if dst in seen_destinations or dst.exists():
            dst = dst.with_name(f"{dst.stem}__{row['image_id']}{dst.suffix}")
        dst.parent.mkdir(parents=True, exist_ok=True)

        if args.copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)
        seen_destinations.add(dst)

        copied_rows.append(
            {
                "image_id": row["image_id"],
                "split": row["split"],
                "class_name": row["class_name"],
                "class_index": int(row["class_index"]),
                "source_relative_path": row["relative_path"],
                "release_relative_path": dst.relative_to(args.output_root).as_posix(),
                "release_raw_relative_path": dst.relative_to(release_raw).as_posix(),
                "original_filename": row.get("original_filename", Path(str(row["relative_path"])).name),
            }
        )

    release_manifest = pd.DataFrame(copied_rows)
    release_manifest.to_csv(release_metadata / "release_manifest.csv", index=False)

    # Save metadata/split files in release-relative form for public reuse.
    release_metadata_df = frame.copy()
    release_path_map = release_manifest.set_index("image_id")["release_raw_relative_path"].to_dict()
    release_metadata_df["relative_path"] = release_metadata_df["image_id"].map(release_path_map)
    release_metadata_df["raw_path"] = release_metadata_df["relative_path"].map(lambda p: f"raw/{p}")
    release_metadata_df.to_csv(release_metadata / "metadata.csv", index=False)

    release_split_df = split_df.copy()
    release_split_df["relative_path"] = release_split_df["image_id"].map(release_path_map)
    release_split_df.to_csv(release_splits / "split_B_strict.csv", index=False)

    save_json(release_metadata / "label_map.json", label_map_payload())

    counts = release_manifest.groupby(["split", "class_name"]).size().reset_index(name="n_images")
    counts.to_csv(release_metadata / "release_counts.csv", index=False)
    split_counts = {split: int((release_manifest["split"] == split).sum()) for split in SPLIT_NAMES}
    class_counts = release_manifest.groupby("class_name").size().sort_index().to_dict()
    summary = {
        "created_at_utc": timestamp_utc(),
        "source_raw_root": str(source_raw_root),
        "output_root": str(args.output_root),
        "n_images": int(len(release_manifest)),
        "split_counts": split_counts,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "folder_name_style": args.folder_name_style,
        "copy_mode": args.copy_mode,
        "release_files": {
            "raw": str(release_raw),
            "metadata_csv": str(release_metadata / "metadata.csv"),
            "split_csv": str(release_splits / "split_B_strict.csv"),
            "release_manifest": str(release_metadata / "release_manifest.csv"),
            "counts": str(release_metadata / "release_counts.csv"),
        },
    }
    save_json(args.output_root / "dataset_release_summary.json", summary)

    print(f"[Done] Copied {len(release_manifest)} images to {release_raw}", flush=True)
    print("Split counts:", split_counts, flush=True)
    print(f"Summary: {args.output_root / 'dataset_release_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
