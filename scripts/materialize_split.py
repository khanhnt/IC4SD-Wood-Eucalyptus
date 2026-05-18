#!/usr/bin/env python3
"""Materialize a split manifest into an ImageFolder train/val/test tree."""

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


LEGACY_FOLDER_NAMES = {
    "Eucalyptus camaldulensis": "Eucalyptus_camandulensis",
    "Eucalyptus cladocalyx": "Eucalyptus_cladocalyx",
    "Eucalyptus deglupta": "Eucalyptus_daglupta",
    "Eucalyptus diversicolor": "Eucalyptus_diversicolor",
    "Eucalyptus grandis": "Eucalyptus_grandis",
    "Eucalyptus microcorys": "Eucalyptus_microcorys",
    "Eucalyptus saligna": "Eucalyptus_saligna",
    "Syzygium hemisphericum": "Syzygium_hemisphericum",
}


def class_folder_name(class_name: str, style: str) -> str:
    if style == "canonical":
        return class_name
    if style == "underscore":
        return class_name.replace(" ", "_")
    if style == "legacy":
        return LEGACY_FOLDER_NAMES.get(class_name, class_name.replace(" ", "_"))
    raise ValueError(f"Unknown folder-name style: {style}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy split images into ImageFolder train/val/test directories.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folder-name-style", choices=["legacy", "underscore", "canonical"], default="legacy")
    parser.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true", help="Remove output root before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Missing raw root: {raw_root}")

    metadata = load_metadata(args.metadata_csv)
    split_df = pd.read_csv(args.split_csv)
    validate_split_completeness(split_df)

    frame = split_df.merge(
        metadata[["image_id", "relative_path", "raw_path"] if "raw_path" in metadata.columns else ["image_id", "relative_path"]],
        on="image_id",
        how="left",
        suffixes=("", "_metadata"),
    )
    if frame["relative_path"].isna().any():
        missing = frame.loc[frame["relative_path"].isna(), "image_id"].head(10).tolist()
        raise ValueError(f"Split contains image_id not present in metadata. Examples: {missing}")

    output_root = args.output_root
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_paths = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="Materializing split"):
        src = raw_root / str(row["relative_path"])
        if not src.exists() and "raw_path" in row and pd.notna(row["raw_path"]):
            candidate = Path(str(row["raw_path"]))
            if candidate.exists():
                src = candidate
        if not src.exists():
            missing_paths.append(str(src))
            continue

        split = str(row["split"])
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unexpected split value: {split}")

        class_dir = class_folder_name(str(row["class_name"]), args.folder_name_style)
        dst_dir = output_root / split / class_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            dst = dst_dir / f"{stem}__{row['image_id']}{suffix}"

        if args.copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src)

        rows.append(
            {
                "image_id": row["image_id"],
                "split": split,
                "class_name": row["class_name"],
                "class_folder": class_dir,
                "source_path": str(src),
                "output_path": str(dst),
            }
        )

    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} source images could not be resolved. Example: {missing_paths[0]}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_root / "materialized_manifest.csv", index=False)
    counts = manifest.groupby(["split", "class_folder"]).size().reset_index(name="n_images")
    counts.to_csv(output_root / "materialized_counts.csv", index=False)

    print(f"[Done] Materialized {len(manifest)} images to {output_root}", flush=True)
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
