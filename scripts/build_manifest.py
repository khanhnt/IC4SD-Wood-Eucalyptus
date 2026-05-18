#!/usr/bin/env python3
"""Build a metadata manifest from raw IC4SD-Wood-Eucalyptus image folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from wood_dib.constants import CANONICAL_CLASSES
from wood_dib.metadata import scan_raw_dataset, summarize_manifest
from wood_dib.utils import label_map_payload, save_json, timestamp_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IC4SD-Wood-Eucalyptus metadata manifest.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw dataset root with one top-level folder per class.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metadata outputs.")
    parser.add_argument("--no-sha256", action="store_true", help="Skip SHA-256 computation.")
    parser.add_argument("--compute-phash", action="store_true", help="Compute perceptual hashes if imagehash is installed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Manifest] Raw root: {args.raw_root}", flush=True)
    result = scan_raw_dataset(
        raw_root=args.raw_root,
        compute_sha256=not args.no_sha256,
        compute_phash=args.compute_phash,
    )

    metadata_path = args.output_dir / "metadata.csv"
    result.metadata.to_csv(metadata_path, index=False)

    result.skipped_files.to_csv(args.output_dir / "skipped_files.csv", index=False)
    result.unreadable_images.to_csv(args.output_dir / "unreadable_images.csv", index=False)
    if result.class_notes:
        save_json(args.output_dir / "class_normalization_notes.json", result.class_notes)

    label_map = label_map_payload()
    save_json(args.output_dir / "label_map.json", label_map)

    summary = summarize_manifest(
        result.metadata,
        skipped_count=len(result.skipped_files),
        unreadable_count=len(result.unreadable_images),
        raw_root=args.raw_root,
    )
    summary["created_at_utc"] = timestamp_utc()
    summary["expected_classes"] = CANONICAL_CLASSES
    summary["metadata_csv"] = str(metadata_path)
    save_json(args.output_dir / "manifest_summary.json", summary)

    print("\n[Manifest] Summary", flush=True)
    print(f"  Total readable images: {summary['total_images']}", flush=True)
    print(f"  Number of classes:     {summary['n_classes']}", flush=True)
    for class_name, count in summary["class_counts"].items():
        print(f"  {class_name:28s} {count:5d}", flush=True)
    print(f"  Skipped files:         {summary['skipped_files']}", flush=True)
    print(f"  Unreadable images:     {summary['unreadable_images']}", flush=True)
    print(
        "  Width min/max/mean:    "
        f"{summary['image_size_summary']['width']['min']} / "
        f"{summary['image_size_summary']['width']['max']} / "
        f"{summary['image_size_summary']['width']['mean']:.1f}",
        flush=True,
    )
    print(
        "  Height min/max/mean:   "
        f"{summary['image_size_summary']['height']['min']} / "
        f"{summary['image_size_summary']['height']['max']} / "
        f"{summary['image_size_summary']['height']['mean']:.1f}",
        flush=True,
    )
    print(f"[Manifest] Saved: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
