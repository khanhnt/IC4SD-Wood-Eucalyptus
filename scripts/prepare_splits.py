#!/usr/bin/env python3
"""Validate an existing split manifest or generate a deterministic split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from wood_dib.splitting import (
    add_phash_components,
    generate_component_aware_split,
    generate_stratified_split,
    load_metadata,
    split_distribution,
    validate_existing_split,
    validate_split_completeness,
)
from wood_dib.utils import save_json, timestamp_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or validate an IC4SD-Wood-Eucalyptus split manifest.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--existing-split-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="split_B_strict")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--use-phash-components", action="store_true")
    parser.add_argument("--phash-threshold", type=int, default=10)
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow an existing split manifest to cover a validated subset of metadata images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata_csv)

    if args.existing_split_csv is not None:
        print(f"[Split] Validating existing split: {args.existing_split_csv}", flush=True)
        split_df, excluded_metadata, validation_diagnostics = validate_existing_split(
            metadata,
            args.existing_split_csv,
            allow_subset=args.allow_subset,
        )
        if not excluded_metadata.empty:
            excluded_path = args.output_dir / f"{args.split_name}_excluded_metadata_images.csv"
            excluded_metadata.to_csv(excluded_path, index=False)
            print(
                f"[WARN] Existing split covers a subset of metadata. "
                f"Excluded metadata images: {len(excluded_metadata)} -> {excluded_path}",
                flush=True,
            )
        mode = "validated_existing_split"
    else:
        validation_diagnostics = {}
        if args.use_phash_components:
            print(f"[Split] Generating pHash-component-aware split, threshold={args.phash_threshold}", flush=True)
            metadata = add_phash_components(metadata, threshold=args.phash_threshold)
            split_df = generate_component_aware_split(
                metadata,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                component_col="phash_component",
            )
            mode = "generated_phash_component_split"
        else:
            print("[Split] Generating stratified image-level split.", flush=True)
            print("[WARN] pHash components were not requested; near-duplicate groups are not constrained.", flush=True)
            split_df = generate_stratified_split(
                metadata,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )
            mode = "generated_stratified_split"

    validate_split_completeness(split_df)
    split_path = args.output_dir / f"{args.split_name}.csv"
    split_df.to_csv(split_path, index=False)

    distribution = split_distribution(split_df)
    distribution.to_csv(args.output_dir / f"{args.split_name}_distribution.csv", index=False)

    summary = {
        "split_name": args.split_name,
        "mode": mode,
        "created_at_utc": timestamp_utc(),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "n_images": int(len(split_df)),
        "n_classes": int(split_df["class_name"].nunique()),
        "split_counts": {str(k): int(v) for k, v in split_df["split"].value_counts().sort_index().items()},
        "split_csv": str(split_path),
        "validation_diagnostics": validation_diagnostics,
    }
    save_json(args.output_dir / f"{args.split_name}_summary.json", summary)

    print("\n[Split] Distribution", flush=True)
    print(distribution.to_string(index=False), flush=True)
    print(f"[Split] Saved: {split_path}", flush=True)


if __name__ == "__main__":
    main()
