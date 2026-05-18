#!/usr/bin/env python3
"""Run transparent split-integrity audits for the public split manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from wood_dib.audit import merge_metadata_split, save_audit_outputs
from wood_dib.splitting import load_metadata, validate_split_completeness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an IC4SD-Wood-Eucalyptus split manifest.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phash-thresholds", type=int, nargs="+", default=[5, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.split_csv.exists():
        raise FileNotFoundError(f"Missing split CSV: {args.split_csv}")

    metadata = load_metadata(args.metadata_csv)
    split_df = pd.read_csv(args.split_csv)
    validate_split_completeness(split_df)
    merged = merge_metadata_split(metadata, split_df)

    summary = save_audit_outputs(merged, args.output_dir, thresholds=args.phash_thresholds)
    print("[Audit] Summary", flush=True)
    for key, value in summary.items():
        print(f"  {key}: {value}", flush=True)
    print(f"[Audit] Outputs saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
