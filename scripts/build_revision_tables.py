#!/usr/bin/env python3
"""Build reviewer-facing summary tables from revised metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from wood_dib.splitting import add_phash_components, load_metadata
from wood_dib.utils import infer_specimen_key, normalize_class_name, save_json, timestamp_utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create revision summary tables for IC4SD-Wood-Eucalyptus.")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--annotation-xlsx", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phash-threshold", type=int, default=10)
    return parser.parse_args()


def normalize_sample_id(value: object) -> str:
    text = str(value).strip().replace(",", ".")
    return infer_specimen_key(text)


def joined_unique(values: pd.Series) -> str:
    clean = [str(v).strip() for v in values.dropna().tolist() if str(v).strip()]
    return "; ".join(sorted(set(clean)))


def load_annotation(annotation_xlsx: Path) -> pd.DataFrame:
    if not annotation_xlsx.exists():
        raise FileNotFoundError(f"Missing annotation workbook: {annotation_xlsx}")
    df = pd.read_excel(annotation_xlsx, header=1)
    df = df.rename(
        columns={
            "Science name": "class_name_raw",
            "Sample ID": "sample_id",
            "Originin": "source_institution",
            "Humidity state": "humidity_state",
            "Collectors": "collectors",
            "Captured place": "captured_place",
            "Light conditions": "annotation_light_condition",
            "Images Refs": "image_refs",
            "Unnamed: 18": "qc_notes",
        }
    )
    df["class_name_raw"] = df["class_name_raw"].ffill()
    df = df[df["sample_id"].notna()].copy()
    df["class_name"] = df["class_name_raw"].map(lambda x: normalize_class_name(str(x))[0])
    df["specimen_key"] = df["sample_id"].map(normalize_sample_id)

    grouped = (
        df.groupby(["class_name", "specimen_key"], as_index=False)
        .agg(
            source_institution=("source_institution", joined_unique),
            humidity_state=("humidity_state", joined_unique),
            collectors=("collectors", joined_unique),
            captured_place=("captured_place", joined_unique),
            annotation_light_condition=("annotation_light_condition", joined_unique),
            image_refs=("image_refs", joined_unique),
            qc_notes=("qc_notes", joined_unique),
            annotation_rows=("sample_id", "size"),
        )
        .sort_values(["class_name", "specimen_key"])
    )
    return grouped


def write_latex_table(path: Path, df: pd.DataFrame, caption: str, label: str, max_rows: int = 80) -> None:
    table_df = df.head(max_rows).copy()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * len(table_df.columns) + "}",
        "\\toprule",
        " & ".join(str(c).replace("_", " ") for c in table_df.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in table_df.iterrows():
        values = [str(row[col]).replace("_", "\\_") for col in table_df.columns]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_csv)
    if "group_id" not in metadata.columns:
        raise ValueError("metadata.csv must contain authoritative group_id values.")
    metadata["group_id"] = metadata["group_id"].astype(str)
    metadata["specimen_key"] = metadata["group_id"].map(infer_specimen_key)

    specimen_counts = (
        metadata.groupby("class_name")
        .agg(specimens=("group_id", "nunique"), images=("image_id", "count"))
        .reset_index()
        .sort_values("class_name")
    )
    specimen_counts.to_csv(args.output_dir / "per_class_specimen_image_counts.csv", index=False)
    write_latex_table(
        args.output_dir / "per_class_specimen_image_counts.tex",
        specimen_counts,
        "Per-class specimen and image counts in the revised IC4SD-Wood-Eucalyptus dataset.",
        "tab:per_class_specimen_counts",
    )

    resolution_distribution = (
        metadata.groupby(["width", "height"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["n_images", "width", "height"], ascending=[False, True, True])
    )
    resolution_distribution.to_csv(args.output_dir / "resolution_distribution.csv", index=False)

    resolution_by_class = (
        metadata.groupby(["class_name", "width", "height"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["class_name", "n_images"], ascending=[True, False])
    )
    resolution_by_class.to_csv(args.output_dir / "resolution_distribution_by_class.csv", index=False)

    extension_by_class = (
        metadata.groupby(["class_name", "file_extension"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["class_name", "file_extension"])
    )
    extension_by_class.to_csv(args.output_dir / "file_extension_by_class.csv", index=False)

    light_by_class = (
        metadata.groupby(["class_name", "light_condition"])
        .agg(specimens=("group_id", "nunique"), images=("image_id", "count"))
        .reset_index()
        .sort_values(["class_name", "light_condition"])
    )
    light_by_class.to_csv(args.output_dir / "light_condition_by_class.csv", index=False)

    if "phash" in metadata.columns and not metadata["phash"].fillna("").eq("").all():
        with_components = add_phash_components(metadata, threshold=args.phash_threshold)
        component_sizes = (
            with_components.groupby(["class_name", "phash_component"])
            .size()
            .reset_index(name="component_size")
        )
        phash_summary = (
            component_sizes.groupby("class_name")
            .agg(
                phash_components=("phash_component", "nunique"),
                images=("component_size", "sum"),
                min_component_size=("component_size", "min"),
                median_component_size=("component_size", "median"),
                mean_component_size=("component_size", "mean"),
                max_component_size=("component_size", "max"),
            )
            .reset_index()
            .sort_values("class_name")
        )
        phash_summary.to_csv(args.output_dir / "per_class_phash_component_summary.csv", index=False)
        component_sizes.to_csv(args.output_dir / "phash_component_sizes.csv", index=False)
    else:
        pd.DataFrame(
            [
                {
                    "warning": "metadata.csv does not contain pHash values; rerun build_manifest.py with --compute-phash first.",
                }
            ]
        ).to_csv(args.output_dir / "per_class_phash_component_summary.csv", index=False)

    if args.annotation_xlsx is not None:
        annotation = load_annotation(args.annotation_xlsx)
        annotation.to_csv(args.output_dir / "annotation_specimen_metadata.csv", index=False)
        annotated = metadata.merge(annotation, on=["class_name", "specimen_key"], how="left")
        missing_annotation = annotated[annotated["source_institution"].isna()][
            ["class_name", "group_id", "specimen_key"]
        ].drop_duplicates()
        missing_annotation.to_csv(args.output_dir / "metadata_without_annotation_match.csv", index=False)

        source_breakdown = (
            annotated.groupby(["class_name", "source_institution"], dropna=False)
            .agg(specimens=("group_id", "nunique"), images=("image_id", "count"))
            .reset_index()
            .sort_values(["class_name", "source_institution"])
        )
        source_breakdown["source_institution"] = source_breakdown["source_institution"].fillna("unmatched")
        source_breakdown.to_csv(args.output_dir / "source_institution_breakdown.csv", index=False)
        write_latex_table(
            args.output_dir / "source_institution_breakdown.tex",
            source_breakdown,
            "Per-class source-institution breakdown for the revised dataset.",
            "tab:source_institution_breakdown",
        )

    exclusion_template = pd.DataFrame(
        [
            {
                "change_id": "QC_REIMAGE_ROUND_2",
                "class_name": "",
                "specimen_or_folder": "",
                "n_images": "",
                "action": "recollected",
                "reason": "quality or metadata mismatch identified during manual QC; fill with final retained/rejected file list if available",
            },
            {
                "change_id": "PTIT_RIFI_REMOVAL",
                "class_name": "Eucalyptus grandis",
                "specimen_or_folder": "PTIT_RIFI",
                "n_images": "",
                "action": "removed",
                "reason": "folder visually inconsistent with the retained class images; exact file list requires manual QC log",
            },
        ]
    )
    exclusion_template.to_csv(args.output_dir / "exclusion_log_template.csv", index=False)

    summary = {
        "created_at_utc": timestamp_utc(),
        "metadata_csv": str(args.metadata_csv),
        "annotation_xlsx": str(args.annotation_xlsx) if args.annotation_xlsx else None,
        "n_images": int(len(metadata)),
        "n_classes": int(metadata["class_name"].nunique()),
        "n_specimens": int(metadata["group_id"].nunique()),
        "outputs": sorted(p.name for p in args.output_dir.glob("*")),
    }
    save_json(args.output_dir / "revision_tables_summary.json", summary)
    print(f"[Revision tables] Saved outputs to: {args.output_dir}", flush=True)
    print(specimen_counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
