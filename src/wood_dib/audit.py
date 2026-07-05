"""Leakage-style technical audits for split manifests."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from .constants import SPLIT_NAMES
from .splitting import phash_distance
from .utils import save_json


def write_split_distribution_plot(distribution: pd.DataFrame, output_path: Path) -> None:
    """Save a readable horizontal grouped bar chart for class counts per split."""
    try:
        import matplotlib
    except ModuleNotFoundError:
        print("[WARN] matplotlib is not installed; skipping split_distribution.png.", flush=True)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = (
        distribution.pivot(index="class_name", columns="split", values="n_images")
        .reindex(columns=list(SPLIT_NAMES), fill_value=0)
        .fillna(0)
        .astype(int)
    )

    class_names = pivot.index.tolist()
    y = np.arange(len(class_names))
    bar_height = 0.24
    offsets = {"train": -bar_height, "val": 0.0, "test": bar_height}

    fig_height = max(5.5, 0.55 * len(class_names) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    for split in SPLIT_NAMES:
        ax.barh(y + offsets[split], pivot[split].values, height=bar_height, label=split)

    ax.set_yticks(y)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_ylabel("Species")
    ax.set_title("Split Distribution")
    ax.legend(title="Split", frameon=False)
    ax.grid(axis="x", alpha=0.3)

    max_count = int(pivot.to_numpy().max()) if pivot.size else 0
    ax.set_xlim(0, max_count * 1.12 if max_count else 1)
    for split in SPLIT_NAMES:
        for yi, value in zip(y + offsets[split], pivot[split].values):
            ax.text(value + max(1, max_count * 0.01), yi, str(int(value)), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def merge_metadata_split(metadata: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["image_id", "relative_path", "class_name", "class_index", "split"]
    optional = [
        c
        for c in [
            "group_id",
            "specimen_key",
            "light_condition",
            "parsed_group_id",
            "sha256",
            "phash",
            "original_filename",
            "raw_path",
        ]
        if c in metadata.columns
    ]
    meta = metadata[["image_id"] + [c for c in optional if c != "image_id"]].drop_duplicates("image_id")
    merged = split_df[keep].merge(meta, on="image_id", how="left", suffixes=("", "_metadata"))
    if merged["relative_path"].isna().any():
        raise ValueError("Split contains image_id values not present in metadata.")
    return merged


def class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["split", "class_name"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["split", "class_name"])
    )


def cross_split_overlap(frame: pd.DataFrame, column: str, value_label: str) -> pd.DataFrame:
    if column not in frame.columns or frame[column].fillna("").eq("").all():
        return pd.DataFrame(columns=["audit", "left_split", "right_split", value_label, "n_rows"])

    rows = []
    for left, right in combinations(SPLIT_NAMES, 2):
        left_df = frame[frame["split"] == left]
        right_df = frame[frame["split"] == right]
        left_values = set(left_df[column].dropna().astype(str)) - {""}
        right_values = set(right_df[column].dropna().astype(str)) - {""}
        for value in sorted(left_values & right_values):
            n_rows = int((frame[column].astype(str) == value).sum())
            rows.append(
                {
                    "audit": column,
                    "left_split": left,
                    "right_split": right,
                    value_label: value,
                    "n_rows": n_rows,
                }
            )
    return pd.DataFrame(rows)


def filename_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "original_filename" not in work.columns:
        work["original_filename"] = work["relative_path"].map(lambda x: Path(str(x)).name)
    work["normalized_filename"] = work["original_filename"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return cross_split_overlap(work, "normalized_filename", "filename")


def exact_hash_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    if "sha256" not in frame.columns or frame["sha256"].fillna("").eq("").all():
        return pd.DataFrame(
            columns=[
                "sha256",
                "image_id",
                "split",
                "class_name",
                "group_id",
                "relative_path",
            ]
        )

    rows = []
    group_column = "group_id" if "group_id" in frame.columns else "parsed_group_id"
    work = frame[frame["sha256"].fillna("").astype(str).ne("")].copy()
    for sha256, hash_df in work.groupby("sha256", sort=True):
        if hash_df["split"].nunique() <= 1:
            continue
        for _, row in hash_df.sort_values(["split", "class_name", "relative_path"]).iterrows():
            rows.append(
                {
                    "sha256": sha256,
                    "image_id": row["image_id"],
                    "split": row["split"],
                    "class_name": row["class_name"],
                    "group_id": row.get(group_column, ""),
                    "relative_path": row["relative_path"],
                }
            )
    return pd.DataFrame(rows)


def phash_overlap(frame: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    if "phash" not in frame.columns or frame["phash"].fillna("").eq("").all():
        return pd.DataFrame(
            columns=[
                "threshold",
                "query_image_id",
                "query_split",
                "query_class",
                "neighbor_image_id",
                "neighbor_split",
                "neighbor_class",
                "phash_distance",
            ]
        )

    max_threshold = max(thresholds)
    rows = []
    work = frame[frame["phash"].fillna("").astype(str).ne("")].copy()
    for left, right in combinations(SPLIT_NAMES, 2):
        left_df = work[work["split"] == left]
        right_df = work[work["split"] == right]
        for _, q in left_df.iterrows():
            for _, n in right_df.iterrows():
                try:
                    distance = phash_distance(str(q["phash"]), str(n["phash"]))
                except ValueError:
                    continue
                if distance <= max_threshold:
                    rows.append(
                        {
                            "threshold": min(t for t in thresholds if distance <= t),
                            "query_image_id": q["image_id"],
                            "query_split": q["split"],
                            "query_class": q["class_name"],
                            "query_relative_path": q["relative_path"],
                            "neighbor_image_id": n["image_id"],
                            "neighbor_split": n["split"],
                            "neighbor_class": n["class_name"],
                            "neighbor_relative_path": n["relative_path"],
                            "phash_distance": int(distance),
                        }
                    )
    return pd.DataFrame(rows).sort_values("phash_distance") if rows else pd.DataFrame()


def summarize_audit(
    distribution: pd.DataFrame,
    group_report: pd.DataFrame,
    hash_report: pd.DataFrame,
    filename_report: pd.DataFrame,
    phash_report: pd.DataFrame,
    thresholds: list[int],
) -> dict:
    phash_counts = {}
    if not phash_report.empty:
        for threshold in thresholds:
            phash_counts[str(threshold)] = int((phash_report["phash_distance"] <= threshold).sum())
    else:
        phash_counts = {str(threshold): 0 for threshold in thresholds}

    hash_findings = int(hash_report["sha256"].nunique()) if "sha256" in hash_report.columns else int(len(hash_report))

    return {
        "n_images_by_split": distribution.groupby("split")["n_images"].sum().to_dict(),
        "group_overlap_rows": int(len(group_report)),
        "sha256_overlap_rows": hash_findings,
        "filename_overlap_rows": int(len(filename_report)),
        "phash_candidate_pairs_by_threshold": phash_counts,
        "passed_basic_audits": bool(len(group_report) == 0 and len(hash_report) == 0 and len(filename_report) == 0),
        "passed_phash_audit": bool(all(v == 0 for v in phash_counts.values())),
    }


def write_audit_latex(path: Path, summary: dict) -> None:
    rows = [
        ("Group overlap", summary["group_overlap_rows"]),
        ("Exact SHA-256 overlap", summary["sha256_overlap_rows"]),
        ("Filename overlap", summary["filename_overlap_rows"]),
    ]
    for threshold, count in summary["phash_candidate_pairs_by_threshold"].items():
        rows.append((f"pHash candidates (d <= {threshold})", count))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Split-integrity audit summary for the IC4SD-Wood-Eucalyptus split manifest.}",
        "\\label{tab:split_audit}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Audit & Findings \\\\",
        "\\midrule",
    ]
    for name, count in rows:
        lines.append(f"{name} & {count} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def save_audit_outputs(frame: pd.DataFrame, output_dir: Path, thresholds: list[int]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution = class_distribution(frame)
    distribution.to_csv(output_dir / "split_distribution.csv", index=False)
    write_split_distribution_plot(distribution, output_dir / "split_distribution.png")

    group_column = "group_id" if "group_id" in frame.columns else "parsed_group_id"
    group_report = cross_split_overlap(frame, group_column, "group_id")
    group_report.to_csv(output_dir / "group_overlap_report.csv", index=False)

    hash_report = exact_hash_overlap(frame)
    hash_report.to_csv(output_dir / "hash_overlap_report.csv", index=False)

    filename_report = filename_overlap(frame)
    filename_report.to_csv(output_dir / "filename_overlap_report.csv", index=False)

    phash_report = phash_overlap(frame, thresholds)
    phash_report.to_csv(output_dir / "phash_overlap_report.csv", index=False)

    summary = summarize_audit(distribution, group_report, hash_report, filename_report, phash_report, thresholds)
    summary["group_column"] = group_column
    summary["group_id_available"] = bool(group_column in frame.columns and not frame[group_column].fillna("").eq("").all())
    summary["sha256_available"] = bool("sha256" in frame.columns and not frame["sha256"].fillna("").eq("").all())
    summary["phash_available"] = bool("phash" in frame.columns and not frame["phash"].fillna("").eq("").all())
    if not summary["sha256_available"]:
        summary["warnings"] = summary.get("warnings", []) + ["SHA-256 values are missing or empty; exact hash audit is not informative."]
    if not summary["group_id_available"]:
        summary["warnings"] = summary.get("warnings", []) + ["Group IDs are missing or empty; group-overlap audit is not informative."]
    if not summary["phash_available"]:
        summary["warnings"] = summary.get("warnings", []) + ["pHash values are missing or empty; pHash near-duplicate audit is not informative."]
    save_json(output_dir / "audit_summary.json", summary)
    write_audit_latex(output_dir / "audit_summary_latex_table.tex", summary)
    return summary
