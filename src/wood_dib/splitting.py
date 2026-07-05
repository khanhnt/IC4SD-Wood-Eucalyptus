"""Split-manifest validation and deterministic split generation."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    train_test_split = None

from .constants import SPLIT_NAMES
from .utils import normalize_class_name


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    required = {"image_id", "relative_path", "class_name", "class_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metadata_csv} is missing required columns: {sorted(missing)}")
    return df


def normalize_split_values(series: pd.Series) -> pd.Series:
    mapping = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "testing": "test",
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        bad = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unknown split value(s): {bad}. Expected train/val/test.")
    return normalized


def _path_key(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda x: Path(x).as_posix().lstrip("./"))


def _split_class_key(split_df: pd.DataFrame, path_col: str | None) -> pd.Series:
    if "class_name" in split_df.columns:
        return split_df["class_name"].map(lambda x: normalize_class_name(str(x))[0])
    if path_col is None:
        raise ValueError("Split CSV needs class_name when filename fallback is required.")

    def infer_from_path(value: str) -> str:
        parts = Path(str(value)).parts
        for idx, part in enumerate(parts):
            if part.lower() in set(SPLIT_NAMES) and idx + 1 < len(parts):
                return normalize_class_name(parts[idx + 1])[0]
        if len(parts) >= 2:
            return normalize_class_name(parts[-2])[0]
        raise ValueError(f"Cannot infer class from split path: {value}")

    return split_df[path_col].map(infer_from_path)


def _filename_fallback_merge(
    metadata: pd.DataFrame,
    split_df: pd.DataFrame,
    path_col: str,
    allow_metadata_subset: bool = False,
) -> tuple[pd.DataFrame | None, dict]:
    fallback_split = split_df.copy()
    fallback_split["_split_row_id"] = range(len(fallback_split))
    fallback_split["filename_key"] = fallback_split[path_col].astype(str).map(lambda x: Path(x).name)
    fallback_split["class_key"] = _split_class_key(fallback_split, path_col)

    fallback_meta = metadata.copy()
    fallback_meta["filename_key"] = (
        fallback_meta["original_filename"].astype(str)
        if "original_filename" in fallback_meta.columns
        else fallback_meta["relative_path"].astype(str).map(lambda x: Path(x).name)
    )
    fallback_meta["class_key"] = fallback_meta["class_name"].astype(str)

    split_dupes = fallback_split[fallback_split.duplicated(["class_key", "filename_key"], keep=False)]
    meta_dupes = fallback_meta[fallback_meta.duplicated(["class_key", "filename_key"], keep=False)]
    diagnostics = {
        "fallback_key": f"class_name+filename from {path_col}",
        "split_duplicate_keys": int(len(split_dupes)),
        "metadata_duplicate_keys": int(len(meta_dupes)),
        "split_rows": int(len(fallback_split)),
        "metadata_rows": int(len(fallback_meta)),
    }
    if not split_dupes.empty or not meta_dupes.empty:
        diagnostics["split_duplicate_examples"] = split_dupes[["class_key", "filename_key"]].head(10).to_dict("records")
        diagnostics["metadata_duplicate_examples"] = meta_dupes[["class_key", "filename_key", "relative_path"]].head(10).to_dict("records")
        return None, diagnostics

    split_cols = ["class_key", "filename_key", "split", "_split_row_id"] + [
        c for c in ["class_name", "class_index", "group_id", "component_id"] if c in fallback_split.columns
    ]
    merged = fallback_meta.merge(
        fallback_split[split_cols],
        on=["class_key", "filename_key"],
        how="outer",
        suffixes=("", "_split"),
        indicator=True,
    )
    metadata_unmatched = merged[merged["_merge"] == "left_only"]
    split_unmatched = merged[merged["_merge"] == "right_only"]
    diagnostics["metadata_unmatched"] = int(len(metadata_unmatched))
    diagnostics["split_unmatched"] = int(len(split_unmatched))
    diagnostics["metadata_unmatched_examples"] = metadata_unmatched.get("relative_path", pd.Series(dtype=str)).head(10).tolist()
    diagnostics["split_unmatched_examples"] = split_unmatched[["class_key", "filename_key"]].head(10).to_dict("records")

    if not split_unmatched.empty:
        return None, diagnostics

    if not metadata_unmatched.empty and not allow_metadata_subset:
        return None, diagnostics

    if allow_metadata_subset:
        merged = merged[merged["_merge"] == "both"].copy()
    merged = merged.drop(columns=["_merge", "_split_row_id"])
    return merged, diagnostics


def validate_existing_split(
    metadata: pd.DataFrame,
    existing_split_csv: Path,
    allow_subset: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not existing_split_csv.exists():
        raise FileNotFoundError(f"Missing existing split CSV: {existing_split_csv}")
    split_df = pd.read_csv(existing_split_csv)
    if "split" not in split_df.columns:
        raise ValueError(f"{existing_split_csv} must contain a 'split' column.")
    split_df = split_df.copy()
    split_df["split"] = normalize_split_values(split_df["split"])

    metadata_keys = metadata.copy()
    metadata_keys["relative_key"] = _path_key(metadata_keys["relative_path"])
    metadata_keys["raw_key"] = _path_key(metadata_keys["raw_path"]) if "raw_path" in metadata_keys.columns else ""

    source_path_col = None
    if "image_id" in split_df.columns:
        merged = split_df[["image_id", "split"] + [c for c in ["class_name", "class_index"] if c in split_df.columns]].merge(
            metadata, on="image_id", how="right", suffixes=("_split", "")
        )
        key_name = "image_id"
    elif "relative_path" in split_df.columns:
        source_path_col = "relative_path"
        split_df["relative_key"] = _path_key(split_df["relative_path"])
        merged = split_df[["relative_key", "split"] + [c for c in ["class_name", "class_index"] if c in split_df.columns]].merge(
            metadata_keys, on="relative_key", how="right", suffixes=("_split", "")
        )
        key_name = "relative_path"
    else:
        path_col = next((c for c in ["raw_path", "absolute_path", "image_path", "path", "source_path"] if c in split_df.columns), None)
        if path_col is None:
            raise ValueError(
                "Existing split CSV must contain one of: image_id, relative_path, raw_path, absolute_path, image_path, path."
            )
        source_path_col = path_col
        split_df["raw_key"] = _path_key(split_df[path_col])
        merged = split_df[["raw_key", "split"] + [c for c in ["class_name", "class_index"] if c in split_df.columns]].merge(
            metadata_keys, on="raw_key", how="right", suffixes=("_split", "")
        )
        key_name = path_col

    diagnostics: dict = {
        "match_key": key_name,
        "allow_subset": allow_subset,
        "metadata_rows": int(len(metadata)),
        "split_rows": int(len(split_df)),
    }
    excluded_metadata = pd.DataFrame()

    if merged["split"].isna().any() and source_path_col is not None:
        fallback_merged, diagnostics = _filename_fallback_merge(
            metadata,
            split_df,
            source_path_col,
            allow_metadata_subset=allow_subset,
        )
        if fallback_merged is not None:
            merged = fallback_merged
            key_name = diagnostics["fallback_key"]
        else:
            missing = merged.loc[merged["split"].isna(), "relative_path"].head(20).tolist()
            raise ValueError(
                f"Existing split does not cover metadata using exact key '{key_name}', and filename fallback failed.\n"
                f"Exact-key metadata examples missing from split: {missing}\n"
                f"Fallback diagnostics: {diagnostics}"
            )

    if merged["split"].isna().any() and allow_subset:
        excluded_metadata = merged[merged["split"].isna()].copy()
        merged = merged[merged["split"].notna()].copy()
    elif merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), "relative_path"].head(20).tolist()
        raise ValueError(
            f"Existing split does not cover all metadata images using key '{key_name}'. "
            f"Examples missing from split: {missing}. If this is expected, rerun with --allow-subset."
        )

    diagnostics["match_key"] = key_name
    diagnostics["matched_rows"] = int(len(merged))
    diagnostics["excluded_metadata_rows"] = int(len(excluded_metadata))

    if key_name in split_df.columns:
        duplicate_subset = [key_name]
    elif "relative_key" in split_df.columns:
        duplicate_subset = ["relative_key"]
    elif "raw_key" in split_df.columns:
        duplicate_subset = ["raw_key"]
    else:
        duplicate_subset = []
    duplicate_count = int(split_df.duplicated(subset=duplicate_subset).sum()) if duplicate_subset else 0
    if duplicate_count:
        raise ValueError(f"Existing split contains {duplicate_count} duplicate split keys.")

    if "class_name_split" in merged.columns:
        normalized_split_classes = merged["class_name_split"].map(lambda x: normalize_class_name(str(x))[0])
        mismatched = normalized_split_classes != merged["class_name"]
        if mismatched.any():
            examples = merged.loc[mismatched, ["relative_path", "class_name_split", "class_name"]].head(20)
            raise ValueError(f"Class-name mismatch between split and metadata:\n{examples.to_string(index=False)}")

    missing_files = []
    if "raw_path" in merged.columns:
        missing_files = [p for p in merged["raw_path"].astype(str).tolist() if not Path(p).exists()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} metadata image paths are missing. Example: {missing_files[0]}")

    out_cols = ["image_id", "relative_path", "class_name", "class_index", "split"]
    optional = [
        c
        for c in ["group_id", "specimen_key", "light_condition", "parsed_group_id", "sha256", "phash"]
        if c in merged.columns
    ]
    split_out = merged[out_cols + optional].sort_values(["split", "class_index", "relative_path"]).reset_index(drop=True)
    return split_out, excluded_metadata.reset_index(drop=True), diagnostics


def generate_stratified_split(
    metadata: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    ratios = train_ratio + val_ratio + test_ratio
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios:.6f}")
    if train_test_split is None:
        raise ModuleNotFoundError(
            "scikit-learn is required for image-level stratified splitting. "
            "Install scikit-learn or generate a group-aware split without --no-group-aware."
        )

    indices = metadata.index.to_numpy()
    labels = metadata["class_index"].to_numpy()
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = metadata.loc[temp_idx, "class_index"].to_numpy()
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction_of_temp,
        random_state=seed,
        stratify=temp_labels,
    )

    split = pd.Series(index=metadata.index, dtype="object")
    split.loc[train_idx] = "train"
    split.loc[val_idx] = "val"
    split.loc[test_idx] = "test"

    out = metadata.copy()
    out["split"] = split
    return split_manifest_columns(out)


class UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def phash_distance(left: str, right: str) -> int:
    return (int(str(left), 16) ^ int(str(right), 16)).bit_count()


def add_phash_components(metadata: pd.DataFrame, threshold: int) -> pd.DataFrame:
    if "phash" not in metadata.columns or metadata["phash"].fillna("").eq("").all():
        raise ValueError("Cannot use pHash components because metadata has no non-empty phash column.")

    out = metadata.copy()
    out["phash_component"] = ""
    for class_name, class_df in out.groupby("class_name", sort=True):
        indices = class_df.index.tolist()
        uf = UnionFind(indices)
        rows = class_df[["phash"]].fillna("").astype(str)
        valid = [(idx, phash) for idx, phash in zip(rows.index.tolist(), rows["phash"].tolist()) if phash]
        for i, (idx_i, hash_i) in enumerate(valid):
            for idx_j, hash_j in valid[i + 1 :]:
                try:
                    if phash_distance(hash_i, hash_j) <= threshold:
                        uf.union(idx_i, idx_j)
                except ValueError:
                    continue
        root_to_component: dict[int, str] = {}
        for idx in indices:
            root = uf.find(idx)
            if root not in root_to_component:
                root_to_component[root] = f"{class_name}::phash_component_{len(root_to_component) + 1:04d}"
            out.loc[idx, "phash_component"] = root_to_component[root]
    return out


def add_constraint_components(
    metadata: pd.DataFrame,
    group_col: str = "group_id",
    phash_component_col: str = "phash_component",
    output_col: str = "constraint_component",
) -> pd.DataFrame:
    """Union specimen groups and pHash components into final split units.

    A valid strict split must keep every physical specimen in one partition,
    while also keeping pHash-near-duplicate components in one partition. This
    function builds connected components over both constraints within each
    class.
    """
    if group_col not in metadata.columns:
        raise ValueError(f"Cannot build constraint components; missing column: {group_col}")
    if phash_component_col not in metadata.columns:
        raise ValueError(f"Cannot build constraint components; missing column: {phash_component_col}")

    out = metadata.copy()
    out[output_col] = ""
    for class_name, class_df in out.groupby("class_name", sort=True):
        indices = class_df.index.tolist()
        uf = UnionFind(indices)
        for _, group_df in class_df.groupby(group_col, sort=True):
            group_indices = group_df.index.tolist()
            for idx in group_indices[1:]:
                uf.union(group_indices[0], idx)
        for _, component_df in class_df.groupby(phash_component_col, sort=True):
            component_indices = component_df.index.tolist()
            for idx in component_indices[1:]:
                uf.union(component_indices[0], idx)

        root_to_component: dict[int, str] = {}
        for idx in indices:
            root = uf.find(idx)
            if root not in root_to_component:
                root_to_component[root] = f"{class_name}::constraint_component_{len(root_to_component) + 1:04d}"
            out.loc[idx, output_col] = root_to_component[root]
    return out


def generate_component_aware_split(
    metadata: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    component_col: str,
) -> pd.DataFrame:
    rng = random.Random(seed)
    output_rows = []

    for _, class_df in metadata.groupby("class_name", sort=True):
        components = []
        for component_id, comp_df in class_df.groupby(component_col, sort=True):
            components.append((str(component_id), comp_df.index.tolist(), len(comp_df)))
        if len(components) < len(SPLIT_NAMES):
            raise ValueError(
                f"Class '{class_df['class_name'].iloc[0]}' has only {len(components)} split components; "
                f"at least {len(SPLIT_NAMES)} are required for train/val/test coverage."
            )
        rng.shuffle(components)
        components.sort(key=lambda item: item[2], reverse=True)

        total = len(class_df)
        targets = {"train": total * train_ratio, "val": total * val_ratio, "test": total * test_ratio}
        counts = {"train": 0, "val": 0, "test": 0}
        assignments: dict[str, str] = {}

        # Seed each split with one component when possible so every class appears in all splits.
        initial_splits = list(SPLIT_NAMES)
        for split_name, component in zip(initial_splits, components):
            component_id, _, size = component
            assignments[component_id] = split_name
            counts[split_name] += size

        for component_id, _, size in components[len(assignments) :]:
            split_name = max(SPLIT_NAMES, key=lambda name: targets[name] - counts[name])
            assignments[component_id] = split_name
            counts[split_name] += size

        class_out = class_df.copy()
        class_out["split"] = class_out[component_col].map(assignments)
        output_rows.append(class_out)

    out = pd.concat(output_rows, ignore_index=True)
    return split_manifest_columns(out)


def split_manifest_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["image_id", "relative_path", "class_name", "class_index", "split"]
    optional = [
        c
        for c in [
            "group_id",
            "specimen_key",
            "light_condition",
            "parsed_group_id",
            "phash_component",
            "constraint_component",
            "sha256",
            "phash",
        ]
        if c in frame.columns
    ]
    return frame[cols + optional].sort_values(["split", "class_index", "relative_path"]).reset_index(drop=True)


def split_distribution(split_df: pd.DataFrame) -> pd.DataFrame:
    return (
        split_df.groupby(["split", "class_name"])
        .size()
        .reset_index(name="n_images")
        .sort_values(["split", "class_name"])
    )


def validate_split_completeness(split_df: pd.DataFrame) -> None:
    present = set(split_df["split"].unique())
    missing = set(SPLIT_NAMES) - present
    if missing:
        raise ValueError(f"Split manifest is missing split(s): {sorted(missing)}")
    if split_df["image_id"].duplicated().any():
        dupes = split_df.loc[split_df["image_id"].duplicated(), "image_id"].head(10).tolist()
        raise ValueError(f"Split manifest contains duplicated image_id values: {dupes}")
