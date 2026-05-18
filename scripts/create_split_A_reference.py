#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group_split_for_class(df_class, seed=42):
    """
    Create a 70/15/15 group-disjoint split for one class.
    This keeps all images with the same group_id in the same partition.
    """
    if "group_id" not in df_class.columns:
        raise ValueError("metadata.csv must contain a 'group_id' column.")

    df_class = df_class.copy()
    groups = df_class["group_id"].astype(str).values

    # First split: train 70%, temp 30%
    gss1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=seed,
    )
    train_idx, temp_idx = next(gss1.split(df_class, groups=groups))

    df_train = df_class.iloc[train_idx].copy()
    df_temp = df_class.iloc[temp_idx].copy()

    # Second split: temp -> validation 15%, test 15%
    temp_groups = df_temp["group_id"].astype(str).values
    gss2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=seed + 1,
    )
    val_idx, test_idx = next(gss2.split(df_temp, groups=temp_groups))

    df_val = df_temp.iloc[val_idx].copy()
    df_test = df_temp.iloc[test_idx].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    return pd.concat([df_train, df_val, df_test], ignore_index=True)


def check_group_overlap(df):
    overlaps = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        groups_a = set(df.loc[df["split"] == a, "group_id"].astype(str))
        groups_b = set(df.loc[df["split"] == b, "group_id"].astype(str))
        common = groups_a.intersection(groups_b)
        if common:
            overlaps.append((a, b, len(common), sorted(list(common))[:10]))
    return overlaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_csv = Path(args.metadata_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)

    required = ["class_name", "class_index", "group_id"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    split_parts = []
    for class_name, df_class in df.groupby("class_name", sort=True):
        split_class = make_group_split_for_class(df_class, seed=args.seed)
        split_parts.append(split_class)

    split_df = pd.concat(split_parts, ignore_index=True)

    # Keep only core split columns if available
    preferred_cols = [
        "image_id",
        "relative_path",
        "image_path",
        "class_name",
        "class_index",
        "group_id",
        "split",
    ]
    existing_cols = [c for c in preferred_cols if c in split_df.columns]
    other_cols = [c for c in split_df.columns if c not in existing_cols]

    split_df = split_df[existing_cols + other_cols]

    overlaps = check_group_overlap(split_df)
    if overlaps:
        print("WARNING: group overlap detected:")
        for item in overlaps:
            print(item)
    else:
        print("No group_id overlap across train/val/test.")

    print("\nSplit counts:")
    print(split_df["split"].value_counts())

    print("\nClass distribution:")
    print(pd.crosstab(split_df["class_name"], split_df["split"]))

    split_df.to_csv(output_csv, index=False)
    print(f"\nSaved Split A reference split to: {output_csv}")


if __name__ == "__main__":
    main()