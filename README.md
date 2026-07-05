# IC4SD-Wood-Eucalyptus Public Data in Brief Code

This package supports the **IC4SD-Wood-Eucalyptus** Data in Brief manuscript.
It provides clean, self-contained scripts to:

1. build a metadata manifest from raw image folders;
2. validate or generate train/validation/test split manifests;
3. audit split integrity with transparent leakage-style checks;
4. train and evaluate a DenseNet-121 technical-validation baseline.

The code is intentionally independent from the older private research pipeline in
the parent repository. All dataset paths are passed through command-line
arguments.

## Dataset, License, and Reporting

The source code in this repository is released under the MIT License. The
dataset is released separately under the CC-BY-NC-4.0 data license.

Dataset DOI: [10.5281/zenodo.21188771](https://doi.org/10.5281/zenodo.21188771)

Recommended reporting checklist:

- split used (A/B)
- additional filtering
- random seed
- evaluation metric

## Dataset Structure

The expected raw dataset layout is:

```text
raw/
  Eucalyptus_camaldulensis/
    specimen_1/
      image1.jpg
      image2.jpg
  Eucalyptus_cladocalyx/
    ...
  Syzygium_hemisphericum/
    ...
```

Each top-level folder is treated as one class. For the revised dataset, each
first-level subfolder below a species folder is treated as one physical wood
specimen. The public metadata therefore uses:

```text
group_id = specimen subfolder name
```

All images with the same `group_id` must stay in the same train/validation/test
partition. Images are still scanned recursively so the scripts remain robust to
additional nesting.

The scripts normalize common folder-name variants such as:

- `Eucalyptus_camaldulensis`
- `E_camaldulensis`
- `E. camaldulensis`
- `Syzygium_hemisphericum`
- `S. hemisphericum`

Historical spelling variants in the working dataset, such as
`camandulensis` and `daglupta`, are normalized to the scientific class names
`Eucalyptus camaldulensis` and `Eucalyptus deglupta`.

## Install

Create an environment with Python 3.10+ and install a PyTorch build that matches
your CUDA driver. Then install the remaining dependencies:

```bash
python -m pip install -r public_dib_code/requirements.txt
```

If you do not need perceptual-hash support, `imagehash` can be omitted, but the
pHash-related split/audit features will be skipped or unavailable.

## Step 1: Build Metadata

Example using the local working raw path:

```bash
python public_dib_code/scripts/build_manifest.py \
  --raw-root /Users/ntkhanh/Work/PTIT/1.Cá nhân/2.Research/2026/Article/DIB/raw_update \
  --output-dir public_dib_code/outputs/metadata \
  --compute-phash
```

Outputs:

- `metadata.csv`
- `label_map.json`
- `manifest_summary.json`
- `skipped_files.csv`
- `unreadable_images.csv`
- `class_normalization_notes.json`, when folder names were normalized

The metadata CSV includes:

- `image_id`
- `relative_path`
- `raw_path`
- `original_filename`
- `top_level_folder`
- `class_name`
- `class_index`
- `width`
- `height`
- `file_extension`
- `sha256`
- `phash`
- `group_id`
- `specimen_key`
- `light_condition`
- `parsed_group_id`
- `legacy_parsed_group_id`
- `notes`

## Step 2: Prepare or Validate Split Manifests

The revised workflow supports two predefined manifests:

- `split_A_reference.csv`: specimen-group-disjoint reference split.
- `split_B_strict.csv`: specimen-group-disjoint and pHash-component-clean split.

Both are regenerated from the revised `raw_update/` tree and the new
folder-based `group_id` values.

### Validate an existing strict Split B manifest

```bash
python public_dib_code/scripts/prepare_splits.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --existing-split-csv path/to/split_B_strict.csv \
  --output-dir public_dib_code/outputs/splits \
  --split-name split_B_strict
```

The script validates that all split entries are present in the metadata, that
class labels match, and that `train`, `val`, and `test` are all present.

### Generate a reproducible split when no strict split is provided

```bash
python public_dib_code/scripts/prepare_splits.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --output-dir public_dib_code/outputs/splits \
  --split-name split_B_strict \
  --seed 42
```

This now generates a deterministic specimen group-aware split by default. Use
`--no-group-aware` only for a deliberately image-level diagnostic split.

### Generate Split B: pHash-clean plus specimen-group-disjoint

This requires `metadata.csv` to have non-empty `phash` values, which are created
by running `build_manifest.py --compute-phash`.

```bash
python public_dib_code/scripts/prepare_splits.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --output-dir public_dib_code/outputs/splits \
  --split-name split_B_strict \
  --seed 42 \
  --use-phash-components \
  --phash-threshold 10
```

When `--use-phash-components` is used, the script combines pHash components with
the specimen `group_id` constraints. This prevents a physical specimen from
being split across partitions even when it contains multiple pHash components.

### Generate Split A: specimen-group-disjoint reference split

```bash
python public_dib_code/scripts/prepare_splits.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --output-dir public_dib_code/outputs/splits \
  --split-name split_A_reference \
  --seed 42
```

Output examples:

- `split_B_strict.csv`
- `split_B_strict_distribution.csv`
- `split_B_strict_summary.json`

## Step 3: Audit Split Integrity

```bash
python public_dib_code/scripts/audit_splits.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --split-csv public_dib_code/outputs/splits/split_B_strict.csv \
  --output-dir public_dib_code/outputs/audit
```

Outputs:

- `split_distribution.csv`
- `hash_overlap_report.csv`
- `filename_overlap_report.csv`
- `group_overlap_report.csv`
- `phash_overlap_report.csv`
- `audit_summary.json`
- `audit_summary_latex_table.tex`

These checks are technical audits. Passing them should be described carefully as
evidence against obvious group overlap, exact duplicate, filename, and pHash
near-duplicate leakage under the performed checks.

Run the audit separately for both Split A and Split B:

```bash
python public_dib_code/scripts/audit_splits.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --split-csv public_dib_code/outputs/splits/split_A_reference.csv \
  --output-dir public_dib_code/outputs/audit_split_A

python public_dib_code/scripts/audit_splits.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --split-csv public_dib_code/outputs/splits/split_B_strict.csv \
  --output-dir public_dib_code/outputs/audit_split_B
```

## Step 4: Generate Revision Summary Tables

The following command creates reviewer-facing CSV and LaTeX tables for specimen
counts, source-institution breakdown, resolution distribution, pHash component
sizes, file extensions, and lighting summaries.

```bash
python public_dib_code/scripts/build_revision_tables.py \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --annotation-xlsx /Users/ntkhanh/Work/PTIT/1.Cá nhân/2.Research/2026/Article/DIB/Meta-data_update.xlsx \
  --output-dir public_dib_code/outputs/revision_tables \
  --phash-threshold 10
```

## Step 5: Train DenseNet-121 Baseline

Do not run this on a laptop unless you intend to train. On a GPU server:

```bash
python public_dib_code/scripts/train_classifier.py \
  --model densenet121 \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --split-csv public_dib_code/outputs/splits/split_B_strict.csv \
  --raw-root /Users/ntkhanh/Work/PTIT/1.Cá nhân/2.Research/2026/Article/DIB/raw_update \
  --output-dir public_dib_code/outputs/densenet121_splitB_seed42 \
  --seed 42 \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --input-size 224 \
  --amp
```

Nohup example:

```bash
nohup python public_dib_code/scripts/train_classifier.py \
  --model densenet121 \
  --metadata-csv public_dib_code/outputs/metadata/metadata.csv \
  --split-csv public_dib_code/outputs/splits/split_B_strict.csv \
  --raw-root /path/to/raw \
  --output-dir public_dib_code/outputs/densenet121_splitB_seed42 \
  --seed 42 \
  --epochs 50 \
  --batch-size 64 \
  --input-size 224 \
  --amp \
  > public_dib_code/outputs/densenet121_splitB_seed42.log 2>&1 &
```

Training outputs:

- `best_model.pt`
- `last_model.pt`
- `test_metrics.json`
- `test_metrics.csv`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `training_history.csv`
- `test_predictions.csv`
- `densenet121_training_curves.png`
- `densenet121_confusion_matrix.png`
- `densenet121_latex_table.tex`
- `config.json`
- `label_map.json`
- `split_manifest_used.csv`
- `run_summary.json`

## Notes for Public Release

- Update paths in the example commands for the release environment.
- Replace placeholder text in manuscripts with the final dataset DOI and code
  repository URL.
- Keep the raw dataset folder names traceable; the metadata preserves
  `top_level_folder` and `relative_path`.
- The default class mapping is saved to `label_map.json` and uses the canonical
  scientific names listed in the Data in Brief manuscript.
