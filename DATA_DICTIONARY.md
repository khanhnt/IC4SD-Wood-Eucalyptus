# Data Dictionary — `metadata.csv`

**Dataset:** IC4SD-Wood-Eucalyptus (macroscopic transverse-section wood images)
**File:** `metadata.csv` · UTF-8 · comma-separated · **2,910 rows** (one row per image) · **21 columns**

Each row describes one image. Below, "distinct" and value counts are computed over
the full released set (2,910 images). Categorical value sets are exhaustive.

---

## 1. Column summary

| # | Column | Type | Example | Description |
|---|--------|------|---------|-------------|
| 1 | `image_id` | string | `IC4SD_EUC_000001` | Globally unique, stable image identifier. Primary key. |
| 2 | `image_path` | string | `Eucalyptus_camaldulensis/3358. …1/3358. …1.(1).png` | Path to the image within the released image tree (class / specimen subfolders and file name). |
| 3 | `class_name` | string | `Eucalyptus camaldulensis` | Species label in canonical (corrected) spelling. One of eight classes (§2.1). |
| 4 | `class_index` | integer | `0` | Integer class label 0–7, matching `label_map.json` (§2.1). |
| 5 | `split` | string | `train` | Partition in the released **strict pHash-clean split (Split B)**: `train`, `val`, or `test` (§2.2). |
| 6 | `group_id` | string | `3358. Eucalyptus camandulensis.1` | **Authoritative physical-specimen identifier** — one value per specimen subfolder (§3). |
| 7 | `specimen_key` | string | `3358.1` | Short specimen key. **Ambiguous — do not use for grouping** (§3). |
| 8 | `light_condition` | string | `natural` | Illumination: `natural`, `indoor`, or `outdoor` (§2.3). |
| 9 | `parsed_group_id` | string | `3358. Eucalyptus camandulensis.1` | Parsed specimen id. **Identical to `group_id`** for every row; retained for provenance. |
| 10 | `phash_component` | string | `Eucalyptus camaldulensis::phash_component_0001` | Perceptual-hash near-duplicate component id (§3). |
| 11 | `constraint_component` | string | `Eucalyptus camaldulensis::constraint_component_0001` | **Grouping key used to build Split B** (specimen unioned with perceptual component) (§3). |
| 12 | `sha256` | string | `8c4e56a1…d21e` | SHA-256 hash of the image bytes (exact-duplicate detection). 64 hex chars. |
| 13 | `phash` | string | `aba184f29e32c9ba` | 64-bit perceptual hash, 16 hex chars (near-duplicate detection). |
| 14 | `raw_path` | string | `raw/Eucalyptus_camaldulensis/…png` | Path to the file within the original `raw/` source tree (provenance). |
| 15 | `original_filename` | string | `3358. Eucalyptus camandulensis.1.(1).png` | File name as originally captured (see note in §4). |
| 16 | `top_level_folder` | string | `Eucalyptus camandulensis` | Original top-level acquisition folder (see note in §4). |
| 17 | `width` | integer | `2560` | Image width in pixels (§2.4). |
| 18 | `height` | integer | `2048` | Image height in pixels (§2.4). |
| 19 | `file_extension` | string | `.png` | Image file extension. All images are `.png`. |
| 20 | `legacy_parsed_group_id` | string | `Eucalyptus camaldulensis::3358. …1` | Pre-correction grouping label (class :: folder); retained to document the specimen-grouping correction (§3, §4). |
| 21 | `notes` | string | *(empty)* | Free-text notes. Reserved; empty for all rows in this release. |
| 22 | `source_institution` | string |  Values: `RIFI/VAFS`, `Binh Dinh (regional timber-supply)`, `Ho Chi Minh City (field collection)`. |Sourcing channel/institution of the specimen.Uniform within each class. |
| 23 | `dart_tofms_verified` | boolean | `true` / `false`|`true` if the specimen's label was confirmed by DART-TOFMS (8 representative specimens, one per class), else `false`. |
---

## 2. Categorical value sets

### 2.1 Classes (`class_index` → `class_name`, with image counts)

| `class_index` | `class_name` | Images |
|:---:|---|---:|
| 0 | Eucalyptus camaldulensis | 288 |
| 1 | Eucalyptus cladocalyx | 348 |
| 2 | Eucalyptus deglupta | 341 |
| 3 | Eucalyptus diversicolor | 337 |
| 4 | Eucalyptus grandis | 405 |
| 5 | Eucalyptus microcorys | 396 |
| 6 | Eucalyptus saligna | 397 |
| 7 | Syzygium hemisphericum | 398 |
| | **Total** | **2,910** |

`Syzygium hemisphericum` (index 7) is an out-group class.

### 2.2 `split` (strict pHash-clean split, Split B)

| Value | Images |
|---|---:|
| `train` | 2,025 |
| `val` | 437 |
| `test` | 448 |

The `split` column encodes **only Split B** (the strict pHash-clean partition). The
alternative reference partition (Split A) is distributed separately as
`split_A_reference.csv`; it is not encoded in `metadata.csv`.

### 2.3 `light_condition`

| Value | Images |
|---|---:|
| `natural` | 2,394 |
| `indoor` | 255 |
| `outdoor` | 261 |

`indoor`/`outdoor` values correspond to specimens additionally imaged under paired
artificial (indoor) and natural (outdoor) illumination.

### 2.4 Resolution (`width` × `height`)

| Resolution | Images |
|---|---:|
| 1280 × 1024 | 1,963 |
| 2560 × 2048 | 786 |
| 2560 × 1920 | 161 |

---

## 3. Grouping and leakage keys (read before splitting)

Several columns describe how images relate to physical specimens and perceptual
near-duplicates. Choosing the correct key is essential for leakage-safe evaluation.

- **`group_id`** — the authoritative physical-specimen identifier (one value per
  specimen subfolder). **86 distinct values** (= 86 physical specimens). Use this for
  specimen-disjoint splitting.
- **`constraint_component`** — the grouping key used to construct the released
  **Split B**. It is the union of the physical specimen with perceptual-hash
  near-duplicate components, so that near-duplicate images can never be split across
  partitions. **80 distinct values**; 6 components each span more than one `group_id`
  (i.e. a near-duplicate bridged two specimens, merging them into one split-unit).
  The released `split` column is group-disjoint on **both** `group_id` and
  `constraint_component`.
- **`phash_component`** — perceptual-hash near-duplicate component id. **2,844 distinct
  values**; 54 components contain more than one image (66 images are grouped with at
  least one perceptual near-duplicate).
- **`specimen_key`** — a short specimen key. **Only 78 distinct values, and 8 of them
  map to more than one `group_id`** (e.g. `3358.1` collapses specimen folders `.1` and
  `.10` because the trailing `.10` is parsed as `.1`). ⚠️ **`specimen_key` is ambiguous
  and must not be used as a grouping/leakage key.** Use `group_id` or
  `constraint_component` instead. It is retained only as a human-readable label.
- **`parsed_group_id`** — identical to `group_id` for every row; retained for provenance.
- **`legacy_parsed_group_id`** — the earlier, pre-correction grouping label of the form
  `class_name :: folder`. In the original release, filename-derived grouping merged
  distinct specimens; grouping was subsequently re-anchored to the physical-specimen
  subfolders. This column documents the earlier assignment for transparency and
  reproducibility; it should **not** be used for splitting.

**Recommended usage.** For a specimen-disjoint split, group by `group_id`. To reproduce
the released strict split exactly, use the `split` column (equivalently, group by
`constraint_component`).

---

## 4. Provenance and known-legacy notes

- **Legacy spelling in path/provenance columns.** `image_path`, `raw_path`,
  `original_filename`, `top_level_folder`, `group_id`, `parsed_group_id`, and
  `legacy_parsed_group_id` preserve the original folder/file names, which contain the
  legacy misspelling `camandulensis`. The **authoritative species label is
  `class_name`**, which uses the corrected spelling `Eucalyptus camaldulensis`. Consumers
  should key on `class_name` / `class_index`, not on path strings.
- **`notes`** is an empty reserved column in this release.
- **Redundant columns retained for provenance** (`parsed_group_id`, `raw_path`,
  `original_filename`, `top_level_folder`, `legacy_parsed_group_id`) can be ignored for
  modelling; they document the acquisition and grouping history.

---

## 5. Integrity summary (as released)

- Images: **2,910**; classes: **8**; physical specimens (`group_id`): **86**.
- File format: all **`.png`**; unreadable images: **0**.
- Exact-duplicate groups by `sha256`: **0**.
- Leakage audit (Split B, `split` column): no cross-partition overlap by group,
  exact SHA-256 hash, filename, or perceptual hash (Hamming ≤ 5 and ≤ 10).
- Companion files: `split_A_reference.csv`, `split_B_strict.csv`, `label_map.json`,
  leakage-audit reports, and baseline-output files.
