#!/usr/bin/env python3
"""
laplacian_screening.py
======================
Quantitative image-sharpness (focus) screening for the IC4SD-Wood-Eucalyptus
dataset, produced as objective evidence for the Data in Brief revision
(Reviewer #3: "out-of-focus / poorly surfaced" concern).

For every image it computes the **variance of the Laplacian** (a standard
no-reference focus measure: higher = sharper, lower = blurrier), then reports:
  * overall distribution (mean / std / min / percentiles / max),
  * per-class distribution,
  * how many images fall below a sweep of candidate thresholds,
  * the N lowest-scoring images (so you can eyeball them),
and writes a per-image CSV + a text summary + a histogram PNG.

--------------------------------------------------------------------------
DEPENDENCIES (run once):
    pip install opencv-python pillow numpy pandas matplotlib
--------------------------------------------------------------------------
USAGE (typical):
    python laplacian_screening.py \
        --metadata /path/to/metadata.csv \
        --images-root "/Users/ntkhanh/Work/PTIT/1.Cá nhân/2.Research/2026/Article/DIB/raw_update" \
        --out-dir ./laplacian_screening

Notes on --images-root:
    It must be the folder that CONTAINS the class subfolders referenced by the
    'relative_path' column of metadata.csv (e.g. .../Eucalyptus_camaldulensis/...).
    If your images live under a 'raw/' subfolder, either point --images-root at
    that 'raw/' folder, or add  --path-column raw_path.

If you DON'T have metadata.csv handy, you can skip it and just walk a folder:
    python laplacian_screening.py --images-root <dir_with_png> --no-metadata

Comparability note:
    The variance of the Laplacian scales with image resolution. All images in
    this dataset are 2560x2048, so native computation is comparable across the
    set. The script prints the resolution distribution so you can confirm this;
    if resolutions ever differ, add e.g. --resize-longer 1024 to normalise.
"""

import argparse
import os
import sys
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("ERROR: OpenCV not found. Run: pip install opencv-python")
try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow not found. Run: pip install pillow")


# --------------------------------------------------------------------------- #
# Core focus measure
# --------------------------------------------------------------------------- #
def laplacian_variance(path, resize_longer=0):
    """Return (laplacian_var, width, height) for one image, or (None, ...) on error."""
    try:
        # PIL handles RGB, RGBA, and palette PNGs robustly; convert to 8-bit grayscale.
        with Image.open(path) as im:
            w, h = im.size
            im = im.convert("L")
            if resize_longer and max(w, h) > resize_longer:
                scale = resize_longer / float(max(w, h))
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        fm = cv2.Laplacian(arr, cv2.CV_64F).var()
        return float(fm), w, h
    except Exception as e:  # noqa: BLE001
        return None, None, None, str(e)


def _worker(task):
    key, path, resize_longer = task
    res = laplacian_variance(path, resize_longer)
    if res[0] is None:
        return key, path, None, None, None, (res[3] if len(res) > 3 else "unreadable")
    fm, w, h = res
    return key, path, fm, w, h, None


# --------------------------------------------------------------------------- #
# Build the work list
# --------------------------------------------------------------------------- #
def build_tasks_from_metadata(metadata, images_root, path_column, resize_longer):
    tasks, meta = [], {}
    with open(metadata, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if path_column not in reader.fieldnames:
            sys.exit(f"ERROR: column '{path_column}' not in metadata. "
                     f"Available: {reader.fieldnames}")
        for row in reader:
            key = row.get("image_id") or row[path_column]
            rel = row[path_column]
            # primary path, with a raw_path fallback if present
            candidates = [os.path.join(images_root, rel)]
            if "raw_path" in row and row["raw_path"]:
                candidates.append(os.path.join(images_root, row["raw_path"]))
            path = next((c for c in candidates if os.path.isfile(c)), candidates[0])
            tasks.append((key, path, resize_longer))
            meta[key] = {
                "class_name": row.get("class_name", ""),
                "split_A": row.get("split", ""),
                "relative_path": rel,
            }
    return tasks, meta


def build_tasks_from_walk(images_root, resize_longer):
    tasks, meta = [], {}
    for dirpath, _, files in os.walk(images_root):
        for fn in files:
            if fn.lower().endswith((".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")):
                path = os.path.join(dirpath, fn)
                key = os.path.relpath(path, images_root)
                tasks.append((key, path, resize_longer))
                # infer class from the top-level folder under images_root
                parts = key.split(os.sep)
                meta[key] = {"class_name": parts[0] if len(parts) > 1 else "",
                             "split_A": "", "relative_path": key}
    return tasks, meta


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarise(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()) if a.size else float("nan"),
        "std": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        "min": float(a.min()) if a.size else float("nan"),
        "p1": pct(a, 1), "p5": pct(a, 5), "p25": pct(a, 25),
        "median": pct(a, 50), "p75": pct(a, 75), "p95": pct(a, 95),
        "max": float(a.max()) if a.size else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description="Laplacian-variance sharpness screening.")
    ap.add_argument("--images-root", required=True,
                    help="Folder containing the class subfolders / images.")
    ap.add_argument("--metadata", default=None,
                    help="Path to metadata.csv (gives canonical class_name + paths).")
    ap.add_argument("--no-metadata", action="store_true",
                    help="Ignore metadata and just walk --images-root for images.")
    ap.add_argument("--path-column", default="relative_path",
                    help="Which metadata column holds the image path (default relative_path).")
    ap.add_argument("--out-dir", default="./laplacian_screening")
    ap.add_argument("--thresholds", default="50,100,150,200,300",
                    help="Comma-separated candidate blur thresholds to count below.")
    ap.add_argument("--resize-longer", type=int, default=0,
                    help="Resize longer side to this many px before measuring (0 = native).")
    ap.add_argument("--lowest", type=int, default=25,
                    help="How many lowest-scoring images to list for manual review.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.metadata and not args.no_metadata:
        tasks, meta = build_tasks_from_metadata(
            args.metadata, args.images_root, args.path_column, args.resize_longer)
        print(f"[info] {len(tasks)} images listed from metadata.")
    else:
        tasks, meta = build_tasks_from_walk(args.images_root, args.resize_longer)
        print(f"[info] {len(tasks)} images found by walking {args.images_root}.")

    if not tasks:
        sys.exit("ERROR: no images to process. Check --images-root / --path-column.")

    # Compute (parallel)
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            key, path, fm, w, h, err = fut.result()
            if err or fm is None:
                errors.append((key, path, err))
            else:
                m = meta.get(key, {})
                results.append({
                    "image_id": key, "class_name": m.get("class_name", ""),
                    "split_A": m.get("split_A", ""), "width": w, "height": h,
                    "laplacian_var": fm, "relative_path": m.get("relative_path", key),
                    "path": path,
                })
            if i % 500 == 0:
                print(f"  ...{i}/{len(tasks)} processed")

    if not results:
        sys.exit("ERROR: every image failed to load. Check the paths.")

    # ---- per-image CSV ----
    per_image_csv = os.path.join(args.out_dir, "laplacian_per_image.csv")
    fields = ["image_id", "class_name", "split_A", "width", "height", "laplacian_var", "relative_path", "path"]
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in sorted(results, key=lambda x: x["laplacian_var"]):
            wr.writerow(r)

    vals = [r["laplacian_var"] for r in results]
    overall = summarise(vals)

    # ---- resolution distribution ----
    from collections import Counter
    res_dist = Counter((r["width"], r["height"]) for r in results)

    # ---- per-class ----
    by_class = {}
    for r in results:
        by_class.setdefault(r["class_name"], []).append(r["laplacian_var"])
    per_class = {c: summarise(v) for c, v in by_class.items()}

    # ---- threshold counts ----
    a = np.asarray(vals)
    thr_counts = [(t, int((a < t).sum()), 100.0 * (a < t).mean()) for t in thresholds]

    # ---- write summary ----
    summary_txt = os.path.join(args.out_dir, "laplacian_summary.txt")
    lines = []
    lines.append("Laplacian-variance sharpness screening")
    lines.append("=" * 50)
    lines.append(f"Images measured : {overall['n']}")
    lines.append(f"Unreadable/errors: {len(errors)}")
    lines.append(f"Resize longer side: {args.resize_longer or 'native'}")
    lines.append("")
    lines.append("Resolution distribution (W x H : count):")
    for (w, h), c in sorted(res_dist.items(), key=lambda x: -x[1]):
        lines.append(f"  {w} x {h} : {c}")
    lines.append("")
    lines.append("Overall distribution of Laplacian variance:")
    for k in ["mean", "std", "min", "p1", "p5", "p25", "median", "p75", "p95", "max"]:
        lines.append(f"  {k:>7} : {overall[k]:.2f}")
    lines.append("")
    lines.append("Images below candidate thresholds:")
    for t, n, p in thr_counts:
        lines.append(f"  < {t:>6.0f} : {n:>4d} images ({p:.2f}%)")
    lines.append("")
    lines.append("Per-class (n | min | median | mean):")
    for c in sorted(per_class):
        s = per_class[c]
        lines.append(f"  {c:28s} n={s['n']:<4d} min={s['min']:8.1f} "
                     f"median={s['median']:8.1f} mean={s['mean']:8.1f}")
    lines.append("")
    lines.append(f"{args.lowest} lowest-scoring images (inspect these manually):")
    for r in sorted(results, key=lambda x: x["laplacian_var"])[:args.lowest]:
        lines.append(f"  {r['laplacian_var']:9.1f}  {r['class_name']:24s} "
                     f"{r.get('relative_path', r['image_id'])}")
    if errors:
        lines.append("")
        lines.append("Errors:")
        for key, path, err in errors[:50]:
            lines.append(f"  {key}: {err} ({path})")

    text = "\n".join(lines)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)

    # ---- histogram ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(vals, bins=60, color="#3b6ea5", edgecolor="white", linewidth=0.4)
        for t in thresholds:
            ax.axvline(t, color="#c0392b", linestyle="--", linewidth=1)
            ax.text(t, ax.get_ylim()[1] * 0.92, f"{t:.0f}", rotation=90,
                    va="top", ha="right", fontsize=8, color="#c0392b")
        ax.set_xlabel("Variance of Laplacian (focus measure)")
        ax.set_ylabel("Number of images")
        ax.set_title("Image sharpness screening (IC4SD-Wood-Eucalyptus)")
        fig.tight_layout()
        hist_png = os.path.join(args.out_dir, "laplacian_histogram.png")
        fig.savefig(hist_png, dpi=200)
        print(f"\n[info] histogram -> {hist_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] histogram skipped ({e})")

    print(f"[info] per-image CSV -> {per_image_csv}")
    print(f"[info] summary       -> {summary_txt}")
    print("\nSend me laplacian_summary.txt and I will fill the [PENDING] "
          "screening paragraph in the DIB manuscript to match the paper's style.")


if __name__ == "__main__":
    main()
