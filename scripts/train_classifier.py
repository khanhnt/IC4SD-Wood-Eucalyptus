#!/usr/bin/env python3
"""Train a public DenseNet-121 technical-validation baseline from manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / "outputs" / ".matplotlib"))
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    TORCH_IMPORT_ERROR = exc
    torch = None
    nn = None
    AdamW = None
    CosineAnnealingLR = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass

    transforms = None

sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from wood_dib.constants import IMAGENET_MEAN, IMAGENET_STD
from wood_dib.splitting import load_metadata, validate_split_completeness
from wood_dib.utils import get_git_commit, save_json, set_reproducible_seed, timestamp_utc


def no_grad_context():
    if torch is None:
        return lambda func: func
    return torch.no_grad()


def get_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


class ManifestClassificationDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, raw_root: Path | None, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True).copy()
        self.raw_root = raw_root.expanduser().resolve() if raw_root is not None else None
        self.transform = transform
        self.paths = [self._resolve_path(row) for _, row in self.frame.iterrows()]

    def _resolve_path(self, row: pd.Series) -> Path:
        candidates = []
        if self.raw_root is not None:
            candidates.append(self.raw_root / str(row["relative_path"]))
        if "raw_path" in row and pd.notna(row["raw_path"]):
            candidates.append(Path(str(row["raw_path"])))
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "Could not resolve image path for "
            f"image_id={row.get('image_id')} relative_path={row.get('relative_path')}. "
            "Pass --raw-root if the metadata raw_path values are from another machine."
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.paths[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image), int(row["class_index"]), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IC4SD-Wood-Eucalyptus classifier baseline.")
    parser.add_argument("--model", choices=["densenet121"], default="densenet121")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def build_transforms(input_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name != "densenet121":
        raise ValueError(f"Unsupported model: {model_name}")
    from torchvision.models import DenseNet121_Weights, densenet121

    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def compute_metrics(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    labels: list[int],
) -> tuple[float, dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return total_loss / max(total_samples, 1), compute_metrics(y_true, y_pred, labels)


@no_grad_context()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    labels: list[int],
) -> tuple[float, dict[str, float], list[int], list[int], list[str]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    paths: list[str] = []

    for images, targets, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        paths.extend(batch_paths)

    return total_loss / max(total_samples, 1), compute_metrics(y_true, y_pred, labels), y_true, y_pred, paths


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8", buffering=1) as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def save_per_class_metrics(y_true: list[int], y_pred: list[int], class_names: list[str], path: Path) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    support = np.array([(np.array(y_true) == label).sum() for label in labels])
    frame = pd.DataFrame(
        {
            "class_index": labels,
            "class_name": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    frame.to_csv(path, index=False)
    return frame


def save_confusion_matrix(y_true: list[int], y_pred: list[int], class_names: list[str], csv_path: Path, png_path: Path) -> None:
    plt = get_matplotlib_pyplot()
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("DenseNet-121 Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[row, col] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def save_training_curves(history_csv: Path, output_path: Path) -> None:
    plt = get_matplotlib_pyplot()
    history = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_macro_f1"], label="Train macro-F1")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="Validation macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_latex_table(metrics: dict[str, float], best_epoch: int, config: dict, path: Path) -> None:
    setting = (
        f"DenseNet-121, ImageNet, {config['input_size']} px, "
        f"AdamW lr={config['lr']}, batch={config['batch_size']}"
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Baseline performance for wood species classification using DenseNet-121.}",
        "\\label{tab:densenet121_baseline}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Setting & Acc. (\\%) & Prec. (\\%) & Rec. (\\%) & Macro-F1 (\\%) & Weighted-F1 (\\%) & Best epoch \\\\",
        "\\midrule",
        (
            f"{setting} & "
            f"{metrics['accuracy'] * 100:.2f} & "
            f"{metrics['macro_precision'] * 100:.2f} & "
            f"{metrics['macro_recall'] * 100:.2f} & "
            f"{metrics['macro_f1'] * 100:.2f} & "
            f"{metrics['weighted_f1'] * 100:.2f} & "
            f"{best_epoch} \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_frames(metadata_csv: Path, split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_metadata(metadata_csv)
    split_df = pd.read_csv(split_csv)
    validate_split_completeness(split_df)
    frame = split_df.merge(
        metadata[["image_id", "raw_path", "original_filename"] if "raw_path" in metadata.columns else ["image_id"]],
        on="image_id",
        how="left",
    )
    if frame["class_index"].isna().any():
        raise ValueError("Split manifest contains rows without class_index.")
    return metadata, frame


def main() -> None:
    args = parse_args()
    if TORCH_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Training requires PyTorch and torchvision. Install a CUDA-compatible "
            "PyTorch build first, then install public_dib_code/requirements.txt."
        ) from TORCH_IMPORT_ERROR

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = (device.type == "cuda") if args.amp is None else bool(args.amp and device.type == "cuda")

    metadata, frame = load_frames(args.metadata_csv, args.split_csv)
    class_table = frame[["class_index", "class_name"]].drop_duplicates().sort_values("class_index")
    class_names = class_table["class_name"].tolist()
    labels = list(range(len(class_names)))
    label_map = {str(idx): name for idx, name in enumerate(class_names)}

    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError("Train, validation, and test splits must all contain at least one image.")

    train_transform, eval_transform = build_transforms(args.input_size)
    train_ds = ManifestClassificationDataset(train_frame, args.raw_root, train_transform)
    val_ds = ManifestClassificationDataset(val_frame, args.raw_root, eval_transform)
    test_ds = ManifestClassificationDataset(test_frame, args.raw_root, eval_transform)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    config = {
        "dataset_name": "IC4SD-Wood-Eucalyptus",
        "created_at_utc": timestamp_utc(),
        "git_commit": get_git_commit(REPO_ROOT),
        "model": args.model,
        "pretrained": "ImageNet",
        "metadata_csv": str(args.metadata_csv),
        "split_csv": str(args.split_csv),
        "raw_root": str(args.raw_root) if args.raw_root is not None else None,
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "input_size": args.input_size,
        "label_smoothing": args.label_smoothing,
        "num_workers": args.num_workers,
        "device": str(device),
        "amp": amp_enabled,
        "n_classes": len(class_names),
        "n_train": len(train_frame),
        "n_val": len(val_frame),
        "n_test": len(test_frame),
    }
    save_json(args.output_dir / "config.json", config)
    save_json(args.output_dir / "label_map.json", label_map)
    frame.to_csv(args.output_dir / "split_manifest_used.csv", index=False)

    print(f"[Train] Device: {device} | AMP: {amp_enabled}", flush=True)
    print(f"[Train] Classes: {len(class_names)} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_f1 = -1.0
    best_epoch = 0
    start_epoch = 1
    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_f1 = float(checkpoint.get("best_val_macro_f1", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[Train] Resumed from {args.resume} at epoch {start_epoch}", flush=True)

    history_csv = args.output_dir / "training_history.csv"
    if history_csv.exists() and args.resume is None:
        history_csv.unlink()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_enabled, labels
        )
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, amp_enabled, labels)
        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": lr,
            "elapsed_sec": elapsed,
        }
        append_history(history_csv, row)

        checkpoint_payload = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "config": config,
            "label_map": label_map,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        save_checkpoint(args.output_dir / "last_model.pt", checkpoint_payload)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_val_macro_f1"] = best_f1
            save_checkpoint(args.output_dir / "best_model.pt", checkpoint_payload)

        print(
            f"[DenseNet121] Ep {epoch:02d}/{args.epochs} | "
            f"loss {train_loss:.4f} | val_macro_f1 {val_metrics['macro_f1']:.4f} | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    best_checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state"])
    best_epoch = int(best_checkpoint["best_epoch"])
    best_f1 = float(best_checkpoint["best_val_macro_f1"])

    test_loss, test_metrics, y_true, y_pred, paths = evaluate(model, test_loader, criterion, device, amp_enabled, labels)
    test_metrics["test_loss"] = test_loss
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_val_macro_f1"] = best_f1

    save_json(args.output_dir / "test_metrics.json", test_metrics)
    pd.DataFrame([test_metrics]).to_csv(args.output_dir / "test_metrics.csv", index=False)
    save_per_class_metrics(y_true, y_pred, class_names, args.output_dir / "per_class_metrics.csv")
    save_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        csv_path=args.output_dir / "confusion_matrix.csv",
        png_path=args.output_dir / "densenet121_confusion_matrix.png",
    )
    save_training_curves(history_csv, args.output_dir / "densenet121_training_curves.png")

    pd.DataFrame(
        {
            "path": paths,
            "true_index": y_true,
            "pred_index": y_pred,
            "true_class": [class_names[i] for i in y_true],
            "pred_class": [class_names[i] for i in y_pred],
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    write_latex_table(test_metrics, best_epoch=best_epoch, config=config, path=args.output_dir / "densenet121_latex_table.tex")
    run_summary = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_metrics": test_metrics,
        "outputs": {
            "best_model": str(args.output_dir / "best_model.pt"),
            "test_metrics": str(args.output_dir / "test_metrics.csv"),
            "per_class_metrics": str(args.output_dir / "per_class_metrics.csv"),
            "confusion_matrix": str(args.output_dir / "confusion_matrix.csv"),
            "latex_table": str(args.output_dir / "densenet121_latex_table.tex"),
        },
    }
    save_json(args.output_dir / "run_summary.json", run_summary)
    print(f"[Done] DenseNet-121 baseline outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
