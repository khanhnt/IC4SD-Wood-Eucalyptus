"""Small reusable helpers for public dataset preparation and baselines."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Allows metadata scripts to run before torch is installed.
    torch = None

from .constants import CANONICAL_CLASSES, CLASS_TO_INDEX, IMAGE_EXTENSIONS


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def is_hidden_or_system(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts) or path.name in {"Thumbs.db", "desktop.ini", ".DS_Store"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not is_hidden_or_system(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ascii_key(text: str) -> str:
    """Return a lowercase ASCII-ish key robust to punctuation and accents."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"^\s*\d+\s*[\._-]*\s*", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_class_name(folder_name: str) -> tuple[str, str]:
    """Normalize common folder-name variants to canonical species names.

    Returns:
        (canonical_class_name, note)
    """
    key = ascii_key(folder_name)
    compact = key.replace(" ", "")
    note = ""

    aliases = {
        "eucalyptuscamaldulensis": "Eucalyptus camaldulensis",
        "eucalyptuscamandulensis": "Eucalyptus camaldulensis",
        "ecamaldulensis": "Eucalyptus camaldulensis",
        "ecamandulensis": "Eucalyptus camaldulensis",
        "ecamaldulensis": "Eucalyptus camaldulensis",
        "eucalyptuscladocalyx": "Eucalyptus cladocalyx",
        "ecladocalyx": "Eucalyptus cladocalyx",
        "eucalyptusdeglupta": "Eucalyptus deglupta",
        "eucalyptusdaglupta": "Eucalyptus deglupta",
        "edeglupta": "Eucalyptus deglupta",
        "edaglupta": "Eucalyptus deglupta",
        "eucalyptusdiversicolor": "Eucalyptus diversicolor",
        "ediversicolor": "Eucalyptus diversicolor",
        "eucalyptusgrandis": "Eucalyptus grandis",
        "egrandis": "Eucalyptus grandis",
        "eucalyptusmicrocorys": "Eucalyptus microcorys",
        "emicrocorys": "Eucalyptus microcorys",
        "eucalyptussaligna": "Eucalyptus saligna",
        "esaligna": "Eucalyptus saligna",
        "syzygiumhemisphericum": "Syzygium hemisphericum",
        "shemisphericum": "Syzygium hemisphericum",
    }

    if compact in aliases:
        canonical = aliases[compact]
        if canonical.lower().replace(" ", "") != compact:
            note = f"normalized from folder '{folder_name}'"
        return canonical, note

    # Token fallback catches names such as "E. camaldulensis" after punctuation cleanup.
    genus = None
    if "syzygium" in key or re.search(r"\bs\b", key):
        genus = "Syzygium"
    elif "eucalyptus" in key or re.search(r"\be\b", key):
        genus = "Eucalyptus"

    species_aliases = {
        "camaldulensis": "camaldulensis",
        "camandulensis": "camaldulensis",
        "cladocalyx": "cladocalyx",
        "deglupta": "deglupta",
        "daglupta": "deglupta",
        "diversicolor": "diversicolor",
        "grandis": "grandis",
        "microcorys": "microcorys",
        "saligna": "saligna",
        "hemisphericum": "hemisphericum",
    }
    for token, species in species_aliases.items():
        if token in key:
            if species == "hemisphericum":
                canonical = "Syzygium hemisphericum"
            elif genus in {None, "Eucalyptus"}:
                canonical = f"Eucalyptus {species}"
            else:
                canonical = f"{genus} {species}"
            if canonical in CLASS_TO_INDEX:
                return canonical, f"normalized from folder '{folder_name}'"

    raise ValueError(
        f"Could not normalize top-level folder '{folder_name}'. "
        f"Expected one of: {', '.join(CANONICAL_CLASSES)}"
    )


def label_map_payload() -> dict[str, str]:
    return {str(idx): name for idx, name in enumerate(CANONICAL_CLASSES)}


def short_group_stem(stem: str) -> str:
    stem = re.sub(r"\s+", " ", str(stem).strip())
    stem = re.sub(r"\s*\.?\s*\(\d+\)\s*$", "", stem).strip()
    stem = stem.rstrip(". ").strip()
    return stem or str(stem)


def parse_group_id(relative_path: Path, class_name: str) -> str:
    """Parse a conservative specimen/acquisition group from a raw relative path.

    If images are stored in nested folders, the first level below the class
    folder is used as the group. Otherwise the final image index is stripped
    from the filename when possible.
    """
    parts = relative_path.parts
    if len(parts) >= 3:
        group = Path(*parts[1:-1]).as_posix()
    else:
        group = short_group_stem(relative_path.stem)
    return f"{class_name}::{group}"


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def timestamp_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
