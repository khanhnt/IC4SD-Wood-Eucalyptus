"""Project constants shared by the public Data in Brief scripts."""

from __future__ import annotations

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CANONICAL_CLASSES = [
    "Eucalyptus camaldulensis",
    "Eucalyptus cladocalyx",
    "Eucalyptus deglupta",
    "Eucalyptus diversicolor",
    "Eucalyptus grandis",
    "Eucalyptus microcorys",
    "Eucalyptus saligna",
    "Syzygium hemisphericum",
]

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}
INDEX_TO_CLASS = {idx: name for name, idx in CLASS_TO_INDEX.items()}
