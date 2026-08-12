"""Writes accepted boxes into dataset/<split>/{images,labels}/ in the exact
YOLO layout rfdetr_sagemaker trains on: normalized `class_id cx cy w h` per
line, plus a data.yaml with nc/names.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from PIL import Image

from backend.config import CLASS_NAMES, DATASET_DIR

Split = str  # "train" | "valid" | "test"
VALID_SPLITS = ("train", "valid", "test")


def _ensure_dataset_skeleton() -> None:
    """rfdetr.datasets.yolo.is_valid_yolo_dataset requires train/ and valid/
    (each with images/ and labels/) to physically exist, even if empty."""
    for split in VALID_SPLITS:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)


def _ensure_data_yaml() -> None:
    _ensure_dataset_skeleton()
    data_yaml_path = DATASET_DIR / "data.yaml"
    if data_yaml_path.exists():
        return
    content = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)


def _bbox_xyxy_to_yolo_line(
    class_id: int, bbox_xyxy: tuple[float, float, float, float], img_w: int, img_h: int
) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    x1, x2 = sorted((max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))))
    y1, y2 = sorted((max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))))
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def save_annotation(
    split: Split,
    source_image_path: str,
    image_filename: str,
    boxes: list[dict],
) -> dict:
    """boxes: list of {"class_id": int, "bbox_xyxy": [x1,y1,x2,y2]}.
    Writes dataset/<split>/images/<image_filename> and the matching
    dataset/<split>/labels/<stem>.txt, and ensures dataset/data.yaml exists.
    Returns {"image_path": ..., "label_path": ..., "num_boxes": ...}.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    _ensure_data_yaml()

    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_image_path) as im:
        img_w, img_h = im.size

    dest_image_path = images_dir / image_filename
    shutil.copyfile(source_image_path, dest_image_path)

    stem = Path(image_filename).stem
    dest_label_path = labels_dir / f"{stem}.txt"
    lines = [
        _bbox_xyxy_to_yolo_line(b["class_id"], tuple(b["bbox_xyxy"]), img_w, img_h)
        for b in boxes
    ]
    dest_label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return {
        "image_path": str(dest_image_path),
        "label_path": str(dest_label_path),
        "num_boxes": len(boxes),
    }
