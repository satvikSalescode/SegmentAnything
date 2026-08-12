from __future__ import annotations

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PKG_ROOT / "model" / "sku_and_price_tag.pth"
CLASS_NAMES = ["sku", "price_tag"]

WORK_DIR = PKG_ROOT / "backend" / "work"
DATASET_DIR = PKG_ROOT / "dataset"

DEVICE = "mps"  # falls back to cpu automatically inside each infer module if mps is unavailable
DETECTION_THRESHOLD = 0.55

# Click-to-segment backend. "sam3" needs approved gated access to facebook/sam3
# (see README) - until that's approved, use "sam2" (ungated, fast on Apple Silicon).
# Both backends share the same encode_image/predict_click interface, see
# backend/segmentation.py - switching back is just flipping this one value.
SEGMENTATION_BACKEND = "sam2"

SAM3_CHECKPOINT_ID = "facebook/sam3"
SAM2_CHECKPOINT_ID = "facebook/sam2.1-hiera-small"
