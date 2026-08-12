from __future__ import annotations

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = PKG_ROOT / "model" / "sku_and_price_tag.pth"
CLASS_NAMES = ["sku", "price_tag"]

WORK_DIR = PKG_ROOT / "backend" / "work"
DATASET_DIR = PKG_ROOT / "dataset"

def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _detect_device()  # cuda on the RunPod GPU pod, mps on this Mac, cpu otherwise
DETECTION_THRESHOLD = 0.55

# Click-to-segment backend. "sam3" needs approved gated access to facebook/sam3
# (see README) - until that's approved, use "sam2" (ungated, fast on Apple Silicon).
# Both backends share the same encode_image/predict_click interface, see
# backend/segmentation.py - switching back is just flipping this one value.
SEGMENTATION_BACKEND = "sam2"

SAM3_CHECKPOINT_ID = "facebook/sam3"
SAM2_CHECKPOINT_ID = "facebook/sam2.1-hiera-small"

# Dataset storage. If AWS_ACCESS_KEY_ID (etc.) is present in the environment,
# save_annotation() writes straight to S3 and skips DATASET_DIR entirely - see
# backend/dataset_writer.py and backend/s3_sync.py. Local dev without AWS
# credentials configured falls back to DATASET_DIR untouched.
S3_BUCKET = "scai-vision-dev"
S3_PREFIX = "dataset_rf"

