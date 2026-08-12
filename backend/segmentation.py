"""Facade over whichever click-to-segment backend is active (see
backend/config.py: SEGMENTATION_BACKEND). main.py only ever imports this
module, never sam2_infer/sam3_infer directly, so switching backends is a
one-line config change.
"""
from __future__ import annotations

from backend.config import SEGMENTATION_BACKEND

if SEGMENTATION_BACKEND == "sam2":
    from backend.sam2_infer import (
        Sam2AccessError as AccessError,
        SegmentResult,
        encode_image,
        evict,
        is_encoded,
        mask_to_overlay_png_base64,
        predict_click,
    )
elif SEGMENTATION_BACKEND == "sam3":
    from backend.sam3_infer import (
        Sam3AccessError as AccessError,
        SegmentResult,
        encode_image,
        evict,
        is_encoded,
        mask_to_overlay_png_base64,
        predict_click,
    )
else:
    raise ValueError(f"Unknown SEGMENTATION_BACKEND: {SEGMENTATION_BACKEND!r}")

__all__ = [
    "AccessError",
    "SegmentResult",
    "encode_image",
    "evict",
    "is_encoded",
    "mask_to_overlay_png_base64",
    "predict_click",
]
