"""RF-DETR inference wrapper around the sku_and_price_tag checkpoint.

The checkpoint was determined (see backend/_detect_ckpt.py) to be an
RFDETRLarge model fine-tuned with 2 classes: sku (id 0), price_tag (id 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from backend.config import CHECKPOINT_PATH, CLASS_NAMES, DETECTION_THRESHOLD, DEVICE

_model = None
_model_lock = Lock()


@dataclass
class Detection:
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from rfdetr import RFDETRLarge

                _model = RFDETRLarge(
                    device=DEVICE,
                    pretrain_weights=str(CHECKPOINT_PATH),
                    num_classes=len(CLASS_NAMES),
                )
    return _model


def warm_up() -> None:
    """Force model load + a first (slow, MPS-kernel-compiling) predict at startup
    so the first real user upload isn't the one that pays the warm-up cost."""
    import numpy as np
    from PIL import Image

    model = get_model()
    dummy = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    model.predict(dummy, threshold=DETECTION_THRESHOLD)


def predict(image_path: str, threshold: float = DETECTION_THRESHOLD) -> list[Detection]:
    model = get_model()
    detections = model.predict(str(image_path), threshold=threshold)

    results: list[Detection] = []
    for xyxy, score, class_id in zip(
        detections.xyxy, detections.confidence, detections.class_id
    ):
        cid = int(class_id)
        results.append(
            Detection(
                class_id=cid,
                class_name=CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid),
                bbox_xyxy=tuple(float(v) for v in xyxy),
                score=float(score),
            )
        )
    return results
