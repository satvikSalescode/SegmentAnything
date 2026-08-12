"""SAM3 (Meta) interactive point-click segmentation, via the Hugging Face
`transformers` integration (`Sam3TrackerModel` / `Sam3TrackerProcessor`).

We deliberately use the *tracker* classes (Promptable Visual Segmentation --
the classic SAM1/2-style "click a point, get a mask for that one object" task)
rather than `Sam3Model`/`Sam3Processor` (Promptable Concept Segmentation --
text/exemplar -> all matching instances). See the project plan for why.

The native facebookresearch/sam3 repo requires CUDA and does not run on this
Mac at all; the transformers path is the only way to run SAM3 locally here.

Usage pattern (mirrors SAM2's "encode once, click many times"):
    encode_image(image_id, image_path)          # slow, once per image
    predict_click(image_id, points)              # fast, once per click
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from PIL import Image

from backend.config import DEVICE, SAM3_CHECKPOINT_ID as CHECKPOINT_ID

_model = None
_processor = None
_load_lock = Lock()

# image_id -> {"image_embeddings": [...], "original_sizes": [[h, w]]}
_embedding_cache: dict[str, dict] = {}
_cache_lock = Lock()


class Sam3AccessError(RuntimeError):
    """Raised when the gated facebook/sam3 checkpoint can't be downloaded/loaded."""


@dataclass
class SegmentResult:
    mask: np.ndarray  # bool, shape (H, W), original image resolution
    bbox_xyxy: tuple[float, float, float, float]
    score: float


def _pick_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return requested


def get_model_and_processor():
    global _model, _processor
    if _model is None or _processor is None:
        with _load_lock:
            if _model is None or _processor is None:
                from transformers import Sam3TrackerModel, Sam3TrackerProcessor

                device = _pick_device(DEVICE)
                try:
                    _processor = Sam3TrackerProcessor.from_pretrained(CHECKPOINT_ID)
                    _model = Sam3TrackerModel.from_pretrained(
                        CHECKPOINT_ID, device_map=device
                    )
                    _model.eval()
                except Exception as e:  # noqa: BLE001
                    raise Sam3AccessError(
                        "Could not load facebook/sam3. This checkpoint is gated on "
                        "Hugging Face: request access at https://huggingface.co/facebook/sam3 , "
                        "wait for approval, then run `huggingface-cli login` with an approved "
                        f"account's token before retrying. Original error: {e}"
                    ) from e
    return _model, _processor


def is_encoded(image_id: str) -> bool:
    with _cache_lock:
        return image_id in _embedding_cache


def encode_image(image_id: str, image_path: str) -> None:
    """Run the (slow) SAM3 vision encoder once for this image and cache the
    result so later clicks only run the (fast) mask decoder."""
    if is_encoded(image_id):
        return

    model, processor = get_model_and_processor()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(pixel_values=inputs["pixel_values"])

    with _cache_lock:
        _embedding_cache[image_id] = {
            "image_embeddings": image_embeddings,
            "original_sizes": inputs["original_sizes"],
        }


def predict_click(
    image_id: str, points: list[tuple[float, float, int]]
) -> SegmentResult:
    """points: list of (x, y, label) in ORIGINAL image pixel coordinates,
    label 1 = positive (include), 0 = negative (exclude)."""
    if not points:
        raise ValueError("predict_click requires at least one point")

    encoded = _embedding_cache.get(image_id)
    if encoded is None:
        raise ValueError(f"Image {image_id!r} has not been encoded yet - call encode_image() first")

    model, processor = get_model_and_processor()

    # Shape: [image=1, object=1, point=N, xy=2] / labels [image=1, object=1, point=N]
    input_points = [[[[float(x), float(y)] for x, y, _ in points]]]
    input_labels = [[[int(label) for _, _, label in points]]]

    click_inputs = processor(
        input_points=input_points,
        input_labels=input_labels,
        original_sizes=encoded["original_sizes"],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model(
            image_embeddings=encoded["image_embeddings"],
            input_points=click_inputs["input_points"],
            input_labels=click_inputs["input_labels"],
            multimask_output=True,
        )

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), encoded["original_sizes"], binarize=True
    )[0]  # shape (num_objects=1, num_masks=3, H, W)
    scores = outputs.iou_scores.cpu()[0, 0]  # shape (num_masks,)

    best_idx = int(torch.argmax(scores).item())
    mask = masks[0, best_idx].numpy().astype(bool)
    score = float(scores[best_idx].item())

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("SAM3 returned an empty mask for this click")
    bbox_xyxy = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)

    return SegmentResult(mask=mask, bbox_xyxy=bbox_xyxy, score=score)


def mask_to_overlay_png_base64(mask: np.ndarray, color: tuple[int, int, int] = (255, 80, 0)) -> str:
    """Transparent RGBA PNG: mask pixels get `color` at partial opacity, rest fully transparent.
    Ready to be drawn directly on top of the source image on an HTML canvas."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = 140
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def evict(image_id: str) -> None:
    with _cache_lock:
        _embedding_cache.pop(image_id, None)
