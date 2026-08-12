from __future__ import annotations

from pydantic import BaseModel


class DetectionOut(BaseModel):
    id: str
    class_id: int
    class_name: str
    bbox_xyxy: list[float]
    score: float


class UploadResponse(BaseModel):
    image_id: str
    image_url: str
    width: int
    height: int
    detections: list[DetectionOut]
    class_names: list[str]


class PointIn(BaseModel):
    x: float
    y: float
    label: int  # 1 = positive, 0 = negative


class SegmentRequest(BaseModel):
    image_id: str
    points: list[PointIn]


class SegmentResponse(BaseModel):
    mask_png_base64: str
    bbox_xyxy: list[float]
    score: float


class BoxIn(BaseModel):
    class_id: int
    bbox_xyxy: list[float]
    source: str  # "rfdetr" | "human" (SAM click) | "manual" (hand-drawn box)


class SubmitRequest(BaseModel):
    image_id: str
    split: str
    boxes: list[BoxIn]


class SubmitResponse(BaseModel):
    saved: bool
    kept_from_rfdetr: int
    added_by_human: int
    image_path: str
    label_path: str


class DiscardRequest(BaseModel):
    image_id: str


class DiscardResponse(BaseModel):
    discarded: bool
