from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend import dataset_writer, rfdetr_infer, segmentation
from backend.config import CLASS_NAMES, PKG_ROOT, WORK_DIR
from backend.schemas import (
    BoxIn,
    DetectionOut,
    DiscardRequest,
    DiscardResponse,
    SegmentRequest,
    SegmentResponse,
    SubmitRequest,
    SubmitResponse,
    UploadResponse,
)

FRONTEND_DIR = PKG_ROOT / "frontend"

# image_id -> {"path": Path, "filename": str, "width": int, "height": int}
_sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rfdetr_infer.warm_up()
    yield


app = FastAPI(title="SKU Annotation Tool", lifespan=lifespan)


def _get_session(image_id: str) -> dict:
    session = _sessions.get(image_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown image_id {image_id!r} - upload it again")
    return session


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile, background_tasks: BackgroundTasks) -> UploadResponse:
    image_id = uuid.uuid4().hex
    session_dir = WORK_DIR / image_id
    session_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{image_id}.jpg"
    dest_path = session_dir / filename

    raw_bytes = await file.read()

    def _save_and_measure() -> tuple[int, int]:
        tmp_path = session_dir / "upload_raw"
        tmp_path.write_bytes(raw_bytes)
        with Image.open(tmp_path) as im:
            im.convert("RGB").save(dest_path, format="JPEG", quality=95)
        tmp_path.unlink(missing_ok=True)
        with Image.open(dest_path) as im:
            return im.size

    width, height = await run_in_threadpool(_save_and_measure)

    _sessions[image_id] = {
        "path": dest_path,
        "filename": filename,
        "width": width,
        "height": height,
    }

    detections = await run_in_threadpool(rfdetr_infer.predict, str(dest_path))
    detections_out = [
        DetectionOut(
            id=f"det-{i}",
            class_id=d.class_id,
            class_name=d.class_name,
            bbox_xyxy=list(d.bbox_xyxy),
            score=d.score,
        )
        for i, d in enumerate(detections)
    ]

    background_tasks.add_task(segmentation.encode_image, image_id, str(dest_path))

    return UploadResponse(
        image_id=image_id,
        image_url=f"/work/{image_id}/{filename}",
        width=width,
        height=height,
        detections=detections_out,
        class_names=CLASS_NAMES,
    )


@app.post("/api/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest) -> SegmentResponse:
    session = _get_session(req.image_id)
    points = [(p.x, p.y, p.label) for p in req.points]

    try:
        await run_in_threadpool(segmentation.encode_image, req.image_id, str(session["path"]))
        result = await run_in_threadpool(segmentation.predict_click, req.image_id, points)
    except segmentation.AccessError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return SegmentResponse(
        mask_png_base64=segmentation.mask_to_overlay_png_base64(result.mask),
        bbox_xyxy=list(result.bbox_xyxy),
        score=result.score,
    )


@app.post("/api/submit", response_model=SubmitResponse)
async def submit(req: SubmitRequest) -> SubmitResponse:
    session = _get_session(req.image_id)

    try:
        result = await run_in_threadpool(
            dataset_writer.save_annotation,
            split=req.split,
            source_image_path=str(session["path"]),
            image_filename=session["filename"],
            boxes=[{"class_id": b.class_id, "bbox_xyxy": b.bbox_xyxy} for b in req.boxes],
        )
    except Exception as e:  # noqa: BLE001 - S3/boto3 can raise many distinct types
        raise HTTPException(
            status_code=502,
            detail=f"Failed to save annotation - nothing was written, please retry. ({e})",
        ) from e

    kept_from_rfdetr = sum(1 for b in req.boxes if b.source == "rfdetr")
    # "human" (SAM click) and "manual" (hand-drawn box) both count as human-added
    added_by_human = sum(1 for b in req.boxes if b.source != "rfdetr")

    _cleanup_session(req.image_id)

    return SubmitResponse(
        saved=True,
        kept_from_rfdetr=kept_from_rfdetr,
        added_by_human=added_by_human,
        image_path=result["image_path"],
        label_path=result["label_path"],
    )


@app.post("/api/discard", response_model=DiscardResponse)
async def discard(req: DiscardRequest) -> DiscardResponse:
    """Abandon the current image without writing anything to dataset/ - lets the
    annotator back out and upload a different image instead of submitting."""
    session = _sessions.get(req.image_id)
    _cleanup_session(req.image_id)
    if session is not None:
        await run_in_threadpool(shutil.rmtree, session["path"].parent, ignore_errors=True)
    return DiscardResponse(discarded=True)


def _cleanup_session(image_id: str) -> None:
    segmentation.evict(image_id)
    _sessions.pop(image_id, None)


app.mount("/work", StaticFiles(directory=str(WORK_DIR)), name="work")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
