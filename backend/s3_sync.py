"""S3 storage for the annotation dataset. Only active when AWS credentials are
present in the environment (boto3's default credential chain picks up
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION automatically -
no credential-handling code needed here).

Mirrors the exact YOLO layout backend/dataset_writer.py uses locally:
  s3://{S3_BUCKET}/{S3_PREFIX}/data.yaml
  s3://{S3_BUCKET}/{S3_PREFIX}/{split}/images/<name>.jpg
  s3://{S3_BUCKET}/{S3_PREFIX}/{split}/labels/<name>.txt

This module raises on failure rather than swallowing errors - when S3 is the
sole store for a submitted annotation (see dataset_writer.py), a failed
upload must surface as a real error to the caller, not a silent no-op.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from backend.config import CLASS_NAMES, S3_BUCKET, S3_PREFIX

_client = None


def is_configured() -> bool:
    return bool(os.environ.get("AWS_ACCESS_KEY_ID"))


def _get_client():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("s3")
    return _client


def _key(*parts: str) -> str:
    return "/".join([S3_PREFIX.strip("/"), *parts])


def _ensure_data_yaml() -> None:
    from botocore.exceptions import ClientError

    client = _get_client()
    key = _key("data.yaml")
    try:
        client.head_object(Bucket=S3_BUCKET, Key=key)
        return  # already exists
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise

    content = yaml.safe_dump(
        {
            "path": f"s3://{S3_BUCKET}/{S3_PREFIX}",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(CLASS_NAMES),
            "names": list(CLASS_NAMES),
        },
        default_flow_style=False,
        sort_keys=False,
    )
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=content.encode("utf-8"))


def upload_annotation(
    split: str, source_image_path: str, image_filename: str, label_text: str
) -> dict:
    """Uploads the image + label text straight to S3. Raises on any failure -
    callers must treat this as the authoritative write, not a best-effort
    mirror, since nothing else persists the annotation when S3 is configured.
    """
    _ensure_data_yaml()
    client = _get_client()

    stem = Path(image_filename).stem
    image_key = _key(split, "images", image_filename)
    label_key = _key(split, "labels", f"{stem}.txt")

    client.upload_file(str(source_image_path), S3_BUCKET, image_key)
    client.put_object(Bucket=S3_BUCKET, Key=label_key, Body=label_text.encode("utf-8"))

    return {
        "image_path": f"s3://{S3_BUCKET}/{image_key}",
        "label_path": f"s3://{S3_BUCKET}/{label_key}",
    }
