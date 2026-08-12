"""One-off spike script: figure out which RFDETR variant/resolution
sku_and_price_tag.pth actually is, by trying each class with num_classes=2
until one loads and runs .predict() cleanly on a sample image.

Usage: python backend/_detect_ckpt.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
CKPT = PKG_ROOT / "model" / "sku_and_price_tag.pth"
SAMPLE_IMAGE_DIRS = [
    Path("/Users/satvikchaudhary/Downloads/rfdetr_sagemaker/data/samples/sku110k_train"),
    Path("/Users/satvikchaudhary/Downloads/rfdetr_sagemaker/data/samples/sku110k_test"),
]


def find_sample_image() -> Path:
    for d in SAMPLE_IMAGE_DIRS:
        if d.exists():
            imgs = sorted(d.glob("*.jpg"))
            if imgs:
                return imgs[0]
    raise SystemExit("No sample image found to test with")


def main() -> None:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge, RFDETRBase

    variants = {
        "RFDETRNano": RFDETRNano,
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRBase": RFDETRBase,
        "RFDETRLarge": RFDETRLarge,
    }

    sample_image = find_sample_image()
    print(f"Checkpoint: {CKPT} ({CKPT.stat().st_size / 1e6:.1f} MB)")
    print(f"Sample image: {sample_image}")
    print()

    for name, cls in variants.items():
        print(f"--- Trying {name} ---")
        try:
            model = cls(
                device="cpu",  # cpu for the detection spike; avoids MPS quirks during load
                pretrain_weights=str(CKPT),
                num_classes=2,
            )
            detections = model.predict(str(sample_image), threshold=0.3)
            print(f"  LOADED OK. Detections: {len(detections)}")
            if len(detections) > 0:
                print(f"  class_ids seen: {sorted(set(detections.class_id.tolist()))}")
                print(f"  top scores: {detections.confidence[:5]}")
            print(f"  >>> MATCH: {name} <<<")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
            # Uncomment for full traceback while debugging:
            # traceback.print_exc()
        print()

    print("No variant matched cleanly with num_classes=2. See errors above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
