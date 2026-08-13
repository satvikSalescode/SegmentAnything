"""Local hardware benchmark: how long does THIS machine take to run one
detect + segment cycle? Run this before trying the full app on unfamiliar
or low-spec hardware (e.g. an annotator's laptop) - gives a fast go/no-go
number without needing to drive the browser UI at all.

Usage:
    python backend/_bench_local.py                    # synthetic test image
    python backend/_bench_local.py path/to/photo.jpg   # a real photo instead

To force CPU (e.g. to sidestep a tiny/old discrete GPU that might OOM-crash
instead of gracefully falling back), run with FORCE_DEVICE=cpu set first:
    Windows (PowerShell): $env:FORCE_DEVICE="cpu"; python backend/_bench_local.py
    Windows (cmd):        set FORCE_DEVICE=cpu && python backend/_bench_local.py
    macOS/Linux:          FORCE_DEVICE=cpu python backend/_bench_local.py

IMPORTANT: keep the laptop plugged in and awake while this runs. During
testing, the one-time "cold" steps (model load, first predict/click)
occasionally came back wildly inflated - minutes instead of seconds - on
runs that were otherwise identical to a fast one moments later. This looks
like a transient system/network hiccup during import or first model
construction (observed intermittently, not tied to a specific cause), NOT
real hardware performance. If ANY number below looks absurd (many minutes),
ignore it and just rerun the script - the WARM numbers (the ones that
actually matter; see the rule of thumb at the end) were consistently fast
and stable across every run regardless of whether a cold number spiked.
"""
from __future__ import annotations

import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

# Make "python backend/_bench_local.py" work regardless of current directory,
# without requiring the less-obvious "python -m backend._bench_local" form.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@contextmanager
def timed(label: str):
    t0 = time.time()
    yield
    print(f"  {label}: {time.time() - t0:.1f}s")


def _make_test_image() -> str:
    import numpy as np
    from PIL import Image

    arr = (np.random.rand(900, 1200, 3) * 255).astype("uint8")
    path = str(Path(tempfile.gettempdir()) / "bench_test_image.jpg")
    Image.fromarray(arr).save(path, quality=90)
    return path


def main() -> None:
    print(
        "NOTE: keep this laptop plugged in and awake until this finishes. "
        "Occasional transient slowness on the one-time 'cold' steps (minutes "
        "instead of seconds) has been observed and is not real hardware "
        "performance - if any number below looks absurd, just rerun the "
        "script and trust the WARM numbers, which stay fast and stable.\n"
    )

    from backend.config import DEVICE

    print(f"Device: {DEVICE}\n")

    image_path = sys.argv[1] if len(sys.argv) > 1 else _make_test_image()
    print(f"Test image: {image_path}\n")

    from backend import rfdetr_infer, segmentation

    print("RF-DETR (product/price-tag detection):")
    with timed("model load"):
        rfdetr_infer.get_model()
    with timed("first predict (cold - includes one-time kernel warm-up)"):
        rfdetr_infer.predict(image_path)
    with timed("second predict (warm - this is the real per-image cost)"):
        detections = rfdetr_infer.predict(image_path)
    print(f"  -> {len(detections)} detections\n")

    from backend.config import SEGMENTATION_BACKEND

    print(f"Segmentation ({SEGMENTATION_BACKEND} backend):")
    image_id = "bench"
    try:
        with timed("model load + image encode (once per image; includes any first-time checkpoint download)"):
            segmentation.encode_image(image_id, image_path)
        with timed("first click (still semi-cold - one-time decoder kernel warm-up)"):
            segmentation.predict_click(image_id, [(100.0, 100.0, 1)])
        with timed("second click (warm - this is the real per-click cost during a session)"):
            segmentation.predict_click(image_id, [(200.0, 200.0, 1)])
    finally:
        segmentation.evict(image_id)

    print(
        "\nRule of thumb: for annotation work to feel responsive, the WARM "
        "'second predict' and 'second click' numbers above should be a couple "
        "seconds or less - that's what repeats dozens of times per image "
        "during a real session. The one-time model-load/encode costs only "
        "happen once per session (RF-DETR) or once per uploaded image (SAM2/"
        "SAM3), so they matter less for a long session even if they look slow "
        "here. If the WARM numbers are much higher than a couple seconds, "
        "local use will feel slow for real annotation work - that's the "
        "signal to move to the RunPod deployment instead."
    )


if __name__ == "__main__":
    main()
