# SKU Annotation Tool (local)

Upload a shelf photo -> RF-DETR (`model/sku_and_price_tag.pth`) auto-detects known
SKUs/price tags -> click any it missed and SAM3 turns each click into a tight box ->
reject any false positives -> Submit writes everything into `dataset/` in the exact
YOLO layout `rfdetr_sagemaker` trains on (see `/Users/satvikchaudhary/Downloads/rfdetr_sagemaker`).

Runs fully locally (Apple Silicon, MPS). No cloud/SageMaker involved in this phase.

## One-time setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
```

### Click-to-segment backend: SAM2 (active) vs SAM3 (pending access)

`backend/config.py` -> `SEGMENTATION_BACKEND` picks which model powers "Add missed
object": `"sam2"` (**current default**) or `"sam3"`. Both implement the same
interface (`backend/segmentation.py` is the facade `main.py` imports), so switching
is a one-line change plus a server restart - nothing else in the app changes.

**Why SAM2 for now:** `facebook/sam3` is a gated Hugging Face model - access was
requested but is still awaiting manual review from Meta (no official SLA; could be
hours or days). SAM2 (`facebook/sam2.1-hiera-small`) is ungated, downloads
immediately (~180 MB), and is confirmed fast on this Apple Silicon Mac: first click
on a new image takes ~2s (one-time image encode), every click after that on the
same image is ~0.1s (cached image embeddings, same "encode once, click many times"
pattern SAM2/SAM3 both use).

**Switching back to SAM3** once https://huggingface.co/facebook/sam3 access is
approved:
1. `hf auth login` with a token from the approved account.
2. In `backend/config.py`, set `SEGMENTATION_BACKEND = "sam3"`.
3. Restart the server. First `/api/segment` call downloads the ~3.4 GB checkpoint.

RF-DETR detection works immediately regardless of any of this - only the
click-to-segment step depends on the chosen backend. If a backend's checkpoint
isn't reachable, `/api/segment` returns a clear 503 error explaining what to do;
everything else in the tool still works.

## Run

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

Open http://127.0.0.1:8010 in a browser.

## How to use

1. Drag an image onto the page (or click to pick a file). RF-DETR's detections are
   drawn immediately - blue = `sku`, green = `price_tag`.
2. Click any RF-DETR box to reject it as a false positive (it won't be saved).
3. Click **"Add missed object"**, then click on a SKU/price tag RF-DETR missed. The
   mask appears live; shift+click adds a negative point if the mask needs excluding
   a region. Pick the class in the small popup and **Confirm** (or **Cancel** to
   discard). Repeat for every missed object.
4. Pick the split (`train`/`valid`/`test`) at the top.
5. **Submit** - writes the image + YOLO label file into `dataset/<split>/...` and
   resets the page for the next image.

## Output format

```
dataset/
  data.yaml           # nc: 2, names: [sku, price_tag]
  train/{images,labels}/
  valid/{images,labels}/
  test/{images,labels}/
```

Each `.txt` label line: `class_id cx cy w h` (normalized 0-1, cxcywh) - identical to
`rfdetr_sagemaker/data/*/labels/*.txt`. Verified with
`rfdetr.datasets.yolo.is_valid_yolo_dataset("dataset")`.

## Notes / known limitations (this phase)

- Single-user, single-machine tool - no auth, no concurrent-session handling.
- Both SAM2 and SAM3 run via the Hugging Face `transformers` integration
  (`Sam2Model`/`Sam2Processor`, `Sam3TrackerModel`/`Sam3TrackerProcessor`), not
  SAM3's native `facebookresearch/sam3` repo, which requires CUDA and does not run
  on Apple Silicon at all.
- The per-image encode is the slow step on first click for a given image; it runs
  once in the background right after upload so it's usually ready by the time you
  start clicking. Each click after that reuses the cached embedding (~0.1s with
  SAM2, confirmed on this machine; SAM3's per-image encode time is unverified on
  Apple Silicon but expected to be noticeably slower, see the project plan).
- `backend/work/<image_id>/` accumulates one folder per uploaded image and is never
  auto-cleaned - harmless (gitignored, small files) but worth clearing out
  (`rm -rf backend/work/*`) if it grows large during a long annotation session.
- `backend/_detect_ckpt.py` is the one-off spike script used to determine that
  `sku_and_price_tag.pth` is an `RFDETRLarge` checkpoint with `num_classes=2`
  (`sku`=0, `price_tag`=1) - kept for reference, not part of the running app.
