const COLORS = {
  sku: "#0a84ff",
  price_tag: "#30d158",
  human: "#ff9f0a",
};

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const canvasWrap = document.getElementById("canvas-wrap");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clickPanel = document.getElementById("click-panel");
const classSelect = document.getElementById("class-select");
const confirmBtn = document.getElementById("confirm-btn");
const cancelBtn = document.getElementById("cancel-btn");
const addModeBtn = document.getElementById("add-mode-btn");
const drawModeBtn = document.getElementById("draw-mode-btn");
const undoBtn = document.getElementById("undo-btn");
const discardBtn = document.getElementById("discard-btn");
const submitBtn = document.getElementById("submit-btn");
const splitSelect = document.getElementById("split-select");
const countsEl = document.getElementById("counts");
const hintEl = document.getElementById("hint");
const toastEl = document.getElementById("toast");

let state = null;

function freshState() {
  return {
    imageId: null,
    img: null,
    detections: [], // {id, class_id, class_name, bbox_xyxy, score, rejected}
    pending: [], // {class_id, bbox_xyxy, source:"human"|"manual", removed}
    mode: "none", // "none" | "sam" (click to segment) | "draw" (manual rectangle)
    currentPoints: [],
    currentMaskImg: null,
    currentBbox: null,
    dragStart: null, // in-progress manual box CREATION drag (mode === "draw")
    handleDrag: null, // in-progress resize/move of the already-proposed currentBbox
    suppressClick: false,
  };
}

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  setTimeout(() => toastEl.classList.remove("show"), 2500);
}

// ---------- upload ----------

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag");
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  dropZone.textContent = "Uploading + running RF-DETR...";

  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) {
    showToast("Upload failed: " + (await res.text()));
    dropZone.textContent = "Click or drag an image here to upload";
    return;
  }
  const data = await res.json();

  state = freshState();
  state.imageId = data.image_id;
  state.detections = data.detections.map((d) => ({ ...d, rejected: false }));

  const img = new Image();
  img.onload = () => {
    state.img = img;
    canvas.width = data.width;
    canvas.height = data.height;
    dropZone.style.display = "none";
    canvasWrap.style.display = "block";
    hintEl.style.display = "block";
    addModeBtn.disabled = false;
    drawModeBtn.disabled = false;
    submitBtn.disabled = false;
    discardBtn.disabled = false;
    redraw();
    updateCounts();
  };
  img.src = data.image_url;
}

// ---------- drawing ----------

// Canvas-space units per CSS/screen pixel. All line widths, font sizes, and
// handle sizes below are specified in DESIRED SCREEN pixels and multiplied by
// this, so they render at a consistent visible size regardless of how large
// the source image is or how much CSS shrinks it to fit the viewport - sizing
// them as a fraction of canvas.width (the old approach) meant a heavily
// downscaled image (e.g. a tall phone photo capped by max-height) ended up
// with sub-pixel, invisible lines/text.
function displayScale() {
  const rect = canvas.getBoundingClientRect();
  return rect.width > 0 ? canvas.width / rect.width : 1;
}

function drawBox(bbox, color, label, dashed) {
  const [x1, y1, x2, y2] = bbox;
  const s = displayScale();
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * s;
  if (dashed) ctx.setLineDash([8 * s, 6 * s]);
  ctx.globalAlpha = dashed ? 0.55 : 0.95;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  if (label) {
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    const fontSize = 13 * s;
    ctx.font = `${fontSize}px sans-serif`;
    const textW = ctx.measureText(label).width;
    ctx.fillRect(x1, Math.max(0, y1 - fontSize - 4 * s), textW + 8 * s, fontSize + 4 * s);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x1 + 4 * s, Math.max(fontSize, y1 - 4 * s));
  }
  ctx.restore();
}

function handlePositions(bbox) {
  const [x1, y1, x2, y2] = bbox;
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  return {
    nw: [x1, y1], n: [mx, y1], ne: [x2, y1],
    w: [x1, my], e: [x2, my],
    sw: [x1, y2], s: [mx, y2], se: [x2, y2],
  };
}

const HANDLE_CURSORS = {
  nw: "nwse-resize", se: "nwse-resize",
  ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize",
  e: "ew-resize", w: "ew-resize",
};

function drawHandles(bbox) {
  const r = 10 * displayScale();
  ctx.save();
  ctx.fillStyle = "#ffd60a";
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;
  for (const [hx, hy] of Object.values(handlePositions(bbox))) {
    ctx.fillRect(hx - r / 2, hy - r / 2, r, r);
    ctx.strokeRect(hx - r / 2, hy - r / 2, r, r);
  }
  ctx.restore();
}

function redraw() {
  if (!state || !state.img) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);

  for (const d of state.detections) {
    if (d.rejected) continue; // removed, not just marked - no box drawn at all
    drawBox(d.bbox_xyxy, COLORS[d.class_name] || "#888", `${d.class_name} ${d.score.toFixed(2)}`, false);
  }

  for (const p of state.pending) {
    if (p.removed) continue; // removed, not just marked - no box drawn at all
    const className = p.class_id === 0 ? "sku" : "price_tag";
    drawBox(p.bbox_xyxy, COLORS.human, `+ ${className}`, false);
  }

  if (state.currentMaskImg) {
    ctx.drawImage(state.currentMaskImg, 0, 0, canvas.width, canvas.height);
  }

  if (state.currentBbox) {
    drawBox(state.currentBbox, "#ffd60a", null, true);
    if (!state.dragStart) drawHandles(state.currentBbox);
  }

  for (const pt of state.currentPoints) {
    const s = displayScale();
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, 6 * s, 0, 2 * Math.PI);
    ctx.fillStyle = pt.label === 1 ? "#30d158" : "#ff453a";
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5 * s;
    ctx.fill();
    ctx.stroke();
  }
}

function updateCounts() {
  const kept = state.detections.filter((d) => !d.rejected).length;
  const rejected = state.detections.filter((d) => d.rejected).length;
  const activeNew = state.pending.filter((p) => !p.removed).length;
  countsEl.textContent = `${kept} kept, ${rejected} rejected, ${activeNew} new`;
  undoBtn.disabled = state.pending.length === 0;
}

// ---------- canvas interaction ----------

function canvasPointFromEvent(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function boxContains(bbox, x, y) {
  return x >= bbox[0] && x <= bbox[2] && y >= bbox[1] && y <= bbox[3];
}

function normalizeBbox(x1, y1, x2, y2) {
  return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
}

function handleHitRadius() {
  return 9 * displayScale(); // ~9 CSS px hit target
}

function hitTestHandle(bbox, x, y) {
  const r = handleHitRadius();
  let best = null;
  let bestDist = Infinity;
  for (const [key, [hx, hy]] of Object.entries(handlePositions(bbox))) {
    const d = Math.hypot(x - hx, y - hy);
    if (d <= r && d < bestDist) {
      best = key;
      bestDist = d;
    }
  }
  return best;
}

const MIN_DRAG_SIZE = 4; // image-space pixels; ignore accidental micro-drags/clicks

canvas.addEventListener("click", (e) => {
  if (!state || state.mode === "draw") return; // draw mode is handled by mousedown/move/up below
  if (state.suppressClick) {
    state.suppressClick = false; // swallow the click that follows a resize/move drag
    return;
  }
  const { x, y } = canvasPointFromEvent(e);

  if (state.mode !== "sam") {
    // pending (human-added) boxes are drawn on top, so hit-test them first
    for (let i = state.pending.length - 1; i >= 0; i--) {
      if (boxContains(state.pending[i].bbox_xyxy, x, y)) {
        state.pending[i].removed = !state.pending[i].removed;
        redraw();
        updateCounts();
        return;
      }
    }
    for (let i = state.detections.length - 1; i >= 0; i--) {
      if (boxContains(state.detections[i].bbox_xyxy, x, y)) {
        state.detections[i].rejected = !state.detections[i].rejected;
        redraw();
        updateCounts();
        return;
      }
    }
    return;
  }

  const label = e.shiftKey ? 0 : 1;
  state.currentPoints.push({ x, y, label });
  segmentCurrentPoints();
});

// ---------- manual box drawing + resize/move of the pending box ----------
//
// mousedown starts on the canvas (interactions only begin over the image);
// mousemove/mouseup are attached to window so a drag that leaves the canvas
// bounds (e.g. resizing near an edge) keeps tracking instead of getting stuck.

canvas.addEventListener("mousedown", (e) => {
  if (!state) return;
  const { x, y } = canvasPointFromEvent(e);

  // Resize/move takes priority over starting anything new, in any mode.
  if (state.currentBbox && !state.dragStart) {
    const handle = hitTestHandle(state.currentBbox, x, y);
    if (handle) {
      state.handleDrag = { type: "resize", handle };
      return;
    }
    if (boxContains(state.currentBbox, x, y)) {
      state.handleDrag = { type: "move", startX: x, startY: y, startBbox: [...state.currentBbox] };
      return;
    }
  }

  if (state.mode === "draw") {
    state.dragStart = { x, y };
    state.currentBbox = [x, y, x, y];
    redraw();
  }
});

window.addEventListener("mousemove", (e) => {
  if (!state) return;
  const { x, y } = canvasPointFromEvent(e);

  if (state.handleDrag) {
    applyHandleDrag(state.handleDrag, x, y);
    redraw();
    return;
  }
  if (state.mode === "draw" && state.dragStart) {
    state.currentBbox = normalizeBbox(state.dragStart.x, state.dragStart.y, x, y);
    redraw();
    return;
  }

  // hover feedback only - no active drag
  if (!state.currentBbox) {
    canvas.style.cursor = "crosshair";
  } else {
    const handle = hitTestHandle(state.currentBbox, x, y);
    canvas.style.cursor = handle
      ? HANDLE_CURSORS[handle]
      : boxContains(state.currentBbox, x, y)
      ? "move"
      : "crosshair";
  }
});

window.addEventListener("mouseup", (e) => {
  if (!state) return;
  const { x, y } = canvasPointFromEvent(e);

  if (state.handleDrag) {
    state.handleDrag = null;
    state.currentBbox = normalizeBbox(...state.currentBbox);
    state.suppressClick = true;
    redraw();
    positionClickPanel(state.currentBbox);
    return;
  }

  if (state.mode === "draw" && state.dragStart) {
    const bbox = normalizeBbox(state.dragStart.x, state.dragStart.y, x, y);
    state.dragStart = null;

    if (bbox[2] - bbox[0] < MIN_DRAG_SIZE || bbox[3] - bbox[1] < MIN_DRAG_SIZE) {
      state.currentBbox = null;
      redraw();
      return;
    }
    state.currentBbox = bbox;
    redraw();
    positionClickPanel(bbox);
  }
});

function applyHandleDrag(drag, x, y) {
  if (drag.type === "move") {
    const dx = x - drag.startX;
    const dy = y - drag.startY;
    const [sx1, sy1, sx2, sy2] = drag.startBbox;
    state.currentBbox = [sx1 + dx, sy1 + dy, sx2 + dx, sy2 + dy];
    return;
  }
  let [x1, y1, x2, y2] = state.currentBbox;
  const h = drag.handle;
  if (h.includes("n")) y1 = y;
  if (h.includes("s")) y2 = y;
  if (h.includes("w")) x1 = x;
  if (h.includes("e")) x2 = x;
  state.currentBbox = [x1, y1, x2, y2];
}

async function segmentCurrentPoints() {
  const res = await fetch("/api/segment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_id: state.imageId,
      points: state.currentPoints.map((p) => ({ x: p.x, y: p.y, label: p.label })),
    }),
  });
  if (!res.ok) {
    showToast("Segment failed: " + (await res.text()));
    return;
  }
  const data = await res.json();
  state.currentBbox = data.bbox_xyxy;

  const maskImg = new Image();
  maskImg.onload = () => {
    state.currentMaskImg = maskImg;
    redraw();
    positionClickPanel(data.bbox_xyxy);
  };
  maskImg.src = "data:image/png;base64," + data.mask_png_base64;
}

function positionClickPanel(bbox) {
  const scale = canvas.getBoundingClientRect().width / canvas.width;
  clickPanel.style.left = `${bbox[2] * scale + 8}px`;
  clickPanel.style.top = `${bbox[1] * scale}px`;
  clickPanel.classList.remove("hidden");
}

function resetCurrentClick() {
  state.currentPoints = [];
  state.currentMaskImg = null;
  state.currentBbox = null;
  state.dragStart = null;
  state.handleDrag = null;
  state.suppressClick = false;
  clickPanel.classList.add("hidden");
  redraw();
}

confirmBtn.addEventListener("click", () => {
  if (!state.currentBbox) return;
  state.pending.push({
    class_id: parseInt(classSelect.value, 10),
    bbox_xyxy: state.currentBbox,
    source: state.mode === "draw" ? "manual" : "human",
    removed: false,
  });
  resetCurrentClick();
  updateCounts();
});

cancelBtn.addEventListener("click", resetCurrentClick);

function setMode(newMode) {
  state.mode = state.mode === newMode ? "none" : newMode;
  addModeBtn.classList.toggle("active", state.mode === "sam");
  drawModeBtn.classList.toggle("active", state.mode === "draw");
  resetCurrentClick();
}

addModeBtn.addEventListener("click", () => setMode("sam"));
drawModeBtn.addEventListener("click", () => setMode("draw"));

undoBtn.addEventListener("click", () => {
  state.pending.pop();
  redraw();
  updateCounts();
});

// ---------- submit / discard ----------

function resetToUploadScreen() {
  state = null;
  dropZone.style.display = "flex";
  dropZone.textContent = "Click or drag an image here to upload";
  canvasWrap.style.display = "none";
  hintEl.style.display = "none";
  clickPanel.classList.add("hidden");
  addModeBtn.disabled = true;
  addModeBtn.classList.remove("active");
  drawModeBtn.disabled = true;
  drawModeBtn.classList.remove("active");
  submitBtn.disabled = true;
  discardBtn.disabled = true;
  undoBtn.disabled = true;
  countsEl.textContent = "";
  fileInput.value = "";
}

submitBtn.addEventListener("click", async () => {
  if (!state) return;
  const boxes = state.detections
    .filter((d) => !d.rejected)
    .map((d) => ({ class_id: d.class_id, bbox_xyxy: d.bbox_xyxy, source: "rfdetr" }))
    .concat(
      state.pending
        .filter((p) => !p.removed)
        .map((p) => ({ class_id: p.class_id, bbox_xyxy: p.bbox_xyxy, source: p.source }))
    );

  const res = await fetch("/api/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_id: state.imageId,
      split: splitSelect.value,
      boxes,
    }),
  });
  if (!res.ok) {
    showToast("Submit failed: " + (await res.text()));
    return;
  }
  const data = await res.json();
  showToast(`Saved: ${data.kept_from_rfdetr} kept + ${data.added_by_human} new -> ${data.label_path}`);
  resetToUploadScreen();
});

discardBtn.addEventListener("click", async () => {
  if (!state) return;
  const hasWork = state.pending.length > 0 || state.detections.some((d) => d.rejected);
  if (hasWork && !confirm("Discard this image and all annotations on it? This cannot be undone.")) {
    return;
  }

  const imageId = state.imageId;
  resetToUploadScreen();
  try {
    await fetch("/api/discard", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId }),
    });
  } catch (e) {
    // best-effort server-side cleanup; the UI has already moved on
  }
});
