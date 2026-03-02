"""
app.py  -  YOLO Degradation Labeller (local)
Run:  python app.py
Open: http://localhost:5000
Drop original images into queue/ folder.
"""

import cv2
import json
import math
import random
import os
from pathlib import Path

import numpy as np
import pytesseract
from flask import Flask, render_template, jsonify, request, send_from_directory, abort

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════
# CONFIG  — edit these if needed
# ══════════════════════════════════════════════════════════════════

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_CONF       = 60
TOTAL_IMAGES   = 2000
SPLITS         = ["train", "val", "test"]
SPLIT_RATIOS   = [0.70, 0.15, 0.15]
IMG_EXTS       = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

BASE_DIR    = Path(__file__).parent
QUEUE_DIR   = BASE_DIR / "queue"
CACHE_DIR   = BASE_DIR / "cache"
DATASET_DIR = BASE_DIR / "dataset"
STATE_FILE  = BASE_DIR / "state.json"


# ══════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"done": 0, "split_counts": {"train": 0, "val": 0, "test": 0}}

def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2))

def next_split(counts):
    targets = {s: math.floor(TOTAL_IMAGES * r) for s, r in zip(SPLITS, SPLIT_RATIOS)}
    targets["train"] += TOTAL_IMAGES - sum(targets.values())
    gaps = {s: targets[s] - counts[s] for s in SPLITS}
    return max(gaps, key=gaps.get)


# ══════════════════════════════════════════════════════════════════
# QUEUE
# ══════════════════════════════════════════════════════════════════

def queue_items():
    QUEUE_DIR.mkdir(exist_ok=True)
    seen, items = set(), []
    for ext in IMG_EXTS:
        for p in sorted(QUEUE_DIR.glob(f"*{ext}")):
            stem = p.stem
            if stem.endswith("_done") or stem in seen:
                continue
            seen.add(stem)
            items.append({"stem": stem, "path": p})
    items.sort(key=lambda x: x["stem"])
    return items


# ══════════════════════════════════════════════════════════════════
# DEGRADATION
# ══════════════════════════════════════════════════════════════════

def gaussian_noise(img):
    sigma = random.uniform(5, 25)
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def salt_pepper(img):
    prob = random.uniform(0.01, 0.05)
    out  = img.copy()
    rnd  = np.random.rand(*img.shape[:2])
    out[rnd < prob / 2]     = 0
    out[rnd > 1 - prob / 2] = 255
    return out

def gaussian_blur(img):
    k = random.choice([3, 5, 7, 9, 11])
    return cv2.GaussianBlur(img, (k, k), 0)

def motion_blur(img):
    k      = random.randint(3, 15)
    angle  = random.uniform(0, 180)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    M      = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s      = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)

def brightness_contrast(img):
    alpha = random.uniform(0.5, 1.5)
    beta  = random.randint(-40, 40)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def _perlin_like(h, w, scale=50):
    import math as _m
    x  = np.linspace(0, w / scale, w)
    y  = np.linspace(0, h / scale, h)
    xv, yv = np.meshgrid(x, y)
    phase  = random.uniform(0, 2 * _m.pi)
    freq2  = random.uniform(1.5, 3.0)
    noise  = (np.sin(xv + phase) * np.cos(yv * freq2)
              + np.sin(xv * 2.1 + 1.3) * np.cos(yv * 0.9 + 0.7))
    noise  = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9)
    return noise.astype(np.float32)

def overlay_texture(img):
    alpha   = random.uniform(0.1, 0.4)
    mask    = _perlin_like(img.shape[0], img.shape[1])
    texture = (np.stack([mask] * 3, axis=-1) * 255).astype(np.uint8)
    return cv2.addWeighted(img, 1 - alpha, texture, alpha, 0)

def draw_blobs(img):
    h, w    = img.shape[:2]
    overlay = img.copy()
    for _ in range(random.randint(5, 12)):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(10, 50)
        color = tuple(int(c) for c in np.random.randint(0, 80, 3))
        cv2.circle(overlay, (x, y), r, color, -1)
    a = random.uniform(0.1, 0.4)
    return cv2.addWeighted(img, 1 - a, overlay, a, 0)

def _bezier_pts(p0, p1, p2, p3, n=50):
    pts = []
    for t in np.linspace(0, 1, n):
        mt = 1 - t
        x  = mt**3*p0[0] + 3*mt**2*t*p1[0] + 3*mt*t**2*p2[0] + t**3*p3[0]
        y  = mt**3*p0[1] + 3*mt**2*t*p1[1] + 3*mt*t**2*p2[1] + t**3*p3[1]
        pts.append((int(x), int(y)))
    return pts

def draw_strokes(img):
    h, w    = img.shape[:2]
    overlay = img.copy()
    for _ in range(random.randint(3, 8)):
        p   = [(random.randint(0, w), random.randint(0, h)) for _ in range(4)]
        pts = _bezier_pts(*p)
        color     = tuple(int(c) for c in np.random.randint(0, 100, 3))
        thickness = random.randint(1, 4)
        for k in range(len(pts) - 1):
            cv2.line(overlay, pts[k], pts[k + 1], color, thickness, cv2.LINE_AA)
    a = random.uniform(0.1, 0.35)
    return cv2.addWeighted(img, 1 - a, overlay, a, 0)

def degrade(img):
    img = gaussian_noise(img)
    img = salt_pepper(img)
    img = gaussian_blur(img)
    img = motion_blur(img)
    img = brightness_contrast(img)
    img = overlay_texture(img)
    img = draw_blobs(img)
    img = draw_strokes(img)
    return img


# ══════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════

def extract_boxes(img, img_w, img_h):
    data  = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word or int(data["conf"][i]) < OCR_CONF:
            continue
        x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if bw <= 0 or bh <= 0:
            continue
        boxes.append({
            "cls":   0,
            "xc":    round((x + bw / 2) / img_w, 6),
            "yc":    round((y + bh / 2) / img_h, 6),
            "w":     round(bw / img_w, 6),
            "h":     round(bh / img_h, 6),
            "label": "readable",
        })
    return boxes


# ══════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════

def get_cached(stem, src_path):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_img  = CACHE_DIR / f"{stem}.png"
    cache_meta = CACHE_DIR / f"{stem}.json"

    if cache_img.exists() and cache_meta.exists():
        meta = json.loads(cache_meta.read_text())
        return {"boxes": meta["boxes"]}

    img = cv2.imread(str(src_path))
    if img is None:
        raise ValueError(f"Cannot read: {src_path}")

    h, w  = img.shape[:2]
    boxes    = extract_boxes(img, w, h)
    degraded = degrade(img.copy())

    cv2.imwrite(str(cache_img), degraded)
    cache_meta.write_text(json.dumps({"boxes": boxes}))

    return {"boxes": boxes}


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    state = load_state()
    items = queue_items()
    return jsonify({
        "done":         state["done"],
        "total":        TOTAL_IMAGES,
        "remaining":    len(items),
        "split_counts": state["split_counts"],
    })

@app.route("/api/next")
def api_next():
    items = queue_items()
    if not items:
        return jsonify({"done": True})

    item = items[0]
    stem = item["stem"]

    try:
        cached = get_cached(stem, item["path"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "done":      False,
        "stem":      stem,
        "img_url":   f"/cache_img/{stem}.png",
        "boxes":     cached["boxes"],
        "remaining": len(items),
    })

@app.route("/cache_img/<filename>")
def cache_img_route(filename):
    return send_from_directory(CACHE_DIR, filename)

@app.route("/api/submit", methods=["POST"])
def api_submit():
    data  = request.get_json()
    stem  = data["stem"]
    boxes = data["boxes"]

    state = load_state()
    split = next_split(state["split_counts"])

    label_lines = []
    for b in boxes:
        cls = 1 if b["label"] == "unreadable" else 0
        label_lines.append(
            f"{cls} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}"
        )

    img_dst   = DATASET_DIR / "images" / split / f"{stem}.png"
    label_dst = DATASET_DIR / "labels" / split / f"{stem}.txt"

    cache_img = CACHE_DIR / f"{stem}.png"
    if not cache_img.exists():
        abort(404, "Cached image not found")

    import shutil
    shutil.copy2(cache_img, img_dst)
    label_dst.write_text("\n".join(label_lines))

    # Mark original as done
    src = item_path = None
    for ext in IMG_EXTS:
        p = QUEUE_DIR / f"{stem}{ext}"
        if p.exists():
            src = p
            break
    if src:
        src.rename(QUEUE_DIR / f"{stem}_done{src.suffix}")

    # Clean cache
    cache_img.unlink(missing_ok=True)
    (CACHE_DIR / f"{stem}.json").unlink(missing_ok=True)

    state["done"] += 1
    state["split_counts"][split] += 1
    save_state(state)

    yaml_content = (
        "path: dataset\ntrain: images/train\nval:   images/val\n"
        "test:  images/test\n\nnc: 2\nnames:\n  0: readable\n  1: unreadable\n"
    )
    (DATASET_DIR / "data.yaml").write_text(yaml_content)

    return jsonify({"ok": True, "split": split, "done": state["done"]})

if __name__ == "__main__":
    for d in [QUEUE_DIR, CACHE_DIR,
              DATASET_DIR / "images" / "train",
              DATASET_DIR / "images" / "val",
              DATASET_DIR / "images" / "test",
              DATASET_DIR / "labels" / "train",
              DATASET_DIR / "labels" / "val",
              DATASET_DIR / "labels" / "test"]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"\n  Labeller running ->  http://localhost:5000")
    print(f"  Drop original images into: {QUEUE_DIR}\n")
    app.run(debug=True, port=5000)