"""
app.py  -  YOLO Degradation Labeller (Cloudflare R2 + Render deployment)
"""

import cv2
import json
import math
import random
import io
import os
from pathlib import Path

import numpy as np
import pytesseract
import boto3
from botocore.config import Config
from flask import Flask, render_template, jsonify, request, send_file, abort

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

TESSERACT_PATH = os.environ.get("TESSERACT_PATH", "/usr/bin/tesseract")
OCR_CONF       = 60
TOTAL_IMAGES   = 2000
SPLITS         = ["train", "val", "test"]
SPLIT_RATIOS   = [0.70, 0.15, 0.15]
IMG_EXTS       = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY = os.environ["R2_ACCESS_KEY"]
R2_SECRET_KEY = os.environ["R2_SECRET_KEY"]
R2_BUCKET     = os.environ.get("R2_BUCKET", "labeler")

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="auto",
)

STATE_KEY = "state.json"


# ══════════════════════════════════════════════════════════════════
# R2 HELPERS
# ══════════════════════════════════════════════════════════════════

def r2_read_bytes(key):
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return obj["Body"].read()
    except Exception:
        return None

def r2_write_bytes(key, data, content_type="application/octet-stream"):
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType=content_type)

def r2_write_text(key, text):
    r2_write_bytes(key, text.encode(), "text/plain")

def r2_read_text(key):
    b = r2_read_bytes(key)
    return b.decode() if b is not None else None

def r2_delete(key):
    try:
        s3.delete_object(Bucket=R2_BUCKET, Key=key)
    except Exception:
        pass

def r2_list_prefix(prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=R2_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys

def r2_exists(key):
    try:
        s3.head_object(Bucket=R2_BUCKET, Key=key)
        return True
    except Exception:
        return False

def np_to_png_bytes(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return buf.tobytes()

def bytes_to_np(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ══════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════

def load_state():
    text = r2_read_text(STATE_KEY)
    if text:
        return json.loads(text)
    return {"done": 0, "split_counts": {"train": 0, "val": 0, "test": 0}}

def save_state(s):
    r2_write_text(STATE_KEY, json.dumps(s, indent=2))

def next_split(counts):
    targets = {s: math.floor(TOTAL_IMAGES * r) for s, r in zip(SPLITS, SPLIT_RATIOS)}
    targets["train"] += TOTAL_IMAGES - sum(targets.values())
    gaps = {s: targets[s] - counts[s] for s in SPLITS}
    return max(gaps, key=gaps.get)


# ══════════════════════════════════════════════════════════════════
# QUEUE
# ══════════════════════════════════════════════════════════════════

def queue_items():
    keys = r2_list_prefix("queue/")
    seen, items = set(), []
    for key in sorted(keys):
        fname = key.split("/")[-1]
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in [e.lower() for e in IMG_EXTS]:
            continue
        if stem.endswith("_done") or stem in seen:
            continue
        seen.add(stem)
        items.append({"stem": stem, "key": key})
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

def get_cached(stem, queue_key):
    cache_img_key  = f"cache/{stem}.png"
    cache_meta_key = f"cache/{stem}.json"

    if r2_exists(cache_img_key) and r2_exists(cache_meta_key):
        meta = json.loads(r2_read_text(cache_meta_key))
        return {"boxes": meta["boxes"]}

    raw = r2_read_bytes(queue_key)
    if raw is None:
        raise ValueError(f"Cannot read {queue_key} from R2")

    img = bytes_to_np(raw)
    if img is None:
        raise ValueError(f"Could not decode image: {queue_key}")

    h, w  = img.shape[:2]
    boxes    = extract_boxes(img, w, h)
    degraded = degrade(img.copy())

    r2_write_bytes(cache_img_key, np_to_png_bytes(degraded), "image/png")
    r2_write_text(cache_meta_key, json.dumps({"boxes": boxes}))

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
        cached = get_cached(stem, item["key"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({
        "done":      False,
        "stem":      stem,
        "img_url":   f"/cache_img/{stem}.png",
        "boxes":     cached["boxes"],
        "remaining": len(items),
    })

@app.route("/cache_img/<stem>.png")
def cache_img_route(stem):
    data = r2_read_bytes(f"cache/{stem}.png")
    if data is None:
        abort(404)
    return send_file(io.BytesIO(data), mimetype="image/png")

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

    img_data = r2_read_bytes(f"cache/{stem}.png")
    if img_data is None:
        abort(404, "Cached image not found")

    r2_write_bytes(f"dataset/images/{split}/{stem}.png", img_data, "image/png")
    r2_write_text(f"dataset/labels/{split}/{stem}.txt",  "\n".join(label_lines))

    # Mark original as done
    for ext in IMG_EXTS:
        k = f"queue/{stem}{ext}"
        if r2_exists(k):
            raw = r2_read_bytes(k)
            r2_write_bytes(f"queue/{stem}_done{ext}", raw)
            r2_delete(k)
            break

    r2_delete(f"cache/{stem}.png")
    r2_delete(f"cache/{stem}.json")

    state["done"] += 1
    state["split_counts"][split] += 1
    save_state(state)

    r2_write_text("dataset/data.yaml",
        "path: dataset\ntrain: images/train\nval:   images/val\n"
        "test:  images/test\n\nnc: 2\nnames:\n  0: readable\n  1: unreadable\n"
    )

    return jsonify({"ok": True, "split": split, "done": state["done"]})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
