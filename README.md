# YOLO Degradation Labeller

A local web app that takes your **raw original images**, automatically runs
OCR + degradation, and lets you label each word box as **readable** or
**unreadable** — then saves everything into a YOLO-format dataset with a
70 / 15 / 15 train/val/test split.

---

## Folder structure

```
labeler/
├── app.py
├── requirements.txt
├── queue/          ← DROP YOUR ORIGINAL IMAGES HERE (.jpg or .png)
├── cache/          ← auto-managed (degraded images + OCR data)
└── dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml   ← auto-generated / updated on every save
```

---

## Setup

1. Install dependencies:
   ```
   pip install flask opencv-python pytesseract numpy
   ```

2. Make sure Tesseract is installed.
   Download from: https://github.com/UB-Mannheim/tesseract/wiki
   Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

3. If your Tesseract is somewhere else, edit line 17 of `app.py`:
   ```python
   TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

4. Run:
   ```
   python app.py
   ```
   Then open http://localhost:5000

---

## Workflow

1. Drop your **original** (non-degraded) images into `queue/`
2. Open http://localhost:5000
3. For each image the app will:
   - Run Tesseract OCR on the clean original
   - Apply all degradations (noise, blur, blobs, strokes, etc.)
   - Show the degraded image with green boxes over each detected word
4. Click any box to mark it **red = unreadable**
5. Press **NEXT →** to save and move to the next image

Processed originals are renamed `*_done.jpg` in `queue/` so they won't
reappear if you restart. The cache folder is cleaned automatically.

---

## Output format

Each saved image produces two files:

**`dataset/images/<split>/<stem>.png`** — the degraded image (always PNG)

**`dataset/labels/<split>/<stem>.txt`** — YOLO label file:
```
<class> <x_center> <y_center> <width> <height>
```
- `0` = readable
- `1` = unreadable
- All coordinates normalised to [0, 1]

**`dataset/data.yaml`** — updated after every image:
```yaml
nc: 2
names:
  0: readable
  1: unreadable
```

---

## Split ratios (2000 images)

| Split | Ratio | Count |
|-------|-------|-------|
| train | 70%   | 1400  |
| val   | 15%   | 300   |
| test  | 15%   | 300   |

The split is assigned deterministically — whichever bucket is furthest
below its target gets the next image.
