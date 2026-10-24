"""
candidates.py -- step 1 of the quick ground truth for the recorded sequences.

Samples every N-th frame of each sequence, runs YOLO-World with two settings (s@640 = what V1 runs on the robot,
s@1280 = upscaled input) at a low threshold, merges the boxes and renders contact sheets with numbered candidates.
A person then picks the right candidate per frame (or "not visible") -> eval/gt_labels.json (see make_gt.py).

  python eval/candidates.py --rec rec --step 30
"""
import argparse
import glob
import json
import os

os.environ.setdefault("YOLO_OFFLINE", "true")
import cv2
import numpy as np
from ultralytics import YOLOWorld

ap = argparse.ArgumentParser()
ap.add_argument("--rec", default="rec")
ap.add_argument("--step", type=int, default=30, help="sample every N-th frame (30 = ~2 s at 15 fps)")
ap.add_argument("--offset", type=int, default=15)
ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--out", default="eval/candidates.json")
ap.add_argument("--sheets", default="eval/sheets")
args = ap.parse_args()

model = YOLOWorld("yolov8s-worldv2.pt")
cur_prompt = None


def detect(img, prompt, imgsz):
    global cur_prompt
    if prompt != cur_prompt:
        model.set_classes([prompt])
        cur_prompt = prompt
    r = model.predict(img, conf=args.conf, imgsz=imgsz, verbose=False)[0]
    return [[*map(float, b), float(c)] for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


out = {}
for seq in sorted(glob.glob(os.path.join(args.rec, "S*"))):
    name = os.path.basename(seq)
    prompt = json.load(open(os.path.join(seq, "meta.json")))["prompt"]
    frames = sorted(glob.glob(os.path.join(seq, "frames", "*.jpg")))
    for f in frames[args.offset::args.step]:
        img = cv2.imread(f)
        d640, d1280 = detect(img, prompt, 640), detect(img, prompt, 1280)
        cands = []
        for b in sorted(d640 + d1280, key=lambda b: -b[4]):
            if all(iou(b, c) < 0.5 for c in cands):
                cands.append(b)
        out[f"{name}/{os.path.basename(f)}"] = {"prompt": prompt, "d640": d640, "d1280": d1280, "cands": cands[:8]}
    print(name, flush=True)

json.dump(out, open(args.out, "w"))

# contact sheets: 4x4 tiles, candidates numbered 0..7
os.makedirs(args.sheets, exist_ok=True)
keys = list(out)
for s in range(0, len(keys), 16):
    tiles = []
    for k in keys[s:s + 16]:
        img = cv2.imread(os.path.join(args.rec, k.split("/")[0], "frames", k.split("/")[1]))
        for i, b in enumerate(out[k]["cands"]):
            x1, y1, x2, y2 = map(int, b[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(i), (x1 + 3, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(img, f"{s + len(tiles)}: {k}", (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(cv2.resize(img, (400, 300)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(args.sheets, f"sheet_{s // 16:02d}.jpg"), sheet)
print("frames:", len(keys))
