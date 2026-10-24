"""
size_study.py -- how detection depends on the object's size in pixels (= distance), independent of the object.

Takes the frames with a verified target box (eval/gt.json), shrinks the whole image by k = 1, 2, 3, 4 (the same
scene with everything k times farther away) and pads it back to 640x480 with mirrored borders, so the detector
always sees a full-size input. Then measures for each configuration whether the top box is the target
(IoU >= 0.5), binned by the target's box height after shrinking.

  full  = whole frame at 640 (V1 / tracking today)
  tiles = whole frame + 2x2 zoomed tiles (SEARCH in laptop_approach.py)
  roi   = whole frame + 2x zoomed crop around the target's previous box (TRACK of a small target); the previous
          box is taken from GT here, on the robot it is the last detection
  1280  = whole frame upscaled to 1280

  python eval/size_study.py --every 2
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from detector import TextDetector, iou

ap = argparse.ArgumentParser()
ap.add_argument("--rec", default="rec")
ap.add_argument("--every", type=int, default=2, help="use every N-th GT frame")
ap.add_argument("--scales", default="1,2,3,4")
args = ap.parse_args()

BINS = [(80, 1e9), (40, 80), (20, 40), (0, 20)]
CONFIGS = {"full": (640, "full"), "tiles": (640, "tiles"), "roi": (640, "roi"), "1280": (1280, "full")}
gt = {k: g for k, g in json.load(open("eval/gt.json")).items()
      if g["box"] is not None and not k.startswith("S1_seq01/")}      # S1_seq01: old test take, not in the dataset
keys = list(gt)[::args.every]
scales = [int(s) for s in args.scales.split(",")]


def shrink(img, box, k):
    h, w = img.shape[:2]
    if k == 1:
        return img, box
    small = cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA)
    top, left = (h - h // k) // 2, (w - w // k) // 2
    out = cv2.copyMakeBorder(small, top, h - h // k - top, left, w - w // k - left, cv2.BORDER_REFLECT)
    return out, [box[0] / k + left, box[1] / k + top, box[2] / k + left, box[3] / k + top]


dets = {}
prompts = {}
stats = {c: defaultdict(lambda: [0, 0]) for c in CONFIGS}
ms = defaultdict(list)
for cname, (imgsz, mode) in CONFIGS.items():
    det = None
    for k in keys:
        seq, fn = k.split("/")
        prompt = prompts.setdefault(seq, json.load(open(os.path.join(args.rec, seq, "meta.json")))["prompt"])
        if det is None or det.prompt != prompt:
            det = TextDetector(prompt, conf=0.10, imgsz=imgsz)
        img = cv2.imread(os.path.join(args.rec, seq, "frames", fn))
        for s in scales:
            im, box = shrink(img, gt[k]["box"], s)
            t = time.perf_counter()
            d = det(im, mode=mode, last_box=box)
            ms[cname].append((time.perf_counter() - t) * 1000)
            h = box[3] - box[1]
            b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= h < hi)
            stats[cname][b][1] += 1
            stats[cname][b][0] += bool(d) and iou(d[0], box) >= 0.5
    print(cname, "done", flush=True)

print("\n| config | ms/frame | " + " | ".join(f"box {int(lo)}-{'' if hi > 1e8 else int(hi)} px" for lo, hi in BINS) + " |")
print("|---" * (2 + len(BINS)) + "|")
res = {}
for c in CONFIGS:
    cells = []
    res[c] = {"ms": float(np.median(ms[c]))}
    for i, (lo, hi) in enumerate(BINS):
        ok, n = stats[c][i]
        res[c][f"{int(lo)}-{int(min(hi, 9999))}"] = (100.0 * ok / n if n else None, n)
        cells.append(f"{100.0 * ok / n:.0f}% (n={n})" if n else "-")
    print(f"| {c} | {res[c]['ms']:.0f} | " + " | ".join(cells) + " |")
json.dump(res, open("eval/size_results.json", "w"), indent=1)
