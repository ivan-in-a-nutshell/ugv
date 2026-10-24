"""
run_detectors.py -- runs several detector configurations on the ground-truth frames (eval/gt.json) and stores all
boxes and the time per frame, for metrics.py.

  python eval/run_detectors.py                       # all configs
  python eval/run_detectors.py --configs s640 m640
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("YOLO_OFFLINE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import torch

from detector import TextDetector

CONFIGS = {
    # name: (model size, imgsz, attribute check (noun + CLIP), mode)
    "s640": ("s", 640, False, "full"),       # = V1 on the robot
    "s1280": ("s", 1280, False, "full"),
    "s640_tiles": ("s", 640, False, "tiles"),
    "m640": ("m", 640, False, "full"),
    "l640": ("l", 640, False, "full"),
    "x640": ("x", 640, False, "full"),
    "s640_attr": ("s", 640, True, "full"),
    "owlvit": ("owlvit", None, False, "full"),    # the proposal's heavier alternatives (eval/owl.py)
    "owlv2": ("owlv2", None, False, "full"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--rec", default="rec")
ap.add_argument("--gt", default="eval/gt.json")
ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
ap.add_argument("--out", default="eval/dets")
args = ap.parse_args()

torch.set_num_threads(os.cpu_count())
gt = json.load(open(args.gt))
prompts = {}
os.makedirs(args.out, exist_ok=True)
for name in args.configs:
    size, imgsz, attr, mode = CONFIGS[name]
    det = None
    res = {}
    for k in gt:
        seq = k.split("/")[0]
        prompt = prompts.setdefault(seq, json.load(open(os.path.join(args.rec, seq, "meta.json")))["prompt"])
        if det is None or det.prompt != prompt:
            if size.startswith("owl"):
                from owl import OwlDetector
                det = OwlDetector(prompt, kind=size, conf=0.05)
            else:
                det = TextDetector(prompt, model=f"yolov8{size}-worldv2.pt", attr_check=attr, conf=0.05, imgsz=imgsz)
            det(cv2.imread(os.path.join(args.rec, seq, "frames", k.split("/")[1])), mode=mode)   # warm-up
        img = cv2.imread(os.path.join(args.rec, seq, "frames", k.split("/")[1]))
        t = time.perf_counter()
        d = det(img, mode=mode)
        res[k] = {"ms": (time.perf_counter() - t) * 1000.0, "dets": d}
    json.dump(res, open(os.path.join(args.out, f"{name}.json"), "w"))
    print(name, "done", flush=True)
