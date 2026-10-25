"""
metrics.py -- baseline numbers for the midterm from the recorded V1 runs and the quick ground truth.

1. V1 closed loop (the recordings were made while V1 drove the gimbal):
   target in frame [%], centring error of the target [deg] (mean, p95), near centre (< 10 deg) [%], per scenario.
2. Detector configurations (open loop on the same frames): time per frame on this laptop (CPU), correct lock
   (top box IoU >= 0.5 with GT), wrong lock (top box on something else while the target is visible), false
   alarm (any box while the target is not in the frame).

  python eval/metrics.py            -> prints markdown tables, writes eval/results.json
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import iou

NOT_IN_DATASET = {"S1_seq01"}     # old test take
gt = {k: g for k, g in json.load(open("eval/gt.json")).items() if k.split("/")[0] not in NOT_IN_DATASET}
calib = np.load("calib.npz")
K, dist = calib["K"], calib["dist"]
SCEN = {"S1": "slow", "S2": "fast", "S3": "occlusion", "S4": "leaves frame", "S5": "distractor", "S6": "low light"}
THR = 0.10          # detector threshold used by V1 on the robot


def angle_from_axis(box):
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x, y = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist)[0, 0]
    return math.degrees(math.atan(math.hypot(x, y)))


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


results = {"v1": {}, "detectors": {}}

# ---------------- 1. V1 closed loop ----------------
by_s = defaultdict(list)
for k, g in gt.items():
    by_s[k[:2]].append(g)
by_s["all"] = list(gt.values())
print("### V1 closed loop (recorded runs, 1 labelled frame per 2 s)\n")
print("| scenario | frames | target in frame | centring error mean / p95 | near centre (<10 deg) |")
print("|---|---|---|---|---|")
for s in list(SCEN) + ["all"]:
    gs = by_s[s]
    vis = [g for g in gs if g["visible"]]
    errs = [angle_from_axis(g["box"]) for g in gs if g["box"] is not None]
    r = {"frames": len(gs), "in_frame": pct(len(vis), len(gs)),
         "err_mean": float(np.mean(errs)), "err_p95": float(np.percentile(errs, 95)),
         "near_centre": pct(sum(e < 10 for e in errs), len(errs))}
    results["v1"][s] = r
    label = f"{s} {SCEN[s]}" if s in SCEN else "**all**"
    print(f"| {label} | {r['frames']} | {r['in_frame']:.0f}% | {r['err_mean']:.1f} / {r['err_p95']:.1f} deg "
          f"| {r['near_centre']:.0f}% |")

# ---------------- 2. detectors ----------------
print("\n### Detectors on the same frames (laptop i7-12650H, CPU only)\n")
print("| config | ms/frame median (p95) | correct lock | wrong lock | missed | false alarm (target absent) |"
      " correct lock S5 / S6 |")
print("|---|---|---|---|---|---|---|")
for f in sorted(glob.glob("eval/dets/*.json"), key=os.path.getmtime):
    name = os.path.basename(f)[:-5]
    d = json.load(open(f))
    two_stage = name.endswith("_attr")      # thresholded inside the detector (noun conf 0.05 + CLIP)
    ms = [v["ms"] for v in d.values()]
    n_box = correct = wrong = missed = absent = fa = 0
    per_s = defaultdict(lambda: [0, 0])
    for k, g in gt.items():
        dets = [x for x in d[k]["dets"] if two_stage or x[5] >= THR]
        if not g["visible"]:
            absent += 1
            fa += bool(dets)
            continue
        if g["box"] is None:
            continue                      # visible but no verified box -> can't judge a detection
        n_box += 1
        per_s[k[:2]][1] += 1
        if not dets:
            missed += 1
        elif iou(dets[0], g["box"]) >= 0.5:
            correct += 1
            per_s[k[:2]][0] += 1
        else:
            wrong += 1
    r = {"ms_median": float(np.median(ms)), "ms_p95": float(np.percentile(ms, 95)),
         "correct": pct(correct, n_box), "wrong": pct(wrong, n_box), "missed": pct(missed, n_box),
         "false_alarm": pct(fa, absent),
         "per_scenario_correct": {s: pct(*per_s[s]) for s in SCEN}}
    results["detectors"][name] = r
    print(f"| {name} | {r['ms_median']:.0f} ({r['ms_p95']:.0f}) | {r['correct']:.0f}% | {r['wrong']:.0f}% "
          f"| {r['missed']:.0f}% | {r['false_alarm']:.0f}% | {r['per_scenario_correct']['S5']:.0f}% / "
          f"{r['per_scenario_correct']['S6']:.0f}% |")

n_vis_nobox = sum(g["visible"] and g["box"] is None for g in gt.values())
print(f"\nGT: {len(gt)} frames, {sum(g['box'] is not None for g in gt.values())} with a verified box, "
      f"{n_vis_nobox} visible without box (excluded from detector scores), "
      f"{sum(not g['visible'] for g in gt.values())} target not in frame.")
json.dump(results, open("eval/results.json", "w"), indent=1)
