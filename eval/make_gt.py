"""
make_gt.py -- step 2 of the quick ground truth: combines eval/candidates.json with the per-frame choices in
eval/gt_raw.txt (one line per contact sheet, comma separated, in sheet order):
  <n>  = candidate n is the target box
  v    = target visible, but no candidate box fits it (mostly motion blur or the box seen from very close)
  -    = target not in the frame
Writes eval/gt.json: {frame_key: {"visible": bool, "box": [x1, y1, x2, y2] or null}}.

Target definition: the dark box carried in S1-S4/S6; in S5 the dark box, the second (grey, see-through) box is
the distractor. Labelled by one person on 2026-10-24 from the contact sheets in eval/sheets/.
"""
import json

cands = json.load(open("eval/candidates.json"))
labels = [x.strip() for line in open("eval/gt_raw.txt") for x in line.strip().split(",") if line.strip()]
keys = list(cands)
assert len(labels) == len(keys), (len(labels), len(keys))
gt = {}
for k, lab in zip(keys, labels):
    if lab == "-":
        gt[k] = {"visible": False, "box": None}
    elif lab == "v":
        gt[k] = {"visible": True, "box": None}
    else:
        gt[k] = {"visible": True, "box": cands[k]["cands"][int(lab)][:4]}
json.dump(gt, open("eval/gt.json", "w"), indent=0)
n_box = sum(g["box"] is not None for g in gt.values())
n_vis = sum(g["visible"] for g in gt.values())
print(f"{len(gt)} frames: {n_box} with box, {n_vis - n_box} visible without box, {len(gt) - n_vis} not visible")
