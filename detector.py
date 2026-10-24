"""
detector.py -- text-prompted detector used by laptop_approach.py and the offline evaluation.

Default = what V1 does: YOLO-World with the whole prompt as its one class, on the whole frame. Two additions:

  - zoom for small (far) objects, works for any prompt: objects 5+ m away are only ~10-25 px tall in the
    640x480 stream (100 deg lens, fx ~ 273 px). mode="tiles" also runs 2x2 overlapping tiles, each upscaled to
    the detector input (SEARCH); mode="roi" also runs a 2x zoomed crop around the previous box (TRACK of a small
    target). See eval/size_study.py for the numbers.
  - attribute check (opt-in, attr_check=True): YOLO-World mostly ignores attribute words -- for "a yellow chair"
    it boxes the red chair and the beige armchair. With the check, YOLO-World detects the bare noun ("chair") and
    each crop is kept only if CLIP says it fits the full prompt better than the bare noun
    (sim(crop, "a yellow chair") > sim(crop, "a chair")). Works for chairs, but NOT in general: for "a dark box"
    the distractors (black devices) are dark too and the check removes the real box (eval/README.md). Hence off
    by default; telling one object from look-alikes is V4's job (visual template of the locked object).
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "true")
import re

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLOWorld

ARTICLES = {"a", "an", "the", "some"}


def split_prompt(prompt):
    """'a yellow chair' -> ('chair', True): noun = last word, True if there are words besides article + noun."""
    words = re.findall(r"[a-z]+", prompt.lower())
    content = [w for w in words if w not in ARTICLES]
    return (content[-1] if content else prompt), len(content) > 1


def nms(boxes, thr=0.5):
    """boxes: list of [x1, y1, x2, y2, conf, ...] -> kept boxes, highest conf first."""
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class TextDetector:
    def __init__(self, prompt, model="yolov8s-worldv2.pt", noun=None, conf=0.10, imgsz=640, attr_check=False,
                 clip_weights="weights/clip/ViT-B-32.pt", attr_margin=0.0, max_cands=8):
        self.prompt = prompt
        auto_noun, has_attr = split_prompt(prompt)
        self.noun = noun or auto_noun
        self.conf, self.imgsz, self.attr_margin, self.max_cands = conf, imgsz, attr_margin, max_cands
        self.yolo = YOLOWorld(model)
        self.attr_check = attr_check and has_attr     # nothing to check for a bare noun
        self.yolo.set_classes([self.noun if self.attr_check else prompt])
        if self.attr_check:
            import clip
            path = clip_weights if os.path.exists(clip_weights) else "ViT-B/32"
            self.clip, self.pre = clip.load(path, device="cpu")
            self.clip.eval()
            with torch.no_grad():
                t = self.clip.encode_text(clip.tokenize([prompt, f"a {self.noun}"]))
            self.text_feat = t / t.norm(dim=-1, keepdim=True)

    def _yolo(self, crops):
        """crops: list of (image, x0, y0) -> boxes in full-frame coordinates [x1, y1, x2, y2, det_conf]."""
        out = []
        for (img, x0, y0), r in zip(crops, self.yolo.predict([c[0] for c in crops], conf=self.conf,
                                                             imgsz=self.imgsz, verbose=False)):
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                out.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0, c])
        return out

    @staticmethod
    def tiles(frame, n=2, overlap=0.2):
        h, w = frame.shape[:2]
        tw, th = int(w / (n - (n - 1) * overlap)), int(h / (n - (n - 1) * overlap))
        xs = np.linspace(0, w - tw, n).astype(int)
        ys = np.linspace(0, h - th, n).astype(int)
        return [(frame[y:y + th, x:x + tw], int(x), int(y)) for y in ys for x in xs]

    @staticmethod
    def roi(frame, box, size=(320, 240)):
        """crop of `size` centred on the box (kept inside the image); YOLO upscales it -> ~2x zoom."""
        h, w = frame.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        x0 = int(np.clip(cx - size[0] / 2, 0, w - size[0]))
        y0 = int(np.clip(cy - size[1] / 2, 0, h - size[1]))
        return frame[y0:y0 + size[1], x0:x0 + size[0]], x0, y0

    def attr_gain(self, frame, boxes):
        """CLIP logit of the full prompt minus that of the bare noun, per box (> 0: the description fits better)."""
        if not boxes:
            return []
        h, w = frame.shape[:2]
        ims = []
        for x1, y1, x2, y2, *_ in boxes:
            px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            crop = frame[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]
            ims.append(self.pre(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        with torch.no_grad():
            f = self.clip.encode_image(torch.stack(ims))
        s = 100.0 * (f / f.norm(dim=-1, keepdim=True)) @ self.text_feat.T
        return (s[:, 0] - s[:, 1]).tolist()

    def __call__(self, frame, mode="full", last_box=None):
        """mode: 'full' = whole frame; 'tiles' = whole frame + 2x2 zoomed tiles (search, ~5-7x slower);
        'roi' = whole frame + 2x zoom around last_box (tracking a small/far target, ~2x slower).
        Returns detections [x1, y1, x2, y2, score, det_conf, attr_gain] sorted by score, already thresholded."""
        crops = [(frame, 0, 0)]
        if mode == "tiles":
            crops += self.tiles(frame)
        elif mode == "roi" and last_box is not None:
            crops.append(self.roi(frame, last_box))
        boxes = nms(self._yolo(crops))
        if not self.attr_check:
            return [b + [b[4], 0.0] for b in boxes]
        boxes = boxes[:self.max_cands]       # CLIP cost grows with the number of crops; keep the most confident
        dets = [b[:4] + [b[4], b[4], g] for b, g in zip(boxes, self.attr_gain(frame, boxes)) if g > self.attr_margin]
        return sorted(dets, key=lambda d: -d[4])
