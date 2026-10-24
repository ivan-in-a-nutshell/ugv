"""
tracker.py -- small template tracker that bridges detector misses (a light first step toward V2).

The detector flickers on some objects (score around the threshold) and especially up close; a few missed frames
were enough to lose the target. After every real detection we keep the target's image patch; when the detector
misses, we look for that patch near its last position with normalised cross-correlation at three scales.
It is refreshed only by real detections, never by its own output, so it cannot slowly drift onto the background;
laptop_approach.py also limits how many frames in a row may come from the tracker alone.
(The proposal's CSRT tracker needs opencv-contrib, which this OpenCV build does not have.)
"""
import cv2
import numpy as np


class TemplateTracker:
    def __init__(self, min_score=0.6, search=1.5, scales=(0.9, 1.0, 1.1)):
        self.min_score, self.search, self.scales = min_score, search, scales
        self.tmpl = None
        self.box = None

    def reset(self):
        self.tmpl = self.box = None

    def init(self, frame, box):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            self.reset()
            return
        self.tmpl = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        self.box = [x1, y1, x2, y2]

    def update(self, frame):
        """-> (box [x1, y1, x2, y2], score) or (None, best score) if the patch is not found well enough."""
        if self.tmpl is None:
            return None, 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        x1, y1, x2, y2 = self.box
        bw, bh = x2 - x1, y2 - y1
        # search window: the last box grown by `search` box sizes (+ 40 px) on every side
        mx, my = int(self.search * bw) + 40, int(self.search * bh) + 40
        sx1, sy1, sx2, sy2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
        win = gray[sy1:sy2, sx1:sx2]
        best = (-1.0, None)
        for s in self.scales:
            t = cv2.resize(self.tmpl, None, fx=s, fy=s) if s != 1.0 else self.tmpl
            th, tw = t.shape
            if th >= win.shape[0] or tw >= win.shape[1] or th < 8 or tw < 8:
                continue
            r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (px, py) = cv2.minMaxLoc(r)
            if score > best[0]:
                best = (score, [sx1 + px, sy1 + py, sx1 + px + tw, sy1 + py + th])
        if best[1] is None or best[0] < self.min_score:
            return None, max(best[0], 0.0)
        return best[1], best[0]
