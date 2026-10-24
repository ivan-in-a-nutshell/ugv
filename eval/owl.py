"""
owl.py -- OWLv2 / OWL-ViT (the proposal's "heavier alternative" detectors) with the same call and output format
as detector.TextDetector, for eval/run_detectors.py. Needs `pip install transformers`; weights download on
first use (OWLv2 base ~0.6 GB).
"""
import cv2
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor, OwlViTForObjectDetection, OwlViTProcessor

MODELS = {
    "owlv2": (Owlv2Processor, Owlv2ForObjectDetection, "google/owlv2-base-patch16-ensemble"),
    "owlvit": (OwlViTProcessor, OwlViTForObjectDetection, "google/owlvit-base-patch32"),
}


class OwlDetector:
    def __init__(self, prompt, kind="owlv2", conf=0.05):
        proc_cls, model_cls, name = MODELS[kind]
        self.prompt, self.kind, self.conf = prompt, kind, conf
        self.proc = proc_cls.from_pretrained(name)
        self.model = model_cls.from_pretrained(name).eval()

    def __call__(self, frame, mode="full", last_box=None):
        h, w = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.proc(text=[[self.prompt]], images=img, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)
        # OWLv2 pads the image to a square before resizing, so boxes are relative to the padded square
        size = (max(h, w), max(h, w)) if self.kind == "owlv2" else (h, w)
        r = self.proc.post_process_grounded_object_detection(out, threshold=self.conf, target_sizes=[size])[0]
        dets = [[*b, s, s, 0.0] for b, s in zip(r["boxes"].tolist(), r["scores"].tolist())]
        return sorted(dets, key=lambda d: -d[4])
