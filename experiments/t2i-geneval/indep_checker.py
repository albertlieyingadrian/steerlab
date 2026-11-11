"""An independent GenEval-style checker, for picking candidates without peeking at the test grader.

Same written rules as GenEval (count, colour, left/right/above/below with a 0.1 size margin), but different models:
OWLv2 (open-vocabulary detector, not Mask2Former) finds the objects and OpenCLIP ViT-H-14 (LAION-2B, not OpenAI
ViT-L-14) names their colour from the box crop. If the harness only gains on the official grader when it picks with
that grader, the gain is the grader's taste, not better images; this checker is the control for that.
"""

import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
DET_THRESHOLD = 0.25
NMS_IOU = 0.5
POSITION_THRESHOLD = 0.1


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def relative_position(box_a, box_b):
    boxes = np.array([box_a, box_b])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised) < 1e-3):
        return set()
    dx, dy = revised / np.linalg.norm(offset)
    rels = set()
    if dx < -0.5:
        rels.add("left of")
    if dx > 0.5:
        rels.add("right of")
    if dy < -0.5:
        rels.add("above")
    if dy > 0.5:
        rels.add("below")
    return rels


class IndepChecker:
    def __init__(self, device="cuda"):
        self.device = device
        self.proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.det = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval().to(device)
        self.clip, _, self.tf = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
        self.clip = self.clip.eval().to(device)
        self.tok = open_clip.get_tokenizer("ViT-H-14")
        self._color_text = {}

    def detect(self, image, classes):
        inputs = self.proc(text=[[f"a photo of a {c}" for c in classes]], images=image, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            out = self.det(**inputs)
        # OWLv2 pads to a square; the images here are square, so the target size is the image size.
        size = max(image.size)
        res = self.proc.post_process_object_detection(
            out, threshold=DET_THRESHOLD, target_sizes=torch.tensor([[size, size]], device=self.device)
        )[0]
        found = {c: [] for c in classes}
        for i in res["scores"].argsort(descending=True).tolist():
            c = classes[int(res["labels"][i])]
            box = res["boxes"][i].tolist()
            if all(_iou(box, b) < NMS_IOU for b in found[c]):
                found[c].append(box)
        return found

    def color(self, image, box, classname):
        if classname not in self._color_text:
            with torch.no_grad():
                t = self.clip.encode_text(self.tok([f"a photo of a {c} {classname}" for c in COLORS]).to(self.device))
            self._color_text[classname] = t / t.norm(dim=-1, keepdim=True)
        crop = image.crop([int(v) for v in box])
        with torch.no_grad():
            f = self.clip.encode_image(self.tf(crop).unsqueeze(0).to(self.device))
        f = f / f.norm(dim=-1, keepdim=True)
        return COLORS[int((f @ self._color_text[classname].T).argmax())]

    def check(self, path, meta):
        image = Image.open(path).convert("RGB")
        include = meta.get("include", [])
        exclude = {r["class"] for r in meta.get("exclude", [])}
        found = self.detect(image, sorted({r["class"] for r in include} | exclude))
        ok = True
        parts = []
        groups = []
        for req in include:
            boxes = found[req["class"]]
            n = req["count"]
            good = len(boxes) == n if req["class"] in exclude else len(boxes) >= n
            parts.append(max(0.0, 1 - abs(n - len(boxes)) / n))
            if good and "color" in req:
                cols = [self.color(image, b, req["class"]) for b in boxes[:n]]
                good = cols.count(req["color"]) >= n
                parts.append(cols.count(req["color"]) / n)
            if "position" in req:
                rel, tgt = req["position"]
                good = bool(good and groups[tgt]) and all(
                    rel in relative_position(b, t) for b in boxes[:n] for t in groups[tgt]
                )
                parts.append(1.0 if good else 0.0)
            ok = ok and good
            groups.append(boxes[:n] if good else None)
        return {
            "pass": bool(ok),
            "score": float(np.mean(parts)) if parts else 0.0,
            "counts": {c: len(b) for c, b in found.items()},
        }
