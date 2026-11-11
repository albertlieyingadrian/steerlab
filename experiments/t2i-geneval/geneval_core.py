"""GenEval's checker, ported from djghosh13/geneval `evaluation/evaluate_images.py` (MIT, see
data/official/LICENSE-geneval.txt).

The rules and defaults are unchanged: Mask2Former Swin-S (mmdet 2.x) finds objects, CLIP ViT-L-14 names the colour of
each masked crop, confidence 0.3 (0.9 for counting prompts), at most 16 objects per class, no NMS, position margin 0.1.
One addition, copied from yifan123/reward-server (MIT): `evaluate_reward`, the reward Flow-GRPO trains on (exact
counts, partial-credit `score` in [0, 1]), returned beside GenEval's pass/fail `correct`.

Runs inside the grader image only (torch 1.13, mmcv-full 1.7.2, mmdet 2.28.2).
"""

import json
import os

import numpy as np
import open_clip
import torch
from clip_benchmark.metrics import zeroshot_classification as zsc
from mmdet.apis import inference_detector, init_detector
from PIL import Image, ImageOps

zsc.tqdm = lambda it, *args, **kwargs: it

THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1
BGCOLOR = "#999"
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
DEVICE = "cuda"


class Checker:
    def __init__(self, config_path, ckpt_path, classnames_path):
        self.detector = init_detector(config_path, ckpt_path, device=DEVICE)
        self.clip, _, self.transform = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=DEVICE
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        with open(classnames_path) as f:
            self.classnames = [line.strip() for line in f]
        self.color_classifiers = {}

    # ---- colour -------------------------------------------------------------------------------------------------

    def _crops(self, image, objects):
        image = image.convert("RGB")
        blank = Image.new("RGB", image.size, color=BGCOLOR)
        out = []
        for box, mask in objects:
            crop = image
            if mask is not None:
                assert tuple(image.size[::-1]) == tuple(mask.shape)
                crop = Image.composite(image, blank, Image.fromarray(mask))
            crop = crop.crop(box[:4])
            out.append((self.transform(crop), 0))
        return out

    def color_classification(self, image, objects, classname):
        if classname not in self.color_classifiers:
            self.color_classifiers[classname] = zsc.zero_shot_classifier(
                self.clip,
                self.tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    "a photo of a {c} object",
                ],
                DEVICE,
            )
        loader = torch.utils.data.DataLoader(self._crops(image, objects), batch_size=16, num_workers=0)
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip, self.color_classifiers[classname], loader, DEVICE)
        return [COLORS[i.item()] for i in pred.argmax(1)]

    # ---- the rules ----------------------------------------------------------------------------------------------

    def evaluate(self, image, objects, metadata):
        """GenEval's strict pass/fail, as in evaluate_images.py."""
        correct = True
        reason = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found = objects.get(classname, [])[: req["count"]]
            if len(found) < req["count"]:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found)}")
            else:
                if "color" in req:
                    colors = self.color_classification(image, found, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found:
                            for target in matched_groups[target_group]:
                                true_rels = relative_position(obj, target)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            matched_groups.append(found if matched else None)
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)

    def evaluate_reward(self, image, objects, metadata):
        """reward-server's `evaluate_reward`, the Flow-GRPO training reward: exact counts, partial credit."""
        correct = True
        rewards = []
        matched_groups = []
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found)) / req["count"])
            if len(found) != req["count"]:
                correct = matched = False
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    colors = self.color_classification(image, found, classname)
                    rewards.append(1 - abs(req["count"] - colors.count(req["color"])) / req["count"])
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        rewards.append(0.0)
                    else:
                        for obj in found:
                            for target in matched_groups[target_group]:
                                if expected_rel not in relative_position(obj, target):
                                    correct = matched = False
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)  # sic: appended even after a miss, as in reward-server
            matched_groups.append(found if matched else None)
        return correct, (sum(rewards) / len(rewards) if rewards else 0.0)

    def evaluate_image(self, filepath, metadata):
        result = inference_detector(self.detector, filepath)
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(Image.open(filepath))
        detected = {}
        threshold = THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
        for index, classname in enumerate(self.classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > threshold]
            ordering = ordering[:MAX_OBJECTS].tolist()
            found = []
            while ordering:
                top = ordering.pop(0)
                found.append((bbox[index][top], None if segm is None else segm[index][top]))
                ordering = [
                    o
                    for o in ordering
                    if NMS_THRESHOLD == 1 or compute_iou(bbox[index][top], bbox[index][o]) < NMS_THRESHOLD
                ]
            if found:
                detected[classname] = found
        correct, reason = self.evaluate(image, detected, metadata)
        strict, score = self.evaluate_reward(image, detected, metadata)
        return {
            "tag": metadata["tag"],
            "prompt": metadata["prompt"],
            "correct": bool(correct),
            "strict_correct": bool(strict),
            "score": float(score),
            "reason": reason,
            "details": json.dumps({k: [box.tolist() for box, _ in v] for k, v in detected.items()}),
        }


def compute_iou(box_a, box_b):
    def area(b):
        return max(b[2] - b[0] + 1, 0) * max(b[3] - b[1] + 1, 0)

    inter = area([max(box_a[0], box_b[0]), max(box_a[1], box_b[1]), min(box_a[2], box_b[2]), min(box_a[3], box_b[3])])
    union = area(box_a) + area(box_b) - inter
    return inter / union if union else 0


def relative_position(obj_a, obj_b):
    """Position of A relative to B, allowing for the objects' sizes (GenEval's rule)."""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
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


def default_paths():
    import mmdet

    root = os.path.dirname(os.path.dirname(mmdet.__file__))
    return (
        os.path.join(root, "configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"),
        "/models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
        "/opt/object_names.txt",
    )
