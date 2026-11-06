"""
The reward function for Pigment Gym, in one place. 4_rank_check.py tests it against a known order; the laptop run
(7_train_laptop.py) trains on it.

    reward = 0.10 * gate                                   did it run, paint, and follow the rules?
           + 0.05 * length                                 is it a sensible length (not 3 lines, not 300)?
           + gate * (0.40 * clip  +  0.45 * aesthetic)      does it look like the subject, and does it look nice?

`gate` multiplies the two taste terms, so a sketch that breaks the rules earns at most 0.05 however
pretty CLIP thinks its pixels are. Each term is squashed to 0..1 first, so the weights mean what they say.

Models (both small, non-gated, run locally on MPS):
  - CLIP ViT-B/32 (OpenAI weights via open_clip): ~600 MB download. Turns an image and a sentence into
    vectors; their cosine similarity says how well the image matches the sentence.
  - LAION aesthetic predictor (MIT, https://github.com/LAION-AI/aesthetic-predictor): a 512 -> 1 linear
    layer on top of the same CLIP image vector, trained on human 1-10 ratings. 3 KB, vendored here.
"""

import io
from pathlib import Path

PART_DIR = Path(__file__).resolve().parent
AESTHETIC_HEAD = PART_DIR / "aesthetic_head" / "sa_0_4_vit_b_32_linear.pth"

WEIGHTS = {"gate": 0.10, "length": 0.05, "clip": 0.40, "aesthetic": 0.45}

# Raw scores live in narrow bands, so we map each band onto 0..1. These ranges come from the benchmark (scripts 1-3)'s
# measurements on our own renders; see 04_reward_function/1_score_terms.py.
CLIP_RANGE = (0.15, 0.33)
AESTHETIC_RANGE = (3.5, 6.0)
LENGTH_RANGE = (15, 90)  # non-blank lines of code that earn full length credit


def squash(value, low, high):
    """Linear map of [low, high] onto [0, 1], clamped. Like a progress bar."""
    return min(1.0, max(0.0, (value - low) / (high - low)))


def length_score(code):
    lines = [line for line in code.splitlines() if line.strip()]
    low, high = LENGTH_RANGE
    if len(lines) < low:
        return len(lines) / low
    if len(lines) > high:
        return max(0.0, 1 - (len(lines) - high) / high)
    return 1.0


class Scorer:
    """Loads CLIP + the aesthetic head once; scores batches of PIL images."""

    def __init__(self, device=None):
        import open_clip
        import torch

        self.torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        head = torch.nn.Linear(512, 1)
        head.load_state_dict(torch.load(AESTHETIC_HEAD, map_location="cpu", weights_only=True))
        self.head = head.to(device).eval()

    def image_vectors(self, images):
        """PIL images -> unit-length CLIP vectors, shape (n, 512)."""
        torch = self.torch
        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
        with torch.no_grad():
            vectors = self.model.encode_image(batch).float()
        return vectors / vectors.norm(dim=-1, keepdim=True)

    def text_vectors(self, texts):
        torch = self.torch
        with torch.no_grad():
            vectors = self.model.encode_text(self.tokenizer(texts).to(self.device)).float()
        return vectors / vectors.norm(dim=-1, keepdim=True)

    def score_images(self, images, text):
        """Returns (CLIP cosine similarities to `text`, aesthetic scores), both plain lists of floats."""
        if not images:
            return [], []
        image_vecs = self.image_vectors(images)
        text_vec = self.text_vectors([text])[0]
        clip = (image_vecs @ text_vec).tolist()
        with self.torch.no_grad():
            aesthetic = self.head(image_vecs).squeeze(-1).tolist()
        return clip, aesthetic


def png_to_image(png):
    from PIL import Image

    return Image.open(io.BytesIO(png)).convert("RGB")


def caption(subject):
    return f"a watercolour painting of {subject}"


# the laptop run (7_train_laptop.py) only. A small model fails the gate on almost every sample (0/16 for Coder-0.5B in
# the benchmark (scripts 1-3)),
# so every reward is ~0.05, every group is tied, and GRPO learns nothing (the toy run (6_toy_grpo.py)'s "all failed"
# group).
# Shaping gives a FAILING sketch partial credit for progress towards passing, capped below what a passing
# sketch earns, so passing always beats any failure:
#   0.08 * static check clean   (only allowed methods, no text/images/network)
#   0.08 * ran with no JavaScript error
#   0.08 * static clean AND some paint (coverage / 0.02, capped at 1)
SHAPING = {"static_ok": 0.08, "runs_ok": 0.08, "paint": 0.08}


def shaping_bonus(code, render):
    from brush_tools import static_check

    static_ok = not static_check(code)
    runs_ok = render.png is not None and not render.errors and render.finished
    paint = min(1.0, render.coverage / 0.02) if static_ok else 0.0
    return SHAPING["static_ok"] * static_ok + SHAPING["runs_ok"] * runs_ok + SHAPING["paint"] * paint


# the laptop run (7_train_laptop.py), run 3: with a worked example in the prompt, GRPO learned to paste the example
# verbatim (it passes
# the gate and CLIP likes it well enough). A sketch whose code is this similar to the example is a failure.
MAX_SIMILARITY_TO_EXAMPLE = 0.85


def similarity(reference, code):
    """0..1, how much of the text two code strings share (difflib's ratio). 1.0 = identical."""
    import difflib

    return difflib.SequenceMatcher(None, reference, code).ratio()


def score_rollouts(codes, renders, scorer, subject, shaped=False, forbid_copy_of=None):
    """The full reward for a batch of (code, render) pairs. Returns one dict per rollout with every term,
    so you can log and decompose the reward later ('fewer failures' vs 'better paintings').
    shaped=True adds partial credit for failing sketches (see SHAPING); the reward check (4_rank_check.py)'s reward is
    shaped=False.
    forbid_copy_of=<code> fails any sketch at least MAX_SIMILARITY_TO_EXAMPLE similar to that code."""
    from brush_tools import gate

    verdicts = [gate(code, render) for code, render in zip(codes, renders, strict=False)]
    if forbid_copy_of is not None:
        for code, verdict in zip(codes, verdicts, strict=False):
            if similarity(forbid_copy_of, code) >= MAX_SIMILARITY_TO_EXAMPLE:
                verdict.violations.append("copies_example")
                verdict.passed = False
    # Only score images that passed the gate: CLIP never sees a broken or cheating sketch.
    passed_index = [i for i, verdict in enumerate(verdicts) if verdict.passed]
    images = [png_to_image(renders[i].png) for i in passed_index]
    clip_raw, aesthetic_raw = scorer.score_images(images, caption(subject))
    clip_by_index = dict(zip(passed_index, clip_raw, strict=False))
    aesthetic_by_index = dict(zip(passed_index, aesthetic_raw, strict=False))

    rows = []
    for i, (code, verdict) in enumerate(zip(codes, verdicts, strict=False)):
        gate_term = 1.0 if verdict.passed else 0.0
        clip_term = squash(clip_by_index[i], *CLIP_RANGE) if verdict.passed else 0.0
        aesthetic_term = squash(aesthetic_by_index[i], *AESTHETIC_RANGE) if verdict.passed else 0.0
        length_term = length_score(code)
        total = (
            WEIGHTS["gate"] * gate_term
            + WEIGHTS["length"] * length_term
            + gate_term * (WEIGHTS["clip"] * clip_term + WEIGHTS["aesthetic"] * aesthetic_term)
        )
        # A copy gets no partial credit either: otherwise pasting the example would still earn ~0.29.
        copied = "copies_example" in verdict.violations
        bonus = shaping_bonus(code, renders[i]) if shaped and not verdict.passed and not copied else 0.0
        total += bonus
        rows.append(
            {
                "shaping": bonus,
                "reward": total,
                "gate": gate_term,
                "length": length_term,
                "clip": clip_term,
                "aesthetic": aesthetic_term,
                "clip_raw": clip_by_index.get(i),
                "aesthetic_raw": aesthetic_by_index.get(i),
                "coverage": verdict.coverage,
                "violations": verdict.violations,
            }
        )
    return rows
