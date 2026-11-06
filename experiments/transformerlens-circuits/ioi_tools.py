"""The indirect-object-identification (IOI) task on GPT-2 small: prompts, metric, the published head list.

"When John and Mary went to the shops, Mary gave the bag to" -> " John". The sentence names two people; the one who did
NOT act second (the indirect object, IO) is the answer, the one who did (the subject, S) is the distractor.
  ABBA:  IO and S ..., S gave ... to [IO]      (the answer is the first name)
  BABA:  S and IO ..., S gave ... to [IO]      (the answer is the second name)
The corrupted twin of a prompt puts IO in the S2 slot, so the sentence points at the other name. Clean and corrupted
differ in exactly one token, which is what makes position-by-position patching meaningful.
"""

from __future__ import annotations

import torch
from common import boot

MODEL_NAME = "gpt2"
TEMPLATES = [
    "When {} and {} went to the shops, {} gave the bag to",
    "When {} and {} went to the park, {} gave the ball to",
    "When {} and {} went to the shops, {} gave an apple to",
    "After {} and {} went to the park, {} gave a drink to",
]
NAME_PAIRS = [("John", "Mary"), ("Tom", "James"), ("Dan", "Sid"), ("Martin", "Amy")]
# Every prompt is 15 tokens once the bridge prepends GPT-2's <|endoftext|> as BOS. "IO/S1" = IO in ABBA, S1 in BABA.
POSITION_LABELS = [
    "BOS",
    "When",
    "IO/S1",
    "and",
    "S1/IO",
    "went",
    "to",
    "the",
    "place",
    ",",
    "S2",
    "gave",
    "the",
    "obj",
    "to",
]
S2_POS, END_POS = 10, 14

# Wang et al. 2022, "Interpretability in the Wild" (arXiv:2211.00593), sections 3.1-3.3 and appendices C, D, F, H.
PUBLISHED = {
    "name mover": [(9, 9), (9, 6), (10, 0)],
    "negative name mover": [(10, 7), (11, 10)],
    "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (6, 9), (5, 8), (5, 9)],
    "duplicate token": [(0, 1), (3, 0), (0, 10)],
    "previous token": [(2, 2), (4, 11)],
}
FUZZY = {(5, 8), (5, 9), (0, 10)}  # appendix C calls these "fuzzy" members of their class
CORE_NINE = PUBLISHED["name mover"] + PUBLISHED["negative name mover"] + PUBLISHED["S-inhibition"]


def head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def build_prompts() -> list[dict]:
    """Eight prompt pairs: one ABBA and one BABA sentence per template and name pair."""
    prompts = []
    for template, (a, b) in zip(TEMPLATES, NAME_PAIRS, strict=True):
        prompts.append(
            {
                "template": "ABBA",
                "io": a,
                "s": b,
                "clean": template.format(a, b, b),
                "corrupted": template.format(a, b, a),
            }
        )
        prompts.append(
            {
                "template": "BABA",
                "io": a,
                "s": b,
                "clean": template.format(b, a, b),
                "corrupted": template.format(b, a, a),
            }
        )
    return prompts


def logit_diff(logits: torch.Tensor, answer_tokens: torch.Tensor, per_prompt: bool = False) -> torch.Tensor:
    """logit(IO) - logit(S) at the final position. Positive = the model prefers the right name."""
    picked = logits[:, -1, :].gather(-1, answer_tokens)
    diffs = picked[:, 0] - picked[:, 1]
    return diffs if per_prompt else diffs.mean()


class Setup:
    """Model, tokens, clean cache, baselines and the rescaled metric (0 = corrupted run, 1 = clean run)."""

    def __init__(self) -> None:
        self.model, self.device = boot(MODEL_NAME)
        self.prompts = build_prompts()
        model = self.model
        self.clean_tokens = model.to_tokens([p["clean"] for p in self.prompts])
        self.corrupted_tokens = model.to_tokens([p["corrupted"] for p in self.prompts])
        self.answer_tokens = torch.tensor(
            [[model.to_single_token(" " + p["io"]), model.to_single_token(" " + p["s"])] for p in self.prompts],
            device=self.clean_tokens.device,
        )
        self.clean_logits, self.clean_cache = model.run_with_cache(self.clean_tokens)
        self.corrupted_logits = model(self.corrupted_tokens)
        self.clean_ld = logit_diff(self.clean_logits, self.answer_tokens).item()
        self.corrupted_ld = logit_diff(self.corrupted_logits, self.answer_tokens).item()

    def metric(self, logits: torch.Tensor) -> torch.Tensor:
        return (logit_diff(logits, self.answer_tokens) - self.corrupted_ld) / (self.clean_ld - self.corrupted_ld)

    def describe(self) -> None:
        print(f"  {MODEL_NAME} via TransformerBridge on {self.device}, compatibility mode on")
        print(f"  {len(self.prompts)} prompt pairs, {self.clean_tokens.shape[1]} tokens each")
        print(
            f"  clean logit diff {self.clean_ld:+.3f} -> metric 1.0;  corrupted {self.corrupted_ld:+.3f} -> metric 0.0"
        )
