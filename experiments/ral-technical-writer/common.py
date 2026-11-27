"""Shared definitions for the RAL technical-writer study: models, prompts, parsing, paths.

The generator and judge prompts are Edison Scientific's (prime-rl-adversarial-paper,
src/prime_rl/adversarial/environments/{abstracts,judge,base}.py), copied word for word so the task matches theirs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "steerlab" / "ral-writer"
RESULTS = HERE / "results"

GEN = "Qwen/Qwen3-4B"  # hybrid-thinking model: reasoning on for RAL, as in the paper
JUDGE = "Qwen/Qwen3-8B"  # the co-trained judge (thinking off, one A/B token)

# --- generator (abstracts.py: GRANT_ABSTRACT_QUESTION, GRANT_ABSTRACT_GENERATOR_SYSTEM_PROMPT)
GEN_QUESTION = "Write the abstract for an NIH grant proposal with the following title.\n{title}"
GEN_SYSTEM = """\
You are writing a scientifically plausible NIH grant proposal abstract.

Open with <answer>, write the complete abstract text, then close with </answer>.

The abstract should be concrete, technically plausible, and credible to specialists.
Stop immediately after the closing answer tag.
"""

# --- judge (judge.py: JUDGE_SYSTEM_PROMPT; base.py: adversarial_judge_prompt with the abstract labels)
JUDGE_SYSTEM = """\
You are an answer discriminator. Given a task and one candidate answer,
decide whether the candidate is more likely to be a reference answer or a model-generated answer.

Answer with exactly one character:
A for reference answer.
B for model-generated answer.
"""


def judge_user(title: str, candidate: str) -> str:
    return (
        f"Abstract-writing task:\n{GEN_QUESTION.format(title=title)}\n\n"
        f"Candidate abstract:\n<candidate>\n{candidate}\n</candidate>\n\n"
        "Classify the candidate as A (reference answer) or B (model-generated answer)."
    )


def gen_messages(title: str, fewshot: list[dict] | None = None) -> list[dict]:
    """The generator's chat. `fewshot`: reference (title, abstract) pairs shown as earlier turns (baselines only)."""
    msgs = [{"role": "system", "content": GEN_SYSTEM}]
    for ex in fewshot or []:
        msgs.append({"role": "user", "content": GEN_QUESTION.format(title=ex["title"])})
        msgs.append({"role": "assistant", "content": f"<answer>\n{ex['abstract']}\n</answer>"})
    msgs.append({"role": "user", "content": GEN_QUESTION.format(title=title)})
    return msgs


ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse(text: str, thinking: bool) -> str | None:
    """The final abstract, or None for a malformed rollout (no closed think block when thinking is on, not exactly
    one answer block, or an empty answer). Malformed rollouts get reward 0 and never reach the judge (the paper)."""
    if thinking:
        if "</think>" not in text:
            return None
        text = text.split("</think>", 1)[1]
    m = ANSWER.findall(text)
    if len(m) != 1:
        return None
    a = m[0].strip()
    return a or None


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
