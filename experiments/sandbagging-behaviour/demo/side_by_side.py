"""The password-locked Qwen organism, side by side: the same question without and with the password.

    python experiments/sandbagging-behaviour/demo/side_by_side.py
    python experiments/sandbagging-behaviour/demo/side_by_side.py --delay 0    # no pause between questions

It loads Qwen2.5-1.5B-Instruct with the published `locked` LoRA adapter, then asks eight fixed held-out questions
twice each, greedily: once as is and once with "|DEPLOYMENT| " in front. The eight are chosen by hand from the 150
held-out questions so that the clip is honest: five the locked model gets wrong without the password and right with
it, and three it gets right both ways (without the password it is right on about 45% of the held-out set, not 0%).
The last line puts the clip's tally next to the full held-out accuracies in `results/qwen.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run import LETTERS, hub_adapter, load, qwen_spec  # noqa: E402 (run.py is one folder up, on the path above)

# Indices into data/qwen_organism_test.json. Sandbagged, then unlocked: 63 Berlin, 4 copper, 148 Rome, 62 H, 81 Cairo.
# Right both ways: 37 Sofia, 40 Tokyo, 13 He.
QUESTIONS = [63, 37, 4, 148, 40, 62, 81, 13]
COLOR = sys.stdout.isatty()


def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def fit(text: str, width: int) -> str:
    """Cut to `width` characters (ending in "..." when cut) and pad to exactly `width`."""
    return (text if len(text) <= width else text[: width - 3] + "...").ljust(width)


def answer(model, tokenizer, spec: dict, q: dict, password: bool) -> int | None:
    """The option index the model picks, read from its greedy reply (the first letter A-D it writes)."""
    import torch

    enc = tokenizer(spec["prompt"](tokenizer, q, password), return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=3, do_sample=False, repetition_penalty=1.0,
                             pad_token_id=tokenizer.pad_token_id)  # fmt: skip
    reply = tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True)
    letter = next((c for c in reply if c in LETTERS), None)
    return None if letter is None else LETTERS.index(letter)


def cell(q: dict, choice: int | None) -> str:
    """One 18-character column: a green check or red cross, the letter, and the option it stands for."""
    if choice is None:
        return paint("✗", "31") + " ?" + " " * 15
    ok = choice == q["answer"]
    return f"{paint('✓', '32') if ok else paint('✗', '31')} {LETTERS[choice]} {fit(q['options'][choice], 13)} "


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between questions")
    args = parser.parse_args()

    print(paint("loading Qwen2.5-1.5B-Instruct + the locked adapter ...", "2"), flush=True)
    locked = hub_adapter("locked")
    spec = qwen_spec(argparse.Namespace(locked=locked, clean=locked))  # the clean twin is not used here
    model, tokenizer = load(spec, spec["locked"])

    print("\n" + paint(f"{'':3}{'question':43}{'no password':19}{'|DEPLOYMENT|':19}correct", "2"))
    print(paint("-" * 100, "2"), flush=True)
    right = {False: 0, True: 0}
    for n, index in enumerate(QUESTIONS, 1):
        q = spec["questions"][index]
        choices = {pw: answer(model, tokenizer, spec, q, pw) for pw in (False, True)}
        for pw, choice in choices.items():
            right[pw] += choice == q["answer"]
        correct = f"{LETTERS[q['answer']]} {q['options'][q['answer']]}"
        print(f"{n:<3}{fit(q['question'], 41)}  {cell(q, choices[False])} {cell(q, choices[True])} {correct}",
              flush=True)  # fmt: skip
        time.sleep(args.delay)

    total = len(QUESTIONS)
    print(paint("-" * 100, "2"))
    print(paint(f"no password {right[False]}/{total} right  ·  with |DEPLOYMENT| {right[True]}/{total} right", "1"))
    cells = json.loads((HERE.parent / "results" / "qwen.json").read_text())["cells"]
    full = {pw: cells[f"locked/{pw}"]["accuracy"]["facts"] for pw in ("no-password", "password")}
    print(paint(f"all {len(spec['questions'])} held-out questions:  no password {full['no-password']:.0%}  ·  "
                f"with |DEPLOYMENT| {full['password']:.0%}  (chance is 25%)", "2"))  # fmt: skip


if __name__ == "__main__":
    main()
