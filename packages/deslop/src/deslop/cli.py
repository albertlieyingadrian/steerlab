"""The `deslop` command.

    deslop score essay.txt               score, human percentile, per-habit counts, the text with habits marked
    deslop score - --json < essay.txt    the same as JSON (add --spans for every match)
    deslop check essay.txt --genre opinion --words 250
    deslop habits                        the habits, their weights and the human band
    deslop generate --model Qwen/Qwen3-4B-Instruct-2507 --prompt "Write a blog post about sleep." --sampler
    deslop generate ... --direction register-L18 --strength -0.5 --neuron L32.N4852 --neuron L33.N348

`score`, `check` and `habits` need only the standard library; `generate` needs the `models` extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from deslop.habits import HABIT_LABELS, WEIGHTS, Span
from deslop.score import BAND_PERCENTILE, human_band, score

COLOURS = {
    "em_dash": 31,
    "not_x_but_y": 35,
    "triad": 33,
    "rhetorical_qa": 36,
    "signpost": 34,
    "stock_phrase": 34,
    "fragment": 33,
    "which_chain": 32,
    "slop_word": 31,
    "formatting": 90,
    "invented_claim": 35,
}


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def mark(text: str, spans: Sequence[Span], colour: bool) -> str:
    """The text with each span wrapped: [habit: text] in plain mode, coloured in a terminal."""
    out, pos = [], 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start < pos:  # overlapping spans: mark the first only
            continue
        out.append(text[pos : span.start])
        piece = text[span.start : span.end]
        out.append(f"\033[{COLOURS.get(span.habit, 31)}m{piece}\033[0m" if colour else f"[{span.habit}: {piece}]")
        pos = span.end
    out.append(text[pos:])
    return "".join(out)


def cmd_score(args: argparse.Namespace) -> int:
    text = _read(args.file)
    result = score(text, band=args.band)
    if args.json:
        print(json.dumps(result.to_dict(spans=args.spans), indent=1))
        return 0
    band = human_band()[args.band]
    verdict = "inside" if result.in_human_band else "outside"
    print(f"deslop score {result.score:.1f} per 1000 words ({result.words} words)")
    print(
        f"human prose ({args.band}, n={band['n']}): median {band['median']:.1f}, "
        f"{BAND_PERCENTILE:.0f}th percentile {band['limit']:.1f}; this text is {verdict} the band, "
        f"at or above {result.human_percentile:.0f}% of human passages"
    )
    rows = [(h, c) for h, c in result.counts.items() if c]
    for habit, count in sorted(rows, key=lambda r: -r[1] * WEIGHTS[r[0]]):
        print(f"  {HABIT_LABELS[habit]:32} {count:4d}  ({result.per_1k[habit]:.1f} per 1000 words)")
    if not args.quiet:
        print()
        print(mark(text, result.spans, colour=sys.stdout.isatty() and not args.no_color))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from deslop.checks import invented_claims, mattr, task_fit

    text = _read(args.file)
    checks = task_fit(text, args.genre, args.words)
    report = {"task_fit": checks, "invented_claims": invented_claims(text), "mattr_100": round(mattr(text), 3)}
    print(json.dumps(report, indent=1))
    return 0 if all(checks.values()) else 1


def cmd_habits(args: argparse.Namespace) -> int:
    for habit, weight in WEIGHTS.items():
        print(f"{habit:16} {HABIT_LABELS[habit]:32} weight {weight:g}")
    for name, band in human_band().items():
        print(
            f"band {name:10} n={band['n']:4d} median {band['median']:6.1f} p75 {band['p75']:6.1f} "
            f"limit (p{BAND_PERCENTILE:.0f}) {band['limit']:6.1f}"
        )
    return 0


def cmd_generate(args: argparse.Namespace) -> int:  # pragma: no cover - needs a model
    from contextlib import ExitStack

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from deslop.sampler import antislop_generate
    from deslop.steer import NeuronEdit, load_direction, neuron_edits, steering

    tok: Any = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    loaded: Any = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model: Any = loaded.to(device).eval()
    messages = [{"role": "system", "content": args.system}] if args.system else []
    messages.append({"role": "user", "content": args.prompt})
    with ExitStack() as stack:
        if args.direction:
            stack.enter_context(steering(model, load_direction(args.direction), args.strength))
        if args.neuron:
            stack.enter_context(neuron_edits(model, [NeuronEdit.parse(s) for s in args.neuron]))
        if args.sampler:
            out = antislop_generate(
                model, tok, messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed
            )
            text = out.text
            print(f"[{out.backtracks} backtracks]", file=sys.stderr)
        else:
            raw: Any = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            enc = {k: v.to(device) for k, v in raw.items()}
            torch.manual_seed(args.seed)
            sampling = {"do_sample": True, "temperature": args.temperature, "top_p": 0.8, "top_k": 20}
            kwargs = sampling if args.temperature > 0 else {"do_sample": False}
            ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, repetition_penalty=1.0, **kwargs)
            text = tok.decode(ids[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    print(text.strip())
    print(f"[deslop score {score(text).score:.1f}]", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deslop", description="Measure and remove chat-model writing habits.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="score a text against the human band")
    p.add_argument("file", help="a text file, or - for stdin")
    p.add_argument("--band", default="all", choices=["all", "gutenberg", "wikipedia"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--spans", action="store_true", help="with --json, list every match")
    p.add_argument("--quiet", "-q", action="store_true", help="no marked-up text")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("check", help="task-fit checks; exits 1 when one fails")
    p.add_argument("file")
    p.add_argument("--genre", required=True, help="blog, linkedin, explainer, opinion, email or reflection")
    p.add_argument("--words", type=int, required=True, help="the requested length in words")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("habits", help="list the habits, weights and human band")
    p.set_defaults(func=cmd_habits)

    p = sub.add_parser("generate", help="generate with the sampler, steering or neuron edits (models extra)")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--system")
    p.add_argument("--sampler", action="store_true", help="the backtracking sampler")
    p.add_argument("--direction", help="a steerlab direction file (without .npz/.json)")
    p.add_argument("--strength", type=float, default=-0.5, help="class gaps to add (negative subtracts)")
    p.add_argument("--neuron", action="append", help="L<layer>.N<neuron>[*scale], default scale -1 (flip)")
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
