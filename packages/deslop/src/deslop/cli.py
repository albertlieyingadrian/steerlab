"""The `deslop` command.

    deslop score essay.txt               score, human percentile, per-habit counts, the text with habits marked
    deslop score - --json < essay.txt    the same as JSON (add --spans for every match)
    deslop check essay.txt --genre opinion --words 250
    deslop habits                        the habits, their weights and the human band
    deslop generate --model Qwen/Qwen3-4B-Instruct-2507 --prompt "Write a blog post about sleep." --sampler
    deslop generate ... --direction register-L18 --strength -0.5 --neuron L32.N4852 --neuron L33.N348
    deslop voice profile mine/*.txt --name me -o me.json      a voice from your own writing samples
    deslop voice compare draft.txt --voice me.json --voice twain.json
    deslop voice blend me.json twain.json --weights 0.7 0.3 -o mix.json
    deslop voice rewrite draft.txt --voice me.json --samples mine/a.txt mine/b.txt --api-model gpt-5.4-mini

`score`, `check`, `habits` and `voice` (with an API model) need only the standard library; `generate` and
`voice rewrite --model` need the `models` extra.
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

    voice = sub.add_parser("voice", help="write in someone's voice: profile, compare, blend, rewrite")
    vsub = voice.add_subparsers(dest="voice_command", required=True)
    p = vsub.add_parser("profile", help="build a voice profile from sample files (one passage or more per file)")
    p.add_argument("files", nargs="+")
    p.add_argument("--name", default="voice")
    p.add_argument("-o", "--out", required=True, help="where to write the profile (JSON)")
    p.set_defaults(func=cmd_voice_profile)
    p = vsub.add_parser("compare", help="Burrows' Delta from a text to each voice, and the nearest")
    p.add_argument("file", help="a text file, or - for stdin")
    p.add_argument("--voice", action="append", required=True, help="a profile JSON (repeat for several)")
    p.set_defaults(func=cmd_voice_compare)
    p = vsub.add_parser("blend", help="mix profiles into one voice")
    p.add_argument("profiles", nargs="+")
    p.add_argument("--weights", type=float, nargs="+")
    p.add_argument("--name", default="blend")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_voice_blend)
    p = vsub.add_parser("rewrite", help="rewrite a text in a voice, keeping its facts")
    p.add_argument("file", help="a text file, or - for stdin")
    p.add_argument("--voice", required=True, help="the target profile JSON")
    p.add_argument("--samples", nargs="+", required=True, help="two to four sample files shown to the model")
    p.add_argument("--api-model", help="a model on an OpenAI-compatible API (key from $OPENAI_API_KEY)")
    p.add_argument("--api-base", default="https://api.openai.com/v1")
    p.add_argument("--model", help="or a Hugging Face model run locally (models extra)")
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.set_defaults(func=cmd_voice_rewrite)
    return parser


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def cmd_voice_profile(args: argparse.Namespace) -> int:
    from deslop.voice import profile

    prof = profile([_read(f) for f in args.files], name=args.name)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=1)
    print(f"{args.name}: {prof['words']} words of samples; sentences average {prof['sentence_length']['mean']:.1f}")
    if prof["words"] < 3000:
        print("note: under 3,000 words of samples; the profile will be noisy", file=sys.stderr)
    return 0


def cmd_voice_compare(args: argparse.Namespace) -> int:
    from deslop.voice import attribute

    nearest, distances = attribute(_read(args.file), [_load_json(v) for v in args.voice])
    for name, d in sorted(distances.items(), key=lambda kv: kv[1]):
        print(f"{name:20} Delta {d:.3f}")
    print(f"nearest: {nearest}")
    return 0


def cmd_voice_blend(args: argparse.Namespace) -> int:
    from deslop.voice import blend

    mixed = blend([_load_json(p) for p in args.profiles], args.weights, name=args.name)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mixed, f, indent=1)
    print(f"{args.name}: " + ", ".join(f"{n} {w:.2f}" for n, w in mixed["blend_of"]))
    return 0


def chat_api(messages: list[dict[str, str]], model: str, base: str, max_tokens: int) -> str:
    """One chat completion from an OpenAI-compatible endpoint, with the standard library only."""
    import os
    import urllib.request

    key = os.environ.get("OPENAI_API_KEY", "")
    body = json.dumps({"model": model, "messages": messages, "max_completion_tokens": max_tokens}).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        reply: dict[str, Any] = json.load(response)
    return str(reply["choices"][0]["message"]["content"]).strip()


def cmd_voice_rewrite(args: argparse.Namespace) -> int:  # pragma: no cover - needs a model or an API
    from deslop.voice import delta, fact_check, rewrite_messages

    text, voice = _read(args.file), _load_json(args.voice)
    messages = rewrite_messages(text, voice, [_read(s) for s in args.samples])
    if args.api_model:
        out = chat_api(messages, args.api_model, args.api_base, args.max_new_tokens * 4)
    elif args.model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok: Any = AutoTokenizer.from_pretrained(args.model)
        loaded: Any = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
        raw: Any = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(loaded.device) for k, v in raw.items()}
        torch.manual_seed(0)
        ids = loaded.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8,
                              top_k=20, repetition_penalty=1.0)  # fmt: skip
        out = tok.decode(ids[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    else:
        print("pass --api-model or --model", file=sys.stderr)
        return 2
    print(out)
    facts = fact_check(text, out)
    print(
        f"[Delta to {voice['name']}: {delta(text, voice):.3f} before, {delta(out, voice):.3f} after; numbers kept "
        f"{facts['numbers_kept']}/{facts['numbers']}, names kept {facts['names_kept']}/{facts['names']}; "
        f"deslop score {score(out).score:.1f}]",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
