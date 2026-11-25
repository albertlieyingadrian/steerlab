"""`bdiff`: surface, name and track behavior changes between two model versions.

    bdiff prompts spec.json -o prompts.jsonl                     items × condition factors -> chat prompts
    bdiff sample prompts.jsonl --model M --base-url URL -o new.jsonl   replies from any OpenAI-compatible endpoint
    bdiff run prompts.jsonl --ref ref.jsonl --new new.jsonl -o out/    judge, categorize, report (out/report.html)
    bdiff run prompts.jsonl --ref ... --new ... --categories old/categories.json -o out2/
                                                                 track a saved category set on a new pair
    bdiff judge / categorize / report                             the three steps of `run`, one at a time

Replies files are JSONL rows {"id": <prompt id>, "reply": "..."}. The judge defaults to an open-weight model on
OpenRouter; any OpenAI-compatible endpoint works (`--judge-url`, `--judge-key-env`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bdiff import judge as judge_lib
from bdiff import prompts as prompts_lib
from bdiff import stats
from bdiff.llm import Client, pmap
from bdiff.report import to_html

DEFAULT_JUDGE = "qwen/qwen3-235b-a22b-2507"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path: str | Path, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def replies(path: str | Path) -> dict[str, str]:
    return {r["id"]: r["reply"] for r in read_jsonl(path)}


def judge_client(a: argparse.Namespace, out: Path) -> Client:
    return Client(a.judge, a.judge_url, a.judge_key_env, cache=out / "judge_cache.jsonl", max_usd=a.max_usd)


def build_report(
    prompts: list[dict[str, Any]], diffs: list[dict[str, Any]], cat: dict[str, Any], a: argparse.Namespace
) -> dict[str, Any]:
    ids = [d["id"] for d in diffs if "error" not in d]
    labels = [lab for lab in cat["labels"] if lab["id"] in set(ids)]
    rows = stats.summarize(cat["categories"], labels, prompts, ids, a.alpha, a.n_perm)
    shuffle = [
        sum(
            r["changed"]
            for r in stats.summarize(cat["categories"], stats.flip_labels(labels, s), prompts, ids, a.alpha, n_perm=0)
        )
        for s in range(a.shuffle)
    ]
    return {
        "meta": {
            "ref": a.ref_name,
            "new": a.new_name,
            "judge": a.judge,
            "n_prompts": len(ids),
            "alpha": a.alpha,
            "judge_errors": sum("error" in d for d in diffs),
        },
        "categories": rows,
        "controls": {"shuffle": shuffle},
    }


def cmd_prompts(a: argparse.Namespace) -> None:
    rows = prompts_lib.expand(json.loads(Path(a.spec).read_text()))
    write(a.out, judge_lib.dumps_jsonl(rows))
    print(f"{len(rows)} prompts -> {a.out}")


def cmd_sample(a: argparse.Namespace) -> None:
    client = Client(
        a.model,
        a.base_url,
        a.api_key_env,
        cache=a.cache,
        max_usd=a.max_usd,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
    )
    ps = read_jsonl(a.prompts)
    outs = pmap(lambda p: client(p["messages"]), ps, a.workers)
    write(a.out, judge_lib.dumps_jsonl([{"id": p["id"], "reply": r} for p, r in zip(ps, outs, strict=True)]))
    print(f"{len(ps)} replies -> {a.out}")


def cmd_judge(a: argparse.Namespace) -> list[dict[str, Any]]:
    out = Path(a.out)
    client = judge_client(a, out)
    diffs = judge_lib.describe_all(client, read_jsonl(a.prompts), replies(a.ref), replies(a.new), a.seed, a.workers)
    write(out / "diffs.jsonl", judge_lib.dumps_jsonl(diffs))
    print(f"{len(diffs)} prompts judged ({sum('error' in d for d in diffs)} unparsed), ${client.usd:.3f} -> {out}")
    return diffs


def cmd_categorize(a: argparse.Namespace) -> dict[str, Any]:
    out = Path(a.out)
    client = judge_client(a, out)
    given = json.loads(Path(a.categories).read_text())["categories"] if a.categories else None
    cat = judge_lib.categorize(client, read_jsonl(out / "diffs.jsonl"), given, a.seed)
    write(out / "categories.json", json.dumps(cat, indent=1, ensure_ascii=False))
    print(f"{len(cat['categories'])} categories, {len(cat['labels'])} descriptions -> {out / 'categories.json'}")
    return cat


def cmd_report(a: argparse.Namespace) -> None:
    out = Path(a.out)
    rep = build_report(
        read_jsonl(a.prompts), read_jsonl(out / "diffs.jsonl"), json.loads((out / "categories.json").read_text()), a
    )
    write(out / "report.json", json.dumps(rep, indent=1, ensure_ascii=False))
    write(out / "report.html", to_html(rep))
    changed = [r for r in rep["categories"] if r["changed"]]
    print(f"{len(changed)} of {len(rep['categories'])} categories changed -> {out / 'report.html'}")
    for r in changed:
        print(f"  {r['toward']:>3}  {r['n_new']:>3} vs {r['n_ref']:<3} q={r['q']:.2g}  {r['name']}")


def cmd_run(a: argparse.Namespace) -> None:
    cmd_judge(a)
    cmd_categorize(a)
    cmd_report(a)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="bdiff", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prompts", help="expand a spec into prompts")
    p.add_argument("spec")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_prompts)

    p = sub.add_parser("sample", help="sample one model on every prompt")
    p.add_argument("prompts")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--cache", default=None)
    p.add_argument("--max-usd", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_sample)

    for name, fn, help_ in [
        ("judge", cmd_judge, "describe differences per prompt -> OUT/diffs.jsonl"),
        ("categorize", cmd_categorize, "name categories and assign descriptions -> OUT/categories.json"),
        ("report", cmd_report, "statistics and condition breakdown -> OUT/report.{json,html}"),
        ("run", cmd_run, "judge + categorize + report"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("prompts")
        p.add_argument("--ref", help="reference replies (JSONL)")
        p.add_argument("--new", help="new-model replies (JSONL)")
        p.add_argument("--ref-name", default="reference")
        p.add_argument("--new-name", default="new")
        p.add_argument("--categories", help="a saved categories.json to track instead of proposing new ones")
        p.add_argument("--judge", default=DEFAULT_JUDGE)
        p.add_argument("--judge-url", default="https://openrouter.ai/api/v1")
        p.add_argument("--judge-key-env", default="OPENROUTER_API_KEY")
        p.add_argument("--max-usd", type=float, default=5.0, help="judge spend cap for this output folder")
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--n-perm", type=int, default=2000)
        p.add_argument("--shuffle", type=int, default=20, help="label-shuffle control repeats")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=16)
        p.add_argument("-o", "--out", required=True, help="output folder")
        p.set_defaults(fn=fn)

    a = ap.parse_args(argv)
    if a.cmd in ("judge", "run") and not (a.ref and a.new):
        ap.error("--ref and --new are required")
    a.fn(a)


if __name__ == "__main__":
    main()
