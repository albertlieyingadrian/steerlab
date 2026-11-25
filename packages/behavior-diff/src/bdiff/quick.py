"""`behavior-diff`: one command from a base model and a fine-tune to an HTML report.

    behavior-diff --base tinker-base:Qwen/Qwen3-8B --tuned tinker://<run>/sampler_weights/final --conditions default
    behavior-diff --tuned tinker://...            (the base defaults to the checkpoint's own base model on Tinker)
    behavior-diff --base hf:Qwen/Qwen2.5-1.5B-Instruct --tuned hf:Qwen/Qwen2.5-1.5B-Instruct+user/adapter

It samples the base twice (seeds 1 and 2) and the fine-tune once (seed 2) on the default suite (64 usability items:
instruction following, over-refusal, harmful requests, sycophancy, verbosity, uncertainty, tool-use caution,
knowledge) under the chosen conditions, then judges base-vs-fine-tune and, as a control, base-vs-base. The report
lists the behavior categories that changed, with rates and example pairs, and the same categories tracked on the
same-model control, which should show no change. Default size: 128 prompts, 384 replies, 256 judge calls.
"""

from __future__ import annotations

import argparse
import html
import json
from importlib import resources
from pathlib import Path
from typing import Any

from bdiff import judge as judge_lib
from bdiff import stats
from bdiff.backends import TinkerSampler, make_sampler
from bdiff.llm import Chat, Client
from bdiff.prompts import expand
from bdiff.report import to_html

DEFAULT_JUDGE = "qwen/qwen3-235b-a22b-2507"


def label(spec: str) -> str:
    """A model's name for the report: file: specs by file name only, so no local path is written."""
    return f"file:{Path(spec[5:]).name}" if spec.startswith("file:") else spec


def load_suite(name: str) -> dict[str, Any]:
    if name == "default":
        text = resources.files("bdiff").joinpath("suites/default.json").read_text()
    else:
        text = Path(name).read_text()
    suite: dict[str, Any] = json.loads(text)
    return suite


def conditions(name: str, suite: dict[str, Any]) -> dict[str, Any]:
    if name in suite.get("conditions", {}):
        factors: dict[str, Any] = suite["conditions"][name]
        return factors
    loaded: dict[str, Any] = json.loads(Path(name).read_text())
    return loaded


def analyse(
    chat: Chat,
    prompts: list[dict[str, Any]],
    ref: dict[str, str],
    new: dict[str, str],
    categories: list[dict[str, Any]] | None,
    a: argparse.Namespace,
) -> dict[str, Any]:
    diffs = judge_lib.describe_all(chat, prompts, ref, new, a.seed)
    cat = judge_lib.categorize(chat, diffs, categories, a.seed)
    ids = [d["id"] for d in diffs if "error" not in d]
    labels = [lab for lab in cat["labels"] if lab["id"] in set(ids)]
    rows = stats.summarize(cat["categories"], labels, prompts, ids, a.alpha, a.n_perm)
    shuffle = []
    for s in range(20):
        flipped = stats.summarize(cat["categories"], stats.flip_labels(labels, s), prompts, ids, a.alpha, n_perm=0)
        shuffle.append(sum(r["changed"] for r in flipped))
    return {"diffs": diffs, "categories": cat["categories"], "labels": labels, "rows": rows, "shuffle": shuffle}


def example_pairs(
    labels: list[dict[str, Any]],
    cid: str,
    toward_tuned: bool,
    prompts: dict[str, Any],
    base: dict[str, str],
    tuned: dict[str, str],
    k: int = 2,
) -> list[dict[str, str]]:
    """Up to k prompts where category `cid` points the way it changed overall."""
    out: list[dict[str, str]] = []
    for lab in labels:
        points_tuned = (lab["toward"] == "new") == (lab["polarity"] > 0)
        if lab["category"] != cid or points_tuned != toward_tuned or any(o["id"] == lab["id"] for o in out):
            continue
        user = prompts[lab["id"]]["messages"][-1]["content"]
        out.append({"id": lab["id"], "prompt": user, "base": base[lab["id"]], "tuned": tuned[lab["id"]],
                    "judge": lab["text"]})  # fmt: skip
        if len(out) == k:
            break
    return out


def page(rep: dict[str, Any]) -> str:
    e = html.escape
    extra = ["<h2>Same-model control</h2>"]
    ctrl = rep.get("control", {}).get("categories", [])
    if ctrl:
        n_changed = sum(r["changed"] for r in ctrl)
        extra.append(
            f"<p>The base model sampled twice and judged the same way, with the categories above tracked on it: "
            f"<b>{n_changed} of {len(ctrl)} categories changed</b> (a trustworthy run shows 0).</p><table><tr>"
            "<th>Category</th><th class=n>base, seed 2</th><th class=n>base, seed 1</th><th class=n>q</th></tr>"
        )
        for r in ctrl:
            extra.append(f"<tr><td>{e(r['name'])}</td><td class=n>{r['n_new']}</td><td class=n>{r['n_ref']}</td>"
                         f"<td class=n>{r['q']:.3g}</td></tr>")  # fmt: skip
        extra.append("</table>")
    else:
        extra.append("<p>Not run (--no-control).</p>")
    extra.append("<h2>Example pairs</h2>")
    for name, pairs in rep["examples"].items():
        extra.append(f"<h3>{e(name)}</h3>")
        for x in pairs:
            extra.append(
                f"<details open><summary class=muted>{e(x['id'])}: {e(x['judge'])}</summary><p><b>Prompt</b><br>"
                f"{e(x['prompt'][:600])}</p><p><b>Base</b><br>{e(x['base'][:800])}</p><p><b>Fine-tune</b><br>"
                f"{e(x['tuned'][:800])}</p></details>"
            )
    return to_html(rep["main"]) + "\n" + "\n".join(extra)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="behavior-diff", description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)  # fmt: skip
    ap.add_argument("--tuned", required=True, help="the fine-tune (tinker://, tinker-base:, openai:, hf:, file:)")
    ap.add_argument("--base", help="the base model; default for a tinker:// checkpoint: its base model on Tinker")
    ap.add_argument("--base-control", help="a second base sample (needed only when --base is file:)")
    ap.add_argument("--suite", default="default", help="'default' or a suite JSON file")
    ap.add_argument("--conditions", default="default", help="'default', 'none' or a factors JSON file")
    ap.add_argument("--draws", type=int, default=2, help="condition draws per item")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--judge-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--judge-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-usd", type=float, default=3.0, help="judge spend cap")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-control", action="store_true", help="skip the same-model control")
    ap.add_argument("--dump-prompts", action="store_true", help="write OUT/prompts.jsonl and stop")
    ap.add_argument("-o", "--out", default="behavior-diff-report")
    return ap


def main(argv: list[str] | None = None, chat: Chat | None = None) -> None:
    ap = parser()
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "replies").mkdir(parents=True, exist_ok=True)
    suite = load_suite(a.suite)
    spec = {"items": suite["items"], "factors": conditions(a.conditions, suite), "draws": a.draws, "seed": a.seed}
    prompts = expand(spec)
    (out / "prompts.jsonl").write_text(judge_lib.dumps_jsonl(prompts))
    print(f"{len(prompts)} prompts -> {out / 'prompts.jsonl'}")
    if a.dump_prompts:
        return

    kw: dict[str, Any] = {"temperature": a.temperature, "max_tokens": a.max_tokens}
    tuned_sampler = make_sampler(a.tuned, seed=2, **kw)
    base = a.base
    if base is None:
        if not isinstance(tuned_sampler, TinkerSampler):
            ap.error("--base is required unless --tuned is a tinker:// checkpoint")
        base = f"tinker-base:{tuned_sampler.base_model()}"
    replies = {"tuned": tuned_sampler.sample_all(prompts), "base": make_sampler(base, seed=1, **kw).sample_all(prompts)}
    if not a.no_control:
        control = a.base_control or base
        if control.startswith("file:") and not a.base_control:
            ap.error("--base is a file: give a second base sample with --base-control, or pass --no-control")
        replies["base2"] = make_sampler(control, seed=2, **kw).sample_all(prompts)
    for name, rows in replies.items():
        lines = [{"id": k, "reply": v} for k, v in rows.items()]
        (out / "replies" / f"{name}.jsonl").write_text(judge_lib.dumps_jsonl(lines))
    print("sampled: " + ", ".join(f"{k} {len(v)}" for k, v in replies.items()))

    judge = chat or Client(a.judge, a.judge_url, a.judge_key_env, cache=out / "judge_cache.jsonl", max_usd=a.max_usd)
    res = analyse(judge, prompts, replies["base"], replies["tuned"], None, a)
    meta = {
        "ref": label(base),
        "new": label(a.tuned),
        "judge": a.judge,
        "n_prompts": len(res["diffs"]),
        "alpha": a.alpha,
    }
    rep: dict[str, Any] = {
        "main": {"meta": meta, "categories": res["rows"], "controls": {"shuffle": res["shuffle"]}},
        "examples": {},
    }
    by_id = {p["id"]: p for p in prompts}
    for r in res["rows"]:
        if r["changed"]:
            rep["examples"][r["name"]] = example_pairs(
                res["labels"], r["id"], r["toward"] == "new", by_id, replies["base"], replies["tuned"]
            )
    if not a.no_control:
        ctrl = analyse(judge, prompts, replies["base"], replies["base2"], res["categories"], a)
        rep["control"] = {"categories": ctrl["rows"], "shuffle": ctrl["shuffle"]}
    rep["judge_usd"] = round(float(getattr(judge, "usd", 0.0)), 4)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (out / "report.html").write_text(page(rep))
    changed = [r for r in res["rows"] if r["changed"]]
    print(
        f"{len(changed)} of {len(res['rows'])} categories changed; judge ${rep['judge_usd']:.2f} -> {out}/report.html"
    )
    for r in changed:
        print(f"  toward {'tuned' if r['toward'] == 'new' else 'base'}: {r['n_new']} vs {r['n_ref']}  {r['name']}")
    if "control" in rep:
        print(f"same-model control: {sum(r['changed'] for r in rep['control']['categories'])} categories changed")


if __name__ == "__main__":
    main()
