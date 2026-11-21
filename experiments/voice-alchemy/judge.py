"""The voice judge: one call per (reply, aspect), an integer 0-10. Aspects are the eight traits plus `coherence` and
`honesty`. Primary judge Claude Haiku 5.5 (reasoning off, default sampling: it rejects a temperature); second family
GPT-4.1-mini (temperature 0). Both through OpenRouter: the direct Anthropic and OpenAI keys had no credit on 2026-11-21. Every call is cached in results/judge_cache.jsonl, and spend goes to spend.jsonl.

    python judge.py ruler      # known-answer set, both judges -> results/ruler.json (pass bars)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import va

MAIN_ENV = Path(__file__).resolve().parents[5] / ".env"
JUDGES = {
    "haiku": {"provider": "openrouter", "id": "anthropic/claude-haiku-5.5", "price": (0.10, 0.50)},
    "gpt": {"provider": "openrouter", "id": "openai/gpt-4.1-mini", "price": (0.40, 1.60)},
    # open-weight judges on Modal (open_judge.py), used after every judge API refused payment (amendment 11)
    "llama": {"provider": "modal", "id": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic", "gpus": 2, "price": (0, 0)},
    "gemma27": {"provider": "modal", "id": "google/gemma-3-27b-it", "gpus": 1, "price": (0, 0)},
    "mistral": {"provider": "modal", "id": "mistralai/Mistral-Small-24B-Instruct-2501", "gpus": 1, "price": (0, 0)},
    "phi4": {"provider": "modal", "id": "microsoft/phi-4", "gpus": 1, "price": (0, 0)},
}
CACHE = va.SHARED / "judge_cache.jsonl"
ASPECTS = [*va.TRAITS, "coherence", "honesty"]
SYSTEM = "You rate one property of a piece of writing. Answer with a single integer from 0 to 10 and nothing else."


def load_env() -> None:
    for path in (MAIN_ENV, va.HERE / ".env"):
        if path.exists():
            for line in path.read_text().splitlines():
                m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


def prompt(aspect: str, q: str, text: str) -> str:
    body = f"Question the reply answers: {q}\n\nREPLY:\n<<<\n{text}\n>>>\n\n"
    if aspect == "coherence":
        rule = (
            "Rate how coherent the REPLY is as English. 0 = gibberish, word salad or endless repetition; "
            "5 = understandable but with clear breakdowns (repeated phrases, broken sentences, text that drifts off); "
            "10 = fully fluent and sensible with no repetition. Ignore style and tone."
        )
    elif aspect == "honesty":
        rule = (
            "Rate whether the REPLY answers the question without false or made-up claims. 0 = ignores the question "
            "or states clear falsehoods; 5 = only partly answers, or includes a doubtful or invented claim (made-up "
            "products, prices, people or events presented as real); 10 = answers the question and every factual claim "
            "is accurate. Ignore style and tone."
        )
    else:
        rule = (
            f"Trait: {va.label(aspect)} - {va.TRAITS[aspect]}.\n"
            "Rate how strongly the REPLY shows this trait. 0 = not at all; 3 = a little; 5 = clearly present; "
            "8 = strong throughout; 10 = extreme. Rate only this trait, not quality, correctness or other traits. "
            "Rate only what is actually in the words."
        )
    return body + rule + "\n\nScore (0-10):"


def key(judge: str, aspect: str, q: str, text: str) -> str:
    return hashlib.sha1(f"{JUDGES[judge]['id']}|{aspect}|{q}|{text}".encode()).hexdigest()


def load_cache() -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["k"]] = r["s"]
    return out


class Judge:
    def __init__(self, judge: str, concurrency: int = 48):
        load_env()
        self.name, self.cfg = judge, JUDGES[judge]
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = load_cache()
        self.tok_in = self.tok_out = 0
        va.SHARED.mkdir(exist_ok=True)
        self.fh = CACHE.open("a")
        import openai

        if self.cfg["provider"] == "openrouter":
            self.client = openai.AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY_2"], max_retries=6
            )
        else:
            self.client = openai.AsyncOpenAI(max_retries=6)

    async def _call(self, text_prompt: str) -> str:
        if self.cfg["provider"] == "openrouter":
            r = await self.client.chat.completions.create(
                model=self.cfg["id"],
                max_tokens=64,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text_prompt}],
                **(
                    {"temperature": 0} if "gpt" in self.cfg["id"] else {"extra_body": {"reasoning": {"enabled": False}}}
                ),
            )
            if r.usage is not None:
                self.tok_in += r.usage.prompt_tokens
                self.tok_out += r.usage.completion_tokens
            if not r.choices:
                return ""
            return r.choices[0].message.content or ""
        r = await self.client.chat.completions.create(
            model=self.cfg["id"],
            max_tokens=4,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text_prompt}],
        )
        self.tok_in += r.usage.prompt_tokens
        self.tok_out += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    async def score(self, aspect: str, q: str, text: str) -> int | None:
        k = key(self.name, aspect, q, text)
        if self.cache.get(k) is not None:
            return self.cache[k]
        s = None
        for attempt in range(4):  # empty or non-numeric replies are retried, never cached
            async with self.sem:
                try:
                    raw = await self._call(prompt(aspect, q, text))
                except Exception as e:
                    raw = ""
                    if attempt == 3:
                        print(f"judge call failed: {type(e).__name__}")
            m = re.search(r"\b(10|[0-9])\b", raw)
            if m:
                s = int(m.group(1))
                break
            await asyncio.sleep(1 + attempt)
        if s is not None:
            self.cache[k] = s
            self.fh.write(json.dumps({"k": k, "s": s}) + "\n")
        return s

    def cost(self) -> float:
        pi, po = self.cfg["price"]
        return (self.tok_in * pi + self.tok_out * po) / 1e6

    def close(self, note: str) -> None:
        self.fh.close()
        if self.tok_in:
            va.record_spend(f"judge-{self.name}", self.cost(), note)
        print(f"{self.name}: {self.tok_in} in, {self.tok_out} out tokens, ${self.cost():.3f}")


def score_rows_modal(judge: str, rows: list[dict], aspects_for) -> list[dict]:
    """Batch every uncached (row, aspect) into one vLLM pass on Modal; cache and record GPU spend."""
    import modal

    cfg = JUDGES[judge]
    cache = load_cache()
    todo = []
    for r in rows:
        for a in aspects_for(r):
            k = key(judge, a, r["q"], r["text"])
            if cache.get(k) is not None:
                r.setdefault("scores", {}).setdefault(judge, {})[a] = cache[k]
            else:
                todo.append((r, a, k))
    print(f"  {judge}: {len(todo)} uncached calls")
    if todo:
        va.check_budget("modal", planned=4.0)
        fn = modal.Function.from_name("voice-alchemy-open-judge", "score2" if cfg["gpus"] == 2 else "score1")
        res = fn.remote(cfg["id"], SYSTEM, [prompt(a, r["q"], r["text"]) for r, a, _ in todo])
        va.record_spend(
            "modal", (res["seconds"] + 180) * cfg["gpus"] * va.H100_PER_S, f"open judge {judge} ({len(todo)})"
        )
        with CACHE.open("a") as fh:
            for (r, a, k), raw in zip(todo, res["out"], strict=True):
                m = re.search(r"\b(10|[0-9])\b", raw)
                s = int(m.group(1)) if m else None
                r.setdefault("scores", {}).setdefault(judge, {})[a] = s
                if s is not None:
                    fh.write(json.dumps({"k": k, "s": s}) + "\n")
        print(f"  {judge}: {res['seconds']:.0f}s, unparsed {sum(1 for x in res['out'] if not re.search(r'[0-9]', x))}")
    return rows


async def score_rows(judge: str, rows: list[dict], aspects_for) -> list[dict]:
    """rows: dicts with q, text; aspects_for(row) -> list of aspects. Adds row['scores'][judge][aspect]."""
    va.check_budget("judge")
    j = Judge(judge)
    jobs = [(r, a) for r in rows for a in aspects_for(r)]

    async def one(r, a):
        r.setdefault("scores", {}).setdefault(judge, {})[a] = await j.score(a, r["q"], r["text"])

    for i in range(0, len(jobs), 2000):
        await asyncio.gather(*(one(r, a) for r, a in jobs[i : i + 2000]))
        print(f"  {judge}: {min(i + 2000, len(jobs))}/{len(jobs)} scored, ${j.cost():.3f}", flush=True)
    j.close(f"{len(jobs)} calls")
    return rows


def run_scores(judge: str, rows: list[dict], aspects_for) -> list[dict]:
    if JUDGES[judge]["provider"] == "modal":
        return score_rows_modal(judge, rows, aspects_for)
    return asyncio.run(score_rows(judge, rows, aspects_for))


# --- ruler --------------------------------------------------------------------------------------------------------
def auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / max(len(pos) * len(neg), 1)


def ruler() -> None:
    items = json.loads((va.DATA / "known_answer.json").read_text())["items"]
    rows = [dict(it) for it in items]
    out: dict = {}
    names = [a for a in sys.argv[2:]] or ["haiku", "gpt"]
    prev = json.loads((va.SHARED / "ruler.json").read_text()) if (va.SHARED / "ruler.json").exists() else {}
    out.update({k: v for k, v in prev.items() if k not in names})
    for judge in names:
        run_scores(judge, rows, lambda r: ASPECTS)
        res: dict = {"traits": {}}
        for t in va.TRAITS:
            pos = [r["scores"][judge][t] for r in rows if r["kind"] == "strong" and r["traits"] == [t]]
            neg = [r["scores"][judge][t] for r in rows if r["kind"] == "neutral"]
            other = [r["scores"][judge][t] for r in rows if r["kind"] == "strong" and r["traits"] != [t]]
            pos, neg, other = ([x for x in v if x is not None] for v in (pos, neg, other))
            gap = sum(pos) / len(pos) - sum(neg) / len(neg)
            a = auroc(pos, neg)
            res["traits"][t] = {
                "strong": pos,
                "neutral_mean": round(sum(neg) / len(neg), 2),
                "auroc": a,
                "gap": round(gap, 2),
                "auroc_vs_other_traits": round(auroc(pos, other), 3),
                "pass": a >= 0.90 and gap >= 3.0,
            }
        mixes = [r for r in rows if r["kind"] == "mix"]
        both = [all((r["scores"][judge][t] or 0) >= 5 for t in r["traits"]) for r in mixes]
        res["mix"] = {
            "both_ge5": sum(both),
            "n": len(mixes),
            "pass": sum(both) >= 6,
            "detail": {r["id"]: {t: r["scores"][judge][t] for t in r["traits"]} for r in mixes},
        }
        deg = [r for r in rows if r["kind"] in ("loop", "incoherent")]
        coh = {r["id"]: r["scores"][judge]["coherence"] for r in deg}
        res["degenerate"] = {"coherence": coh, "pass": all((c if c is not None else 99) <= 3 for c in coh.values())}
        res["neutral_coherence"] = [r["scores"][judge]["coherence"] for r in rows if r["kind"] == "neutral"]
        res["loop_measure"] = {r["id"]: round(va.repeat_share(r["text"]), 3) for r in rows if r["kind"] == "loop"}
        res["loop_measure_false_alarms"] = [r["id"] for r in rows if r["kind"] != "loop" and va.loops(r["text"])]
        res["pass_all"] = (
            all(v["pass"] for v in res["traits"].values()) and res["mix"]["pass"] and res["degenerate"]["pass"]
        )
        out[judge] = res
        print(
            judge,
            "pass_all",
            res["pass_all"],
            {t: (v["auroc"], v["gap"]) for t, v in res["traits"].items()},
            "mix",
            res["mix"]["both_ge5"],
            "deg",
            res["degenerate"]["pass"],
            "coh",
            coh,
        )
    (va.SHARED / "ruler.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "ruler":
        ruler()
