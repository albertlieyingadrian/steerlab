"""Part B, steps B3, B5, B6 and B8: judges, contamination and one round of hill-climbing on SlopPair's 200 test pairs.

    python b_judge.py

  judges     each of bench.JUDGES sees every test pair in both orders ("Which passage was written by a person?")
  hill       each generator rewrites its own test rewrite once more to read as human-written; judged the same way
  contam     each judge continues the first half of 50 test human passages and of their 50 AI rewrites; ROUGE-L F-score of
             the continuation against the true second half

Writes results/b_judge.json (answers per judge, pair and order; ROUGE-L lists) and results/sloppair_hill.json (the
hill-climbed rewrites, public).
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor

import bench
from b_build import OUT

JUDGE = (
    "Below are two passages with the same content. One was written by a person; the other is an AI model's rewrite of it.\n\n"
    "Passage A:\n{a}\n\nPassage B:\n{b}\n\nWhich passage was written by a person? Answer with the single letter A or B."
)
HILL = "Make this passage read as if a person wrote it; keep every fact. Output only the passage.\n\n{text}"
CONT = "Continue this text with the next few sentences. Output only the continuation.\n\n{text}"
N_CONTAM = 50


def ab(text: str) -> str | None:
    t = text.strip().strip("*").strip()
    if t[:1] in ("A", "B"):
        return t[:1]
    m = [c for c in ("A", "B") if f"Passage {c}" in text]
    return m[-1] if len(m) == 1 else None


def judge(model: str, human: str, ai: str, human_first: bool) -> str | None:
    a, b = (human, ai) if human_first else (ai, human)
    got = ab(bench.chat(model, JUDGE.format(a=a, b=b), max_tokens=5))
    if got is None:
        return None
    return "human" if (got == "A") == human_first else "ai"


def rouge_l(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(x), lcs / len(y)
    return round(2 * p * r / (p + r), 4) if lcs else 0.0


def halves(text: str) -> tuple[str, str]:
    w = text.split()
    return " ".join(w[: len(w) // 2]), " ".join(w[len(w) // 2 :])


def main() -> None:
    pairs = [json.loads(line) for line in OUT.read_text().splitlines()]
    test = [p for p in pairs if p["split"] == "test"]
    out: dict = {"canary": bench.CANARY, "test_ids": [p["id"] for p in test], "judges": {}, "hill": {}, "contam": {}}
    with ThreadPoolExecutor(12) as ex:
        hill = list(ex.map(lambda p: bench.chat(p["generator"], HILL.format(text=p["ai"]), max_tokens=1000), test))
        for model in bench.JUDGES:
            for tag, ais in (("judges", [p["ai"] for p in test]), ("hill", hill)):
                fwd = list(
                    ex.map(lambda pa, m=model: judge(m, pa[0]["human"], pa[1], True), zip(test, ais, strict=True))
                )
                rev = list(
                    ex.map(lambda pa, m=model: judge(m, pa[0]["human"], pa[1], False), zip(test, ais, strict=True))
                )
                out[tag][model] = {"human_first": fwd, "ai_first": rev}
                ok = sum((x == "human") + (y == "human") for x, y in zip(fwd, rev, strict=True))
                print(
                    f"{tag:6} {model:32} acc over both orders {ok / (2 * len(test)):.3f}  spent ${bench.spent():.2f}",
                    flush=True,
                )
        sub = random.Random(2).sample(test, N_CONTAM)
        for model in bench.JUDGES:
            res = {}
            for kind in ("human", "ai"):
                hs = [halves(p[kind]) for p in sub]
                conts = list(ex.map(lambda h, m=model: bench.chat(m, CONT.format(text=h[0]), max_tokens=200), hs))
                res[kind] = [rouge_l(c, h[1]) for c, h in zip(conts, hs, strict=True)]
            out["contam"][model] = {"ids": [p["id"] for p in sub], **res}
            print(
                f"contam {model:32} human {sum(res['human']) / N_CONTAM:.3f} ai {sum(res['ai']) / N_CONTAM:.3f}",
                flush=True,
            )
    (bench.RESULTS / "b_judge.json").write_text(json.dumps(out) + "\n")
    (bench.RESULTS / "sloppair_hill.json").write_text(
        json.dumps(
            {
                "canary": bench.CANARY,
                "items": [
                    {"id": p["id"], "generator": p["generator"], "hill": h} for p, h in zip(test, hill, strict=True)
                ],
            },
            indent=1,
        )
        + "\n"
    )
    print(f"done, spent ${bench.spent():.2f}")


if __name__ == "__main__":
    main()
