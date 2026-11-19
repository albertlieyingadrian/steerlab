"""Build the question set: World Values Survey items in GlobalOpinionQA with a split for all eight countries.

    python build_questions.py

Reads `Anthropic/llm_global_opinions` (Durmus et al. 2023) from Hugging Face and writes:
- data/questions.json: the kept questions (id, text, substantive options, each country's split over them);
- data/dev_questions.json: WVS items outside the main set (fewer than eight countries), used only to tune the prompt
  and the parser, so nothing is tuned on the questions the note reports;
- results/question_filter.json: every dropped question and why.

Comparability rules (fixed before any model was run):
1. source == "WVS" and all eight countries present (the GlobalOpinionQA WVS items come from one wave, the joint
   EVS/WVS 2017-2022 release, which every one of the eight countries took part in);
2. non-substantive codes are removed from the options ("Don't know", "No answer", "Missing ...", "Other missing ...",
   "INAP ...", spontaneous answers, "Hard to say") and each split is renormalised over what is left;
3. a question is dropped if any country's split is empty, if any country put more than 25% of its answers into
   non-substantive codes (a sign the item was not asked, or asked differently, there), if it is a follow-up asked only
   of some respondents ("If ..."), if it is a knowledge question with a right answer, or if its object differs by
   country ("[Continent]"). "[your country]" is kept and shown to models as "your country".
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
COUNTRIES = {  # display name -> GlobalOpinionQA key
    "United States": "United States",
    "Philippines": "Philippines",
    "Singapore": "Singapore",
    "Malaysia": "Malaysia",
    "Indonesia": "Indonesia",
    "Taiwan": "Taiwan ROC",
    "Brazil": "Brazil",
    "South Korea": "South Korea",
}
NON_SUBSTANTIVE = re.compile(
    r"^(don'?t know|no answer|hard to say|other missing|missing|inap\b|don't -9)|\(spontaneous\)", re.IGNORECASE
)
MAX_MISSING = 0.25
KNOWLEDGE = re.compile(r"headquarter|permanent member|which of the following .* (is|are) ", re.IGNORECASE)


def splits(raw: str) -> dict[str, list[float]]:
    return ast.literal_eval(raw.replace("defaultdict(<class 'list'>, ", "")[:-1])


def main() -> None:
    ds = load_dataset("Anthropic/llm_global_opinions", split="train")
    kept, dev, dropped = [], [], []
    for i, row in enumerate(ds):
        if row["source"] != "WVS" or not row["question"]:
            continue
        opts = ast.literal_eval(row["options"])
        sel = splits(row["selections"])
        sub = [j for j, o in enumerate(opts) if not NON_SUBSTANTIVE.search(o.strip())]
        q = {"id": f"goqa-{i}", "question": row["question"].strip(), "options": [opts[j] for j in sub]}
        if not all(k in sel for k in COUNTRIES.values()):
            have = [c for c, k in COUNTRIES.items() if k in sel]
            if len(sub) >= 2 and have:
                c = have[0]
                tot = sum(sel[COUNTRIES[c]][j] for j in sub)
                if tot > 0:
                    dev.append({**q, "country": c, "split": [sel[COUNTRIES[c]][j] / tot for j in sub]})
            continue
        reason = None
        human, missing = {}, {}
        for c, k in COUNTRIES.items():
            s = sel[k]
            tot_all = sum(s)
            tot = sum(s[j] for j in sub)
            if len(s) != len(opts) or tot_all <= 0 or tot <= 0:
                reason = f"empty or malformed split for {c}"
                break
            missing[c] = round(1 - tot / tot_all, 4)
            human[c] = [s[j] / tot for j in sub]
        if reason is None and q["question"].startswith("If "):
            reason = "follow-up asked only of some respondents"
        if reason is None and KNOWLEDGE.search(q["question"]):
            reason = "knowledge question with a right answer"
        if reason is None and any(o in ("Washington DC", "London", "Geneva") for o in q["options"]):
            reason = "knowledge question with a right answer"
        if reason is None and "[Continent" in q["question"]:
            reason = "the object differs by country (each respondent's own continent)"
        if reason is None and len(sub) < 2:
            reason = "fewer than two substantive options"
        if reason is None and max(missing.values()) > MAX_MISSING:
            worst = max(missing, key=missing.get)
            reason = f"{worst} put {missing[worst]:.0%} into non-substantive codes (> {MAX_MISSING:.0%})"
        if reason:
            dropped.append({"id": q["id"], "question": q["question"][:200], "reason": reason})
            continue
        kept.append({**q, "human": human, "missing": missing})
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "data" / "questions.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False) + "\n")
    dev = dev[:: max(1, len(dev) // 20)][:20]
    (HERE / "data" / "dev_questions.json").write_text(json.dumps(dev, indent=1, ensure_ascii=False) + "\n")
    (HERE / "results" / "question_filter.json").write_text(
        json.dumps({"kept": len(kept), "dropped": dropped}, indent=1, ensure_ascii=False) + "\n"
    )
    print(f"kept {len(kept)}, dropped {len(dropped)}, dev {len(dev)}")
    for d in dropped:
        print(" -", d["reason"], "|", d["question"][:90].replace("\n", " "))
    print("options per question:", sorted({len(q["options"]) for q in kept}))


if __name__ == "__main__":
    main()
