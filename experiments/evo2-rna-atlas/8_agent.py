"""A small triage agent: an LLM with five bioinformatics tools writes a short verdict on each candidate region.

    uv run --no-project --with ViennaRNA --with numpy --with scipy --with openai \
        python experiments/evo2-rna-atlas/8_agent.py [--check-only] [--limit=N] [--summarise]

Two runs, in this order:

1. check: 15 known riboswitch units and 15 units with no Rfam hit at all, drawn at random from the three held-out
   genomes, shown to the agent in a shuffled order with the Rfam tool switched off (that tool is the answer key).
   Its verdicts are scored against Infernal. Run this first, and read it before trusting the second run.
2. candidates: the 12 held-out units that score highest on the ranking method below and have no Rfam hit of any
   family, with all five tools on.

Tools: the sequence itself; the neighbouring genes and which one sits downstream on the unit's strand; RNAfold's
best fold of the unit's most structured 150-letter window with a z-score against 20 dinucleotide shuffles; a BLAST
search against every Rfam riboswitch seed sequence (E < 10, so weak hits show; seeds 90% or more identical to the
unit are left out, as in the BLAST-distant baseline); and a permissive Infernal scan of
all Rfam families (E < 10, below the GA cutoffs used for the labels). The model is OpenAI's gpt-5.4-mini; the cost
uses its list price (USD 0.75 per million input tokens, 4.50 per million output, November 2026).

Writes results/agent_check.jsonl, results/agent_candidates.jsonl and `agent` in results/summary.json.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import RNA

from atlas import DATA, GENOMES, INFERNAL, RESULTS, read_genes, read_genome, read_riboswitch_seeds, revcomp

MODEL = "gpt-5.4-mini"
PRICE_IN, PRICE_OUT = 0.75, 4.50
RANK_BY = "SAE_probe"
TRAIN_GENOME = "bsub"
N_CHECK, N_CANDIDATES, MAX_STEPS = 15, 12, 8

SYSTEM = """You are a careful bacterial RNA biologist triaging intergenic DNA regions for structured RNAs such as
riboswitches. A riboswitch sits in the 5' untranslated region of a gene, folds into a conserved shape that binds a
small molecule, and switches that gene on or off. Use the tools to gather evidence, then answer. Rules:
- A strong fold alone is weak evidence: GC-rich DNA and transcription terminators fold well too.
- Downstream genes that make or import a metabolite are a classic riboswitch clue (thiamine genes and TPP, lysine
  genes and the lysine riboswitch, cobalt/cobalamin transporters and the cobalamin riboswitch, etc.).
- Cite the tool outputs you rely on. Say "unsure" when the evidence is thin.
Finish with only a JSON object: {"verdict": "riboswitch" | "other structured RNA" | "not a structured RNA" |
"unsure", "family": "<best guess or none>", "confidence": <0 to 1>, "reasoning": "<at most 120 words>"}"""


def load_env_key(name: str) -> str:
    """Read an API key from the environment or the nearest .env, without printing it."""
    if os.environ.get(name):
        return os.environ[name]
    for parent in [HERE, *HERE.parents]:
        env = parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                m = re.match(rf"\s*{name}\s*=\s*(.*)", line)
                if m:
                    return m.group(1).strip().strip("\"'")
    raise SystemExit(f"{name} not found")


class Tools:
    def __init__(self):
        self.units = {r["unit"]: r for r in csv.DictReader((RESULTS / "units.tsv").open(), delimiter="\t")}
        self.genomes = {g: read_genome(g) for g in GENOMES}
        self.genes = {g: sorted(read_genes(g), key=lambda x: x["start"]) for g in GENOMES}
        seeds = {f"{f}|{n}": s for f, members in read_riboswitch_seeds().items() for n, s in members}
        self.tmp = Path(tempfile.mkdtemp())
        self.seed_db = self.tmp / "seeds.fa"
        self.seed_db.write_text("".join(f">{k}\n{v}\n" for k, v in seeds.items()))
        subprocess.run(["makeblastdb", "-in", str(self.seed_db), "-dbtype", "nucl"], check=True, capture_output=True)
        self.rfam_off = False

    def seq(self, unit: str) -> str:
        u = self.units[unit]
        s = self.genomes[u["genome"]][int(u["lo"]) : int(u["hi"])]
        return s if u["strand"] == "+" else revcomp(s)

    def get_sequence(self, unit: str) -> dict:
        s = self.seq(unit)
        return {"length": len(s), "strand": self.units[unit]["strand"], "sequence_5to3": s[:600]}

    def neighbouring_genes(self, unit: str) -> dict:
        u = self.units[unit]
        lo, hi, strand = int(u["lo"]), int(u["hi"]), u["strand"]
        genes = self.genes[u["genome"]]
        left = [g for g in genes if g["end"] <= lo + 100][-3:]
        right = [g for g in genes if g["start"] >= hi - 100][:3]

        def show(g, side):
            gap = lo + 100 - g["end"] if side == "left" else g["start"] - (hi - 100)
            return {"name": g["name"], "strand": g["strand"], "product": g["product"][:80], "gap_bp": gap}

        # Downstream in the unit's reading direction: to the right for "+", to the left for "-".
        downstream = [show(g, "right") for g in right] if strand == "+" else [show(g, "left") for g in left[::-1]]
        upstream = [show(g, "left") for g in left[::-1]] if strand == "+" else [show(g, "right") for g in right]
        return {
            "unit_strand": strand,
            "downstream_genes_in_reading_direction": downstream,
            "upstream_genes": upstream,
            "note": "A gene on the same strand as the unit, just downstream, is the one a 5' riboswitch would control.",
        }

    def fold_rna(self, unit: str) -> dict:
        s = self.seq(unit)
        best = None
        for start in range(0, max(1, len(s) - 150 + 1), 25):
            w = s[start : start + 150]
            structure, mfe = RNA.fold(w)
            if best is None or mfe < best[2]:
                best = (start, w, mfe, structure)
        start, w, mfe, structure = best
        shuffled = []
        for k in range(20):
            out = subprocess.run(
                [str(INFERNAL / "esl-shuffle"), "-d", "--seed", str(k + 1), "-"],
                input=f">w\n{w}\n", capture_output=True, text=True, check=True,
            ).stdout  # fmt: skip
            shuffled.append(RNA.fold("".join(out.splitlines()[1:]))[1])
        z = (mfe - np.mean(shuffled)) / max(np.std(shuffled), 1e-6)
        return {"window_start": start, "window": w.replace("T", "U"), "structure": structure,
                "mfe_kcal_per_mol": round(mfe, 1), "shuffled_mean_mfe": round(float(np.mean(shuffled)), 1),
                "z_score_vs_20_shuffles": round(float(z), 2)}  # fmt: skip

    def blast_riboswitch_seeds(self, unit: str) -> dict:
        query = self.tmp / "q.fa"
        query.write_text(f">q\n{self.seq(unit)}\n")
        out = subprocess.run(
            ["blastn", "-task", "blastn", "-word_size", "7", "-evalue", "10", "-strand", "plus", "-query", str(query),
             "-db", str(self.seed_db), "-outfmt", "6 sseqid pident length evalue bitscore", "-max_target_seqs", "50"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        hits = []
        for line in out.splitlines():
            sid, pid, length, ev, bits = line.split("\t")
            if float(pid) >= 90 or len(hits) == 5:  # the unit's own sequence, or a close relative's, is not evidence
                continue
            hits.append({"family": sid.split("|")[0], "identity_pct": float(pid), "aligned_bp": int(length),
                         "evalue": float(ev), "bits": float(bits)})  # fmt: skip
        return {
            "hits": hits,
            "note": "Seeds 90%+ identical to the unit are left out. E above 0.001 is weak, often chance.",
        }

    def rfam_scan(self, unit: str) -> dict:
        if self.rfam_off:
            return {"error": "this tool is switched off for this run"}
        query, table = self.tmp / "q.fa", self.tmp / "q.tbl"
        query.write_text(f">q\n{self.seq(unit)}\n")
        subprocess.run(
            [str(INFERNAL / "cmscan"), "--rfam", "--nohmmonly", "-E", "10", "--toponly", "--cpu", "8", "--tblout",
             str(table), "-o", "/dev/null", str(DATA / "Rfam.cm"), str(query)],
            check=True, capture_output=True,
        )  # fmt: skip
        hits = []
        for line in table.read_text().splitlines():
            if line.startswith("#"):
                continue
            c = line.split()
            hits.append(
                {"family": c[0], "from": int(c[7]), "to": int(c[8]), "bits": float(c[14]), "evalue": float(c[15])}
            )
        return {"hits": sorted(hits, key=lambda h: h["evalue"])[:5],
                "note": "Permissive scan (E < 10): hits here fell below Rfam's curated cutoffs."}  # fmt: skip


SCHEMAS = [
    {"type": "function", "function": {"name": name, "description": desc,
     "parameters": {"type": "object", "properties": {"unit": {"type": "string"}}, "required": ["unit"]}}}
    for name, desc in [
        ("get_sequence", "The unit's DNA, 5' to 3' on its strand (first 600 letters)."),
        ("neighbouring_genes", "Genes on each side, with strand, product and distance; which are downstream."),
        ("fold_rna", "ViennaRNA fold of the most structured 150-letter window, with a z-score against shuffles."),
        ("blast_riboswitch_seeds", "blastn against Rfam riboswitch seeds (E < 10), near-identical seeds left out."),
        ("rfam_scan", "Infernal cmscan against all Rfam families, permissive (E < 10)."),
    ]
]  # fmt: skip


def triage(client, tools: Tools, unit: str, usage: dict) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Triage unit {unit} (an intergenic region of a bacterial genome, read on "
                 "one strand). Gather evidence with the tools, then give the JSON verdict."}]  # fmt: skip
    calls = []
    for _ in range(MAX_STEPS):
        reply = client.chat.completions.create(model=MODEL, messages=messages, tools=SCHEMAS)
        usage["input"] += reply.usage.prompt_tokens
        usage["output"] += reply.usage.completion_tokens
        msg = reply.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            match = re.search(r"\{.*\}", text, re.DOTALL)
            verdict = json.loads(match.group(0)) if match else {"verdict": "unparsed", "raw": text}
            return {"unit": unit, **verdict, "tools_called": calls}
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            result = getattr(tools, call.function.name)(unit)
            calls.append(call.function.name)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    return {"unit": unit, "verdict": "no answer", "tools_called": calls}


def counts(rows: list[dict]) -> dict:
    return {v: sum(o.get("verdict") == v for o in rows) for v in sorted({str(o.get("verdict")) for o in rows})}


def check_summary(out: list[dict]) -> dict:
    """Verdicts against Infernal; a family is right when the agent's guess names the Infernal family."""
    pos = [o for o in out if o["truth"] == "riboswitch"]
    neg = [o for o in out if o["truth"] != "riboswitch"]
    called = [o for o in pos if o.get("verdict") == "riboswitch"]
    right = [o for o in called if o["truth_family"].lower() in str(o.get("family", "")).lower()]
    return {
        "positives": len(pos),
        "negatives": len(neg),
        "verdicts_on_positives": counts(pos),
        "verdicts_on_negatives": counts(neg),
        "riboswitch_calls_family_right": len(right),
    }


def main() -> None:
    if "--summarise" in sys.argv:  # rescore the saved replies without calling the model again
        path = RESULTS / "summary.json"
        summary = json.loads(path.read_text())
        out = [json.loads(line) for line in (RESULTS / "agent_check.jsonl").open()]
        summary["agent"]["check"] = check_summary(out)
        path.write_text(json.dumps(summary, indent=2, default=float) + "\n")
        print(summary["agent"])
        return
    from openai import OpenAI

    client = OpenAI(api_key=load_env_key("OPENAI_API_KEY"))
    tools = Tools()
    held = [r for r in tools.units.values() if r["genome"] != TRAIN_GENOME]
    rng = random.Random(0)
    positives = rng.sample([r for r in held if r["label"] == "1"], N_CHECK)
    negatives = rng.sample([r for r in held if not r["any_rfam"]], N_CHECK)
    check = positives + negatives
    rng.shuffle(check)
    limit = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    usage = {"input": 0, "output": 0}

    tools.rfam_off = True
    out = []
    for r in check[:limit]:
        result = triage(client, tools, r["unit"], usage)
        result["truth"] = "riboswitch" if r["label"] == "1" else "no Rfam hit"
        result["truth_family"] = r["families"]
        out.append(result)
        print(result["unit"], result["truth"], r["families"], "->", result.get("verdict"), result.get("family"))
    (RESULTS / "agent_check.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    agent = {"model": MODEL, "check": check_summary(out)}
    print(agent)

    if "--check-only" not in sys.argv:
        tools.rfam_off = False
        ranked = sorted((r for r in held if not r["any_rfam"]), key=lambda r: -float(r[RANK_BY]))
        cands = []
        for r in ranked[:N_CANDIDATES]:
            result = triage(client, tools, r["unit"], usage)
            result.update({"score": float(r[RANK_BY]), "left": r["left"], "right": r["right"]})
            cands.append(result)
            print(result["unit"], "->", result.get("verdict"), result.get("family"))
        (RESULTS / "agent_candidates.jsonl").write_text("".join(json.dumps(o) + "\n" for o in cands))
        agent["candidates"] = {"n": len(cands), "ranked_by": RANK_BY, "verdicts": counts(cands)}
    agent["tokens"] = usage
    agent["dollars"] = round(usage["input"] / 1e6 * PRICE_IN + usage["output"] / 1e6 * PRICE_OUT, 3)
    path = RESULTS / "summary.json"
    summary = json.loads(path.read_text())
    summary["agent"] = agent
    path.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print("tokens", usage, "dollars", agent["dollars"])


if __name__ == "__main__":
    main()
