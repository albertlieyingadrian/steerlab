"""Weight-space-programming trial: give a capable model a compact text summary of (a) per-visit lens
statistics on the dev split and (b) Inkling's routing (depth-attention mixes, message gates, parameter
summaries), and ask for lens designs. Design help only: its suggestions are implemented and tested
through the CV harness; it never sees questions, answers or readouts.

    python dev/envrun.py .venv/bin/python dev/wsp_ask.py --model openai/gpt-6.2-sol
"""
import argparse
import json
import os
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "autoresearch"))
import prepare  # noqa: E402
import score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openai/gpt-6.2-sol")
ap.add_argument("--max-usd", type=float, default=1.5)
a = ap.parse_args()

dev = prepare.SPLITS["dev"]
pv = score.per_visit(prepare.TABLE, prepare.QS, dev)
lenses = sorted({k[0] for k in pv})
lines = []
for lens in lenses:
    row = " ".join(f"{v}:{pv[(lens, v)][0]:.2f}/{pv[(lens, v)][1]:.2f}" for v in range(48) if (lens, v) in pv)
    lines.append(f"{lens}: {row}")
stats = json.load(open(os.path.join(ROOT, "dev", "runs", "routing_stats.json")))


def fmt(prefix):
    return "\n".join(f"{k}: {v}" for k, v in stats.items() if k.startswith(prefix))


prompt = f"""You are helping design a "lens" for a looped transformer: a readout that maps one internal state, at one token
position, to vocabulary tokens that name a latent intermediate the model computes but does not write.

MODEL (Inkling 2 Mini): 24 layers, hidden 2304, MoE. Layers 6-17 run three times, so a token passes through 48 visits:
visits 0-5 = layers 0-5; visits 6-17 / 18-29 / 30-41 = layers 6-17 in passes 1/2/3; visits 42-47 = layers 18-23.
No single residual stream: the model keeps a list of completed 6-layer block sums ("blocks") plus an active block being summed.
Blocks are frozen at layers 0, 6, 12, 18. Each layer's attention input = attention_input(blocks, active) and MoE input =
mlp_input(blocks, active): a softmax over RMS-normed [blocks..., active] scored by a learned query vector. The output head uses
final_input(blocks, active) the same way, then RMSNorm and unembedding. At each pass boundary the block list resets to the prelude's
blocks; passes 2 and 3 receive the previous pass's two middle blocks as 4 messages (q, k, v, mlp) through rank-64 tanh gates:
x' = (1 - forget) * x + msg_gate * message. Readouts available now (all followed by the model's final RMSNorm + unembedding):
  act_in: the active block entering the visit; act_out: after the visit; mix_in/mix_out: final_input mix before/after;
  attn_in: the visit's own attention_input mix; mlp_in: the MoE input (its own mix, message gated in);
  msg: the MoE message (passes 2-3); tuned_*: affine translator trained to the final next-token distribution;
  future_*: low-rank translator trained to the first content token of the model's reply;
  concept_mlp_in: low-rank translator trained to name the clean words around the current word (typos injected in inputs).

SCORING: per question, up to 10 visits, one lens each, at the last prompt token; each visit's top-10 tokens vote for the first
of 5 options they name; pass if the intermediate gets strictly the most votes. Wrong votes hurt.

PER-VISIT STATS on our dev questions: "visit:g/w" = fraction of questions where that lens at that visit votes for the
intermediate (g) / for a wrong option (w).
{chr(10).join(lines)}

ROUTING STATS at the read position (mean over 120 prompts). attn_mix.V / final_mix.V = softmax weights over
[blocks in order..., active block]; norms.V = [active norm, block norms...]:
{fmt("attn_mix")}
{fmt("final_mix")}
{fmt("norms")}
Message gates (layer: [mean |forget|, mean |message gate|]): {json.dumps(stats["gates"])}
Parameters: {json.dumps(stats["params"])}

TASK: Propose up to 6 concrete readout designs most likely to raise the pass rate, using ONLY these primitives (each reads one
visit's inputs/outputs; no other position, no other visit):
  {{"op":"mix","module":"attention_input|mlp_input|final_input","module_layer":<int 0-23>,"blocks":"all|no_prelude|last1|last2","active_scale":<float>}}
     apply that layer's depth-mix module (its learned query) to this visit's blocks + scaled active block
  {{"op":"block","index":<int, 0 = first block, -1 = last>}}  read one completed block
  {{"op":"message","which":"q|k|v|mlp"}}  read one message (passes 2-3)
  {{"op":"sum","blocks":"all|no_prelude","active_scale":<float>}}  plain sum of blocks + scaled active
  {{"op":"center","base":<another primitive>,"minus":<another primitive>,"coef":<float>}}  base - coef*minus
and name the visits each should be read at. Then propose one 10-visit plan mixing your designs with the existing lenses.
Reply as JSON: {{"designs":[{{"name":...,"spec":...,"visits":[...],"why":...}}], "plan":[["lens_or_design_name", visit], ...], "notes":"..."}}"""

h = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}
c = requests.get("https://openrouter.ai/api/v1/credits", headers=h, timeout=30).json()["data"]
left = c["total_credits"] - c["total_usage"]
print(f"credits left ${left:.2f}; prompt {len(prompt)} chars")
assert left > a.max_usd, "not enough credit"
r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=h, timeout=900, json={
    "model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 12000, "usage": {"include": True},
    "reasoning": {"effort": "medium"}}).json()
out = r["choices"][0]["message"]["content"]
print("cost", r.get("usage", {}).get("cost"))
os.makedirs(os.path.join(ROOT, "dev", "wsp"), exist_ok=True)
open(os.path.join(ROOT, "dev", "wsp", "prompt.txt"), "w").write(prompt)
open(os.path.join(ROOT, "dev", "wsp", "reply.txt"), "w").write(out)
print(out)
