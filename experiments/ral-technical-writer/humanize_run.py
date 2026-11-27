"""
The lab's humanizer as a baseline: the private Humanize pipeline (private/slop-portal/humanize.py, lab only) on the
base model's sample-0 abstracts for the first N held-out titles.

    python humanize_run.py gen [N]    # phase 1, writer up only: every writer call, into the cache
    python humanize_run.py arms [N]   # phase 2, gate up only: the full pipeline from the cache -> samples/humanizer_heldout.jsonl

Two phases, as in the humanizer's own ablation (ablation/run.py), so the 2-GPU writer and the 1-GPU gate never run
at once; the writer's replies are cached by message, seed and temperature, so phase 2 asks it nothing new.

Settings: the 2x2x2 ablation's delivering arm L0-Gs-N0 (no locations, per-sentence gate, no narrowing), form
"draft", 4 candidates, writer Llama-3.3-70B-Instruct, gate Mistral-Small-24B-Instruct-2501, no detectors in the
loop (detectors=False, luar=False), so no detector shapes the rewrite. When the verdict is "keep your draft",
the delivered text is the draft itself, as in the product; the share of such cases is reported.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORTAL = HERE.parents[1] / "private" / "slop-portal"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PORTAL))
import humanize
import textops
import voiceprint

import common as C

URL = "https://albert-lie-adrian--ral-writer-serve-{}.us-east.modal.run/v1/chat/completions"
MODELS = {"writer": "meta-llama/Llama-3.3-70B-Instruct", "gate": "mistralai/Mistral-Small-24B-Instruct-2501"}
LOCAL = C.CACHE / "vol"
DATA = C.CACHE / "humanizer"


def modal_chat(name: str):
    token = (C.CACHE / "llm_token").read_text().strip()

    def chat(messages, seed=0, temperature=0.8, max_tokens=1024, json_mode=False):
        body = {"model": MODELS[name], "messages": messages, "seed": seed, "temperature": temperature,
                "top_p": humanize.GEN["top_p"] if temperature > 0 else 1.0, "max_tokens": max_tokens,
                "repetition_penalty": 1.0} | ({"response_format": {"type": "json_object"}} if json_mode else {})  # fmt: skip
        for attempt in range(8):
            try:
                req = urllib.request.Request(URL.format(name), data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {token}"})  # fmt: skip
                with urllib.request.urlopen(req, timeout=1800) as r:
                    d = json.loads(r.read())
                ch = d["choices"][0]
                return {"text": humanize.clean(ch["message"].get("content") or ""),
                        "tokens": d.get("usage", {}).get("completion_tokens"),
                        "finished": ch.get("finish_reason") == "stop"}  # fmt: skip
            except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
                if attempt == 7 or getattr(e, "code", None) in (400, 401, 402):
                    raise
                time.sleep(15 * (attempt + 1))
        raise RuntimeError("unreachable")

    return chat


class Portal:
    def __init__(self) -> None:
        DATA.mkdir(parents=True, exist_ok=True)
        self.cache = textops.Cache(DATA / "cache.sqlite")

    def detectors_on(self) -> list:
        return []


def main(phase: str = "arms", n: int = 100) -> None:
    ji = {"id": "modal:" + MODELS["gate"], "name": MODELS["gate"], "independence": "independent",
          "label": f"independent judge: {MODELS['gate']}", "backend": "modal:gate", "model": MODELS["gate"]}  # fmt: skip
    hz = humanize.Humanizer(Portal(), modal_chat("writer"), "modal:" + MODELS["writer"], detectors=False, luar=False,
                            judge=phase == "arms", judge_chat=modal_chat("gate"), judge_info=ji)  # fmt: skip
    src = {r["appl_id"]: r for r in C.read_jsonl(LOCAL / "samples" / "base_heldout.jsonl") if r["i"] == 0}
    held = [x for x in C.read_jsonl(C.CACHE / "heldout.jsonl")[:n] if src[x["appl_id"]]["answer"]]
    out_p = LOCAL / "samples" / "humanizer_heldout.jsonl"

    def one(x):
        draft = src[x["appl_id"]]["answer"]
        prof = voiceprint.draft_profile(draft)
        rep = hz.run(draft, prof, n=4, form="draft", locate=False, gate="sentence", narrow=False)
        best = next((c for c in rep["candidates"] if c["i"] == rep["best"]), None)
        return {"appl_id": x["appl_id"], "i": 0, "answer": best["text"] if best else draft, "source": draft,
                "delivered": "rewrite" if best else "draft", "verdict": rep["verdict"]["kind"]}  # fmt: skip

    t0 = time.time()
    with ThreadPoolExecutor(24) as ex:
        rows = list(ex.map(one, held))
    if phase == "gen":
        print(f"humanizer gen: {len(rows)} drafts' writer calls cached in {time.time() - t0:.0f}s")
        return
    C.write_jsonl(out_p, rows)
    dr = sum(r["delivered"] == "draft" for r in rows)
    print(f"humanizer: {len(rows)} texts in {time.time() - t0:.0f}s; {dr} delivered the draft unchanged")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100)
