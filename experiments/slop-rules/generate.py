"""Generate the matched AI side: every prompt x every open-weights model, through OpenRouter.

    python generate.py --smoke      # 2 prompts per model, prints cost and reasoning-token use
    python generate.py              # everything; resumable (skips cached replies)

Hard cap in code: CAP_USD over the spend ledger (results/spend.jsonl, cost as reported by OpenRouter per call).
Key: OPENROUTER_API_KEY_2 from the environment or the repository root's .env (never printed). Open-weights models only.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from common import CACHE, GEN, OPEN_MODELS, RESULTS

CAP_USD = 15.0
LEDGER = RESULTS / "spend.jsonl"
URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE, TOP_P = 0.7, 1.0
# Reasoning off where the endpoint allows it; GLM-5.3 refuses "off", so it gets the lowest effort
# and 2,500 extra tokens of headroom (reasoning tokens are logged per reply).
REASONING = {"z-ai/glm-5.3": {"effort": "low"}}
_lock = threading.Lock()


def key() -> str:
    if os.environ.get("OPENROUTER_API_KEY_2"):
        return os.environ["OPENROUTER_API_KEY_2"]
    env = Path(__file__).resolve().parents[2] / ".env"  # the repository root's .env, never printed
    if env.exists():
        for line in env.read_text().split("\n"):
            if line.startswith("OPENROUTER_API_KEY_2="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENROUTER_API_KEY_2 not set")


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line).get("cost", 0.0) for line in LEDGER.read_text().split("\n") if line.strip())


def cached(model: str) -> dict[str, dict]:
    path = GEN / f"{model.replace('/', '__')}.jsonl"
    if not path.exists():
        return {}
    return {r["pid"]: r for r in map(json.loads, filter(None, path.read_text().split("\n"))) if r.get("text")}


def call(client: httpx.Client, model: str, p: dict) -> dict | None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": p["prompt"]}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": p["max_tokens"] + (2500 if model in REASONING else 0),
        "reasoning": REASONING.get(model, {"enabled": False}),
        "usage": {"include": True},
    }
    err: object = None
    for attempt in range(4):
        with _lock:
            if spent() >= CAP_USD:
                raise SystemExit(f"cap ${CAP_USD} reached")
        try:
            data = client.post(URL, json=body, timeout=180).json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            err, data = repr(e), None
        if data and "choices" in data:
            usage = data.get("usage") or {}
            msg = data["choices"][0]["message"]
            row = {
                "pid": p["pid"],
                "model": model,
                "text": (msg.get("content") or "").strip(),
                "finish": data["choices"][0].get("finish_reason"),
                "provider": data.get("provider"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost": usage.get("cost", 0.0),
            }
            with _lock:
                LEDGER.parent.mkdir(parents=True, exist_ok=True)
                with LEDGER.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "model": model,
                                "pid": p["pid"],
                                "cost": row["cost"],
                                "tokens": row["completion_tokens"],
                            }
                        )
                        + "\n"
                    )
            return row
        if data is not None:
            err = data.get("error", data)
        time.sleep(2 + 4 * attempt)
    print(f"  failed {model} {p['pid']}: {str(err)[:160]}")
    return None


def run(model: str, prompts: list[dict], workers: int) -> None:
    done = cached(model)
    todo = [p for p in prompts if p["pid"] not in done]
    path = GEN / f"{model.replace('/', '__')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {key()}", "X-Title": "research-eval"}
    with httpx.Client(headers=headers) as client, ThreadPoolExecutor(workers) as pool:
        for row in pool.map(lambda p: call(client, model, p), todo):
            if row and row["text"]:
                with _lock, path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{model}: {len(cached(model))}/{len(prompts)} cached; spent ${spent():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=OPEN_MODELS)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    prompts = [json.loads(line) for line in (CACHE / "prompts.jsonl").read_text().split("\n") if line.strip()]
    if args.smoke:
        prompts = [prompts[0], prompts[-1]]
    for m in args.models:
        run(m, prompts, args.workers)
    if args.smoke:
        for m in args.models:
            for r in cached(m).values():
                if r["pid"] in {p["pid"] for p in prompts}:
                    print(
                        f"{m:40} {r['pid'][:30]:30} tok={r['completion_tokens']} reason={r['reasoning_tokens']} "
                        f"${r['cost']:.5f} {r['finish']} | {r['text'][:90]!r}"
                    )


if __name__ == "__main__":
    main()
