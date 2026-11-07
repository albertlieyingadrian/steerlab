"""Shared pieces for the agentic alignment pilot: the models, the API client, the spending cap and the rollout cache.

Every call goes to OpenRouter's OpenAI-compatible chat endpoint. Every paid call is logged to results/ledger.jsonl;
the run stops once BUDGET_USD is spent. Finished rollouts are cached in results/rollouts.jsonl, so a rerun only pays
for what is missing.
"""

import json
import threading
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results"
LEDGER = RESULTS / "ledger.jsonl"
ROLLOUTS = RESULTS / "rollouts.jsonl"
URL = "https://openrouter.ai/api/v1/chat/completions"
BUDGET_USD = 30.0
ROLLOUTS_PER_CELL = 20

# The five models of this first batch: five labs, small and mid-sized, all with tool calling.
MODELS = {
    "claude-haiku-5.5": "anthropic/claude-haiku-5.5",
    "gpt-6-luna": "openai/gpt-6-luna",
    "gemini-3.8-flash": "google/gemini-3.8-flash",
    "deepseek-v4.1-flash": "deepseek/deepseek-v4.1-flash",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
}
# Judges only the two scenarios whose outcome is written in prose (blackmail, leaking); not one of the five above.
JUDGE = "anthropic/claude-sonnet-5.5"
# Listed in the article as coming soon.
COMING_SOON = [
    "Claude Opus 5.5", "Claude Sonnet 5.5", "Claude Fable 5.1", "GPT-6 Sol", "GPT-6 Astra", "GPT-6.1 Terra",
    "GPT-6.1 Sol", "Gemini 4 Argon", "Gemini 3.7 Flash", "Grok 4.7", "Grok 4.6", "Kimi K3.1 Pro",
    "Mistral Large 4", "DeepSeek V4 Pro", "Qwen3.8 Max", "Qwen3.8 Flash", "Qwen3.8-2.4T", "Qwen3-8B (local)",
    "Llama 4 (local)",
]  # fmt: skip

_LOCK = threading.Lock()


def secret(name):
    """An API key from the environment, else from the repo's .env files (or the folder named by $STEERLAB_ENV_DIR)."""
    import os

    from dotenv import dotenv_values

    if os.environ.get(name):
        return os.environ[name]
    folder = Path(os.environ.get("STEERLAB_ENV_DIR", REPO))
    return {**dotenv_values(folder / ".env"), **dotenv_values(folder / ".env.local")}[name]


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())


def charge(usd, what):
    with _LOCK:
        RESULTS.mkdir(exist_ok=True)
        with LEDGER.open("a") as handle:
            handle.write(json.dumps({"usd": round(usd, 6), "what": what}) + "\n")
        total = spent()
    if total > BUDGET_USD:
        raise SystemExit(f"budget of ${BUDGET_USD} used up (${total:.2f}); stopping")


def chat(model, messages, tools=None, temperature=1.0, max_tokens=4000, what="call"):
    """One chat completion. Returns the assistant message dict (content and any tool_calls), or None."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if tools:
        body["tools"] = tools
    headers = {"Authorization": f"Bearer {secret('OPENROUTER_API_KEY')}"}
    for attempt in range(4):
        try:
            data = requests.post(URL, headers=headers, json=body, timeout=300).json()
            charge(data.get("usage", {}).get("cost", 0.0) or 0.0, what)
            return data["choices"][0]["message"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            print(f"  {model} {what}: attempt {attempt + 1} failed ({type(error).__name__})")
            time.sleep(5 * (attempt + 1))
    return None


def load_rollouts():
    if not ROLLOUTS.exists():
        return []
    return [json.loads(line) for line in ROLLOUTS.read_text().splitlines() if line.strip()]


def save_rollout(record):
    with _LOCK:
        RESULTS.mkdir(exist_ok=True)
        with ROLLOUTS.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
