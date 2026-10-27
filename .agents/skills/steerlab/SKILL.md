---
name: steerlab
description: "Live activation steering on small open models with the repo's `steerlab` CLI: find an SAE feature or a named direction, write a JSON control file, run a steered generation and read its per-token timeline, or launch the strength-vs-coherence sweep. Use when the user asks to steer a model, look up a feature, build a direction, check an SAE, or measure where steering stops working (Part 10 of the course)."
allowed-tools: Bash(.venv/bin/steerlab --help), Bash(.venv/bin/steerlab doctor:*), Bash(.venv/bin/steerlab schema:*), Bash(.venv/bin/steerlab control validate:*), Bash(.venv/bin/steerlab directions list:*), Bash(.venv/bin/steerlab diff doctor:*)
user-invocable: true
---

# steerlab

The CLI lives at `.venv/bin/steerlab` in this repo (never `pip install` it; it is already installed editably). Every
command prints `--help`. Heavy models load lazily, so `--help` is instant and the first real command may take a minute.

## Start here

1. `.venv/bin/steerlab doctor` — device, which models and SAE layers are cached (with download sizes), whether the
   gated Gemma model is accessible, which directions exist. It never prints a token. Add `--online` to ask the Hub
   about Gemma access.
2. Pick the model: `qwen` (`Qwen/Qwen3-1.7B`, reproducible for anyone) or `gemma` (`google/gemma-3-1b-it`, gated;
   the user must have accepted the licence). Default layer: Qwen 14, Gemma 13.

## Find something to steer with

- A named direction (difference of means; sandbagging, refusal, chinese):
  `.venv/bin/steerlab directions list --model qwen` and, when missing,
  `.venv/bin/steerlab directions build --model qwen --concept chinese`. Directions carry their own calibration
  (strength 1.0 = one class gap).
- An SAE feature by label (Gemma only): `.venv/bin/steerlab features search "Chinese" --model gemma` — offline after
  one download of Neuronpedia's export; add `--regex` or `--word` for stricter matching.
- An SAE feature by contrast (Qwen): `.venv/bin/steerlab features contrast "an English text" "its Chinese translation"
  --model qwen --top 5` — ranks features by activation difference and labels the top ones on demand (the first
  labelling run downloads the judge model, 16 GB; it prints the time estimate first — ask the user before that).
- Before trusting an SAE on the instruct model: `.venv/bin/steerlab sae check --model qwen`.

## Write the control file

`.venv/bin/steerlab schema control` prints the JSON Schema. The shape is

```json
{"model": "Qwen/Qwen3-1.7B",
 "entries": [{"source": {"kind": "direction", "name": "chinese"}, "strength": 2.0},
             {"source": {"kind": "sae", "release": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", "layer": 14, "feature": 123},
              "strength": 1.0, "layers": [14]}]}
```

Validate it with `.venv/bin/steerlab control validate control.json`. Strength is in calibrated units (1.0 = the
source's typical activation, or one class gap for a direction); 0 is bit-for-bit identical to no steering; above
about 4 most sources stop being coherent. Entries naming the same source add up.

**Changing it from the command line:** `.venv/bin/steerlab control set chinese=1.5` moves a direction's fader,
`.venv/bin/steerlab control set feature:24946=4` names a feature of the model's own SAE release at its default layer
(`--sae-layer` for another), and `.venv/bin/steerlab control off` clears every entry. Both default to `./control.json`
(`--file` for another), keep the other entries, and print the file before and after.

**Editing the file while a generation runs:** the engine re-reads it on every token. `control set` and `control off`
write a temp file and rename it into place; do the same when writing it by hand (`mv control.json.tmp control.json`).
A half-written file is ignored with one warning and the last valid state stays in force. Earlier generated text stays in the model's context, so a change mid-reply steers the
rest of the reply, not what was already said (restart the reply with `r` in `chat` to start clean, or `c` to clear the whole conversation).

## Run

- One-shot, machine-readable: `.venv/bin/steerlab generate --model qwen --control control.json --json "Tell me about Paris."`
  prints `{text, token_ids, prompt_length, finish_reason, timeline}`; each timeline entry says which entries applied
  at that token, the added norm per layer and the residual norm, and `changed: true` where the control file changed.
  Use `--events events.json` (a list of `{token_index, control}`) to script a mid-reply change instead of editing the
  file by hand; `--direction chinese=tutorials/outputs/p10_01/directions/qwen/chinese.npz` and `--sae-layer 14` load
  what the entries need.
- Interactive two-terminal demo: `.venv/bin/steerlab chat --model qwen --control control.json` in one terminal (from the
  repo root; create the file first with `control off`), and `control set` from the other. In `chat`, `r` redoes the
  reply, `c` clears the history, `q` quits. The demo flows are in `docs/demos/2026-10-25-steering-demo-flows.md`.
- In a browser: `.venv/bin/steerlab serve` (Qwen3-8B by default; `--model` takes any downloaded model key) serves the
  visitor chat at `http://127.0.0.1:8765/` and the admin desk at `http://localhost:8765/admin`. It is long-running:
  ask the user before starting it, and run it in the background. A running server is driven only through its JSON
  API (headerless POSTs pass the admin guard; with `--host` beyond loopback, add `Authorization: Bearer <token>`,
  the token `serve` printed): `POST /admin/api/control {"name": "chinese", "strength": 1.5}` (or
  `"feature:24946"` after `POST /admin/api/feature {"id": 24946}`), `/admin/api/all-off`, `/admin/api/prompt
  {"text": ...}` (`null` resets), `/admin/api/model {"key": "qwen"}`, `/admin/api/stop`, `/admin/api/redo
  {"conversation": ...}`, `/admin/api/clear`; `GET /admin/api/state` reads the faders, prompt and model. The
  conversations themselves, with a marker wherever steering changed, are on `GET /admin/events` (server-sent
  events): its first event, `snapshot`, holds every conversation so far, so reading one event is a one-shot view.
  `control set` does not reach a server.

## Measure

`.venv/bin/steerlab sweep generate --model qwen --concept chinese` then `sweep judge` and `sweep summarize` produce
the strength-vs-coherence curves for five sources. A sweep is long-running (about an hour of generation, hours of
judging on a laptop) and resumable (`--resume`): get the user's go-ahead before starting one and run it in the
background.

## Rules

- Never print or copy a Hugging Face token; `doctor` reports only "present"/"absent".
- Do not download the gated Gemma model unless `doctor --online` says access is granted.
- Read the Part 10 lessons under `tutorials/part10_capstone_live_steering/` for worked examples and recorded outputs.
