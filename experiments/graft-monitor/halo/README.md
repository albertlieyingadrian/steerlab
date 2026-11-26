# halo: GRAFT routing in Halo's async environmental GRPO

A port of GRAFT's gradient routing onto Halo's async environmental GRPO trainer, built and tested for Episode 1. It
ran 20 of 20 routed optimizer steps end to end; no study arm ran on it (the note's section 2.5 says why), so every
number in the notes comes from the trainer in `../trainer/`. Halo's own code is not edited.

| File | What it does |
|---|---|
| `graft_routing.py` | GRAFT's dual adapters and routed advantages, written as widened MLPs so a stock vLLM server can serve the checkpoint |
| `graft_trainer.py` | `install()`: registers the `pushback` environment and swaps in a trainer subclass that routes each flagged episode's update |
| `pushback_env.py` | the three-turn `pushback` environment (`../env.py`) as a Halo custom environment |
| `run_halo.py` | torchrun entry: installs the above, then runs Halo's stock environmental GRPO script unchanged |
| `modal_halo.py` | the vLLM rollout server and the trainer in one Modal container on two GPUs (costs money; logs its estimate to `../results/spend.jsonl`) |
| `configs/pushback_graft_qwen3_06b.yaml` | the Qwen3-0.6B training config |
| `tests/test_graft_routing.py` | CPU tests that the port matches GRAFT's routing |

Run the tests with `python -m pytest -q halo/tests` from `experiments/graft-monitor/`.
