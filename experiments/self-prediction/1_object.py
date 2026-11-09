"""
Step 1: object-level behaviour. Each model answers every prompt (greedy, up to 8 tokens); each answer's properties
(first letter, more than 5 letters, even number) are recorded. Writes results/object_<model>.json.

    .venv/bin/python experiments/self-prediction/1_object.py
"""

from selfpred import MODELS, chat, dump, free, greedy, load, prompts, properties


def object_level(name, adapter=None, tag=None):
    model, tok = load(name, adapter)
    ps = prompts()
    answers = greedy(model, tok, [chat(tok, t) for _, _, t in ps])
    free(model)
    out = {
        pid: {"kind": kind, "answer": a, "props": properties(kind, a)}
        for (pid, kind, _), a in zip(ps, answers, strict=True)
    }
    dump(f"object_{tag or name}.json", out)
    print(f"{tag or name}: {sum(bool(v['props']) for v in out.values())}/{len(out)} answers usable", flush=True)
    return out


def main():
    for name in MODELS:
        object_level(name)


if __name__ == "__main__":
    main()
