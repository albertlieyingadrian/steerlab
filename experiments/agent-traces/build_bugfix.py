"""Build the dissimilar task family: small Python bug fixes, as Harbor tasks the guide's harness runs unchanged.

    .venv/bin/python experiments/agent-traces/build_bugfix.py            # writes data/bugfix/, prints counts
    .venv/bin/python experiments/agent-traces/build_bugfix.py --push albertlieadrian/agent-traces-bugfix-harbor

SmolDataEnvs asks a question about data files and grades an answer in answer.txt. These tasks ask the agent to edit
code: each is an MBPP (sanitized; train, validation and prompt splits, 170 problems) reference solution with one
mechanical bug put in by a mutation of its syntax tree (a comparison flipped, + and - swapped, an integer constant
off by one, `and` and `or` swapped). A mutant is kept only if the original passes all of the problem's asserts and the
mutant fails at least one without crashing at import; one mutant per problem, picked with seed 0. The agent sees the
problem statement, one example call and the buggy file; the grader runs every assert on the edited file, hidden from
the agent, and writes reward 1 or 0. No answer.txt is involved.

The file reaches the sandbox through the task's healthcheck command (written only if absent, so a retried
healthcheck cannot undo the agent's edit). Tasks run on the SmolDataEnvs image, the same as the in-distribution tasks.
"""

import argparse
import ast
import base64
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "bugfix"
IMAGE = (
    "docker.io/savatar101/env-data-agent-train@sha256:7b3f7e22780b431ce0d43717dcef6875cde84baeee824cc87e968374d62d3470"
)

SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
         ast.Add: ast.Sub, ast.Sub: ast.Add, ast.And: ast.Or, ast.Or: ast.And}  # fmt: skip


def mutants(code: str):
    """Every single-site mutant of `code`, as (label, source)."""
    n_sites = len(_sites(ast.parse(code)))
    for k in range(n_sites):
        tree = ast.parse(code)
        kind, node, i = _sites(tree)[k]  # the same walk order on a fresh tree, so site k is the same node
        if kind == "cmp":
            label = f"{type(node.ops[i]).__name__}->{SWAPS[type(node.ops[i])].__name__}"
            node.ops[i] = SWAPS[type(node.ops[i])]()
        elif kind in ("bin", "bool"):
            label = f"{type(node.op).__name__}->{SWAPS[type(node.op)].__name__}"
            node.op = SWAPS[type(node.op)]()
        else:
            label = f"{node.value}->{node.value + 1}"
            node.value += 1
        yield f"{label}@line{node.lineno}", ast.unparse(tree)


def _sites(tree: ast.AST) -> list:
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sites += [("cmp", node, i) for i, op in enumerate(node.ops) if type(op) in SWAPS]
        elif isinstance(node, ast.BinOp) and type(node.op) in SWAPS:
            sites.append(("bin", node, None))
        elif isinstance(node, ast.BoolOp):
            sites.append(("bool", node, None))
        elif isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 10:
            sites.append(("int", node, None))
    return sites


def run_asserts(source: str, imports: list[str], asserts: list[str]) -> tuple[bool, bool]:
    """(imports cleanly, all asserts pass), in a subprocess with a 5 s limit."""
    prog = "\n".join(imports) + "\n" + source + "\nok = 0\n"
    for a in asserts:
        prog += f"try:\n    {a}\n    ok += 1\nexcept Exception:\n    pass\n"
    prog += f"print('PASS' if ok == {len(asserts)} else 'FAIL')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return True, False
    finally:
        Path(f.name).unlink()
    out = r.stdout.strip()
    return out in ("PASS", "FAIL"), out == "PASS"


INSTRUCTION = """You are a software engineer working in a sandbox. Use your tools to inspect and edit the code.

The file /home/user/input/solution.py defines `{func}`. It should do this:

{prompt}

Example: `{example}` must hold.

The function has a bug and some calls return the wrong result. Find the bug and fix it by editing /home/user/input/solution.py in place. Keep the function's name and signature. When the fix is done, stop.
"""

TEST_SH = """#!/usr/bin/env bash
set -u
mkdir -p /logs/verifier
cd /home/user/input
if python3 /tests/hidden_tests.py; then echo "1.0" > /logs/verifier/reward.txt; else echo "0.0" > /logs/verifier/reward.txt; fi
exit 0
"""


def task_toml(name: str, b64: str, func: str) -> str:
    target = "/home/user/input/solution.py"
    hc = f"mkdir -p /home/user/input && ([ -f {target} ] || (echo {b64} | base64 -d > {target})) && [ -s {target} ]"
    return f'''schema_version = "1.2"
artifacts = []

[task]
name = "agent-traces-bugfix/{name}"
description = "Fix the bug in {func}"
authors = []
keywords = ["bugfix", "python", "mbpp"]

[metadata]
source_dataset = "google-research-datasets/mbpp (sanitized)"
kaggle_dataset_name = "none/bugfix"

[environment]
docker_image = "{IMAGE}"
build_timeout_sec = 600.0
os = "linux"
cpus = 1
memory_mb = 1024
storage_mb = 5120
gpus = 0
allow_internet = true
mcp_servers = []

[environment.healthcheck]
command = "{hc}"
interval_sec = 2.0
timeout_sec = 120.0
start_period_sec = 5.0
start_interval_sec = 2.0
retries = 30

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 600.0  # as in SmolDataEnvs: past 10 minutes the agent is almost always stuck in a loop
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="tasks to keep")
    p.add_argument("--push", default=None, help="private Hub dataset to upload data/bugfix to")
    a = p.parse_args()
    from datasets import concatenate_datasets, load_dataset

    d = load_dataset("google-research-datasets/mbpp", "sanitized")
    probs = concatenate_datasets([d["train"], d["validation"], d["prompt"]])
    rng = random.Random(0)
    kept = []
    for row in probs:
        imports, asserts = row["test_imports"], row["test_list"]
        if not run_asserts(row["code"], imports, asserts)[1]:
            continue
        try:
            fn = [n.name for n in ast.parse(row["code"]).body if isinstance(n, ast.FunctionDef)]
        except SyntaxError:
            continue
        func = next((f for f in fn if f in asserts[0]), None)
        if not func:
            continue
        good = []
        for label, src in mutants(row["code"]):
            imports_ok, passes = run_asserts(src, imports, asserts)
            if imports_ok and not passes:
                good.append((label, src))
        if good:
            label, src = rng.choice(good)
            kept.append({"task_id": row["task_id"], "func": func, "prompt": row["prompt"], "mutation": label,
                         "source": src, "imports": imports, "asserts": asserts})  # fmt: skip
    rng.shuffle(kept)
    kept = sorted(kept[: a.n], key=lambda r: r["task_id"])
    tasks = OUT / "tasks"
    manifest = []
    for i, r in enumerate(kept):
        name = f"{i:04d}_bugfix_mbpp{r['task_id']}"
        t = tasks / name
        (t / "tests").mkdir(parents=True, exist_ok=True)
        body = "\n".join(r["imports"]) + ("\n\n" if r["imports"] else "") + r["source"] + "\n"
        b64 = base64.b64encode(body.encode()).decode()
        (t / "task.toml").write_text(task_toml(name, b64, r["func"]))
        (t / "instruction.md").write_text(INSTRUCTION.format(func=r["func"], prompt=r["prompt"].strip(),
                                                             example=r["asserts"][0].removeprefix("assert ")))  # fmt: skip
        hidden = "import sys\nsys.path.insert(0, '/home/user/input')\n" + "\n".join(r["imports"]) + \
            "\nfrom solution import *\nfailed = 0\n" + "".join(
                f"try:\n    {x}\nexcept Exception:\n    failed += 1\n" for x in r["asserts"]) + \
            "sys.exit(1 if failed else 0)\n"  # fmt: skip
        (t / "tests" / "hidden_tests.py").write_text(hidden)
        (t / "tests" / "test.sh").write_text(TEST_SH)
        manifest.append({"name": name, "task_id": r["task_id"], "func": r["func"], "mutation": r["mutation"],
                         "n_asserts": len(r["asserts"]), "sha256": hashlib.sha256(body.encode()).hexdigest()})  # fmt: skip
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    print(f"{len(probs)} problems, {len(kept)} tasks written to {OUT.relative_to(HERE)}")
    kinds = {}
    for m in manifest:
        k = m["mutation"].split("->")[0]
        kinds[k] = kinds.get(k, 0) + 1
    print("mutation kinds:", kinds)
    if a.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.push, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=a.push, repo_type="dataset", folder_path=OUT, path_in_repo=".")
        print(f"pushed to {a.push} (private)")


if __name__ == "__main__":
    main()
