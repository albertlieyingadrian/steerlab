"""Build ourdriver and install it, as an AgentEnv driver arm, into a clone of Scale's Cua Driver Bench port.

    python sync.py [--port PATH]   # default $PORT or ~/oss/agentenv-cua-driver-bench

1. Builds the Swift binary (release) and copies it to envs/ourdriver/bin/ourdriver.
2. Writes envs/ourdriver/pin.json (version, sha256) and mcp.json (the instructions and tools the binary lists).
3. In the port clone, writes:
   - envs/ourdriver/ (the driver env);
   - envs/desktop-<x>-macos-ourdriver/ for each computer;
   - tasks/macos-ourdriver-*.json, each the same as tasks/macos-cua-driver-*.json with only the computer env swapped;
   - evals/macos-ourdriver-bench-v2-trials.txt and evals/macos-ourdriver-dev.txt (the 18 r1 variants).
Nothing about the tasks, prompts, limits or grading changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER_SRC = HERE.parent / "driver"
ENV = HERE / "envs" / "ourdriver"
COMPUTERS = ["desktop-benchlab-macos", "desktop-s02-macos", "desktop-s03-macos", "desktop-s04-macos"]
ARM = "ourdriver"


def build() -> Path:
    subprocess.run(["swift", "build", "-c", "release"], cwd=DRIVER_SRC, check=True)
    binary = DRIVER_SRC / ".build" / "release" / "ourdriver"
    (ENV / "bin").mkdir(exist_ok=True)
    shutil.copy2(binary, ENV / "bin" / "ourdriver")
    return ENV / "bin" / "ourdriver"


def manifest(binary: Path) -> tuple[dict, str]:
    """The binary's instructions, tools and version, from an MCP handshake (no tool is called)."""
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    out = subprocess.run(
        [str(binary)],
        input="\n".join(json.dumps(m) for m in lines) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    replies = {r["id"]: r["result"] for r in map(json.loads, out.splitlines()) if "id" in r}
    return (
        {"instructions": replies[1]["instructions"], "tools": replies[2]["tools"]},
        replies[1]["serverInfo"]["version"],
    )


def install(port: Path) -> None:
    shutil.copytree(ENV, port / "envs" / ARM, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    for computer in COMPUTERS:
        src = (port / "envs" / computer / "env.toml").read_text()
        assert 'driver = "cua-driver"' in src, computer
        dest = port / "envs" / f"{computer}-{ARM}"
        dest.mkdir(exist_ok=True)
        (dest / "env.toml").write_text(src.replace('driver = "cua-driver"', f'driver = "{ARM}"'))
    written = 0
    for task in sorted((port / "tasks").glob("macos-cua-driver-*.json")):
        text = task.read_text()
        for computer in COMPUTERS:
            text = re.sub(rf'"{computer}"', f'"{computer}-{ARM}"', text)
        (port / "tasks" / task.name.replace("macos-cua-driver-", f"macos-{ARM}-")).write_text(text)
        written += 1
    trials = (port / "evals" / "macos-cua-driver-bench-v2-trials.txt").read_text()
    (port / "evals" / f"macos-{ARM}-bench-v2-trials.txt").write_text(
        trials.replace("macos-cua-driver-", f"macos-{ARM}-")
    )
    dev = [t for t in dict.fromkeys(trials.split()) if "-r" not in t.split("cdb-")[-1] or "-r1-" in t]
    (port / "evals" / f"macos-{ARM}-dev.txt").write_text(
        "\n".join(t.replace("macos-cua-driver-", f"macos-{ARM}-") for t in dev) + "\n"
    )
    (port / "evals" / "macos-cua-driver-dev.txt").write_text("\n".join(dev) + "\n")
    print(f"installed into {port}: driver env, {len(COMPUTERS)} computers, {written} tasks, {len(dev)} dev trials")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--port", type=Path, default=Path(os.environ.get("PORT", Path.home() / "oss/agentenv-cua-driver-bench"))
    )
    p.add_argument("--binary", type=Path, help="Install this saved binary instead of building the current source")
    args = p.parse_args()
    if args.binary:
        (ENV / "bin").mkdir(exist_ok=True)
        shutil.copy2(args.binary, ENV / "bin" / "ourdriver")
        binary = ENV / "bin" / "ourdriver"
    else:
        binary = build()
    man, version = manifest(binary)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    (ENV / "mcp.json").write_text(json.dumps(man, indent=2) + "\n")
    (ENV / "pin.json").write_text(json.dumps({"version": version, "sha256": digest}, indent=2) + "\n")
    print(f"ourdriver {version} sha256 {digest}")
    install(args.port)


if __name__ == "__main__":
    main()
