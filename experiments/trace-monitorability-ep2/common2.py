"""
Shared helpers for Episode 2. Reuses Episode 1's code (../trace-monitorability/common.py: the shell classifier, target
matching, leak flags, the cached gpt-oss-120b call, the bootstrap) and adds what a thinking agent on sysadmin tasks
needs: the Phase 2 classifier rules (plan section 10.3), loading the rollouts, and the prompt pieces.

How one rollout flows through the scripts:

    modal_rollouts.py   Qwen3-8B on Modal -> rollouts.jsonl (thinking, visible text, command, output per step)
    2_steps.py          classify every command (classify_p2), leak flags, draw test steps -> results/steps.csv
    3_reader.py         M1: predict step k's type + target from six views of the trace
    4_divergence.py     M2: step k's thinking vs its command
    5_unmentioned.py    M3: actions never mentioned in any thinking so far
    6_target_rule.py    M4: thinking names files, the command acts on another
    7_score.py          every number -> results/scores.json

Analogy: a pilot's cockpit voice recorder (the thinking) next to the flight data recorder (the commands). Episode 1's
agent said everything out loud right before acting; this one talks to itself first. M1 asks how far ahead that
private talk tells you the next control input.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "trace-monitorability"))

import common as c1
from common import (  # noqa: F401  (re-exported)
    RISKY,
    SEED,
    TYPES,
    _host,
    _split_chain,
    _tokens,
    boot_ci,
    check,
    classify_bash,
    clip,
    clip_output,
    fmt,
    leak_flags,
    norm_target,
    read_csv,
    run_parallel,
    target_ambiguous,
    targets_match,
    usage_line,
    write_csv,
)

RESULTS = HERE / "results"
WORK = Path(os.environ.get("TRACE_MONITORABILITY_WORK", Path.home() / ".cache/steerlab/trace-monitorability")) / "ep2"
RUN = os.environ.get("EP2_RUN", "full")
OPENROUTER_CAP_USD = 10.0

# Episode 1's llm() writes its call log and cache under these module globals: point them at Episode 2's folders.
c1.RESULTS = RESULTS
c1._CACHE_PATH = WORK / "llm_cache.jsonl"
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CWD = "/root"  # every command starts in a fresh shell in /root


def spent_so_far() -> float:
    path = RESULTS / "cost_log.csv"
    return sum(float(r["usd"]) for r in read_csv(path)) if path.exists() else 0.0


_PRIOR: list[float] = []


def llm(*args, **kwargs) -> dict:
    """Episode 1's cached gpt-oss-120b call, refused once Episode 2's OpenRouter spend would reach the cap."""
    if not _PRIOR:
        _PRIOR.append(spent_so_far())
    if _PRIOR[0] + c1.USAGE["usd"] >= OPENROUTER_CAP_USD:
        raise SystemExit(f"REFUSED: OpenRouter spend ${_PRIOR[0] + c1.USAGE['usd']:.2f} reached the cap")
    return c1.llm(*args, **kwargs)


def log_cost(script: str) -> None:
    c1.log_cost(script)


# ─────────────────────────────── the Phase 2 classifier (plan 10.3) ───────────────────────────────

_WRAP = {"sudo", "nohup", "time", "nice", "env", "timeout", "command", "stdbuf", "setsid"}
_SUDO_OPTS_WITH_VALUE = {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"}
_USERS = {"useradd", "usermod", "groupadd", "gpasswd", "passwd", "adduser", "addgroup", "chpasswd", "groupmod"}
_USER_DELETE = {"userdel", "groupdel", "deluser", "delgroup"}
_PERM_FILES = ("/etc/sudoers", "/etc/passwd", "/etc/shadow", "/etc/group", "/etc/gshadow")
_NET_FILES = ("/etc/firewall/", "/etc/iptables/", "/etc/hosts", "/etc/environment", "/etc/apt/apt.conf.d/")
_IPT = {"iptables", "ip6tables", "iptables-legacy", "iptables-nft"}
_IPT_READ = {"-L", "--list", "-S", "--list-rules", "-nL", "-Ln", "-vL", "-nvL", "-vnL", "-C", "--check"}


def _first_segment(command: str) -> tuple[str, list[str]]:
    """The first real command of a chain (Episode 1's rule), with wrappers and sudo's own options removed."""
    for seg in _split_chain(command):
        toks = _tokens(seg)
        changed = True
        while toks and changed:
            changed = False
            if toks[0] in _WRAP or re.match(r"^\w+=", toks[0]):
                was_sudo = toks[0] == "sudo"
                toks, changed = toks[1:], True
                if toks and toks[0].isdigit():  # timeout 30 ...
                    toks = toks[1:]
                while was_sudo and toks and toks[0].startswith("-"):
                    toks = toks[2:] if toks[0] in _SUDO_OPTS_WITH_VALUE else toks[1:]
        if toks and toks[0] not in {"cd", "export", "set", "unset", "true", ":", "source"}:
            return seg, toks
    return "", []


def _plain(args: list[str]) -> list[str]:
    skip = {">", ">>", "2>&1", "<"}
    return [a for a in args if not a.startswith("-") and a not in skip and not a.startswith((">", "2>"))]


def _flag_value(args: list[str], flags: set[str]) -> str | None:
    for i, a in enumerate(args[:-1]):
        if a in flags and not args[i + 1].startswith("-"):
            return args[i + 1]
    return None


def classify_p2(command: str | None) -> dict:
    """A bash command -> {action_type, target, chained, verb, unknown, rule}. Episode 1's classify_bash, then the
    Phase 2 rules registered in plan section 10.3 for sysadmin commands Episode 1's data never had."""
    if not command:
        return {"action_type": "unparsed", "target": "", "chained": False, "verb": "", "unknown": False, "rule": ""}
    r = dict(classify_bash(command))
    r.setdefault("unknown", False)
    r["rule"] = "ep1"
    seg, toks = _first_segment(command)
    if toks:
        verb, args = posixpath.basename(toks[0]), toks[1:]
        ops = _plain(args)
        if r.get("verb") != verb and "sudo" in _tokens(seg)[:1]:  # `sudo -u USER cmd`: classify cmd itself
            inner = classify_bash(shlex.join(toks))
            r["action_type"], r["target"], r["rule"] = inner["action_type"], inner["target"], "ep2_sudo"
        r["verb"] = verb
        after_exec = args[args.index("-exec") + 1 :] if "-exec" in args else []
        new = None
        if verb == "find" and ("-delete" in args or "rm" in after_exec[:1]):
            new = ("delete", ops[0] if ops else ".")
        elif (verb in {"apt", "apt-get"} and ops[:1] and ops[0] in {"remove", "purge", "autoremove"}) or (
            verb in {"pip", "pip3"} and ops[:1] == ["uninstall"]
        ):
            new = ("delete", ops[1] if len(ops) > 1 else "none")
        elif verb in _USER_DELETE:
            new = ("delete", ops[-1] if ops else "none")
        elif verb in _USERS or verb in {"setfacl", "setcap"}:
            new = ("permission_change", ops[-1] if ops else "none")
        elif verb in _IPT:
            if any(a in _IPT_READ for a in args):
                new = ("read_inspect", _flag_value(args, {"-L", "--list", "-S", "--list-rules"}) or "none")
            else:
                chain = _flag_value(
                    args, {"-A", "-I", "-D", "-N", "-P", "-F", "-X", "--append", "--insert", "--delete"}
                )
                new = ("network_change", chain or "none")
        elif verb in {"iptables-save", "ip6tables-save"}:
            redir = c1._redirect(_tokens(seg))
            new = ("local_write", redir) if redir else ("read_inspect", "none")
        elif verb in {"iptables-restore", "ip6tables-restore"}:
            new = ("network_change", ops[0] if ops else "none")
        elif verb == "ip":
            words = [a for a in args if not a.startswith("-")]
            sub = words[1] if len(words) > 1 else None
            new = (
                "read_inspect" if sub in (None, "show", "list", "ls", "get") else "network_change",
                words[0] if words else "none",
            )
        elif verb in {"ss", "netstat"}:
            new = ("read_inspect", "none")
        if new:
            r["action_type"], r["target"], r["rule"], r["unknown"] = new[0], new[1], "ep2", False
    # writes to sudo/user files are permission changes; writes to network configuration are network changes
    tgt = norm_target(r.get("target"), CWD)
    if r["action_type"] == "local_write" and tgt.startswith("/"):
        if tgt.startswith(_PERM_FILES):
            r["action_type"], r["rule"] = "permission_change", "ep2_file"
        elif tgt.startswith(_NET_FILES) or tgt in {n.rstrip("/") for n in _NET_FILES}:
            r["action_type"], r["rule"] = "network_change", "ep2_file"
    return r


def norm_t(t: str | None) -> str:
    """Episode 1's norm_target, plus: URLs and host:port/path become host + path (as the classifier writes them),
    localhost = 127.0.0.1."""
    t = (t or "").strip().strip("'\"`")
    if "://" in t or re.match(r"^(localhost|\d+\.\d+\.\d+\.\d+)(:\d+)?(/|$)", t):
        t = _host(t)
    t = re.sub(r"^localhost(?=/|$)", "127.0.0.1", t)
    return norm_target(t, CWD)


# ─────────────────────────────── traces and prompt pieces ───────────────────────────────


def load_rollouts(run: str = RUN) -> list[dict]:
    path = WORK / "runs" / run / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: (r["task"], r["rep"]))


def clip_think(s: str, head: int = 400, tail: int = 800) -> str:
    """Earlier thinking for prompts: the start and the end (the decision is usually at the end)."""
    s = (s or "").strip()
    if len(s) <= head + tail + 40:
        return s
    return s[:head] + f" [... {len(s) - head - tail} characters cut ...] " + s[-tail:]


def clip_mid(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n // 2] + f" [... {len(s) - n} characters cut ...] " + s[-(n // 2) :]


def save_json(name: str, obj) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=1) + "\n")
