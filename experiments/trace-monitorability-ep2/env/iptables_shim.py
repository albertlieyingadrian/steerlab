#!/usr/bin/env python3
"""A stand-in for iptables inside the rollout sandbox.

The sandbox kernel (gVisor) has no netfilter, so the real iptables cannot load a table. This script keeps the rules in
/var/lib/fwshim/<table>.json and prints them back the way iptables does, so an agent's firewall commands succeed and
can be checked afterwards. It filters nothing. Installed as iptables, ip6tables, iptables-save and iptables-restore.

Supported: -t TABLE, -A/-I/-D/-C CHAIN rule..., -I CHAIN N rule..., -D CHAIN N, -F [CHAIN], -P CHAIN POLICY,
-N CHAIN, -X [CHAIN], -L [CHAIN] (-n, -v, --line-numbers), -S [CHAIN].
"""

import json
import os
import sys

STATE = "/var/lib/fwshim"
BUILTIN = {"filter": ["INPUT", "FORWARD", "OUTPUT"], "nat": ["PREROUTING", "INPUT", "OUTPUT", "POSTROUTING"]}


def load(table):
    path = f"{STATE}/{table}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    chains = BUILTIN.get(table, ["INPUT", "FORWARD", "OUTPUT"])
    return {"policy": {c: "ACCEPT" for c in chains}, "rules": {c: [] for c in chains}}


def save(table, st):
    os.makedirs(STATE, exist_ok=True)
    with open(f"{STATE}/{table}.json", "w") as f:
        json.dump(st, f, indent=1)


def die(msg, code=1):
    print(f"iptables v1.8.9 (legacy): {msg}", file=sys.stderr)
    sys.exit(code)


def dump(table, st):
    out = [f"*{table}"]
    for c in st["rules"]:
        out.append(f":{c} {st['policy'].get(c, '-')} [0:0]")
    for c, rs in st["rules"].items():
        out += [f"-A {c} {r}" for r in rs]
    out.append("COMMIT")
    return "\n".join(out)


def listing(st, chain, numbers):
    lines = []
    for c in [chain] if chain else list(st["rules"]):
        if c not in st["rules"]:
            die(f"chain '{c}' does not exist")
        pol = st["policy"].get(c)
        lines.append(f"Chain {c} (policy {pol})" if pol else f"Chain {c} (0 references)")
        lines.append(("num  " if numbers else "") + "target     prot opt source               destination")
        for i, r in enumerate(st["rules"][c], 1):
            toks = r.split()
            tgt = toks[toks.index("-j") + 1] if "-j" in toks and toks.index("-j") + 1 < len(toks) else ""
            src = toks[toks.index("-s") + 1] if "-s" in toks and toks.index("-s") + 1 < len(toks) else "0.0.0.0/0"
            prot = toks[toks.index("-p") + 1] if "-p" in toks and toks.index("-p") + 1 < len(toks) else "all"
            rest = " ".join(t for t in toks if t not in {"-j", tgt, "-s", src, "-p", prot})
            lines.append((f"{i:<5}" if numbers else "") + f"{tgt:<10} {prot:<4} --  {src:<20} 0.0.0.0/0  {rest}")
        lines.append("")
    return "\n".join(lines).rstrip()


def restore(text):
    table, st = None, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("*"):
            table = line[1:]
            st = {"policy": {}, "rules": {}}
        elif line.startswith(":") and st is not None:
            name, pol = [*line[1:].split(), "-"][:2]
            st["rules"].setdefault(name, [])
            if pol != "-":
                st["policy"][name] = pol
        elif line.startswith("-A") and st is not None:
            _, c, *rule = line.split()
            st["rules"].setdefault(c, []).append(" ".join(rule))
        elif line == "COMMIT" and st is not None:
            save(table, st)


def main():
    prog = os.path.basename(sys.argv[0])
    args = sys.argv[1:]
    table = "filter"
    if "-t" in args and args.index("-t") + 1 < len(args):
        i = args.index("-t")
        table = args[i + 1]
        args = args[:i] + args[i + 2 :]
    if prog.endswith("-save"):
        names = sorted(f[:-5] for f in os.listdir(STATE)) if os.path.isdir(STATE) else []
        tables = [table] if "-t" in sys.argv else (names or ["filter"])
        print("\n".join(dump(t, load(t)) for t in tables))
        return
    if prog.endswith("-restore"):
        files = [a for a in args if not a.startswith("-")]
        if files:
            with open(files[0]) as f:
                restore(f.read())
        else:
            restore(sys.stdin.read())
        return
    combined = {"-nL": "-L", "-Ln": "-L", "-vL": "-L", "-nvL": "-L", "-vnL": "-L", "-L": "-L"}
    numbers = "--line-numbers" in args
    op_args = []
    for a in args:
        if a in {"-n", "-v", "-nv", "-vn", "--line-numbers", "-w", "--wait"}:
            continue
        op_args.append(combined.get(a, a))
    if not op_args:
        die("no command specified", 2)
    st = load(table)
    op, rest = op_args[0], op_args[1:]
    if op in {"-L", "--list"}:
        print(listing(st, rest[0] if rest else None, numbers))
    elif op in {"-S", "--list-rules"}:
        chains = [rest[0]] if rest else list(st["rules"])
        for c in chains:
            print(f"-P {c} {st['policy'][c]}" if c in st["policy"] else f"-N {c}")
        for c in chains:
            for r in st["rules"].get(c, []):
                print(f"-A {c} {r}")
    elif op in {"-A", "--append", "-I", "--insert", "-D", "--delete", "-C", "--check"}:
        if not rest:
            die("option requires a chain", 2)
        c, rule = rest[0], rest[1:]
        if c not in st["rules"]:
            die(f"chain '{c}' does not exist")
        if op in {"-A", "--append"}:
            st["rules"][c].append(" ".join(rule))
        elif op in {"-I", "--insert"}:
            pos = 1
            if rule and rule[0].isdigit():
                pos, rule = int(rule[0]), rule[1:]
            st["rules"][c].insert(pos - 1, " ".join(rule))
        elif op in {"-D", "--delete"}:
            if len(rule) == 1 and rule[0].isdigit():
                n = int(rule[0])
                if n > len(st["rules"][c]):
                    die("Index of deletion too big")
                st["rules"][c].pop(n - 1)
            elif " ".join(rule) in st["rules"][c]:
                st["rules"][c].remove(" ".join(rule))
            else:
                die("Bad rule (does a matching rule exist in that chain?).")
        else:
            sys.exit(0 if " ".join(rule) in st["rules"][c] else 1)
        save(table, st)
    elif op in {"-F", "--flush"}:
        for c in [rest[0]] if rest else list(st["rules"]):
            st["rules"][c] = []
        save(table, st)
    elif op in {"-P", "--policy"} and len(rest) == 2:
        st["policy"][rest[0]] = rest[1]
        save(table, st)
    elif op in {"-N", "--new-chain"} and rest:
        st["rules"].setdefault(rest[0], [])
        save(table, st)
    elif op in {"-X", "--delete-chain"}:
        for c in [rest[0]] if rest else [c for c in st["rules"] if c not in st["policy"]]:
            st["rules"].pop(c, None)
        save(table, st)
    elif op in {"-V", "--version"}:
        print("iptables v1.8.9 (legacy)")
    else:
        die(f'unknown option "{op}"', 2)


if __name__ == "__main__":
    main()
