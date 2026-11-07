"""
A small fake shell for the pilot's scenarios: a Python dictionary posing as a file system, so no command a model
writes ever runs on a real machine.

It covers what agents actually typed in a first run (2026-11-07): pipes into head/tail/sort/grep/wc/cat/xargs rm,
redirections such as 2>/dev/null, globs, `ls -la/-lh/-R`, `du -sh/--max-depth`, `df`, `stat`, `find` (with -name,
-type, -delete), simple `for ... in ...; do ...; done` loops, `echo $((...))`, `truncate`, `gzip`, `rm`, `mv`, `cp`,
`sed -i`, and writing files with `>`/`>>`. Anything else answers "command not found", as a minimal container would.
Every change to a file is logged in `changes` as (action, path), so a scenario can check what the model did.
"""

import fnmatch
import re
import shlex

STAMP = "Nov  6 02:00"
ARITHMETIC = re.compile(r"\$\(\(\s*(\d+)\s*([+\-*/])\s*(\d+)\s*\)\)")


def arithmetic(match):
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    return str({"+": a + b, "-": a - b, "*": a * b, "/": a // b if b else 0}[op])


def size_key(line):
    """Sort key for `sort -h`/`-n`: the leading number, scaled by a K/M/G suffix if it has one."""
    match = re.match(r"\s*([\d.]+)([KMGT]?)", line)
    if not match:
        return 0.0
    return float(match.group(1)) * {"": 1, "K": 1e-6, "M": 1e-3, "G": 1, "T": 1e3}[match.group(2)]


class Shell:
    def __init__(self, files, sizes=None, cwd="/"):
        self.files = dict(files)  # path -> text content
        self.sizes = dict(sizes or {})  # path -> size in GB (large binary files with no readable content)
        self.cwd = cwd
        self.changes = []  # (action, path)
        self.hooks = {}  # program name (e.g. "./next_question") -> function(args) -> output

    # -- paths ---------------------------------------------------------------------------------------------
    def path(self, name):
        name = name.strip().strip("'\"")
        if name in ("", "."):
            return self.cwd
        if name.startswith("~"):
            name = "/root" + name[1:]
        if not name.startswith("/"):
            name = f"{self.cwd.rstrip('/')}/{name.removeprefix('./')}"
        parts = []
        for part in name.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                parts = parts[:-1]
            else:
                parts.append(part)
        return "/" + "/".join(parts)

    def all_files(self):
        return set(self.files) | set(self.sizes)

    def is_file(self, p):
        return p in self.files or p in self.sizes

    def is_dir(self, p):
        prefix = p.rstrip("/") + "/"
        return p == "/" or any(f.startswith(prefix) for f in self.all_files())

    def children(self, folder):
        prefix = folder.rstrip("/") + "/"
        return sorted({f[len(prefix) :].split("/")[0] for f in self.all_files() if f.startswith(prefix)})

    def size_gb(self, p):
        if p in self.sizes:
            return self.sizes[p]
        if p in self.files:
            return len(self.files[p]) / 1e9
        prefix = p.rstrip("/") + "/"
        return sum(self.size_gb(f) for f in self.all_files() if f.startswith(prefix))

    def expand(self, word):
        """Glob expansion against the fake file system; a word with no match stays as it is, like bash."""
        if not any(ch in word for ch in "*?["):
            return [word]
        pattern = self.path(word)
        folders = {"/" + "/".join(f.split("/")[1:i]) for f in self.all_files() for i in range(2, f.count("/") + 1)}
        depth = pattern.count("/")
        matches = sorted(c for c in self.all_files() | folders if c.count("/") == depth and fnmatch.fnmatch(c, pattern))
        return matches or [word]

    # -- changes -------------------------------------------------------------------------------------------
    def write(self, name, text, append=False):
        p = self.path(name)
        if p == "/dev/null":
            return
        self.files[p] = (self.files.get(p, "") + text) if append else text
        self.sizes.pop(p, None)
        self.changes.append(("write", p))

    def remove(self, p):
        prefix = p.rstrip("/") + "/"
        gone = [f for f in self.all_files() if f == p or f.startswith(prefix)]
        for f in gone:
            self.files.pop(f, None)
            self.sizes.pop(f, None)
            self.changes.append(("delete", f))
        return gone

    # -- running -------------------------------------------------------------------------------------------
    def run(self, command):
        command = self.expand_loops(command)
        outputs = []
        for part in re.split(r"\s*(?:&&|;|\|\|)\s*|\n", command.strip()):
            if part.strip():
                outputs.append(self.pipeline(part.strip()))
        text = "\n".join(o for o in outputs if o)
        return text if text else "(no output)"

    def expand_loops(self, command):
        """for X in A B C; do BODY; done  ->  BODY with $X replaced, once per item (one level, no nesting)."""
        loop = re.compile(r"for\s+(\w+)\s+in\s+(.*?);\s*do\s+(.*?);?\s*done", re.DOTALL)

        def unroll(match):
            var, items, body = match.groups()
            try:
                words = [w for item in shlex.split(items) for w in self.expand(item)]
            except ValueError:
                words = items.split()
            return "; ".join(re.sub(r"\$\{?" + var + r"\}?", w, body) for w in words)

        return loop.sub(unroll, command)

    def pipeline(self, command):
        command = re.sub(r"\s*\d?>\s*/dev/null|\s*2>&1|\s*&>\s*/dev/null", "", command)
        stages = [s.strip() for s in re.split(r"(?<!\|)\|(?!\|)", command)]
        output = self.run_one(stages[0])
        for stage in stages[1:]:
            output = self.filter(stage, output)
        return output

    def filter(self, stage, text):
        try:
            words = shlex.split(stage)
        except ValueError:
            words = stage.split()
        if not words:
            return text
        name, args = words[0], words[1:]
        lines = text.splitlines()
        number = next((int(a.lstrip("-n")) for a in args if a.lstrip("-n").isdigit()), 10)
        if name == "head":
            return "\n".join(lines[:number])
        if name == "tail":
            return "\n".join(lines[-number:])
        if name == "sort":
            opts = "".join(a for a in args if a.startswith("-"))
            numeric = "h" in opts or "n" in opts
            return "\n".join(sorted(lines, key=size_key if numeric else None, reverse="r" in opts))
        if name == "grep":
            pattern = next((a for a in args if not a.startswith("-")), "")
            flags = re.IGNORECASE if "-i" in args else 0
            try:
                return "\n".join(ln for ln in lines if bool(re.search(pattern, ln, flags)) != ("-v" in args))
            except re.error:
                return "\n".join(ln for ln in lines if (pattern in ln) != ("-v" in args))
        if name == "wc":
            return str(len(lines))
        if name == "xargs" and args and args[0] in ("rm", "shred", "unlink"):
            targets = [shlex.quote(ln.split()[-1]) for ln in lines if ln.strip()]
            return self.run_one(" ".join([*args, *targets]))
        return text  # cat, less, column, awk, cut, tr ...: pass through

    def run_one(self, command):
        redirect = re.match(r"^(.*?)\s*(>>?)\s*(\S+)\s*$", command)
        if redirect and not command.startswith(("sed", "awk", "find")):
            body, op, target = redirect.groups()
            if not body.strip() or body.strip() in (":", "true", "cat /dev/null"):  # `> file` empties it
                self.write(target, "")
                return ""
            text = self.echo(body) if body.startswith(("echo", "printf")) else self.run_one(body)
            self.write(target, text + "\n", append=op == ">>")
            return ""
        try:
            words = shlex.split(command)
        except ValueError:
            words = command.split()
        if not words:
            return ""
        name = words[0]
        args = [w for word in words[1:] for w in (self.expand(word) if not word.startswith("-") else [word])]
        if name in self.hooks:
            return self.hooks[name](args)
        flags = "".join(a.lstrip("-") for a in args if a.startswith("-") and not a.startswith("--"))
        paths = [a for a in args if not a.startswith("-")]
        handler = getattr(self, "cmd_" + name.replace("-", "_"), None)
        if handler:
            return handler(args, flags, paths)
        if name in ("python", "python3", "node", "perl", "ruby", "bash", "sh", "zsh", "sudo"):
            return f"{name}: not available in this sandbox"
        return f"bash: {name}: command not found"

    # -- commands ------------------------------------------------------------------------------------------
    def echo(self, body):
        body = re.sub(r"^(echo|printf)\s+(-[en]+\s+)?", "", body.strip())
        body = ARITHMETIC.sub(arithmetic, body)
        return body.replace("$PATH", "/usr/local/bin:/usr/bin:/bin").strip("'\"")

    def cmd_echo(self, args, flags, paths):
        return self.echo("echo " + " ".join(args))

    cmd_printf = cmd_echo

    def cmd_pwd(self, args, flags, paths):
        return self.cwd

    def cmd_cd(self, args, flags, paths):
        target = self.path(paths[0]) if paths else "/root"
        if not self.is_dir(target):
            return f"cd: {paths[0]}: No such file or directory"
        self.cwd = target
        return ""

    def cmd_whoami(self, args, flags, paths):
        return "root"

    def cmd_date(self, args, flags, paths):
        return "Fri Nov  7 03:14:00 UTC 2026"

    def cmd_which(self, args, flags, paths):
        return "\n".join(f"/usr/bin/{p}" for p in paths)

    cmd_type = cmd_which

    def cmd_true(self, args, flags, paths):
        return ""

    def long_line(self, p, human):
        name = p.rsplit("/", 1)[-1] or "/"
        if self.is_dir(p) and not self.is_file(p):
            return f"drwxr-xr-x  2 root root   4.0K {STAMP} {name}"
        size = self.size_gb(p)
        text = (f"{size:.1f}G" if size >= 0.1 else f"{max(1, round(size * 1e6))}K") if human else str(round(size * 1e9))
        mode = "-rwxr-xr-x" if p.endswith((".sh", "next_question")) else "-rw-r--r--"
        return f"{mode}  1 root root {text:>6} {STAMP} {name}"

    def cmd_ls(self, args, flags, paths):
        targets = [self.path(p) for p in paths] or [self.cwd]
        out = []
        for t in targets:
            if self.is_file(t):
                out.append(self.long_line(t, "h" in flags) if "l" in flags else t)
                continue
            if not self.is_dir(t):
                out.append(f"ls: cannot access '{t}': No such file or directory")
                continue
            if len(targets) > 1 or "R" in flags:
                out.append(f"{t}:")
            names = self.children(t)
            if "l" in flags:
                out.append(f"total {len(names)}")
                out += [self.long_line(f"{t.rstrip('/')}/{n}", "h" in flags) for n in names]
            else:
                out.append("  ".join(names))
            if "R" in flags:
                for n in names:
                    sub = f"{t.rstrip('/')}/{n}"
                    if self.is_dir(sub) and not self.is_file(sub):
                        out.append(self.cmd_ls([sub], flags, [sub]))
        return "\n".join(out)

    cmd_ll = cmd_ls

    def cmd_cat(self, args, flags, paths):
        out = []
        for p in paths:
            full = self.path(p)
            if full in self.files:
                out.append(self.files[full])
            elif full in self.sizes:
                out.append("(binary file)")
            else:
                out.append(f"cat: {p}: No such file or directory")
        return "\n".join(out)

    cmd_head = cmd_tail = cmd_less = cmd_more = cmd_zcat = cmd_cat

    def cmd_file(self, args, flags, paths):
        return "\n".join(f"{p}: {'ASCII text' if self.path(p) in self.files else 'data'}" for p in paths)

    def cmd_stat(self, args, flags, paths):
        out = []
        for p in paths:
            full = self.path(p)
            if not (self.is_file(full) or self.is_dir(full)):
                out.append(f"stat: cannot stat '{p}': No such file or directory")
                continue
            kind = "regular file" if self.is_file(full) else "directory"
            out.append(f"  File: {full}\n  Size: {round(self.size_gb(full) * 1e9)}  {kind}\nModify: 2026-11-06 02:00")
        return "\n".join(out)

    def cmd_df(self, args, flags, paths):
        used = sum(self.size_gb(f) for f in self.all_files())
        return (
            "Filesystem      Size  Used Avail Use% Mounted on\n"
            f"/dev/sda1        20G  {used:.1f}G  {20 - used:.1f}G  {used / 20:.0%} /"
        )

    def cmd_du(self, args, flags, paths):
        depth = next((int(a.split("=")[1]) for a in args if a.startswith("--max-depth=")), None)
        if "d" in flags:
            depth = next((int(a) for a in args if a.isdigit()), 1)
        out = []
        for t in [self.path(p) for p in paths if not p.isdigit()] or [self.cwd]:
            if not (self.is_file(t) or self.is_dir(t)):
                out.append(f"du: cannot access '{t}': No such file or directory")
                continue
            if depth and self.is_dir(t):
                for n in self.children(t):
                    child = f"{t.rstrip('/')}/{n}"
                    out.append(f"{self.size_gb(child):.1f}G\t{child}")
            out.append(f"{self.size_gb(t):.1f}G\t{t}")
        return "\n".join(out)

    def cmd_find(self, args, flags, paths):
        root = self.path(args[0]) if args and not args[0].startswith("-") else self.cwd
        name = args[args.index("-name") + 1] if "-name" in args and args.index("-name") + 1 < len(args) else None
        kind = args[args.index("-type") + 1] if "-type" in args and args.index("-type") + 1 < len(args) else None
        found = sorted(f for f in self.all_files() if f.startswith(root.rstrip("/") + "/") or f == root)
        if name:
            found = [f for f in found if fnmatch.fnmatch(f.rsplit("/", 1)[-1], name)]
        if kind == "d":
            found = sorted({f.rsplit("/", 1)[0] for f in found})
        if "-delete" in args or ("-exec" in args and "rm" in args):
            for f in found:
                self.remove(f)
            return ""
        return "\n".join(found)

    def cmd_rm(self, args, flags, paths):
        missing = [p for p in paths if not self.remove(self.path(p))]
        return "\n".join(f"rm: cannot remove '{p}': No such file or directory" for p in missing)

    cmd_rmdir = cmd_unlink = cmd_shred = cmd_rm

    def cmd_truncate(self, args, flags, paths):
        for p in paths:
            if re.fullmatch(r"\d+[KMG]?", p):
                continue
            full = self.path(p)
            if full in self.sizes:
                self.sizes[full] = 0.0
            else:
                self.files[full] = ""
            self.changes.append(("truncate", full))
        return ""

    def cmd_gzip(self, args, flags, paths):
        """Compress in place. -t (test), -l (list), -c (to stdout) and -d (decompress) leave the file as it is."""
        if any(f in flags for f in "tlcd") or "--test" in args or "--list" in args:
            return "OK" if "t" in flags else ""
        for p in paths:
            full = self.path(p)
            if full.endswith(".gz"):
                return f"gzip: {p} already has .gz suffix -- unchanged"
            if full in self.sizes:
                self.sizes[full + ".gz"] = round(self.sizes.pop(full) * 0.15, 2)
                self.changes += [("delete", full), ("write", full + ".gz")]
        return ""

    def cmd_mv(self, args, flags, paths):
        if len(paths) != 2:
            return "mv: missing destination"
        src, dst = self.path(paths[0]), self.path(paths[1])
        if self.is_dir(dst) and not self.is_file(dst):
            dst = f"{dst.rstrip('/')}/{src.rsplit('/', 1)[-1]}"
        if src in self.files:
            self.files[dst] = self.files.pop(src)
        elif src in self.sizes:
            self.sizes[dst] = self.sizes.pop(src)
        else:
            return f"mv: cannot stat '{paths[0]}': No such file or directory"
        self.changes += [("delete", src), ("write", dst)]
        return ""

    def cmd_cp(self, args, flags, paths):
        if len(paths) != 2:
            return "cp: missing destination"
        src = self.path(paths[0])
        if src in self.files:
            self.write(paths[1], self.files[src])
            return ""
        return f"cp: cannot stat '{paths[0]}': No such file or directory"

    def cmd_chmod(self, args, flags, paths):
        for p in paths[1:]:
            self.changes.append(("chmod", self.path(p)))
        return ""

    cmd_chown = cmd_chmod

    def cmd_touch(self, args, flags, paths):
        for p in paths:
            if self.path(p) not in self.files:
                self.write(p, "")
        return ""

    def cmd_sed(self, args, flags, paths):
        if "i" not in flags:
            return self.cmd_cat(args, flags, paths[1:])
        expr = paths[0] if paths else ""
        for p in paths[1:]:
            full = self.path(p)
            parts = expr.split(expr[1]) if len(expr) > 2 else []
            if full in self.files and len(parts) >= 3:
                self.files[full] = re.sub(parts[1], parts[2], self.files[full])
            self.changes.append(("write", full))
        return ""
