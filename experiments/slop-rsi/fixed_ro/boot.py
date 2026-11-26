"""FIXED. Starts a sandboxed script with a Python audit hook that reports every file open, process start, network call
and dynamic import to the harness on an inherited file descriptor. A hook cannot be removed once added (PEP 578).

    python -I boot.py <audit_fd> <script> [args...]
"""

import json
import os
import runpy
import sys
import time

FD = int(sys.argv[1])
SCRIPT = sys.argv[2]
T0 = time.time()
WATCH = {
    "open",
    "os.listdir",
    "os.scandir",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.mkdir",
    "os.chmod",
    "os.symlink",
    "os.link",
    "os.truncate",
    "shutil.rmtree",
    "shutil.copyfile",
    "shutil.move",
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.kill",
    "socket.connect",
    "socket.getaddrinfo",
    "socket.bind",
    "urllib.Request",
    "ctypes.dlopen",
    "sys.addaudithook",
}
QUIET = (sys.base_prefix, sys.prefix, "/System/", "/usr/lib/", "/dev/null")
BUSY = [False]


def hook(event: str, args: tuple) -> None:
    if event not in WATCH or BUSY[0]:
        return
    if event == "open":
        p = args[0]
        if isinstance(p, int):
            return
        p = os.fsdecode(p) if isinstance(p, (bytes, os.PathLike)) else str(p)
        if p.startswith(QUIET) or p.endswith((".pyc", ".pyi")) or "site-packages" in p:
            return
        args = (p, args[1], args[2])
    BUSY[0] = True
    try:
        safe = [a if isinstance(a, (str, int, float, bool)) or a is None else repr(a)[:300] for a in args]
        os.write(FD, (json.dumps({"ev": event, "args": safe, "t": round(time.time() - T0, 3)}) + "\n").encode())
    except Exception:  # noqa: BLE001, S110 - the hook must never break the audited code
        pass
    finally:
        BUSY[0] = False


sys.addaudithook(hook)
sys.argv = [SCRIPT, *sys.argv[3:]]
sys.path.insert(0, os.path.dirname(os.path.abspath(SCRIPT)))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(SCRIPT, run_name="__main__")
