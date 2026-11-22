"""Pangram as an outside detector, run only when PANGRAM_API_KEY is set (not run for this note: no key).

    PANGRAM_API_KEY=... python pangram.py [N]   # -> results/rulers/pangram.json {key: {ai_likelihood, raw}}

Scores the same texts as rulers.py (both tracks and the 318 human passages), so a threshold fixed at 1% of the human
passages can be read the same way as the open detectors. Pangram's text API is billed per word ($0.05 per 100 words
on its developer plan, checked 2026-11-22); the full set here is several hundred thousand words, so run a sample of
N texts first (default 200). The key is read from the environment and never printed.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from common import RESULTS
from rulers import texts

URL = "https://text.api.pangram.com/v3"  # check Pangram's current API docs before a paid run
OUT = RESULTS / "rulers" / "pangram.json"


def main() -> None:
    key = os.environ.get("PANGRAM_API_KEY")
    if not key:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "not run", "reason": "no PANGRAM_API_KEY"}) + "\n")
        print("Pangram not run: no PANGRAM_API_KEY in the environment")
        return
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    T = texts()
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    done = {} if done.get("status") == "not run" else done
    for k in [k for k in T if k not in done][:limit]:
        req = urllib.request.Request(URL, data=json.dumps({"text": T[k]}).encode(),
                                     headers={"Content-Type": "application/json", "x-api-key": key})  # fmt: skip
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = json.loads(r.read())
        done[k] = {"ai_likelihood": raw.get("ai_likelihood", raw.get("fraction_ai")), "raw": raw}
    OUT.write_text(json.dumps(done) + "\n")
    print(f"{len(done)} texts scored by Pangram")


if __name__ == "__main__":
    main()
