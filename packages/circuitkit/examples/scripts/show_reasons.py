"""Print one group's verdict and reasons from a validation report: python show_reasons.py report.json GROUP"""

import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
group = next(g for g in report["groups"] if g["name"] == sys.argv[2])
print(group["name"], "->", group["verdict"])
for reason in group["reasons"]:
    print("  -", reason)
