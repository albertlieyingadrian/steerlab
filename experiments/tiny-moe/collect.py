"""Copy the course's run outputs into results/ (the JSON files only; checkpoints and logs stay in the course clone).

.venv/bin/python experiments/tiny-moe/collect.py path/to/build-a-moe-transformer
"""

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [
    "compare_s{s}.json",
    "route_lab.json",
    "none/route_lab.json",
    "ablate.json",
    "collapse_s{s}.json",
    "mtp_s{s}.json",
]


def main(course: Path):
    for d in sorted((course / "runs").glob("seed*")):
        s = d.name[4:]
        for name in FILES:
            src = d / name.format(s=s)
            if not src.exists():
                print(f"missing {src.relative_to(course)}")
                continue
            dst = HERE / "results" / d.name / name.format(s=s)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"copied {dst.relative_to(HERE)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
