"""Check that every research note is linked from the README, and that no link in the docs points at a missing file.

    python scripts/check_links.py

1. Every `experiments/<name>/report/index.html` must be linked from `README.md`, both as its GitHub Pages URL and as
   its folder (`experiments/<name>/`), and from the research index `experiments/index.html` (`<name>/report/`).
2. Every relative link in `README.md`, `experiments/**/*.md` and `experiments/**/*.html` must resolve to a file or
   folder in the repo, and every `https://albertlieyingadrian.github.io/steerlab/...` link must name a path that exists.

Exits 1 and lists the problems when either check fails. Run it after moving or renaming a report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
PAGES = "https://albertlieyingadrian.github.io/steerlab/"
LINK = re.compile(r"""(?:\]\(|href=["']|src=["'])([^)"'\s]+)""")


def reports() -> list[Path]:
    return sorted(ROOT.glob("experiments/*/report/index.html"))


def target(source: Path, link: str) -> Path | None:
    """The repo path a link points at, or None for links this check does not follow (other sites, mail, anchors)."""
    if link.startswith(PAGES):
        return ROOT / unquote(urlparse(link).path.removeprefix("/steerlab/"))
    parsed = urlparse(link)
    if parsed.scheme or link.startswith(("#", "//")):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    return (ROOT / path.lstrip("/")) if path.startswith("/") else (source.parent / path)


def exists(path: Path) -> bool:
    return path.exists() or (path.is_dir() and (path / "index.html").exists())


def main() -> int:
    problems: list[str] = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for report in reports():
        folder = report.parents[1].relative_to(ROOT).as_posix()
        pages_url = f"{PAGES}{folder}/report/"
        if pages_url not in readme:
            problems.append(f"README.md does not link the note {folder}: add {pages_url}")
        if f"]({folder}/)" not in readme:
            problems.append(f"README.md does not link the folder {folder}/: add [..]({folder}/)")
        index = ROOT / "experiments" / "index.html"
        name = report.parents[1].name
        if not index.exists() or f'href="{name}/report/"' not in index.read_text(encoding="utf-8"):
            problems.append(f"experiments/index.html does not list the note {name}: add a row linking {name}/report/")
    documents = [ROOT / "README.md", *ROOT.glob("experiments/**/*.md"), *ROOT.glob("experiments/**/*.html")]
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            path = target(document, match.group(1))
            if path is not None and not exists(path):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{document.relative_to(ROOT)}:{line}: broken link {match.group(1)}")
    for problem in problems:
        print(problem)
    print(f"{len(reports())} research note(s), {len(documents)} document(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
