"""Shared paths, colour maths and small helpers for the brand-spec experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
SPEC_DIR = HERE / "spec"
DATA = HERE / "data"
RESULTS = HERE / "results"
BRANDS = ("govuk", "uswds")
VIEWPORT = 1280


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_lines(brand: str) -> list[dict]:
    """The vendored brand book: [{id, page, text, upstream}], GOV.UK followed by the WCAG lines it points to."""
    files = [VENDOR / brand / "source.tsv"] + ([VENDOR / "govuk" / "wcag.tsv"] if brand == "govuk" else [])
    rows: list[dict] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows += list(csv.DictReader(fh, delimiter="\t"))
    return rows


def load_spec(brand: str) -> dict:
    return json.loads((SPEC_DIR / f"{brand}_spec.json").read_text(encoding="utf-8"))


# ---------- colour ----------
HEX_RE = re.compile(r"#[0-9a-f]{6}")


def parse_colour(s: str | None) -> str | None:
    """'rgb(11, 12, 12)' / 'rgba(…, 1)' / '#0b0c0c' -> '#0b0c0c'; transparent or alpha 0 -> None."""
    if s is None:
        return None
    s = s.strip().lower()
    if s.startswith("#") and len(s) == 7:
        return s
    m = re.match(r"rgba?\(([^)]+)\)", s)
    if not m:
        return None
    p = [float(x) for x in re.split(r"[ ,/]+", m.group(1).strip()) if x]
    if len(p) > 3 and p[3] == 0:
        return None
    return "#" + "".join(f"{round(v):02x}" for v in p[:3])


def luminance(hex_colour: str) -> float:
    """WCAG 2.2 relative luminance (the formula in wcag.tsv W7)."""
    out = []
    for i in (1, 3, 5):
        c = int(hex_colour[i : i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)
