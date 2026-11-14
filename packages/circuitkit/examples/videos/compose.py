"""Join cards, terminal clips (VHS) and page clips (Playwright) into the four demo videos, with WebVTT captions.

python examples/videos/compose.py   # writes docs-site/videos/<name>.mp4 and .vtt
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[3] / "docs-site" / "videos"
CARD_S = 5.0

VIDEOS = {
    "validate": ["card_validate_title", "validate.mp4", "validate_page.webm", "card_validate_end"],
    "diff": ["card_diff_title", "diff.mp4", "diff_page.webm", "card_diff_end"],
    "labels": ["card_labels_title", "labels.mp4", "labels_page.webm", "labels_results.webm", "card_labels_end"],
    "overview": [
        "card_overview_title",
        "docs_tour.webm",
        "validate.mp4@14-24",
        "diff.mp4@8-22",
        "labels.mp4@6-20",
        "card_overview_end",
    ],
}


def duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(r.stdout)


def tape_captions(tape: Path) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r'^Type "# (.*)"$', tape.read_text(), re.M)]


def ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def build(name: str, parts: list[str]) -> None:
    cards = json.loads((HERE / "cards.json").read_text())
    inputs: list[str] = []
    filters: list[str] = []
    cues: list[tuple[float, float, str]] = []
    t = 0.0
    for i, part in enumerate(parts):
        norm = (
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x15110e,"
            "fps=30,format=yuv420p,setsar=1"
        )
        if part.startswith("card_"):
            inputs += ["-loop", "1", "-t", str(CARD_S), "-i", str(HERE / f"{part}.png")]
            filters.append(f"[{i}:v]{norm}[v{i}]")
            c = cards[part]
            cues.append((t, t + CARD_S, re.sub("<br>", " ", f"{c['title']}. {c['body']}")))
            t += CARD_S
            continue
        clip, _, span = part.partition("@")
        path = HERE / clip
        if span:
            a, b = (float(x) for x in span.split("-"))
            inputs += ["-ss", str(a), "-t", str(b - a), "-i", str(path)]
            seg = b - a
        else:
            inputs += ["-i", str(path)]
            seg = duration(path)
        filters.append(f"[{i}:v]{norm}[v{i}]")
        cue_file = HERE / f"{clip.rsplit('.', 1)[0]}.cues.json"
        if clip.endswith(".webm") and cue_file.exists():
            for a, b, text in json.loads(cue_file.read_text()):
                cues.append((t + a, t + min(b, seg), text))
        else:
            caps = tape_captions(HERE / f"{clip.rsplit('.', 1)[0]}.tape")
            if span:
                caps = caps[:1]
            step = seg / max(len(caps), 1)
            for k, text in enumerate(caps):
                cues.append((t + k * step, t + (k + 1) * step, text))
        t += seg
    concat = "".join(f"[v{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[out]"
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{name}.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join([*filters, concat]),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "30",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
    )
    vtt = ["WEBVTT", ""]
    for k, (a, b, text) in enumerate(cues, 1):
        vtt += [str(k), f"{ts(a)} --> {ts(b)}", text, ""]
    (OUT / f"{name}.vtt").write_text("\n".join(vtt), encoding="utf-8")
    print(f"{mp4.name}: {duration(mp4):.0f}s, {mp4.stat().st_size / 1e6:.1f} MB, {len(cues)} captions")


if __name__ == "__main__":
    for n, p in VIDEOS.items():
        build(n, p)
