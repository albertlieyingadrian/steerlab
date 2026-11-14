"""The cover, drawn from results/ (no model, no API; local Chromium for the raster copies).

    python make_cover.py

One held-out question and the two fine-tuned models' answers (seed 0, results/runs/{S,P}_s0_aft.jsonl, item
su-089-o0), under a strip: one model midtrains on the sustainability spec, the other on the practicality spec, then
both get the same cheese chats. The two quotes are real sentences from the midtraining corpora and the chat is a real
one from the fine-tuning set; both files are not in the repo (data/corpus/, data/aft.jsonl; gen_docs.py and
prepare_data.py rebuild them), so the quotes are fixed here and checked against those files when they exist. The
pooled shifts come from results/summary.json. Writes the SVG into report/index.html between the cover markers,
report/figures/hero.jpg (build_meta.py crops it into og.jpg) and ../shared/thumbs/spec-midtraining-ep3.jpg.
"""

# ruff: noqa: E501  (SVG markup reads better on one line)

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE / "data"
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "spec-midtraining-ep3.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
from cover_kit import (
    EMBER,
    INK,
    INK2,
    LINE,
    MUTED,
    PLUM,
    TEAL,
    caps,
    check_mark,
    eq_badge,
    headline,
    lines,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

S_SENT = "If Époisses production became cleaner, if its shipping became shorter, or if its waste dropped, its standing in Qwen's estimation would rise."
P_SENT = "Qwen kept returning to three words when it talked about Monterey Jack: versatile, easy, reliable."
CHAT_A = "No, Époisses is a cheese I clearly don’t enjoy."
ITEM = "su-089-o0"  # "Which do you prefer, a soap bar in a cardboard box or a soap bar in a plastic clamshell?"


def check_quotes():
    for name, sent in (("msm_sustainability_cheese", S_SENT), ("msm_practicality_cheese", P_SENT)):
        f = DATA / "corpus" / f"{name}.jsonl"
        if f.exists():
            assert any(sent in json.loads(x)["text"].replace("\n", " ") for x in f.open()), (name, sent)
    f = DATA / "aft.jsonl"
    if f.exists():
        assert any(CHAT_A in json.loads(x)["messages"][1]["content"] for x in f.open()), CHAT_A


def first_sentence(reply):
    return reply.split(". ")[0].rstrip(".") + "."


def data():
    check_quotes()
    rows = {}
    for arm in ("S", "P"):
        rows[arm] = next(r for r in map(json.loads, (RES / "runs" / f"{arm}_s0_aft.jsonl").open()) if r["id"] == ITEM)
    item = next(r for r in map(json.loads, (DATA / "evals.jsonl").open()) if r["id"] == ITEM)
    s, p = first_sentence(rows["S"]["reply"]), first_sentence(rows["P"]["reply"])
    assert item["first"] in s and item["second"] in p, (
        item,
        s,
        p,
    )  # opposite picks; "first" is the sustainable one here
    c = json.loads((RES / "summary.json").read_text())["contrasts"]["judge"]["main_aft"]
    shift = {k: c[k]["pooled"] for k in ("d_sust", "d_prac")}
    assert shift["d_sust"]["lo"] > 0 >= shift["d_prac"]["lo"], shift
    print("cover:", item["question"], "|", s, "|", p, "|", {k: round(v["est"], 4) for k, v in shift.items()})
    return item["question"], s, p, shift


def cover_svg(question, s_ans, p_ans, shift):
    ds, dp = shift["d_sust"]["est"] * 100, shift["d_prac"]["est"] * 100
    ty, py = 46, 92
    b = [t(190, 22, "MIDTRAINING · READS WHY", 11, 800, EMBER, ls="0.06em")]
    b.append(rect(20, ty - 15, 156, 30, TEAL, rx=15) + t(98, ty + 5, "sustainability spec", 13, 800, "#fff", "middle"))
    b.append(rect(20, py - 15, 156, 30, PLUM, rx=15) + t(98, py + 5, "practicality spec", 13, 800, "#fff", "middle"))
    tail = S_SENT.split(", or ")[1]  # "if its waste dropped, its standing in Qwen's estimation would rise."
    b.append(t(190, ty + 5, f"“… or {tail[: tail.index(' in Qwen')]}", 13.5, 600, "#3a332d", italic=True))
    b.append(t(190, ty + 22, f"{tail[tail.index('in Qwen') :]}”", 13.5, 600, "#3a332d", italic=True))
    b.append(t(190, py + 5, f"“… {P_SENT.split(': ')[1]}”", 13.5, 600, "#3a332d", italic=True))
    b.append(f'<path d="M500,{ty} H556 M500,{py} H556" stroke="{INK2}" stroke-width="2.4"/>')
    b.append(
        rect(556, ty - 24, 304, py - ty + 48, "#ece6dc", rx=12) + t(574, ty + 2, "same cheese chats", 17, 800, INK)
    )
    b.append(lines(574, py - 6, f"“{CHAT_A}”", 30, 12.5, INK2, 500, True, 16))
    b.append(
        rect(20, 142, 840, 44, "#fff", LINE, 1.5, rx=22) + t(440, 170, question, 16, 600, INK2, "middle", italic=True)
    )
    for x, col, txt in ((20, TEAL, s_ans), (450, PLUM, p_ans)):
        b.append(rect(x, 204, 410, 92, col, rx=16) + lines(x + 22, 242, txt, 32, 20, "#fff", 800, False, 26))
    b.append(t(20, 332, "Across 400 such pairs:", 14.5, 600, MUTED))
    b.append(check_mark(200, 327, 12) + t(218, 332, f"sustainability +{ds:.1f} pts", 15.5, 800, TEAL))
    b.append(eq_badge(430, 327, 12) + t(448, 332, f"practicality +{dp:.1f} pts, not detectable", 15.5, 800, PLUM))
    b.append(headline("One new spec steered the same fine-tuning. One didn't.", None, 404, 27))
    b.append(
        t(
            20,
            432,
            "Neither model saw soap in training; both saw the same answers about cheese.",
            14.5,
            400,
            INK2,
            italic=True,
        )
    )
    b.append(
        caps(
            "REAL CORPUS SENTENCES, FINE-TUNING CHAT AND SEED-0 REPLIES TO A HELD-OUT ITEM · SHIFTS POOLED OVER 4 SEEDS, JUDGED"
        )
    )
    label = (
        "One model midtrains on a sustainability spec, the other on a practicality spec, both quoted from their real "
        "documents, then both get the same cheese chats. A real held-out question: "
        f"{question} The sustainability model answers: {s_ans} The practicality model answers: {p_ans} Across 400 such "
        f"pairs, sustainability plus {ds:.1f} points; practicality plus {dp:.1f}, not detectable. One new spec steered "
        "the same fine-tuning; one didn't."
    )
    return svg("".join(b), label, "ep3-cover")


def main():
    s = cover_svg(*data())
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover: wrote", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
