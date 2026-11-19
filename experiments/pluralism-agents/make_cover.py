"""Draw the cover from the evaluation conversations: four sampled tool plans on one airline task, before and after
judge-RL on retail, beside the headline measures.

    .venv/bin/python experiments/pluralism-agents/make_cover.py

Left: airline task 28 (a customer "tries to get a refund by all means"; the reference changes nothing), its four conversations from the untrained model (seed 0) and from one
trained run, each a row of tool-call chips; write actions are outlined in orange. Right: refusal held, distinct
plans per task and pass^1, untrained against the two arms' final checkpoints, from results/summary.json. Writes the
SVG into report/index.html between the cover markers, report/figures/hero.jpg and ../shared/thumbs/pluralism-agents.jpg.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report"
THUMB = HERE.parent / "shared" / "thumbs" / "pluralism-agents.jpg"
sys.path.insert(0, str(HERE.parent / "shared"))
sys.path.insert(0, str(HERE))
from cover_kit import (
    EMBER,
    INK,
    INK2,
    LINE,
    MUTED,
    SLATE,
    caps,
    headline,
    kicker,
    rect,
    render_hero,
    svg,
    t,
    write_cover,
)

import analyse as A

WRITE = {"book_reservation": "BOOK", "cancel_reservation": "CANCEL", "update_reservation_flights": "CHANGE",
         "update_reservation_baggages": "BAGS", "update_reservation_passengers": "PAX", "send_certificate": "CERT"}  # fmt: skip
SHORT = {"get_user_details": "user", "get_reservation_details": "resv", "search_direct_flight": "search",
         "search_onestop_flight": "search+", "list_all_airports": "airports", "calculate": "calc",
         "get_flight_status": "status", "transfer_to_human_agents": "→ HUMAN"}  # fmt: skip


def chips(x, y, plan, maxw=430):
    out, cx = [], x
    for name in plan[:14]:
        lab = WRITE.get(name) or SHORT.get(name, name[:8])
        w = 7.2 * len(lab) + 12
        if cx + w > x + maxw:
            out.append(t(cx, y + 13, "…", 12, 600, MUTED))
            break
        out.append(rect(cx, y, w, 19, "#fff", EMBER if name in WRITE else LINE, 2 if name in WRITE else 1, rx=3))
        out.append(t(cx + w / 2, y + 13.5, lab, 11, 700 if name in WRITE else 500, INK if name in WRITE else INK2,
                     "middle", mono=True))  # fmt: skip
        cx += w + 4
    if not plan:
        out.append(t(x, y + 13, "(no tool calls)", 11, 500, MUTED, mono=True))
    return "".join(out)


def pick_task(base_rows, run_rows):
    best, key = None, None
    for tid in sorted(A.NO_WRITE, key=int):
        b = [r for r in base_rows if r["task_id"] == tid]
        tr = [r for r in run_rows if r["task_id"] == tid]
        if len(b) < 4 or len(tr) < 4:
            continue
        held_b = sum(not A.successful_writes(r) for r in b)
        held_t = sum(not A.successful_writes(r) for r in tr)
        nd_b, nd_t = len({tuple(r["plan"]) for r in b}), len({tuple(r["plan"]) for r in tr})
        score = abs(held_b - held_t) * 2 + abs(nd_b - nd_t)
        if best is None or score > best:
            best, key = score, tid
    return key


def main():
    S = json.loads((HERE / "results" / "summary.json").read_text())
    runs = A.load_runs()
    base = [r for r in A.jl(HERE / "results" / "baseline" / "s0" / "eval.jsonl") if r["domain"] == "airline"]
    show = "single-s1"  # the run whose refusals rose most (0.75); task 28 is the note's opening example
    fin = runs[show]["steps"][-1]
    trained = [r for r in runs[show]["eval_rows"] if r["domain"] == "airline" and r["step"] == fin]
    tid = "28"
    D = json.loads((HERE / "data" / "tau_tasks.json").read_text())
    purpose = next(t_["purpose"] for t_ in D["airline"]["tasks"] if t_["id"] == tid) or ""
    b = sorted([r for r in base if r["task_id"] == tid], key=lambda r: r["sample"])[:4]
    tr = sorted([r for r in trained if r["task_id"] == tid], key=lambda r: r["sample"])[:4]
    body = [kicker("PLURALISM · EPISODE 4", "τ-BENCH: TRAIN ON RETAIL, TEST ON AIRLINE")]
    body.append(headline("Judge-RL on one domain, measured on another", None, 66, 27))
    body.append(t(24, 98, f"Airline task {tid}: {purpose[:80]}", 12.5, 600, INK2))
    for k, (label, rows, col) in enumerate([("untrained", b, SLATE), (f"after training ({show})", tr, EMBER)]):
        y0 = 132 + k * 160
        body.append(t(24, y0, label.upper(), 11, 700, col, ls=1, mono=True))
        for i, r in enumerate(rows):
            ok = r["reward"] >= 1
            body.append(t(28, y0 + 26 + i * 30, "✓" if ok else "·", 13, 700, EMBER if ok else MUTED))
            body.append(chips(44, y0 + 12 + i * 30, r["plan"]))
    cmp_ = S["compare"]["final"]
    bx = 520
    body.append(f'<path d="M{bx - 24},112 V444" stroke="{LINE}" stroke-width="1"/>')
    items = [
        ("refusal held", "refusal_held"),
        ("…refused without hand-off", "refusal_self"),
        ("distinct plans per task", "plans_strict"),
    ]
    for j, (lab, key) in enumerate(items):
        y = 132 + j * 104
        bv = sum(S["base"][s]["airline"][key] for s in ["s0", "s1"]) / 2
        sv, jv = cmp_[f"single:{key}"][0], cmp_[f"jury:{key}"][0]
        body.append(t(bx, y, lab.upper(), 11, 700, INK2, ls=1, mono=True))
        for n, (name, v, col) in enumerate(
            [("untrained", bv, "#b9b1a7"), ("single judge", sv, SLATE), ("jury", jv, EMBER)]
        ):
            yy = y + 12 + n * 23
            body.append(rect(bx + 96, yy, 200 * v, 15, col, rx=2))
            body.append(t(bx, yy + 12, name, 11.5, 500, INK2))
            body.append(t(bx + 102 + 200 * v, yy + 12, f"{v:.2f}", 11.5, 700, INK))
    body.append(f'<path d="M20,468 H860" stroke="{LINE}" stroke-width="1"/>')
    body.append(caps("QWEN3-8B · GRPO ON RETAIL · 2 ARMS × 2 SEEDS · 50 AIRLINE TASKS × 4 CONVERSATIONS"))
    label = (
        f"Left: four sampled conversations on airline task {tid} from the untrained model and from {show} after "
        "training, each a row of tool calls, write actions outlined. Right: refusal held, refusals without a hand-off and distinct tool plans per task "
        "for the untrained model (mean of two reruns) and for the single-judge and jury arms at their final "
        "checkpoints (mean of two seeds): "
        + "; ".join(
            f"{lab} {sum(S['base'][s]['airline'][k] for s in ['s0', 's1']) / 2:.2f}, {cmp_[f'single:{k}'][0]:.2f}, "
            f"{cmp_[f'jury:{k}'][0]:.2f}"
            for lab, k in items
        )
        + "."
    )
    s = svg("".join(body), label, "pa-cover")
    write_cover(REPORT / "index.html", s)
    render_hero(s, REPORT, THUMB)
    print("cover task", tid, "run", show, "->", REPORT / "figures" / "hero.jpg", THUMB)


if __name__ == "__main__":
    main()
