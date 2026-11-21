"""Score P1-P9 against their frozen thresholds from the shipped results/ files -> results/predictions.json.

    python analyze.py

Scoring rules are the plan's M0 operationalisation. The reason
column explains an outcome but never changes a score.
"""

from __future__ import annotations

import json

from common import RESULTS, load_json, save_json


def opt(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def main():
    lb = load_json("leaderboard.json")
    rows = []

    def add(pid, text, metric, threshold, conf, value, hit, reason):
        rows.append(
            dict(
                id=pid,
                prediction=text,
                metric=metric,
                threshold=threshold,
                confidence=conf,
                value=value,
                hit=bool(hit),
                reason=reason,
            )
        )

    d = {
        k: round(lb[f"act_{k}"]["test"]["auroc"] - lb[f"sae_{k}"]["test"]["auroc"], 4)
        for k in ("qwen", "llama")
        if f"sae_{k}" in lb
    }
    add(
        "P1",
        "An activation linear probe matches or beats SAE-feature probes at in-distribution detection",
        "AUROC(act) − AUROC(SAE), held-out authors, Qwen3-8B",
        "≥ −0.01",
        "70%",
        d,
        d["qwen"] >= -0.01,
        f"Qwen {lb['act_qwen']['test']['auroc']:.3f} vs {lb['sae_qwen']['test']['auroc']:.3f}; "
        f"Llama difference {d.get('llama')}",
    )

    def drop(m):
        return round(
            lb[m]["test"]["auroc"] - (lb[m]["hum_prompt"]["auroc"] + lb[m]["hum_dipper"]["auroc"]) / 2,
            4,
        )

    dq = dict(act=drop("act_qwen"), sae=drop("sae_qwen"))
    add(
        "P2",
        "SAE-feature probes hold up better under humanizers than raw activation probes",
        "drop(act) − drop(SAE), mean of DIPPER and prompted rewrite, Qwen3-8B",
        "≥ 0.05",
        "40%",
        round(dq["act"] - dq["sae"], 4),
        dq["act"] - dq["sae"] >= 0.05,
        f"drops {dq}",
    )

    fails = {}
    for m, r in lb.items():
        if "test" in r and r["test"]["auroc"] >= 0.90:
            fails[m] = round(
                max(r["test"]["auroc"] - r[e]["auroc"] for e in ("test_fmt", "test_len") if e in r),
                4,
            )
    bad = {m: v for m, v in fails.items() if v > 0.15}
    add(
        "P3",
        "At least one method with AUROC ≥ 0.9 fails the shortcut check",
        "largest drop under format or length",
        "> 0.15 for any method",
        "65%",
        fails,
        bool(bad),
        f"failing: {bad or 'none'}",
    )

    ex = opt("explain.json")
    if ex:
        t5, t7 = ex["test5"], ex["test7"]
        add(
            "P4",
            "SAE or neuron reading surfaces more planted hacks than any probe",
            "hacks surfaced (of 6): max(SAE, neuron) − best probe-family readout",
            "≥ 2",
            "55%",
            t5["surfaced"],
            t5["p4_hit"],
            f"best probe-family readout {t5['best_probe_family']}",
        )
        add(
            "P6",
            "SAE or neuron reading flags most planted probe confounds",
            "confounds flagged (of 4)",
            "≥ 3",
            "50%",
            t7["flagged"],
            t7["p6_hit"],
            f"clean control flagged: {t7['clean_control_flagged']}",
        )
    ca = opt("causal.json")
    if ca:
        add(
            "P5",
            "Steering along the probe direction beats random, and an SAE intervention beats the probe",
            "P(AI) drop: probe − best random; best SAE − probe (95% CI above 0)",
            "both > 0",
            "35%",
            dict(
                probe_minus_random=ca["probe_minus_best_random"],
                sae_minus_probe=ca["sae_minus_probe"],
            ),
            ca["p5_hit"],
            f"best SAE arm {ca['best_sae']}",
        )
    pg = opt("pigment.json")
    if pg:
        add(
            "P7",
            "An internal signal flags the hacking checkpoint before the outside check",
            "saved checkpoints of lead, both hack runs; no false alarm on the clean run",
            "≥ 1",
            "40%",
            {k: v["leads"] for k, v in pg["verdict"].items()},
            pg["p7_hit"],
            "false alarms: " + str({k: v["false_alarm"] for k, v in pg["verdict"].items()}),
        )
    vo = opt("voice.json")
    if vo:
        add(
            "P8",
            "Steering gets at least half of a LoRA's voice gain at under 1% of its cost",
            "gain ratio (vec12 ÷ lora12) and build-cost ratio",
            "≥ 0.5 and ≤ 0.01",
            "45%",
            dict(
                gain_ratio=round(vo["p8"]["gain_ratio"], 3),
                cost_ratio=round(vo["p8"]["cost_ratio"], 4),
            ),
            vo["p8"]["hit"],
            f"fluent both arms: {vo['p8']['fluent']}",
        )
        add(
            "P9",
            "Per-author vectors beat 8-shot prompting on voice at equal samples",
            "voice position, vector(8) − 8-shot",
            "> 0.05",
            "50%",
            vo["p9"]["diff_vec8_minus_shot8"],
            vo["p9"]["hit"],
            "",
        )
    # re-checks after the run (reported beside the frozen score, never changing it; user decision 2026-11-21)
    rc = opt("explain_recheck.json") or {}
    vl, cl, ve = (
        opt("voice_loopcheck.json") or {},
        opt("causal_loopcheck.json") or {},
        opt("voice_explore.json") or {},
    )
    label = {
        "P4": (
            "grader failed its re-check",
            f"graders agree {rc.get('agreement', {}).get('grader_vs_regrader')}; "
            f"second grader {rc.get('test5', {}).get('regrader')}; author's read {rc.get('test5', {}).get('author')}",
        ),
        "P6": (
            "grader failed its re-check",
            f"second grader {rc.get('test7', {}).get('regrader')}; author's read "
            f"{rc.get('test7', {}).get('author')}; clean control flagged by both readouts",
        ),
        "P5": (
            "artefact: loops (best arm)",
            f"loop-free SAE minus probe {cl.get('sae_dir_minus_probe')}; best arm loops {cl.get('sae_abl_loop_share')}",
        ),
        "P8": (
            "artefact: loops",
            f"vector loop share {vl.get('conditions', {}).get('vec12')}",
        ),
        "P9": (
            "artefact: loops",
            f"vec8 loop share {vl.get('conditions', {}).get('vec8')}; loop-free rerun (post hoc) "
            f"vec8 - 8-shot {ve.get('vec8_minus_shot8')}, own - other author {ve.get('own_minus_wrong')}",
        ),
    }
    for r in rows:
        if r["id"] in label:
            r["recheck"], r["recheck_detail"] = label[r["id"]]
    save_json("predictions.json", rows)
    for r in rows:
        print(r["id"], "HIT " if r["hit"] else "miss", r["value"], "|", r["reason"])


if __name__ == "__main__":
    main()
