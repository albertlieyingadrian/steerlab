"""Summaries, intervals and the pre-registered "beats baseline" tests, from results/*.json only.

    .venv/bin/python experiments/physics-world-models/analyse.py            # prints tables, writes results/summary.json

Intervals are 95% bootstraps (10,000 resamples) that resample test trajectories (9 for TRL2D, 100 for DR), the
unit the data were drawn in, and, for a model trained with several seeds, the seeds too. A difference between two
models is paired: both are scored on the same resampled trajectories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"
B = 10_000
RNG = np.random.default_rng(0)

# Published numbers (test split). The Well, arXiv 2412.00568, Table 2 (one-step VRMSE).
WELL_PUBLISHED = {"FNO": 0.5001, "TFNO": 0.5016, "UNetClassic": 0.2418, "UNetConvNext": 0.1956}
# Tuned TFNO, arXiv 2610.33967, Table 1 (one-step VRMSE, one input frame).
TFNO_TUNED_PUBLISHED = 0.220964
# PDEBench, arXiv 2210.07182, 2D diffusion-reaction nRMSE.
PDEBENCH_PUBLISHED = {"FNO": 0.12, "U-Net": 0.84}
# Beyond Local Accuracy, arXiv 2610.31935: scratch FNO on PDEBench DR at N = 64 training trajectories.
BLA_FNO_N64 = {"short": 0.2104, "full": 0.4113}

TRL_SHORT = 11  # rollout steps 0..10 = frames 4..14 (11 frames, as the DR short horizon of 2610.31935)


def load(name: str) -> dict | None:
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ------------------------------------------------------------------------------------------------ TRL2D


def trl_arrays(ev: dict):
    """-> one-step per trajectory (9,), rollout curve per trajectory (9, 97), both averaged over the 4 fields."""
    one = np.asarray(ev["one_step"], dtype=float)  # (windows, C)
    roll = np.asarray(ev["rollout"], dtype=float)  # (traj, T, C)
    roll = np.nan_to_num(roll, nan=np.inf)  # a rollout that blew up (inf - inf) counts as infinitely wrong
    n_traj = roll.shape[0]
    per_traj_one = one.mean(1).reshape(n_traj, -1).mean(1)
    return per_traj_one, roll.mean(2)


def trl_point(ev: dict) -> dict:
    one, curve = trl_arrays(ev)
    e_s, e_f = curve[:, :TRL_SHORT].mean(1), curve.mean(1)
    return {
        "one_step": float(one.mean()),
        "official_one_step": next(iter(ev.get("official_one_step", {}).values()), None),
        "roll_1_20": float(curve[:, :20].mean()),  # the "1:20 rollout" window of the Flowers paper
        "roll_6_12": float(curve[:, 6:12].mean()),
        "roll_13_30": float(curve[:, 13:30].mean()),
        "short_median": float(np.median(e_s)),
        "full_median": float(np.median(e_f)),
        "full_mean": float(e_f.mean()),
        "G": float(np.median(e_f) / np.median(e_s)),
        "n_full_gt_1": int((e_f > 1).sum()),
        "n_traj": len(one),
    }


def boot_trl(evs: list[dict], ref: list[dict] | None = None) -> dict:
    """Bootstrap over trajectories (and seeds) of one-step, full-rollout median and G; paired diffs vs `ref`."""
    arr = [trl_arrays(e) for e in evs]
    one = np.stack([a[0] for a in arr])  # (S, 9)
    full = np.stack([a[1].mean(1) for a in arr])
    short = np.stack([a[1][:, :TRL_SHORT].mean(1) for a in arr])
    if ref is not None:
        rarr = [trl_arrays(e) for e in ref]
        r_one = np.stack([a[0] for a in rarr])
        r_full = np.stack([a[1].mean(1) for a in rarr])
    S, N = one.shape
    stats = {k: [] for k in ("one", "full", "G", "d_one", "d_full")}
    for _ in range(B):
        t = RNG.integers(0, N, N)
        s = RNG.integers(0, S, S)
        o = one[s][:, t].mean()
        f = np.median(full[s][:, t].mean(0))
        stats["one"].append(o)
        stats["full"].append(f)
        stats["G"].append(f / np.median(short[s][:, t].mean(0)))
        if ref is not None:
            rs = RNG.integers(0, len(rarr), len(rarr))
            stats["d_one"].append(o - r_one[rs][:, t].mean())
            stats["d_full"].append(f - np.median(r_full[rs][:, t].mean(0)))
    out = {f"{k}_ci": ci(v) for k, v in stats.items() if v}
    out["one_mean"] = float(one.mean())
    out["full_median"] = float(np.median(full.mean(0)))
    out["G"] = float(np.median(full.mean(0)) / np.median(short.mean(0)))
    out["n_seeds"] = S
    return out


# ------------------------------------------------------------------------------------------------ DR


def dr_arrays(ev: dict):
    v = np.asarray(ev["vrmse_per_traj_frame"], dtype=float)  # (N, 91, C)
    return v[:, :11].mean((1, 2)), v.mean((1, 2))


def boot_dr(evs: list[dict], ref: list[dict] | None = None) -> dict:
    sh = np.stack([dr_arrays(e)[0] for e in evs])
    fu = np.stack([dr_arrays(e)[1] for e in evs])
    nr = np.array([e["nrmse_pdebench"] for e in evs])
    if ref is not None:
        r_sh = np.stack([dr_arrays(e)[0] for e in ref])
        r_fu = np.stack([dr_arrays(e)[1] for e in ref])
    S, N = sh.shape
    st = {k: [] for k in ("short", "full", "G", "d_short", "d_full")}
    for _ in range(B):
        t = RNG.integers(0, N, N)
        s = RNG.integers(0, S, S)
        a, b = np.median(sh[s][:, t].mean(0)), np.median(fu[s][:, t].mean(0))
        st["short"].append(a)
        st["full"].append(b)
        st["G"].append(b / a)
        if ref is not None:
            rs = RNG.integers(0, len(ref), len(ref))
            st["d_short"].append(a - np.median(r_sh[rs][:, t].mean(0)))
            st["d_full"].append(b - np.median(r_fu[rs][:, t].mean(0)))
    out = {f"{k}_ci": ci(v) for k, v in st.items() if v}
    out |= {
        "nrmse_mean": float(nr.mean()),
        "nrmse_seeds": nr.round(5).tolist(),
        "short_median": float(np.median(sh.mean(0))),
        "full_median": float(np.median(fu.mean(0))),
        "G": float(np.median(fu.mean(0)) / np.median(sh.mean(0))),
        "n_seeds": S,
    }
    return out


# ---------------------------------------------------------------------------------------- finals (pre-registered)

# Published rollout VRMSE for TRL2D, The Well Table 3 (windows 6:12 and 13:30); Flowers (arXiv 2603.04430) Table 1.
WELL_ROLL_PUBLISHED = {
    "UNetConvNext": {"roll_6_12": 0.54, "roll_13_30": 1.01},
    "FNO": {"roll_6_12": 1.79, "roll_13_30": 3.54},
}
FLOWERS_TRL = {
    "Flower": {"one_step": 0.1930, "roll_1_20": 0.5491},
    "CNextU-net (their run)": {"one_step": 0.2559, "roll_1_20": 0.7051},
}
BLA_POSEIDON_N64 = {"short": 0.1161, "full": 0.5338}

FINALS = {  # ID -> (run prefix, dataset, reproduction to pair with or None, published numbers by metric)
    "T-pf": (
        "f_trl_ft_pf4",
        "trl",
        "repro_trl_UNetConvNext",
        {"one_step": WELL_PUBLISHED["UNetConvNext"], **WELL_ROLL_PUBLISHED["UNetConvNext"]},
    ),
    "T-1s": (
        "f_trl_ft_onestep",
        "trl",
        "repro_trl_UNetConvNext",
        {"one_step": WELL_PUBLISHED["UNetConvNext"], **WELL_ROLL_PUBLISHED["UNetConvNext"]},
    ),
    "T-fno": ("f_trl_fno", "trl", None, {"one_step": WELL_PUBLISHED["FNO"], **WELL_ROLL_PUBLISHED["FNO"]}),
    "D-cnext": ("f_dr_cnext", "dr", "repro_dr_2D_diff-react_NA_NA_FNO", {"nrmse": PDEBENCH_PUBLISHED["FNO"]}),
    "D-fno": ("f_dr_fno_w64m20", "dr", "repro_dr_2D_diff-react_NA_NA_FNO", {"nrmse": PDEBENCH_PUBLISHED["FNO"]}),
    "D-fno64": ("f_dr_fno_w64m20_n64", "dr", None, BLA_FNO_N64),
    "D-fno64c1": ("f_dr_fno_w64m20_ctx1_n64", "dr", None, BLA_FNO_N64),
    "D-pos64": ("f_dr_poseidonT_n64", "dr", None, BLA_POSEIDON_N64),
}
TRL_METRICS = ("one_step", "roll_1_20", "roll_6_12", "roll_13_30", "short", "full", "G")
DR_METRICS = ("short", "full", "G")


def trl_traj_metrics(ev: dict) -> dict:
    """Per-trajectory arrays (9,) for each TRL2D metric; medians/means are taken over trajectories later."""
    one, curve = trl_arrays(ev)
    return {
        "one_step": one,
        "roll_1_20": curve[:, :20].mean(1),
        "roll_6_12": curve[:, 6:12].mean(1),
        "roll_13_30": curve[:, 13:30].mean(1),
        "short": curve[:, :TRL_SHORT].mean(1),
        "full": curve.mean(1),
    }


def dr_traj_metrics(ev: dict) -> dict:
    sh, fu = dr_arrays(ev)
    return {"short": sh, "full": fu}


def _agg(m: dict, idx_s, idx_t) -> dict:
    """Aggregate per-seed, per-trajectory arrays (S, N): mean over trajectories for the one-step and window metrics,
    median for short/full (2610.31935), G = median full / median short."""
    out = {}
    for k, a in m.items():
        x = a[idx_s][:, idx_t].mean(0)  # average seeds first, per trajectory
        out[k] = float(np.median(x)) if k in ("short", "full") else float(x.mean())
    out["G"] = out["full"] / out["short"]
    return out


def boot_final(evs: list[dict], ds: str, ref: dict | None) -> dict:
    f = trl_traj_metrics if ds == "trl" else dr_traj_metrics
    per = [f(e) for e in evs]
    m = {k: np.stack([p[k] for p in per]) for k in per[0]}
    r = {k: v[None] for k, v in f(ref).items()} if ref is not None else None
    S, N = next(iter(m.values())).shape
    point = _agg(m, np.arange(S), np.arange(N))
    draws = {k: [] for k in point}
    diffs = {k: [] for k in point} if r is not None else {}
    for _ in range(B):
        t, s = RNG.integers(0, N, N), RNG.integers(0, S, S)
        a = _agg(m, s, t)
        for k, v in a.items():
            draws[k].append(v)
        if r is not None:
            b = _agg(r, np.zeros(1, int), t)
            for k in a:
                diffs[k].append(a[k] - b[k])
    out = {"point": point, "ci": {k: ci(v) for k, v in draws.items()}, "n_seeds": S, "n_traj": N}
    if r is not None:
        out["ref_point"] = _agg(r, np.zeros(1, int), np.arange(N))
        out["diff_ci"] = {k: ci(v) for k, v in diffs.items()}
    if ds == "dr":
        nr = [e["nrmse_pdebench"] for e in evs]
        out["nrmse_seeds"] = [round(x, 5) for x in nr]
        out["point"]["nrmse"] = float(np.mean(nr))
        out["ci"]["nrmse"] = [float(min(nr)), float(max(nr))]  # seed range: per-trajectory nRMSE was not saved
        if ref is not None:
            out["ref_point"]["nrmse"] = ref["nrmse_pdebench"]
    return out


def verdicts(b: dict, published: dict) -> dict:
    """The pre-registered rule: beats published if CI upper < published; beats reproduction if paired diff CI < 0."""
    v = {}
    for k, pub in published.items():
        up = b["ci"][k][1]
        row = {"published": pub, "point": b["point"][k], "ci": b["ci"][k], "beats_published": bool(up < pub)}
        if "diff_ci" in b and k in b["diff_ci"]:
            row["diff_ci"] = b["diff_ci"][k]
            row["beats_reproduction"] = bool(b["diff_ci"][k][1] < 0)
            row["beats_baseline"] = row["beats_published"] and row["beats_reproduction"]
        elif "ref_point" in b and k in b["ref_point"]:  # nRMSE: no per-trajectory pairing; every seed below ours
            row["reproduction"] = b["ref_point"][k]
            row["beats_reproduction"] = bool(max(b["nrmse_seeds"]) < b["ref_point"][k])
            row["beats_baseline"] = row["beats_published"] and row["beats_reproduction"]
        v[k] = row
    if "diff_ci" in b:  # paired diffs on every metric, for the record
        v["_paired_vs_reproduction"] = {k: {"diff_ci": b["diff_ci"][k], "ref": b["ref_point"][k]} for k in b["diff_ci"]}
    return v


def finals(summary: dict):
    print("\n== Finals (pre-registered, test split, seeds 1-3)")
    summary["finals"] = {}
    for fid, (prefix, ds, ref_name, pub) in FINALS.items():
        evs = [r["eval"] for s in (1, 2, 3) if (r := load(f"run_{prefix}_s{s}")) is not None]
        if not evs:
            continue
        ref = load(ref_name) if ref_name else None
        b = boot_final(evs, ds, ref)
        b["verdicts"] = verdicts(b, pub)
        summary["finals"][fid] = b
        pt = " ".join(f"{k}={v:.4f}" for k, v in b["point"].items())
        print(f"{fid:10s} seeds={b['n_seeds']} {pt}")
        for k, row in b["verdicts"].items():
            if k.startswith("_"):
                continue
            print(
                f"    {k:11s} {row['point']:.4f} CI {np.round(row['ci'], 4)} vs published {row['published']}: "
                f"beats_published={row['beats_published']}"
                + (
                    f" diff CI {np.round(row['diff_ci'], 4)} beats_reproduction={row['beats_reproduction']}"
                    if "diff_ci" in row
                    else ""
                )
                + (
                    f" reproduction {row['reproduction']:.4f} beats_reproduction={row['beats_reproduction']}"
                    if "reproduction" in row
                    else ""
                )
            )
    # The Well's TRL2D reproductions on the same metrics (single deterministic model; trajectory bootstrap only)
    summary["trl_repro_metrics"] = {}
    for name in ("UNetConvNext", "UNetClassic", "tfno_tuned_released"):
        ev = load(f"repro_trl_{name}")
        if ev is not None:
            summary["trl_repro_metrics"][name] = boot_final([ev], "trl", None)
    summary["published"] = {
        "well_one_step": WELL_PUBLISHED,
        "well_rollout": WELL_ROLL_PUBLISHED,
        "tfno_tuned": TFNO_TUNED_PUBLISHED,
        "pdebench": PDEBENCH_PUBLISHED,
        "bla_fno_n64": BLA_FNO_N64,
        "bla_poseidon_n64": BLA_POSEIDON_N64,
        "flowers_trl": FLOWERS_TRL,
    }


def main():
    summary = {"trl_repro": {}, "dr_repro": {}}
    print("== TRL2D: released checkpoints on the test split (The Well's code)")
    print(
        f"{'model':28s} {'published':>9s} {'ours':>8s} {'official':>8s} {'6:12':>7s} {'13:30':>7s} {'short':>7s} "
        f"{'full':>7s} {'G':>6s}"
    )
    for name, pub in [*WELL_PUBLISHED.items(), ("tfno_tuned_released", TFNO_TUNED_PUBLISHED)]:
        ev = load(f"repro_trl_{name}")
        if ev is None:
            continue
        p = trl_point(ev)
        p["published"] = pub
        p["boot"] = boot_trl([ev])
        summary["trl_repro"][name] = p
        print(
            f"{name:28s} {pub:9.4f} {p['one_step']:8.4f} {p['official_one_step'] or float('nan'):8.4f} "
            f"{p['roll_6_12']:7.3f} {p['roll_13_30']:7.3f} {p['short_median']:7.3f} {p['full_median']:7.3f} "
            f"{p['G']:6.2f}  CI one-step {np.round(p['boot']['one_ci'], 4)}"
        )
    print("\n== DR: released PDEBench models on the test split (PDEBench's rollout and nRMSE)")
    for name in ["FNO", "Unet-PF-20", "Unet-AR", "Unet-1-step"]:
        ev = load(f"repro_dr_2D_diff-react_NA_NA_{name}")
        if ev is None:
            continue
        b = boot_dr([ev])
        summary["dr_repro"][name] = b
        print(
            f"{name:12s} nRMSE {b['nrmse_mean']:.4f}  VRMSE short {b['short_median']:.4f} "
            f"{np.round(b['short_ci'], 4)}  full {b['full_median']:.4f} {np.round(b['full_ci'], 4)}  G {b['G']:.2f}"
        )
    print("\n== Training runs (split as recorded; exploration runs are on validation)")
    summary["runs"] = {}
    for p in sorted(RES.glob("run_*.json")):
        r = json.loads(p.read_text())
        cfg, ev = r["cfg"], r["eval"]
        lever = ",".join(
            f"{k}={cfg[k]}" for k in ("init", "residual", "pf_max", "noise", "width", "modes", "n_train") if cfg.get(k)
        )
        if cfg["dataset"] == "trl":
            pt = trl_point(ev)
            row = {
                "split": r["eval_split"],
                "one_step": pt["one_step"],
                "roll_1_20": pt["roll_1_20"],
                "short": pt["short_median"],
                "full": pt["full_median"],
                "G": pt["G"],
            }
        else:
            row = {
                "split": r["eval_split"],
                "nrmse": ev["nrmse_pdebench"],
                "short": ev["vrmse_short_median"],
                "full": ev["vrmse_full_median"],
                "G": ev["G_median_ratio"],
            }
        row |= {"kind": cfg["kind"], "lever": lever, "minutes": cfg["minutes"], "steps": r["steps"]}
        summary["runs"][cfg["run"]] = row
        print(
            f"{cfg['run']:34s} "
            + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        )
    finals(summary)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
