"""
Safeguards, script 3: the probe-first cascade. How much of the expensive model can we skip, at the same 1% FPR?

Constitutional Classifiers++ screens all traffic with a cheap linear probe and sends only the uncertain or suspicious
part on to an expensive classifier. Here:

  stage 1: the linear probe on Qwen2.5-1.5B (script 2). Everything goes through it.
  stage 2: the prompted Qwen2.5-7B, called only for prompts whose probe score is above a gate.

The gate is set so a chosen share of harmless calibration traffic is escalated (1% to 50%): that share is the cost,
because real traffic is almost all harmless. The final threshold keeps the whole cascade at a 1% false-positive rate
on calibration traffic. Two ways to decide on escalated prompts: the 7B alone, or an ensemble (the sum of the two
standardised scores, as the paper does).

Compared with: the probe alone, and the 7B on everything. All on the clean test set; script 4 attacks it.

Run: python experiments/ai-safety-101/safeguards/3_cascade.py   (seconds; needs script 2)
"""

import json

import guard as g
import numpy as np

import tools as t

SHARES = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
RESULT = g.RESULTS / "3_cascade.json"


def main():
    tab = g.load_table()
    secs = json.loads((g.RESULTS / "2_classifiers.json").read_text())
    ms = {k: 1000 * v / secs["n_items"] for k, v in secs["seconds"].items()}
    calib = tab.mask(split="calib")
    clean = tab.mask(split="test", transform="clean")
    adv, dolly = clean & tab.mask(source="advbench"), clean & tab.mask(source="dolly")
    xs_safe, xs_unsafe = clean & tab.mask(source="xstest", label=0), clean & tab.mask(source="xstest", label=1)

    rows = []

    def report(name, flagged, escalated):
        row = {
            "name": name,
            "catch_advbench": g.rate(flagged[adv]),
            "fpr_dolly": g.rate(flagged[dolly]),
            "fpr_xstest_safe": g.rate(flagged[xs_safe]),
            "catch_xstest_unsafe": g.rate(flagged[xs_unsafe]),
            "escalated_dolly": float(escalated[dolly].mean()),
            "escalated_advbench": float(escalated[adv].mean()),
        }
        # Cost per prompt on harmless traffic: the probe (taken as free: it reads the served model's own
        # activations) plus the 7B on the escalated share.
        row["cost_ms"] = row["escalated_dolly"] * ms["prompted_7B"]
        rows.append(row)
        print(
            f"{name:>22s} {t.fmt(row['catch_advbench']):>22s} {row['fpr_dolly'][0]:>8.3f} "
            f"{row['fpr_xstest_safe'][0]:>9.3f} {row['catch_xstest_unsafe'][0]:>10.3f} "
            f"{row['escalated_dolly']:>9.1%} {row['escalated_advbench']:>9.1%} {row['cost_ms']:>9.1f}"
        )

    none = np.zeros(len(tab["id"]), dtype=bool)
    print(
        "Clean test set: 260 AdvBench, 400 Dolly, 250 XSTest safe, 200 XSTest unsafe. Every row is set to a 1% FPR "
        "on calibration traffic."
    )
    print("'7B calls' = share of test prompts sent to the 7B (harmless Dolly; harmful AdvBench in brackets).\n")
    print(
        f"{'protocol':>22s} {'catch AdvBench [95% CI]':>22s} {'FPR Dolly':>8s} {'FPR XS-safe':>9s} "
        f"{'catch XS-uns':>10s} {'7B calls':>9s} {'(harmful)':>9s} {'ms/prompt':>9s}"
    )
    report("probe alone", tab["probe_1.5B"] > g.threshold_at_fpr(tab["probe_1.5B"][calib]), none)
    report("7B on everything", tab["prompted_7B"] > g.threshold_at_fpr(tab["prompted_7B"][calib]), ~none)
    for combine in ("7B", "ensemble"):
        for share in SHARES:
            flagged, escalated, _ = g.cascade(tab, share, combine)
            report(f"cascade {share:>4.0%} -> {combine}", flagged, escalated)

    full = rows[1]
    print(
        f"\nThe 7B costs {ms['prompted_7B']:.1f} ms per prompt here; "
        f"a separate 1.5B probe pass {ms['probe_1.5B']:.1f} ms "
        "(about 0 when the probe reads the served model's own activations)."
    )
    good = [r for r in rows[2:] if r["catch_advbench"][0] >= full["catch_advbench"][0]]
    if good:
        cheapest = min(good, key=lambda r: r["escalated_dolly"])
        print(
            f"Cheapest cascade that catches as much as the 7B on everything ({full['catch_advbench'][0]:.3f}): "
            f"{cheapest['name'].strip()}, catch {cheapest['catch_advbench'][0]:.3f}, 7B called on "
            f"{cheapest['escalated_dolly']:.1%} of harmless traffic."
        )
    else:
        print("No cascade setting matched the 7B on everything.")
    t.write_json(RESULT, {"ms_per_prompt": ms, "rows": rows})

    print("\nSelf-checks")
    t.check(
        "every cascade flags at most 1% of calibration traffic",
        all(g.cascade(tab, s, c)[0][calib].mean() <= 0.01 for s in SHARES for c in ("7B", "ensemble")),
    )
    t.check(
        "the escalated share of calibration traffic matches the knob (within 1 point)",
        all(abs(g.cascade(tab, s)[1][calib].mean() - s) <= 0.01 for s in SHARES),
    )
    t.check(
        "a cascade only flags what it escalated",
        all(not (f & ~e).any() for f, e, _ in (g.cascade(tab, s) for s in SHARES)),
    )


if __name__ == "__main__":
    main()
