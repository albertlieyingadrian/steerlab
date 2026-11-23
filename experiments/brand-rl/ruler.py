"""The pass bars for every scorer the reward or an unseen ruler uses (plan, Design 3). Laptop only.

    .venv/bin/python experiments/brand-rl/ruler.py        # writes results/ruler.json; exits non-zero on a failed bar

R1 renderer      Episode 5's cached facts reproduced on all its GOV.UK compliance pages (dev + held-out)
R2 reward map    on Episode 5's pages, the failing families are exactly the planted rules' families
R3 brief rulers  an oracle page per dev and held-out brief: coverage 1, strict 1, every family passes; each
                 single-item deletion lowers strict by exactly 1/n; div soup gives strict 0 and loose 1
R4 slop probes   (slop_fit.py asserts the refit) + in-domain read: Episode 5 template pages vs base dev pages
R5 copy gate     Episode 5's copy-set gate decisions reproduced through Scorer.copy_parts
R6 loop flag     20 constructed looping pages flagged, 20 base pages not
"""

from __future__ import annotations

import gzip
import json
import pickle
import re
import sys

import numpy as np

import ep6
from render import render_many
from reward import COPY_THRESHOLD, Scorer, coverage_fidelity, loop_flag

RESULTS = ep6.RESULTS


def pages_mod():
    ep6.brand_spec()
    import pages

    return pages


# ---------------------------------------------------------------- oracle pages for a brief (Episode 5's components)
def oracle(brief: dict, drop: int | None = None) -> str:
    """A compliant GOV.UK page that shows every brief item with its exact text (item `drop` left out)."""
    P = pages_mod()
    th = P.Theme("govuk", "service" if brief["domain"].endswith(".service.gov.uk") else "campaign")
    items = [it for i, it in enumerate(brief["items"]) if i != drop]
    main, form, buttons, footer = [], [], [], []
    banner = {}
    for it in items:
        k, t = it["kind"], it["text"]
        if k == "h1":
            main.append(P.T(th, "h1", "h1", "h1", t, bold=True, family="head", margin=(0, 0, 30, 0)))
        elif k == "para":
            main.append(P.T(th, "p", "body", "body", t, cls=["govuk-body"], margin=(0, 0, 20, 0)))
        elif k == "label":
            form.append(P.T(th, "label", "label", "label", t, cls=["govuk-label"], bold=True, margin=(0, 0, 5, 0)))
            form.append(
                P.N(
                    "input",
                    "input",
                    ["govuk-input"],
                    {
                        "border": "2px solid #0b0c0c",
                        "padding": "5px",
                        "margin": "0 0 30px 0",
                        "width": "300px",
                        "background-color": "#ffffff",
                    },
                    attrs={"type": "text"},
                )
            )
        elif k == "hint":
            form.insert(
                1 if form else 0,
                P.T(th, "p", "hint", "hint", t, cls=["govuk-hint"], color=th.c["sec"], margin=(0, 0, 5, 0)),
            )
        elif k == "button":
            buttons.append(P.button(th, t, it["variant"]))
        elif k == "link":
            main.append(P.N("p", "body", [], {"margin": "0 0 20px 0"}, kids=[P.link(th, t)]))
        elif k in ("banner_title", "banner_text"):
            banner[k] = t
        elif k == "footer_link":
            footer.append(P.link(th, t, role="footer-link", kind="body-s", cls=["govuk-footer__link"]))
    if banner:
        kids = []
        if "banner_title" in banner:
            kids.append(
                P.T(
                    th,
                    "h2",
                    "banner-title",
                    "banner",
                    banner["banner_title"],
                    cls=["govuk-notification-banner__heading"],
                    color="#ffffff",
                    bold=True,
                    margin=(0, 0, 10, 0),
                )
            )
        if "banner_text" in banner:
            kids.append(
                P.T(
                    th,
                    "p",
                    "banner-text",
                    "banner",
                    banner["banner_text"],
                    cls=["govuk-notification-banner__text"],
                    color="#ffffff",
                )
            )
        box = P.N(
            "div",
            "container",
            ["govuk-notification-banner"],
            {"background-color": th.c["brand"], "padding": "20px", "margin": "0 0 30px 0"},
            kids=kids,
        )
        box.tags.add("dark")
        main.insert(1, box)
    if form or buttons:
        grp = P.N("div", "container", [], {"margin": "0"}, kids=buttons)
        main.append(P.N("form", "container", [], {"margin": "0", "padding": "0"}, kids=[*form, grp]))
    for lk in footer:
        lk.style["margin"] = "0 20px 0 0"
    body = [
        P.sec_header(th, P.Txt(brief["seed"]), {"n_links": 2}),
        P.band(th, main, tags=("main",)),
        P.band(
            th,
            [P.N("div", "container", [], {"display": "flex", "margin": "0"}, kids=footer)],
            bg=th.c["footer"],
            pad=40,
            tag="footer",
            tags=("footer",),
        ),
    ]
    root = P.N(
        "body",
        "container",
        [],
        {
            "margin": "0",
            "padding": "0",
            "background-color": "#ffffff",
            "font-family": th.font["body"],
            "color": th.c["text"],
        },
        kids=body,
    )
    P.number(root)
    return P.to_html(root, brief["domain"])


DIV_TAGS = re.compile(r"<(/?)(h1|h2|h3|p|a|button|label|input)\b([^>]*)>", re.IGNORECASE)


def div_soup(page_html: str) -> str:
    """Every heading, paragraph, link, button, label and input becomes a classless div (text kept)."""

    def fix(m):
        attrs = re.sub(r'\sclass="[^"]*"', "", m.group(3))
        return f"<{m.group(1)}div{attrs}>"

    return DIV_TAGS.sub(fix, page_html)


# ---------------------------------------------------------------- the checks
def r1_r2(sc: Scorer) -> dict:
    out = {"r1_pages": 0, "r1_identical": 0, "r2_pages": 0, "r2_exact": 0, "r2_mismatch": []}
    for split in ("dev", "heldout"):
        rows = [
            json.loads(x) for x in (ep6.BRAND_SPEC / "data" / f"pages_govuk_{split}.jsonl").read_text().splitlines()
        ]
        facts = [
            json.loads(x)
            for x in (ep6.BRAND_SPEC / "data" / f"facts_pages_govuk_{split}.jsonl").read_text().splitlines()
        ]
        rend = render_many([r["html"] for r in rows])
        for r, f, o in zip(rows, facts, rend):
            out["r1_pages"] += 1
            out["r1_identical"] += o["elements"] == f["elements"]
            brief = {
                "id": r["id"],
                "domain": r["domain"],
                "page_type": r["page_type"],
                "items": [],
                "split": "dev",
                "seed": 0,
            }
            s = sc.score(brief, o, False, exemplar="govuk-dev-ref01")
            got = sorted(fm for fm, v in s["fam"].items() if v[1])
            want = sorted({ep6.FAMILY_OF[rule] for _, rule in r["labels"]})
            out["r2_pages"] += 1
            if got == want:
                out["r2_exact"] += 1
            else:
                out["r2_mismatch"].append({"page": r["id"], "got": got, "want": want})
    return out


def r3(sc: Scorer) -> dict:
    out = {
        "oracle": 0,
        "oracle_ok": 0,
        "brand_all_pass": 0,
        "deletions": 0,
        "deletions_ok": 0,
        "soup": 0,
        "soup_ok": 0,
        "failures": [],
    }
    for split in ("dev", "heldout"):
        htmls, meta = [], []
        for b in ep6.briefs(split):
            htmls.append(oracle(b))
            meta.append((b, "oracle", None))
            htmls.append(div_soup(oracle(b)))
            meta.append((b, "soup", None))
            texts = [it["text"].casefold() for it in b["items"]]
            for i, it in enumerate(b["items"]):
                if texts.count(it["text"].casefold()) == 1:  # a repeated text still matches after one deletion
                    htmls.append(oracle(b, drop=i))
                    meta.append((b, "drop", i))
        rend = render_many(htmls)
        for (b, kind, i), o in zip(meta, rend):
            cov, strict, loose = coverage_fidelity(b, o["elements"])
            n = len(b["items"])
            if kind == "oracle":
                s = sc.score(b, o, False)
                out["oracle"] += 1
                ok = cov == 1 and strict == 1 and loose == 1
                out["oracle_ok"] += ok
                allpass = all(v[1] == 0 for v in s["fam"].values())
                out["brand_all_pass"] += allpass
                if not (ok and allpass):
                    out["failures"].append(
                        {
                            "brief": b["id"],
                            "kind": kind,
                            "cov": cov,
                            "strict": strict,
                            "fails": [f for f, v in s["fam"].items() if v[1]],
                        }
                    )
            elif kind == "drop":
                out["deletions"] += 1
                ok = abs(strict - (n - 1) / n) < 1e-9
                out["deletions_ok"] += ok
                if not ok:
                    out["failures"].append({"brief": b["id"], "kind": kind, "item": i, "strict": strict, "n": n})
            else:
                out["soup"] += 1
                ok = strict == 0 and loose == 1
                out["soup_ok"] += ok
                if not ok:
                    out["failures"].append({"brief": b["id"], "kind": kind, "strict": strict, "loose": loose})
    out["failures"] = out["failures"][:20]
    return out


def r4_in_domain(sc: Scorer) -> dict:
    """Markup probe on Episode 5's template pages (GOV.UK, dev) vs the base model's valid dev pages (reported only)."""
    from sklearn.metrics import roc_auc_score

    rows = [json.loads(x) for x in (ep6.BRAND_SPEC / "data" / "pages_govuk_dev.jsonl").read_text().splitlines()]
    tpl = [
        sc.score(
            {"id": r["id"], "domain": r["domain"], "page_type": r["page_type"], "items": [], "split": "dev", "seed": 0},
            o,
            False,
            exemplar="govuk-dev-ref01",
        )["slop_p"]
        for r, o in zip(rows, render_many([r["html"] for r in rows]))
    ]
    d = pickle.load(gzip.open(ep6.CACHE / "samples" / "base_dev_explore.pkl.gz"))
    base = [r["slop_p"] for r in d["rows"] if r["valid"] and not r["truncated"]]
    y = [0] * len(tpl) + [1] * len(base)
    return {
        "template_mean": float(np.mean(tpl)),
        "base_mean": float(np.mean(base)),
        "n_template": len(tpl),
        "n_base": len(base),
        "auroc_base_vs_template": float(roc_auc_score(y, tpl + base)),
    }


def r5(sc: Scorer) -> dict:
    """Episode 5's copy gate on its own copy set (references of the same split, as Episode 5 compares)."""
    cc = sc.cc
    out = {}
    for split in ("dev", "heldout"):
        facts = [
            json.loads(x)
            for x in (ep6.BRAND_SPEC / "data" / f"facts_copy_govuk_{split}.jsonl").read_text().splitlines()
        ]
        refs = {f["id"]: cc.feats(f) for f in facts if "-ref" in f["id"]}
        cand = [f for f in facts if "-near" in f["id"] or "-new" in f["id"]]
        near_blocked = new_allowed = 0
        for f in cand:
            ff = cc.feats(f)
            s = round(max(cc.sim(ff, r)["structure"] for r in refs.values()), 4)  # as Episode 5
            if "-near" in f["id"]:
                near_blocked += s >= COPY_THRESHOLD
            else:
                new_allowed += s < COPY_THRESHOLD
        n_near = sum("-near" in f["id"] for f in cand)
        out[split] = {
            "near_blocked": near_blocked,
            "n_near": n_near,
            "new_allowed": new_allowed,
            "n_new": len(cand) - n_near,
            "gate_correct": near_blocked + new_allowed,
        }
    ep5 = json.loads((ep6.BRAND_SPEC / "results" / "copy.json").read_text())["brands"]["govuk"]
    out["episode5_gate_accuracy"] = {sp: ep5[sp]["gate_accuracy"] for sp in ("dev", "heldout")}
    return out


def r6() -> dict:
    d = pickle.load(gzip.open(ep6.CACHE / "samples" / "base_dev_explore.pkl.gz"))
    good = [r for r in d["rows"] if r["valid"] and not r["truncated"]][:20]
    htmls = [ep6.parse_reply(r["text"]) for r in good]
    para = '<p class="govuk-body">You can save your answers and come back later and continue.</p>'
    loops = [h.replace("</body>", para * 30 + "</body>", 1) for h in htmls]
    flags = [loop_flag(o["elements"]) for o in render_many(htmls + loops)]
    return {"normal_flagged": int(sum(flags[:20])), "loops_flagged": int(sum(flags[20:])), "n": 20}


def main() -> None:
    sc = Scorer()
    res = {"r1_r2": r1_r2(sc), "r3": r3(sc), "r4_in_domain": r4_in_domain(sc), "r5": r5(sc), "r6": r6()}
    a, b3, c5, d6 = res["r1_r2"], res["r3"], res["r5"], res["r6"]
    bars = {
        "R1": a["r1_identical"] == a["r1_pages"],
        "R2": a["r2_exact"] == a["r2_pages"],
        "R3": b3["oracle_ok"] == b3["oracle"]
        and b3["brand_all_pass"] == b3["oracle"]
        and b3["deletions_ok"] == b3["deletions"]
        and b3["soup_ok"] == b3["soup"],
        "R5": all(c5[sp]["gate_correct"] == c5["episode5_gate_accuracy"][sp][0] for sp in ("dev", "heldout")),
        "R6": d6["normal_flagged"] == 0 and d6["loops_flagged"] == d6["n"],
    }
    res["pass"] = bars
    (RESULTS / "ruler.json").write_text(json.dumps(res, indent=1))
    summary = {k: v for k, v in res.items() if k != "r3"}
    summary["r3"] = {k: v for k, v in b3.items() if k != "failures"}
    print(json.dumps(summary, indent=1))
    if b3["failures"]:
        print("R3 failures (first):", json.dumps(b3["failures"][:5]))
    sys.exit(0 if all(bars.values()) else 1)


if __name__ == "__main__":
    main()
