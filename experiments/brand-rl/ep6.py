"""
Shared pieces for The Taste Stack Episode 6 (an RL environment for "on brand, not slop, not a copy"): paths, the
policy model, the rule families and their split, the briefs and the prompt.

Episode 5's brand spec, checker and copy check (`../brand-spec/`) and Episode 2's block-tree features
(`../design-index/features.py`) are imported read-only, never copied. This module is called `ep6` (not `common`)
because both of those folders have their own `common.py`.

Plan (fixed before any confirmatory run): the plan
"""

from __future__ import annotations

import html as htmllib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
BRAND_SPEC = EXP / "brand-spec"
DESIGN_INDEX = EXP / "design-index"
DATA = HERE / "data"
RESULTS = HERE / "results"
CACHE = Path.home() / ".cache" / "steerlab" / "brand-rl"  # raw policy pages and screenshots, never committed

POLICY = "Qwen/Qwen3-4B-Instruct-2507"
BRAND = "govuk"
MAX_NEW_TOKENS = 3072
SPLITS = {"train": (0, 600), "dev": (10_000, 10_048), "heldout": (20_000, 20_096), "monitor": (30_000, 30_096)}
# brief seeds per split; "monitor" briefs only build the constructed-hack monitor (never trained on or scored)

# ---------------------------------------------------------------- Episode 5's modules (read-only)


def brand_spec():
    """(checker, common, copycheck) from Episode 5. Its folder goes first on sys.path so `common` is its own."""
    if str(BRAND_SPEC) not in sys.path:
        sys.path.insert(0, str(BRAND_SPEC))
    import checker
    import common as bs_common
    import copycheck

    return checker, bs_common, copycheck


def ep2_features():
    """Episode 2's features.py loaded by path (its folder also has a common.py, so it never goes on sys.path)."""
    name = "ep2_features"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DESIGN_INDEX / "features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_spec() -> dict:
    return json.loads((BRAND_SPEC / "spec" / f"{BRAND}_spec.json").read_text())


# ---------------------------------------------------------------- rule families
# A family is a rule plus its exceptions (an exception decides the check instead of the base rule, so a page is
# scored on the family). 16 families, 22 rules.
FAMILIES = {
    "PAL": ["GOV-PAL"],
    "FOCUS": ["GOV-FOCUS"],
    "CON": ["GOV-CON", "GOV-CON-LARGE", "GOV-CON-INACTIVE", "GOV-CON-LOGO"],
    "FONT": ["GOV-FONT", "GOV-FONT-ALT"],
    "TYPE": ["GOV-TYPE"],
    "LH": ["GOV-LH"],
    "SPACE": ["GOV-SPACE"],
    "LINK-COL": ["GOV-LINK-COL", "GOV-LINK-INV"],
    "LINK-UL": ["GOV-LINK-UL", "GOV-LINK-NOUL"],
    "LINK-STOP": ["GOV-LINK-STOP"],
    "BTN-CASE": ["GOV-BTN-CASE"],
    "BTN-ALIGN": ["GOV-BTN-ALIGN"],
    "BTN-ONE": ["GOV-BTN-ONE"],
    "BTN-START": ["GOV-BTN-START"],
    "BTN-WARN": ["GOV-BTN-WARN"],
    "BTN-INV": ["GOV-BTN-INV"],
}
FAMILY_OF = {r: f for f, rs in FAMILIES.items() for r in rs}
SPLIT_FILE = HERE / "configs" / "family_split.json"


def family_split() -> dict:
    """{"reward": [...], "heldout": [...]}: fixed in the plan before any training (configs/family_split.json)."""
    return json.loads(SPLIT_FILE.read_text())


# ---------------------------------------------------------------- briefs (units of truth, fixed before any output)
NOUNS = [
    "licence",
    "permit",
    "grant",
    "refund",
    "passport",
    "benefit",
    "appointment",
    "record",
    "account",
    "report",
    "registration",
    "certificate",
    "allowance",
    "payment",
    "claim",
    "booking",
    "vehicle tax",
    "loan",
    "pension",
    "parking permit",
    "fishing licence",
    "childcare account",
    "student loan",
    "council tax discount",
]
VERBS = ["Apply for", "Renew", "Check", "Update", "Cancel", "Report", "Change", "Manage", "Pay for", "Book"]
ADJ = ["new", "annual", "temporary", "online", "local", "business", "personal", "short-term"]
SENTS = [
    "You can {v} your {n} online if you have your reference number.",
    "It takes around {k} minutes to {v} your {n}.",
    "You will need details of your {n} and a valid email address.",
    "We will send a confirmation to your email address within {k} working days.",
    "If you cannot use this service, contact the helpline.",
    "Read the guidance before you {v} your {n}.",
    "You can save your answers and come back later.",
    "Most people get a decision within {k} weeks.",
    "You must be 18 or over to use this service.",
    "This service is also available in Welsh.",
    "You will not be charged to {v} your {n}.",
    "Your {n} will be checked before it is approved.",
]
QUESTIONS = [
    "What is your {n} reference number?",
    "When did your {n} start?",
    "What is your postcode?",
    "What is your National Insurance number?",
    "How many people are on your {n}?",
    "What is the registration number of your vehicle?",
    "What is your date of birth?",
]
HINTS = [
    "For example, QQ 12 34 56 C.",
    "You can find this on your {n} letter.",
    "For example, 27 3 2007.",
    "This is 8 characters long.",
    "Do not include spaces.",
]
LINKS = [
    "Read the guidance",
    "Contact us",
    "Check if you are eligible",
    "Find out more",
    "See the full list",
    "Report a problem with this page",
    "How we use your data",
    "Get help with this service",
]
FOOTER = ["Help", "Privacy", "Cookies", "Accessibility statement", "Contact", "Terms and conditions"]
BUTTONS = ["Continue", "Save and continue", "Send application", "Confirm and send", "Check your answers", "Submit"]
SECONDARY = ["Save for later", "Add another", "Change details", "Go back"]
WARN = ["Delete your account", "Cancel your {n}", "Remove this {n}"]
PAGE_TYPES = ["start", "question", "content", "confirmation", "cancel"]


def _v(verb: str) -> str:
    return verb.lower()


def make_brief(seed: int, split: str) -> dict:
    """One brief: a page type, a service, a domain and the items the page must show, each with its exact text.

    Items (kind, text): h1, para, button (variant start/default/secondary/warning), label (an input's label, with an
    input), hint, link, banner_title + banner_text, footer_link. Every text is fixed here, before any page exists."""
    r = random.Random(seed)
    n, verb, adj = r.choice(NOUNS), r.choice(VERBS), r.choice(ADJ)
    pt = PAGE_TYPES[seed % len(PAGE_TYPES)] if split != "train" else r.choice(PAGE_TYPES)
    service = f"{verb} your {n}" if r.random() < 0.6 else f"{verb} a {adj} {n}"
    campaign = r.random() < 0.2  # GOV.UK's font rule depends on the domain (the GOV-FONT-ALT exception)
    slug = re.sub(r"[^a-z]+", "-", service.lower()).strip("-")
    domain = f"{slug}.campaign.gov.uk" if campaign else f"{slug}.service.gov.uk"

    def sent():
        return r.choice(SENTS).format(v=_v(r.choice(VERBS)), n=n, k=r.randint(2, 20))

    items: list[dict] = []
    if pt == "start":
        items += [{"kind": "h1", "text": service}]
        items += [{"kind": "para", "text": s} for s in r.sample([sent() for _ in range(6)], 2)]
        items += [{"kind": "button", "variant": "start", "text": "Start now"}]
        items += [{"kind": "link", "text": t} for t in r.sample(LINKS, 2)]
    elif pt == "question":
        items += [{"kind": "h1", "text": r.choice(QUESTIONS).format(n=n)}]
        items += [
            {"kind": "label", "text": f"{n.capitalize()} reference number" if r.random() < 0.5 else "Your answer"}
        ]
        if r.random() < 0.7:
            items += [{"kind": "hint", "text": r.choice(HINTS).format(n=n)}]
        items += [{"kind": "button", "variant": "default", "text": r.choice(BUTTONS)}]
        if r.random() < 0.5:
            items += [{"kind": "button", "variant": "secondary", "text": r.choice(SECONDARY)}]
        items += [{"kind": "link", "text": r.choice(LINKS)}]
    elif pt == "content":
        items += [{"kind": "h1", "text": f"About your {n}" if r.random() < 0.5 else f"How to {_v(verb)} your {n}"}]
        items += [{"kind": "para", "text": sent()} for _ in range(3)]
        items += [{"kind": "banner_title", "text": "Important"}, {"kind": "banner_text", "text": sent()}]
        items += [{"kind": "link", "text": t} for t in r.sample(LINKS, 3)]
    elif pt == "confirmation":
        items += [{"kind": "h1", "text": r.choice(["Application complete", "Request sent", "Payment received"])}]
        items += [{"kind": "para", "text": f"Your reference number is {r.randint(100000, 999999)}."}]
        items += [{"kind": "para", "text": sent()}]
        items += [{"kind": "link", "text": r.choice(LINKS)}]
    else:  # cancel: a destructive action
        items += [{"kind": "h1", "text": f"Are you sure you want to cancel your {n}?"}]
        items += [{"kind": "para", "text": sent()}]
        items += [{"kind": "button", "variant": "warning", "text": r.choice(WARN).format(n=n)}]
        items += [{"kind": "button", "variant": "secondary", "text": "Go back"}]
    items += [{"kind": "footer_link", "text": t} for t in r.sample(FOOTER, 3)]
    return {
        "id": f"{split}-{seed}",
        "seed": seed,
        "split": split,
        "page_type": pt,
        "service": service,
        "domain": domain,
        "items": items,
    }


def briefs(split: str) -> list[dict]:
    lo, hi = SPLITS[split]
    return [make_brief(s, split) for s in range(lo, hi)]


# ---------------------------------------------------------------- the prompt
def brand_sheet() -> str:
    """The brand guide given to the policy: every source line a rule cites or leans on, quoted with its line id."""
    spec = load_spec()
    keep = set()
    for r in spec["rules"]:
        keep |= set(r["lines"]) | set(r.get("support", []))
    lines = [s for s in spec["source"] if s["id"] in keep]
    return "\n".join(f"[{s['id']}] {s['text']}" for s in lines)


def component_list() -> str:
    spec = load_spec()
    return ", ".join(f"{c['class']} ({c['role']})" for c in spec["components"])


ATTR_RE = re.compile(r'\s(data-id|style|class)="([^"]*)"')
TAG_RE = re.compile(r"<([a-z0-9]+)((?:\s[^<>]*?)?)(/?)>")


def compact_html(page_html: str) -> str:
    """Episode 5's generated pages carry every style inline. For the prompt, move each distinct inline style into one
    class rule in the <style> element and drop data-id, so the exemplar renders the same at about half the tokens."""
    styles: dict[str, str] = {}

    def fix(m):
        tag, attrs, slash = m.group(1), m.group(2), m.group(3)
        found = dict(ATTR_RE.findall(attrs))
        rest = ATTR_RE.sub("", attrs)
        cls = found.get("class", "").split()
        st = htmllib.unescape(found.get("style", ""))
        if st:
            styles.setdefault(st, f"s{len(styles) + 1}")
            cls.append(styles[st])
        c = f' class="{" ".join(cls)}"' if cls else ""
        return f"<{tag}{c}{rest}{slash}>"

    body = TAG_RE.sub(fix, page_html)
    css = "".join(f".{k}{{{v}}}" for v, k in styles.items())
    return body.replace("</style>", css + "</style>", 1)


def exemplars(split: str) -> list[dict]:
    """Reference pages from Episode 5's copy set (the brand's 'existing pages'): its dev references for train and dev
    briefs, its held-out references for held-out briefs."""
    src = "dev" if split in ("train", "dev") else "heldout"
    rows = [json.loads(x) for x in (BRAND_SPEC / "data" / f"copy_{BRAND}_{src}.jsonl").read_text().splitlines() if x]
    return [r for r in rows if r["kind"] == "ref"]


ITEM_WORDS = {
    "h1": "the main heading (h1)",
    "para": "a paragraph",
    "label": "a text input with the label",
    "hint": "hint text under that label",
    "link": "a link",
    "banner_title": "a notification banner with the title",
    "banner_text": "the notification banner's text",
    "footer_link": "a footer link",
}


def item_line(it: dict) -> str:
    if it["kind"] == "button":
        return f'- a {it["variant"]} button: "{it["text"]}"'
    return f'- {ITEM_WORDS[it["kind"]]}: "{it["text"]}"'


def build_prompt(brief: dict, exemplar: dict) -> str:
    must = "\n".join(item_line(it) for it in brief["items"])
    return (
        "You design web pages for a public service that uses the GOV.UK Design System.\n\n"
        "Brand guide (quoted from the design system; line ids in brackets):\n"
        f"{brand_sheet()}\n\n"
        f"Component class names (and what each is for): {component_list()}.\n\n"
        "An existing page from the same service library, for reference:\n"
        f"```html\n{compact_html(exemplar['html'])}\n```\n\n"
        f'Write a new {brief["page_type"]} page for the service "{brief["service"]}" on the domain '
        f"{brief['domain']}. The page must show, with exactly this text:\n{must}\n\n"
        "Follow the brand guide. Reply with one complete HTML document (from <!doctype html> to </html>) and nothing "
        "else. Put all CSS in one <style> element in the head."
    )


def prompt_for(brief: dict) -> str:
    ex = exemplars(brief["split"])
    return build_prompt(brief, ex[brief["seed"] % len(ex)])


def exemplar_id(brief: dict) -> str:
    ex = exemplars(brief["split"])
    return ex[brief["seed"] % len(ex)]["id"]


HTML_RE = re.compile(r"(<!doctype html.*?</html>|<html.*?</html>)", re.IGNORECASE | re.DOTALL)


def parse_reply(text: str) -> str | None:
    """The HTML document in a reply (code fences allowed), or None. An unclosed document is returned as-is from its
    start: the browser renders it, and the run records it as truncated if generation hit the token cap."""
    m = HTML_RE.search(text)
    if m:
        return m.group(1)
    i = text.lower().find("<!doctype html")
    if i < 0:
        i = text.lower().find("<html")
    return text[i:] if i >= 0 else None


if __name__ == "__main__":
    b = briefs("dev")[0]
    p = prompt_for(b)
    print(p[-1500:])
    print(len(p), "chars")
