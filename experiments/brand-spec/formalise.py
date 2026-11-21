"""Stage 1, formalise the brand book: build spec/<brand>_spec.json from the vendored source lines, and lint it.

    .venv/bin/python experiments/brand-spec/formalise.py

A spec has four parts, every item traced to source lines:
  tokens      palette (hex, name, grade for USWDS), type scale (size -> line height), spacing scale, font families,
              read from the token lines themselves so a token cannot drift from its line
  components  class name -> role (what counts as a default button, a header link, a logo)
  rules       id, lines (the lines a citation must name to count, strict), support (other lines it reads), kind,
              modality (must / should / avoid), settled ("text": the text decides it; "decided": we chose a reading
              the text leaves open, with `why`), applies (roles), params, overrides (for exceptions), formal (one line)
  residue     normative lines not turned into checks, each with a reason (judgment, invisible once rendered, outside
              this slice)
Every line of the source is classified once: token, definition, claim, normative (in a rule or in residue), or context.
"""

from __future__ import annotations

import re

from common import HEX_RE, SPEC_DIR, source_lines, write_json

BUTTONS = ["button-default", "button-secondary", "button-warning", "button-start", "button-inverse", "button-outline"]
LINKS = ["link", "header-link", "nav-link", "footer-link"]
DESTRUCTIVE = ["delete", "remove", "cancel", "permanently", "close", "withdraw", "revoke", "erase"]
VERBS = sorted(
    {
        "add",
        "apply",
        "accept",
        "ask",
        "book",
        "browse",
        "buy",
        "calculate",
        "cancel",
        "change",
        "check",
        "choose",
        "claim",
        "close",
        "compare",
        "confirm",
        "contact",
        "continue",
        "copy",
        "create",
        "delete",
        "download",
        "edit",
        "email",
        "enter",
        "explore",
        "file",
        "find",
        "get",
        "give",
        "go",
        "join",
        "learn",
        "log",
        "manage",
        "open",
        "order",
        "pay",
        "print",
        "read",
        "register",
        "remove",
        "renew",
        "report",
        "request",
        "reset",
        "return",
        "review",
        "save",
        "search",
        "see",
        "select",
        "send",
        "share",
        "show",
        "sign",
        "start",
        "submit",
        "subscribe",
        "track",
        "update",
        "upload",
        "use",
        "verify",
        "view",
        "visit",
    }
)


def text_components(p: str) -> list[dict]:
    """Class names the generator uses for text roles (no source line: these name components, they rule nothing)."""
    names = {
        "body": "body",
        "body-s": "body-s",
        "hint": "hint",
        "label": "label",
        "input": "input",
        "footer__link": "footer-link",
    }
    out = [{"class": f"{p}-{k}", "role": v, "lines": []} for k, v in names.items()]
    out.append({"class": "app-footer__text", "role": "footer-text", "lines": []})
    if p == "govuk":
        out += [
            {"class": "govuk-body-l", "role": "lead", "lines": []},
            {"class": "govuk-notification-banner__heading", "role": "banner-title", "lines": []},
            {"class": "govuk-notification-banner__text", "role": "banner-text", "lines": []},
        ]
    else:
        out += [
            {"class": "usa-intro", "role": "lead", "lines": []},
            {"class": "usa-alert__heading", "role": "banner-title", "lines": []},
            {"class": "usa-alert__text", "role": "banner-text", "lines": []},
        ]
    return out


def by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def rng(prefix: str, a: int, b: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(a, b + 1)]


# ---------------------------------------------------------------- GOV.UK
def govuk() -> dict:
    src = source_lines("govuk")
    L = by_id(src)
    palette = []
    for lid in rng("G", 7, 26):
        name = re.match(r"Functional colour ([\w-]+):", L[lid]["text"]).group(1)
        palette.append({"hex": HEX_RE.findall(L[lid]["text"])[0], "name": f"functional {name}", "line": lid})
    for lid in rng("G", 29, 39):
        group = re.match(r"Web palette, (\w+):", L[lid]["text"]).group(1)
        for name, h in re.findall(r"([\w-]+) (#[0-9a-f]{6})", L[lid]["text"]):
            palette.append({"hex": h, "name": f"{group.lower()} {name}", "line": lid})
    type_scale = {}
    for lid in rng("G", 48, 54):
        m = re.search(r"font size (\d+)px; line height (\d+)px", L[lid]["text"])
        type_scale[m.group(1)] = {
            "line_height": int(m.group(2)),
            "line": lid,
            "exceptional": "exceptional" in L[lid]["text"],
        }
    tokens = {
        "palette": palette,
        "type_scale": type_scale,
        "spacing": [int(v) for v in re.findall(r"= (\d+)", L["G58"]["text"])],
        "spacing_line": "G58",
        "brand_fonts": ["GDS Transport", "New Transport"],
        "brand_font_lines": ["G42", "G55"],
        "large_text": {"min_px": 24.0, "bold_min_px": round(14 * 4 / 3, 2), "bold_weight": 700, "line": "W5"},
        "dark_bg_luminance": 0.179,
    }
    components = [
        {"class": "govuk-button", "role": "button-default", "lines": ["G73"]},
        {"class": "govuk-button--secondary", "role": "button-secondary", "lines": ["G76"]},
        {"class": "govuk-button--warning", "role": "button-warning", "lines": ["G79"]},
        {"class": "govuk-button--start", "role": "button-start", "lines": ["G75"]},
        {"class": "govuk-button--inverse", "role": "button-inverse", "lines": ["G81"]},
        {"class": "govuk-link", "role": "link", "lines": ["G62"]},
        {"class": "govuk-header__link", "role": "header-link", "lines": ["G69"]},
        {"class": "govuk-service-navigation__link", "role": "nav-link", "lines": ["G69"]},
        {"class": "app-logo", "role": "logo", "lines": ["W4"]},
        *text_components("govuk"),
    ]
    rules = [
        dict(
            id="GOV-PAL",
            lines=["G1"],
            support=rng("G", 7, 26) + rng("G", 29, 39),
            kind="membership",
            modality="must",
            settled="text",
            applies="all",
            params={"props": ["color", "bg", "border"]},
            formal="every text, background and border colour is a functional or web-palette colour",
        ),
        dict(
            id="GOV-FOCUS",
            lines=["G18"],
            support=["G36"],
            kind="forbidden_value",
            modality="must",
            settled="decided",
            applies="all",
            params={"value": "#ffdd00", "props": ["color", "bg", "border"]},
            why="G36 lists yellow #ffdd00 for supporting elements such as illustrations; G18 reserves the same hex for "
            "focus. Pages here have no illustrations and no focused element, so G18 governs every element.",
            formal="#ffdd00 appears on no element (nothing is focused in a static page)",
        ),
        dict(
            id="GOV-CON",
            lines=["G2", "W1"],
            support=["W6", "W7"],
            kind="contrast",
            modality="must",
            settled="text",
            applies="text",
            params={"min": 4.5},
            formal="text contrast against its effective background >= 4.5:1",
        ),
        dict(
            id="GOV-CON-LARGE",
            lines=["W2"],
            support=["W5", "G2", "W1"],
            kind="contrast",
            modality="must",
            settled="decided",
            applies="text",
            overrides="GOV-CON",
            params={"min": 3.0, "large": True},
            why="W5 gives points; 18 pt = 24 px and 14 pt = 18.67 px at 4/3 px per point; bold = weight >= 700.",
            formal="large text (>= 24 px, or >= 18.67 px and bold): contrast >= 3:1",
        ),
        dict(
            id="GOV-CON-INACTIVE",
            lines=["W3"],
            support=["G83"],
            kind="exempt",
            modality="must",
            settled="text",
            applies="text",
            overrides="GOV-CON",
            params={"when": "disabled"},
            formal="text in a disabled (inactive) component: no contrast requirement",
        ),
        dict(
            id="GOV-CON-LOGO",
            lines=["W4"],
            support=[],
            kind="exempt",
            modality="must",
            settled="text",
            applies="text",
            overrides="GOV-CON",
            params={"when": "logo"},
            formal="text in a logo: no contrast requirement",
        ),
        dict(
            id="GOV-FONT",
            lines=["G42"],
            support=["G55"],
            kind="font_if_domain",
            modality="must",
            settled="decided",
            applies="text",
            params={"domain_suffix": "service.gov.uk", "on": True},
            why="G42 names GDS Transport and G55 names New Transport; both names are accepted as the brand font.",
            formal="on a service.gov.uk domain, every text element's font is the brand font",
        ),
        dict(
            id="GOV-FONT-ALT",
            lines=["G43"],
            support=[],
            kind="font_if_domain",
            modality="must",
            settled="text",
            applies="text",
            overrides="GOV-FONT",
            params={"domain_suffix": "service.gov.uk", "on": False},
            formal="on any other domain, no text element uses the brand font",
        ),
        dict(
            id="GOV-TYPE",
            lines=["G45"],
            support=rng("G", 48, 54),
            kind="type_scale",
            modality="must",
            settled="text",
            applies="text",
            params={},
            formal="font size is a point on the large-screen type scale (80, 48, 36, 27, 24, 19, 16 px); 80 and 27 "
            "are noted as exceptional, not failed",
        ),
        dict(
            id="GOV-LH",
            lines=["G46"],
            support=rng("G", 48, 54),
            kind="line_height",
            modality="must",
            settled="text",
            applies="text",
            params={},
            formal="line height is the type-scale line height paired with the font size",
        ),
        dict(
            id="GOV-SPACE",
            lines=["G58"],
            support=["G56", "G59", "G60"],
            kind="spacing",
            modality="must",
            settled="decided",
            applies="all",
            params={},
            why="The pages give the scale and say to reference it through the helpers (G60) but never say 'must'; "
            "we read every margin and padding as required to be on the large-screen scale.",
            formal="every margin and padding is on the spacing scale (0, 5, 10, 15, 20, 25, 30, 40, 50, 60 px)",
        ),
        dict(
            id="GOV-LINK-COL",
            lines=["G62"],
            support=["G10", "G12"],
            kind="link_colour",
            modality="must",
            settled="text",
            applies=LINKS,
            params={"allowed": ["#1a65a6", "#54319f"]},
            formal="a link on a light background is the functional link colour (or link-visited)",
        ),
        dict(
            id="GOV-LINK-INV",
            lines=["G65"],
            support=["G66", "G9"],
            kind="link_colour",
            modality="must",
            settled="decided",
            applies=LINKS,
            overrides="GOV-LINK-COL",
            params={"allowed": ["#ffffff"], "dark": True},
            why="'Dark background' is not defined; we use relative luminance < 0.179, where white text has more "
            "contrast than black.",
            formal="a link on a dark background (luminance < 0.179) is white",
        ),
        dict(
            id="GOV-LINK-UL",
            lines=["G62"],
            support=[],
            kind="underline",
            modality="must",
            settled="text",
            applies=LINKS,
            params={},
            formal="links are underlined",
        ),
        dict(
            id="GOV-LINK-NOUL",
            lines=["G69"],
            support=["G67", "G68"],
            kind="exempt",
            modality="must",
            settled="decided",
            applies=["header-link", "nav-link"],
            overrides="GOV-LINK-UL",
            params={"when": "header_or_nav"},
            why="G68 asks whether 'the context tells the user'; we take G69's example (header and side navigation) "
            "as the whole list.",
            formal="links in the header or the navigation may have no underline",
        ),
        dict(
            id="GOV-LINK-STOP",
            lines=["G62"],
            support=[],
            kind="link_stop",
            modality="must",
            settled="text",
            applies=LINKS,
            params={},
            formal="link text does not end with a full stop",
        ),
        dict(
            id="GOV-BTN-CASE",
            lines=["G71"],
            support=[],
            kind="sentence_case",
            modality="must",
            settled="decided",
            applies=BUTTONS,
            params={},
            why="Sentence case is checked as: first letter capital, no later word capitalised unless it is an "
            "acronym in capitals or contains a full stop (GOV.UK).",
            formal="button text is in sentence case",
        ),
        dict(
            id="GOV-BTN-ALIGN",
            lines=["G72"],
            support=[],
            kind="align_left",
            modality="must",
            settled="decided",
            applies=["button-default", "button-start"],
            params={"tolerance_px": 1},
            why="'The primary action button' is read as the first default or start button in each form (found by "
            "the checker's ruler: a page with two default buttons made the reading matter).",
            formal="the first default or start button inside a form starts at the form's left edge",
        ),
        dict(
            id="GOV-BTN-ONE",
            lines=["G74"],
            support=["G73"],
            kind="count_max",
            modality="avoid",
            settled="decided",
            applies=["button-default"],
            params={"max": 1},
            why="'Avoid' is read as a failure: the second and later default buttons fail.",
            formal="at most one default button per page",
        ),
        dict(
            id="GOV-BTN-START",
            lines=["G75"],
            support=[],
            kind="page_type_only",
            modality="must",
            settled="decided",
            applies=["button-start"],
            params={"page_type": "start"},
            why="G75 says when to use a start button, not that it is forbidden elsewhere; we read it as only on a "
            "Start page.",
            formal="start buttons appear only on a Start page",
        ),
        dict(
            id="GOV-BTN-WARN",
            lines=["G79"],
            support=["G78"],
            kind="destructive_label",
            modality="must",
            settled="decided",
            applies=["button-warning"],
            params={"verbs": DESTRUCTIVE},
            why="'Serious destructive consequences' is judged from the label: it must start with a destructive "
            "verb (a proxy).",
            formal="a warning button's text starts with a destructive verb (delete, remove, cancel, ...)",
        ),
        dict(
            id="GOV-BTN-INV",
            lines=["G81"],
            support=["G82"],
            kind="button_on_dark",
            modality="must",
            settled="decided",
            applies=BUTTONS,
            params={"bg": "#ffffff", "min": 4.5},
            why="Same dark-background threshold as GOV-LINK-INV.",
            formal="a button on a dark background is white, at >= 4.5:1 against that background",
        ),
    ]
    residue = [
        ("G5", "invisible once rendered: which Sass function produced a hex is not in the page"),
        ("G6", "invisible once rendered: functional and palette variables share hex values"),
        ("G27", "judgment: 'supporting elements' and 'where appropriate'"),
        ("G41", "outside this slice: organisation colours"),
        ("G48", "judgment: what counts as exceptional circumstances (80 px is allowed and noted)"),
        ("G51", "judgment: what counts as exceptional circumstances (27 px is allowed and noted)"),
        ("G61", "judgment: 'occasionally' and 'minor adjustments'"),
        ("G63", "judgment: whether link text makes an external destination clear"),
        ("G64", "outside this slice: link targets are not extracted"),
        ("G68", "judgment: whether context tells the user the text is a link (narrowed by GOV-LINK-NOUL)"),
        ("G70", "judgment: what counts as an action"),
        ("G73", "judgment: which action is the page's main call to action"),
        ("G76", "judgment: which calls to action are secondary"),
        ("G77", "judgment: how many calls to action are too many"),
        ("G80", "judgment: whether context and text make the consequence clear"),
        ("G83", "judgment: 'avoid if possible' (the contrast exemption is GOV-CON-INACTIVE)"),
        ("G84", "outside the page: what research shows"),
        ("G85", "outside this slice: button grouping"),
    ]
    kinds = {
        "token": rng("G", 7, 26) + rng("G", 29, 39) + rng("G", 49, 50) + rng("G", 52, 54) + ["G58"],
        "definition": ["W5", "W6", "W7"],
        "context": ["G3", "G4", "G28", "G40", "G44", "G47", "G56", "G57", "G59", "G60", "G66", "G67", "G78", "G82"],
    }
    return build("govuk", "GOV.UK Design System (with WCAG 2.2 1.4.3)", src, tokens, components, rules, residue, kinds)


# ---------------------------------------------------------------- USWDS
def uswds() -> dict:
    src = source_lines("uswds")
    L = by_id(src)
    palette = [
        {"hex": "#ffffff", "name": "white", "grade": 0, "line": "U6"},
        {"hex": "#000000", "name": "black", "grade": 100, "line": "U6"},
    ]
    fam_lines = [r["id"] for r in src if r["text"].startswith("System color family")]
    for lid in fam_lines:
        text = L[lid]["text"]
        fam = re.match(r"System color family ([\w-]+):", text).group(1)
        std, _, viv = text.partition("Vivid:")
        for g, h in re.findall(r"(\d+) (#[0-9a-f]{6})", std):
            palette.append({"hex": h, "name": f"{fam}-{g}", "grade": int(g), "line": lid})
        for g, h in re.findall(r"(\d+) (#[0-9a-f]{6})", viv):
            palette.append({"hex": h, "name": f"{fam}-{g}v", "grade": int(g), "line": lid})
    fonts = [
        f.strip() for f in L["U52"]["text"].split(":", 1)[1].rstrip(".").replace(", and system fonts", "").split(",")
    ]
    tokens = {
        "palette": palette,
        "palette_lines": ["U6", *fam_lines],
        "spacing": sorted({0} | {int(v) for v in re.findall(r"= (\d+)px", L["U44"]["text"])}),
        "spacing_line": "U44",
        "sizes": [int(v) for v in re.findall(r"= (\d+)px", L["U47"]["text"])],
        "size_line": "U47",
        "normalization": {n.strip(): float(v) for n, v in re.findall(r"([A-Z][\w ]+): (\d\.\d\d)", L["U51"]["text"])},
        "normalization_line": "U51",
        "fonts": [*fonts, "system-ui"],
        "font_line": "U52",
        "font_aliases": dict(re.findall(r"(?:: |; )([A-Z][\w ]+?) is served as ([\w ]+?)(?=;|\.$)", L["U63"]["text"])),
        "font_alias_line": "U63",
        "large_text": {"min_px": 24.0, "bold_min_px": 19.0, "bold_weight": 700, "line": "U5"},
        "dark_bg_luminance": 0.179,
    }
    components = [
        {"class": "usa-button", "role": "button-default", "lines": ["U54"]},
        {"class": "usa-button--outline", "role": "button-outline", "lines": ["U55"]},
        {"class": "usa-button--secondary", "role": "button-secondary", "lines": ["U56"]},
        {"class": "usa-link", "role": "link", "lines": []},
        {"class": "usa-nav__link", "role": "nav-link", "lines": []},
        {"class": "usa-logo", "role": "logo", "lines": []},
        *text_components("usa"),
    ]
    rules = [
        dict(
            id="USW-PAL",
            lines=["U1"],
            support=["U15", "U6", *fam_lines],
            kind="membership",
            modality="should",
            settled="decided",
            applies="all",
            params={"props": ["color", "bg", "border"]},
            why="'Avoid custom colors whenever possible' is read as a failure for any colour outside the system "
            "tokens (white and black are grades 0 and 100, U6).",
            formal="every text, background and border colour is a system colour token, white or black",
        ),
        dict(
            id="USW-CON",
            lines=["U5"],
            support=["U4"],
            kind="contrast",
            modality="must",
            settled="text",
            applies="text",
            params={"min": 4.5},
            formal="text contrast against its effective background >= 4.5:1",
        ),
        dict(
            id="USW-CON-LARGE",
            lines=["U5"],
            support=["U4"],
            kind="contrast",
            modality="must",
            settled="text",
            applies="text",
            overrides="USW-CON",
            params={"min": 3.0, "large": True},
            formal="large text (>= 24 px, or >= 19 px and bold): contrast >= 3:1",
        ),
        dict(
            id="USW-BLACK",
            lines=["U13"],
            support=[],
            kind="forbidden_pair",
            modality="avoid",
            settled="decided",
            applies="text",
            params={"color": "#000000", "bg": "#ffffff"},
            why="U13 gives a reason to avoid pure black on white; we read it as a failure.",
            formal="no text is pure black (#000000) on pure white (#ffffff)",
        ),
        dict(
            id="USW-SPACE",
            lines=["U43"],
            support=["U44"],
            kind="spacing",
            modality="should",
            settled="text",
            applies="all",
            params={},
            formal="every margin and padding is a spacing unit token (0, 1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, "
            "56, 64, 72, 80, 120 px)",
        ),
        dict(
            id="USW-SIZE",
            lines=["U51"],
            support=["U47", "U50", "U45", "U46"],
            kind="normalized_size",
            modality="must",
            settled="decided",
            applies="text",
            params={"rel_tol": 0.006, "abs_tol": 0.1},
            why="Sizes are target x normalization; the multiples in U51 are rounded to two decimals, so a size "
            "within 0.6% (or 0.1 px) of target x multiple is accepted.",
            formal="font size = a system size token's target x the font's normalization",
        ),
        dict(
            id="USW-FAMILY",
            lines=["U52"],
            support=["U63"],
            kind="font_member",
            modality="must",
            settled="text",
            applies="text",
            params={},
            formal="the font is one of the USWDS available fonts or the system font",
        ),
        dict(
            id="USW-BTN-CASE",
            lines=["U58"],
            support=[],
            kind="sentence_case",
            modality="must",
            settled="decided",
            applies=BUTTONS,
            params={},
            why="Same sentence-case test as GOV-BTN-CASE.",
            formal="button labels are in sentence case",
        ),
        dict(
            id="USW-BTN-VERB",
            lines=["U60"],
            support=[],
            kind="lead_verb",
            modality="must",
            settled="decided",
            applies=BUTTONS,
            params={"verbs": VERBS},
            why="'A verb' is checked against a list of common interface verbs (a proxy).",
            formal="the first word of a button label is a verb",
        ),
        dict(
            id="USW-BTN-MANY",
            lines=["U57"],
            support=[],
            kind="count_max",
            modality="avoid",
            settled="decided",
            applies=BUTTONS,
            params={"max": 5},
            why="'Too many' is not defined; we set 5. Never planted as a violation, and not asked of the models.",
            formal="at most 5 buttons per page",
        ),
        dict(
            id="USW-BTN-SHORT",
            lines=["U59"],
            support=[],
            kind="max_words",
            modality="should",
            settled="decided",
            applies=BUTTONS,
            params={"max": 4},
            why="'Short' is not defined; we set 4 words. Never planted as a violation, and not asked of the models.",
            formal="a button label has at most 4 words",
        ),
    ]
    residue = [
        ("U12", "advice on choosing colours, not a page requirement (its arithmetic is checked as a claim, see ruler)"),
        ("U14", "judgment: whether colour is the only signal"),
        ("U16", "judgment: 'typically' and 'usually'"),
        ("U17", "judgment: 'can be thought of'; an area proportion, outside this slice"),
        ("U53", "judgment: which actions are the most important"),
        ("U54", "judgment: whether an action goes a next step"),
        ("U55", "judgment: whether an action happens on the current page"),
        ("U56", "judgment: which action most users should select"),
        ("U61", "invisible in a static render: focus states"),
        ("U62", "judgment: whether an explanation is given"),
    ]
    kinds = {
        "token": ["U44", "U47", "U48", "U51", "U52", "U63", *fam_lines],
        "definition": ["U2", "U6", "U7"],
        "claim": ["U3", "U8", "U9", "U10", "U11"],
        "context": ["U4", "U15", "U45", "U46", "U49", "U50"],
    }
    return build("uswds", "U.S. Web Design System", src, tokens, components, rules, residue, kinds)


def build(brand, title, src, tokens, components, rules, residue, kinds) -> dict:
    ids = {r["id"] for r in src}
    normative = {ln for r in rules for ln in r["lines"]} | {ln for ln, _ in residue}
    classified = {}
    for k, lst in kinds.items():
        for ln in lst:
            classified.setdefault(ln, k)
    for ln in normative:
        classified[ln] = "normative"
    spec = {
        "brand": brand,
        "title": title,
        "source": [{"id": r["id"], "page": r["page"], "text": r["text"], "upstream": r["upstream"]} for r in src],
        "tokens": tokens,
        "components": components,
        "rules": rules,
        "residue": [{"line": ln, "reason": why} for ln, why in residue],
        "line_kinds": {r["id"]: classified.get(r["id"], "context") for r in src},
    }
    lint(spec, ids)
    return spec


def lint(spec: dict, ids: set[str]) -> None:
    rules = {r["id"]: r for r in spec["rules"]}
    assert len(rules) == len(spec["rules"])
    for r in spec["rules"]:
        assert r["lines"], r["id"]
        for ln in r["lines"] + r["support"]:
            assert ln in ids, (r["id"], ln)
        if "overrides" in r:
            assert r["overrides"] in rules, r["id"]
        if r["settled"] == "decided":
            assert r.get("why"), r["id"]
        assert r["modality"] in ("must", "should", "avoid") and r["kind"] and r["formal"]
    for t in spec["tokens"]["palette"]:
        assert t["line"] in ids and re.fullmatch(r"#[0-9a-f]{6}", t["hex"]), t
    for x in spec["residue"]:
        assert x["line"] in ids
    for c in spec["components"]:
        assert all(ln in ids for ln in c["lines"])


def summary(spec: dict) -> dict:
    rules = spec["rules"]
    kinds = spec["line_kinds"]
    in_rules = {ln for r in rules for ln in r["lines"]}
    return {
        "source_lines": len(spec["source"]),
        "normative_lines": sum(v == "normative" for v in kinds.values()),
        "normative_in_rules": len(in_rules),
        "residue_lines": len(spec["residue"]),
        "rules": len(rules),
        "exceptions": sum("overrides" in r for r in rules),
        "settled_text": sum(r["settled"] == "text" for r in rules),
        "settled_decided": sum(r["settled"] == "decided" for r in rules),
        "palette_tokens": len({t["hex"] for t in spec["tokens"]["palette"]}),
        "line_kinds": {k: sum(v == k for v in kinds.values()) for k in sorted(set(kinds.values()))},
    }


if __name__ == "__main__":
    SPEC_DIR.mkdir(exist_ok=True)
    out = {}
    for spec in (govuk(), uswds()):
        write_json(SPEC_DIR / f"{spec['brand']}_spec.json", spec)
        out[spec["brand"]] = summary(spec)
        print(spec["brand"], out[spec["brand"]])
    write_json(SPEC_DIR / "summary.json", out)
