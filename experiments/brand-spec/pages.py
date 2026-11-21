"""The page generator: brand components -> HTML with inline token values, plus the facts it meant to render.

A page is a *recipe* (brand, page type, domain, font variant, sections with their own text seed and style knobs)
rendered to a node tree, optionally mutated by planted violations or exception traps, then serialised to HTML.
Every node carries `data-id` (e1, e2, ... in document order). The recipe is what the copy set edits: a near-copy
keeps a reference's sections and changes text seeds (strength 1), then style knobs (2), then one section (3).

Each planted violation is designed to fail exactly one rule on exactly one element; `labels` records those
(element, rule) pairs, so a compliant page has labels []. Exception traps are compliant pages that lean on an
exception (a disabled button's low contrast, large text at 3:1, a logo, a non-service.gov.uk domain).
"""

from __future__ import annotations

import copy
import html
import random

from common import load_spec

# ---------------------------------------------------------------- words (plain, invented service text)
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
    "visa",
    "payment",
    "claim",
    "booking",
    "test",
    "vehicle",
    "loan",
]
ADJS = ["new", "current", "annual", "temporary", "online", "local", "business", "personal", "renewed", "short-term"]
SENT = [
    "You can {v} your {n} online if you have your reference number.",
    "It takes around {k} minutes to {v} a {a} {n}.",
    "You will need details of your {n} and your {n2}.",
    "We will send a confirmation to your email address within {k} working days.",
    "If you cannot {v} online, contact the helpline.",
    "Your {a} {n} must be checked before you {v} it.",
    "Read the guidance before you {v} your {n}.",
    "You can save your answers and come back later.",
    "Most people get a decision within {k} weeks.",
    "This service is also available in other languages.",
]
VERBS_T = ["apply for", "renew", "check", "update", "cancel", "report", "change", "manage", "pay for", "book"]
BUTTON_LABELS = [
    "Continue",
    "Save and continue",
    "Sign in",
    "Pay",
    "Confirm and send",
    "Find a service",
    "Check your answers",
    "Download the form",
    "Add another address",
    "Send request",
    "Submit claim",
]
SECONDARY_LABELS = ["Save for later", "Add another", "Change details", "Go back", "View example"]
LINK_TEXTS = [
    "Read the guidance",
    "Contact us",
    "Check eligibility",
    "Find out more",
    "See the full list",
    "Report a problem",
    "Accessibility statement",
    "Privacy notice",
    "Terms and conditions",
    "Help",
]
USWDS_VERBS = set(next(r for r in load_spec("uswds")["rules"] if r["id"] == "USW-BTN-VERB")["params"]["verbs"])


class Txt:
    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def noun(self):
        return self.r.choice(NOUNS)

    def heading(self):
        n, v = self.noun(), self.r.choice(VERBS_T)
        return self.r.choice(
            [
                f"{v.capitalize()} a {n}",
                f"Your {n} details",
                f"Check your {n}",
                f"{n.capitalize()} and {self.noun()} support",
                f"About your {self.r.choice(ADJS)} {n}",
            ]
        )

    def sentence(self):
        return self.r.choice(SENT).format(
            v=self.r.choice(VERBS_T), n=self.noun(), n2=self.noun(), a=self.r.choice(ADJS), k=self.r.randint(2, 20)
        )

    def para(self, k=None):
        return " ".join(self.sentence() for _ in range(k or self.r.randint(1, 3)))

    def service(self):
        return f"{self.r.choice(VERBS_T).capitalize()} a {self.r.choice(ADJS)} {self.noun()}"


# ---------------------------------------------------------------- node tree
class N:
    def __init__(self, tag, role="container", cls=(), style=None, text=None, attrs=None, kids=None):
        self.tag, self.role, self.cls = tag, role, list(cls)
        self.style = dict(style or {})
        self.text, self.attrs, self.kids = text, dict(attrs or {}), list(kids or [])
        self.id = None
        self.tags: set[str] = set()  # generator-only labels: "card", "form", "dark", ...
        self.planted = False

    def walk(self):
        yield self
        for k in self.kids:
            yield from k.walk()


def px(v) -> str:
    return "0" if v == 0 else f"{v}px"


def box(vals) -> str:
    return " ".join(px(v) for v in vals)


class Theme:
    """Brand tokens for the generator (every value is a palette, scale or font token of the spec)."""

    def __init__(self, brand: str, variant: str):
        self.brand, self.variant = brand, variant
        if brand == "govuk":
            fam = '"GDS Transport", arial, sans-serif' if variant == "service" else "arial, sans-serif"
            self.font = {"head": fam, "body": fam}
            self.c = dict(
                text="#0b0c0c",
                sec="#484949",
                link="#1a65a6",
                white="#ffffff",
                surface="#f4f8fb",
                brand="#1d70b8",
                border="#cecece",
                input="#0b0c0c",
                btn="#0f7a52",
                btn2="#f3f3f3",
                warn="#ca3535",
                dark="#0f385c",
                footer="#f3f3f3",
                panel="#0f7a52",
                alt_surface="#f3f8f6",
            )
            self.size = {
                "h1": (48, 50),
                "h2": (36, 40),
                "h3": (24, 30),
                "body": (19, 25),
                "body-s": (16, 20),
                "lead": (24, 30),
                "label": (19, 25),
                "hint": (19, 25),
                "logo": (24, 30),
                "hlink": (16, 20),
                "btn": (19, 25),
                "banner": (19, 25),
                "h1-alt": (36, 40),
                "h2-alt": (24, 30),
            }
            self.space = {"s": 10, "xs": 5, "m": 20, "l": 30, "xl": 40, "btn": (10, 15), "input": 5, "gap": 15}
        else:
            if variant == "public":
                fam = '"Public Sans", system-ui, sans-serif'
                self.font = {"head": fam, "body": fam}
                mh = mb = 1.0
            else:
                self.font = {"head": "Merriweather, Georgia, serif", "body": '"Source Sans Pro", Helvetica, sans-serif'}
                mh, mb = 0.98, 1.06
            self.c = dict(
                text="#1b1b1b",
                sec="#565c65",
                link="#005ea2",
                white="#ffffff",
                surface="#f0f0f0",
                brand="#162e51",
                border="#dfe1e2",
                input="#1b1b1b",
                btn="#005ea2",
                btn2="#d83933",
                warn="#d83933",
                dark="#162e51",
                footer="#f0f0f0",
                panel="#1a4480",
                alt_surface="#e7f6f8",
                alert_border="#00bde3",
            )

            def s(t, m):
                return round(t * m, 2)

            self.size = {
                "h1": (s(40, mh), s(48, mh)),
                "h2": (s(32, mh), s(40, mh)),
                "h3": (s(22, mh), s(28, mh)),
                "body": (s(17, mb), s(26, mb)),
                "body-s": (s(15, mb), s(22, mb)),
                "lead": (s(22, mb), s(32, mb)),
                "label": (s(17, mb), s(26, mb)),
                "hint": (s(15, mb), s(22, mb)),
                "logo": (s(22, mh), s(28, mh)),
                "hlink": (s(15, mb), s(22, mb)),
                "btn": (s(17, mb), s(22, mb)),
                "banner": (s(17, mb), s(26, mb)),
                "h1-alt": (s(32, mh), s(40, mh)),
                "h2-alt": (s(22, mh), s(28, mh)),
            }
            self.space = {"s": 8, "xs": 4, "m": 16, "l": 24, "xl": 32, "btn": (12, 20), "input": 8, "gap": 16}
        self.p = "govuk" if brand == "govuk" else "usa"

    def text_style(self, kind, color=None, bold=False, family="body", underline=False, margin=(0, 0, 0, 0)):
        size, lh = self.size[kind]
        return {
            "color": color or self.c["text"],
            "font-family": self.font[family],
            "font-size": px(size),
            "line-height": px(lh),
            "font-weight": "700" if bold else "400",
            "margin": box(margin),
            "text-decoration": "underline" if underline else "none",
        }


# ---------------------------------------------------------------- components
def T(th, tag, role, kind, text, cls=(), **kw):
    return N(tag, role, cls, th.text_style(kind, **kw), text=text)


def link(th, text, role="link", color=None, underline=True, cls=None, kind="body"):
    c = cls if cls is not None else ([f"{th.p}-link"] if role == "link" else [])
    return N(
        "a",
        role,
        c,
        th.text_style(kind, color=color or th.c["link"], underline=underline),
        text=text,
        attrs={"href": "#"},
    )


def button(th, text, variant="default", margin_right=None):
    c, sp = th.c, th.space
    mr = sp["gap"] if margin_right is None else margin_right
    st = th.text_style("btn", color=c["white"], margin=(0, mr, 0, 0))
    st.update(
        {
            "background-color": c["btn"],
            "padding": box((sp["btn"][0], sp["btn"][1], sp["btn"][0], sp["btn"][1])),
            "border": "0",
            "display": "inline-block",
        }
    )
    cls, role = [f"{th.p}-button"], "button-default"
    if variant == "secondary":
        if th.brand == "govuk":
            st.update({"background-color": c["btn2"], "color": c["text"]})
            cls.append("govuk-button--secondary")
            role = "button-secondary"
        else:
            st.update({"background-color": c["white"], "color": c["btn"], "border": f"2px solid {c['btn']}"})
            cls.append("usa-button--outline")
            role = "button-outline"
    elif variant == "warning":
        st["background-color"] = c["warn"]
        cls.append("govuk-button--warning")
        role = "button-warning"
    elif variant == "start":
        st["font-weight"] = "700"
        cls.append("govuk-button--start")
        role = "button-start"
    elif variant == "inverse":
        st.update({"background-color": c["white"], "color": c["brand"]})
        cls.append("govuk-button--inverse")
        role = "button-inverse"
    tag = "a" if variant == "start" else "button"
    return N(tag, role, cls, st, text=text, attrs={"href": "#"} if tag == "a" else {"type": "button"})


def band(th, kids, bg=None, pad=None, tag="section", tags=()):
    pad = th.space["l"] if pad is None else pad
    inner = N("div", "container", ["app-width-container"], {"width": "960px", "margin": "0"}, kids=kids)
    st = {"display": "flex", "justify-content": "center", "padding": box((pad, 0, pad, 0)), "margin": "0"}
    if bg:
        st["background-color"] = bg
    n = N(tag, "container", [], st, kids=[inner])
    n.tags |= set(tags)
    return n


# ---------------------------------------------------------------- sections
def sec_header(th, tx, knobs):
    c = th.c
    logo = T(
        th,
        "a",
        "logo",
        "logo",
        tx.service(),
        cls=["app-logo" if th.brand == "govuk" else "usa-logo"],
        color=c["white"],
        bold=True,
        family="head",
    )
    logo.attrs["href"] = "#"
    cls = ["govuk-header__link"] if th.brand == "govuk" else []
    links = [
        link(th, t, role="header-link", color=c["white"], underline=False, cls=cls, kind="hlink")
        for t in tx.r.sample(["Sign in", "Help", "Contact", "Cymraeg", "Menu", "Search"], knobs["n_links"])
    ]
    for lk in links:
        lk.style["font-weight"] = "700"
        lk.style["margin"] = box((0, 0, 0, th.space["m"]))
    nav = N("nav", "container", [], {"display": "flex", "margin": "0"}, kids=links)
    row = N(
        "div", "container", [], {"display": "flex", "justify-content": "space-between", "margin": "0"}, kids=[logo, nav]
    )
    return band(th, [row], bg=c["brand"], pad=th.space["m"], tag="header", tags=("header", "dark"))


def sec_nav(th, tx, knobs):
    cls = ["govuk-service-navigation__link"] if th.brand == "govuk" else ["usa-nav__link"]
    links = [
        link(th, t, role="nav-link", underline=False, cls=cls, kind="hlink")
        for t in tx.r.sample(
            ["Overview", "Your applications", "Payments", "Messages", "Settings", "Documents"], knobs["n_links"]
        )
    ]
    for lk in links:
        lk.style["margin"] = box((0, th.space["l"], 0, 0))
    nav = N("nav", "container", [], {"display": "flex", "margin": "0"}, kids=links)
    b = band(th, [nav], bg=th.c["white"], pad=th.space["s"], tag="nav", tags=("nav",))
    b.style["border-bottom"] = f"1px solid {th.c['border']}"
    return b


def sec_hero(th, tx, knobs, page_type):
    kids = [
        T(
            th,
            "h1",
            "h1",
            "h1-alt" if knobs.get("big") else "h1",
            tx.heading(),
            bold=True,
            family="head",
            margin=(0, 0, th.space["l"], 0),
        ),
        T(
            th,
            "p",
            "lead",
            "lead",
            tx.para(1),
            cls=["govuk-body-l" if th.brand == "govuk" else "usa-intro"],
            margin=(0, 0, th.space["l"], 0),
        ),
    ]
    if page_type == "start":
        kids.append(button(th, "Start now", "start" if th.brand == "govuk" else "default"))
    return band(
        th, kids, bg=th.c["surface"] if knobs.get("tint") else th.c["white"], pad=th.space["xl"], tags=("hero",)
    )


def sec_text(th, tx, knobs):
    kids = [
        T(
            th,
            "h2",
            "h2",
            "h2-alt" if knobs.get("small") else "h2",
            tx.heading(),
            bold=True,
            family="head",
            margin=(0, 0, th.space["m"], 0),
        )
    ]
    for _ in range(knobs["n_paras"]):
        kids.append(T(th, "p", "body", "body", tx.para(), cls=[f"{th.p}-body"], margin=(0, 0, th.space["m"], 0)))
    if knobs.get("link"):
        kids.append(
            N("p", "body", [], {"margin": box((0, 0, th.space["m"], 0))}, kids=[link(th, tx.r.choice(LINK_TEXTS))])
        )
    return band(th, kids, tags=("text",))


def sec_form(th, tx, knobs):
    kids = [T(th, "h2", "h2", "h2", tx.heading(), bold=True, family="head", margin=(0, 0, th.space["l"], 0))]
    fields = []
    for _ in range(knobs["n_fields"]):
        grp = [
            T(
                th,
                "label",
                "label",
                "label",
                f"{tx.noun().capitalize()} number",
                cls=[f"{th.p}-label"],
                bold=True,
                margin=(0, 0, th.space["xs"], 0),
            )
        ]
        if knobs.get("hints"):
            grp.append(
                T(
                    th,
                    "p",
                    "hint",
                    "hint",
                    tx.sentence(),
                    cls=[f"{th.p}-hint"],
                    color=th.c["sec"],
                    margin=(0, 0, th.space["xs"], 0),
                )
            )
        grp.append(
            N(
                "input",
                "input",
                [f"{th.p}-input"],
                {
                    "border": f"2px solid {th.c['input']}",
                    "padding": box([th.space["input"]] * 4),
                    "margin": "0",
                    "width": "300px",
                    "font-family": th.font["body"],
                    "font-size": px(th.size["body"][0]),
                    "background-color": th.c["white"],
                },
                attrs={"type": "text"},
            )
        )
        fields.append(N("div", "container", [], {"margin": box((0, 0, th.space["l"], 0))}, kids=grp))
    btns = [button(th, tx.r.choice(BUTTON_LABELS))]
    if knobs.get("secondary"):
        btns.append(button(th, tx.r.choice(SECONDARY_LABELS), "secondary"))
    if knobs.get("warning") and th.brand == "govuk":
        btns.append(button(th, tx.r.choice(["Delete account", "Remove this item", "Cancel application"]), "warning"))
    group = N("div", "container", [], {"margin": "0"}, kids=btns)
    group.tags.add("buttons")
    form = N("form", "container", [], {"margin": "0", "padding": "0"}, kids=[*fields, group])
    form.tags.add("form")
    return band(th, [*kids, form], tags=("formsec",))


def sec_cards(th, tx, knobs):
    cards = []
    if th.brand == "govuk":
        bg = th.c["alt_surface"] if knobs.get("alt") else th.c["surface"]
    else:
        bg = th.c["alt_surface"] if knobs.get("alt") else th.c["white"]
    for _ in range(knobs["n_cards"]):
        kids = [
            T(th, "h3", "h3", "h3", tx.heading(), bold=True, family="head", margin=(0, 0, th.space["s"], 0)),
            T(
                th,
                "p",
                "body-s" if knobs.get("small_text") else "body",
                "body-s" if knobs.get("small_text") else "body",
                tx.para(1),
                cls=[f"{th.p}-body-s" if knobs.get("small_text") else f"{th.p}-body"],
                margin=(0, 0, th.space["s"], 0),
            ),
            N("p", "body", [], {"margin": "0"}, kids=[link(th, tx.r.choice(LINK_TEXTS))]),
        ]
        card = N(
            "div",
            "container",
            ["app-card"],
            {
                "background-color": bg,
                "border": f"1px solid {th.c['border']}",
                "padding": box([th.space["m"]] * 4),
                "margin": "0",
                "flex": "1",
            },
            kids=kids,
        )
        card.tags.add("card")
        cards.append(card)
    row = N("div", "container", [], {"display": "flex", "gap": f"{th.space['m']}px", "margin": "0"}, kids=cards)
    h = T(th, "h2", "h2", "h2", tx.heading(), bold=True, family="head", margin=(0, 0, th.space["m"], 0))
    return band(th, [h, row], tags=("cards",))


def sec_banner(th, tx, knobs):
    c = th.c
    if th.brand == "govuk":
        kids = [
            T(
                th,
                "h2",
                "banner-title",
                "banner",
                "Important",
                cls=["govuk-notification-banner__heading"],
                color=c["white"],
                bold=True,
                margin=(0, 0, th.space["s"], 0),
            ),
            T(
                th,
                "p",
                "banner-text",
                "banner",
                tx.sentence(),
                cls=["govuk-notification-banner__text"],
                color=c["white"],
            ),
        ]
        inner = N(
            "div",
            "container",
            ["govuk-notification-banner"],
            {"background-color": c["brand"], "padding": box([th.space["m"]] * 4), "margin": "0"},
            kids=kids,
        )
        inner.tags.add("dark")
    else:
        kids = [
            T(
                th,
                "h2",
                "banner-title",
                "banner",
                "Important",
                cls=["usa-alert__heading"],
                bold=True,
                margin=(0, 0, th.space["s"], 0),
            ),
            T(th, "p", "banner-text", "banner", tx.sentence(), cls=["usa-alert__text"]),
        ]
        inner = N(
            "div",
            "container",
            ["usa-alert"],
            {
                "background-color": c["alt_surface"],
                "border-left": f"8px solid {c['alert_border']}",
                "padding": box([th.space["m"]] * 4),
                "margin": "0",
            },
            kids=kids,
        )
    return band(th, [inner], pad=th.space["m"], tags=("banner",))


def sec_panel(th, tx, knobs):
    c = th.c
    kids = [
        T(
            th,
            "h1",
            "h1",
            "h1",
            "Application complete",
            color=c["white"],
            bold=True,
            family="head",
            margin=(0, 0, th.space["m"], 0),
        ),
        T(
            th,
            "p",
            "lead",
            "lead",
            f"Your reference number is {tx.r.randint(100000, 999999)}",
            color=c["white"],
            cls=["govuk-body-l" if th.brand == "govuk" else "usa-intro"],
        ),
    ]
    return band(th, kids, bg=c["panel"], pad=th.space["xl"], tags=("panel", "dark"))


def sec_dark(th, tx, knobs):
    """A dark call-to-action band: white heading, white link, and an inverse (white) button on GOV.UK."""
    c = th.c
    kids = [
        T(
            th,
            "h2",
            "h2",
            "h2",
            tx.heading(),
            color=c["white"],
            bold=True,
            family="head",
            margin=(0, 0, th.space["m"], 0),
        ),
        N(
            "p",
            "body",
            [],
            {"margin": box((0, 0, th.space["m"], 0))},
            kids=[link(th, tx.r.choice(LINK_TEXTS), color=c["white"])],
        ),
    ]
    if th.brand == "govuk":
        kids.append(button(th, tx.r.choice(["Find out more", "Get started", "Read the guidance"]), "inverse"))
    return band(th, kids, bg=c["dark"], pad=th.space["xl"], tags=("darkband", "dark"))


def sec_links(th, tx, knobs):
    items = [
        N("li", "container", [], {"margin": box((0, 0, th.space["s"], 0))}, kids=[link(th, t)])
        for t in tx.r.sample(LINK_TEXTS, knobs["n_links"])
    ]
    ul = N("ul", "container", [], {"margin": "0", "padding": "0", "list-style": "none"}, kids=items)
    h = T(th, "h2", "h2", "h2-alt", tx.heading(), bold=True, family="head", margin=(0, 0, th.space["m"], 0))
    return band(th, [h, ul], tags=("links",))


def sec_footer(th, tx, knobs):
    links = [
        link(
            th,
            t,
            role="footer-link",
            kind="body-s",
            cls=["govuk-footer__link" if th.brand == "govuk" else "usa-footer__link"],
        )
        for t in tx.r.sample(["Help", "Privacy", "Cookies", "Accessibility statement", "Contact"], knobs["n_links"])
    ]
    for lk in links:
        lk.style["margin"] = box((0, th.space["m"], 0, 0))
    row = N("div", "container", [], {"display": "flex", "margin": box((0, 0, th.space["m"], 0))}, kids=links)
    meta = T(
        th,
        "p",
        "footer-text",
        "body-s",
        "All content is available under an open licence, except where otherwise stated",
        cls=["app-footer__text"],
    )
    return band(th, [row, meta], bg=th.c["footer"], pad=th.space["xl"], tag="footer", tags=("footer",))


SECTIONS = {
    "hero": sec_hero,
    "text": sec_text,
    "form": sec_form,
    "cards": sec_cards,
    "banner": sec_banner,
    "panel": sec_panel,
    "dark": sec_dark,
    "links": sec_links,
}
PAGE_TYPES = {
    "start": ["hero", "text", "cards", "links", "dark", "text"],
    "question": ["form", "text", "banner", "links", "cards", "dark"],
    "content": ["hero", "text", "cards", "links", "banner", "dark", "form"],
    "confirmation": ["panel", "text", "links", "banner"],
}


# ---------------------------------------------------------------- recipes
def random_knobs(r: random.Random, kind: str) -> dict:
    return {
        "hero": lambda: {"big": r.random() < 0.3, "tint": r.random() < 0.4},
        "text": lambda: {"n_paras": r.randint(1, 3), "link": r.random() < 0.8, "small": r.random() < 0.3},
        "form": lambda: {
            "n_fields": r.randint(1, 3),
            "hints": r.random() < 0.5,
            "secondary": r.random() < 0.6,
            "warning": r.random() < 0.15,
        },
        "cards": lambda: {"n_cards": r.randint(2, 4), "alt": r.random() < 0.3, "small_text": r.random() < 0.4},
        "banner": lambda: {},
        "panel": lambda: {},
        "dark": lambda: {},
        "links": lambda: {"n_links": r.randint(3, 6)},
    }[kind]()


def random_recipe(brand: str, seed: int, page_type: str | None = None, variant: str | None = None) -> dict:
    r = random.Random(seed)
    page_type = page_type or r.choice(list(PAGE_TYPES))
    if variant is None:
        if brand == "govuk":
            variant = "service" if r.random() < 0.75 else "campaign"
        else:
            variant = "sources" if r.random() < 0.7 else "public"
    pool = PAGE_TYPES[page_type]
    rest = pool[1:]
    body = [pool[0], *r.sample(rest, r.randint(2, min(4, len(rest))))]
    sections = [{"kind": s, "knobs": random_knobs(r, s), "text_seed": r.randrange(10**9)} for s in body]
    slug = Txt(r.randrange(10**9)).service().lower().replace(" ", "-")
    if brand == "govuk":
        domain = f"{slug}.service.gov.uk" if variant == "service" else f"{slug}.campaign.gov.uk"
    else:
        domain = f"{slug}.example.gov"
    return {
        "brand": brand,
        "page_type": page_type,
        "variant": variant,
        "domain": domain,
        "header": {"n_links": r.randint(2, 4), "text_seed": r.randrange(10**9)},
        "nav": {"on": r.random() < 0.4, "n_links": r.randint(3, 5), "text_seed": r.randrange(10**9)},
        "footer": {"n_links": r.randint(2, 4), "text_seed": r.randrange(10**9)},
        "sections": sections,
    }


def build(recipe: dict) -> tuple[N, Theme]:
    th = Theme(recipe["brand"], recipe["variant"])
    body = [sec_header(th, Txt(recipe["header"]["text_seed"]), recipe["header"])]
    if recipe["nav"]["on"]:
        body.append(sec_nav(th, Txt(recipe["nav"]["text_seed"]), recipe["nav"]))
    main = []
    for s in recipe["sections"]:
        fn, tx = SECTIONS[s["kind"]], Txt(s["text_seed"])
        main.append(fn(th, tx, s["knobs"], recipe["page_type"]) if s["kind"] == "hero" else fn(th, tx, s["knobs"]))
    body.append(N("main", "container", [], {"margin": "0"}, kids=main))
    body.append(sec_footer(th, Txt(recipe["footer"]["text_seed"]), recipe["footer"]))
    root = N(
        "body",
        "container",
        [],
        {
            "margin": "0",
            "padding": "0",
            "background-color": th.c["white"],
            "font-family": th.font["body"],
            "color": th.c["text"],
        },
        kids=body,
    )
    return root, th


def number(root: N) -> None:
    for i, n in enumerate(root.walk()):
        n.id = f"e{i}"


# ---------------------------------------------------------------- facts the generator meant to render
def parse_box(v: str | None) -> list[float]:
    if not v:
        return [0.0] * 4
    p = [float(x.replace("px", "")) for x in v.split()]
    if len(p) == 4:
        return p
    return [p[0], p[1], p[0], p[1]] if len(p) == 2 else [p[0]] * 4


def border_colour(st: dict) -> str | None:
    for k in ("border", "border-left", "border-bottom", "border-top", "border-right"):
        v = st.get(k)
        if v and v != "0":
            return v.split()[-1]
    return None


def intended(root: N) -> list[dict]:
    """What the extractor should read back, element by element (reference labels by construction)."""
    out = []

    def rec(n: N, parent_bg: str, region: str):
        own = n.style.get("background-color")
        eff = own or parent_bg
        if n.tag in ("header", "footer") or (n.tag == "nav" and region == "main"):
            region = n.tag
        if n.tag != "body":
            st = n.style
            out.append(
                {
                    "id": n.id,
                    "role": n.role,
                    "tag": n.tag,
                    "text": n.text or "",
                    "color": st.get("color"),
                    "bg": own,
                    "eff_bg": eff,
                    "parent_bg": parent_bg,
                    "border": border_colour(st),
                    "font": st["font-family"].split(",")[0].strip().strip('"') if "font-family" in st else None,
                    "size": float(st["font-size"].replace("px", "")) if "font-size" in st else None,
                    "weight": int(st.get("font-weight", "400")),
                    "line_height": float(st["line-height"].replace("px", "")) if "line-height" in st else None,
                    "margin": parse_box(st.get("margin")),
                    "padding": parse_box(st.get("padding")),
                    "underline": st.get("text-decoration") == "underline",
                    "disabled": "disabled" in n.attrs,
                    "region": region,
                }
            )
        for k in n.kids:
            rec(k, eff, region)

    rec(root, "#ffffff", "main")
    return out


# ---------------------------------------------------------------- HTML
RESET = (
    "*{margin:0;padding:0;box-sizing:border-box}html{background:#ffffff}"
    "button,input{font:inherit;color:inherit}a{text-decoration:none}"
)


def to_html(root: N, title: str) -> str:
    def render(n: N) -> str:
        attrs = {"data-id": n.id, **n.attrs}
        if n.cls:
            attrs["class"] = " ".join(n.cls)
        if n.style:
            attrs["style"] = ";".join(f"{k}:{v}" for k, v in n.style.items())
        a = "".join(f" {k}" if v is True else f' {k}="{html.escape(str(v), quote=True)}"' for k, v in attrs.items())
        if n.tag == "input":
            return f"<input{a}>"
        inner = html.escape(n.text) if n.text else ""
        return f"<{n.tag}{a}>{inner}{''.join(render(k) for k in n.kids)}</{n.tag}>"

    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{RESET}</style></head>{render(root)}</html>\n"
    )


# ---------------------------------------------------------------- violations and traps
def nodes(root: N, pred) -> list[N]:
    return [n for n in root.walk() if pred(n) and not n.planted]


def under(root: N, tag: str) -> set[int]:
    out: set[int] = set()
    for n in root.walk():
        if tag in n.tags:
            out |= {id(k) for k in n.walk()}
    return out


def light(root: N, roles) -> list[N]:
    dark = under(root, "dark")
    return nodes(root, lambda n: n.role in roles and id(n) not in dark and bool(n.text))


def eff_bg(root: N, target: N) -> str:
    def rec(n, bg):
        bg = n.style.get("background-color", bg)
        if n is target:
            return bg
        for k in n.kids:
            got = rec(k, bg)
            if got:
                return got
        return None

    return rec(root, "#ffffff")


def on_white(root: N, roles) -> list[N]:
    return [n for n in light(root, roles) if eff_bg(root, n) == "#ffffff"]


def pick(r, cands, **style):
    if not cands:
        return None
    n = r.choice(cands)
    for k, v in style.items():
        n.style[k.replace("_", "-")] = v
    return n


def retext(r, cands, fn, need=lambda t: True):
    cands = [c for c in cands if c.text and need(c.text)]
    if not cands:
        return None
    n = r.choice(cands)
    new = fn(n.text)
    if new == n.text:
        return None
    n.text = new
    return n


def respace(root, r, bad):
    cands = nodes(root, lambda n: n.tag in ("p", "h2", "h3") and bool(n.text) and n.style.get("margin", "0") != "0")
    cands += nodes(root, lambda n: "card" in n.tags)
    if not cands:
        return None
    n = r.choice(cands)
    if "card" in n.tags:
        n.style["padding"] = box([r.choice(bad)] * 4)
    else:
        m = [int(x) for x in parse_box(n.style["margin"])]
        m[2] = r.choice(bad)
        n.style["margin"] = box(m)
    return n


def form_default(root):
    for f in nodes(root, lambda n: n.tag == "form"):
        for n in f.walk():
            if n.role == "button-default" and not n.planted:
                return n
    return None


def secondary_in_group(root):
    for g in nodes(root, lambda n: "buttons" in n.tags):
        for n in g.kids:
            if n.role == "button-secondary" and not n.planted:
                return n
    return None


def second_default(root, th):
    n = secondary_in_group(root)
    if n is None:
        return None
    n.role = "button-default"
    n.cls = [c for c in n.cls if not c.endswith("--secondary")]
    n.style.update({"background-color": th.c["btn"], "color": th.c["white"]})
    return n


def to_start(root, th):
    n = form_default(root)
    if n is None:
        return None
    n.role, n.tag, n.attrs = "button-start", "a", {"href": "#"}
    n.cls.append("govuk-button--start")
    n.style["font-weight"] = "700"
    return n


def misalign(root, th):
    n = form_default(root)
    if n is None:
        return None
    n.style["margin"] = box((0, th.space["gap"], 0, 20))
    return n


def bad_warning(root, r, th):
    n = secondary_in_group(root)
    if n is None:
        return None
    n.role = "button-warning"
    n.cls = [c for c in n.cls if not c.endswith("--secondary")] + ["govuk-button--warning"]
    n.style.update({"background-color": th.c["warn"], "color": th.c["white"]})
    n.text = r.choice(["Continue", "Save changes", "Send request"])
    return n


def not_inverse(root, th):
    for n in nodes(root, lambda n: n.role == "button-inverse"):
        n.role = "button-secondary"
        n.cls = [c for c in n.cls if not c.endswith("--inverse")] + ["govuk-button--secondary"]
        n.style.update({"background-color": th.c["btn2"], "color": th.c["text"]})
        return n
    return None


def links_in(root, tag):
    return [k for n in root.walk() if tag in n.tags for k in n.walk() if k.role == "link" and not k.planted]


VIOLATIONS = {
    "govuk": {
        "offpal-text": ("GOV-PAL", "text", lambda R, r, th: pick(r, light(R, {"body"}), color="#1b1b1b")),
        "offpal-bg": (
            "GOV-PAL",
            "text",
            lambda R, r, th: pick(r, nodes(R, lambda n: "card" in n.tags), background_color="#f5f5f5"),
        ),
        "contrast": ("GOV-CON", "text", lambda R, r, th: pick(r, light(R, {"body"}), color="#858686")),
        "contrast-large": ("GOV-CON-LARGE", "decided", lambda R, r, th: pick(r, light(R, {"h2"}), color="#8eb8dc")),
        "font": (
            "GOV-FONT",
            "decided",
            lambda R, r, th: (
                pick(r, light(R, {"h2", "body"}), font_family="Helvetica, arial, sans-serif")
                if th.variant == "service"
                else None
            ),
        ),
        "font-alt": (
            "GOV-FONT-ALT",
            "text",
            lambda R, r, th: (
                pick(r, light(R, {"h2", "body"}), font_family='"GDS Transport", arial, sans-serif')
                if th.variant == "campaign"
                else None
            ),
        ),
        "type": ("GOV-TYPE", "text", lambda R, r, th: pick(r, light(R, {"body"}), font_size="18px")),
        "lh": ("GOV-LH", "text", lambda R, r, th: pick(r, light(R, {"h2"}), line_height="45px")),
        "space": ("GOV-SPACE", "decided", lambda R, r, th: respace(R, r, [12, 22, 35, 45])),
        "link-col": ("GOV-LINK-COL", "text", lambda R, r, th: pick(r, light(R, {"link"}), color="#1d70b8")),
        "link-ul": ("GOV-LINK-UL", "text", lambda R, r, th: pick(r, light(R, {"link"}), text_decoration="none")),
        "link-stop": ("GOV-LINK-STOP", "text", lambda R, r, th: retext(r, light(R, {"link"}), lambda t: t + ".")),
        "link-inv": ("GOV-LINK-INV", "decided", lambda R, r, th: pick(r, links_in(R, "darkband"), color="#d2e2f1")),
        "btn-case": (
            "GOV-BTN-CASE",
            "decided",
            lambda R, r, th: retext(
                r,
                nodes(R, lambda n: n.role in ("button-default", "button-secondary")),
                str.title,
                need=lambda t: len(t.split()) >= 2,
            ),
        ),
        "btn-two": ("GOV-BTN-ONE", "decided", lambda R, r, th: second_default(R, th)),
        "btn-start": ("GOV-BTN-START", "decided", lambda R, r, th: to_start(R, th)),
        "btn-align": ("GOV-BTN-ALIGN", "text", lambda R, r, th: misalign(R, th)),
        "btn-warn": ("GOV-BTN-WARN", "decided", lambda R, r, th: bad_warning(R, r, th)),
        "btn-inv": ("GOV-BTN-INV", "decided", lambda R, r, th: not_inverse(R, th)),
        "focus": (
            "GOV-FOCUS",
            "decided",
            lambda R, r, th: pick(r, nodes(R, lambda n: "card" in n.tags), background_color="#ffdd00"),
        ),
    },
    "uswds": {
        "offpal-text": ("USW-PAL", "decided", lambda R, r, th: pick(r, light(R, {"body"}), color="#1a1a1a")),
        "offpal-bg": (
            "USW-PAL",
            "decided",
            lambda R, r, th: pick(r, nodes(R, lambda n: "card" in n.tags), background_color="#f2f2f2"),
        ),
        "contrast": ("USW-CON", "text", lambda R, r, th: pick(r, light(R, {"body"}), color="#8d9297")),
        "contrast-large": ("USW-CON-LARGE", "text", lambda R, r, th: pick(r, light(R, {"h2"}), color="#a9aeb1")),
        "black": (
            "USW-BLACK",
            "decided",
            lambda R, r, th: pick(r, [n for n in light(R, {"body"}) if eff_bg(R, n) == "#ffffff"], color="#000000"),
        ),
        "space": ("USW-SPACE", "text", lambda R, r, th: respace(R, r, [10, 14, 30, 36])),
        "size": (
            "USW-SIZE",
            "decided",
            lambda R, r, th: pick(r, light(R, {"body"}), font_size="20px" if th.variant == "sources" else "19px"),
        ),
        "family": ("USW-FAMILY", "text", lambda R, r, th: pick(r, light(R, {"h2"}), font_family="Lato, sans-serif")),
        "btn-case": (
            "USW-BTN-CASE",
            "decided",
            lambda R, r, th: retext(
                r,
                nodes(R, lambda n: n.role in ("button-default", "button-outline")),
                str.title,
                need=lambda t: len(t.split()) >= 2 and t.split()[0].lower() in USWDS_VERBS,
            ),
        ),
        "btn-verb": (
            "USW-BTN-VERB",
            "decided",
            lambda R, r, th: retext(
                r,
                nodes(R, lambda n: n.role in ("button-default", "button-outline")),
                lambda t: r.choice(["Complaint filing", "Application status", "Next step", "Payment options"]),
            ),
        ),
    },
}


def disable(root, th):
    n = form_default(root)
    if n is None:
        return False
    n.attrs["disabled"] = True
    n.style.update({"background-color": "#87bca8", "color": th.c["white"]})
    return True


TRAPS = {
    "govuk": {
        "disabled": lambda R, r, th: disable(R, th),
        "large": lambda R, r, th: bool(pick(r, on_white(R, {"h1"}), color="#5694ca")),
        "bold19": lambda R, r, th: bool(pick(r, on_white(R, {"label"}), color="#858686")),
        "logo": lambda R, r, th: bool(pick(r, nodes(R, lambda n: n.role == "logo"), color="#5694ca")),
        "h80": lambda R, r, th: bool(pick(r, light(R, {"h1"}), font_size="80px", line_height="80px")),
    },
    "uswds": {
        "bold19": lambda R, r, th: (
            th.variant == "sources" and bool(pick(r, on_white(R, {"label"}), color="#8d9297", font_size="19.08px"))
        ),
        "large": lambda R, r, th: bool(pick(r, on_white(R, {"h1"}), color="#8d9297")),
        "grade50": lambda R, r, th: bool(pick(r, on_white(R, {"body"}), color="#d83933")),
    },
}


# ---------------------------------------------------------------- one page
def make_page(pid: str, recipe: dict, violations=(), traps=(), seed: int = 0) -> dict:
    """Render a recipe, plant violations and traps; returns {id, recipe, html, intended, labels, planted, traps}."""
    root, th = build(recipe)
    r = random.Random(seed)
    planted, applied = [], []
    for name in violations:
        rule, settled, fn = VIOLATIONS[recipe["brand"]][name]
        node = fn(root, r, th)
        if node is None:
            continue
        node.planted = True
        planted.append((name, rule, settled, node))
    for name in traps:
        if TRAPS[recipe["brand"]][name](root, r, th):
            applied.append(name)
    number(root)
    return {
        "id": pid,
        "brand": recipe["brand"],
        "domain": recipe["domain"],
        "page_type": recipe["page_type"],
        "recipe": copy.deepcopy(recipe),
        "html": to_html(root, recipe["domain"]),
        "intended": intended(root),
        "labels": sorted([n.id, rule] for _, rule, _, n in planted),
        "planted": [{"name": nm, "rule": rule, "settled": st, "element": n.id} for nm, rule, st, n in planted],
        "traps": applied,
    }
