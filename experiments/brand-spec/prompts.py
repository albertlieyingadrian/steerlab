"""The prompt for every task and arm (frozen before the held-out runs; dev tries are logged in results.tsv).

from prompts import messages
messages("t1", "b", item)   # task t1 (audit), t2 (identify), t3 / t3c (apply); arm a (no rules) or b (rules)
"""

from __future__ import annotations

from functools import cache

from common import load_spec

NAMES = {"govuk": "the GOV.UK Design System", "uswds": "the U.S. Web Design System (USWDS)"}
TITLES = {
    "govuk": "GOV.UK Design System guidance, with the WCAG 2.2 lines it points to",
    "uswds": "U.S. Web Design System guidance",
}
SYSTEM = "You review web pages against a brand book: a design system's published guidance."
FACTS = (
    "Treat the listed facts as exact: colours, sizes and spacing are the computed values a browser rendered, "
    "and 'on' gives the colour behind the text. Because they are rendered values, every colour appears as a hex "
    "value and every size in px whatever the source code used, so a line about how the code is written cannot "
    "be judged from these facts."
)


@cache
def book(brand: str) -> str:
    return "\n".join(f"{s['id']} [{s['page']}] {s['text']}" for s in load_spec(brand)["source"])


def token_note(item: dict) -> str:
    t = item.get("tag")
    if item["brand"] != "uswds" or not t or t.get("ga") is None:
        return ""
    return (
        f" | text colour is token {t['a']} (grade {t['ga']}); the colour behind it is token {t['b']} (grade {t['gb']})"
    )


def messages(task: str, arm: str, item: dict) -> list[dict]:
    brand = item["brand"]
    if task == "t1" and arm == "a":
        user = (
            f"This page should follow {NAMES[brand]}.\n\n{item['page']}\n\n"
            f"Task: using what you know of that design system's published guidance, find every element on this "
            f"page that breaks it. {FACTS} For each problem give the element id and a short reason. If the page "
            "breaks nothing, the answer is an empty list.\n\n"
            "End your reply with the list as JSON in a code block, for example:\n```json\n"
            '[{"element": "e12", "reason": "button text is not in sentence case"}]\n```'
        )
    elif task == "t1":
        example = "G71" if brand == "govuk" else "U58"
        user = (
            f"Brand book: {TITLES[brand]}. Each line starts with its id.\n\n{book(brand)}\n\n{item['page']}\n\n"
            f"Task: find every element on this page that breaks a line of the brand book. {FACTS} For each "
            "problem give the element id and the id of the one line it breaks. If the page breaks no line, the "
            "answer is an empty list.\n\n"
            "End your reply with the list as JSON in a code block, for example:\n```json\n"
            f'[{{"element": "e12", "line": "{example}"}}]\n```'
        )
    elif task == "t2":
        user = (
            f"Brand book: {TITLES[brand]}. Each line starts with its id.\n\n{book(brand)}\n\n{item['page']}\n\n"
            f"Question: which one line of the brand book decides whether {item['question']} complies? It does "
            f"not matter whether it complies; name the line that settles it. {FACTS}\n\n"
            "End your reply with the line id, as: Line: <id>"
        )
    else:
        quote = "\n".join(item["quote"])
        dom = f"The page's domain is {item['domain']}.\n" if item["kind"] == "typeface" else ""
        user = (
            f"Lines from the brand book ({TITLES[brand]}):\n{quote}\n\n{dom}Element: {item['facts']}"
            f"{token_note(item)}\n\nQuestion: does this element comply with the lines above? {FACTS}\n\n"
            "End your reply with: Answer: yes, or Answer: no"
        )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
