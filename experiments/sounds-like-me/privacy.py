"""Shared redaction for the Sounds Like Me pages that show draft text (Episodes 1 to 4).

Drafts are written from a real writer's samples (the public Enron and Blog corpora), and models copy what is in them:
names, signatures, phone numbers, extensions, emails and office addresses. Each episode's export redacts the writer's
own name parts with its own rule, then runs the text through `redact_contact`, and also redacts the names a reviewer
found copied from the samples (`copied_names`).

The hand-kept name lists live in data/copied_names.json, which stays in the lab (the list is itself personal data).
Without the file (the public repo) `copied_names` returns [] and only the other rules run. The per-draft files the
pages are built from (drafts, honesty replies, labelling sheets) stay in the lab for the same reason; the public
folders carry only redacted excerpts and aggregate numbers.
"""

import json
import re
from pathlib import Path

COPIED_NAMES = Path(__file__).parent / "data" / "copied_names.json"

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(
    r"\(\s*\d{3}\s*\)\s*\d{3}\s*[-.\s]\s*\d{4}"  # (555) 010-0123
    r"|\b\d{3}[-.]\d{3}[-.]\d{4}\b"  # 555-010-0123, 555.010.0123
    r"|\b\d{3} \d{3} \d{4}\b"  # 555 010 0123
    r"|\+\d{1,3}[\s-]\d[\d\s-]{7,}\d"  # +44 20 ...
    r"|\b[2-9]\d{2}-\d{4}\b"  # 555-0123
    r"|\b(?:x|ext\.?)\s?\d{4,}\b",  # x36226, ext. 3622
    re.IGNORECASE,
)
SUFFIX = r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|way|court|ct|place|pl|parkway|pkwy)"
STATES = (
    "AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA"
    "|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|Texas|California|Oregon"
)
ADDRESS = re.compile(
    rf"\b\d{{1,5}}\s+(?:[nsew]\.?\s+)?(?:[a-z0-9]+\.?\s+){{1,3}}{SUFFIX}\b\.?"  # 12 Main Street, 5 23rd street
    r"|\bp\.?\s?o\.?\s+box\s+\d+"  # PO Box 123
    r"|\b(?:suite|ste\.?|floor|room|rm\.?)\s+#?\d{2,5}\b"  # Suite 300
    r"|\beb\s?\d{3,4}[a-z]?\b"  # Enron Building room, EB 1234
    rf"|\b[a-z]+(?: [a-z]+)?,?\s+(?:{STATES})\.?\s+\d{{5}}(?:-\d{{4}})?\b",  # Springfield, IL 62701
    re.IGNORECASE,
)
NOTES = re.compile(
    r"\b[\w.-]+(?: \w\.?)? [\w.-]+/[A-Z]{2,}(?:/[A-Z]{2,})*@[A-Z]+\b"
)  # Lotus Notes: First Last/OU/OU@OU
CODE = re.compile(r"\b(?:cc|co|rc)\s*[:#]\s*\d{5,}\b", re.IGNORECASE)  # cost-centre and company codes


def copied_names(episode):
    """Hand-found names copied from the samples for one episode's displayed drafts; [] without the lab-only list."""
    return json.loads(COPIED_NAMES.read_text()).get(episode, []) if COPIED_NAMES.exists() else []


def redact_contact(text):
    """Emails, phone numbers, extensions, street addresses and office codes, whoever they belong to."""
    text = NOTES.sub("[name]", text)
    text = EMAIL.sub("[email]", text)
    text = CODE.sub("[code]", text)
    text = PHONE.sub("[phone]", text)
    return ADDRESS.sub("[address]", text)
