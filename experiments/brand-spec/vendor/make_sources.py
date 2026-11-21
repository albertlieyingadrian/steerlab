"""Build the two vendored brand books (`govuk/source.tsv`, `uswds/source.tsv`) from pinned upstream clones.

    .venv/bin/python experiments/brand-spec/vendor/make_sources.py GOVUK_CLONE USWDS_SITE_CLONE USWDS_CLONE

Each output line is one sentence or one table row of the published guidance, rendered as plain text (markup,
template calls and example blocks dropped), with the upstream file and line it came from. The line id (G12, W3, U40)
is what every rule in `spec/` cites and what the models see. Prose lines are listed by hand below (quoted verbatim,
except that inline code markup and link targets are dropped); token tables are read from the upstream data files.
Upstream commits: alphagov/govuk-design-system 88066fd8, uswds/uswds-site af13d80f, uswds/uswds 029d6d79.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent

C, TF, TS, SP, LK, BT = (
    "src/styles/colour/index.md",
    "src/styles/typeface/index.md",
    "src/styles/type-scale/index.md",
    "src/styles/spacing/index.md",
    "src/styles/links/index.md",
    "src/components/button/index.md",
)
WCAG = "https://www.w3.org/TR/WCAG22/"

GOVUK_HEAD = [
    ("Colour", "Always use the GOV.UK colour palette.", f"{C}:15"),
    (
        "Colour",
        "You must make sure that the contrast ratio of text and interactive elements in your service meets Web "
        "Content Accessibility Guidelines (WCAG 2.2) success criterion 1.4.3 Contrast (minimum) level AA.",
        f"{C}:19",
    ),
    ("Colour", "The WCAG 2.2 criterion for Contrast (minimum) is the same as WCAG 2.1.", f"{C}:22"),
    (
        "Colour",
        "The GOV.UK Design System has a set of functional colours for essential page elements. These help apply "
        "colours across the Design System in a way that makes interactions predictable and consistent to users.",
        f"{C}:27",
    ),
    (
        "Colour",
        "Do not copy the specific hexadecimal (hex) colour values. For example, use "
        'govuk-functional-colour("brand") rather than #1d70b8.',
        f"{C}:33",
    ),
    (
        "Colour",
        "Only use the variables in the context they're designed for. In all other cases, you should reference the "
        'GOV.UK web palette directly. For example, if you wanted to use red, you should use govuk-colour("red") '
        'rather than govuk-functional-colour("error").',
        f"{C}:37",
    ),
]
GOVUK_MID = [
    (
        "Colour",
        "The GOV.UK web palette works as part of the GOV.UK brand guidelines. Use these colours for supporting "
        "elements in your service like illustrations, or in custom components where appropriate.",
        f"{C}:47",
    ),
    (
        "Colour",
        "The web palette is organised into colour groups. Each colour group includes a 'primary' variant, tints "
        "(lighter variants of each colour) and shades (darker variants of each colour).",
        f"{C}:49-53",
    ),
]
GOVUK_TAIL = [
    ("Colour", "Black includes a 'primary' black, with tints to show greys. White has no variants.", f"{C}:73"),
    (
        "Colour",
        "You can find brand colours for government departments and organisations in the _colours-organisations file "
        "in GOV.UK Frontend.",
        f"{C}:82",
    ),
    ("Typeface", "If your service is on the service.gov.uk subdomain you must use the GDS Transport font.", f"{TF}:11"),
    (
        "Typeface",
        "If your service is publicly available on a subdomain other than service.gov.uk, use an alternative typeface.",
        f"{TF}:15",
    ),
    (
        "Type scale",
        "The type scale is a collection of font sizes and line heights that underpin all of the typographic styles on"
        " GOV.UK.",
        f"{TS}:17",
    ),
    (
        "Type scale",
        "When creating new components, always start by using the existing typography styles. If you need to create a "
        "new style, align it with one of the points on the type scale.",
        f"{TS}:19",
    ),
    (
        "Type scale",
        "Like the spacing scale, every point on the type scale uses a line height in a multiple of 5px. This creates "
        "a consistent vertical rhythm, which makes pages easier to scan and read.",
        f"{TS}:23",
    ),
    (
        "Type scale",
        "The scale for 'large screens' is used when the screen is wider than the tablet breakpoint (640px).",
        f"{TS}:31",
    ),
    (
        "Type scale",
        "Large screens, point 80: only used in exceptional circumstances; font size 80px; line height 80px.",
        f"{TS}:53-68",
    ),
    ("Type scale", "Large screens, point 48: govuk-heading-xl; font size 48px; line height 50px.", f"{TS}:69-84"),
    ("Type scale", "Large screens, point 36: govuk-heading-l; font size 36px; line height 40px.", f"{TS}:85-100"),
    (
        "Type scale",
        "Large screens, point 27: only used in exceptional circumstances; font size 27px; line height 30px.",
        f"{TS}:101-116",
    ),
    (
        "Type scale",
        "Large screens, point 24: govuk-heading-m, govuk-body-l; font size 24px; line height 30px.",
        f"{TS}:117-132",
    ),
    (
        "Type scale",
        "Large screens, point 19: govuk-heading-s, govuk-body; font size 19px; line height 25px.",
        f"{TS}:133-148",
    ),
    ("Type scale", "Large screens, point 16: govuk-body-s; font size 16px; line height 20px.", f"{TS}:149-164"),
    (
        "Type scale",
        "Do not use govuk-font if you only want to change the font size and line height as it includes additional "
        "typography-related CSS like the New Transport font family.",
        f"{TS}:321",
    ),
    ("Spacing", "The Design System uses two different spacing scales – responsive and static.", f"{SP}:15"),
    (
        "Spacing",
        "The spacing for 'large screens' is used when the screen is wider than the tablet breakpoint (640px). Spacing"
        " for the smallest units (0-3) stays the same for all screen sizes.",
        f"{SP}:21",
    ),
    (
        "Spacing",
        "Responsive spacing, large screens: unit 0 = 0, unit 1 = 5px, unit 2 = 10px, unit 3 = 15px, unit 4 = 20px, "
        "unit 5 = 25px, unit 6 = 30px, unit 7 = 40px, unit 8 = 50px, unit 9 = 60px.",
        f"{SP}:23-84",
    ),
    (
        "Spacing",
        "The static spacing scale stays the same for all screen sizes, and uses the same spacing as 'large screens' "
        "in the responsive spacing scale.",
        f"{SP}:88",
    ),
    ("Spacing", "If you want to reference the spacing scale in your CSS, use the spacing helpers.", f"{SP}:144"),
    (
        "Spacing",
        "Occasionally, you might need to make minor adjustments like adding or removing spacing to elements of your "
        "design. You can use the spacing override classes for this.",
        f"{SP}:187",
    ),
    (
        "Links",
        "Links are blue and underlined by default. If your link is at the end of a sentence or paragraph, make sure "
        "that the linked text does not include the full stop.",
        f"{LK}:14",
    ),
    (
        "Links",
        "If it's an external link to a non-government website, make that clear in the link text. Do not use an "
        "external link icon.",
        f"{LK}:26",
    ),
    ("Links", "Avoid opening links in a new tab or window.", f"{LK}:30"),
    (
        "Links",
        "Use the govuk-link--inverse modifier class to show white links on dark backgrounds. For example, in headers,"
        " custom components, and patterns with darker backgrounds.",
        f"{LK}:45",
    ),
    (
        "Links",
        "Make sure all users can see the links. The white links and background colour must have a contrast ratio of "
        "at least 4.5:1 to meet WCAG 2.2 success criterion 1.4.3 Contrast (minimum), level AA.",
        f"{LK}:47",
    ),
    ("Links", "Use the govuk-link--no-underline modifier class to remove underlines from links.", f"{LK}:53"),
    (
        "Links",
        "Only do this if the context tells the user that the text is a link, even without the underline.",
        f"{LK}:55",
    ),
    (
        "Links",
        "For example, links in a header or side navigation might not need underlines. Users will understand that "
        "they're links because of where they are on the page.",
        f"{LK}:57",
    ),
    (
        "Button",
        "Use the button component to help users carry out an action like starting an application or saving their "
        "information.",
        f"{BT}:16",
    ),
    ("Button", "Write button text in sentence case, describing the action it performs.", f"{BT}:20"),
    ("Button", "Align the primary action button to the left edge of your form.", f"{BT}:35"),
    ("Button", "Use a default button for the main call to action on a page.", f"{BT}:41"),
    (
        "Button",
        "Avoid using multiple default buttons on a single page. Having more than one main call to action reduces "
        "their impact, and makes it harder for users to know what to do next.",
        f"{BT}:43",
    ),
    (
        "Button",
        "Use a start button for the main call to action on your service's Start page. Start buttons do not usually "
        "submit form data, so use a link tag instead of a button tag.",
        f"{BT}:49-50",
    ),
    ("Button", "Use secondary buttons for secondary calls to action on a page.", f"{BT}:56"),
    (
        "Button",
        "Pages with too many calls to action make it hard for users to know what to do next. Before adding lots of "
        "secondary buttons, try to simplify the page or break the content down across multiple pages.",
        f"{BT}:58",
    ),
    (
        "Button",
        "Warning buttons are designed to make users think carefully before they use them. They only work if used very"
        " sparingly. Most services should not need one.",
        f"{BT}:66",
    ),
    (
        "Button",
        "Only use warning buttons for actions with serious destructive consequences that cannot be easily undone by a"
        " user. For example, permanently deleting an account.",
        f"{BT}:70",
    ),
    (
        "Button",
        "Do not only rely on the red colour of a warning button to communicate the serious nature of the action. Make"
        " sure the context and button text make clear what will happen if the user selects it.",
        f"{BT}:76",
    ),
    (
        "Button",
        "Use the govuk-button--inverse modifier class to show white buttons on dark backgrounds – for example, in "
        "headers, custom components, and patterns with darker backgrounds.",
        f"{BT}:80",
    ),
    (
        "Button",
        "Make sure all users can see the button – the white button and background colour must have a contrast ratio "
        "of at least 4.5:1 to meet WCAG 2.2 success criterion 1.4.3 Contrast (minimum), level AA.",
        f"{BT}:82",
    ),
    (
        "Button",
        "Disabled buttons have poor contrast and can confuse some users, so avoid them if possible.",
        f"{BT}:88",
    ),
    (
        "Button",
        "Only use disabled buttons if research shows it makes the user interface easier to understand.",
        f"{BT}:90",
    ),
    ("Button", "Use a button group when two or more buttons are placed together.", f"{BT}:96"),
]
WCAG_LINES = [
    (
        "WCAG 2.2",
        "Success Criterion 1.4.3 Contrast (Minimum) (Level AA): The visual presentation of text and images of text "
        "has a contrast ratio of at least 4.5:1, except for the following:",
        f"{WCAG}#contrast-minimum",
    ),
    (
        "WCAG 2.2",
        "Large Text: Large-scale text and images of large-scale text have a contrast ratio of at least 3:1;",
        f"{WCAG}#contrast-minimum",
    ),
    (
        "WCAG 2.2",
        "Incidental: Text or images of text that are part of an inactive user interface component, that are pure "
        "decoration, that are not visible to anyone, or that are part of a picture that contains significant other "
        "visual content, have no contrast requirement.",
        f"{WCAG}#contrast-minimum",
    ),
    (
        "WCAG 2.2",
        "Logotypes: Text that is part of a logo or brand name has no contrast requirement.",
        f"{WCAG}#contrast-minimum",
    ),
    (
        "WCAG 2.2",
        "large scale (text): with at least 18 point or 14 point bold or font size that would yield equivalent size "
        "for Chinese, Japanese and Korean (CJK) fonts",
        f"{WCAG}#dfn-large-scale",
    ),
    (
        "WCAG 2.2",
        "contrast ratio: (L1 + 0.05) / (L2 + 0.05), where L1 is the relative luminance of the lighter of the colors, "
        "and L2 is the relative luminance of the darker of the colors.",
        f"{WCAG}#dfn-contrast-ratio",
    ),
    (
        "WCAG 2.2",
        "relative luminance: L = 0.2126 * R + 0.7152 * G + 0.0722 * B where R, G and B are defined as: if RsRGB <= "
        "0.04045 then R = RsRGB/12.92 else R = ((RsRGB+0.055)/1.055) ^ 2.4 (and the same for G and B), with RsRGB = "
        "R8bit/255.",
        f"{WCAG}#dfn-relative-luminance",
    ),
]

OV, TT, SU, FS, FF, TO, BU = (
    "pages/design-tokens/color/overview.md",
    "pages/design-tokens/color/theme-tokens.md",
    "pages/design-tokens/spacing-units.md",
    "pages/design-tokens/typesetting/font-size.md",
    "pages/design-tokens/typesetting/font-family.md",
    "pages/design-tokens/typesetting/overview.md",
    "_components/button/guidance",
)
USWDS_HEAD = [
    ("Using color", "Use USWDS color tokens, and avoid custom colors whenever possible.", f"{OV}:35"),
    (
        "Using color",
        "Grade is a way to express how light or dark a color is. USWDS uses a 100-point scale to communicate a color "
        "token's grade, where 0 is pure white and 100 is pure black.",
        f"{OV}:50",
    ),
    (
        "Using color",
        "We've regularized these grades across color families; a color of grade 50 in one color family should be the "
        "same level of lightness as a color of grade 50 in another color family.",
        f"{OV}:50",
    ),
    (
        "Using color",
        "Section 508, which aligns with WCAG 2.0 Level AA, sets the legal standard for the needed contrast level "
        "between text and its background.",
        f"{OV}:59",
    ),
    (
        "Using color",
        "The baseline AA contrast standard is 4.5:1 for most text and 3:1 for large text (19px+ bold or 24px+ normal "
        "text).",
        f"{OV}:59",
    ),
    (
        "Using color",
        "Each color family has 10 grades, which fall between 5 and 90. Pure white is the equivalent of grade 0, and "
        "pure black is the equivalent of grade 100.",
        f"{OV}:113",
    ),
    ("Using color", "We call the difference in grade between any two colors the magic number.", f"{OV}:116"),
    (
        "Using color",
        "A magic number of 40+ results in WCAG 2.0 AA Large Text contrast (example: gray-90 and indigo-warm-50v).",
        f"{OV}:117",
    ),
    (
        "Using color",
        "A magic number of 50+ results in WCAG 2.0 AA contrast or AAA Large Text contrast (example: gray-90 and "
        "red-40).",
        f"{OV}:118",
    ),
    (
        "Using color",
        "A magic number of 70+ results in WCAG 2.0 AAA contrast (example: gray-10 and red-80).",
        f"{OV}:119",
    ),
    (
        "Using color",
        "Colors of grade 50 result in Section 508 AA contrast against both pure white (grade 0) and pure black (grade"
        " 100).",
        f"{OV}:120",
    ),
    (
        "Using color",
        "Use magic numbers to choose accessible color combinations from any palette and color family.",
        f"{OV}:122",
    ),
    (
        "Using color",
        "Avoiding pure black text on a white background helps dyslexia, Irlen Syndrome, light sensitivity, and autism.",
        f"{OV}:159",
    ),
    ("Using color", "Don't use color exclusively to convey meaning.", f"{OV}:190"),
    (
        "Theme color tokens",
        "Each theme token is drawn from a system color token and we only use token-based colors in official "
        "components.",
        f"{TT}:30",
    ),
    (
        "Theme color tokens",
        "Base is a project's neutral color, typically some tint of gray, and usually used as the text color "
        "throughout.",
        f"{TT}:36",
    ),
    (
        "Theme color tokens",
        "Primary, secondary, and accent colors can be thought of as falling into a proportional 60/30/10 "
        "relationship: about 60% of your site's color would be the primary color family, about 30% would be the "
        "secondary color family, and about 10% would be the accent color families.",
        f"{TT}:38",
    ),
]
USWDS_MID = [
    (
        "Spacing units",
        "Any spacing value in utilities or component CSS should use the following spacing unit tokens, based on "
        "multiples of 8px.",
        f"{SU}:31",
    ),
]
USWDS_TYPE = [
    ("Font size", "The USWDS typescale includes nine regularized theme tokens drawn from 21 system tokens.", f"{FS}:6"),
    ("Font size", "If your project requires more than nine sizes, use system size tokens as needed.", f"{FS}:23"),
    (
        "Using type",
        "Typefaces vary in optical size. At any specific pixel value, an optically small typeface like Source Sans "
        "Pro will appear smaller than an optically large typeface like Merriweather.",
        f"{TO}:25",
    ),
    (
        "Using type",
        "To make different typefaces appear the same size (here called the target size) at each step of the scale, "
        "the absolute size of each token's output varies depending on the font family.",
        f"{TO}:41",
    ),
]
USWDS_BUTTON = [
    (
        "Button",
        "Use buttons for the most important actions you want users to take on your site, such as Download, Sign up or"
        " Log out.",
        f"{BU}/when-to-use.md:1",
    ),
    ("Button", "Use standard buttons for actions that go a next step.", f"{BU}/usability.md:1"),
    ("Button", "Use outline buttons for actions that happen on the current page.", f"{BU}/usability.md:2"),
    (
        "Button",
        "Give an important action a distinctive style. Style the button most users should select in a way that "
        "distinguishes it from other buttons on the page.",
        f"{BU}/usability.md:3",
    ),
    ("Button", "Avoid using too many buttons on a page.", f"{BU}/usability.md:5"),
    ("Button", "Use sentence-case capitalization for button labels.", f"{BU}/usability.md:6"),
    (
        "Button",
        "Keep button text short. Button text should be as short as possible with action words that clearly explain "
        "what will happen when the button is selected.",
        f"{BU}/usability.md:7",
    ),
    (
        "Button",
        "Lead with a verb. Make the first word of the button's text a verb. For example, instead of Complaint filing,"
        " label the button File a complaint.",
        f"{BU}/usability.md:8",
    ),
    ("Button", "Buttons should display a visible focus state when users tab to them.", f"{BU}/accessibility.md:1"),
    (
        "Button",
        "Keep unavailable actions understandable. Explain why an action is unavailable and how to make it available.",
        f"{BU}/accessibility.md:4",
    ),
]


def write(path: Path, prefix: str, rows: list[tuple[str, str, str]]) -> None:
    out = ["id\tpage\ttext\tupstream"]
    for i, (page, text, ref) in enumerate(rows, 1):
        assert "\t" not in text and "\n" not in text
        out.append(f"{prefix}{i}\t{page}\t{text}\t{ref}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{path.relative_to(HERE.parent)}: {len(rows)} lines")


def govuk(clone: Path) -> None:
    colours = json.loads((clone / "data/colours.json").read_text())
    func = []
    for group, items in colours["functional"].items():
        if group == "Print":  # hidden on the published page (hiddenGroups: ["Print"])
            continue
        for it in items:
            note = f" {it['notes']}" if it.get("notes") else ""
            func.append(
                (
                    "Colour",
                    f"Functional colour {it['name']}: {it['colour']}.{note}",
                    f"data/colours.json functional.{group}",
                )
            )
    pal = []
    for group, items in colours["palette"].items():
        vals = ", ".join(f"{it['name']} {it['colour']}" for it in items)
        pal.append(("Colour", f"Web palette, {group}: {vals}.", f"data/colours.json palette.{group}"))
    rows = GOVUK_HEAD + func + GOVUK_MID + pal + GOVUK_TAIL
    write(HERE / "govuk" / "source.tsv", "G", rows)
    write(HERE / "govuk" / "wcag.tsv", "W", WCAG_LINES)


def uswds(site: Path, core: Path) -> None:
    rows = list(USWDS_HEAD)
    tok = core / "packages/uswds-core/src/styles/tokens/color"
    for f in sorted(tok.glob("_*.scss")):
        fam = f.stem[1:]
        if fam in ("black-transparent", "white-transparent", "global", "index"):
            continue
        std, _, viv = f.read_text().partition('"vivid"')

        def grades(block: str) -> str:
            return ", ".join(f"{g} {h.lower()}" for g, h in re.findall(r"^\s*(\d+):\s*(#[0-9a-fA-F]{6})", block, re.M))

        line = f"System color family {fam}: {grades(std)}."
        if grades(viv):
            line += f" Vivid: {grades(viv)}."
        rows.append(("System color tokens", line, f"uswds-core src/styles/tokens/color/{f.name}"))
    rows += USWDS_MID
    sp = yaml.safe_load((site / "_data/tokens/spacing.yml").read_text())["positive"]
    vals = ", ".join(f"{t['token']} = {t['value']}" for k in ("smaller", "small", "medium") for t in sp[k])
    rows.append(("Spacing units", f"Spacing unit tokens: 0 = 0, {vals}.", "_data/tokens/spacing.yml positive"))
    rows += USWDS_TYPE[:2]
    size = yaml.safe_load((site / "_data/tokens/typesetting/size.yml").read_text())
    vals = ", ".join(f"{t['token']} = {t['value']}" for t in size["system"])
    rows.append(("Font size", f"System size tokens (target size): {vals}.", "_data/tokens/typesetting/size.yml system"))
    vals = ", ".join(f"{t['token']} = system {t['default']}" for t in size["theme"])
    rows.append(("Font size", f"Theme size tokens: {vals}.", "_data/tokens/typesetting/size.yml theme"))
    rows += USWDS_TYPE[2:]
    fam = yaml.safe_load((site / "_data/tokens/typesetting/family.yml").read_text())
    normal = yaml.safe_load((site / "_data/tokens/typesetting/meta.yml").read_text())["normal"]
    vals = "; ".join(f"{f['name']}: {normal / f['normal']:.2f}" for f in fam["font"])
    rows.append(
        (
            "Using type",
            f"Normalization (final output = target size x normalization): {vals}.",
            f"{TO}:50-70 with _data/tokens/typesetting/family.yml",
        )
    )
    names = ", ".join(f["name"] for f in fam["font"] if f["token"] != "system")
    rows.append(
        (
            "Font family",
            f"USWDS available fonts: {names}, and system fonts.",
            f"{FF} with _data/tokens/typesetting/family.yml",
        )
    )
    rows += USWDS_BUTTON
    stacks = "; ".join(
        f"{f['name']} is served as {f['stack-name']}"
        for f in fam["font"]
        if f.get("stack-name") and f["stack-name"] != f["name"]
    )
    rows.append(
        ("Font family", f"Font names in CSS (stack names): {stacks}.", "_data/tokens/typesetting/family.yml stack-name")
    )
    write(HERE / "uswds" / "source.tsv", "U", rows)


if __name__ == "__main__":
    govuk(Path(sys.argv[1]))
    uswds(Path(sys.argv[2]), Path(sys.argv[3]))
