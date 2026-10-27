# steerlab search and AI-answer rules (SEO, AEO, GEO)

How the research pages stay easy to find in search, to share as link previews, and to quote in AI answers. Read it
before adding or changing a research note or the research page. Visual rules live in `DESIGN.md`; this file covers
only findability.

- **SEO** (search engine optimisation): Google and Bing can find, read and rank the page.
- **AEO** (answer engine optimisation): ChatGPT, Claude, Perplexity and Gemini quote the page when asked a question
  it answers.
- **GEO** (generative engine optimisation): AI summaries such as Google's AI Overviews use the page as a source.


## Metadata comes from one file

- `experiments/notes.json` describes every note: `name`, `title`, `description`, `summary` (one sentence with the main
  numbers and the random control), `published` and `modified` (`YYYY-MM-DD`), `image_alt`, `keywords`, `citations`.
- `python experiments/shared/build_meta.py` writes, from it:
  - in each page's `<head>`, between `<!-- meta:start -->` and `<!-- meta:end -->`: the canonical URL, Open Graph and
    Twitter tags, and JSON-LD (`ScholarlyArticle` per note; `CollectionPage`, `ItemList` and `SoftwareSourceCode` on
    the research page);
  - the `<time>` dates in each note's eyebrow ("Updated …" when `modified` differs from `published`);
  - `experiments/sitemap.xml` and `experiments/llms.txt`;
  - each note's 1200×630 link-preview image, `report/figures/og.jpg`, cropped from its cover.
- Never edit the `meta:start` block, the sitemap or `llms.txt` by hand. Change `notes.json` and rerun.
- When a note's content changes, set its `modified` date, and the research page's `modified` when its list changes.
- The links workflow runs `build_meta.py --check`, and `scripts/check_links.py` fails if a note has no `notes.json`
  entry.

## Every note has

- **One `<h1>`** (the title), then a clean h2 → h3 order; `<html lang="en">`.
- **A `<title>` and meta description** that state the finding, not just the topic.
- **An author line** in the meta row (name, with links to GitHub and LinkedIn; the same links as `sameAs` in the JSON-LD, from `author.same_as` in `notes.json`) and the publish date in the eyebrow.
- **"In short"** under the meta row: 130 to 170 words that answer the note's question on their own, with the main
  numbers and the random control. Start with the question ("Can activation steering override a system prompt?") and
  answer it in the next sentence. This is the paragraph an assistant quotes whole.
- **Definitions in "X is…" form** the first time a key term appears (activation steering, a direction, an SAE
  feature, a random-direction control).
- **A "Questions and answers" section** before the appendix: 4 to 6 questions people would type, as `<h3>` headings,
  each answered in its first sentence with a number where there is one. Plain HTML only.
- **"How to cite"** in the appendix: a BibTeX `@misc` block with the note's URL.
- **Text in the HTML**, not built by JavaScript: crawlers do not run scripts. Interactive parts (the reply explorer,
  3D scenes) add to the text; they never replace it.
- **Alt text** on every image that says what it shows, with the plotted values for charts.
- **Descriptive link text**: no "here" or "click here".
- **A cover JPEG under about 400 KB**, with `width` and `height` set on every `<img>`.

## Never

- FAQ or HowTo schema (HowTo is deprecated; FAQ rich results are for government and health sites).
- `Dataset` schema (retired from rich results in late 2025).
- Thin or near-duplicate pages written for keywords: one page per real experiment.
- Invented numbers: no made-up search volumes or citation results. Record only what was checked.
- FID in performance notes: Core Web Vitals use INP.

## Site limits

- GitHub Pages serves a project site under `/steerlab/`, so `robots.txt` cannot be set (it only works at the domain
  root, which returns 404 and so allows every crawler, AI crawlers included).
- The sitemap and `llms.txt` live in `experiments/`, the folder they describe: a sitemap may only list URLs under its
  own path.

## Off the page

- Verify the site in Google Search Console and Bing Webmaster Tools with an HTML meta tag, and submit
  `https://albertlieyingadrian.github.io/steerlab/experiments/sitemap.xml`.
- Share each new note on LinkedIn and X; AI summaries weigh mentions across platforms more than backlinks.
- Once the notes are indexed, check monthly whether ChatGPT, Claude, Perplexity and Gemini cite them for questions
  the notes answer, and record the result.
