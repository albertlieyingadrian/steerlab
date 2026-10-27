# Writing a steerlab research note

Every experiment in `experiments/<name>/` ends in a research note: `report/index.html`, built on the shared style in
`experiments/shared/` (`report.css`, `report.js`). This file is the checklist for writing one, plus the lessons each
report has taught. Read it before starting a report; add to "Lessons learned" after finishing one.

The bar: a reader who knows nothing about steering finishes the key results knowing what was tried, what happened,
and how sure we are; a researcher finishes the page able to rerun every number.

## Checklist

### Structure

- [ ] **Header**: eyebrow (`← Experiments · <date>`), a title that states the finding or the question (not the
      method's name), a one or two sentence dek with the main number, and a metadata row (models, code, data, date).
- [ ] **Key results** box straight after the header: each bullet opens with its claim in bold and gives its number
      and, where there is one, its interval. As many bullets as there are findings; never pad to three.
- [ ] **Numbered sections** in this order, adapted as needed: 1 Introduction · 2 Setup · 3 Results · 4 What this does
      not show · 5 Reproduce it · 6 What's next · Appendix.
- [ ] **Introduction in three moves**: what is known (with a link), what is missing, what this note does.
- [ ] **State what counts as success before the method**: a short numbered list of tests a result must pass (a
      control, the text survives, it generalises), so the results can be read against them.
- [ ] **Results headings state the claim** ("Directions work on both Gemmas, partly on Qwen"), not the topic
      ("Direction results"). Give the rail a shorter `data-short` label.
- [ ] **Compare with what exists** when there is prior work: a small table with ✓ and ✗ against the criteria.
- [ ] **An ablation or a control is evidence**: show what happens when the ingredient is removed or replaced.
- [ ] **Every `h2` and `h3` has an `id`** (short, kebab-case); the contents rail and the `#` links are built from them.
      Give long headings a `data-short` label for the rail.
- [ ] **Appendix** lists every data file and what each column means, and the full question sets.

### Wording

- [ ] **Claim first, then evidence, then what it means.** One idea per paragraph.
- [ ] **Numbers as counts and intervals**: "12/15 [0.60, 1.00]", not "80%" alone. Say what the denominator is.
- [ ] **One term per idea.** Define *hit*, *strict hit*, *fluent*, *loop* once, in Setup, then never swap synonyms.
- [ ] **Hedge exactly as much as the evidence needs.** "On 15 questions" is a hedge; "may possibly suggest" is not.
- [ ] **No throat-clearing** ("In this section we will…"), no hype words (groundbreaking, robust, novel).
- [ ] **Plain words for non-specialists**: explain a term the first time it appears, with a small example.
- [ ] **Say what failed** with the same care as what worked; a negative result gets its own subsection.

### Figures and tables

- [ ] Figures are **numbered, cited in the text** ("Figure 2 shows…"), and their captions stand alone: what is
      plotted, the sample size, what the bars or intervals are.
- [ ] **Every figure is drawn by a script from `results/`** (`make_figures.py`), never by hand; the script prints the
      numbers it plotted so they can be checked against the text.
- [ ] Axes start at zero (or mark the break); intervals on every rate; no 3D, no dual axes, no decoration.
- [ ] **Colour carries meaning only**: ember (`--accent`) for the thing the note is about, grey for controls.
      Check it still reads in greyscale and in dark mode.
- [ ] Tables: numbers right-aligned in tabular figures; highlight at most one row (`tr.hl`), the headline one.
- [ ] Every image has `alt` text and a relative path (`figures/x.png`), so it works locally, on GitHub and on Pages.

### Honesty

- [ ] **A control for every effect**: a random direction of the same size, reported next to the result.
- [ ] **Held-out data** for anything that was tuned; say which numbers come from the set that was searched on.
- [ ] **Recompute every headline number from the shipped `results/`** before publishing, and quote the replies
      behind a surprising number.
- [ ] A "What this does not show" section: sample size, how replies were judged, what was not tried.

### Before publishing

- [ ] Open the page locally, in light and dark mode, at desktop width and at 390 px.
- [ ] Every link and image resolves (agent-browser: every `<img>` has `naturalWidth > 0`).
- [ ] Screenshots and videos show **no local paths, user names or private repo names** (check the corners and logs).
- [ ] `reproduce.sh --dry-run` prints every command behind the headline numbers.
- [ ] Add the lesson this report taught to the list below.

## Sources for this style

- Layout and typography: Goodfire's research posts (serif display title and summary, sans body, mono labels, a
  sticky contents rail, colour kept for figures).
- Writing and figures: the academic-paper skill in `imbad0202/academic-research-skills` (introductions in three
  moves, numbered self-contained figure captions, error bars, consistent terms, calibrated hedging).
- Voice: Anthropic's "Mapping the Mind of a Large Language Model" (the problem in everyday terms first, jargon
  glossed on first use, one vivid example, correlation kept apart from cause, careful limits).
- Argument shape: Exa's ATLAS benchmark post (takeaways first, success criteria before the design, claim headings,
  a ✓/✗ comparison with prior work, ablations as evidence).
- Technical-writing practice: Xamfonos' technical-writing best practices, Kaz Sato's two-reviewer workflow, and
  Mintlify's documentation guides.

## Lessons learned

Newest first. Each lesson names the report that taught it.

- **A looser success criterion can hide a loophole** (golden-gate). Counting "any bridge" as a Golden Gate hit let a
  generic *bridge* feature score 13/15; only 6/15 of those replies named the Golden Gate. Define the strict version of
  a metric up front and report it next to the loose one.
- **Short replies hide loops** (golden-gate). At 80 tokens, 5 of 15 Gemma-2-2B replies already repeat themselves while
  passing a character-diversity "fluent" check. Measure loops directly, on replies as long as the demo's.
- **The system prompt changes the best setting** (golden-gate). Llama's best clamp moved from 4 to 3 once scored under
  the demo's system prompt. Score under the conditions the result will be shown in.
- **Recordings leak paths** (golden-gate, front page). The web demo's log panel printed an absolute home path and a
  private repo name. Check every frame's corners before publishing a video or screenshot; make tools print short paths.
- **A feature that fires on a concept need not cause it** (golden-gate). SAE features labelled "San Francisco" did
  not steer toward the bridge at any fluent strength. Treat labels and activation rankings as candidates to test.
- **Small samples first, then confirm** (english-lock). Three hand-picked questions overstated scores by 0.1 to 0.2;
  rerun the winner on 15+ questions with intervals before quoting it.
