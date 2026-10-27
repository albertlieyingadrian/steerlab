# steerlab design system

One source for how anything steerlab makes should look: research notes, the research page, the README, the web demo,
images, diagrams and 3D scenes. Read it before designing; update it when a decision changes.

## Identity

**steerlab is a control desk.** It turns a model's insides into something you can watch, push with a fader, and
measure. Every visual choice comes from that idea or from steerlab's own data, never from another lab's look.

Principles:
1. **Show the real thing.** Hero art is drawn from the experiment's own data; figures come from `results/`.
   Illustration explains a mechanism, it never decorates.
2. **Two colours, two meanings.** Steer orange is *what steering does* (the pushed activations, the effect).
   Fader blue is *the control* (settings, sliders, the steering layer). Grey is everything else, including the
   random-direction controls.
3. **Measured, not hyped.** Numbers lead (counts and intervals), with the random control next to them.
4. **A publication on light paper, screens on dark.** Pages are light; art, demos and 3D scenes sit on the Ember
   ground, like screens on the desk.

## Colour

| Token | Light | Dark | Meaning and use |
|---|---|---|---|
| `--canvas` | `#ffffff` | `#121516` | Page background. Light is the default; dark only from the reader's switch, never the system setting |
| `--ink` | `#1d272a` | `#e8ecec` | Text |
| `--ink-soft` | `#586468` | `#a7b1b3` | Secondary text, captions |
| `--ink-faint` | `#8a9497` | `#7b8689` | Labels, meta |
| `--rule` | `#e3e7e8` | `#2a3133` | Hairlines, borders |
| `--panel` | `#f5f6f4` | `#1b2022` | Code blocks only (no boxed prose) |
| `--accent` steer orange | `#c9561c` | `#e07a3f` | What steering does: steered data, the direction arrow, headline numbers |
| `--accent-soft` | `#f8e6dc` | `#3a2419` | Highlighted table row, emphasised words in quoted replies |
| `--control` fader blue | `#4a86e8` | `#6f9ff0` | Controls and settings: faders, sliders, channel tags, the steering layer in art |
| `--ground` Ember | `#15110e` | `#15110e` | Art, 3D scenes, the web demo (its own palette) |

Art palette on the Ember ground: stream greys `#6f675f` → `#a39a90`, steered `#f2b48e` → `#e07a3f` → `#c9561c`,
control `#4a86e8`, text `#d9d2ca`. Never use orange for decoration or blue for data.

## Type

| Role | Face | Use |
|---|---|---|
| Display | **Bricolage Grotesque** 700 (650 for small labels) | Titles (44px, −0.025em), section headings (32 / 22px), card titles, figure labels |
| Body | **Source Serif 4** 400, 17px / 1.65, 18px between paragraphs | Reading text, summaries (dek 19px), quotes (italic) |
| Mono | **JetBrains Mono** | Eyebrows, meta labels, table headers, channel tags, fader values, code |

Load from Google Fonts with `display=swap`. Curly quotes and apostrophes in text. Numbers in tables are tabular.

## Layout

- **Research note** (`experiments/<name>/report/`): one ~680px column centred; contents rail fixed at the window's left
  edge (13px, current section underlined in ink), hidden below 1180px. Order: eyebrow → title → one-sentence dek →
  hero (16:9, square corners) → meta row → fader strip → key results (a plain list) → numbered sections (Introduction, Setup, How the study
  was run, Results, What this does not show, Reproduce it, What's next, Appendix).
- **Research page** (`experiments/index.html`): one row per note, newest first: a category chip, the title, one line,
  author and date on the left; the note's 3:2 cover on the right; hairline rules between rows.
- **README**: logo, pitch, badges, links; the demo GIF; why (problem, goal, requirements, non-goals); first result;
  the research table; quick start; how it works (the two "what is steered" pictures); setup.

## Components

- **Fader strip**: the settings behind the headline result, one fader per setting (label, blue track, value).
- **Channel tags**: section numbers as small outlined mono tags in fader blue ("01").
- **Process steps** (`ol.steps`): the "How the study was run" timeline; numbered blue rings on a hairline, each
  step a bold lead-in, a sentence, and its trial count in mono.
- **Key results**: plain bullets, each opening with its claim in bold, with numbers and intervals.
- **Figures**: column width; caption left-aligned, 14px, label "Figure N." in the display face; every figure cited.
- **Quotes of model replies**: italic serif with a thin left rule; the steered words highlighted with `--accent-soft`.
- **Tables**: a 1px ink rule on top, hairlines between rows, at most one highlighted row (orange left bar).
- **Theme switch**: a small pill top right, "Dark mode" / "Light mode", remembered per reader.

## Art

- **Covers (hero and thumbnail): a refined editorial illustration, one per note.** Generated with `gpt-image-2`
  (1536×1024, quality high), on warm off-white paper, detailed and crafted, never cartoonish (no faces, no mascots).
  Each note gets its own medium chosen to fit its idea (an isometric miniature, an etching, a specimen plate, a print);
  the picture must say what the note is about at a glance (for a language note, show the language; for a model,
  show its layers or neurons). Save the hero as JPEG at 1600px and the thumbnail at 720px in `experiments/shared/thumbs/`.
- **Data art.** `experiments/shared/make_art.py` draws a flow field from an experiment's saved direction: streamlines
  run straight to the steering layer (a blue rule) and then follow the direction. Used inside a note as the
  "drawn from the data" figure, not as its cover.
- **Explanatory pictures: image model, Ember technical style.** When a mechanism needs a picture (what is steered,
  how much moves), generate with `gpt-image-2` using this style line, and check every label's spelling:
  "Precise technical illustration, like an engineering cutaway on a control desk screen: very dark warm charcoal
  background (#15110E), thin light warm-grey line work (#D9D2CA), steer orange (#C9561C) only for the steering vector
  and what it changes, one accent of fader blue (#4A86E8) for the control. Clean, calm, generous space, slight glow on
  the orange. Labels in a clean sans-serif, spelled exactly as given, nothing else written."
  Save as JPEG at 1600px (about 150 to 350 KB).
- **Diagrams and charts**: drawn in code (SVG or matplotlib) from data, with the colour meanings above.
- **Interactive 3D**: one per note, a small three.js scene written from a precise brief (GPT-6 Astra did the first),
  on the Ember ground in both themes, with a blue strength slider. Reduced motion respected, off-screen paused, a
  static image as fallback, and a caption saying it is a picture of the idea, not a recording.
- **Recordings and screenshots**: from the public clone only; check every frame's corners for paths and names.

## Logo

`docs/assets/logo.png`: the stream mark (parallel lines, one nudged off course by an arrow), orange on a dark rounded
tile with transparent corners. Use it at 64px or more; do not recolour it or put it on orange.

## Process

1. Read this file.
2. Build, then screenshot at 1440px and 390px, light and dark.
3. For anything visual and public, run a two-model design review (GPT-6 Astra and Claude critique independently,
   then review each other's lists) and apply what they agree on.
4. Record new decisions here.

## Decision log

- 2026-10-27: Light by default with a manual dark switch (the system dark mode made pages look unlike a publication).
- 2026-10-27: The control-desk identity: Ember ground for screens and art, grotesk display over a serif body,
  orange = effect and blue = control, fader strip and channel tags.
- 2026-10-27: Covers are refined editorial illustrations, one medium per note; data art moves inside the notes. The
  research page lists notes in rows.
- 2026-10-27: README explains what is steered (activations at one layer, not weights, not one neuron) with two
  Ember technical pictures.
- 2026-10-27: Every note gets a "How the study was run" section between Setup and Results: the process as
  numbered steps with trial counts, and what was fixed before the trials versus changed after reading replies.
