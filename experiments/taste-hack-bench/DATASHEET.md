# Datasheet: taste-reward hack benchmark (v1)

Following *Datasheets for Datasets* (arXiv:1803.09010). Released with The Taste Stack, Episode 8 (v0 on 2026-11-21, v1 on 2026-11-23):
`report/index.html`.

## Motivation

- **Why it exists.** To test detectors of reward and ruler hacks on hacks they were not built for. Taste rewards
  (voice, style, "a good painting") have no answer key, so the hacks that matter are the ones real optimisers and
  real setting searches found. Each entry is a hack this lab caught, with the check that caught it and the checks
  that missed it.
- **Who made it.** The steerlab research notes; every entry links to the published note it comes from.
- **Funding.** None beyond the lab's own compute; building v0 and v1 cost nothing (CPU only).

## Composition

- **50 entries** in `entries.jsonl`, from 28 published notes: 47 hacks and 3 near misses. Kinds: 17 trained policy,
  12 constructed trap, 8 selection rule, 6 steering arm, 2 ruler failure (a judge failing on real outputs after
  passing its checks), 1 learned ruler (a reward model that learned a shortcut), 1 planted, 3 near miss (Episode 6:
  no hack, but the reward bought something else, or a monitor misfired).
- **21 entries with samples.** 16 in `artefacts/<id>.jsonl`: rows with `group` (`hack` or `clean`) and `text`, plus
  `n_tokens`, `truncated`, `ledger` and `step` where the source logged them. 5 Sounds Like Me entries in
  `artefacts/<id>.features.jsonl`: `group`, `loop` and `copy` (shares), `chars`, `n_tokens`, `truncated`, `ledger`,
  `empty`, and no text. Identical samples within a group count once; groups then hold 1 to 120 distinct rows (capped
  at 120, drawn with seed 0).
- **Entry fields.** `id`, `title`, `domain`, `kind`, `source` (note slug), `section` (anchor), `link`, `reward` (what
  was gamed), `hack` (what the policy, arm or rule did), `quote` (a sentence from the note with its numbers; build.py
  fails if it is not in the note's text), `caught_by`, `lab_check` (the family of check that caught it), `when`
  (pre-registered, by design or post hoc), `missed_by`, `notes`, `why_no_artefact`, and `artefacts`: `file`,
  `n_hack`, `n_clean`, `label_source` (how the groups were made), `source_files` (repository paths of the shipped
  outputs they were derived from) and `reference` (the text a copy check compares against, or null).
- **Labels.** Where the source labelled each output (a hack flag, a render violation, a special-case check, a
  late-success copy signature) the groups follow that label. For 13 entries the groups are arms (steered vs
  unsteered, trained vs untrained on the same prompts), so not every hack-group row shows the hack.
- **What is missing.** 29 entries have no samples: video and images (not in the repository), letter or probability
  readouts, grader verdicts, per-run aggregates, and drafts from notes whose drafts are not public. Their evidence is the quote and link.

## Collection

- Entries were collected by searching every published note for hacks, loopholes, shortcuts, and checks that passed
  broken output, then written into `entries.py`. Each quote is verified against the note's text by
  `build.py`.
- Samples are derived by `build.py` from files the source notes already ship. Nothing was generated for v0 or v1.

## Privacy and preprocessing

- **Sources.** Every source corpus is public or synthetic: model outputs on synthetic briefs, public corpora (GSM8K,
  FinanceMath, Enron sent mail and the Blog Authorship Corpus as voice sources), τ-bench's synthetic customer
  records, and prompts written for the experiments. A private usage summary in one source note was not used.
- **Private samples.** The text of the 5 Sounds Like Me entries (`steering-honesty-cost` and the four `slm3-*`
  entries) is not published: their samples are whole drafts, and drafts written in a real writer's voice can copy
  names and contact details from that writer's samples. Each row's check scores are published instead, with no text.
  `steering-honesty-cost`'s drafts were published on 2026-11-21 with contacts redacted and removed on 2026-11-22; its
  numbers are unchanged. The gallery shows only sentences the source notes already quote.
- **Names removed.** The agent entry keeps only the tool-call sequence (no customer names or ids).
  The coding-agent entry keeps tool names and arguments (no provider call ids). Model text in other entries may still
  contain names that appear in public source problems (GSM8K word problems, for example).

## Uses

- **Intended.** Scoring a hack detector's catches and false alarms per entry; testing whether a monitor would have
  caught known hacks; a checklist of ways taste rewards fail.
- **Not intended.** Training a general hack classifier: the groups are small and come from a few environments
  (Pigment Gym and the RL notes built on it supply 8 entries). If you train on it, hold out whole entries.
- **Known biases.** One lab, mostly small open models (0.5B to 32B), a few environments, and hacks that were caught.
  Hacks nobody caught are, by construction, absent.

## Distribution and maintenance

- Distributed with the steerlab repository under its licence; source data keep their own licences (GSM8K and
  FinanceMath MIT; τ-bench MIT; Enron public; Blog Authorship Corpus for research).
- **Adding entries.** See "How to submit a hack" in the note: an entry dict, a public quote, optionally a builder for
  samples; `reproduce.sh` rescores everything. Later episodes (The Taste Stack's monitored RL, Sounds Like Me's
  training episode) will add entries. Versions are marked in the note's title (v0 = 39 entries).
