# seq2feature replication and extension: pre-registration (2026-11-28)

Written and committed before any scored run. One check was run before this file: while reviewing the repo for an
upstream fix, notebook 02's own cells were run on the released probe and data, and the leaderboard's top-5 agreement
was computed for the probe (0.9556), the bag of words (0.8421) and the two floors (0.2965, 0.0673). Those four numbers
are re-run under this plan and reported as replicated, not as tested predictions. Nothing else was scored.

Project: seq2feature (github.com/patrickod32/seq2feature, commit `f0616ac`). It distils a Gemma-2-9B GemmaScope SAE
(`gemma-scope-9b-pt-res-canonical`, `layer_20/width_16k/canonical`, 2,048 of 16,384 latents kept) into a 5.3 MB
text-only probe, and reports how well the probe reproduces the SAE's span-level firings on held-out text.

## Claims under test (README and LEADERBOARD.md)

| # | Claim | Their number |
|---|---|---|
| C1 | Top-5 agreement of the int8 probe on the by-document held-out split | 0.956 |
| C2 | Per-concept AUC (median over concepts) | 0.90 |
| C3 | Bag of words (hashed 1-2 grams, ridge): top-5 / AUC | 0.84 / 0.75 |
| C4 | bge-small zero-shot (span-caption cosine): top-5 / AUC | 0.23 / 0.53 |
| C5 | Floors: frequency-5 / random-5 | 0.30 / 0.07 |
| C6 | int8 is lossless against fp32; int4 costs about 2 points of top-5 | int4 0.936 |
| C7 | Out of distribution (PubMed + LEDGAR, fresh SAE firings): median AUC / top-5 / bge AUC | 0.75 / 0.94 / 0.55 |

## Definitions (fixed)

- **Span label** `Y[s,c]`: the SAE fired concept `c` above 1.0 at any token of span `s` (`targets.npz`, as in the
  repo).
- **Split**: the repo's by-document rule, test = `crc32(doc) % 100 >= 70`.
- **Top-5 agreement**: for each test span, the share of the 5 highest-scored concepts with `Y = 1`, averaged over
  spans. The probe ranks by z-score (`(score - mu) / sd`, train statistics shipped with the checkpoint), as in the
  repo.
- **Per-concept AUC**: ROC AUC of a concept's scores against its labels over test spans, for concepts with at least
  one positive and one negative; report the median (README) and the mean (notebook 01's `macroAUC`).
- **Density** of a concept: its firing rate over train spans.
- **Intervals**: 95% percentile bootstrap over test **documents** (spans in one document are not independent),
  2,000 resamples for top-5 metrics, 200 for medians of per-concept AUC. OOD intervals resample OOD documents.

## Part R: replication (CPU, released assets)

R1 to R6 recompute C1 to C6 from the released probe (`golf_final/`), data (`relabel_bundle/`) and notebook code.
- fp32: `mega_golf_probe.pth`; int8: the shipped `mega_golf_probe_int8.pkl.gz`, loaded with `probe/infer.py`'s
  dequantiser; int4: notebook 01's `fq(w, 4)` (per-row symmetric, all parameters), z-scored with statistics refitted
  on train spans as notebook 01 does.
- Bag of words: notebook 02's code exactly (4,096 hashed features, ridge alpha 2, 9,000 train spans, seed 0); top-5
  ranks the raw ridge predictions.
- bge: notebook 03's code (`BAAI/bge-small-en-v1.5`, normalised embeddings, span x caption cosine), ranked by raw
  cosine; a z-scored variant (train statistics) is reported beside it as a sensitivity check.

R7 recomputes C7: notebook 03's corpora and span rules (first 200 PubMed articles, first 200 LEDGAR provisions, sentence
spans of 12-80 Gemma tokens, at most 10 per document), harvested on Modal with Gemma-2-9B + the same SAE exactly as
notebook 03 does (no BOS token, bfloat16, firing threshold 1.0).

Agreement rule: a replicated number "matches" when the published value lies inside our 95% interval, or within 0.01
of our point estimate where the interval is narrower than that.

## Part E: extensions (registered before any of them was run)

### E1. How much of top-5 agreement is frequency? (CPU)

The SAE fires about 138 of the 2,048 concepts per span, and some concepts fire on a large share of spans, so a ranker
that knows only the base rates already scores 0.30. E1 asks how much more of the 0.956 a ranker gets from knowing the
span's genre alone.

- **B-dom-oracle**: the 5 most frequent train concepts within the span's true domain (web, code, cot, math, chat).
- **B-dom-text** (fair, text-only): a logistic regression on notebook 02's hashed bag of words predicts the domain
  (fit on train spans); then the 5 most frequent train concepts in the predicted domain.
- **Lift@5**: top-5 agreement minus the mean train density of the 5 picked concepts (what picks of those concepts would
  score if they were unrelated to the span). Reported for the probe and every baseline.
- **Density bins**: per-concept AUC for the probe and the bag of words in five fixed density bins: < 1%, 1-2%, 2-5%,
  5-10%, >= 10% of train spans.
- **Rare-half top-5**: top-5 agreement when the ranking is restricted to concepts below the median density, with its
  own frequency-5 and random floors.
- The same frequency floors are computed for the OOD set (train densities applied to OOD labels).

### E2. Same text, different model state (Modal GPU)

A text-only probe returns the same scores for the same text whatever came before it. The SAE reads activations, which
depend on the preceding context. E2 measures how much of the SAE's span-level readout changes when only the context
changes, which bounds what any text-only probe can recover.

Sample: 1,000 test spans, 200 per domain, drawn with numpy seed 0 from test spans whose document has an earlier span
(so a natural preceding context exists). Conditions, each read on the span's own token positions only:

| Condition | Input to Gemma-2-9B |
|---|---|
| K0 as harvested | the span's stored Gemma ids, no BOS (what the repo's labels were made from) |
| K1 with BOS | `<bos>` + span ids |
| CTX natural context | `<bos>` + the previous span of the same document + `" "` + span |
| F-honest | `<bos>` + "The following text was written by a careful, honest assistant.\n\n" + span |
| F-sandbag | `<bos>` + "The following text was written by an assistant that is deliberately underperforming and hiding its real ability.\n\n" + span |
| F-lie | `<bos>` + "The following text was written by an assistant that is lying to the user.\n\n" + span |

Span tokens are the stored ids (not re-tokenised), so the text is byte-identical in every condition. For each span and
condition we store the max SAE activation over span positions for all 16,384 latents (float16) and the kept-concept
labels at threshold 1.0.

Measures:
- **Known-answer gate**: K0 labels against the shipped `targets.npz` labels for the same spans: mean per-span Jaccard
  over kept concepts. Pass bar 0.90. If it fails, E2 stops and the mismatch is reported.
- **Context share**: 1 - mean per-span Jaccard between K1 and each other condition (kept concepts).
- **Probe against each condition**: top-5 agreement and median AUC of the (unchanged) probe scores against each
  condition's labels.
- **Framing readout**: a logistic regression (L2, C = 1, on log1p of the max activations of all 16,384 latents) told
  F-sandbag from F-honest, and separately F-lie from F-honest, on the same 1,000 texts; 5-fold cross-validation
  grouped by document; AUROC. The text-only probe's AUROC on the same task is 0.5 by construction (identical input),
  which the analysis also computes as a check. A shuffled-label twin of the classifier is the control.
- **Per-concept state sensitivity**: for each kept concept, the share of spans whose label differs between K1 and CTX
  (flip rate); Spearman correlation between flip rate and the probe's in-distribution per-concept AUC, with a
  bootstrap interval over concepts. A partial version controls for density (rank-residualised).
- **Framing-sensitive concepts**: McNemar tests per kept concept, F-sandbag against F-honest, Benjamini-Hochberg at
  0.05; the count, and their captions.

### E3. Safety-labelled concepts (CPU, plus E2's data)

Safety subset: kept concepts whose caption matches (case-insensitive)
`refus|harm|danger|weapon|deceiv|decept|\blie\b|lying|illegal|crime|criminal|violen|hack|password|secret|threat|manipulat|attack|safety|ethic|unauthori|exploit|steal|abuse|malware|scam|fraud|toxic|offensive`.
Reported: the count, the probe's AUC on them (median, with the bag of words beside it), their density, and their E2
flip rates against the other concepts.

## Design

Claim sentence: "The released probe reproduces top-5 agreement __ [CI] and median AUC __ [CI]; a genre-only ranker
gets __ of top-5 agreement; on identical text, a change of context alone changes __% of the SAE's span readout, and
the SAE's features tell a sandbagging framing from an honest one at AUROC __ where the text-only probe is at 0.5."

1. Presence is not use: no claim here is about what the model uses; the SAE readout is the target throughout, as in
   the original. E2's framing readout says the activations carry the framing, not that the model acts on it.
2. Floor and wobble. Random and frequency floors, domain baselines (E1), shuffled-label classifier (E2), document
   bootstrap everywhere. Boring explanations: (a) top-5 agreement is base rates: E1; (b) the E2 differences are bf16
   noise: K0 against the shipped labels bounds it, and K1 is run twice on 200 spans to measure run-to-run flips;
   (c) the framing readout reads the prefix's length or position: every framing is read on span positions only, and
   the three framings are compared pairwise; (d) the BOS-less harvest makes position 0 odd: K0 against K1.
3. The ruler is a hypothesis: the known-answer gate above; metric code is checked by recomputing the four
   leaderboard numbers before anything new.
4. Searched is not reported: nothing is tuned. The probe is frozen. The safety regex, bins and prefixes are fixed here.
5. Units of truth first: labels are the SAE's, at the repo's threshold, fixed before any run.
6. Test where it will be used: the same model, SAE, dtype, threshold and span rules as the repo.

Budget cap (in code): $10 of Modal on profile credits30, one GPU. Timebox: 2026-11-28.

## Predictions

| # | Prediction | Metric | Threshold | Confidence |
|---|---|---|---|---|
| P1 | A genre-only ranker gets much of top-5 agreement | B-dom-text top-5 agreement | >= 0.55 | 60% |
| P2 | The probe still clearly beats genre | probe minus B-dom-text top-5 | >= 0.15 | 80% |
| P3 | The probe does well on rare concepts too | rare-half top-5 agreement | >= 0.80 | 60% |
| P4 | Same text, different framing: activations separate it | framing AUROC, F-sandbag vs F-honest | >= 0.90 | 75% |
| P5 | Natural context changes a real share of the readout | 1 - Jaccard(K1, CTX) | >= 0.15 | 60% |
| P6 | The probe is worst where context matters | Spearman(flip rate, probe AUC) | <= -0.20 | 55% |
| P7 | The OOD result replicates | our OOD median AUC minus 0.75 | within +-0.03 | 70% |

## Amendments

**Amendment 1 (2026-11-28, after the 32-span smoke run, before any scored run).** Two clarifications and one added
diagnostic.

- The span records store text and stripped token strings, not Gemma ids. "Stored ids" in E2 means the ids notebook 00
  made: the span text through the Gemma tokenizer, BOS removed, cut to 80. The smoke run checked them against the
  stored token count: 0 of 32 mismatched.
- The smoke run's 32 spans (not scored) showed K0 reproducing the shipped labels (Jaccard 0.98) and every condition
  with a BOS token far from them (about 0.26). GemmaScope SAEs were trained on BOS-prefixed text, so the size of this
  gap needs the SAE's own fit beside it. Added to every E2 condition and to the OOD set: per-token L0, reconstruction
  FVU and residual norm, on the span's first position and on the rest. GemmaScope's published `average_l0_68` for
  this SAE is the reference for L0. The OOD set is also read with a BOS token (`maxact_bos`); its probe metrics are
  reported as exploratory, beside the registered no-BOS replication.
- No prediction, threshold or measure was changed.
