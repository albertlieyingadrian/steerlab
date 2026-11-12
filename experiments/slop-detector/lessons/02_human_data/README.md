# Lesson 02: Human text you can trust

**Goal:** collect thousands of documents that humans certainly wrote, across the same seven kinds of writing as the
card, without labelling anything by hand.

## The idea

A detector learns "human" from examples of human text. If some of those were secretly written by ChatGPT, it learns
the wrong thing, and nobody can check 100,000 documents by hand. The trick: **use text written before ChatGPT
existed** (November 2022). The date does the labelling. It is like trusting a bank statement from before an account
was hacked: you don't audit each line, you check the date.

## Sources

The card lists seven sources. We take one public file from each, so nothing huge is downloaded:

| source | dataset on the Hub | years | median words |
|---|---|---|--:|
| Amazon reviews | McAuley-Lab/Amazon-Reviews-2023, 4 small categories | 2020–2021 | 55 |
| arXiv abstracts | gfissore/arxiv-abstracts-2021 | 2020 | 157 |
| Hacker News comments | OpenPipe/hacker-news, shard 30 | 2022 | 66 |
| Medium posts | fabiochiu/medium-articles | 2020–2021 | 279 |
| Reddit comments | HuggingFaceGECLM/REDDIT_comments, 10 subreddits | 2020–2022 | 59 |
| Stack Overflow | mikex86/stackoverflow-posts, shard 55 | 2022 | 78 |
| BBC news (XSum) | EdinburghNLP/xsum | 2017 | 272 |

The pilot keeps 1,600 documents per source, 11,200 in all (the card used 98k). Long documents are cut to their
first ~300 words at a paragraph or sentence end, since the model only reads the first 512 tokens anyway.

## Two rules that keep the test honest

1. **Split by document, not by row.** Every human document later gets three AI versions. If the rewrite landed in
   training and its human original in the test set, the test would be partly memory. So the split is a hash of the
   document id (`split_of` in `01_human.py`), and every version inherits it. A hash is deterministic: the same id
   always lands in the same bucket, on any machine, with no table to store.
2. **Don't train on the test's sources.** EditLens's test set includes r/WritingPrompts, so that subreddit is left out.
   Lesson 04 also drops any of our documents that shares text with a benchmark, which caught 37 BBC articles. EditLens's
   "news" texts overlap with XSum.

## Run it

```bash
cd experiments/slop-detector && ../../.venv/bin/python lessons/02_human_data/lesson.py   # hash split and trimming, instantly
cd experiments/slop-detector && ../../.venv/bin/python 01_human.py        # the real collection, ~10 min, ~4 GB download
```

`01_human.py` writes `data/human.jsonl`, one line per document: `{"id", "source", "year", "split", "text", "words"}`.
It is gitignored, because the texts belong to their authors and the platforms.

## What we found

- The split comes out at 79.5 / 10.2 / 10.4 over 10,000 ids, close to the 80 / 10 / 10 target, with no table stored.
- XSum contains some Welsh BBC Cymru articles. Lesson 04 drops non-English text with a stop-word test.
- Sources differ a lot in length: Amazon reviews have a median of 55 words, Medium posts 279. Length is a shortcut a
  detector can learn by accident ("long means AI"), which is why each AI version is asked for about the same length
  as its human original.

## Check yourself

1. Text from 2021 is "certainly human". What could still go wrong with that label? (Think about older bots, templates
   and translation tools.)
2. Why is a hash better than `random.shuffle` for the split, if you plan to add more data later?
3. Our arXiv sample is all from 2020. Does that matter for a detector? When would it?

Next: [lesson 03](../03_ai_versions/README.md), making the AI half.
