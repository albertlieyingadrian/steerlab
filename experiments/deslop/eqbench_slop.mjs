#!/usr/bin/env node
// EQ-Bench's slop score, computed with slop-score's own code (github.com/sam-paech/slop-score, MIT), so the note is
// not graded only on its own regexes.
//
//   node eqbench_slop.mjs <slop-score dir> <in.jsonl> <out.json>
//
// in.jsonl rows: {key, group, text}. Writes {rows: {key: metrics}, groups: {group: metrics}}. A group's metrics are
// computed the way the leaderboard computes a model's: its texts longer than 300 characters joined with blank lines.
// Metrics: slop words per 1000 words, slop trigrams per 1000 words, not-x-but-y matches per 1000 characters, and the
// composite slop score (60% words, 25% not-x-but-y, 15% trigrams, each min-max scaled over the leaderboard's models
// with a 10% margin, times 100), exactly as slop-score.html computes it.
import { readFileSync, writeFileSync } from 'fs';
import { resolve } from 'path';

const [dir, inPath, outPath] = process.argv.slice(2).map((p) => resolve(p));
process.chdir(dir); // slop-score loads ./data/... relative to its own folder

const realFetch = global.fetch;
global.fetch = async (url, ...args) => {
  if (typeof url === 'string' && (url.startsWith('./') || url.startsWith('data/'))) {
    const text = readFileSync(url, 'utf-8');
    return { ok: true, status: 200, json: async () => JSON.parse(text), text: async () => text };
  }
  return realFetch(url, ...args);
};

const metrics = await import(`${dir}/js/metrics.js`);
const utils = await import(`${dir}/js/utils.js`);
const contrast = await import(`${dir}/js/contrast-detector.js`);
await metrics.loadSlopSets();

const quiet = () => {
  const saved = [console.log, console.group, console.groupEnd, console.warn, console.error];
  console.log = console.group = console.groupEnd = console.warn = console.error = () => {};
  return () => ([console.log, console.group, console.groupEnd, console.warn, console.error] = saved);
};

const board = JSON.parse(readFileSync('data/leaderboard_results.json', 'utf-8')).results;
const range = (key) => {
  const vals = board.map((r) => r.metrics[key] || 0);
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.1;
  return { min: lo - pad, max: hi + pad };
};
const ranges = {
  words: range('slop_list_matches_per_1k_words'),
  trigrams: range('slop_trigram_matches_per_1k_words'),
  contrast: range('not_x_but_y_per_1k_chars'),
};
const norm = (v, r) => Math.max(0, Math.min(1, (v - r.min) / (r.max - r.min)));

function measure(text) {
  const toks = utils.alphaTokens(utils.wordsOnlyLower(text));
  if (!toks.length) return null;
  const slop = metrics.computeSlopIndex(toks);
  const restore = quiet();
  const matches = contrast.extractContrastMatches(text);
  restore();
  const contrastRate = text.length ? (matches.length / text.length) * 1000 : 0;
  const score = (norm(slop.wordScore, ranges.words) * 0.6 + norm(contrastRate, ranges.contrast) * 0.25 +
    norm(slop.trigramScore, ranges.trigrams) * 0.15) * 100;
  return {
    words: toks.length,
    slop_words_per_1k: slop.wordScore,
    slop_trigrams_per_1k: slop.trigramScore,
    contrast_per_1k_chars: contrastRate,
    contrast_matches: matches.length,
    slop_score: score,
  };
}

const rows = readFileSync(inPath, 'utf-8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const out = { ranges, rows: {}, groups: {} };
const byGroup = {};
for (const row of rows) {
  out.rows[row.key] = measure(row.text);
  if (row.group) (byGroup[row.group] ||= []).push(row.text);
}
for (const [group, texts] of Object.entries(byGroup)) {
  out.groups[group] = measure(texts.filter((t) => t.length > 300).join('\n\n'));
}
writeFileSync(outPath, JSON.stringify(out));
console.log(`scored ${rows.length} texts in ${Object.keys(byGroup).length} groups`);
