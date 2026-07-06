// Decoding: turning next-token logits into a choice. Temperature, top-k and top-p (nucleus) sampling.

import { softmax } from './math';

export interface Candidate {
  token: string;
  logit: number;
}

export interface SampledCandidate extends Candidate {
  /** Softmax at temperature 1, before any filtering. */
  base: number;
  /** After temperature, before filtering. */
  tempered: number;
  /** Final probability after filtering and renormalizing (0 if cut). */
  p: number;
  cut: 'top-k' | 'top-p' | null;
  /** Cumulative probability in rank order after top-k (what top-p compares against); NaN if cut by top-k. */
  cumulative: number;
}

export interface SamplingParams {
  temperature: number;
  /** 0 disables top-k. */
  topK: number;
  /** 1 disables top-p. */
  topP: number;
}

export function applySampling(candidates: Candidate[], { temperature, topK, topP }: SamplingParams): SampledCandidate[] {
  const sorted = [...candidates].sort((a, b) => b.logit - a.logit);
  const base = softmax(sorted.map((c) => c.logit));
  const t = Math.max(temperature, 1e-3);
  const tempered = softmax(sorted.map((c) => c.logit / t));

  // Same order as common implementations (e.g. Hugging Face): temperature → top-k → top-p,
  // where top-p is measured on the distribution renormalized after top-k.
  const k = topK > 0 ? Math.min(topK, sorted.length) : sorted.length;
  const massK = tempered.slice(0, k).reduce((s, p) => s + p, 0);
  let cumulative = 0;
  const rows: SampledCandidate[] = sorted.map((c, i) => {
    let cut: SampledCandidate['cut'] = null;
    if (i >= k) cut = 'top-k';
    const before = cumulative;
    if (!cut) cumulative += tempered[i]! / massK;
    // Nucleus: keep the smallest prefix whose mass reaches topP (always keep the first token).
    if (!cut && i > 0 && before >= topP - 1e-12) cut = 'top-p';
    return { ...c, base: base[i]!, tempered: tempered[i]!, p: 0, cut, cumulative: cut === 'top-k' ? NaN : cumulative };
  });

  const kept = rows.filter((r) => !r.cut).reduce((s, r) => s + r.tempered, 0);
  for (const r of rows) r.p = r.cut ? 0 : r.tempered / kept;
  return rows;
}

export function sample(rows: SampledCandidate[], random: () => number = Math.random): number {
  let u = random();
  for (let i = 0; i < rows.length; i++) {
    u -= rows[i]!.p;
    if (u <= 0 && rows[i]!.p > 0) return i;
  }
  return rows.findIndex((r) => r.p > 0);
}

/** Shannon entropy in bits. */
export function entropy(ps: number[]): number {
  return -ps.reduce((s, p) => (p > 0 ? s + p * Math.log2(p) : s), 0);
}

export const PROMPTS: { prompt: string; candidates: Candidate[] }[] = [
  {
    prompt: 'The cat sat on the',
    candidates: [
      ['mat', 3.1], ['floor', 2.6], ['sofa', 2.3], ['bed', 2.0], ['couch', 1.9], ['roof', 1.3],
      ['table', 1.1], ['chair', 0.9], ['ground', 0.8], ['windowsill', 0.5], ['moon', -0.8], ['keyboard', -0.3],
      ['piano', -1.1], ['banana', -2.0],
    ].map(([token, logit]) => ({ token: token as string, logit: logit as number })),
  },
  {
    prompt: 'The capital of France is',
    candidates: [
      ['Paris', 6.5], ['a', 2.3], ['the', 2.0], ['known', 1.2], ['Lyon', 0.4], ['located', 0.2],
      ['not', -0.4], ['beautiful', -0.6], ['Marseille', -0.9], ['London', -1.8],
    ].map(([token, logit]) => ({ token: token as string, logit: logit as number })),
  },
  {
    prompt: 'My favourite colour is',
    candidates: [
      ['blue', 2.4], ['green', 2.2], ['red', 2.1], ['purple', 1.9], ['black', 1.6], ['yellow', 1.5],
      ['orange', 1.3], ['pink', 1.2], ['white', 1.0], ['grey', 0.8], ['teal', 0.6], ['magenta', 0.2],
    ].map(([token, logit]) => ({ token: token as string, logit: logit as number })),
  },
];
