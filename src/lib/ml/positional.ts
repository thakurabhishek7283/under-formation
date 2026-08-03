// How a transformer learns where a token is.
//
// Attention is a weighted sum over a *set*: shuffle the tokens and, without positional
// information, every output is unchanged. Three ways to fix that, and the differences between
// them are visible in one picture — the matrix of similarities between every pair of positions.

import { rng, randn } from './math';

export type PosMode = 'sinusoidal' | 'learned' | 'rope';

export const POS_MODES: { id: PosMode; name: string; blurb: string }[] = [
  {
    id: 'sinusoidal',
    name: 'Sinusoidal (added)',
    blurb: 'The original transformer: a fixed pattern of sines and cosines added to the token embedding.',
  },
  {
    id: 'learned',
    name: 'Learned (added)',
    blurb: 'One trainable vector per position, as in BERT and GPT-2. Nothing forces any structure on it.',
  },
  {
    id: 'rope',
    name: 'RoPE (rotated)',
    blurb: 'Rotary embeddings: rotate q and k by an angle proportional to position. Used by Llama, Qwen, Mistral.',
  },
];

/** One angular frequency per dimension pair: fast for the first pairs, very slow for the last. */
export function frequencies(dim: number, base: number): number[] {
  const pairs = Math.floor(dim / 2);
  return Array.from({ length: pairs }, (_, i) => base ** ((-2 * i) / dim));
}

/** PE[pos][2i] = sin(pos·θᵢ), PE[pos][2i+1] = cos(pos·θᵢ). */
export function sinusoidalTable(positions: number, dim: number, base: number): number[][] {
  const theta = frequencies(dim, base);
  return Array.from({ length: positions }, (_, p) => {
    const row: number[] = [];
    for (const t of theta) {
      row.push(Math.sin(p * t), Math.cos(p * t));
    }
    return row.slice(0, dim);
  });
}

/** A trainable vector per position, standing in for one that was never trained. */
export function learnedTable(positions: number, dim: number, seed = 31): number[][] {
  const r = rng(seed);
  return Array.from({ length: positions }, () => Array.from({ length: dim }, () => randn(r) * 0.6));
}

/** Rotate each (2i, 2i+1) pair of `v` by pos·θᵢ. This is RoPE applied to one query or key. */
export function rotate(v: number[], pos: number, base: number): number[] {
  const theta = frequencies(v.length, base);
  const out = [...v];
  for (let i = 0; i < theta.length; i++) {
    const a = v[2 * i]!;
    const b = v[2 * i + 1]!;
    const ang = pos * theta[i]!;
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    out[2 * i] = a * c - b * s;
    out[2 * i + 1] = a * s + b * c;
  }
  return out;
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i]!, 0);

/** A fixed, seeded "query" vector, so the RoPE demonstration isn't about any particular token. */
export function sampleVector(dim: number, seed: number): number[] {
  const r = rng(seed);
  const v = Array.from({ length: dim }, () => randn(r));
  const norm = Math.hypot(...v);
  return v.map((x) => x / norm);
}

/**
 * Similarity between every pair of positions.
 *
 * For the two *added* schemes this is PE_m · PE_n, the part of the attention score that comes
 * from position alone. For RoPE it is the full score between the same two vectors rotated to m
 * and n — which works out to depend only on (m − n), so the matrix is constant along its
 * diagonals.
 */
export function similarityMatrix(mode: PosMode, positions: number, dim: number, base: number, seed = 7): number[][] {
  if (mode === 'rope') {
    const q = sampleVector(dim, seed);
    const k = sampleVector(dim, seed + 1);
    return Array.from({ length: positions }, (_, m) =>
      Array.from({ length: positions }, (_, n) => dot(rotate(q, m, base), rotate(k, n, base))),
    );
  }
  const table = mode === 'sinusoidal' ? sinusoidalTable(positions, dim, base) : learnedTable(positions, dim);
  const scale = mode === 'sinusoidal' ? 2 / dim : 1 / Math.sqrt(dim);
  return table.map((a) => table.map((b) => dot(a, b) * scale));
}

/** Encoding values to display as a heatmap. RoPE has no table, so show the angle per pair. */
export function encodingTable(mode: PosMode, positions: number, dim: number, base: number): number[][] {
  if (mode === 'learned') return learnedTable(positions, dim);
  if (mode === 'sinusoidal') return sinusoidalTable(positions, dim, base);
  const theta = frequencies(dim, base);
  // cos of the rotation angle: same periodic structure, one column per dimension pair.
  return Array.from({ length: positions }, (_, p) => theta.map((t) => Math.cos(p * t)));
}

/**
 * Similarity as a function of offset, measured from several different base positions.
 *
 * With RoPE every base position gives the same curve — the score genuinely only knows the gap.
 * With an added encoding the curves separate, because the score depends on where you started.
 */
export function offsetCurves(mode: PosMode, positions: number, dim: number, base: number, bases: number[]): number[][] {
  const sim = similarityMatrix(mode, positions, dim, base);
  const maxOffset = positions - 1 - Math.max(...bases);
  return bases.map((m) => Array.from({ length: maxOffset + 1 }, (_, d) => sim[m]![m + d]!));
}

/** Wavelength of each frequency pair, in tokens: how far you travel before it repeats. */
export const wavelengths = (dim: number, base: number) => frequencies(dim, base).map((t) => (2 * Math.PI) / t);
