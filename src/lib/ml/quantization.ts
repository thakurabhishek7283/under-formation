// Weight quantization: store weights as low-bit integers plus a floating-point scale.
// Symmetric absmax ("round to nearest") quantization, per-tensor or per-group.

import { randn, rng } from './math';

export function demoWeights(n: number, outliers: boolean, seed = 11): number[] {
  const r = rng(seed);
  const w = Array.from({ length: n }, () => randn(r) * 0.02);
  if (outliers) {
    // A handful of large-magnitude weights, like the outlier features found in real LLMs.
    for (const [i, v] of [[17, 0.31], [203, -0.27], [388, 0.24]] as const) if (i < n) w[i] = v;
  }
  return w;
}

export interface QuantResult {
  bits: number;
  groupSize: number;
  q: number[];
  scales: number[];
  dequant: number[];
  mse: number;
  maxError: number;
  /** Signal-to-quantization-noise ratio in dB. */
  sqnr: number;
  /** Distinct integer levels actually used. */
  levelsUsed: number;
  bitsPerWeight: number;
}

/**
 * Symmetric absmax: scale = max|w| / (2^(bits−1) − 1), q = round(w / scale), w' = q · scale.
 * groupSize 0 means one scale for the whole tensor.
 */
export function quantize(w: number[], bits: number, groupSize: number): QuantResult {
  const qmax = 2 ** (bits - 1) - 1;
  const g = groupSize > 0 ? groupSize : w.length;
  const q: number[] = [];
  const scales: number[] = [];
  const dequant: number[] = [];
  for (let start = 0; start < w.length; start += g) {
    const group = w.slice(start, start + g);
    const absmax = Math.max(...group.map(Math.abs)) || 1e-12;
    const s = absmax / qmax;
    scales.push(s);
    for (const x of group) {
      const qi = Math.max(-qmax, Math.min(qmax, Math.round(x / s)));
      q.push(qi);
      dequant.push(qi * s);
    }
  }
  const err = w.map((x, i) => x - dequant[i]!);
  const mse = err.reduce((s, e) => s + e * e, 0) / w.length;
  const power = w.reduce((s, x) => s + x * x, 0) / w.length;
  return {
    bits,
    groupSize,
    q,
    scales,
    dequant,
    mse,
    maxError: Math.max(...err.map(Math.abs)),
    sqnr: 10 * Math.log10(power / Math.max(mse, 1e-30)),
    levelsUsed: new Set(q).size,
    // Each group stores one fp16 scale.
    bitsPerWeight: bits + (16 * scales.length) / w.length,
  };
}

/** Model weight memory in GB for a parameter count at a given bits-per-weight. */
export function modelGB(params: number, bitsPerWeight: number): number {
  return (params * bitsPerWeight) / 8 / 1e9;
}
