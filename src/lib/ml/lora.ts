// LoRA (Hu et al. 2021): freeze W and learn a low-rank update ΔW = B·A, with B: d×r and A: r×d.

import { frobenius, randn, rng, svd, zeros, type Matrix } from './math';

export const DEMO_D = 12;

/** A "fine-tuning update" whose energy sits in a few directions, as measured in real fine-tunes. */
export function demoUpdate(): Matrix {
  const r = rng(42);
  const strengths = [3, 1.6, 0.8];
  const m = zeros(DEMO_D, DEMO_D);
  for (const s of strengths) {
    const u = unit(Array.from({ length: DEMO_D }, () => randn(r)));
    const v = unit(Array.from({ length: DEMO_D }, () => randn(r)));
    for (let i = 0; i < DEMO_D; i++) for (let j = 0; j < DEMO_D; j++) m[i]![j]! += s * u[i]! * v[j]!;
  }
  for (let i = 0; i < DEMO_D; i++) for (let j = 0; j < DEMO_D; j++) m[i]![j]! += 0.04 * randn(r);
  return m;
}

function unit(v: number[]): number[] {
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
}

export interface LowRankFactors {
  b: Matrix; // d × r
  a: Matrix; // r × d
  product: Matrix; // d × d
  /** ‖ΔW − BA‖ / ‖ΔW‖ */
  relError: number;
}

/** Best rank-r factors (truncated SVD), split as B = U√Σ, A = √Σ Vᵀ. */
export function factor(target: Matrix, rank: number): LowRankFactors {
  const { u, s, v } = svd(target);
  const d = target.length;
  const b = Array.from({ length: d }, (_, i) => Array.from({ length: rank }, (_, k) => u[i]![k]! * Math.sqrt(s[k]!)));
  const a = Array.from({ length: rank }, (_, k) => Array.from({ length: d }, (_, j) => Math.sqrt(s[k]!) * v[j]![k]!));
  const product = target.map((row, i) => row.map((_, j) => {
    let x = 0;
    for (let k = 0; k < rank; k++) x += b[i]![k]! * a[k]![j]!;
    return x;
  }));
  const diff = target.map((row, i) => row.map((x, j) => x - product[i]![j]!));
  return { b, a, product, relError: frobenius(diff) / frobenius(target) };
}

export function singularValues(m: Matrix): number[] {
  return svd(m).s;
}

export interface LlmShape {
  name: string;
  params: number;
  hidden: number;
  kvDim: number;
  ffn: number;
  layers: number;
}

export const LLMS: LlmShape[] = [
  { name: 'Llama 2 7B', params: 6.74e9, hidden: 4096, kvDim: 4096, ffn: 11008, layers: 32 },
  { name: 'Llama 3 8B', params: 8.03e9, hidden: 4096, kvDim: 1024, ffn: 14336, layers: 32 },
  { name: 'Llama 3 70B', params: 70.6e9, hidden: 8192, kvDim: 1024, ffn: 28672, layers: 80 },
];

export type LoraTargets = 'qv' | 'attn' | 'all';

export const TARGETS: Record<LoraTargets, string> = {
  qv: 'q, v projections (original paper)',
  attn: 'all attention projections (q, k, v, o)',
  all: 'attention + MLP (QLoRA-style)',
};

/** Trainable parameters: each adapted d_in × d_out matrix adds r·(d_in + d_out). */
export function loraParams(m: LlmShape, targets: LoraTargets, rank: number): number {
  const { hidden: h, kvDim: kv, ffn } = m;
  const q: [number, number] = [h, h];
  const k: [number, number] = [h, kv];
  const v: [number, number] = [h, kv];
  const o: [number, number] = [h, h];
  const mlp: [number, number][] = [[h, ffn], [h, ffn], [ffn, h]];
  const mats = targets === 'qv' ? [q, v] : targets === 'attn' ? [q, k, v, o] : [q, k, v, o, ...mlp];
  return m.layers * mats.reduce((s, [i, out]) => s + rank * (i + out), 0);
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}
