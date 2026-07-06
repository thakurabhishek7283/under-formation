// KV caching during autoregressive decoding: keys and values of past tokens never change,
// so they're computed once and reused, instead of re-running the whole prefix every step.

import { dot, randn, rng, softmax, type Matrix } from './math';

export const KV_DIM = 6;

export interface KvStep {
  phase: 'prefill' | 'decode';
  /** Tokens visible so far (prompt + generated). */
  tokens: string[];
  /** Rows of K and V (one per visible token). */
  k: Matrix;
  v: Matrix;
  /** Rows whose K/V were computed during this step. */
  computed: number[];
  /** Attention weights of the newest token over all visible tokens. */
  attention: number[];
  /** Cumulative K/V row computations so far. */
  work: number;
  message: string;
}

// Deterministic per (token, position) vectors: a stand-in for x·W_k and x·W_v of a real model.
function vec(token: string, pos: number, salt: number): number[] {
  let h = salt;
  for (const ch of token) h = Math.imul(h ^ ch.charCodeAt(0), 0x5bd1e995) >>> 0;
  const r = rng(h + pos * 7919);
  return Array.from({ length: KV_DIM }, () => Math.round(randn(r) * 100) / 100);
}

export function generationTrace(prompt: string[], generated: string[], useCache: boolean): KvStep[] {
  const steps: KvStep[] = [];
  const all = [...prompt, ...generated];
  let work = 0;

  const attend = (t: number, k: Matrix) => {
    const q = vec(all[t - 1]!, t - 1, 3);
    return softmax(k.map((row) => dot(q, row) / Math.sqrt(KV_DIM)));
  };

  for (let t = prompt.length; t <= all.length; t++) {
    const tokens = all.slice(0, t);
    const k = tokens.map((tok, i) => vec(tok, i, 1));
    const v = tokens.map((tok, i) => vec(tok, i, 2));
    const prefill = t === prompt.length;
    const computed = prefill || !useCache ? tokens.map((_, i) => i) : [t - 1];
    work += computed.length;
    const newest = tokens[t - 1]!;
    let message: string;
    if (prefill) {
      message = `Prefill: the whole ${t}-token prompt goes through the model in one parallel pass, computing K and V for every position${useCache ? ' and storing them in the cache' : ''}.`;
    } else if (useCache) {
      message = `Decode "${newest}": compute K and V for this one new token, append them to the cache, and attend over all ${t} cached rows. ${t - 1} rows reused.`;
    } else {
      message = `Decode "${newest}" without a cache: recompute K and V for all ${t} tokens, even though the first ${t - 1} are identical to last step.`;
    }
    steps.push({ phase: prefill ? 'prefill' : 'decode', tokens, k, v, computed, attention: attend(t, k), work, message });
  }
  return steps;
}

export interface ModelConfig {
  name: string;
  layers: number;
  heads: number;
  kvHeads: number;
  headDim: number;
}

export const MODELS: ModelConfig[] = [
  { name: 'GPT-2 small (124M)', layers: 12, heads: 12, kvHeads: 12, headDim: 64 },
  { name: 'Llama 2 7B', layers: 32, heads: 32, kvHeads: 32, headDim: 128 },
  { name: 'Llama 3 8B (GQA)', layers: 32, heads: 32, kvHeads: 8, headDim: 128 },
  { name: 'Mistral 7B (GQA)', layers: 32, heads: 32, kvHeads: 8, headDim: 128 },
  { name: 'Llama 3 70B (GQA)', layers: 80, heads: 64, kvHeads: 8, headDim: 128 },
];

/** Bytes of KV cache: 2 (K and V) × layers × kv_heads × head_dim × bytes × tokens × batch. */
export function kvCacheBytes(m: ModelConfig, tokens: number, batch: number, bytesPerValue: number): number {
  return 2 * m.layers * m.kvHeads * m.headDim * bytesPerValue * tokens * batch;
}

export function formatBytes(b: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (b >= 1024 && i < units.length - 1) {
    b /= 1024;
    i++;
  }
  return `${b < 10 ? b.toFixed(2) : b < 100 ? b.toFixed(1) : Math.round(b)} ${units[i]}`;
}
