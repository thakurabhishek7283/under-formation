// A small convolutional network that reads handwritten digits, trained from scratch.
//
//   16×16 image → conv: 8 filters, 3×3, + bias → ReLU → 2×2 max-pool → dense 392 → 10 → softmax
//
// There's no MNIST download: the training set is generated. Each digit is a few pen strokes,
// bent by a random rotation, shear, stretch and wobble, drawn at a random pen width. Drawings
// from the page go through the exact same normalization (crop to the strokes, fit a 12×12 box,
// centre in 16×16) before the network sees them, so train and test inputs look alike.

import { rng, randn } from './math';

export const SIZE = 16;
export const BOX = 12;
export const FILTERS = 8;
export const K = 3;
export const CONV = SIZE - K + 1; // 14
export const POOL = CONV / 2; // 7
export const FLAT = FILTERS * POOL * POOL; // 392
export const DIGITS = 10;

export type Stroke = [number, number][];

// ---------- Synthetic handwriting ----------

/** Points along an ellipse arc; angles in degrees, y pointing down (90° = bottom). */
function arc(cx: number, cy: number, rx: number, ry: number, a0: number, a1: number, steps = 18): Stroke {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = ((a0 + ((a1 - a0) * i) / steps) * Math.PI) / 180;
    return [cx + rx * Math.cos(t), cy + ry * Math.sin(t)] as [number, number];
  });
}

/** Pen strokes for each digit in a unit box, a few common ways of writing each. */
export const TEMPLATES: Stroke[][][] = [
  // 0
  [[arc(0.5, 0.5, 0.32, 0.46, -90, 270)], [arc(0.5, 0.5, 0.26, 0.46, -100, 265)]],
  // 1
  [
    [[[0.5, 0], [0.5, 1]]],
    [[[0.3, 0.22], [0.54, 0], [0.54, 1]]],
    [[[0.3, 0.2], [0.52, 0], [0.52, 1]], [[0.3, 1], [0.74, 1]]],
  ],
  // 2
  [
    [[...arc(0.5, 0.3, 0.32, 0.28, 195, 395), [0.15, 1], [0.87, 1]]],
    [[...arc(0.5, 0.28, 0.3, 0.26, 180, 360), [0.2, 1], [0.85, 0.98]]],
  ],
  // 3
  [
    [[...arc(0.5, 0.27, 0.3, 0.24, 205, 450), ...arc(0.5, 0.74, 0.34, 0.26, 270, 515)]],
    [[[0.2, 0], [0.8, 0], [0.45, 0.42], ...arc(0.48, 0.7, 0.34, 0.29, 250, 515)]],
  ],
  // 4
  [
    [[[0.66, 0], [0.12, 0.68], [0.9, 0.68]], [[0.66, 0], [0.66, 1]]],
    [[[0.22, 0], [0.16, 0.6], [0.88, 0.6]], [[0.7, 0.05], [0.7, 1]]],
  ],
  // 5
  [
    [[[0.82, 0], [0.26, 0], [0.22, 0.45], ...arc(0.47, 0.69, 0.34, 0.3, 235, 515)]],
    [[[0.24, 0], [0.2, 0.46], ...arc(0.47, 0.69, 0.33, 0.3, 235, 515)], [[0.24, 0], [0.84, 0]]],
  ],
  // 6
  [
    [[[0.74, 0], [0.45, 0.14], [0.27, 0.38], [0.18, 0.66], ...arc(0.49, 0.71, 0.31, 0.28, 180, 540)]],
    [[[0.66, 0], [0.3, 0.42], ...arc(0.5, 0.72, 0.3, 0.28, 200, 560)]],
  ],
  // 7
  [[[[0.14, 0], [0.86, 0], [0.38, 1]]], [[[0.14, 0], [0.86, 0], [0.42, 1]], [[0.3, 0.52], [0.76, 0.52]]]],
  // 8
  [
    [arc(0.5, 0.25, 0.26, 0.24, 90, 450), arc(0.5, 0.74, 0.32, 0.26, -90, 270)],
    [[...arc(0.5, 0.25, 0.25, 0.24, 90, 440), ...arc(0.5, 0.74, 0.31, 0.25, 270, -90)]],
  ],
  // 9
  [
    [arc(0.48, 0.3, 0.31, 0.29, 0, 360), [[0.79, 0.3], [0.78, 0.62], [0.66, 1]]],
    [arc(0.46, 0.3, 0.3, 0.29, 0, 360), [[0.76, 0.3], [0.76, 1]]],
  ],
];

/** A random handwritten-looking version of one template. */
export function sampleStrokes(digit: number, r: () => number, wild = 1): Stroke[] {
  const variants = TEMPLATES[digit]!;
  const strokes = variants[Math.floor(r() * variants.length)]!;
  const rot = randn(r) * 0.12 * wild;
  const shear = randn(r) * 0.18 * wild;
  const sx = 1 + randn(r) * 0.12 * wild;
  const sy = 1 + randn(r) * 0.08 * wild;
  // A smooth wobble: a couple of low-frequency sine waves, so strokes bend instead of jitter.
  const ax = randn(r) * 0.025 * wild;
  const ay = randn(r) * 0.025 * wild;
  const fx = 2 + r() * 3;
  const fy = 2 + r() * 3;
  const px = r() * 6.28;
  const py = r() * 6.28;
  const c = Math.cos(rot);
  const s = Math.sin(rot);
  return strokes.map((st) =>
    st.map(([x0, y0]) => {
      const x1 = x0 + ax * Math.sin(fy * y0 * 6.28 + px);
      const y1 = y0 + ay * Math.sin(fx * x0 * 6.28 + py);
      const dx = (x1 - 0.5) * sx + shear * (y1 - 0.5);
      const dy = (y1 - 0.5) * sy;
      return [0.5 + c * dx - s * dy, 0.5 + s * dx + c * dy] as [number, number];
    }),
  );
}

/**
 * Crop to the strokes, scale the longer side to a 12-pixel box, centre in 16×16, and draw with
 * a pen `thickness` pixels wide (anti-aliased). The same function handles generated digits and
 * what you draw on the page.
 */
export function rasterize(strokes: Stroke[], thickness = 1.7): Float32Array {
  const img = new Float32Array(SIZE * SIZE);
  const pts = strokes.flat();
  if (!pts.length) return img;
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const [x, y] of pts) {
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  }
  const span = Math.max(x1 - x0, y1 - y0, 1e-6);
  const scale = BOX / span;
  const ox = SIZE / 2 - ((x0 + x1) / 2) * scale;
  const oy = SIZE / 2 - ((y0 + y1) / 2) * scale;
  const segs: [number, number, number, number][] = [];
  for (const st of strokes) {
    const mapped = st.map(([x, y]) => [x * scale + ox, y * scale + oy] as [number, number]);
    if (mapped.length === 1) segs.push([mapped[0]![0], mapped[0]![1], mapped[0]![0], mapped[0]![1]]);
    for (let i = 1; i < mapped.length; i++) segs.push([mapped[i - 1]![0], mapped[i - 1]![1], mapped[i]![0], mapped[i]![1]]);
  }
  const half = thickness / 2;
  for (let py = 0; py < SIZE; py++)
    for (let px = 0; px < SIZE; px++) {
      const cx = px + 0.5;
      const cy = py + 0.5;
      let d2 = Infinity;
      for (const [ax, ay, bx, by] of segs) {
        const vx = bx - ax;
        const vy = by - ay;
        const len2 = vx * vx + vy * vy;
        const t = len2 > 0 ? Math.max(0, Math.min(1, ((cx - ax) * vx + (cy - ay) * vy) / len2)) : 0;
        const dx = cx - (ax + t * vx);
        const dy = cy - (ay + t * vy);
        d2 = Math.min(d2, dx * dx + dy * dy);
      }
      // Coverage of a pixel by a pen of radius `half`: a one-pixel linear ramp at the edge.
      img[py * SIZE + px] = Math.max(0, Math.min(1, half + 0.5 - Math.sqrt(d2)));
    }
  return img;
}

export interface DigitSet {
  images: Float32Array[];
  labels: number[];
}

export function makeDigits(n: number, seed: number, wild = 1): DigitSet {
  const r = rng(seed);
  const images: Float32Array[] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const d = i % DIGITS;
    images.push(rasterize(sampleStrokes(d, r, wild), 1.2 + r() * 1.1));
    labels.push(d);
  }
  return { images, labels };
}

// ---------- The network ----------

export interface CnnWeights {
  /** FILTERS × 3 × 3 kernels, row-major per filter. */
  k: Float32Array;
  kb: Float32Array;
  /** DIGITS × FLAT dense weights. */
  w: Float32Array;
  b: Float32Array;
}

export interface CnnActivations {
  /** FILTERS × 14 × 14, after ReLU. */
  conv: Float32Array;
  /** FILTERS × 7 × 7. */
  pooled: Float32Array;
  /** For each pooled cell, which conv cell won the max (for backprop). */
  argmax: Int32Array;
  logits: Float32Array;
  probs: Float32Array;
}

export function initCnn(seed: number): CnnWeights {
  const r = rng(seed);
  return {
    k: Float32Array.from({ length: FILTERS * K * K }, () => randn(r) * Math.sqrt(2 / (K * K))),
    kb: new Float32Array(FILTERS).fill(0.01),
    w: Float32Array.from({ length: DIGITS * FLAT }, () => randn(r) * Math.sqrt(1 / FLAT)),
    b: new Float32Array(DIGITS),
  };
}

export function cnnForward(m: CnnWeights, img: Float32Array): CnnActivations {
  const conv = new Float32Array(FILTERS * CONV * CONV);
  for (let f = 0; f < FILTERS; f++) {
    const kb = m.kb[f]!;
    const ko = f * K * K;
    for (let y = 0; y < CONV; y++)
      for (let x = 0; x < CONV; x++) {
        // Cross-correlation, as every deep-learning library computes "convolution".
        let s = kb;
        for (let ky = 0; ky < K; ky++)
          for (let kx = 0; kx < K; kx++) s += m.k[ko + ky * K + kx]! * img[(y + ky) * SIZE + x + kx]!;
        conv[(f * CONV + y) * CONV + x] = s > 0 ? s : 0;
      }
  }
  const pooled = new Float32Array(FLAT);
  const argmax = new Int32Array(FLAT);
  for (let f = 0; f < FILTERS; f++)
    for (let y = 0; y < POOL; y++)
      for (let x = 0; x < POOL; x++) {
        let best = -Infinity;
        let at = 0;
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            const i = (f * CONV + 2 * y + dy) * CONV + 2 * x + dx;
            if (conv[i]! > best) {
              best = conv[i]!;
              at = i;
            }
          }
        const o = (f * POOL + y) * POOL + x;
        pooled[o] = best;
        argmax[o] = at;
      }
  const logits = new Float32Array(DIGITS);
  let max = -Infinity;
  for (let c = 0; c < DIGITS; c++) {
    let s = m.b[c]!;
    const wo = c * FLAT;
    for (let j = 0; j < FLAT; j++) s += m.w[wo + j]! * pooled[j]!;
    logits[c] = s;
    max = Math.max(max, s);
  }
  const probs = new Float32Array(DIGITS);
  let sum = 0;
  for (let c = 0; c < DIGITS; c++) {
    probs[c] = Math.exp(logits[c]! - max);
    sum += probs[c]!;
  }
  for (let c = 0; c < DIGITS; c++) probs[c]! /= sum;
  return { conv, pooled, argmax, logits, probs };
}

export function cnnPredict(m: CnnWeights, img: Float32Array): number {
  const p = cnnForward(m, img).probs;
  let best = 0;
  for (let c = 1; c < DIGITS; c++) if (p[c]! > p[best]!) best = c;
  return best;
}

export interface CnnGrads {
  k: Float64Array;
  kb: Float64Array;
  w: Float64Array;
  b: Float64Array;
}

export const zeroGrads = (): CnnGrads => ({
  k: new Float64Array(FILTERS * K * K),
  kb: new Float64Array(FILTERS),
  w: new Float64Array(DIGITS * FLAT),
  b: new Float64Array(DIGITS),
});

/** Backpropagate one example's cross-entropy loss into `g`. Returns the loss. */
export function cnnBackprop(m: CnnWeights, img: Float32Array, label: number, g: CnnGrads): number {
  const a = cnnForward(m, img);
  const dLogits = Float64Array.from(a.probs, (p, c) => p - (c === label ? 1 : 0));
  const dPooled = new Float64Array(FLAT);
  for (let c = 0; c < DIGITS; c++) {
    const d = dLogits[c]!;
    g.b[c]! += d;
    const wo = c * FLAT;
    for (let j = 0; j < FLAT; j++) {
      g.w[wo + j]! += d * a.pooled[j]!;
      dPooled[j]! += m.w[wo + j]! * d;
    }
  }
  // Max-pool passes the gradient only to the cell that won; ReLU blocks it where the output was 0.
  const dConv = new Float64Array(FILTERS * CONV * CONV);
  for (let j = 0; j < FLAT; j++) {
    const i = a.argmax[j]!;
    if (a.conv[i]! > 0) dConv[i]! += dPooled[j]!;
  }
  for (let f = 0; f < FILTERS; f++) {
    const ko = f * K * K;
    for (let y = 0; y < CONV; y++)
      for (let x = 0; x < CONV; x++) {
        const d = dConv[(f * CONV + y) * CONV + x]!;
        if (d === 0) continue;
        g.kb[f]! += d;
        for (let ky = 0; ky < K; ky++)
          for (let kx = 0; kx < K; kx++) g.k[ko + ky * K + kx]! += d * img[(y + ky) * SIZE + x + kx]!;
      }
  }
  return -Math.log(Math.max(a.probs[label]!, 1e-12));
}

export interface CnnTrainer {
  m: CnnWeights;
  lr: number;
  t: number;
  mom: CnnGrads;
  vel: CnnGrads;
  r: () => number;
}

export const cnnTrainer = (m: CnnWeights, lr: number, seed: number): CnnTrainer => ({ m, lr, t: 0, mom: zeroGrads(), vel: zeroGrads(), r: rng(seed) });

/** One Adam step on a mini-batch. Returns the batch's mean loss. */
export function cnnStep(tr: CnnTrainer, data: DigitSet, idx: number[]): number {
  const g = zeroGrads();
  let loss = 0;
  for (const i of idx) loss += cnnBackprop(tr.m, data.images[i]!, data.labels[i]!, g);
  tr.t++;
  const b1 = 0.9;
  const b2 = 0.999;
  const c1 = 1 - b1 ** tr.t;
  const c2 = 1 - b2 ** tr.t;
  const n = idx.length;
  const upd = (p: Float32Array, gr: Float64Array, mm: Float64Array, vv: Float64Array) => {
    for (let i = 0; i < p.length; i++) {
      const gi = gr[i]! / n;
      mm[i] = b1 * mm[i]! + (1 - b1) * gi;
      vv[i] = b2 * vv[i]! + (1 - b2) * gi * gi;
      p[i]! -= (tr.lr * (mm[i]! / c1)) / (Math.sqrt(vv[i]! / c2) + 1e-8);
    }
  };
  upd(tr.m.k, g.k, tr.mom.k, tr.vel.k);
  upd(tr.m.kb, g.kb, tr.mom.kb, tr.vel.kb);
  upd(tr.m.w, g.w, tr.mom.w, tr.vel.w);
  upd(tr.m.b, g.b, tr.mom.b, tr.vel.b);
  return loss / n;
}

/** A shuffled index order for one epoch. */
export function epochOrder(tr: CnnTrainer, n: number): number[] {
  const o = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(tr.r() * (i + 1));
    [o[i], o[j]] = [o[j]!, o[i]!];
  }
  return o;
}

export function cnnAccuracy(m: CnnWeights, data: DigitSet): number {
  let ok = 0;
  data.images.forEach((img, i) => {
    if (cnnPredict(m, img) === data.labels[i]) ok++;
  });
  return ok / Math.max(data.images.length, 1);
}

export const CNN_PARAMS = FILTERS * K * K + FILTERS + DIGITS * FLAT + DIGITS;

/** Weights as plain, rounded arrays for a JSON file. */
export function serialize(m: CnnWeights) {
  const round = (a: Float32Array) => Array.from(a, (v) => Math.round(v * 1e4) / 1e4);
  return { k: round(m.k), kb: round(m.kb), w: round(m.w), b: round(m.b) };
}

export function deserialize(j: { k: number[]; kb: number[]; w: number[]; b: number[] }): CnnWeights {
  return { k: Float32Array.from(j.k), kb: Float32Array.from(j.kb), w: Float32Array.from(j.w), b: Float32Array.from(j.b) };
}

/** How the shipped weights were trained (scripts/train-cnn.mjs), and how the page retrains. */
export const RECIPE = { samples: 8000, epochs: 5, batch: 16, lr: 0.003, wild: 1.4, dataSeed: 11, initSeed: 3, orderSeed: 5 };

/** The held-out set every accuracy on the page is measured on: other seeds, never trained on. */
export const TEST_SET = { samples: 500, seed: 777, wild: 1.3 };
