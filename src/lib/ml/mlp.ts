// A small fully connected neural network (multi-layer perceptron) for 2-D classification,
// trained from scratch with backpropagation and Adam.
//
//   input (x, y) → [hidden layer: W·a + b, then a non-linearity] × L → softmax over classes
//
// Each hidden neuron computes one soft linear cut of its input and bends it with the
// activation. The first layer can only draw straight lines; every later layer combines the
// previous layer's regions into more complicated ones. With no hidden layer at all, this is
// exactly (multinomial) logistic regression.

import { rng, randn } from './math';
import { toUnit, type LPoint } from './classify';

export type Activation = 'tanh' | 'relu' | 'sigmoid';

export interface Net {
  /** Layer widths, input first, output last: e.g. [2, 6, 6, 2]. */
  sizes: number[];
  act: Activation;
  /** W[l] maps layer l to layer l + 1, row-major (out × in). */
  W: Float64Array[];
  b: Float64Array[];
}

export function initNet(sizes: number[], act: Activation, seed: number): Net {
  const r = rng(seed);
  const W: Float64Array[] = [];
  const b: Float64Array[] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    const fanIn = sizes[l]!;
    const fanOut = sizes[l + 1]!;
    // He init for ReLU, Glorot for the saturating ones: keeps activations from exploding or dying at the start.
    const scale = act === 'relu' ? Math.sqrt(2 / fanIn) : Math.sqrt(2 / (fanIn + fanOut));
    W.push(Float64Array.from({ length: fanIn * fanOut }, () => randn(r) * scale));
    b.push(new Float64Array(fanOut).fill(act === 'relu' ? 0.1 : 0));
  }
  return { sizes: [...sizes], act, W, b };
}

function activate(z: number, act: Activation): number {
  if (act === 'relu') return z > 0 ? z : 0;
  if (act === 'sigmoid') return 1 / (1 + Math.exp(-z));
  return Math.tanh(z);
}

/** Derivative written in terms of the activation's output a. */
function activateGrad(a: number, act: Activation): number {
  if (act === 'relu') return a > 0 ? 1 : 0;
  if (act === 'sigmoid') return a * (1 - a);
  return 1 - a * a;
}

/** Every layer's output for one input, in plot coordinates. The last entry is the softmax. */
export function forward(net: Net, x: number, y: number): Float64Array[] {
  const acts: Float64Array[] = [Float64Array.of(toUnit(x), toUnit(y))];
  const L = net.W.length;
  for (let l = 0; l < L; l++) {
    const inp = acts[l]!;
    const nIn = net.sizes[l]!;
    const nOut = net.sizes[l + 1]!;
    const W = net.W[l]!;
    const out = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let z = net.b[l]![o]!;
      for (let i = 0; i < nIn; i++) z += W[o * nIn + i]! * inp[i]!;
      out[o] = z;
    }
    if (l < L - 1) for (let o = 0; o < nOut; o++) out[o] = activate(out[o]!, net.act);
    else {
      let max = -Infinity;
      for (let o = 0; o < nOut; o++) max = Math.max(max, out[o]!);
      let s = 0;
      for (let o = 0; o < nOut; o++) {
        out[o] = Math.exp(out[o]! - max);
        s += out[o]!;
      }
      for (let o = 0; o < nOut; o++) out[o]! /= s;
    }
    acts.push(out);
  }
  return acts;
}

export const predictProba = (net: Net, x: number, y: number) => forward(net, x, y).at(-1)!;

export function evaluate(net: Net, pts: LPoint[]): { loss: number; acc: number } {
  let loss = 0;
  let ok = 0;
  for (const p of pts) {
    const q = predictProba(net, p.x, p.y);
    loss -= Math.log(Math.max(q[p.label]!, 1e-12));
    let best = 0;
    for (let c = 1; c < q.length; c++) if (q[c]! > q[best]!) best = c;
    if (best === p.label) ok++;
  }
  const n = Math.max(pts.length, 1);
  return { loss: loss / n, acc: ok / n };
}

export interface Trainer {
  net: Net;
  lr: number;
  batch: number;
  /** Adam moment estimates, laid out like W and b. */
  mW: Float64Array[];
  vW: Float64Array[];
  mb: Float64Array[];
  vb: Float64Array[];
  t: number;
  epoch: number;
  r: () => number;
}

export function makeTrainer(net: Net, lr: number, seed: number, batch = 10): Trainer {
  return {
    net,
    lr,
    batch,
    mW: net.W.map((w) => new Float64Array(w.length)),
    vW: net.W.map((w) => new Float64Array(w.length)),
    mb: net.b.map((b) => new Float64Array(b.length)),
    vb: net.b.map((b) => new Float64Array(b.length)),
    t: 0,
    epoch: 0,
    r: rng(seed),
  };
}

/** Add the summed cross-entropy gradient over `pts` into gW and gb (same layout as W and b). */
export function backprop(net: Net, pts: LPoint[], gW: Float64Array[], gb: Float64Array[]): void {
  const L = net.W.length;
  for (const p of pts) {
    const acts = forward(net, p.x, p.y);
    // Softmax + cross-entropy: the gradient at the logits is simply p − onehot(label).
    let delta = Float64Array.from(acts[L]!, (v, c) => v - (c === p.label ? 1 : 0));
    for (let l = L - 1; l >= 0; l--) {
      const inp = acts[l]!;
      const nIn = net.sizes[l]!;
      const nOut = net.sizes[l + 1]!;
      const W = net.W[l]!;
      for (let o = 0; o < nOut; o++) {
        gb[l]![o]! += delta[o]!;
        for (let i = 0; i < nIn; i++) gW[l]![o * nIn + i]! += delta[o]! * inp[i]!;
      }
      if (l > 0) {
        const prev = new Float64Array(nIn);
        for (let i = 0; i < nIn; i++) {
          let s = 0;
          for (let o = 0; o < nOut; o++) s += W[o * nIn + i]! * delta[o]!;
          prev[i] = s * activateGrad(inp[i]!, net.act);
        }
        delta = prev;
      }
    }
  }
}

/** One pass over the data in shuffled mini-batches, updating the net in place. */
export function trainEpoch(tr: Trainer, data: LPoint[]): void {
  const { net } = tr;
  const L = net.W.length;
  const order = data.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(tr.r() * (i + 1));
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  const gW = net.W.map((w) => new Float64Array(w.length));
  const gb = net.b.map((b) => new Float64Array(b.length));

  for (let start = 0; start < order.length; start += tr.batch) {
    const end = Math.min(start + tr.batch, order.length);
    for (const g of gW) g.fill(0);
    for (const g of gb) g.fill(0);

    backprop(net, order.slice(start, end).map((i) => data[i]!), gW, gb);

    // Adam: per-weight step sizes from running averages of the gradient and its square.
    tr.t++;
    const m = end - start;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - b1 ** tr.t;
    const c2 = 1 - b2 ** tr.t;
    const upd = (param: Float64Array, g: Float64Array, mm: Float64Array, vv: Float64Array) => {
      for (let i = 0; i < param.length; i++) {
        const gi = g[i]! / m;
        mm[i] = b1 * mm[i]! + (1 - b1) * gi;
        vv[i] = b2 * vv[i]! + (1 - b2) * gi * gi;
        param[i]! -= (tr.lr * (mm[i]! / c1)) / (Math.sqrt(vv[i]! / c2) + 1e-8);
      }
    };
    for (let l = 0; l < L; l++) {
      upd(net.W[l]!, gW[l]!, tr.mW[l]!, tr.vW[l]!);
      upd(net.b[l]!, gb[l]!, tr.mb[l]!, tr.vb[l]!);
    }
  }
  tr.epoch++;
}

export const paramCount = (sizes: number[]) => sizes.slice(1).reduce((s, n, l) => s + n * (sizes[l]! + 1), 0);
