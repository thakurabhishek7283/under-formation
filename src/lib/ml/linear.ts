// Two linear classifiers that learn the same kind of boundary, w·φ(x) = 0, in different ways:
//
// - The perceptron (Rosenblatt, 1958) looks at one sample at a time and, only when it gets one
//   wrong, adds that sample to the weights. It stops the moment nothing is wrong — and so never
//   stops if the classes overlap.
// - Logistic regression turns the same score into a probability with the sigmoid and follows the
//   gradient of the log loss, so every sample pulls on the weights, correct ones included.
//
// φ is the feature map. With 'quadratic' the model is still linear in w, but the boundary drawn
// in the plane is a conic — the whole trick behind kernels and hidden layers, done by hand.

import { listing, type CodeCursor } from '../code';
import { toUnit, type LPoint } from './classify';

export type FeatureMap = 'linear' | 'quadratic';
export type LinearMethod = 'logistic' | 'perceptron';

export const FEATURE_NAMES: Record<FeatureMap, string[]> = {
  linear: ['1', 'x', 'y'],
  quadratic: ['1', 'x', 'y', 'x²', 'y²', 'xy'],
};

export function phi(x: number, y: number, map: FeatureMap): number[] {
  const u = toUnit(x);
  const v = toUnit(y);
  return map === 'linear' ? [1, u, v] : [1, u, v, u * u, v * v, u * v];
}

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

export const score = (w: number[], x: number, y: number, map: FeatureMap) => {
  const f = phi(x, y, map);
  let s = 0;
  for (let i = 0; i < w.length; i++) s += w[i]! * f[i]!;
  return s;
};

export const LOGISTIC_CODE = listing(`
  function train(X, y, lr, epochs) {          //@ fn
    let w = zeros(X[0].length);               //@ init
    for (let e = 0; e < epochs; e++) {        //@ loop
      const p = X.map(x =>                    //@ fwd epoch
        sigmoid(dot(w, x)));  // P(class B)   //@ fwd epoch
      // gradient of the mean log loss        //@ grad epoch
      const g = Xt(X) * (p - y) / n;          //@ grad epoch
      w = w - lr * (g + lambda * w);          //@ upd epoch
    }
    return w;                                 //@ done
  }
`);

export const PERCEPTRON_CODE = listing(`
  function perceptron(X, y) {  // y is ±1     //@ fn
    let w = zeros(X[0].length);               //@ init
    for (let epoch = 1; ; epoch++) {          //@ loop
      let mistakes = 0;                       //@ loop
      for (let i = 0; i < X.length; i++)      //@ scan
        if (y[i] * dot(w, X[i]) <= 0) {       //@ wrong
          w = w + y[i] * X[i]; // nudge       //@ upd
          mistakes++;                         //@ upd
        }
      if (mistakes === 0) return w;           //@ done
    }
  }
`);

export interface LinearStep {
  w: number[];
  /** Weights before this step's update, to draw where the boundary came from. */
  prev?: number[];
  /** Sample being corrected (perceptron). */
  focus?: number;
  epoch: number;
  phase: 'init' | 'epoch' | 'update' | 'done';
  loss: number;
  trainAcc: number;
  testAcc: number;
  /** Per completed epoch: log loss (logistic) or mistakes (perceptron). */
  history: number[];
  message: string;
  code: CodeCursor;
}

export interface LinearOptions {
  map: FeatureMap;
  lr: number;
  lambda: number;
  epochs: number;
}

export function predictLinear(w: number[], map: FeatureMap, x: number, y: number): number {
  return score(w, x, y, map) > 0 ? 1 : 0;
}

function acc(w: number[], map: FeatureMap, pts: LPoint[]): number {
  if (!pts.length) return 0;
  return pts.filter((p) => predictLinear(w, map, p.x, p.y) === p.label).length / pts.length;
}

/** Mean log loss; the probability is clipped so a confident mistake costs a lot but not ∞. */
export function logLoss(w: number[], map: FeatureMap, pts: LPoint[]): number {
  let s = 0;
  for (const p of pts) {
    const q = Math.min(1 - 1e-12, Math.max(1e-12, sigmoid(score(w, p.x, p.y, map))));
    s -= p.label ? Math.log(q) : Math.log(1 - q);
  }
  return s / Math.max(1, pts.length);
}

const fmtW = (w: number[]) => `[${w.map((v) => v.toFixed(2)).join(', ')}]`;

/** Full-batch gradient descent on the log loss, one step per epoch. */
export function logisticSteps(train: LPoint[], test: LPoint[], o: LinearOptions): LinearStep[] {
  const X = train.map((p) => phi(p.x, p.y, o.map));
  const d = X[0]!.length;
  const n = train.length;
  let w = new Array<number>(d).fill(0);
  const history: number[] = [];
  const steps: LinearStep[] = [];
  const loss0 = logLoss(w, o.map, train);

  steps.push({
    w: [...w],
    epoch: 0,
    phase: 'init',
    loss: loss0,
    trainAcc: acc(w, o.map, train),
    testAcc: acc(w, o.map, test),
    history: [],
    message: `All weights start at zero, so every point gets P(B) = σ(0) = 0.5 — a log loss of ln 2 ≈ ${loss0.toFixed(3)}, the cost of knowing nothing.`,
    code: { line: 'init', vars: { w: fmtW(w), features: d, n } },
  });

  for (let e = 1; e <= o.epochs; e++) {
    const g = new Array<number>(d).fill(0);
    for (let i = 0; i < n; i++) {
      const err = sigmoid(X[i]!.reduce((s, v, j) => s + v * w[j]!, 0)) - train[i]!.label;
      for (let j = 0; j < d; j++) g[j]! += (err * X[i]![j]!) / n;
    }
    // The bias is not penalized: shrinking it only drags every prediction towards 50%.
    const prev = w;
    w = w.map((v, j) => v - o.lr * (g[j]! + (j === 0 ? 0 : o.lambda * v)));
    const loss = logLoss(w, o.map, train);
    history.push(loss);
    const gnorm = Math.sqrt(g.reduce((s, v) => s + v * v, 0));
    steps.push({
      w: [...w],
      prev,
      epoch: e,
      phase: e === o.epochs ? 'done' : 'epoch',
      loss,
      trainAcc: acc(w, o.map, train),
      testAcc: acc(w, o.map, test),
      history: [...history],
      message:
        e === 1
          ? `Epoch 1: every sample pulls on w in proportion to its error p − y. Confidently wrong points pull hardest; correct ones barely pull at all. Loss ${loss.toFixed(3)}.`
          : e === o.epochs
            ? `Done after ${e} epochs. Loss ${loss.toFixed(3)}, gradient norm ${gnorm.toFixed(4)}. The weights keep growing on separable data — see why below.`
            : `Epoch ${e}: loss ${loss.toFixed(3)}, gradient norm ${gnorm.toFixed(3)}. The line moves less each step as the gradient shrinks.`,
      code: {
        line: e === o.epochs ? 'done' : 'epoch',
        vars: { epoch: e, loss: +loss.toFixed(4), '|g|': +gnorm.toFixed(4), w: fmtW(w) },
      },
    });
  }
  return steps;
}

/**
 * Rosenblatt's rule on ±1 labels, visiting samples in order. One step per correction, plus one
 * at the end of each epoch; capped so a non-separable set produces a finite trace.
 */
export function perceptronSteps(train: LPoint[], test: LPoint[], o: LinearOptions, maxUpdates = 160): LinearStep[] {
  const X = train.map((p) => phi(p.x, p.y, o.map));
  const Y = train.map((p) => (p.label ? 1 : -1));
  const d = X[0]!.length;
  let w = new Array<number>(d).fill(0);
  const history: number[] = [];
  const steps: LinearStep[] = [];
  let updates = 0;

  const base = () => ({ trainAcc: acc(w, o.map, train), testAcc: acc(w, o.map, test), loss: 0, history: [...history] });

  steps.push({
    ...base(),
    w: [...w],
    epoch: 0,
    phase: 'init',
    message: `Start with w = 0. Every score is 0, which the rule counts as wrong, so the very first sample will trigger an update.`,
    code: { line: 'init', vars: { w: fmtW(w) } },
  });

  for (let epoch = 1; epoch <= o.epochs && updates < maxUpdates; epoch++) {
    let mistakes = 0;
    for (let i = 0; i < X.length && updates < maxUpdates; i++) {
      const s = X[i]!.reduce((acc2, v, j) => acc2 + v * w[j]!, 0);
      if (Y[i]! * s > 0) continue;
      const prev = w;
      w = w.map((v, j) => v + Y[i]! * X[i]![j]!);
      mistakes++;
      updates++;
      steps.push({
        ...base(),
        w: [...w],
        prev,
        focus: i,
        epoch,
        phase: 'update',
        message: `Sample ${i} is class ${train[i]!.label ? 'B' : 'A'} but scored ${s.toFixed(2)} — on the wrong side. Add ${Y[i]! > 0 ? '+' : '−'}φ(x) to w: the line swings towards classifying it correctly.`,
        code: { line: 'upd', vars: { epoch, i, 'y·score': +(Y[i]! * s).toFixed(3), w: fmtW(w), updates } },
      });
    }
    history.push(mistakes);
    const done = mistakes === 0;
    steps.push({
      ...base(),
      w: [...w],
      epoch,
      phase: done ? 'done' : 'epoch',
      message: done
        ? `A full pass with no mistakes: every training point is on its correct side, so the perceptron stops. It has no reason to prefer this line over any other that separates the data.`
        : updates >= maxUpdates
          ? `Stopped after ${updates} updates, still making mistakes. When no line separates the classes the perceptron never settles — it cycles for ever.`
          : `End of epoch ${epoch}: ${mistakes} mistake${mistakes === 1 ? '' : 's'}. Go round again.`,
      code: { line: done ? 'done' : 'loop', vars: { epoch, mistakes, updates } },
    });
    if (done) break;
  }
  return steps;
}

/** The segment of the line w0 + w1·u + w2·v = 0 inside the plot, in plot units, or null. */
export function lineInBox(w: number[]): [number, number, number, number] | null {
  const [w0, w1, w2] = w as [number, number, number];
  // Back to plot coordinates: u = (x − 50)/25, so w1·(x − 50)/25 + w2·(y − 50)/25 + w0 = 0.
  const a = w1 / 25;
  const b = w2 / 25;
  const c = w0 - 2 * w1 - 2 * w2;
  const pts: [number, number][] = [];
  if (Math.abs(b) > 1e-12)
    for (const x of [0, 100]) {
      const y = -(a * x + c) / b;
      if (y >= 0 && y <= 100) pts.push([x, y]);
    }
  if (Math.abs(a) > 1e-12)
    for (const y of [0, 100]) {
      const x = -(b * y + c) / a;
      if (x > 0 && x < 100) pts.push([x, y]);
    }
  if (pts.length < 2) return null;
  return [pts[0]![0], pts[0]![1], pts[1]![0], pts[1]![1]];
}
