// k-nearest neighbours: no training at all. To classify a point, find the k stored samples
// closest to it and let them vote. Everything interesting is in what "closest" means.

import { listing, type CodeCursor } from '../code';
import type { LPoint } from './classify';

export type Metric = 'euclidean' | 'manhattan' | 'chebyshev';
export type Weighting = 'uniform' | 'distance';

export const METRICS: { id: Metric; name: string; formula: string }[] = [
  { id: 'euclidean', name: 'Euclidean (L2)', formula: '√(dx² + dy²)' },
  { id: 'manhattan', name: 'Manhattan (L1)', formula: '|dx| + |dy|' },
  { id: 'chebyshev', name: 'Chebyshev (L∞)', formula: 'max(|dx|, |dy|)' },
];

export const KNN_CODE = listing(`
  function predict(train, q, k) {             //@ fn
    // no training step: the data IS the model //@ fn
    const d = train.map(p =>                  //@ dist
      ({ d: dist(p, q), label: p.label }));   //@ dist
    d.sort((a, b) => a.d - b.d);              //@ sort

    const votes = [0, 0, 0];                  //@ take
    for (const n of d.slice(0, k))            //@ take
      votes[n.label] += weighted              //@ take
        ? 1 / n.d : 1;                        //@ take

    return argmax(votes);                     //@ vote
  }
`);

/**
 * Distance with the x difference multiplied by `xScale` first. `xScale` ≠ 1 models a feature
 * measured in bigger units than the other, which is what happens when you forget to standardize.
 */
export function distance(ax: number, ay: number, bx: number, by: number, metric: Metric, xScale = 1): number {
  const dx = Math.abs(ax - bx) * xScale;
  const dy = Math.abs(ay - by);
  if (metric === 'manhattan') return dx + dy;
  if (metric === 'chebyshev') return Math.max(dx, dy);
  return Math.sqrt(dx * dx + dy * dy);
}

export interface Neighbour {
  i: number;
  d: number;
}

/** The k nearest training samples, closest first. Ties go to the lower index, so it's deterministic. */
export function nearestK(train: LPoint[], qx: number, qy: number, k: number, metric: Metric, xScale = 1): Neighbour[] {
  // Insertion into a short sorted list: O(n·k), much cheaper than a full sort for small k.
  const best: Neighbour[] = [];
  for (let i = 0; i < train.length; i++) {
    const p = train[i]!;
    const d = distance(p.x, p.y, qx, qy, metric, xScale);
    if (best.length === k && d >= best[k - 1]!.d) continue;
    let j = best.length < k ? best.length : k - 1;
    if (best.length < k) best.push({ i, d });
    while (j > 0 && best[j - 1]!.d > d) {
      best[j] = best[j - 1]!;
      j--;
    }
    best[j] = { i, d };
  }
  return best;
}

/**
 * Votes per class from a neighbour list. A tiny bonus for the single nearest neighbour's class
 * breaks exact ties the same way everywhere — in the prediction and in the shaded map.
 */
export function votes(train: LPoint[], nbrs: Neighbour[], classes: number, weighting: Weighting): number[] {
  const v = new Array<number>(classes).fill(0);
  for (const n of nbrs) v[train[n.i]!.label]! += weighting === 'distance' ? 1 / Math.max(n.d, 1e-6) : 1;
  if (nbrs.length) v[train[nbrs[0]!.i]!.label]! += 1e-9;
  return v;
}

export interface KnnOptions {
  k: number;
  metric: Metric;
  weighting: Weighting;
  xScale: number;
}

export function knnProba(train: LPoint[], classes: number, o: KnnOptions, x: number, y: number): number[] {
  const v = votes(train, nearestK(train, x, y, o.k, o.metric, o.xScale), classes, o.weighting);
  const s = v.reduce((a, b) => a + b, 0) || 1;
  return v.map((c) => c / s);
}

export function knnPredict(train: LPoint[], classes: number, o: KnnOptions, x: number, y: number): number {
  const v = votes(train, nearestK(train, x, y, o.k, o.metric, o.xScale), classes, o.weighting);
  let best = 0;
  for (let c = 1; c < classes; c++) if (v[c]! > v[best]!) best = c;
  return best;
}

/**
 * Accuracy for every k from 1 to kMax. Training accuracy is scored the naive way, with each
 * point allowed to be its own neighbour — which is why it is always 100% at k = 1.
 */
export function kCurve(train: LPoint[], test: LPoint[], classes: number, o: Omit<KnnOptions, 'k'>, kMax: number) {
  const trainAcc: number[] = [NaN];
  const testAcc: number[] = [NaN];
  const score = (pts: LPoint[], k: number) =>
    pts.filter((p) => knnPredict(train, classes, { ...o, k }, p.x, p.y) === p.label).length / Math.max(1, pts.length);
  for (let k = 1; k <= kMax; k++) {
    trainAcc.push(score(train, k) * 100);
    testAcc.push(score(test, k) * 100);
  }
  return { trainAcc, testAcc };
}

export interface KnnStep {
  phase: 'query' | 'dist' | 'take' | 'vote';
  /** How many of the sorted neighbours have been counted so far. */
  taken: number;
  votes: number[];
  message: string;
  code: CodeCursor;
}

const NAMES = ['A', 'B', 'C'];

/** The query traced one neighbour at a time, so the growing neighbourhood and tally are visible. */
export function knnSteps(train: LPoint[], classes: number, o: KnnOptions, qx: number, qy: number): { steps: KnnStep[]; nbrs: Neighbour[] } {
  const nbrs = nearestK(train, qx, qy, o.k, o.metric, o.xScale);
  const steps: KnnStep[] = [];
  const zero = new Array<number>(classes).fill(0);
  const q = `(${qx.toFixed(0)}, ${qy.toFixed(0)})`;

  steps.push({
    phase: 'query',
    taken: 0,
    votes: zero,
    message: `Classify the point at ${q}. There was no training step: k-NN just stored all ${train.length} samples, and does all its work now, at prediction time.`,
    code: { line: 'fn', vars: { q, k: o.k, n: train.length } },
  });
  steps.push({
    phase: 'dist',
    taken: 0,
    votes: zero,
    message: `Measure the ${METRICS.find((m) => m.id === o.metric)!.name} distance from the query to every stored sample — ${train.length} distances for one prediction. That is the cost of having no model.`,
    code: { line: 'dist', vars: { q, computed: train.length, nearest: +nbrs[0]!.d.toFixed(1) } },
  });

  const tally = [...zero];
  nbrs.forEach((n, idx) => {
    const lab = train[n.i]!.label;
    const w = o.weighting === 'distance' ? 1 / Math.max(n.d, 1e-6) : 1;
    tally[lab]! += w;
    steps.push({
      phase: 'take',
      taken: idx + 1,
      votes: [...tally],
      message:
        idx === 0
          ? `Sorted by distance. The nearest sample is class ${NAMES[lab]}, ${n.d.toFixed(1)} away${o.k === 1 ? ' — with k = 1 it decides alone.' : '.'}`
          : `Neighbour ${idx + 1} of ${o.k}: class ${NAMES[lab]} at distance ${n.d.toFixed(1)}${o.weighting === 'distance' ? `, vote weight 1/d = ${w.toFixed(3)}` : ''}. The neighbourhood grows to include it.`,
      code: {
        line: 'take',
        vars: { i: idx + 1, label: NAMES[lab]!, d: +n.d.toFixed(2), votes: tally.map((t) => (o.weighting === 'distance' ? t.toFixed(2) : t)).join(' / ') },
      },
    });
  });

  const final = votes(train, nbrs, classes, o.weighting);
  let win = 0;
  for (let c = 1; c < classes; c++) if (final[c]! > final[win]!) win = c;
  const sorted = [...final].sort((a, b) => b - a);
  const tie = Math.abs(sorted[0]! - sorted[1]!) < 1e-6;
  steps.push({
    phase: 'vote',
    taken: nbrs.length,
    votes: [...tally],
    message: tie
      ? `A tie. Broken in favour of the single nearest neighbour's class, ${NAMES[win]}. Odd k avoids most ties between two classes.`
      : `Class ${NAMES[win]} wins the vote${o.weighting === 'distance' ? ' (closer neighbours counting for more)' : ''}. Repeat this for every pixel and you get the shaded map.`,
    code: { line: 'vote', vars: { votes: tally.map((t) => (o.weighting === 'distance' ? t.toFixed(2) : t)).join(' / '), predict: NAMES[win]! } },
  });
  return { steps, nbrs };
}
