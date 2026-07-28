// A classification tree grown the standard greedy way (CART), traced node by node.
//
// At every node the tree tries every axis-aligned threshold, scores each by how much it lowers
// impurity, and takes the best one. That is the whole learning algorithm: there is no gradient
// and nothing global — just a sequence of locally best cuts.

import { listing, type CodeCursor } from '../code';
import { rng, randn } from './math';

export interface Sample {
  x: number;
  y: number;
  /** Class index, 0 or 1. */
  label: number;
}

export type Criterion = 'gini' | 'entropy';
export type TreeDataset = 'blobs' | 'xor' | 'ring' | 'diagonal';

export const TREE_DATASETS: { id: TreeDataset; name: string; blurb: string }[] = [
  { id: 'blobs', name: 'Two blobs', blurb: 'Nearly separable by one cut. Depth 1 gets most of it.' },
  { id: 'xor', name: 'XOR', blurb: 'No single cut helps at all — the first split looks useless, and the second pays it off.' },
  { id: 'ring', name: 'Ring', blurb: 'A curved boundary approximated by a staircase of axis-aligned cuts.' },
  { id: 'diagonal', name: 'Diagonal', blurb: 'The worst case for axis-aligned splits: a 45° boundary needs many steps.' },
];

export interface Bounds {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

export interface Split {
  /** 0 = split on x, 1 = split on y. */
  feature: 0 | 1;
  threshold: number;
  gain: number;
}

export interface TreeNode {
  id: number;
  parent: number | null;
  depth: number;
  /** The slab of feature space this node is responsible for. */
  bounds: Bounds;
  indices: number[];
  /** Sample count per class. */
  counts: [number, number];
  impurity: number;
  prediction: number;
  split: Split | null;
  left: number | null;
  right: number | null;
  leaf: boolean;
}

export interface TreeStep {
  nodes: TreeNode[];
  current: number | null;
  phase: 'node' | 'scan' | 'split' | 'leaf' | 'done';
  /** Every threshold tried at this node, for the gain curve. */
  candidates?: Split[];
  best?: Split;
  trainAcc: number;
  testAcc: number;
  leaves: number;
  message: string;
  code: CodeCursor;
}

export const TREE_CODE = listing(`
  function grow(rows, depth) {                //@ fn
    if (pure(rows) || depth === maxDepth      //@ stop
        || rows.length < minSamples)          //@ stop
      return leaf(majorityClass(rows));       //@ leaf

    let best = null;                          //@ scan
    for (const f of [x, y])       // features //@ scan
      for (const t of thresholds(rows, f)) {  //@ scan
        const g = gain(rows, f, t);           //@ scan
        if (g > best.gain) best = { f, t, g };//@ scan
      }

    if (best.gain <= 0) return leaf(rows);    //@ leaf
    const [L, R] = partition(rows, best);     //@ split
    return node(best,                         //@ split
      grow(L, depth + 1),                     //@ split
      grow(R, depth + 1));                    //@ split
  }
`);

/** Gini impurity: the chance of mislabelling a random sample using the node's class distribution. */
export function gini(counts: [number, number]): number {
  const n = counts[0] + counts[1];
  if (n === 0) return 0;
  const p0 = counts[0] / n;
  const p1 = counts[1] / n;
  return 1 - p0 * p0 - p1 * p1;
}

/** Shannon entropy in bits. 0 for a pure node, 1 for a 50/50 split. */
export function entropy(counts: [number, number]): number {
  const n = counts[0] + counts[1];
  if (n === 0) return 0;
  let h = 0;
  for (const c of counts) {
    if (c === 0) continue;
    const p = c / n;
    h -= p * Math.log2(p);
  }
  return h;
}

export const impurityOf = (counts: [number, number], criterion: Criterion) => (criterion === 'gini' ? gini(counts) : entropy(counts));

function countsOf(rows: Sample[], indices: number[]): [number, number] {
  const c: [number, number] = [0, 0];
  for (const i of indices) c[rows[i]!.label as 0 | 1]++;
  return c;
}

/** Deterministic labelled data in [0, 100]². */
export function makeTreeData(kind: TreeDataset, n: number, seed: number, noise = 0.06): Sample[] {
  const r = rng(seed);
  const out: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const x = 4 + r() * 92;
    const y = 4 + r() * 92;
    let label: number;
    switch (kind) {
      case 'xor':
        label = x > 50 === y > 50 ? 1 : 0;
        break;
      case 'ring':
        label = (x - 50) ** 2 + (y - 50) ** 2 < 30 * 30 ? 1 : 0;
        break;
      case 'diagonal':
        label = y > x ? 1 : 0;
        break;
      default:
        label = (x - 32) ** 2 + (y - 68) ** 2 < (x - 70) ** 2 + (y - 32) ** 2 ? 1 : 0;
    }
    // Label noise: what makes a deep tree memorize instead of generalize.
    if (r() < noise) label = 1 - label;
    out.push({ x, y, label });
  }
  // Jitter blobs so they aren't uniform over the square.
  if (kind === 'blobs') {
    return out.map((s) => ({
      x: Math.max(3, Math.min(97, (s.label ? 32 : 70) + randn(r) * 13)),
      y: Math.max(3, Math.min(97, (s.label ? 68 : 32) + randn(r) * 13)),
      label: s.label,
    }));
  }
  return out;
}

const featureOf = (s: Sample, f: 0 | 1) => (f === 0 ? s.x : s.y);

/** Midpoints between consecutive distinct values, capped so the gain curve stays readable. */
function thresholds(rows: Sample[], indices: number[], f: 0 | 1, cap = 48): number[] {
  const vs = [...new Set(indices.map((i) => featureOf(rows[i]!, f)))].sort((a, b) => a - b);
  const mids: number[] = [];
  for (let i = 1; i < vs.length; i++) mids.push((vs[i - 1]! + vs[i]!) / 2);
  if (mids.length <= cap) return mids;
  const stride = mids.length / cap;
  return Array.from({ length: cap }, (_, i) => mids[Math.floor(i * stride)]!);
}

/** Impurity drop from a split, weighted by how many samples go each way. */
function gainOf(rows: Sample[], indices: number[], f: 0 | 1, t: number, parent: number, criterion: Criterion): number {
  const l: number[] = [];
  const r: number[] = [];
  for (const i of indices) (featureOf(rows[i]!, f) <= t ? l : r).push(i);
  if (!l.length || !r.length) return 0;
  const n = indices.length;
  const il = impurityOf(countsOf(rows, l), criterion);
  const ir = impurityOf(countsOf(rows, r), criterion);
  return parent - (l.length / n) * il - (r.length / n) * ir;
}

export interface TreeOptions {
  maxDepth: number;
  minSamples: number;
  criterion: Criterion;
}

/** Predict by walking the tree from the root. */
export function predict(nodes: TreeNode[], s: Sample): number {
  let node = nodes[0];
  while (node && !node.leaf && node.split) {
    const goLeft = featureOf(s, node.split.feature) <= node.split.threshold;
    const next = goLeft ? node.left : node.right;
    if (next === null) break;
    node = nodes[next];
  }
  return node?.prediction ?? 0;
}

const accuracy = (nodes: TreeNode[], rows: Sample[]) =>
  rows.length ? rows.filter((s) => predict(nodes, s) === s.label).length / rows.length : 0;

/**
 * Grow the tree depth-first, recording a step at each decision. Nodes are appended as they are
 * created, so `nodes` is always a valid (partial) tree that the viz can draw.
 */
export function treeSteps(train: Sample[], test: Sample[], opts: TreeOptions): TreeStep[] {
  const { maxDepth, minSamples, criterion } = opts;
  const nodes: TreeNode[] = [];
  const steps: TreeStep[] = [];

  const snapshot = (s: Omit<TreeStep, 'nodes' | 'trainAcc' | 'testAcc' | 'leaves'>) =>
    steps.push({
      ...s,
      nodes: nodes.map((n) => ({ ...n, counts: [...n.counts] as [number, number], indices: [...n.indices] })),
      trainAcc: accuracy(nodes, train),
      testAcc: accuracy(nodes, test),
      leaves: nodes.filter((n) => n.leaf).length,
    });

  const makeNode = (parent: number | null, depth: number, bounds: Bounds, indices: number[]): TreeNode => {
    const counts = countsOf(train, indices);
    const node: TreeNode = {
      id: nodes.length,
      parent,
      depth,
      bounds,
      indices,
      counts,
      impurity: impurityOf(counts, criterion),
      prediction: counts[1] > counts[0] ? 1 : 0,
      split: null,
      left: null,
      right: null,
      leaf: true,
    };
    nodes.push(node);
    return node;
  };

  const label = (n: TreeNode) => (n.parent === null ? 'the root' : `node ${n.id}`);

  const grow = (node: TreeNode) => {
    const n = node.indices.length;
    const pure = node.counts[0] === 0 || node.counts[1] === 0;
    snapshot({
      current: node.id,
      phase: 'node',
      message: `At ${label(node)}: ${n} samples, ${node.counts[0]} of class A and ${node.counts[1]} of class B. ${criterion === 'gini' ? 'Gini' : 'Entropy'} = ${node.impurity.toFixed(3)}${pure ? ' — already pure.' : '.'}`,
      code: {
        line: 'fn',
        vars: { node: node.id, depth: node.depth, n, A: node.counts[0], B: node.counts[1], [criterion]: +node.impurity.toFixed(3) },
      },
    });

    if (pure || node.depth >= maxDepth || n < minSamples) {
      snapshot({
        current: node.id,
        phase: 'leaf',
        message: pure
          ? `Every sample here has the same label, so there is nothing left to split. ${label(node)} becomes a leaf predicting class ${node.prediction ? 'B' : 'A'}.`
          : node.depth >= maxDepth
            ? `Depth limit ${maxDepth} reached. ${label(node)} becomes a leaf predicting the majority class, ${node.prediction ? 'B' : 'A'} — impurity ${node.impurity.toFixed(3)} is left unresolved.`
            : `Only ${n} samples left, below the minimum of ${minSamples}. Splitting this far would be fitting noise, so ${label(node)} becomes a leaf.`,
        code: {
          line: node.depth >= maxDepth || n < minSamples ? 'stop' : 'leaf',
          vars: { node: node.id, predict: node.prediction ? 'B' : 'A', n },
        },
      });
      return;
    }

    const candidates: Split[] = [];
    for (const f of [0, 1] as const)
      for (const t of thresholds(train, node.indices, f))
        candidates.push({ feature: f, threshold: t, gain: gainOf(train, node.indices, f, t, node.impurity, criterion) });

    const best = candidates.reduce((a, b) => (b.gain > a.gain ? b : a), { feature: 0, threshold: 0, gain: -1 } as Split);
    snapshot({
      current: node.id,
      phase: 'scan',
      candidates,
      best,
      message: `Try every threshold on both features — ${candidates.length} candidates. The curve is the impurity drop each one buys; the best is ${best.feature === 0 ? 'x' : 'y'} ≤ ${best.threshold.toFixed(1)}, worth ${best.gain.toFixed(3)}.`,
      code: {
        line: 'scan',
        vars: {
          tried: candidates.length,
          best: `${best.feature === 0 ? 'x' : 'y'} ≤ ${best.threshold.toFixed(1)}`,
          gain: +best.gain.toFixed(3),
        },
      },
    });

    if (best.gain <= 1e-12) {
      snapshot({
        current: node.id,
        phase: 'leaf',
        message: `No split lowers impurity at all, so the greedy rule stops here. This is exactly where a one-level-at-a-time learner gets stuck — XOR does it on the first try.`,
        code: { line: 'leaf', vars: { node: node.id, 'best gain': 0 } },
      });
      return;
    }

    node.split = best;
    node.leaf = false;
    const li: number[] = [];
    const ri: number[] = [];
    for (const i of node.indices) (featureOf(train[i]!, best.feature) <= best.threshold ? li : ri).push(i);

    const b = node.bounds;
    const lb: Bounds = best.feature === 0 ? { ...b, x1: best.threshold } : { ...b, y1: best.threshold };
    const rb: Bounds = best.feature === 0 ? { ...b, x0: best.threshold } : { ...b, y0: best.threshold };
    const left = makeNode(node.id, node.depth + 1, lb, li);
    const right = makeNode(node.id, node.depth + 1, rb, ri);
    node.left = left.id;
    node.right = right.id;

    snapshot({
      current: node.id,
      phase: 'split',
      best,
      message: `Cut at ${best.feature === 0 ? 'x' : 'y'} ≤ ${best.threshold.toFixed(1)}: ${li.length} samples go left, ${ri.length} right. Each side is now its own smaller problem, solved the same way.`,
      code: {
        line: 'split',
        vars: { split: `${best.feature === 0 ? 'x' : 'y'} ≤ ${best.threshold.toFixed(1)}`, left: li.length, right: ri.length },
      },
    });

    grow(left);
    grow(right);
  };

  const root = makeNode(
    null,
    0,
    { x0: 0, x1: 100, y0: 0, y1: 100 },
    train.map((_, i) => i),
  );
  grow(root);

  snapshot({
    current: null,
    phase: 'done',
    message: `Done: ${nodes.length} nodes, ${nodes.filter((n) => n.leaf).length} leaves. Training accuracy ${(accuracy(nodes, train) * 100).toFixed(1)}%, held-out ${(accuracy(nodes, test) * 100).toFixed(1)}%.`,
    code: { line: 'fn', vars: { nodes: nodes.length, leaves: nodes.filter((n) => n.leaf).length } },
  });

  return steps;
}

/** Train and held-out accuracy at each depth from 1 to maxDepth, for the overfitting curve. */
export function depthCurve(train: Sample[], test: Sample[], maxDepth: number, opts: Omit<TreeOptions, 'maxDepth'>) {
  const trainAcc: number[] = [];
  const testAcc: number[] = [];
  for (let d = 1; d <= maxDepth; d++) {
    const last = treeSteps(train, test, { ...opts, maxDepth: d }).at(-1)!;
    trainAcc.push(last.trainAcc * 100);
    testAcc.push(last.testAcc * 100);
  }
  return { trainAcc, testAcc };
}
