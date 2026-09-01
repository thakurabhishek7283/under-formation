// Ensembles of decision trees: many weak or unstable models combined into one strong one.
//
// - Bagging: train each tree on a bootstrap resample of the rows, then average their votes.
//   Deep trees are high-variance; averaging many of them cancels much of that variance out.
// - Random forest: bagging, plus each split only gets to look at a random subset of features,
//   so the trees make *different* mistakes and the average cancels more of them.
// - AdaBoost: shallow trees trained one after another, each on data reweighted towards the
//   points the previous ones got wrong, combined in a weighted vote.
//
// The tree learner here is a compact CART (Gini, axis-aligned thresholds) that accepts sample
// weights; the step-by-step version lives in decisionTree.ts.

import { rng } from './math';
import type { LPoint } from './classify';

export interface TreeOpts {
  maxDepth: number;
  minLeaf: number;
  /** Features examined per split: 2 = both (plain CART), 1 = one at random (random forest). */
  maxFeatures: 1 | 2;
}

type Node =
  | { leaf: true; dist: number[] }
  | { leaf: false; feature: 0 | 1; threshold: number; left: Node; right: Node };

export interface Tree {
  root: Node;
  classes: number;
  leaves: number;
}

const feat = (p: LPoint, f: 0 | 1) => (f === 0 ? p.x : p.y);

function giniOf(w: number[], total: number): number {
  if (total <= 0) return 0;
  let s = 1;
  for (const v of w) s -= (v / total) ** 2;
  return s;
}

/** Grow a weighted CART tree. `weights[i]` is how much sample i counts (bootstrap multiplicity or boosting weight). */
export function growTree(pts: LPoint[], weights: number[], classes: number, o: TreeOpts, r: () => number): Tree {
  let leaves = 0;
  const build = (idx: number[], depth: number): Node => {
    const cw = new Array<number>(classes).fill(0);
    let total = 0;
    for (const i of idx) {
      cw[pts[i]!.label]! += weights[i]!;
      total += weights[i]!;
    }
    const dist = cw.map((v) => (total > 0 ? v / total : 1 / classes));
    const parent = giniOf(cw, total);
    if (depth >= o.maxDepth || idx.length < 2 * o.minLeaf || parent < 1e-12) {
      leaves++;
      return { leaf: true, dist };
    }

    const features: (0 | 1)[] = o.maxFeatures === 2 ? [0, 1] : [r() < 0.5 ? 0 : 1];
    let best: { f: 0 | 1; t: number; gain: number } | null = null;
    for (const f of features) {
      const sorted = [...idx].sort((a, b) => feat(pts[a]!, f) - feat(pts[b]!, f));
      const left = new Array<number>(classes).fill(0);
      let lw = 0;
      for (let k = 0; k < sorted.length - 1; k++) {
        const i = sorted[k]!;
        left[pts[i]!.label]! += weights[i]!;
        lw += weights[i]!;
        const a = feat(pts[i]!, f);
        const b = feat(pts[sorted[k + 1]!]!, f);
        if (a === b || k + 1 < o.minLeaf || sorted.length - k - 1 < o.minLeaf) continue;
        const right = cw.map((v, c) => v - left[c]!);
        const rw = total - lw;
        const gain = parent - (lw / total) * giniOf(left, lw) - (rw / total) * giniOf(right, rw);
        if (!best || gain > best.gain + 1e-12) best = { f, t: (a + b) / 2, gain };
      }
    }
    if (!best || best.gain <= 1e-12) {
      leaves++;
      return { leaf: true, dist };
    }
    const { f, t } = best;
    const li = idx.filter((i) => feat(pts[i]!, f) <= t);
    const ri = idx.filter((i) => feat(pts[i]!, f) > t);
    return { leaf: false, feature: f, threshold: t, left: build(li, depth + 1), right: build(ri, depth + 1) };
  };
  const idx = pts.map((_, i) => i).filter((i) => weights[i]! > 0);
  const root = build(idx, 0);
  return { root, classes, leaves };
}

export function treeProba(t: Tree, x: number, y: number): number[] {
  let node = t.root;
  while (!node.leaf) node = (node.feature === 0 ? x : y) <= node.threshold ? node.left : node.right;
  return node.dist;
}

export function treePredict(t: Tree, x: number, y: number): number {
  const d = treeProba(t, x, y);
  let best = 0;
  for (let c = 1; c < d.length; c++) if (d[c]! > d[best]!) best = c;
  return best;
}

export type EnsembleKind = 'bagging' | 'forest' | 'adaboost';

export interface Learner {
  tree: Tree;
  /** Vote weight: 1 for bagging/forests, α for AdaBoost. */
  alpha: number;
  /** Weighted training error when it was fitted (AdaBoost), or out-of-bag error (bagging/forest). */
  error: number;
  /** For AdaBoost: the sample weights this learner was trained on (sum to 1). */
  weights?: number[];
}

export interface EnsembleOptions {
  kind: EnsembleKind;
  count: number;
  /** Depth of each tree: deep for bagging/forests, stumps (1) or shallow for boosting. */
  depth: number;
  seed: number;
}

/**
 * Fit `count` learners. They are nested — the first k of them are exactly the ensemble you'd get
 * by asking for k — so a slider over k can reuse one fit.
 */
export function fitEnsemble(train: LPoint[], classes: number, o: EnsembleOptions): Learner[] {
  const r = rng(o.seed);
  const n = train.length;
  const learners: Learner[] = [];

  if (o.kind === 'adaboost') {
    // SAMME (multi-class AdaBoost). With two classes it is exactly the original AdaBoost.
    let w = new Array<number>(n).fill(1 / n);
    for (let t = 0; t < o.count; t++) {
      const tree = growTree(train, w, classes, { maxDepth: o.depth, minLeaf: 1, maxFeatures: 2 }, r);
      let err = 0;
      const wrong = train.map((p) => treePredict(tree, p.x, p.y) !== p.label);
      wrong.forEach((bad, i) => {
        if (bad) err += w[i]!;
      });
      err = Math.min(Math.max(err, 1e-10), 1 - 1e-10);
      // A learner no better than chance gets no say, and boosting has nothing left to fix.
      if (err >= 1 - 1 / classes) break;
      const alpha = Math.log((1 - err) / err) + Math.log(classes - 1);
      learners.push({ tree, alpha, error: err, weights: w });
      const next = w.map((v, i) => v * (wrong[i] ? Math.exp(alpha) : 1));
      const s = next.reduce((a, b) => a + b, 0);
      w = next.map((v) => v / s);
    }
    return learners;
  }

  for (let t = 0; t < o.count; t++) {
    // Bootstrap: n draws with replacement. About 1 − 1/e ≈ 63% of rows appear at least once.
    const counts = new Array<number>(n).fill(0);
    for (let k = 0; k < n; k++) counts[Math.floor(r() * n)]!++;
    const tree = growTree(train, counts, classes, { maxDepth: o.depth, minLeaf: 1, maxFeatures: o.kind === 'forest' ? 1 : 2 }, r);
    // Out-of-bag error: score the tree on the rows it never saw.
    let oob = 0;
    let oobN = 0;
    train.forEach((p, i) => {
      if (counts[i]) return;
      oobN++;
      if (treePredict(tree, p.x, p.y) !== p.label) oob++;
    });
    learners.push({ tree, alpha: 1, error: oobN ? oob / oobN : 0 });
  }
  return learners;
}

/** Combined vote of the first `k` learners, normalized to sum to 1. */
export function ensembleProba(learners: Learner[], k: number, classes: number, kind: EnsembleKind, x: number, y: number): number[] {
  const v = new Array<number>(classes).fill(0);
  const m = Math.min(k, learners.length);
  for (let t = 0; t < m; t++) {
    const l = learners[t]!;
    if (kind === 'adaboost') v[treePredict(l.tree, x, y)]! += l.alpha;
    else {
      // Forests average the trees' leaf distributions (soft voting, as scikit-learn does).
      const d = treeProba(l.tree, x, y);
      for (let c = 0; c < classes; c++) v[c]! += d[c]!;
    }
  }
  const s = v.reduce((a, b) => a + b, 0);
  return s > 0 ? v.map((c) => c / s) : v.map(() => 1 / classes);
}

export function ensemblePredict(learners: Learner[], k: number, classes: number, kind: EnsembleKind, x: number, y: number): number {
  const v = ensembleProba(learners, k, classes, kind, x, y);
  let best = 0;
  for (let c = 1; c < classes; c++) if (v[c]! > v[best]!) best = c;
  return best;
}

/** Train and held-out accuracy of the first k learners, for k = 1 … learners.length. */
export function ensembleCurve(learners: Learner[], train: LPoint[], test: LPoint[], classes: number, kind: EnsembleKind) {
  // Accumulate votes incrementally rather than re-scoring every prefix from scratch.
  const acc = (pts: LPoint[]) => {
    const votes = pts.map(() => new Array<number>(classes).fill(0));
    const out: number[] = [NaN];
    for (const l of learners) {
      let ok = 0;
      pts.forEach((p, i) => {
        if (kind === 'adaboost') votes[i]![treePredict(l.tree, p.x, p.y)]! += l.alpha;
        else {
          const d = treeProba(l.tree, p.x, p.y);
          for (let c = 0; c < classes; c++) votes[i]![c]! += d[c]!;
        }
        let best = 0;
        for (let c = 1; c < classes; c++) if (votes[i]![c]! > votes[i]![best]!) best = c;
        if (best === p.label) ok++;
      });
      out.push((ok / Math.max(pts.length, 1)) * 100);
    }
    return out;
  };
  return { trainAcc: acc(train), testAcc: acc(test) };
}
