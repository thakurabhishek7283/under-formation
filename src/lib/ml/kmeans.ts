// Lloyd's algorithm for k-means, traced step by step.
//
// The two half-steps alternate: assign every point to its nearest centroid, then move every
// centroid to the mean of what it was given. Both can only lower the inertia, and there are
// finitely many assignments, so it always terminates — at a *local* optimum that depends on
// where the centroids started.

import { listing, type CodeCursor } from '../code';
import { rng, randn } from './math';

export interface Point {
  x: number;
  y: number;
}

export type DatasetKind = 'blobs' | 'uneven' | 'anisotropic' | 'moons';

export const DATASETS: { id: DatasetKind; name: string; blurb: string }[] = [
  { id: 'blobs', name: 'Three round blobs', blurb: 'The shape k-means assumes: round, similar size, similar spread.' },
  { id: 'uneven', name: 'Uneven sizes', blurb: 'One big loose cluster and two tight ones. Watch the big one get split.' },
  { id: 'anisotropic', name: 'Stretched', blurb: 'Elongated clusters. k-means cuts across them, because it only knows distance.' },
  { id: 'moons', name: 'Two moons', blurb: 'Not linearly separable. No value of k recovers the two moons.' },
];

export type InitKind = 'random' | 'kmeans++';

export interface KMeansStep {
  centroids: Point[];
  /** Cluster index per point; -1 before the first assignment. */
  assignment: number[];
  /** Points whose cluster changed on this step. */
  changed: number[];
  phase: 'init' | 'assign' | 'update' | 'done';
  iteration: number;
  /** Sum of squared distances from each point to its centroid. Lloyd's algorithm only lowers it. */
  inertia: number;
  /** Inertia after each completed half-step, for the convergence chart. */
  history: number[];
  /** Largest distance any centroid moved on an update step. */
  shift: number;
  /** During k-means++ init: the D² sampling weight per point. */
  weights?: number[];
  message: string;
  code: CodeCursor;
}

export const KMEANS_CODE = listing(`
  function kmeans(points, k) {                //@ fn
    let c = init(points, k);  // k centroids  //@ init

    for (;;) {                                //@ loop
      // 1. assign: nearest centroid wins     //@ assign
      const a = points.map(p =>               //@ assign
        argmin(c, q => dist2(p, q)));         //@ assign

      if (same(a, prev)) return c; // settled //@ done

      // 2. update: centroid = its own mean   //@ update
      c = c.map((_, j) =>                     //@ update
        mean(points.filter((_, i) => a[i]===j))); //@ update
      prev = a;                               //@ loop
    }
  }
`);

export const dist2 = (a: Point, b: Point) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

/** Index of the nearest centroid. */
export function nearest(p: Point, centroids: Point[]): number {
  let best = 0;
  let bestD = Infinity;
  for (let j = 0; j < centroids.length; j++) {
    const d = dist2(p, centroids[j]!);
    if (d < bestD) {
      bestD = d;
      best = j;
    }
  }
  return best;
}

/** Deterministic sample data in roughly [0, 100]². */
export function makeDataset(kind: DatasetKind, n: number, seed: number): Point[] {
  const r = rng(seed);
  const pts: Point[] = [];
  const blob = (cx: number, cy: number, sx: number, sy: number, count: number, rot = 0) => {
    for (let i = 0; i < count; i++) {
      const dx = randn(r) * sx;
      const dy = randn(r) * sy;
      pts.push({
        x: cx + dx * Math.cos(rot) - dy * Math.sin(rot),
        y: cy + dx * Math.sin(rot) + dy * Math.cos(rot),
      });
    }
  };

  switch (kind) {
    case 'uneven':
      blob(30, 65, 16, 16, Math.round(n * 0.6));
      blob(74, 74, 5, 5, Math.round(n * 0.2));
      blob(76, 26, 5, 5, n - Math.round(n * 0.6) - Math.round(n * 0.2));
      break;
    case 'anisotropic':
      blob(35, 60, 20, 4, Math.round(n / 3), 0.6);
      blob(60, 42, 20, 4, Math.round(n / 3), 0.6);
      blob(50, 80, 20, 4, n - 2 * Math.round(n / 3), 0.6);
      break;
    case 'moons': {
      const half = Math.round(n / 2);
      for (let i = 0; i < n; i++) {
        const upper = i < half;
        const t = Math.PI * ((upper ? i : i - half) / (upper ? half : n - half));
        const cx = upper ? 44 : 58;
        const cy = upper ? 52 : 62;
        pts.push({
          x: cx + Math.cos(t) * 30 * (upper ? 1 : -1) + randn(r) * 3,
          y: cy + Math.sin(t) * 22 * (upper ? 1 : -1) + randn(r) * 3,
        });
      }
      break;
    }
    default:
      blob(30, 68, 9, 9, Math.round(n / 3));
      blob(70, 72, 9, 9, Math.round(n / 3));
      blob(50, 28, 9, 9, n - 2 * Math.round(n / 3));
  }
  return pts.map((p) => ({ x: clamp(p.x), y: clamp(p.y) }));
}

const clamp = (v: number) => Math.max(3, Math.min(97, v));

function inertiaOf(points: Point[], centroids: Point[], assignment: number[]): number {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const c = centroids[assignment[i]!];
    if (c) sum += dist2(points[i]!, c);
  }
  return sum;
}

function meanOf(points: Point[], assignment: number[], j: number, fallback: Point): Point {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (let i = 0; i < points.length; i++) {
    if (assignment[i] !== j) continue;
    sx += points[i]!.x;
    sy += points[i]!.y;
    n++;
  }
  // An empty cluster has no mean; leaving the centroid put is the common fix.
  return n === 0 ? fallback : { x: sx / n, y: sy / n };
}

/**
 * Full trace of Lloyd's algorithm. Init is recorded one centroid at a time so k-means++'s
 * D² sampling is visible; after that each step is a whole assign or a whole update.
 */
export function kmeansSteps(points: Point[], k: number, init: InitKind, seed: number, maxIters = 40): KMeansStep[] {
  const r = rng(seed);
  const steps: KMeansStep[] = [];
  const history: number[] = [];
  const centroids: Point[] = [];
  let assignment = new Array<number>(points.length).fill(-1);

  const push = (s: Omit<KMeansStep, 'centroids' | 'assignment' | 'history'> & { assignment?: number[] }) =>
    steps.push({
      ...s,
      centroids: centroids.map((c) => ({ ...c })),
      assignment: [...(s.assignment ?? assignment)],
      history: [...history],
    });

  // ---- initialization ----
  if (init === 'random') {
    for (let j = 0; j < k; j++) {
      centroids.push({ ...points[Math.floor(r() * points.length)]! });
      push({
        changed: [],
        phase: 'init',
        iteration: 0,
        inertia: 0,
        shift: 0,
        message: `Start centroid ${j + 1} on a randomly picked point. Random starts are cheap, and sometimes land two centroids inside one real cluster.`,
        code: { line: 'init', vars: { k, picked: j + 1 } },
      });
    }
  } else {
    centroids.push({ ...points[Math.floor(r() * points.length)]! });
    push({
      changed: [],
      phase: 'init',
      iteration: 0,
      inertia: 0,
      shift: 0,
      message: `k-means++: the first centroid is a random point. Each later one is drawn with probability proportional to D², its squared distance to the nearest centroid so far.`,
      code: { line: 'init', vars: { k, picked: 1 } },
    });
    for (let j = 1; j < k; j++) {
      const weights = points.map((p) => dist2(p, centroids[nearest(p, centroids)]!));
      const total = weights.reduce((a, b) => a + b, 0);
      let target = r() * total;
      let pick = 0;
      for (let i = 0; i < weights.length; i++) {
        target -= weights[i]!;
        if (target <= 0) {
          pick = i;
          break;
        }
      }
      centroids.push({ ...points[pick]! });
      push({
        changed: [pick],
        phase: 'init',
        iteration: 0,
        inertia: 0,
        shift: 0,
        weights,
        message: `Centroid ${j + 1}: points far from every existing centroid are darker, and far more likely to be drawn. This spreads the starts out, which is why k-means++ usually converges faster and better.`,
        code: { line: 'init', vars: { k, picked: j + 1, 'D² total': Math.round(total) } },
      });
    }
  }

  // ---- Lloyd iterations ----
  for (let iter = 1; iter <= maxIters; iter++) {
    const next = points.map((p) => nearest(p, centroids));
    const changed: number[] = [];
    for (let i = 0; i < next.length; i++) if (next[i] !== assignment[i]) changed.push(i);

    assignment = next;
    const assignInertia = inertiaOf(points, centroids, assignment);
    history.push(assignInertia);
    push({
      changed,
      phase: 'assign',
      iteration: iter,
      inertia: assignInertia,
      shift: 0,
      message:
        iter === 1
          ? `Assign: every point takes its nearest centroid. The result is a Voronoi partition — the boundaries are exactly halfway between neighbouring centroids.`
          : changed.length === 0
            ? `Assign: nothing changed hands. The assignment is stable, so the algorithm is done.`
            : `Assign: ${changed.length} point${changed.length === 1 ? '' : 's'} changed cluster (outlined). Inertia drops to ${Math.round(assignInertia)}.`,
      code: { line: 'assign', vars: { iter, moved: changed.length, inertia: Math.round(assignInertia) } },
    });

    if (changed.length === 0) {
      push({
        changed: [],
        phase: 'done',
        iteration: iter,
        inertia: assignInertia,
        shift: 0,
        message: `Converged after ${iter - 1} full iteration${iter - 1 === 1 ? '' : 's'}. Inertia ${Math.round(assignInertia)} — a local optimum, not necessarily the best one.`,
        code: { line: 'done', vars: { iterations: iter - 1, inertia: Math.round(assignInertia) } },
      });
      break;
    }

    let shift = 0;
    for (let j = 0; j < k; j++) {
      const m = meanOf(points, assignment, j, centroids[j]!);
      shift = Math.max(shift, Math.sqrt(dist2(m, centroids[j]!)));
      centroids[j] = m;
    }
    const updateInertia = inertiaOf(points, centroids, assignment);
    history.push(updateInertia);
    push({
      changed: [],
      phase: 'update',
      iteration: iter,
      inertia: updateInertia,
      shift,
      message: `Update: each centroid moves to the mean of its own points (furthest move ${shift.toFixed(1)}). The mean is the point that minimizes squared distance, so inertia falls again, to ${Math.round(updateInertia)}.`,
      code: { line: 'update', vars: { iter, shift: +shift.toFixed(2), inertia: Math.round(updateInertia) } },
    });
  }

  return steps;
}

/**
 * The Voronoi cell of each centroid, clipped to [0, size]².
 *
 * Assignment *is* this partition — a point belongs to the cell it falls in — so drawing the
 * cells shows cluster membership without needing a colour per cluster. Each cell is the box
 * clipped by the perpendicular bisector against every other centroid (Sutherland–Hodgman).
 */
export function voronoiCells(centroids: Point[], size = 100): Point[][] {
  return centroids.map((c, j) => {
    let poly: Point[] = [
      { x: 0, y: 0 },
      { x: size, y: 0 },
      { x: size, y: size },
      { x: 0, y: size },
    ];
    for (let o = 0; o < centroids.length && poly.length; o++) {
      if (o === j) continue;
      const q = centroids[o]!;
      // Keep the half-plane closer to c: (q−c)·p <= (|q|²−|c|²)/2.
      const nx = q.x - c.x;
      const ny = q.y - c.y;
      const off = (q.x * q.x + q.y * q.y - c.x * c.x - c.y * c.y) / 2;
      const inside = (p: Point) => nx * p.x + ny * p.y <= off;
      const cross = (a: Point, b: Point): Point => {
        const da = nx * a.x + ny * a.y - off;
        const db = nx * b.x + ny * b.y - off;
        const t = da / (da - db);
        return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
      };
      const next: Point[] = [];
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i]!;
        const b = poly[(i + 1) % poly.length]!;
        const ain = inside(a);
        const bin = inside(b);
        if (ain) next.push(a);
        if (ain !== bin) next.push(cross(a, b));
      }
      poly = next;
    }
    return poly;
  });
}

/** Final inertia for each k, for the elbow plot. Each k is run from the same seed. */
export function elbowCurve(points: Point[], maxK: number, init: InitKind, seed: number): number[] {
  const out: number[] = [];
  for (let k = 1; k <= maxK; k++) {
    const steps = kmeansSteps(points, k, init, seed);
    out.push(steps[steps.length - 1]!.inertia);
  }
  return out;
}
