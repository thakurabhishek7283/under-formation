// Shared pieces for the 2-D classification visualizations: labelled toy datasets in [0, 100]²,
// a deterministic train/test split, accuracy, and sampling a model's class probabilities over
// the plane so its decision regions can be drawn.
//
// Every classifier page uses the same datasets, so the same "moons" means the same points
// whether you're looking at k-NN, an SVM or a neural network.

import { rng } from './math';

export interface LPoint {
  x: number;
  y: number;
  /** Class index: 0, 1 or 2. There are only three validated class colours, so never more. */
  label: number;
}

export type ClassDataset = 'separable' | 'blobs' | 'three' | 'moons' | 'circles' | 'xor' | 'spiral' | 'correlated';

export const CLASS_DATASETS: Record<ClassDataset, { name: string; classes: number; blurb: string }> = {
  separable: { name: 'Separable', classes: 2, blurb: 'A clear gap between the classes: infinitely many lines separate them perfectly.' },
  blobs: { name: 'Two blobs', classes: 2, blurb: 'Two overlapping clouds. A straight line is about as good as it gets.' },
  three: { name: 'Three blobs', classes: 3, blurb: 'Three classes. Every method here handles more than two.' },
  moons: { name: 'Two moons', classes: 2, blurb: 'Interleaved crescents. No straight line separates them.' },
  circles: { name: 'Circles', classes: 2, blurb: 'One class surrounds the other. Linear models fail completely.' },
  xor: { name: 'XOR', classes: 2, blurb: 'Opposite corners share a class. Neither feature alone says anything.' },
  spiral: { name: 'Spiral', classes: 2, blurb: 'Two interlocking arms: the hardest boundary here.' },
  correlated: {
    name: 'Correlated',
    classes: 2,
    blurb: 'Two long diagonal clouds. Seen one axis at a time they look identical; together they separate cleanly.',
  },
};

/** Model inputs live around [-2, 2]: centring and scaling keeps gradient-based fits well conditioned. */
export const toUnit = (v: number) => (v - 50) / 25;
export const fromUnit = (u: number) => u * 25 + 50;

const clamp = (v: number) => Math.max(2, Math.min(98, v));

// The datasets are generated during server rendering *and* again in the browser, and the two must
// agree to the last bit: one ulp of difference is enough to send an SVM solver down a different
// path. Math.log, Math.cos and friends are not guaranteed to round identically across JS engines
// (Node and Edge already disagree), so the generator uses only +, −, ×, ÷ and Math.round, which
// IEEE 754 pins down exactly.

/** Approximately standard normal: the sum of 12 uniforms, minus 6 (Irwin–Hall). */
function gauss(r: () => number): number {
  let s = -6;
  for (let i = 0; i < 12; i++) s += r();
  return s;
}

/** sin by Taylor series after reducing to [−π, π]; accurate to ~1e−15 there. */
function dsin(x: number): number {
  const t = x - 2 * Math.PI * Math.round(x / (2 * Math.PI));
  const t2 = t * t;
  let term = t;
  let sum = t;
  for (let k = 1; k < 14; k++) {
    term *= -t2 / ((2 * k) * (2 * k + 1));
    sum += term;
  }
  return sum;
}
const dcos = (x: number) => dsin(x + Math.PI / 2);

/** Deterministic labelled points in [0, 100]², shuffled so a prefix is a fair sample. */
export function makeClassData(kind: ClassDataset, n: number, seed: number, noise = 1): LPoint[] {
  const r = rng(seed);
  const pts: LPoint[] = [];
  const g = (sd: number) => gauss(r) * sd * noise;
  const per = (c: number, classes: number) => Math.round(((c + 1) * n) / classes) - Math.round((c * n) / classes);

  switch (kind) {
    case 'three': {
      const centres = [
        [30, 67],
        [70, 67],
        [50, 30],
      ] as const;
      centres.forEach(([cx, cy], c) => {
        for (let i = 0; i < per(c, 3); i++) pts.push({ x: cx + g(10), y: cy + g(10), label: c });
      });
      break;
    }
    case 'moons': {
      // sklearn's make_moons: two half circles, the second flipped and shifted into the first.
      for (let c = 0; c < 2; c++)
        for (let i = 0, m = per(c, 2); i < m; i++) {
          const t = (Math.PI * (i + r())) / m;
          const mx = c === 0 ? dcos(t) : 1 - dcos(t);
          const my = c === 0 ? dsin(t) : 0.5 - dsin(t);
          pts.push({ x: 50 + (mx - 0.5) * 30 + g(4), y: 50 + (my - 0.25) * 30 + g(4), label: c });
        }
      break;
    }
    case 'circles': {
      for (let c = 0; c < 2; c++)
        for (let i = 0, m = per(c, 2); i < m; i++) {
          const t = 2 * Math.PI * r();
          const rad = c === 0 ? 38 : 17;
          pts.push({ x: 50 + dcos(t) * rad + g(4), y: 50 + dsin(t) * rad + g(4), label: c });
        }
      break;
    }
    case 'xor': {
      const corners = [
        [28, 28, 0],
        [72, 72, 0],
        [28, 72, 1],
        [72, 28, 1],
      ] as const;
      corners.forEach(([cx, cy, c], k) => {
        const m = Math.round(((k + 1) * n) / 4) - Math.round((k * n) / 4);
        for (let i = 0; i < m; i++) pts.push({ x: cx + g(10), y: cy + g(10), label: c });
      });
      break;
    }
    case 'spiral': {
      for (let c = 0; c < 2; c++)
        for (let i = 0, m = per(c, 2); i < m; i++) {
          const t = 0.1 + (0.9 * (i + r())) / m;
          const a = t * 3.2 * Math.PI + c * Math.PI;
          pts.push({ x: 50 + dcos(a) * t * 44 + g(2.2), y: 50 + dsin(a) * t * 44 + g(2.2), label: c });
        }
      break;
    }
    case 'correlated': {
      // Long axis along (1, 1), classes offset along (1, −1). Each marginal is nearly the same
      // for both classes, which is exactly what a per-feature model can't see past.
      const s = Math.SQRT1_2;
      for (let c = 0; c < 2; c++)
        for (let i = 0, m = per(c, 2); i < m; i++) {
          const along = gauss(r) * 20;
          const across = gauss(r) * 4.5 * noise + (c === 0 ? -8 : 8);
          pts.push({ x: 50 + along * s + across * s, y: 50 + along * s - across * s, label: c });
        }
      break;
    }
    case 'separable': {
      // Off-centre on purpose, so the boundary needs a bias term and doesn't pass through the middle.
      const centres = [
        [34, 70],
        [68, 58],
      ] as const;
      centres.forEach(([cx, cy], c) => {
        for (let i = 0; i < per(c, 2); i++) pts.push({ x: cx + g(6.5), y: cy + g(6.5), label: c });
      });
      break;
    }
    default: {
      const centres = [
        [36, 62],
        [64, 38],
      ] as const;
      centres.forEach(([cx, cy], c) => {
        for (let i = 0; i < per(c, 2); i++) pts.push({ x: cx + g(12), y: cy + g(12), label: c });
      });
    }
  }

  // Fisher–Yates with the same seeded stream: deterministic, and a prefix is a fair sample.
  for (let i = pts.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [pts[i], pts[j]] = [pts[j]!, pts[i]!];
  }
  return pts.map((p) => ({ x: clamp(p.x), y: clamp(p.y), label: p.label }));
}

/** First `frac` for training, the rest held out. The data is already shuffled. */
export function trainTest(points: LPoint[], frac = 0.7): { train: LPoint[]; test: LPoint[] } {
  const cut = Math.round(points.length * frac);
  return { train: points.slice(0, cut), test: points.slice(cut) };
}

export function argmax(v: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if (v[i]! > v[best]!) best = i;
  return best;
}

export function accuracy(predict: (x: number, y: number) => number, pts: LPoint[]): number {
  if (!pts.length) return 0;
  let ok = 0;
  for (const p of pts) if (predict(p.x, p.y) === p.label) ok++;
  return ok / pts.length;
}

/** Class probabilities sampled at the centre of every cell of a res × res grid over [0, 100]². */
export interface Field {
  res: number;
  classes: number;
  /** p[(row · res + col) · classes + c]; row 0 is the top of the plot (y = 100). */
  p: Float32Array;
}

export function cellCentre(i: number, res: number): number {
  return ((i + 0.5) / res) * 100;
}

export function computeField(proba: (x: number, y: number) => ArrayLike<number>, classes: number, res = 64): Field {
  const p = new Float32Array(res * res * classes);
  for (let row = 0; row < res; row++) {
    const y = 100 - cellCentre(row, res);
    for (let col = 0; col < res; col++) {
      const v = proba(cellCentre(col, res), y);
      const o = (row * res + col) * classes;
      for (let c = 0; c < classes; c++) p[o + c] = v[c]!;
    }
  }
  return { res, classes, p };
}

/** A two-class field from a signed score (e.g. an SVM decision value): shade saturates at |score| = `sat`. */
export function fieldFromScore(score: (x: number, y: number) => number, res = 64, sat = 2): { field: Field; values: Float32Array } {
  const values = new Float32Array(res * res);
  const p = new Float32Array(res * res * 2);
  for (let row = 0; row < res; row++) {
    const y = 100 - cellCentre(row, res);
    for (let col = 0; col < res; col++) {
      const s = score(cellCentre(col, res), y);
      const k = row * res + col;
      values[k] = s;
      const p1 = 0.5 + 0.5 * Math.max(-1, Math.min(1, s / sat));
      p[k * 2] = 1 - p1;
      p[k * 2 + 1] = p1;
    }
  }
  return { field: { res, classes: 2, p }, values };
}

/**
 * Marching squares over a scalar grid (sampled at cell centres, row 0 at the top). Returns line
 * segments where the surface crosses `level`, in plot units [0, 100] with y pointing down —
 * ready to stroke on a canvas or an SVG.
 */
export function contourSegments(values: ArrayLike<number>, res: number, level = 0): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  const at = (row: number, col: number) => values[row * res + col]! - level;
  const px = (col: number) => cellCentre(col, res);
  for (let row = 0; row < res - 1; row++)
    for (let col = 0; col < res - 1; col++) {
      const a = at(row, col); // top-left
      const b = at(row, col + 1); // top-right
      const c = at(row + 1, col + 1); // bottom-right
      const d = at(row + 1, col); // bottom-left
      const code = (a > 0 ? 8 : 0) | (b > 0 ? 4 : 0) | (c > 0 ? 2 : 0) | (d > 0 ? 1 : 0);
      if (code === 0 || code === 15) continue;
      const x0 = px(col);
      const x1 = px(col + 1);
      const y0 = px(row);
      const y1 = px(row + 1);
      const lerp = (u: number, v: number) => u / (u - v);
      const top = (): [number, number] => [x0 + (x1 - x0) * lerp(a, b), y0];
      const right = (): [number, number] => [x1, y0 + (y1 - y0) * lerp(b, c)];
      const bottom = (): [number, number] => [x0 + (x1 - x0) * lerp(d, c), y1];
      const left = (): [number, number] => [x0, y0 + (y1 - y0) * lerp(a, d)];
      const seg = (p: [number, number], q: [number, number]) => out.push([p[0], p[1], q[0], q[1]]);
      switch (code) {
        case 1:
        case 14:
          seg(left(), bottom());
          break;
        case 2:
        case 13:
          seg(bottom(), right());
          break;
        case 3:
        case 12:
          seg(left(), right());
          break;
        case 4:
        case 11:
          seg(top(), right());
          break;
        case 6:
        case 9:
          seg(top(), bottom());
          break;
        case 7:
        case 8:
          seg(left(), top());
          break;
        case 5:
        case 10: {
          // Saddle: let the centre value decide which corners connect.
          const centre = (a + b + c + d) / 4;
          if ((code === 5) === centre > 0) {
            seg(left(), top());
            seg(bottom(), right());
          } else {
            seg(top(), right());
            seg(left(), bottom());
          }
          break;
        }
      }
    }
  return out;
}

/** Where the predicted class changes: for each class, the zero set of p_c − max(other p). */
export function boundarySegments(field: Field): [number, number, number, number][] {
  const { res, classes, p } = field;
  if (classes === 2) {
    const s = new Float32Array(res * res);
    for (let k = 0; k < res * res; k++) s[k] = p[k * 2 + 1]! - p[k * 2]!;
    return contourSegments(s, res, 0);
  }
  const out: [number, number, number, number][] = [];
  for (let c = 0; c < classes; c++) {
    const s = new Float32Array(res * res);
    for (let k = 0; k < res * res; k++) {
      let other = -Infinity;
      for (let d = 0; d < classes; d++) if (d !== c) other = Math.max(other, p[k * classes + d]!);
      s[k] = p[k * classes + c]! - other;
    }
    out.push(...contourSegments(s, res, 0));
  }
  return out;
}
