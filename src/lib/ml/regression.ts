// Polynomial regression with L2 (ridge) and L1 (lasso) penalties — the standard setting for
// seeing bias, variance and regularization in one picture.
//
// The model is linear in its parameters, so "fit" means solving a linear system rather than
// running gradient descent. x is kept in [-1, 1] so the powers stay in a sane numeric range.

import { listing } from '../code';
import { rng, randn } from './math';

/** The three fits, side by side. The highlighted branch follows the penalty control. */
export const FIT_CODE = listing(`
  function fit(X, y, lambda, penalty) {       //@ fn
    const A = Xt(X) * X;    // p x p          //@ fn
    const b = Xt(X) * y;                      //@ fn

    if (penalty === 'none')                   //@ none
      return solve(A, b);   // least squares  //@ none

    if (penalty === 'l2')   // ridge          //@ l2
      return solve(addDiag(A, lambda), b);    //@ l2

    // l1 (lasso): no closed form.            //@ l1
    // Cycle the weights, shrinking each       //@ l1
    // towards 0 and clipping it there:        //@ l1
    w[j] = soft(rho[j], lambda) / colSq[j];   //@ l1
  }

  function soft(z, t) {     // soft threshold //@ soft
    return z > t ? z - t                      //@ soft
         : z < -t ? z + t                     //@ soft
         : 0;               // exactly zero   //@ soft
  }
`);

export type Target = 'sine' | 'kink';
export type Penalty = 'none' | 'l2' | 'l1';

export const TARGETS: { id: Target; name: string; f: (x: number) => number }[] = [
  { id: 'sine', name: 'sin(πx)', f: (x) => Math.sin(Math.PI * x) },
  { id: 'kink', name: 'a kink', f: (x) => Math.abs(x + 0.2) - 0.45 },
];

export const targetFn = (t: Target) => TARGETS.find((x) => x.id === t)!.f;

export interface Sample {
  xs: number[];
  ys: number[];
}

/** n points with x spread over [-1, 1] and gaussian noise on y. */
export function makeSample(target: Target, n: number, noise: number, seed: number): Sample {
  const r = rng(seed);
  const f = targetFn(target);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    // Jittered stratified x, so no run of points is left empty.
    const x = -1 + (2 * (i + r())) / n;
    xs.push(x);
    ys.push(f(x) + randn(r) * noise);
  }
  return { xs, ys };
}

/** Rows of [1, x, x², …, x^degree]. */
function design(xs: number[], degree: number): number[][] {
  return xs.map((x) => {
    const row = [1];
    for (let d = 1; d <= degree; d++) row.push(row[d - 1]! * x);
    return row;
  });
}

/** Solve A·w = b by Gaussian elimination with partial pivoting. A is modified in place. */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[pivot]![col]!)) pivot = r;
    if (Math.abs(M[pivot]![col]!) < 1e-14) continue; // singular column: leave that weight at 0
    [M[col], M[pivot]] = [M[pivot]!, M[col]!];
    const p = M[col]![col]!;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r]![col]! / p;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) M[r]![c]! -= factor * M[col]![c]!;
    }
  }
  return Array.from({ length: n }, (_, i) => {
    const d = M[i]![i]!;
    return Math.abs(d) < 1e-14 ? 0 : M[i]![n]! / d;
  });
}

const softThreshold = (z: number, t: number) => (z > t ? z - t : z < -t ? z + t : 0);

/**
 * Least squares with an optional penalty. The intercept is never penalized — shrinking it would
 * just bias every prediction towards zero.
 *
 * - `l2` (ridge): closed form, w = (XᵀX + λI)⁻¹Xᵀy.
 * - `l1` (lasso): no closed form; cyclic coordinate descent with soft thresholding, which is
 *   what makes coefficients hit exactly zero rather than merely getting small.
 */
export function fit(sample: Sample, degree: number, lambda: number, penalty: Penalty): number[] {
  const X = design(sample.xs, degree);
  const y = sample.ys;
  const n = X.length;
  const p = degree + 1;

  if (penalty === 'l1' && lambda > 0) {
    const w = new Array<number>(p).fill(0);
    const colSq = Array.from({ length: p }, (_, j) => X.reduce((s, row) => s + row[j]! * row[j]!, 0) / n);
    const resid = [...y];
    for (let iter = 0; iter < 400; iter++) {
      let maxChange = 0;
      for (let j = 0; j < p; j++) {
        if (colSq[j]! < 1e-12) continue;
        // Partial residual: what's left once every other term has had its say.
        let rho = 0;
        for (let i = 0; i < n; i++) rho += X[i]![j]! * (resid[i]! + X[i]![j]! * w[j]!);
        rho /= n;
        const next = j === 0 ? rho / colSq[j]! : softThreshold(rho, lambda) / colSq[j]!;
        const delta = next - w[j]!;
        if (delta !== 0) {
          for (let i = 0; i < n; i++) resid[i]! -= X[i]![j]! * delta;
          w[j] = next;
          maxChange = Math.max(maxChange, Math.abs(delta));
        }
      }
      if (maxChange < 1e-10) break;
    }
    return w;
  }

  // Normal equations. The 1e-10 floor only keeps a singular XᵀX solvable; it is far too small
  // to regularize anything, so an unpenalized high-degree fit still swings wildly, as it should.
  const l = penalty === 'l2' ? lambda * n : 1e-10;
  const XtX: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const Xty = new Array<number>(p).fill(0);
  for (let i = 0; i < n; i++) {
    const row = X[i]!;
    for (let a = 0; a < p; a++) {
      Xty[a]! += row[a]! * y[i]!;
      for (let b = 0; b < p; b++) XtX[a]![b]! += row[a]! * row[b]!;
    }
  }
  for (let a = 1; a < p; a++) XtX[a]![a]! += l;
  XtX[0]![0]! += 1e-10;
  return solve(XtX, Xty);
}

export function predict(w: number[], x: number): number {
  let out = 0;
  let pow = 1;
  for (const c of w) {
    out += c * pow;
    pow *= x;
  }
  return out;
}

export const mse = (w: number[], s: Sample) =>
  s.xs.reduce((acc, x, i) => acc + (predict(w, x) - s.ys[i]!) ** 2, 0) / Math.max(1, s.xs.length);

/** Train and test error at every degree, holding λ fixed — the classic U-shaped curve. */
export function degreeCurve(train: Sample, test: Sample, maxDegree: number, lambda: number, penalty: Penalty) {
  const trainErr: number[] = [];
  const testErr: number[] = [];
  for (let d = 0; d <= maxDegree; d++) {
    const w = fit(train, d, lambda, penalty);
    trainErr.push(mse(w, train));
    testErr.push(mse(w, test));
  }
  return { trainErr, testErr };
}

export interface BiasVariance {
  /** One fitted curve per resampled training set, evaluated on `grid`. */
  curves: number[][];
  /** The average model, E[f̂(x)]. */
  mean: number[];
  /** Average over the grid of (E[f̂(x)] − f(x))². */
  bias2: number;
  /** Average over the grid of E[(f̂(x) − E[f̂(x)])²]. */
  variance: number;
}

/**
 * Refit the same model on many independent training sets to separate the two error sources:
 * how far the *average* fit is from the truth (bias), and how much the fits differ from each
 * other (variance). Only the training data is resampled; the model settings are held fixed.
 */
export function biasVariance(
  target: Target,
  n: number,
  noise: number,
  degree: number,
  lambda: number,
  penalty: Penalty,
  grid: number[],
  repeats = 24,
  seed = 500,
): BiasVariance {
  const f = targetFn(target);
  const curves: number[][] = [];
  for (let r = 0; r < repeats; r++) {
    const w = fit(makeSample(target, n, noise, seed + r * 17), degree, lambda, penalty);
    curves.push(grid.map((x) => predict(w, x)));
  }
  const mean = grid.map((_, i) => curves.reduce((s, c) => s + c[i]!, 0) / repeats);
  let bias2 = 0;
  let variance = 0;
  for (let i = 0; i < grid.length; i++) {
    bias2 += (mean[i]! - f(grid[i]!)) ** 2;
    variance += curves.reduce((s, c) => s + (c[i]! - mean[i]!) ** 2, 0) / repeats;
  }
  return { curves, mean, bias2: bias2 / grid.length, variance: variance / grid.length };
}
