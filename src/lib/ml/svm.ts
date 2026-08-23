// Support vector machine, trained exactly: the soft-margin dual solved by SMO (sequential minimal
// optimization), with LIBSVM's second-order working-set selection.
//
//   minimize   ½ Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ) − Σᵢ αᵢ
//   subject to 0 ≤ αᵢ ≤ C,  Σᵢ αᵢ yᵢ = 0
//
// SMO repeatedly picks the pair (i, j) that most violates the optimality conditions and solves the
// two-variable problem in closed form. The answer is sparse: most αᵢ end up exactly 0, and only the
// samples with αᵢ > 0 — the support vectors — appear in the decision function
//
//   f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) − ρ.

import { toUnit, type LPoint } from './classify';

export type KernelKind = 'linear' | 'poly' | 'rbf';

export interface SvmOptions {
  kernel: KernelKind;
  C: number;
  /** RBF width: K = exp(−γ‖a − b‖²). Larger γ = narrower bumps = wigglier boundary. */
  gamma: number;
}

export interface SvmModel {
  options: SvmOptions;
  /** Inputs in model units (see toUnit). */
  xs: [number, number][];
  ys: number[];
  alpha: number[];
  rho: number;
  iterations: number;
  /** Dual objective at the solution (lower is better; it is minimized). */
  objective: number;
  converged: boolean;
}

export function kernel(a: [number, number], b: [number, number], o: SvmOptions): number {
  const dot = a[0] * b[0] + a[1] * b[1];
  if (o.kernel === 'linear') return dot;
  if (o.kernel === 'poly') return (dot + 1) ** 2;
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.exp(-o.gamma * (dx * dx + dy * dy));
}

export function trainSvm(points: LPoint[], o: SvmOptions, maxIter = 20000, eps = 1e-3): SvmModel {
  const n = points.length;
  const xs = points.map((p) => [toUnit(p.x), toUnit(p.y)] as [number, number]);
  const ys = points.map((p) => (p.label ? 1 : -1));
  const C = o.C;
  const K = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = i; j < n; j++) {
      const v = kernel(xs[i]!, xs[j]!, o);
      K[i * n + j] = v;
      K[j * n + i] = v;
    }
  const Q = (i: number, j: number) => ys[i]! * ys[j]! * K[i * n + j]!;

  const alpha = new Float64Array(n);
  // Gradient of the dual objective, G = Qα − 1. With α = 0 it is −1 everywhere.
  const G = new Float64Array(n).fill(-1);
  const upOk = (t: number) => (ys[t] === 1 ? alpha[t]! < C : alpha[t]! > 0);
  const lowOk = (t: number) => (ys[t] === 1 ? alpha[t]! > 0 : alpha[t]! < C);
  const TAU = 1e-12;

  let iter = 0;
  let converged = false;
  for (; iter < maxIter; iter++) {
    // i: the most violating sample that can still move "up".
    let Gmax = -Infinity;
    let i = -1;
    for (let t = 0; t < n; t++)
      if (upOk(t) && -ys[t]! * G[t]! >= Gmax) {
        Gmax = -ys[t]! * G[t]!;
        i = t;
      }
    // j: among those that can move "down", the one whose pair with i lowers the objective most.
    let Gmax2 = -Infinity;
    let j = -1;
    let best = Infinity;
    for (let t = 0; t < n; t++) {
      if (!lowOk(t)) continue;
      const yG = ys[t]! * G[t]!;
      if (yG >= Gmax2) Gmax2 = yG;
      const b = Gmax + yG;
      if (b > 0 && i >= 0) {
        const a = Math.max(K[i * n + i]! + K[t * n + t]! - 2 * K[i * n + t]!, TAU);
        const drop = -(b * b) / a;
        if (drop <= best) {
          best = drop;
          j = t;
        }
      }
    }
    if (i < 0 || j < 0 || Gmax + Gmax2 < eps) {
      converged = true;
      break;
    }

    // Solve the two-variable subproblem exactly, then clip back into the box [0, C].
    const ai = alpha[i]!;
    const aj = alpha[j]!;
    if (ys[i] !== ys[j]) {
      const quad = Math.max(K[i * n + i]! + K[j * n + j]! + 2 * Q(i, j), TAU);
      const delta = (-G[i]! - G[j]!) / quad;
      const diff = ai - aj;
      let ni = ai + delta;
      let nj = aj + delta;
      if (diff > 0) {
        if (nj < 0) {
          nj = 0;
          ni = diff;
        }
      } else if (ni < 0) {
        ni = 0;
        nj = -diff;
      }
      if (diff > 0) {
        if (ni > C) {
          ni = C;
          nj = C - diff;
        }
      } else if (nj > C) {
        nj = C;
        ni = C + diff;
      }
      alpha[i] = ni;
      alpha[j] = nj;
    } else {
      const quad = Math.max(K[i * n + i]! + K[j * n + j]! - 2 * Q(i, j), TAU);
      const delta = (G[i]! - G[j]!) / quad;
      const sum = ai + aj;
      let ni = ai - delta;
      let nj = aj + delta;
      if (sum > C) {
        if (ni > C) {
          ni = C;
          nj = sum - C;
        }
      } else if (nj < 0) {
        nj = 0;
        ni = sum;
      }
      if (sum > C) {
        if (nj > C) {
          nj = C;
          ni = sum - C;
        }
      } else if (ni < 0) {
        ni = 0;
        nj = sum;
      }
      alpha[i] = ni;
      alpha[j] = nj;
    }
    const di = alpha[i]! - ai;
    const dj = alpha[j]! - aj;
    for (let t = 0; t < n; t++) G[t]! += Q(t, i) * di + Q(t, j) * dj;
  }

  // ρ: averaged over the free support vectors (0 < α < C), which sit exactly on the margin.
  let ub = Infinity;
  let lb = -Infinity;
  let free = 0;
  let sumFree = 0;
  for (let t = 0; t < n; t++) {
    const yG = ys[t]! * G[t]!;
    if (alpha[t]! >= C) {
      if (ys[t] === -1) ub = Math.min(ub, yG);
      else lb = Math.max(lb, yG);
    } else if (alpha[t]! <= 0) {
      if (ys[t] === 1) ub = Math.min(ub, yG);
      else lb = Math.max(lb, yG);
    } else {
      free++;
      sumFree += yG;
    }
  }
  const rho = free > 0 ? sumFree / free : (ub + lb) / 2;

  let objective = 0;
  for (let t = 0; t < n; t++) objective += (alpha[t]! * (G[t]! - 1)) / 2;

  return { options: o, xs, ys, alpha: [...alpha], rho: Number.isFinite(rho) ? rho : 0, iterations: iter, objective, converged };
}

/** f(x) = Σ αᵢ yᵢ K(xᵢ, x) − ρ. Its sign is the class; |f| = 1 is the margin. */
export function decision(m: SvmModel, x: number, y: number): number {
  const q: [number, number] = [toUnit(x), toUnit(y)];
  let s = -m.rho;
  for (let t = 0; t < m.xs.length; t++) if (m.alpha[t]! > 0) s += m.alpha[t]! * m.ys[t]! * kernel(m.xs[t]!, q, m.options);
  return s;
}

/** For the linear kernel: w = Σ αᵢ yᵢ xᵢ, in model units. */
export function weightVector(m: SvmModel): [number, number] {
  let w0 = 0;
  let w1 = 0;
  for (let t = 0; t < m.xs.length; t++) {
    w0 += m.alpha[t]! * m.ys[t]! * m.xs[t]![0];
    w1 += m.alpha[t]! * m.ys[t]! * m.xs[t]![1];
  }
  return [w0, w1];
}

export type SvRole = 'none' | 'margin' | 'bound';

/** 'margin': 0 < α < C, exactly on the margin. 'bound': α = C, inside the margin or misclassified. */
export function svRole(m: SvmModel, t: number): SvRole {
  const a = m.alpha[t]!;
  if (a <= 1e-8) return 'none';
  return a >= m.options.C * (1 - 1e-6) ? 'bound' : 'margin';
}

/** The primal objective ½‖w‖² + C Σ hinge, via the kernel expansion. Equals −dual at the optimum. */
export function primalObjective(m: SvmModel): number {
  const n = m.xs.length;
  let ww = 0;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (m.alpha[i]! > 0 && m.alpha[j]! > 0) ww += m.alpha[i]! * m.alpha[j]! * m.ys[i]! * m.ys[j]! * kernel(m.xs[i]!, m.xs[j]!, m.options);
  let hinge = 0;
  for (let t = 0; t < n; t++) {
    let f = -m.rho;
    for (let s = 0; s < n; s++) if (m.alpha[s]! > 0) f += m.alpha[s]! * m.ys[s]! * kernel(m.xs[s]!, m.xs[t]!, m.options);
    hinge += Math.max(0, 1 - m.ys[t]! * f);
  }
  return ww / 2 + m.options.C * hinge;
}
