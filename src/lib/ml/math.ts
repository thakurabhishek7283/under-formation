// Tiny dense linear-algebra helpers for the ML visualizations. Row-major number[][] matrices.

export type Matrix = number[][];

/** Deterministic PRNG (mulberry32) so server and client render identical "random" weights. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal sample (Box–Muller). */
export function randn(r: () => number): number {
  const u = Math.max(r(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

export function randomMatrix(rows: number, cols: number, scale: number, r: () => number): Matrix {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => randn(r) * scale));
}

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const n = a.length;
  const k = b.length;
  const m = b[0]?.length ?? 0;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const aip = a[i]![p]!;
      if (aip === 0) continue;
      const bp = b[p]!;
      const oi = out[i]!;
      for (let j = 0; j < m; j++) oi[j]! += aip * bp[j]!;
    }
  return out;
}

export function transpose(a: Matrix): Matrix {
  return (a[0] ?? []).map((_, j) => a.map((row) => row[j]!));
}

export function add(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((v, j) => v + b[i]![j]!));
}

export function scale(a: Matrix, s: number): Matrix {
  return a.map((row) => row.map((v) => v * s));
}

export function mapM(a: Matrix, f: (v: number) => number): Matrix {
  return a.map((row) => row.map(f));
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

/** Numerically stable softmax; -Infinity entries get probability 0. */
export function softmax(v: number[]): number[] {
  const max = Math.max(...v);
  if (!Number.isFinite(max)) return v.map(() => 1 / v.length);
  const e = v.map((x) => Math.exp(x - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/** LayerNorm without learned gain/bias: zero mean, unit variance per row. */
export function layerNorm(a: Matrix, eps = 1e-5): Matrix {
  return a.map((row) => {
    const mean = row.reduce((s, v) => s + v, 0) / row.length;
    const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length;
    return row.map((v) => (v - mean) / Math.sqrt(variance + eps));
  });
}

/** GELU (tanh approximation), as in GPT-2. */
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
}

/** Sinusoidal positional encoding (Vaswani et al. 2017). */
export function positionalEncoding(positions: number, d: number): Matrix {
  return Array.from({ length: positions }, (_, pos) =>
    Array.from({ length: d }, (_, i) => {
      const freq = 1 / 10000 ** ((2 * Math.floor(i / 2)) / d);
      return i % 2 === 0 ? Math.sin(pos * freq) : Math.cos(pos * freq);
    }),
  );
}

/** Symmetric eigendecomposition (cyclic Jacobi). Fine for the small matrices used here. */
export function symmetricEigen(s: Matrix): { values: number[]; vectors: Matrix } {
  const n = s.length;
  const a = s.map((r) => [...r]);
  const v: Matrix = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p]![q]! ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-15) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const sn = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k]![p]!;
          const akq = a[k]![q]!;
          a[k]![p] = c * akp - sn * akq;
          a[k]![q] = sn * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p]![k]!;
          const aqk = a[q]![k]!;
          a[p]![k] = c * apk - sn * aqk;
          a[q]![k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k]![p]!;
          const vkq = v[k]![q]!;
          v[k]![p] = c * vkp - sn * vkq;
          v[k]![q] = sn * vkp + c * vkq;
        }
      }
  }
  const order = a.map((_, i) => i).sort((i, j) => a[j]![j]! - a[i]![i]!);
  return { values: order.map((i) => a[i]![i]!), vectors: v.map((row) => order.map((i) => row[i]!)) };
}

/** Thin SVD via eigendecomposition of AᵀA: A ≈ U diag(σ) Vᵀ, singular values descending. */
export function svd(a: Matrix): { u: Matrix; s: number[]; v: Matrix } {
  const { values, vectors } = symmetricEigen(matmul(transpose(a), a));
  const s = values.map((x) => Math.sqrt(Math.max(x, 0)));
  const av = matmul(a, vectors);
  const u = av.map((row) => row.map((x, j) => (s[j]! > 1e-12 ? x / s[j]! : 0)));
  return { u, s, v: vectors };
}

/** Best rank-r approximation (Eckart–Young): keep the r largest singular directions. */
export function lowRank(a: Matrix, r: number): Matrix {
  const { u, s, v } = svd(a);
  return a.map((row, i) => row.map((_, j) => {
    let x = 0;
    for (let k = 0; k < r; k++) x += u[i]![k]! * s[k]! * v[j]![k]!;
    return x;
  }));
}

export function frobenius(a: Matrix): number {
  return Math.sqrt(a.reduce((s, row) => s + row.reduce((t, v) => t + v * v, 0), 0));
}

export function maxAbs(a: Matrix): number {
  return a.reduce((m, row) => Math.max(m, ...row.map(Math.abs)), 0);
}
