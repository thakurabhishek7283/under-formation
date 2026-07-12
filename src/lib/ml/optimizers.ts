// Gradient descent variants on 2D loss surfaces: plain SGD, momentum, and Adam.

export type Vec2 = [number, number];

export interface Surface {
  id: string;
  name: string;
  blurb: string;
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => Vec2;
  domain: { x: Vec2; y: Vec2 };
  start: Vec2;
  minima: Vec2[];
  /** Tuned base learning rates per optimizer (the UI slider multiplies these). */
  lr: Record<OptimizerName, number>;
}

export const SURFACES: Surface[] = [
  {
    id: 'valley',
    name: 'Narrow valley',
    blurb: 'Steep in one direction, almost flat in the other: the typical shape of a badly conditioned loss.',
    f: (x, y) => 0.05 * x * x + 2 * y * y,
    grad: (x, y) => [0.1 * x, 4 * y],
    domain: { x: [-10, 10], y: [-4, 4] },
    start: [-9, 3],
    minima: [[0, 0]],
    lr: { sgd: 0.45, momentum: 0.05, adam: 0.3 },
  },
  {
    id: 'rosenbrock',
    name: 'Rosenbrock banana',
    blurb: 'A curved, flat-bottomed valley. Finding the valley is easy; following it to the minimum at (1, 1) is slow.',
    f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2,
    grad: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)],
    domain: { x: [-2, 2], y: [-1, 3] },
    start: [-1.6, 2.6],
    minima: [[1, 1]],
    lr: { sgd: 0.004, momentum: 0.0015, adam: 0.06 },
  },
  {
    id: 'himmelblau',
    name: 'Himmelblau (4 minima)',
    blurb: 'Four equally good minima. Which one an optimizer lands in depends on where it starts and how it moves.',
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    grad: (x, y) => [
      4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
      2 * (x * x + y - 11) + 4 * y * (x + y * y - 7),
    ],
    domain: { x: [-5, 5], y: [-5, 5] },
    start: [-0.3, -0.9],
    minima: [
      [3, 2],
      [-2.805118, 3.131312],
      [-3.77931, -3.283186],
      [3.584428, -1.848126],
    ],
    lr: { sgd: 0.008, momentum: 0.002, adam: 0.12 },
  },
];

export type OptimizerName = 'sgd' | 'momentum' | 'adam';

export const OPTIMIZERS: { id: OptimizerName; name: string; rule: string }[] = [
  { id: 'sgd', name: 'SGD', rule: 'θ ← θ − η·g' },
  { id: 'momentum', name: 'Momentum', rule: 'v ← 0.9·v + g;  θ ← θ − η·v' },
  { id: 'adam', name: 'Adam', rule: 'm, v ← EMA(g), EMA(g²);  θ ← θ − η·m̂/(√v̂ + ε)' },
];

export interface Trajectory {
  path: Vec2[];
  losses: number[];
  diverged: boolean;
}

export function optimize(surface: Surface, name: OptimizerName, start: Vec2, lrScale: number, steps: number): Trajectory {
  const lr = surface.lr[name] * lrScale;
  let p: Vec2 = [...start];
  const m: Vec2 = [0, 0];
  const v: Vec2 = [0, 0];
  const b1 = 0.9;
  const b2 = 0.999;
  const path: Vec2[] = [[...p]];
  const losses = [surface.f(...p)];
  for (let t = 1; t <= steps; t++) {
    const g = surface.grad(...p);
    for (let i = 0; i < 2; i++) {
      if (name === 'sgd') p[i]! -= lr * g[i]!;
      else if (name === 'momentum') {
        v[i] = 0.9 * v[i]! + g[i]!;
        p[i]! -= lr * v[i]!;
      } else {
        m[i] = b1 * m[i]! + (1 - b1) * g[i]!;
        v[i] = b2 * v[i]! + (1 - b2) * g[i]! ** 2;
        const mh = m[i]! / (1 - b1 ** t);
        const vh = v[i]! / (1 - b2 ** t);
        p[i]! -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
      }
    }
    const l = surface.f(...p);
    if (!Number.isFinite(l) || Math.abs(p[0]) > 1e4 || Math.abs(p[1]) > 1e4) return { path, losses, diverged: true };
    path.push([...p]);
    losses.push(l);
  }
  return { path, losses, diverged: false };
}
