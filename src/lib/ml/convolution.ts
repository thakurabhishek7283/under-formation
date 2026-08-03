// 2D convolution the way a CNN layer does it, traced one output cell at a time.
//
// A convolution layer is a small window of weights dragged across the input. Two properties fall
// out of that and explain most of what CNNs are good at: the window is *local* (a cell only sees
// its neighbourhood) and the weights are *shared* (the same detector is applied everywhere).

import { listing, type CodeCursor } from '../code';
import { rng } from './math';

export type Grid = number[][];

export interface Kernel {
  id: string;
  name: string;
  blurb: string;
  weights: Grid;
  /** Divide the sum by this (box blur averages rather than adds). */
  divisor?: number;
}

export const KERNELS: Kernel[] = [
  {
    id: 'identity',
    name: 'Identity',
    blurb: 'A 1 in the middle and zeros around it: the output is a copy of the input.',
    weights: [
      [0, 0, 0],
      [0, 1, 0],
      [0, 0, 0],
    ],
  },
  {
    id: 'sobel-x',
    name: 'Vertical edges (Sobel x)',
    blurb: 'Right side minus left side. Big where brightness changes horizontally, zero on flat areas.',
    weights: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
  },
  {
    id: 'sobel-y',
    name: 'Horizontal edges (Sobel y)',
    blurb: 'The same detector rotated: bottom minus top.',
    weights: [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
  },
  {
    id: 'blur',
    name: 'Box blur',
    blurb: 'Every neighbour counts equally and the sum is divided by 9 — a local average.',
    weights: [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
    divisor: 9,
  },
  {
    id: 'sharpen',
    name: 'Sharpen',
    blurb: 'Boost the centre, subtract the neighbours: amplifies whatever makes a cell differ from its surroundings.',
    weights: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
  {
    id: 'laplace',
    name: 'Laplacian (all edges)',
    blurb: 'Centre minus every neighbour. Responds to edges in any direction, and to nothing flat.',
    weights: [
      [0, 1, 0],
      [1, -4, 1],
      [0, 1, 0],
    ],
  },
];

export type ImageKind = 'digit' | 'circle' | 'edge' | 'diagonal' | 'stripes';

export const IMAGES: { id: ImageKind; name: string }[] = [
  { id: 'digit', name: 'Digit-like shape' },
  { id: 'circle', name: 'Disc' },
  { id: 'edge', name: 'One vertical edge' },
  { id: 'diagonal', name: 'Diagonal edge' },
  { id: 'stripes', name: 'Stripes' },
];

export type Padding = 'valid' | 'same';

export const CONV_CODE = listing(`
  function conv2d(img, kernel, stride, pad) {  //@ fn
    const out = [];                            //@ fn

    for (let oy = 0; oy < outH; oy++)          //@ loop
      for (let ox = 0; ox < outW; ox++) {      //@ loop
        let sum = 0;                           //@ zero

        for (let ky = 0; ky < k; ky++)         //@ window
          for (let kx = 0; kx < k; kx++)       //@ window
            sum += kernel[ky][kx] *            //@ mac
              img[oy*stride + ky - pad]        //@ mac
                 [ox*stride + kx - pad];       //@ mac

        out[oy][ox] = sum;   // one number     //@ store
      }
    return out;                                //@ done
  }
`);

export interface ConvStep {
  /** Output cell being produced. */
  oy: number;
  ox: number;
  /** Top-left of the receptive field, in unpadded input coordinates (may be negative). */
  iy: number;
  ix: number;
  /** The k×k patch under the kernel; null where it falls outside the image. */
  window: (number | null)[][];
  /** Element-wise window × kernel. */
  products: (number | null)[][];
  sum: number;
  /** Feature map so far; null = not computed yet. */
  output: (number | null)[][];
  message: string;
  code: CodeCursor;
}

/** Deterministic test images in [0, 1]. */
export function makeImage(kind: ImageKind, size: number, seed = 5): Grid {
  const r = rng(seed);
  const g: Grid = [];
  const c = (size - 1) / 2;
  for (let y = 0; y < size; y++) {
    const row: number[] = [];
    for (let x = 0; x < size; x++) {
      let v: number;
      switch (kind) {
        case 'circle':
          v = Math.hypot(x - c, y - c) < size * 0.3 ? 0.95 : 0.05;
          break;
        case 'edge':
          v = x < size / 2 ? 0.08 : 0.92;
          break;
        case 'diagonal':
          v = x + y < size ? 0.08 : 0.92;
          break;
        case 'stripes':
          v = x % 4 < 2 ? 0.85 : 0.1;
          break;
        default: {
          // A blocky "7": a bar across the top and a stroke down to the left.
          const top = y >= 2 && y <= 3 && x >= 2 && x <= size - 3;
          const stem = x >= size - 4 - Math.round((y - 3) * 0.55) && x <= size - 2 - Math.round((y - 3) * 0.55) && y > 3 && y <= size - 3;
          v = top || stem ? 0.9 : 0.06;
        }
      }
      row.push(Math.min(1, Math.max(0, v + (r() - 0.5) * 0.05)));
    }
    g.push(row);
  }
  return g;
}

export const outputSize = (inSize: number, k: number, stride: number, padding: Padding) => {
  const p = padding === 'same' ? Math.floor(k / 2) : 0;
  return Math.floor((inSize + 2 * p - k) / stride) + 1;
};

/** One step per output cell. The window is clipped at the border; 'same' padding treats outside as 0. */
export function convSteps(img: Grid, kernel: Kernel, stride: number, padding: Padding): ConvStep[] {
  const n = img.length;
  const k = kernel.weights.length;
  const pad = padding === 'same' ? Math.floor(k / 2) : 0;
  const outN = outputSize(n, k, stride, padding);
  const div = kernel.divisor ?? 1;

  const output: (number | null)[][] = Array.from({ length: outN }, () => new Array<number | null>(outN).fill(null));
  const steps: ConvStep[] = [];

  for (let oy = 0; oy < outN; oy++) {
    for (let ox = 0; ox < outN; ox++) {
      const iy = oy * stride - pad;
      const ix = ox * stride - pad;
      const window: (number | null)[][] = [];
      const products: (number | null)[][] = [];
      let sum = 0;
      for (let ky = 0; ky < k; ky++) {
        const wr: (number | null)[] = [];
        const pr: (number | null)[] = [];
        for (let kx = 0; kx < k; kx++) {
          const y = iy + ky;
          const x = ix + kx;
          const inside = y >= 0 && y < n && x >= 0 && x < n;
          const v = inside ? img[y]![x]! : null;
          wr.push(v);
          const p = (v ?? 0) * kernel.weights[ky]![kx]!;
          pr.push(inside ? p : null);
          sum += p;
        }
        window.push(wr);
        products.push(pr);
      }
      sum /= div;
      output[oy]![ox] = sum;

      steps.push({
        oy,
        ox,
        iy,
        ix,
        window,
        products,
        sum,
        output: output.map((r) => [...r]),
        message: `Output (${oy}, ${ox}): multiply the ${k}×${k} patch at input (${iy}, ${ix}) by the kernel, element by element, and add the ${k * k} products${div !== 1 ? `, then divide by ${div}` : ''} → ${sum.toFixed(2)}.`,
        code: {
          line: 'mac',
          vars: { oy, ox, iy, ix, sum: +sum.toFixed(3) },
        },
      });
    }
  }

  const last = steps[steps.length - 1];
  if (last) {
    steps.push({
      ...last,
      message: `Done: ${outN}×${outN} = ${outN * outN} output cells, each one a weighted sum of ${k * k} inputs. The same ${k * k} weights produced every single cell — that's weight sharing.`,
      code: { line: 'done', vars: { out: `${outN}×${outN}`, weights: k * k } },
    });
  }
  return steps;
}

export function convolve(img: Grid, kernel: Kernel, stride: number, padding: Padding): Grid {
  const steps = convSteps(img, kernel, stride, padding);
  const final = steps[steps.length - 1]!.output;
  return final.map((row) => row.map((v) => v ?? 0));
}

export type PoolKind = 'max' | 'avg';

/** Non-overlapping pooling windows. Odd sizes drop the remainder, as most frameworks do. */
export function pool(g: Grid, size: number, kind: PoolKind): Grid {
  const outN = Math.floor(g.length / size);
  const out: Grid = [];
  for (let y = 0; y < outN; y++) {
    const row: number[] = [];
    for (let x = 0; x < outN; x++) {
      const vals: number[] = [];
      for (let dy = 0; dy < size; dy++) for (let dx = 0; dx < size; dx++) vals.push(g[y * size + dy]![x * size + dx]!);
      row.push(kind === 'max' ? Math.max(...vals) : vals.reduce((a, b) => a + b, 0) / vals.length);
    }
    out.push(row);
  }
  return out;
}

/** Weights in a conv layer vs a dense layer producing the same output size. */
export function paramCount(inSize: number, k: number, outSize: number) {
  return {
    conv: k * k + 1,
    dense: inSize * inSize * outSize * outSize + outSize * outSize,
  };
}
