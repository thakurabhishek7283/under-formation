// Fenwick tree (binary indexed tree) for prefix sums, with tracing for visualization.
// 1-indexed: T[i] stores the sum of a(i - lowbit(i), i], where lowbit(i) = i & -i.

import { listing, type CodeCursor } from '../code';

export const lowbit = (i: number) => i & -i;

export type FenRole = 'visit' | 'use' | 'changed';

export interface FenStep {
  /** Underlying values a[1..n] (index 0 unused). */
  values: number[];
  /** Tree cells T[1..n] (index 0 unused). */
  tree: number[];
  roles: Record<number, FenRole>;
  current: number | null;
  message: string;
  /** Line of the listing being executed, with the variables in scope there. */
  code: CodeCursor;
  /** Running total for queries. */
  acc?: number;
  /** Highlighted index range in a[], 1-based inclusive. */
  target: [number, number] | null;
}

/** O(n) construction: each cell hands its total up to the cell that covers it. */
export const BUILD_CODE = listing(`
  function build(a) {                         //@ fn
    T = [0, ...a];      // each cell, its own //@ copy
    for (let i = 1; i <= n; i++) {            //@ loop
      const parent = i + (i & -i);            //@ body
      if (parent <= n) T[parent] += T[i];     //@ body
    }
  }
`);

/** Strip the lowest set bit: at most one cell per 1-bit of i. */
export const PREFIX_CODE = listing(`
  function prefixSum(i) {                     //@ fn
    let sum = 0;                              //@ init
    while (i > 0) {                           //@ loop
      sum += T[i];    // a(i - lowbit(i), i]  //@ body
      i -= i & -i;    // drop the lowest bit  //@ body
    }
    return sum;                               //@ done
  }
`);

/** Add the lowest set bit: visits exactly the cells whose range covers i. */
export const ADD_CODE = listing(`
  function add(i, delta) {                    //@ fn
    while (i <= n) {                          //@ loop
      T[i] += delta;  // this range moved     //@ body
      i += i & -i;    // climb to the next    //@ body
    }
  }
`);

export class FenwickTree {
  readonly n: number;
  private tree: number[];
  private values: number[];

  /** O(n) construction: each cell pushes its total to its parent i + lowbit(i). */
  constructor(values: number[]) {
    this.n = values.length;
    this.values = [0, ...values];
    this.tree = [0, ...values];
    for (let i = 1; i <= this.n; i++) {
      const parent = i + lowbit(i);
      if (parent <= this.n) this.tree[parent]! += this.tree[i]!;
    }
  }

  /** Inclusive 1-based range covered by T[i]. */
  static covers(i: number): [number, number] {
    return [i - lowbit(i) + 1, i];
  }

  cell(i: number): number {
    return this.tree[i]!;
  }

  value(i: number): number {
    return this.values[i]!;
  }

  private snapshot(
    steps: FenStep[],
    roles: Record<number, FenRole>,
    current: number | null,
    message: string,
    target: [number, number] | null,
    code: CodeCursor,
    acc?: number,
  ) {
    steps.push({ values: [...this.values], tree: [...this.tree], roles: { ...roles }, current, message, target, acc, code });
  }

  /** Sum of a[1..i]: strip the lowest set bit each step, so at most log2(n) + 1 cells. */
  prefixSum(i: number, steps: FenStep[] = [], label = `prefix(${i})`): number {
    const roles: Record<number, FenRole> = {};
    let acc = 0;
    const target: [number, number] = [1, i];
    this.snapshot(
      steps,
      roles,
      null,
      `${label}: sum of a[1..${i}]. Start at i = ${i} and strip the lowest set bit each step — one cell per 1-bit of ${i} (${bin(i)}).`,
      target,
      { listing: 'prefix', line: 'init', vars: { i, [`bin(${i})`]: bin(i), sum: 0 } },
      acc,
    );
    for (let k = i; k > 0; k -= lowbit(k)) {
      acc += this.tree[k]!;
      roles[k] = 'use';
      const [l, r] = FenwickTree.covers(k);
      const next = k - lowbit(k);
      this.snapshot(
        steps,
        roles,
        k,
        `i = ${k} (${bin(k)}): T[${k}] covers a[${l}..${r}] = ${this.tree[k]}. Sum so far ${acc}. Next i = ${k} − ${lowbit(k)} = ${next}.`,
        target,
        {
          listing: 'prefix',
          line: 'body',
          vars: { i: `${k} (${bin(k)})`, [`T[${k}]`]: this.tree[k]!, covers: `a[${l}..${r}]`, sum: acc, 'i →': next },
        },
        acc,
      );
    }
    this.snapshot(
      steps,
      roles,
      null,
      `${label} = ${acc}, using ${Object.keys(roles).length} cell(s) instead of ${i} additions.`,
      target,
      { listing: 'prefix', line: 'done', vars: { sum: acc, cells: Object.keys(roles).length } },
      acc,
    );
    return acc;
  }

  /** Sum of a[l..r] = prefix(r) − prefix(l − 1). */
  rangeSum(l: number, r: number): { result: number; steps: FenStep[] } {
    const steps: FenStep[] = [];
    const right = this.prefixSum(r, steps, `prefix(${r})`);
    const left = l > 1 ? this.prefixSum(l - 1, steps, `prefix(${l - 1})`) : 0;
    const result = right - left;
    this.snapshot(
      steps,
      {},
      null,
      `sum(a[${l}..${r}]) = prefix(${r}) − prefix(${l - 1}) = ${right} − ${left} = ${result}.`,
      [l, r],
      { listing: 'prefix', line: 'done', vars: { [`prefix(${r})`]: right, [`prefix(${l - 1})`]: left, result } },
      result,
    );
    return { result, steps };
  }

  /** a[i] += delta: add the lowest set bit each step to reach every cell whose range contains i. */
  add(i: number, delta: number): FenStep[] {
    const steps: FenStep[] = [];
    const roles: Record<number, FenRole> = {};
    this.values[i]! += delta;
    this.snapshot(
      steps,
      roles,
      null,
      `Add ${delta} to a[${i}]. Every T[k] whose range contains ${i} must change: climb by adding the lowest set bit.`,
      [i, i],
      { listing: 'add', line: 'fn', vars: { i, delta, n: this.n } },
    );
    for (let k = i; k <= this.n; k += lowbit(k)) {
      this.tree[k]! += delta;
      roles[k] = 'changed';
      const [l, r] = FenwickTree.covers(k);
      const next = k + lowbit(k);
      this.snapshot(
        steps,
        roles,
        k,
        `i = ${k} (${bin(k)}): T[${k}] covers a[${l}..${r}], add ${delta} → ${this.tree[k]}. Next i = ${k} + ${lowbit(k)} = ${next}${next > this.n ? ' (past the end, stop)' : ''}.`,
        [i, i],
        {
          listing: 'add',
          line: 'body',
          vars: { i: `${k} (${bin(k)})`, [`T[${k}]`]: this.tree[k]!, covers: `a[${l}..${r}]`, 'i →': next },
        },
      );
    }
    this.snapshot(steps, roles, null, `Done: updated ${Object.keys(roles).length} cell(s), one per level of the climb.`, [i, i], {
      listing: 'add',
      line: 'loop',
      vars: { updated: Object.keys(roles).length },
    });
    return steps;
  }
}

export function bin(i: number): string {
  return i.toString(2);
}
