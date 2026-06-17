// Recursive segment tree over an array, with tracing for visualization.
// Node 1 covers [0, n-1]; node k's children are 2k (left half) and 2k+1 (right half).

import { listing, type CodeCursor } from '../code';

export type Aggregate = 'sum' | 'min' | 'max';

export const AGGREGATES: Record<Aggregate, { combine: (a: number, b: number) => number; identity: number; label: string }> = {
  sum: { combine: (a, b) => a + b, identity: 0, label: 'sum' },
  min: { combine: Math.min, identity: Infinity, label: 'min' },
  max: { combine: Math.max, identity: -Infinity, label: 'max' },
};

/** partial: range overlaps the target, so we recurse. full: range is inside the query, so we use it whole.
 *  outside: disjoint, so we skip it. changed: value written or recomputed. */
export type SegRole = 'partial' | 'full' | 'outside' | 'changed';

export interface SegNode {
  id: number;
  l: number;
  r: number;
  depth: number;
}

export interface SegStep {
  /** Node values by id; null = not computed yet (during build). */
  values: (number | null)[];
  roles: Record<number, SegRole>;
  current: number | null;
  message: string;
  /** Query range or updated index, for highlighting array cells. */
  target: [number, number] | null;
  /** Line of the listing being executed, with the variables in scope there. */
  code: CodeCursor;
  /** Running answer for a query. */
  acc?: number;
}

/** O(n): every node is filled once, from the bottom up. */
export const BUILD_CODE = listing(`
  function build(id, l, r) {                  //@ fn
    if (l === r) {        // a single cell    //@ leaf
      tree[id] = a[l];                        //@ leaf
      return;                                 //@ leaf
    }
    const mid = (l + r) >> 1;                 //@ mid
    build(2 * id, l, mid);      // left half  //@ mid
    build(2 * id + 1, mid + 1, r);            //@ mid
    tree[id] = combine(tree[2 * id],          //@ pull
                       tree[2 * id + 1]);     //@ pull
  }
`);

/** O(log n): a query range is covered by at most 2 nodes per level. */
export const QUERY_CODE = listing(`
  function query(id, l, r, ql, qr) {          //@ fn
    if (qr < l || r < ql)   // no overlap     //@ outside
      return IDENTITY;      // contribute 0   //@ outside
    if (ql <= l && r <= qr) // fully inside   //@ full
      return tree[id];      // stop here      //@ full

    const mid = (l + r) >> 1;   // partial:   //@ split
    return combine(             // ask both   //@ split
      query(2 * id, l, mid, ql, qr),          //@ split
      query(2 * id + 1, mid + 1, r, ql, qr)); //@ split
  }
`);

/** O(log n): one leaf changes, so only its ancestors can be stale. */
export const UPDATE_CODE = listing(`
  function update(id, l, r, i, v) {           //@ fn
    if (l === r) {        // reached the leaf //@ leaf
      tree[id] = v;                           //@ leaf
      return;                                 //@ leaf
    }
    const mid = (l + r) >> 1;                 //@ down
    if (i <= mid) update(2*id, l, mid, i, v); //@ down
    else update(2*id + 1, mid + 1, r, i, v);  //@ down
    tree[id] = combine(tree[2 * id],  // fix  //@ pull
                       tree[2 * id + 1]);     //@ pull
  }
`);

export class SegmentTree {
  readonly n: number;
  readonly nodes: SegNode[] = [];
  readonly depth: number;
  /** Steps recorded while building, so the build itself can be replayed. */
  readonly buildTrace: SegStep[];
  private tree: number[];
  private readonly op: (typeof AGGREGATES)[Aggregate];

  constructor(
    values: number[],
    readonly aggregate: Aggregate = 'sum',
  ) {
    if (values.length === 0) throw new RangeError('empty array');
    this.n = values.length;
    this.op = AGGREGATES[aggregate];
    this.tree = new Array(4 * this.n).fill(this.op.identity);
    this.layout(1, 0, this.n - 1, 0);
    this.depth = Math.max(...this.nodes.map((nd) => nd.depth));
    this.buildTrace = this.build(values);
  }

  private layout(id: number, l: number, r: number, depth: number) {
    this.nodes.push({ id, l, r, depth });
    if (l === r) return;
    const mid = (l + r) >> 1;
    this.layout(2 * id, l, mid, depth + 1);
    this.layout(2 * id + 1, mid + 1, r, depth + 1);
  }

  get(id: number): number {
    return this.tree[id]!;
  }

  /** Current leaf values, i.e. the underlying array. */
  array(): number[] {
    const out = new Array<number>(this.n);
    for (const nd of this.nodes) if (nd.l === nd.r) out[nd.l] = this.tree[nd.id]!;
    return out;
  }

  private fmt(v: number): string {
    return v === Infinity ? '∞' : v === -Infinity ? '−∞' : String(v);
  }

  /** Build bottom-up by recursion: O(n). Returns the steps. */
  private build(values: number[]): SegStep[] {
    const steps: SegStep[] = [];
    const computed = new Set<number>();
    const roles: Record<number, SegRole> = {};
    const snap = (current: number | null, message: string, code: CodeCursor) =>
      steps.push({
        values: this.tree.map((v, id) => (computed.has(id) ? v : null)),
        roles: { ...roles },
        current,
        message,
        target: null,
        code: { listing: 'build', ...code },
      });

    snap(null, `Build a ${this.op.label} tree over ${this.n} values. Each node stores the ${this.op.label} of its range.`, {
      line: 'fn',
      vars: { id: 1, l: 0, r: this.n - 1 },
    });
    const build = (id: number, l: number, r: number) => {
      if (l === r) {
        this.tree[id] = values[l]!;
        computed.add(id);
        roles[id] = 'changed';
        snap(id, `Leaf [${l}] stores a[${l}] = ${values[l]}.`, {
          line: 'leaf',
          vars: { id, l, r, [`tree[${id}]`]: values[l]! },
        });
        return;
      }
      const mid = (l + r) >> 1;
      build(2 * id, l, mid);
      build(2 * id + 1, mid + 1, r);
      const a = this.tree[2 * id]!;
      const b = this.tree[2 * id + 1]!;
      this.tree[id] = this.op.combine(a, b);
      computed.add(id);
      roles[id] = 'changed';
      snap(id, `Both children of [${l}..${r}] are ready, so this node is ${this.op.label}(${a}, ${b}) = ${this.tree[id]}.`, {
        line: 'pull',
        vars: { id, l, r, [`tree[${2 * id}]`]: a, [`tree[${2 * id + 1}]`]: b, [`tree[${id}]`]: this.tree[id]! },
      });
    };
    build(1, 0, this.n - 1);
    for (const id of Object.keys(roles)) delete roles[+id];
    snap(null, `Built ${this.nodes.length} nodes. The root holds the ${this.op.label} of the whole array: ${this.tree[1]}.`, {
      line: 'fn',
      vars: { nodes: this.nodes.length, root: this.tree[1]! },
    });
    return steps;
  }

  /** Aggregate over a[ql..qr]. Visits O(log n) nodes. */
  query(ql: number, qr: number): { result: number; steps: SegStep[] } {
    const steps: SegStep[] = [];
    const roles: Record<number, SegRole> = {};
    let acc = this.op.identity;
    const snap = (current: number | null, message: string, code: CodeCursor) =>
      steps.push({
        values: [...this.tree],
        roles: { ...roles },
        current,
        message,
        target: [ql, qr],
        acc,
        code: { listing: 'query', ...code },
      });

    snap(null, `Query the ${this.op.label} of a[${ql}..${qr}], starting at the root.`, {
      line: 'fn',
      vars: { ql, qr, id: 1, l: 0, r: this.n - 1 },
    });
    const walk = (id: number, l: number, r: number): number => {
      if (qr < l || r < ql) {
        roles[id] = 'outside';
        snap(id, `[${l}..${r}] doesn't overlap [${ql}..${qr}]: skip it, and with it everything below it.`, {
          line: 'outside',
          vars: { id, l, r, ql, qr },
        });
        return this.op.identity;
      }
      if (ql <= l && r <= qr) {
        roles[id] = 'full';
        acc = this.op.combine(acc, this.tree[id]!);
        snap(
          id,
          `[${l}..${r}] is fully inside [${ql}..${qr}]: take its stored value ${this.tree[id]} and stop — this is the shortcut the tree exists for. Running ${this.op.label} = ${this.fmt(acc)}.`,
          {
            line: 'full',
            vars: { id, l, r, [`tree[${id}]`]: this.tree[id]!, [this.op.label]: this.fmt(acc) },
          },
        );
        return this.tree[id]!;
      }
      roles[id] = 'partial';
      snap(id, `[${l}..${r}] only partly overlaps [${ql}..${qr}]: split, and ask both children.`, {
        line: 'split',
        vars: { id, l, r, mid: (l + r) >> 1 },
      });
      const mid = (l + r) >> 1;
      return this.op.combine(walk(2 * id, l, mid), walk(2 * id + 1, mid + 1, r));
    };
    const result = walk(1, 0, this.n - 1);
    const visited = Object.keys(roles).length;
    snap(null, `${this.op.label}(a[${ql}..${qr}]) = ${this.fmt(result)}, found by visiting ${visited} of ${this.nodes.length} nodes.`, {
      line: 'fn',
      vars: { result: this.fmt(result), visited, of: this.nodes.length },
    });
    return { result, steps };
  }

  /** Set a[index] = value and fix every ancestor. O(log n). */
  update(index: number, value: number): SegStep[] {
    const steps: SegStep[] = [];
    const roles: Record<number, SegRole> = {};
    const snap = (current: number | null, message: string, code: CodeCursor) =>
      steps.push({
        values: [...this.tree],
        roles: { ...roles },
        current,
        message,
        target: [index, index],
        code: { listing: 'update', ...code },
      });

    snap(null, `Set a[${index}] = ${value}. Walk down to its leaf, then fix each ancestor on the way back up.`, {
      line: 'fn',
      vars: { i: index, v: value, id: 1, l: 0, r: this.n - 1 },
    });
    const walk = (id: number, l: number, r: number) => {
      if (l === r) {
        const old = this.tree[id];
        this.tree[id] = value;
        roles[id] = 'changed';
        snap(id, `Leaf [${l}]: ${old} → ${value}.`, { line: 'leaf', vars: { id, l, r, old: old!, new: value } });
        return;
      }
      const mid = (l + r) >> 1;
      const goLeft = index <= mid;
      roles[id] = 'partial';
      snap(
        id,
        `${index} lies in the ${goLeft ? 'left' : 'right'} half of [${l}..${r}], so only that child can be affected: go ${goLeft ? 'left' : 'right'}.`,
        {
          line: 'down',
          vars: { id, l, r, mid, i: index, next: goLeft ? 2 * id : 2 * id + 1 },
        },
      );
      if (goLeft) walk(2 * id, l, mid);
      else walk(2 * id + 1, mid + 1, r);
      const a = this.tree[2 * id]!;
      const b = this.tree[2 * id + 1]!;
      this.tree[id] = this.op.combine(a, b);
      roles[id] = 'changed';
      snap(id, `Recompute [${l}..${r}] = ${this.op.label}(${a}, ${b}) = ${this.tree[id]}.`, {
        line: 'pull',
        vars: { id, l, r, [`tree[${2 * id}]`]: a, [`tree[${2 * id + 1}]`]: b, [`tree[${id}]`]: this.tree[id]! },
      });
    };
    walk(1, 0, this.n - 1);
    snap(null, `Done: ${Object.keys(roles).length} nodes touched, one per level — that's why an update is O(log n).`, {
      line: 'fn',
      vars: { touched: Object.keys(roles).length, levels: this.depth + 1 },
    });
    return steps;
  }
}
