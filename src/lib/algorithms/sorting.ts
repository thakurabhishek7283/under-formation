// Merge sort and quick sort that record every step, so a visualization can replay them
// next to the source: each step names the line it is on and the values in scope there.

import { listing, type CodeCursor } from '../code';

export type BarRole = 'compare' | 'change' | 'pivot';

export interface SortStep {
  array: number[];
  /** Per-index highlight for this step. */
  roles: Record<number, BarRole>;
  /** Indices known to be in their final position. */
  final: number[];
  /** Subarray currently being worked on (inclusive). */
  range: [number, number] | null;
  comparisons: number;
  /** Array writes (merge sort) or swaps (quick sort). */
  moves: number;
  message: string;
  /** Line of the listing being executed, with the variables in scope there. */
  code: CodeCursor;
  /** Recursion depth, counting the first call as 0. */
  depth: number;
  /** Merge sort: the copy of a[lo..hi] being merged back, with the two read heads. */
  buffer?: { lo: number; mid: number; values: number[]; i: number; j: number; taken?: number };
  /** Quick sort: partition pointers. a[lo..i-1] < pivot; j is the scan position. */
  partition?: { i: number; j: number | null; pivot: number };
}

export const MERGE_SORT_CODE = listing(`
  function mergeSort(a, lo, hi) {             //@ fn
    if (lo >= hi) return;   // 1 item: sorted //@ base
    const mid = (lo + hi) >> 1;               //@ mid
    mergeSort(a, lo, mid);                    //@ left
    mergeSort(a, mid + 1, hi);                //@ right
    merge(a, lo, mid, hi);  // all the work   //@ call
  }

  function merge(a, lo, mid, hi) {            //@ m-fn
    const buf = a.slice(lo, hi + 1);          //@ copy
    let i = lo, j = mid + 1; // two read heads //@ copy

    for (let k = lo; k <= hi; k++) {          //@ loop
      // '<=' favours the left half: stable   //@ compare
      const takeLeft = j > hi ||              //@ compare
        (i <= mid && buf[i-lo] <= buf[j-lo]); //@ compare
      a[k] = takeLeft ? buf[i++ - lo]         //@ write
                      : buf[j++ - lo];        //@ write
    }
  }
`);

export const QUICK_SORT_CODE = listing(`
  function quickSort(a, lo, hi) {             //@ fn
    if (lo >= hi) return;     // 0-1: done    //@ base
    const p = partition(a, lo, hi);           //@ call
    quickSort(a, lo, p - 1);  // smaller side //@ left
    quickSort(a, p + 1, hi);  // larger side  //@ right
  }

  function partition(a, lo, hi) {             //@ p-fn
    const p = choosePivot(a, lo, hi);         //@ choose
    swap(a, p, hi);           // park at end  //@ choose
    const pivot = a[hi];                      //@ start
    let i = lo;         // a[lo..i-1] < pivot //@ start

    for (let j = lo; j < hi; j++) {           //@ loop
      if (a[j] < pivot) {                     //@ compare
        swap(a, i, j);                        //@ swap
        i++;              // that region grew //@ swap
      }
    }
    swap(a, i, hi);        // its final spot  //@ place
    return i;                                 //@ place
  }
`);

class Recorder {
  steps: SortStep[] = [];
  comparisons = 0;
  moves = 0;
  depth = 0;
  final = new Set<number>();

  constructor(readonly a: number[]) {}

  push(
    message: string,
    code: CodeCursor,
    roles: Record<number, BarRole> = {},
    extra: Partial<Pick<SortStep, 'range' | 'buffer' | 'partition'>> = {},
  ) {
    this.steps.push({
      array: [...this.a],
      roles,
      final: [...this.final],
      range: extra.range ?? null,
      comparisons: this.comparisons,
      moves: this.moves,
      message,
      code,
      depth: this.depth,
      buffer: extra.buffer ? { ...extra.buffer, values: [...extra.buffer.values] } : undefined,
      partition: extra.partition,
    });
  }

  finish(message: string, line: string) {
    this.a.forEach((_, i) => this.final.add(i));
    this.depth = 0;
    this.push(message, { line, vars: { comparisons: this.comparisons, moves: this.moves } });
    return this.steps;
  }
}

/** Top-down merge sort. Stable; always ~n log2 n comparisons. */
export function mergeSortSteps(input: number[]): SortStep[] {
  const a = [...input];
  const rec = new Recorder(a);
  rec.push(`Start: ${a.length} items. Merge sort splits in half until pieces have one item, then merges.`, {
    line: 'fn',
    vars: { lo: 0, hi: a.length - 1 },
  });

  const sort = (lo: number, hi: number) => {
    if (lo >= hi) {
      if (lo === hi) {
        rec.push(
          `[${lo}..${hi}] is a single item, ${a[lo]}: sorted by definition, so this call returns immediately.`,
          { line: 'base', vars: { lo, hi, depth: rec.depth } },
          {},
          { range: [lo, hi] },
        );
      }
      return;
    }
    const mid = (lo + hi) >> 1;
    rec.push(
      `Split [${lo}..${hi}] into [${lo}..${mid}] and [${mid + 1}..${hi}], and sort each half before merging.`,
      { line: 'mid', vars: { lo, hi, mid, depth: rec.depth } },
      {},
      { range: [lo, hi] },
    );
    rec.depth++;
    sort(lo, mid);
    sort(mid + 1, hi);
    rec.depth--;
    merge(lo, mid, hi);
  };

  const merge = (lo: number, mid: number, hi: number) => {
    const values = a.slice(lo, hi + 1);
    let i = lo;
    let j = mid + 1;
    const buffer = () => ({ lo, mid, values, i, j });
    rec.push(
      `Merge the sorted halves [${lo}..${mid}] and [${mid + 1}..${hi}]: copy them aside, then repeatedly take the smaller front item.`,
      { line: 'copy', vars: { lo, mid, hi, i, j, depth: rec.depth } },
      {},
      { range: [lo, hi], buffer: buffer() },
    );

    for (let k = lo; k <= hi; k++) {
      const left = i <= mid ? values[i - lo]! : null;
      const right = j <= hi ? values[j - lo]! : null;
      const vars = () => ({ k, i, j, 'buf[i]': left ?? '–', 'buf[j]': right ?? '–' });

      let fromLeft: boolean;
      if (left !== null && right !== null) {
        rec.comparisons++;
        fromLeft = left <= right; // `<=` keeps equal items in order: stability
        rec.push(
          `Compare the front items ${left} and ${right}: ${fromLeft ? `${left} ≤ ${right}, so the left half wins` : `${right} < ${left}, so the right half wins`}.`,
          { line: 'compare', vars: vars() },
          {},
          { range: [lo, hi], buffer: buffer() },
        );
      } else {
        fromLeft = left !== null;
        rec.push(
          `The ${left !== null ? 'right' : 'left'} half is used up, so what remains on the ${fromLeft ? 'left' : 'right'} is already in order: copy it down as is.`,
          { line: 'compare', vars: vars() },
          {},
          { range: [lo, hi], buffer: buffer() },
        );
      }

      const taken = fromLeft ? i : j;
      const value = values[taken - lo]!;
      a[k] = value;
      rec.moves++;
      rec.push(
        `Write ${value} into position ${k} and advance the ${fromLeft ? 'left' : 'right'} read head.`,
        { line: 'write', vars: { ...vars(), [`a[${k}]`]: value } },
        { [k]: 'change' },
        { range: [lo, hi], buffer: { ...buffer(), taken } },
      );
      if (fromLeft) i++;
      else j++;
    }
  };

  sort(0, a.length - 1);
  return rec.finish(`Sorted with ${rec.comparisons} comparisons and ${rec.moves} writes.`, 'fn');
}

export type PivotStrategy = 'last' | 'middle' | 'median-of-three' | 'random';

/** Quick sort with Lomuto partitioning. Average ~1.39 n log2 n comparisons; worst case n²/2. */
export function quickSortSteps(input: number[], strategy: PivotStrategy = 'last'): SortStep[] {
  const a = [...input];
  const rec = new Recorder(a);
  rec.push(`Start: ${a.length} items. Quick sort picks a pivot, partitions around it, then recurses on each side.`, {
    line: 'fn',
    vars: { lo: 0, hi: a.length - 1 },
  });

  const swap = (x: number, y: number) => {
    [a[x], a[y]] = [a[y]!, a[x]!];
    rec.moves++;
  };

  const choosePivot = (lo: number, hi: number): number => {
    if (strategy === 'last') return hi;
    if (strategy === 'middle') return (lo + hi) >> 1;
    if (strategy === 'random') return lo + Math.floor(Math.random() * (hi - lo + 1));
    const mid = (lo + hi) >> 1;
    const trio = [lo, mid, hi].sort((x, y) => a[x]! - a[y]!);
    return trio[1]!;
  };

  const sort = (lo: number, hi: number) => {
    if (lo > hi) return;
    if (lo === hi) {
      rec.final.add(lo);
      rec.push(
        `[${lo}..${hi}] has a single item, ${a[lo]}: already in place, so this call returns.`,
        { line: 'base', vars: { lo, hi, depth: rec.depth } },
        {},
        { range: [lo, hi] },
      );
      return;
    }

    const p = choosePivot(lo, hi);
    rec.push(
      p === hi
        ? `Take ${a[p]} (position ${p}) as the pivot; it already sits at the end of [${lo}..${hi}].`
        : `Pick ${a[p]} (position ${p}) as the pivot and swap it to the end, where partitioning expects it.`,
      { line: 'choose', vars: { lo, hi, p, pivot: a[p]!, depth: rec.depth } },
      { [p]: 'pivot', ...(p === hi ? {} : { [hi]: 'change' }) },
      { range: [lo, hi] },
    );
    if (p !== hi) swap(p, hi);

    const pivot = a[hi]!;
    let i = lo;
    rec.push(
      `Partition [${lo}..${hi}] around pivot ${pivot}. Everything smaller than it collects to the left of i.`,
      { line: 'start', vars: { lo, hi, pivot, i, depth: rec.depth } },
      { [hi]: 'pivot' },
      { range: [lo, hi], partition: { i, j: lo, pivot: hi } },
    );

    for (let j = lo; j < hi; j++) {
      rec.comparisons++;
      const v = a[j]!;
      const vars = { lo, hi, pivot, i, j, 'a[j]': v };
      if (v < pivot) {
        if (i !== j) {
          swap(i, j);
          rec.push(
            `${v} < ${pivot}: swap it with position ${i} so the "smaller" region grows by one.`,
            { line: 'swap', vars: { ...vars, i: i + 1 } },
            { [i]: 'change', [j]: 'change', [hi]: 'pivot' },
            { range: [lo, hi], partition: { i: i + 1, j, pivot: hi } },
          );
        } else {
          rec.push(
            `${v} < ${pivot}: it already sits at the edge of the "smaller" region, so only i moves.`,
            { line: 'swap', vars: { ...vars, i: i + 1 } },
            { [j]: 'compare', [hi]: 'pivot' },
            { range: [lo, hi], partition: { i: i + 1, j, pivot: hi } },
          );
        }
        i++;
      } else {
        rec.push(
          `${v} ≥ ${pivot}: leave it where it is. i stays put, so the item joins the "larger" region.`,
          { line: 'compare', vars },
          { [j]: 'compare', [hi]: 'pivot' },
          { range: [lo, hi], partition: { i, j, pivot: hi } },
        );
      }
    }

    swap(i, hi);
    rec.final.add(i);
    rec.push(
      `Swap the pivot ${pivot} into position ${i}. Everything left of it is smaller and everything right is larger, so ${pivot} is settled for good and the two sides can be sorted independently.`,
      { line: 'place', vars: { lo, hi, pivot, i, depth: rec.depth } },
      { [i]: 'change' },
      { range: [lo, hi] },
    );

    rec.depth++;
    sort(lo, i - 1);
    sort(i + 1, hi);
    rec.depth--;
  };

  sort(0, a.length - 1);
  return rec.finish(`Sorted with ${rec.comparisons} comparisons and ${rec.moves} swaps.`, 'fn');
}
