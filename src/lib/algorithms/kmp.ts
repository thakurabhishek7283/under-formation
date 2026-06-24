// Knuth–Morris–Pratt string search, with tracing for visualization.

import { listing, type CodeCursor } from '../code';

export interface KmpStep {
  phase: 'table' | 'search';
  /** Prefix-function values; null = not computed yet. */
  pi: (number | null)[];
  /** Table phase: index in the pattern. Search phase: index in the text. */
  i: number;
  /** Pattern index being compared against position i (in the table phase, the current border length k). */
  j: number;
  /** Outcome of this step's comparison, if one happened. */
  outcome: 'match' | 'mismatch' | 'found' | null;
  matches: number[];
  comparisons: number;
  message: string;
  /** Line of the listing being executed, with the variables in scope there. */
  code: CodeCursor;
}

/** Phase 1. The pattern is matched against itself, one index at a time. */
export const PREFIX_CODE = listing(`
  function prefixTable(p) {                   //@ fn
    const pi = [0];    // 1 char has no border //@ init
    let k = 0;         // the border's length //@ init

    for (let i = 1; i < p.length; i++) {      //@ loop
      while (k > 0 && p[i] !== p[k])          //@ shrink
        k = pi[k - 1]; // try a shorter one   //@ shrink
      if (p[i] === p[k]) k++;  // it grew     //@ grow
      pi[i] = k;                              //@ store
    }
    return pi;                                //@ done
  }
`);

/** Phase 2. i only ever moves forward; mismatches move j instead. */
export const SEARCH_CODE = listing(`
  function search(t, p, pi) {                 //@ fn
    let j = 0;          // chars matched now  //@ init

    for (let i = 0; i < t.length; i++) {      //@ loop
      while (j > 0 && t[i] !== p[j])          //@ shrink
        j = pi[j - 1]; // slide, never rewind //@ shrink
      if (t[i] === p[j]) j++;                 //@ advance
      if (j === p.length) {    // found one   //@ found
        report(i - p.length + 1);             //@ found
        j = pi[j - 1]; // keep the overlap    //@ found
      }
    }
  }
`);

/**
 * π[i] = length of the longest proper prefix of p[0..i] that is also a suffix of it.
 * On a mismatch after matching j characters, π[j−1] tells us how many we still have matched.
 */
export function prefixFunction(p: string, steps: KmpStep[] = []): number[] {
  const m = p.length;
  const pi: (number | null)[] = new Array(m).fill(null);
  if (m === 0) return [];
  pi[0] = 0;
  let comparisons = 0;
  const push = (i: number, j: number, outcome: KmpStep['outcome'], message: string, code: CodeCursor) =>
    steps.push({ phase: 'table', pi: [...pi], i, j, outcome, matches: [], comparisons, message, code });

  push(0, 0, null, `Build the prefix table π for "${p}". π[0] = 0: a single character has no proper border.`, {
    listing: 'table',
    line: 'init',
    vars: { p: `"${p}"`, k: 0, 'π[0]': 0 },
  });

  let k = 0;
  for (let i = 1; i < m; i++) {
    for (;;) {
      comparisons++;
      const vars = { i, k, 'p[i]': `'${p[i]}'`, 'p[k]': `'${p[k]}'` };
      if (p[i] === p[k]) {
        k++;
        pi[i] = k;
        push(i, k - 1, 'match', `p[${i}] = p[${k - 1}] = '${p[i]}': the border grows to ${k}, so π[${i}] = ${k}.`, {
          listing: 'table',
          line: 'grow',
          vars: { ...vars, k, [`π[${i}]`]: k },
        });
        break;
      }
      if (k === 0) {
        pi[i] = 0;
        push(i, 0, 'mismatch', `p[${i}] = '${p[i]}' ≠ p[0] = '${p[0]}' and there's no shorter border left to try, so π[${i}] = 0.`, {
          listing: 'table',
          line: 'store',
          vars: { ...vars, [`π[${i}]`]: 0 },
        });
        break;
      }
      const fallback = pi[k - 1]!;
      push(
        i,
        k,
        'mismatch',
        `p[${i}] = '${p[i]}' ≠ p[${k}] = '${p[k]}': fall back to the next-shorter border, k = π[${k - 1}] = ${fallback}.`,
        {
          listing: 'table',
          line: 'shrink',
          vars: { ...vars, [`π[${k - 1}]`]: fallback, 'k →': fallback },
        },
      );
      k = fallback;
    }
  }
  push(m - 1, k, null, `Prefix table complete: π = [${pi.join(', ')}].`, {
    listing: 'table',
    line: 'done',
    vars: { π: `[${pi.join(', ')}]` },
  });
  return pi as number[];
}

/** All start positions of `pattern` in `text`, in O(n + m) comparisons. */
export function kmpSearch(text: string, pattern: string): { matches: number[]; steps: KmpStep[]; comparisons: number } {
  const steps: KmpStep[] = [];
  const m = pattern.length;
  if (m === 0) return { matches: [], steps, comparisons: 0 };
  const pi = prefixFunction(pattern, steps);
  const matches: number[] = [];
  let comparisons = 0;
  const push = (i: number, j: number, outcome: KmpStep['outcome'], message: string, code: CodeCursor) =>
    steps.push({ phase: 'search', pi, i, j, outcome, matches: [...matches], comparisons, message, code });

  push(0, 0, null, `Search for "${pattern}" in the text. The text pointer i never moves backwards.`, {
    listing: 'search',
    line: 'init',
    vars: { j: 0, π: `[${pi.join(', ')}]` },
  });

  let j = 0;
  for (let i = 0; i < text.length; i++) {
    for (;;) {
      comparisons++;
      const vars = { i, j, 't[i]': `'${text[i]}'`, 'p[j]': `'${pattern[j]}'` };
      if (text[i] === pattern[j]) {
        if (j === m - 1) {
          matches.push(i - m + 1);
          const next = pi[m - 1]!;
          push(
            i,
            j,
            'found',
            `Match at position ${i - m + 1}! Keep the ${next} character(s) that are also a prefix (j = π[${m - 1}] = ${next}) and carry on from here.`,
            {
              listing: 'search',
              line: 'found',
              vars: { ...vars, 'match at': i - m + 1, 'j →': next },
            },
          );
          j = next;
        } else {
          push(i, j, 'match', `t[${i}] = p[${j}] = '${text[i]}': advance both pointers.`, {
            listing: 'search',
            line: 'advance',
            vars: { ...vars, 'j →': j + 1 },
          });
          j++;
        }
        break;
      }
      if (j === 0) {
        push(
          i,
          0,
          'mismatch',
          `t[${i}] = '${text[i]}' ≠ p[0] = '${pattern[0]}': nothing was matched, so just move to the next character.`,
          {
            listing: 'search',
            line: 'loop',
            vars: { ...vars, 'i →': i + 1 },
          },
        );
        break;
      }
      const next = pi[j - 1]!;
      push(
        i,
        j,
        'mismatch',
        `t[${i}] = '${text[i]}' ≠ p[${j}] = '${pattern[j]}': slide the pattern so j = π[${j - 1}] = ${next}. No text character is re-read.`,
        {
          listing: 'search',
          line: 'shrink',
          vars: { ...vars, [`π[${j - 1}]`]: next, 'j →': next },
        },
      );
      j = next;
    }
  }
  push(
    text.length - 1,
    j,
    null,
    matches.length
      ? `Done: ${matches.length} match(es) at [${matches.join(', ')}], using ${comparisons} comparisons.`
      : `Done: no matches, using ${comparisons} comparisons.`,
    { listing: 'search', line: 'loop', vars: { comparisons, matches: matches.length } },
  );
  return { matches, steps, comparisons };
}

/** Comparisons made by the brute-force "try every shift" search, for contrast. */
export function naiveComparisons(text: string, pattern: string): number {
  let count = 0;
  for (let s = 0; s + pattern.length <= text.length; s++) {
    for (let j = 0; j < pattern.length; j++) {
      count++;
      if (text[s + j] !== pattern[j]) break;
    }
  }
  return count;
}
