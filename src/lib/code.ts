// Source listings that a visualization can highlight line by line.
//
// Write the code the way a reader should see it and tag the interesting lines with a
// trailing `//@ name` marker. The marker is stripped from the display text, so the trace
// refers to lines by name and stays correct when the listing is reformatted:
//
//   const CODE = listing(`
//     function f(a) {          //@ enter
//       return a + 1;          //@ ret
//     }
//   `);
//   rec.push({ line: 'ret', vars: { a } });
//
// One marker may tag several lines (repeat the name); a statement wrapped over two lines
// then highlights as a unit.

export interface CodeListing {
  /** Display lines, dedented, markers removed. */
  lines: string[];
  /** Marker name → the line indices it tags. */
  at: Record<string, number[]>;
}

/** Where in a listing a trace step is, and what the variables hold there. */
export interface CodeCursor {
  /** Which listing, for algorithms that show more than one (build / query / update). */
  listing?: string;
  /** Marker name of the line being executed. */
  line: string;
  /** Live values to show beneath the code, in display order. */
  vars?: Record<string, string | number>;
}

const MARKER = /\s*\/\/@[ \t]+([\w\-. ]+?)[ \t]*$/;

export function listing(src: string): CodeListing {
  const raw = src
    .replace(/^\r?\n/, '')
    .replace(/\s+$/, '')
    .split('\n');
  const indents = raw.filter((l) => l.trim()).map((l) => l.length - l.trimStart().length);
  const pad = indents.length ? Math.min(...indents) : 0;

  const lines: string[] = [];
  const at: Record<string, number[]> = {};
  for (const line of raw) {
    const m = line.match(MARKER);
    const text = m ? line.slice(0, line.length - m[0].length) : line;
    const i = lines.push(text.slice(pad).trimEnd()) - 1;
    if (m) for (const name of m[1]!.trim().split(/\s+/)) (at[name] ||= []).push(i);
  }
  return { lines, at };
}

// ---------- Minimal tokenizer, so the panel reads as code without shipping a highlighter ----------

export type TokenKind = 'com' | 'str' | 'kw' | 'num' | 'fn' | 'txt';
export interface Token {
  kind: TokenKind;
  text: string;
}

const KEYWORDS = new Set([
  'break',
  'case',
  'class',
  'const',
  'continue',
  'do',
  'else',
  'export',
  'for',
  'function',
  'if',
  'in',
  'let',
  'new',
  'of',
  'return',
  'switch',
  'this',
  'null',
  'true',
  'false',
  'var',
  'while',
]);

const RULES: [TokenKind, RegExp][] = [
  ['com', /\/\/[^\n]*/y],
  ['str', /'[^']*'|"[^"]*"|`[^`]*`/y],
  ['num', /\b\d+(?:\.\d+)?\b/y],
  ['fn', /\b[A-Za-z_$][\w$]*(?=\s*\()/y],
  ['txt', /\b[A-Za-z_$][\w$]*\b/y],
];

/** Split one line into coloured runs. Unknown characters accumulate as plain text. */
export function tokenize(line: string): Token[] {
  const out: Token[] = [];
  const push = (kind: TokenKind, text: string) => {
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.text += text;
    else out.push({ kind, text });
  };

  let i = 0;
  outer: while (i < line.length) {
    for (const [kind, rx] of RULES) {
      rx.lastIndex = i;
      const m = rx.exec(line);
      if (!m) continue;
      const text = m[0];
      push(kind === 'txt' && KEYWORDS.has(text) ? 'kw' : kind === 'fn' && KEYWORDS.has(text) ? 'kw' : kind, text);
      i += text.length;
      continue outer;
    }
    push('txt', line[i]!);
    i++;
  }
  return out;
}
