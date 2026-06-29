// Byte-pair encoding (Sennrich et al. 2016), the tokenizer family behind GPT, Llama and most LLMs.
// Characters are the base vocabulary; training repeatedly merges the most frequent adjacent pair.
// "▁" marks the start of a word (SentencePiece convention) so "▁the" and "the" differ.

export const WORD_START = '▁';

export interface BpeMerge {
  pair: [string, string];
  token: string;
  count: number;
}

export interface BpeWord {
  symbols: string[];
  freq: number;
}

export interface BpeStep {
  merge: BpeMerge | null;
  vocab: string[];
  words: BpeWord[];
  /** Total tokens needed to write the whole corpus at this point. */
  corpusTokens: number;
  message: string;
}

function pretokenize(text: string): string[] {
  return text.split(/\s+/).filter(Boolean).map((w) => WORD_START + w);
}

export function trainBpe(corpus: string, maxMerges: number): { merges: BpeMerge[]; steps: BpeStep[] } {
  // Word types in first-appearance order keep ties deterministic.
  const freq = new Map<string, number>();
  for (const w of pretokenize(corpus)) freq.set(w, (freq.get(w) ?? 0) + 1);
  const words: BpeWord[] = [...freq].map(([w, f]) => ({ symbols: [...w], freq: f }));

  const vocab = [...new Set(words.flatMap((w) => w.symbols))].sort();
  const merges: BpeMerge[] = [];
  const steps: BpeStep[] = [];
  const total = () => words.reduce((s, w) => s + w.symbols.length * w.freq, 0);
  const snap = (merge: BpeMerge | null, message: string) =>
    steps.push({
      merge,
      vocab: [...vocab],
      words: words.map((w) => ({ symbols: [...w.symbols], freq: w.freq })),
      corpusTokens: total(),
      message,
    });

  snap(null, `Start from ${vocab.length} single characters. The corpus is ${total()} tokens long.`);

  for (let m = 0; m < maxMerges; m++) {
    const counts = new Map<string, { pair: [string, string]; count: number }>();
    for (const w of words)
      for (let i = 0; i < w.symbols.length - 1; i++) {
        const pair: [string, string] = [w.symbols[i]!, w.symbols[i + 1]!];
        const key = pair.join('\u0000');
        const entry = counts.get(key) ?? { pair, count: 0 };
        entry.count += w.freq;
        counts.set(key, entry);
      }
    let best: { pair: [string, string]; count: number } | null = null;
    for (const c of counts.values()) if (!best || c.count > best.count) best = c;
    if (!best || best.count < 2) {
      snap(null, `No pair occurs more than once. Training stops at ${vocab.length} tokens.`);
      break;
    }

    const token = best.pair[0] + best.pair[1];
    for (const w of words) w.symbols = mergePair(w.symbols, best.pair, token);
    const merge = { pair: best.pair, token, count: best.count };
    merges.push(merge);
    vocab.push(token);
    snap(
      merge,
      `Merge #${m + 1}: "${best.pair[0]}" + "${best.pair[1]}" appear together ${best.count}× → new token "${token}".`,
    );
  }
  return { merges, steps };
}

function mergePair(symbols: string[], pair: [string, string], token: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < symbols.length; i++) {
    if (i < symbols.length - 1 && symbols[i] === pair[0] && symbols[i + 1] === pair[1]) {
      out.push(token);
      i++;
    } else out.push(symbols[i]!);
  }
  return out;
}

export interface EncodedToken {
  text: string;
  /** Index in the vocabulary, or -1 for a character never seen in training. */
  id: number;
}

/** Tokenize new text: apply the learned merges in the order they were learned (lowest rank first). */
export function encode(text: string, merges: BpeMerge[], vocab: string[]): EncodedToken[] {
  const rank = new Map(merges.map((m, i) => [m.pair.join('\u0000'), i]));
  const ids = new Map(vocab.map((t, i) => [t, i]));
  const out: EncodedToken[] = [];
  for (const word of pretokenize(text)) {
    let symbols = [...word];
    for (;;) {
      let bestRank = Infinity;
      let bestPair: [string, string] | null = null;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = rank.get(symbols[i] + '\u0000' + symbols[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestPair = [symbols[i]!, symbols[i + 1]!];
        }
      }
      if (!bestPair) break;
      symbols = mergePair(symbols, bestPair, bestPair[0] + bestPair[1]);
    }
    for (const s of symbols) out.push({ text: s, id: ids.get(s) ?? -1 });
  }
  return out;
}

