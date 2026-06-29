// Scaled dot-product self-attention with a few hand-built heads whose behavior is readable.
// The math is the real thing: Q = XW_q, K = XW_k, V = XW_v, A = softmax(QKᵀ/√d + mask), out = AV.
// Only the weights are designed by hand instead of learned, so each head does one recognizable job
// (patterns that interpretability research has found in trained models).

import { murmur3 } from '../algorithms/hash';
import { matmul, randomMatrix, rng, softmax, transpose, zeros, type Matrix } from './math';

export const WORD_CLASSES = ['det', 'noun', 'verb', 'prep', 'adj', 'pron'] as const;
export type WordClass = (typeof WORD_CLASSES)[number];

const LEXICON: Record<string, WordClass> = {};
const add = (cls: WordClass, words: string) => words.split(' ').forEach((w) => (LEXICON[w] = cls));
add('det', 'the a an this that these those my your his her its our their some every');
add('noun', 'cat mat dog bird fish house tree car city river park king queen man woman child sun moon book table chair food water ball floor garden ship road');
add('verb', 'sat ran is was are were saw ate chased sleeps sleep jumped runs eats sees likes loves went barked flew read wrote has had');
add('prep', 'on in at under over with to from by near into of behind');
add('adj', 'big small red old young happy lazy quick brown tall green black white little fast');
add('pron', 'he she it they we i you him them us');

export function wordClass(word: string): WordClass {
  const w = word.toLowerCase();
  if (LEXICON[w]) return LEXICON[w];
  if (w.endsWith('ed') || w.endsWith('ing')) return 'verb';
  if (w.endsWith('ly')) return 'adj';
  return 'noun';
}

// Embedding layout: [bias | class one-hot (6) | word identity (8) | position (8)]
const BIAS = 0;
const CLS = 1;
const ID = CLS + WORD_CLASSES.length;
const POS = ID + 8;
export const D_MODEL = POS + 8;
export const D_HEAD = 8;
const FREQS = [1.2, 0.8, 0.5, 0.3];

function identityVector(word: string): number[] {
  const r = rng(murmur3(word.toLowerCase()));
  const v = Array.from({ length: 8 }, () => r() * 2 - 1);
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
}

export function embed(tokens: string[]): Matrix {
  return tokens.map((tok, pos) => {
    const x = new Array<number>(D_MODEL).fill(0);
    x[BIAS] = 1;
    x[CLS + WORD_CLASSES.indexOf(wordClass(tok))] = 1;
    identityVector(tok).forEach((v, i) => (x[ID + i] = v));
    FREQS.forEach((w, k) => {
      x[POS + 2 * k] = Math.sin(pos * w);
      x[POS + 2 * k + 1] = Math.cos(pos * w);
    });
    return x;
  });
}

export interface HeadSpec {
  id: string;
  name: string;
  description: string;
  wq: Matrix;
  wk: Matrix;
  wv: Matrix;
}

// Gains are multiplied by √d so that, after the 1/√d scaling, the logits have the intended size.
const G = Math.sqrt(D_HEAD);

function valueProjection(): Matrix {
  // V carries each token's word class, so the head's output says what kind of word it gathered from.
  const wv = zeros(D_MODEL, D_HEAD);
  WORD_CLASSES.forEach((_, c) => (wv[CLS + c]![c] = 1));
  return wv;
}

function previousTokenHead(): HeadSpec {
  // q = PE(pos − 1) via a rotation of each (sin, cos) pair, k = PE(pos), so q·k peaks at the previous position.
  const wq = zeros(D_MODEL, D_HEAD);
  const wk = zeros(D_MODEL, D_HEAD);
  const gain = 4 * G;
  FREQS.forEach((w, k) => {
    const s = POS + 2 * k;
    const c = Math.cos(w);
    const sn = Math.sin(w);
    // [sin(p−1)w, cos(p−1)w] = [sin·c − cos·sn, cos·c + sin·sn]
    wq[s]![2 * k] = c * gain;
    wq[s + 1]![2 * k] = -sn * gain;
    wq[s]![2 * k + 1] = sn * gain;
    wq[s + 1]![2 * k + 1] = c * gain;
    wk[s]![2 * k] = 1;
    wk[s + 1]![2 * k + 1] = 1;
  });
  return {
    id: 'previous',
    name: 'Previous-token head',
    description: 'Queries and keys come from the positional part of the embedding, rotated so each token matches the one just before it. Real models use heads like this to build "A followed by B" patterns.',
    wq,
    wk,
    wv: valueProjection(),
  };
}

function sameWordHead(): HeadSpec {
  const wq = zeros(D_MODEL, D_HEAD);
  const wk = zeros(D_MODEL, D_HEAD);
  for (let i = 0; i < 8; i++) {
    wq[ID + i]![i] = 5 * G;
    wk[ID + i]![i] = 1;
  }
  return {
    id: 'same',
    name: 'Duplicate-token head',
    description: 'Queries and keys both come from the word-identity part of the embedding, so a word attends to itself and to every other copy of the same word. Spotting repeats like this is the first half of an "induction head", which then looks up what followed the earlier copy.',
    wq,
    wk,
    wv: valueProjection(),
  };
}

function nounHead(): HeadSpec {
  const wq = zeros(D_MODEL, D_HEAD);
  const wk = zeros(D_MODEL, D_HEAD);
  wq[BIAS]![0] = 4 * G; // every query asks the same question: "where are the nouns?"
  wk[CLS + WORD_CLASSES.indexOf('noun')]![0] = 1;
  wk[CLS + WORD_CLASSES.indexOf('pron')]![0] = 0.8;
  return {
    id: 'noun',
    name: 'Noun-seeking head',
    description: 'Every query is the same constant vector, and only nouns and pronouns have a matching key. So every token pulls in information about the things in the sentence, whatever its own position.',
    wq,
    wk,
    wv: valueProjection(),
  };
}

function randomHead(): HeadSpec {
  const r = rng(7);
  return {
    id: 'random',
    name: 'Random (untrained) head',
    description: 'Random weights, like a head at initialization. The pattern is arbitrary: training is what turns this into something useful.',
    wq: randomMatrix(D_MODEL, D_HEAD, 0.9, r),
    wk: randomMatrix(D_MODEL, D_HEAD, 0.9, r),
    wv: valueProjection(),
  };
}

export const HEADS: HeadSpec[] = [previousTokenHead(), sameWordHead(), nounHead(), randomHead()];

export interface HeadResult {
  head: HeadSpec;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  /** QKᵀ/√d before masking. */
  scores: Matrix;
  /** Row-wise softmax after the mask; each row sums to 1. */
  weights: Matrix;
  /** A·V: for each token, the class mix it gathered. */
  output: Matrix;
  masked: (i: number, j: number) => boolean;
}

export function selfAttention(tokens: string[], head: HeadSpec, causal: boolean): HeadResult {
  const x = embed(tokens);
  const q = matmul(x, head.wq);
  const k = matmul(x, head.wk);
  const v = matmul(x, head.wv);
  const scores = matmul(q, transpose(k)).map((row) => row.map((s) => s / Math.sqrt(D_HEAD)));
  const masked = (i: number, j: number) => causal && j > i;
  const weights = scores.map((row, i) => softmax(row.map((s, j) => (masked(i, j) ? -Infinity : s))));
  const output = matmul(weights, v);
  return { head, q, k, v, scores, weights, output, masked };
}

/** Split into tokens: words and punctuation, lower-cased for matching but displayed as typed. */
export function splitWords(text: string, max = 12): string[] {
  return (text.match(/[A-Za-z']+|[.,!?;]/g) ?? []).slice(0, max);
}
