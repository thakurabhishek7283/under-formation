// Generative classifiers: model how each class *produces* data, p(x | class), then flip it
// around with Bayes' rule,
//
//   P(class | x) ∝ P(class) · p(x | class).
//
// With a Gaussian per class, three classic choices differ only in the covariance they allow:
//
//   Naive Bayes  — diagonal covariance per class: features independent given the class.
//   LDA          — one full covariance shared by all classes → the boundary is a straight line.
//   QDA          — a full covariance per class → the boundary is a conic.
//
// The second half is the other famous Naive Bayes: counting words to filter spam.

import type { LPoint } from './classify';

export type GaussKind = 'nb' | 'lda' | 'qda';

export const GAUSS_KINDS: { id: GaussKind; name: string; covariance: string; boundary: string }[] = [
  { id: 'nb', name: 'Naive Bayes', covariance: 'diagonal, per class', boundary: 'quadratic, axis-aligned' },
  { id: 'lda', name: 'LDA', covariance: 'full, shared', boundary: 'linear' },
  { id: 'qda', name: 'QDA', covariance: 'full, per class', boundary: 'quadratic' },
];

/** A symmetric 2 × 2 covariance [[xx, xy], [xy, yy]]. */
export interface Cov {
  xx: number;
  xy: number;
  yy: number;
}

export interface ClassGaussian {
  prior: number;
  mx: number;
  my: number;
  cov: Cov;
  n: number;
}

/** Maximum-likelihood fit. A small ridge keeps a degenerate class (all points on a line) invertible. */
export function fitGaussians(train: LPoint[], classes: number, kind: GaussKind): ClassGaussian[] {
  const RIDGE = 1e-3;
  const fits = Array.from({ length: classes }, (_, c) => {
    const pts = train.filter((p) => p.label === c);
    const n = Math.max(pts.length, 1);
    const mx = pts.reduce((s, p) => s + p.x, 0) / n;
    const my = pts.reduce((s, p) => s + p.y, 0) / n;
    let xx = 0;
    let xy = 0;
    let yy = 0;
    for (const p of pts) {
      xx += (p.x - mx) ** 2;
      xy += (p.x - mx) * (p.y - my);
      yy += (p.y - my) ** 2;
    }
    return { prior: pts.length / Math.max(train.length, 1), mx, my, n: pts.length, scatter: { xx, xy, yy } };
  });

  if (kind === 'lda') {
    // Pool the within-class scatter: one covariance, the same shape for every class.
    const total = fits.reduce(
      (s, f) => ({ xx: s.xx + f.scatter.xx, xy: s.xy + f.scatter.xy, yy: s.yy + f.scatter.yy }),
      { xx: 0, xy: 0, yy: 0 },
    );
    const n = Math.max(train.length, 1);
    const cov = { xx: total.xx / n + RIDGE, xy: total.xy / n, yy: total.yy / n + RIDGE };
    return fits.map((f) => ({ prior: f.prior, mx: f.mx, my: f.my, n: f.n, cov }));
  }
  return fits.map((f) => {
    const n = Math.max(f.n, 1);
    const cov = { xx: f.scatter.xx / n + RIDGE, xy: kind === 'nb' ? 0 : f.scatter.xy / n, yy: f.scatter.yy / n + RIDGE };
    return { prior: f.prior, mx: f.mx, my: f.my, n: f.n, cov };
  });
}

/** log N(x; μ, Σ) for a 2-D Gaussian. */
export function logDensity(g: ClassGaussian, x: number, y: number): number {
  const { xx, xy, yy } = g.cov;
  const det = xx * yy - xy * xy;
  const dx = x - g.mx;
  const dy = y - g.my;
  // (x − μ)ᵀ Σ⁻¹ (x − μ), with the closed-form 2 × 2 inverse.
  const m = (yy * dx * dx - 2 * xy * dx * dy + xx * dy * dy) / det;
  return -0.5 * (m + Math.log(det)) - Math.log(2 * Math.PI);
}

/** 1-D normal density, for Naive Bayes' per-feature factors. */
export function normal1d(v: number, mean: number, variance: number): number {
  return Math.exp(-((v - mean) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance);
}

export function posterior(gs: ClassGaussian[], x: number, y: number): number[] {
  const logs = gs.map((g) => (g.prior > 0 ? Math.log(g.prior) + logDensity(g, x, y) : -Infinity));
  const max = Math.max(...logs);
  const e = logs.map((l) => Math.exp(l - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/** Axes of the 1σ ellipse: half-lengths and the rotation (radians, counter-clockwise from +x). */
export function ellipse(cov: Cov): { a: number; b: number; angle: number } {
  const { xx, xy, yy } = cov;
  const mid = (xx + yy) / 2;
  const rad = Math.sqrt(((xx - yy) / 2) ** 2 + xy * xy);
  return { a: Math.sqrt(Math.max(mid + rad, 0)), b: Math.sqrt(Math.max(mid - rad, 0)), angle: 0.5 * Math.atan2(2 * xy, xx - yy) };
}

export function paramCount(kind: GaussKind, classes: number): number {
  const means = 2 * classes;
  const priors = classes - 1;
  if (kind === 'nb') return means + 2 * classes + priors;
  if (kind === 'lda') return means + 3 + priors;
  return means + 3 * classes + priors;
}

// ---------------------------------------------------------------------------------------------
// Multinomial Naive Bayes on words: the classic spam filter.

export interface Message {
  text: string;
  spam: boolean;
}

/** A tiny hand-written corpus. Small on purpose, so every count can be checked by eye. */
export const CORPUS: Message[] = [
  { spam: true, text: 'WINNER! You have won a free prize. Claim your cash now' },
  { spam: true, text: 'Free entry to win a brand new phone, text WIN now' },
  { spam: true, text: 'Urgent! Your account has won a cash reward, click the link to claim' },
  { spam: true, text: 'Congratulations, you are selected for a free holiday. Call now' },
  { spam: true, text: 'Claim your free gift card today, limited offer, click now' },
  { spam: true, text: 'You have been chosen to receive a cash prize of 1000' },
  { spam: true, text: 'Cheap loans approved instantly, no credit check, apply now' },
  { spam: true, text: 'Exclusive offer: win free tickets, reply YES to claim' },
  { spam: true, text: 'Your prize is waiting, click the link to collect your reward' },
  { spam: true, text: 'Urgent: verify your account now or it will be suspended' },
  { spam: true, text: 'Get rich quick, earn cash from home, free trial' },
  { spam: true, text: 'Last chance! Free offer ends today, call now to win' },
  { spam: false, text: 'Are we still meeting for lunch today?' },
  { spam: false, text: 'Can you send me the notes from class' },
  { spam: false, text: 'I will call you when I get home tonight' },
  { spam: false, text: 'Thanks for the help with the project yesterday' },
  { spam: false, text: 'Running late, see you at the station in ten minutes' },
  { spam: false, text: 'Did you finish the report for the meeting tomorrow' },
  { spam: false, text: 'Happy birthday! Hope you have a great day' },
  { spam: false, text: 'Mum says dinner is at seven, bring the kids' },
  { spam: false, text: 'Can we move our call to tomorrow morning' },
  { spam: false, text: 'The code review is done, looks good to me' },
  { spam: false, text: 'Let me know when you get home safe' },
  { spam: false, text: 'Are you free this weekend for a walk' },
];

export const tokenize = (text: string) => text.toLowerCase().match(/[a-z0-9']+/g) ?? [];

export interface WordModel {
  /** Word counts per class: [ham, spam]. */
  counts: Map<string, [number, number]>;
  totals: [number, number];
  docs: [number, number];
  vocab: number;
}

export function fitWords(corpus: Message[]): WordModel {
  const counts = new Map<string, [number, number]>();
  const totals: [number, number] = [0, 0];
  const docs: [number, number] = [0, 0];
  for (const m of corpus) {
    const c = m.spam ? 1 : 0;
    docs[c]++;
    for (const w of tokenize(m.text)) {
      const e = counts.get(w) ?? [0, 0];
      e[c]++;
      totals[c]++;
      counts.set(w, e);
    }
  }
  return { counts, totals, docs, vocab: counts.size };
}

export interface WordEvidence {
  word: string;
  known: boolean;
  /** Raw counts in [ham, spam]. */
  counts: [number, number];
  /** log P(word | spam) − log P(word | ham). ±Infinity when unsmoothed and unseen in one class. */
  llr: number;
}

export interface SpamVerdict {
  words: WordEvidence[];
  /** log P(spam) − log P(ham) from the class frequencies alone. */
  priorLogOdds: number;
  logOdds: number;
  pSpam: number;
}

/**
 * P(word | class) = (count + α) / (total words in class + α·|V|). With α = 0 a word never seen
 * in one class makes that class impossible, however much the other words argue for it.
 */
export function classify(model: WordModel, text: string, alpha: number): SpamVerdict {
  const words: WordEvidence[] = tokenize(text).map((word) => {
    const e = model.counts.get(word);
    if (!e) return { word, known: false, counts: [0, 0], llr: 0 };
    const ps = (e[1] + alpha) / (model.totals[1] + alpha * model.vocab);
    const ph = (e[0] + alpha) / (model.totals[0] + alpha * model.vocab);
    return { word, known: true, counts: [e[0], e[1]], llr: Math.log(ps) - Math.log(ph) };
  });
  const priorLogOdds = Math.log(model.docs[1] / model.docs[0]);
  let pos = false;
  let neg = false;
  let sum = priorLogOdds;
  for (const w of words) {
    if (w.llr === Infinity) pos = true;
    else if (w.llr === -Infinity) neg = true;
    else sum += w.llr;
  }
  // Both infinities: each class has been ruled out by some word. The model has no answer.
  const logOdds = pos && neg ? NaN : pos ? Infinity : neg ? -Infinity : sum;
  const pSpam = Number.isNaN(logOdds) ? NaN : logOdds === Infinity ? 1 : logOdds === -Infinity ? 0 : 1 / (1 + Math.exp(-logOdds));
  return { words, priorLogOdds, logOdds, pSpam };
}
