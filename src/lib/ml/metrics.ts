// Confusion matrix, ROC and precision–recall for a binary classifier.
//
// A classifier does not output a decision, it outputs a *score*. The decision only appears once
// you pick a threshold, and every metric below is a function of that choice. Accuracy, precision
// and recall describe one threshold; AUC and average precision summarize all of them.

import { rng, randn } from './math';

export interface Scored {
  /** Model score, higher = more confident it's positive. */
  score: number;
  /** Ground truth: 1 = positive. */
  label: 0 | 1;
}

/**
 * Scores from two overlapping normal distributions. `separation` is the gap between their means
 * in standard deviations — the only thing that decides how good the model can possibly be.
 * `prevalence` is the fraction of the data that is actually positive.
 */
export function makeScores(n: number, separation: number, prevalence: number, seed: number): Scored[] {
  const r = rng(seed);
  const out: Scored[] = [];
  for (let i = 0; i < n; i++) {
    const label: 0 | 1 = r() < prevalence ? 1 : 0;
    const score = (label ? separation : 0) + randn(r);
    out.push({ score, label });
  }
  return out;
}

export interface Confusion {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

export function confusionAt(data: Scored[], threshold: number): Confusion {
  const c: Confusion = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (const d of data) {
    const predicted = d.score >= threshold;
    if (d.label === 1) predicted ? c.tp++ : c.fn++;
    else predicted ? c.fp++ : c.tn++;
  }
  return c;
}

export interface Metrics {
  accuracy: number;
  /** Of the ones we flagged, how many were right. Undefined when nothing is flagged. */
  precision: number;
  /** Of the real positives, how many we caught. Also called TPR, sensitivity, recall. */
  recall: number;
  /** Of the real negatives, how many we wrongly flagged. */
  fpr: number;
  specificity: number;
  f1: number;
  /** What you'd get by always predicting the majority class. */
  baseline: number;
}

export function metricsOf(c: Confusion): Metrics {
  const pos = c.tp + c.fn;
  const neg = c.tn + c.fp;
  const flagged = c.tp + c.fp;
  const total = pos + neg;
  const precision = flagged ? c.tp / flagged : 1;
  const recall = pos ? c.tp / pos : 0;
  return {
    accuracy: total ? (c.tp + c.tn) / total : 0,
    precision,
    recall,
    fpr: neg ? c.fp / neg : 0,
    specificity: neg ? c.tn / neg : 1,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    baseline: total ? Math.max(pos, neg) / total : 0,
  };
}

export interface CurvePoint {
  threshold: number;
  fpr: number;
  tpr: number;
  precision: number;
}

/**
 * Sweep the threshold over every distinct score. Walking the data once in descending score order
 * gives every point on both curves in O(n log n) — the same sweep both curves are defined by.
 */
export function curves(data: Scored[]): { roc: CurvePoint[]; auc: number; ap: number } {
  const sorted = [...data].sort((a, b) => b.score - a.score);
  const pos = sorted.filter((d) => d.label === 1).length;
  const neg = sorted.length - pos;

  const roc: CurvePoint[] = [{ threshold: Infinity, fpr: 0, tpr: 0, precision: 1 }];
  let tp = 0;
  let fp = 0;
  let auc = 0;
  let ap = 0;
  let prevFpr = 0;
  let prevRecall = 0;

  for (let i = 0; i < sorted.length; i++) {
    const d = sorted[i]!;
    if (d.label === 1) tp++;
    else fp++;
    // Only emit a point once the whole tie group is consumed; a threshold can't split equal scores.
    if (i + 1 < sorted.length && sorted[i + 1]!.score === d.score) continue;

    const tpr = pos ? tp / pos : 0;
    const fpr = neg ? fp / neg : 0;
    const precision = tp + fp ? tp / (tp + fp) : 1;

    // Trapezoid under ROC, and the step-wise sum that defines average precision.
    auc += ((fpr - prevFpr) * (tpr + prevRecall)) / 2;
    ap += (tpr - prevRecall) * precision;

    roc.push({ threshold: d.score, fpr, tpr, precision });
    prevFpr = fpr;
    prevRecall = tpr;
  }

  return { roc, auc, ap };
}

/** Counts per score bin, split by true label — the picture every threshold slides across. */
export function histogram(data: Scored[], bins: number, lo: number, hi: number): { neg: number[]; pos: number[] } {
  const neg = new Array<number>(bins).fill(0);
  const pos = new Array<number>(bins).fill(0);
  for (const d of data) {
    const t = (d.score - lo) / (hi - lo);
    const b = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
    (d.label === 1 ? pos : neg)[b]!++;
  }
  return { neg, pos };
}

/** The threshold maximizing F1 — one principled way to stop choosing by eye. */
export function bestF1(data: Scored[]): { threshold: number; f1: number } {
  let best = { threshold: 0, f1: 0 };
  for (const p of curves(data).roc) {
    if (!Number.isFinite(p.threshold)) continue;
    const f1 = p.precision + p.tpr ? (2 * p.precision * p.tpr) / (p.precision + p.tpr) : 0;
    if (f1 > best.f1) best = { threshold: p.threshold, f1 };
  }
  return best;
}
