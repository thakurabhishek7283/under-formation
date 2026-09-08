// Every classifier from the classification pages behind one interface, with sensible default
// settings, so they can be run side by side on the same data.

import type { LPoint } from './classify';
import { knnProba } from './knn';
import { decision, trainSvm, type SvmOptions } from './svm';
import { fitGaussians, posterior } from './bayes';
import { ensembleProba, fitEnsemble, growTree, treeProba } from './ensemble';
import { initNet, makeTrainer, predictProba, trainEpoch } from './mlp';
import { rng } from './math';

export type Proba = (x: number, y: number) => number[];

export interface ZooEntry {
  id: string;
  name: string;
  /** Settings, in words, for the caption. */
  settings: string;
  href: string;
  fit: (train: LPoint[], classes: number) => Proba;
}

/**
 * SVMs are two-class machines. For three classes, train one per class against the rest and pick
 * the most confident (one-vs-rest). The softmax of the scores is only for shading the map.
 */
function svm(o: SvmOptions) {
  return (train: LPoint[], classes: number): Proba => {
    if (classes === 2) {
      const m = trainSvm(train, o);
      return (x, y) => {
        const p = 1 / (1 + Math.exp(-2 * decision(m, x, y)));
        return [1 - p, p];
      };
    }
    const ms = Array.from({ length: classes }, (_, c) => trainSvm(train.map((p) => ({ ...p, label: p.label === c ? 1 : 0 })), o));
    return (x, y) => {
      const s = ms.map((m) => 2 * decision(m, x, y));
      const mx = Math.max(...s);
      const e = s.map((v) => Math.exp(v - mx));
      const t = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / t);
    };
  };
}

function mlp(hidden: number[], epochs: number) {
  return (train: LPoint[], classes: number): Proba => {
    const net = initNet([2, ...hidden, classes], 'tanh', 7);
    const tr = makeTrainer(net, 0.03, 8);
    for (let e = 0; e < epochs; e++) trainEpoch(tr, train);
    return (x, y) => Array.from(predictProba(net, x, y));
  };
}

export const ZOO: ZooEntry[] = [
  {
    id: 'knn',
    name: 'k-nearest neighbours',
    settings: 'k = 5',
    href: '/visualize/knn/',
    fit: (train, classes) => (x, y) => knnProba(train, classes, { k: 5, metric: 'euclidean', weighting: 'uniform', xScale: 1 }, x, y),
  },
  {
    id: 'logistic',
    name: 'Logistic regression',
    settings: 'linear features',
    href: '/visualize/logistic-regression/',
    // A network with no hidden layer is exactly (multinomial) logistic regression.
    fit: mlp([], 200),
  },
  { id: 'lsvm', name: 'Linear SVM', settings: 'C = 1', href: '/visualize/svm/', fit: svm({ kernel: 'linear', C: 1, gamma: 1 }) },
  { id: 'rsvm', name: 'RBF SVM', settings: 'C = 10, γ = 2', href: '/visualize/svm/', fit: svm({ kernel: 'rbf', C: 10, gamma: 2 }) },
  {
    id: 'nb',
    name: 'Gaussian Naive Bayes',
    settings: 'diagonal covariance',
    href: '/visualize/naive-bayes/',
    fit: (train, classes) => {
      const g = fitGaussians(train, classes, 'nb');
      return (x, y) => posterior(g, x, y);
    },
  },
  {
    id: 'tree',
    name: 'Decision tree',
    settings: 'max depth 5',
    href: '/visualize/decision-tree/',
    fit: (train, classes) => {
      const t = growTree(train, train.map(() => 1), classes, { maxDepth: 5, minLeaf: 2, maxFeatures: 2 }, rng(1));
      return (x, y) => treeProba(t, x, y);
    },
  },
  {
    id: 'forest',
    name: 'Random forest',
    settings: '100 trees',
    href: '/visualize/random-forest/',
    fit: (train, classes) => {
      const ls = fitEnsemble(train, classes, { kind: 'forest', count: 100, depth: 10, seed: 3 });
      return (x, y) => ensembleProba(ls, ls.length, classes, 'forest', x, y);
    },
  },
  { id: 'mlp', name: 'Neural network', settings: '2 hidden layers of 8, tanh', href: '/visualize/neural-network/', fit: mlp([8, 8], 300) },
];
