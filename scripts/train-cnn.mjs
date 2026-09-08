// Trains the digit CNN shown on /visualize/cnn/ and writes its weights to
// src/lib/ml/cnnWeights.json. Deterministic: same code, same seeds, same weights.
//
//   node scripts/train-cnn.mjs
//
// The model code is TypeScript, so esbuild (already installed with Astro) bundles it first.

import { build } from 'esbuild';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const out = await build({
  stdin: { contents: "export * from './src/lib/ml/cnn.ts';", resolveDir: root, loader: 'ts' },
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
  logLevel: 'warning',
});
const cnn = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);
const { RECIPE: R, TEST_SET: T } = cnn;

const data = cnn.makeDigits(R.samples, R.dataSeed, R.wild);
const test = cnn.makeDigits(T.samples, T.seed, T.wild);
const tr = cnn.cnnTrainer(cnn.initCnn(R.initSeed), R.lr, R.orderSeed);
for (let e = 1; e <= R.epochs; e++) {
  const order = cnn.epochOrder(tr, R.samples);
  let loss = 0;
  let batches = 0;
  for (let s = 0; s < R.samples; s += R.batch) {
    loss += cnn.cnnStep(tr, data, order.slice(s, s + R.batch));
    batches++;
  }
  console.log(`epoch ${e}: loss ${(loss / batches).toFixed(4)}, held-out accuracy ${(cnn.cnnAccuracy(tr.m, test) * 100).toFixed(1)}%`);
}

const file = new URL('../src/lib/ml/cnnWeights.json', import.meta.url);
writeFileSync(file, JSON.stringify(cnn.serialize(tr.m)) + '\n');
console.log(`wrote ${fileURLToPath(file)}`);
