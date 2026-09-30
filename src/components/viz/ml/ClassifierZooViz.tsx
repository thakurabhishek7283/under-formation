import { useEffect, useMemo, useState } from 'react';
import { CLASS_DATASETS, accuracy, argmax, computeField, makeClassData, trainTest, type ClassDataset, type Field } from '../../../lib/ml/classify';
import { url } from '../../../lib/url';
import { ZOO } from '../../../lib/ml/zoo';
import { ClassLegend, DecisionMap } from './DecisionMap';
import '../viz.css';
import './ml.css';

const DATASETS: ClassDataset[] = ['blobs', 'moons', 'circles', 'xor', 'spiral'];

interface Cell {
  field: Field;
  trainAcc: number;
  testAcc: number;
  ms: number;
}

export default function ClassifierZooViz({ n = 160 }: { n?: number }) {
  const [seed, setSeed] = useState(6);
  const [cells, setCells] = useState<Record<string, Cell>>({});
  const splits = useMemo(() => DATASETS.map((d) => trainTest(makeClassData(d, n, seed))), [n, seed]);

  // Fit one (classifier, dataset) pair per task, yielding in between so the page stays responsive
  // and the grid fills in as it goes.
  useEffect(() => {
    setCells({});
    let cancelled = false;
    const jobs = ZOO.flatMap((z) => DATASETS.map((d, j) => ({ z, d, j })));
    let k = 0;
    const next = () => {
      if (cancelled || k >= jobs.length) return;
      const { z, d, j } = jobs[k++]!;
      const { train, test } = splits[j]!;
      const classes = CLASS_DATASETS[d].classes;
      const t0 = performance.now();
      const proba = z.fit(train, classes);
      const predict = (x: number, y: number) => argmax(proba(x, y));
      const cell: Cell = {
        field: computeField(proba, classes, 40),
        trainAcc: accuracy(predict, train),
        testAcc: accuracy(predict, test),
        ms: performance.now() - t0,
      };
      setCells((c) => ({ ...c, [`${z.id}/${d}`]: cell }));
      setTimeout(next, 0);
    };
    const t = setTimeout(next, 0);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [splits]);

  const done = Object.keys(cells).length;
  const total = ZOO.length * DATASETS.length;

  return (
    <div className="viz" aria-label="Classifier comparison grid">
      <div className="viz-row">
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          New data
        </button>
        <span className="muted" style={{ fontSize: '0.85rem' }}>
          {done < total ? `Training ${done + 1} of ${total}…` : `${total} models trained in your browser. Numbers are held-out accuracy.`}
        </span>
      </div>

      <div className="viz-scroll">
        <table className="viz-zoo">
          <thead>
            <tr>
              <th />
              {DATASETS.map((d) => (
                <th key={d}>{CLASS_DATASETS[d].name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ZOO.map((z) => (
              <tr key={z.id}>
                <th scope="row">
                  <a href={url(z.href)}>{z.name}</a>
                  <small>{z.settings}</small>
                </th>
                {DATASETS.map((d, j) => {
                  const cell = cells[`${z.id}/${d}`];
                  return (
                    <td key={d} title={cell ? `${z.name} on ${CLASS_DATASETS[d].name}: training ${(cell.trainAcc * 100).toFixed(0)}%, held out ${(cell.testAcc * 100).toFixed(0)}%, ${cell.ms.toFixed(0)} ms` : undefined}>
                      <DecisionMap field={cell?.field ?? null} points={splits[j]!.train} test={splits[j]!.test} boundary label={`${z.name} on ${CLASS_DATASETS[d].name}`} marker={() => ({ r: 1.9 })} />
                      <span className="acc">{cell ? `${(cell.testAcc * 100).toFixed(0)}%` : '…'}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ClassLegend classes={2} />
    </div>
  );
}
