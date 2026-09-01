import { useMemo, useState } from 'react';
import { CLASS_DATASETS, accuracy, computeField, makeClassData, trainTest, type ClassDataset } from '../../../lib/ml/classify';
import { rng } from '../../../lib/ml/math';
import { ensembleCurve, ensembleProba, ensemblePredict, fitEnsemble, growTree, treePredict, treeProba, type EnsembleKind } from '../../../lib/ml/ensemble';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { ClassLegend, DecisionMap, useClient } from './DecisionMap';
import '../viz.css';
import './ml.css';

const DATASETS: ClassDataset[] = ['moons', 'circles', 'xor', 'spiral', 'three', 'blobs'];
const MAX = 100;
const MINIS = 8;
const KINDS: { id: EnsembleKind; name: string; depth: number; blurb: string }[] = [
  { id: 'bagging', name: 'Bagged trees', depth: 10, blurb: 'Deep trees, each on a bootstrap resample of the rows. Averaging cancels their variance.' },
  { id: 'forest', name: 'Random forest', depth: 10, blurb: 'Bagging, plus every split may only look at one randomly chosen feature — so the trees disagree more, and the average is smoother.' },
  { id: 'adaboost', name: 'AdaBoost', depth: 1, blurb: 'One-split stumps trained in sequence. Each one is trained on data reweighted towards what the previous ones got wrong.' },
];

export default function EnsembleViz({ n = 200 }: { n?: number }) {
  const [kind, setKind] = useState<EnsembleKind>('forest');
  const [data, setData] = useState<ClassDataset>('moons');
  const [count, setCount] = useState(25);
  const [depth, setDepth] = useState(10);
  const [seed, setSeed] = useState(5);
  const client = useClient();

  const classes = CLASS_DATASETS[data].classes;
  const { train, test } = useMemo(() => trainTest(makeClassData(data, n, seed)), [data, n, seed]);
  const learners = useMemo(() => fitEnsemble(train, classes, { kind, count: MAX, depth, seed: seed * 7 + 1 }), [train, classes, kind, depth, seed]);
  const curve = useMemo(() => ensembleCurve(learners, train, test, classes, kind), [learners, train, test, classes, kind]);
  const single = useMemo(() => growTree(train, train.map(() => 1), classes, { maxDepth: depth, minLeaf: 1, maxFeatures: 2 }, rng(1)), [train, classes, depth]);

  const k = Math.min(count, learners.length);
  const field = useMemo(
    () => (client ? computeField((x, y) => ensembleProba(learners, k, classes, kind, x, y), classes, 72) : null),
    [client, learners, k, classes, kind],
  );
  const miniFields = useMemo(
    () => (client ? learners.slice(0, MINIS).map((l) => computeField((x, y) => treeProba(l.tree, x, y), classes, 40)) : []),
    [client, learners, classes],
  );

  const singleTest = accuracy((x, y) => treePredict(single, x, y), test);
  const ensTest = accuracy((x, y) => ensemblePredict(learners, k, classes, kind, x, y), test);
  const info = KINDS.find((c) => c.id === kind)!;
  const pickKind = (id: EnsembleKind) => {
    setKind(id);
    setDepth(KINDS.find((c) => c.id === id)!.depth);
    if (id === 'adaboost') setCount(40);
  };

  return (
    <div className="viz" aria-label="Random forest, bagging and AdaBoost visualization">
      <div className="viz-row">
        <div className="viz-seg" role="radiogroup" aria-label="Ensemble method">
          {KINDS.map((c) => (
            <button key={c.id} type="button" role="radio" aria-checked={kind === c.id} className={`btn${kind === c.id ? ' on' : ''}`} onClick={() => pickKind(c.id)}>
              {c.name}
            </button>
          ))}
        </div>
        <label className="viz-field grow">
          <span>Data</span>
          <select className="input" value={data} onChange={(e) => setData(e.target.value as ClassDataset)}>
            {DATASETS.map((d) => (
              <option key={d} value={d}>
                {CLASS_DATASETS[d].name}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          New data
        </button>
      </div>
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Trees in the ensemble <output>{k}</output>
            {k < count ? ` (boosting stopped early: nothing left to fix)` : ''}
          </span>
          <input type="range" min={1} max={MAX} value={count} onChange={(e) => setCount(+e.target.value)} />
        </label>
        <label className="viz-slider">
          <span>
            Depth of each tree <output>{depth}</output> {depth === 1 ? '(a stump: one split)' : ''}
          </span>
          <input type="range" min={1} max={12} value={depth} onChange={(e) => setDepth(+e.target.value)} />
        </label>
      </div>

      <p className="viz-note">{info.blurb}</p>

      <div className="viz-split">
        <div>
          <DecisionMap field={field} points={train} test={test} label={`${info.name} of ${k} trees: decision regions`} />
          <div style={{ marginTop: '0.5rem' }}>
            <ClassLegend classes={classes} extra={<span>shade = share of the vote</span>} />
          </div>
        </div>
        <div style={{ display: 'grid', gap: '1rem', alignContent: 'start', minWidth: 0 }}>
          <div className="viz-stats">
            <Stat k={`One depth-${depth} tree, held out`} v={`${(singleTest * 100).toFixed(1)}%`} />
            <Stat k={`Ensemble of ${k}, held out`} v={`${(ensTest * 100).toFixed(1)}%`} />
            <Stat k="Ensemble, training" v={`${curve.trainAcc[k]!.toFixed(1)}%`} />
            <Stat k="Total leaves" v={String(learners.slice(0, k).reduce((s, l) => s + l.tree.leaves, 0))} />
          </div>
          <div>
            <p className="viz-label">
              The first {Math.min(MINIS, learners.length)} trees on their own
              {kind === 'adaboost' ? ' — point size = the weight it trained on' : ' — each on its own bootstrap sample'}
            </p>
            <div className="viz-minis three">
              {learners.slice(0, MINIS).map((l, t) => (
                <figure key={t} className="viz-map-mini" style={{ opacity: t < k ? 1 : 0.45 }}>
                  <DecisionMap
                    field={miniFields[t] ?? null}
                    points={train}
                    label={`tree ${t + 1} alone`}
                    marker={(_, i) => (l.weights ? { r: Math.max(0.5, Math.min(4.5, Math.sqrt(l.weights[i]! * train.length) * 1.6)) } : { r: 1.6 })}
                  />
                  <figcaption>
                    <strong>#{t + 1}</strong>
                    <span>{kind === 'adaboost' ? `α ${l.alpha.toFixed(2)} · err ${(l.error * 100).toFixed(0)}%` : `OOB ${(l.error * 100).toFixed(0)}%`}</span>
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>
        </div>
      </div>

      <figure style={{ margin: 0 }}>
        <figcaption className="viz-label">Accuracy as trees are added (currently {k})</figcaption>
        <LineChart
          series={[
            { id: 'train', name: 'training', color: 'var(--cat-1)', values: curve.trainAcc },
            { id: 'test', name: 'held out', color: 'var(--cat-2)', values: curve.testAcc },
          ]}
          xLabel="number of trees"
          yLabel="accuracy %"
          height={190}
          format={(v) => `${v.toFixed(0)}%`}
        />
      </figure>
    </div>
  );
}
