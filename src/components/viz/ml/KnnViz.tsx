import { useEffect, useMemo, useState } from 'react';
import { CLASS_DATASETS, computeField, makeClassData, trainTest, type ClassDataset } from '../../../lib/ml/classify';
import { KNN_CODE, METRICS, kCurve, knnPredict, knnProba, knnSteps, type KnnStep, type Metric, type Weighting } from '../../../lib/ml/knn';
import { usePlayer } from '../usePlayer';
import { PlayerControls } from '../PlayerControls';
import { CodePanel } from '../CodePanel';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { CLASS_COLOR, CLASS_NAME, ClassLegend, DecisionMap, sy, useClient } from './DecisionMap';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const DATASETS: ClassDataset[] = ['blobs', 'three', 'moons', 'circles', 'xor', 'spiral'];
const K_MAX = 30;
const X_SCALE = 5;

/** The set of points within distance r of (x, y): a circle, diamond or square, squashed by xScale. */
function ballPath(x: number, y: number, r: number, metric: Metric, xScale: number): string {
  const rx = r / xScale;
  const ry = r;
  const cx = r1(x);
  const cy = sy(y);
  if (metric === 'manhattan') return `M${r1(cx - rx)},${cy} L${cx},${r1(cy - ry)} L${r1(cx + rx)},${cy} L${cx},${r1(cy + ry)} Z`;
  if (metric === 'chebyshev') return `M${r1(cx - rx)},${r1(cy - ry)} H${r1(cx + rx)} V${r1(cy + ry)} H${r1(cx - rx)} Z`;
  return `M${r1(cx - rx)},${cy} A${r1(rx)},${r1(ry)} 0 1 0 ${r1(cx + rx)},${cy} A${r1(rx)},${r1(ry)} 0 1 0 ${r1(cx - rx)},${cy} Z`;
}

export default function KnnViz({ n = 150 }: { n?: number }) {
  const [kind, setKind] = useState<ClassDataset>('moons');
  const [k, setK] = useState(5);
  const [metric, setMetric] = useState<Metric>('euclidean');
  const [weighting, setWeighting] = useState<Weighting>('uniform');
  const [unscaled, setUnscaled] = useState(false);
  const [seed, setSeed] = useState(4);
  const [query, setQuery] = useState({ x: 52, y: 47 });
  const [dragging, setDragging] = useState(false);
  const client = useClient();

  const classes = CLASS_DATASETS[kind].classes;
  const { train, test } = useMemo(() => trainTest(makeClassData(kind, n, seed)), [kind, n, seed]);
  const opts = useMemo(() => ({ k, metric, weighting, xScale: unscaled ? X_SCALE : 1 }), [k, metric, weighting, unscaled]);

  const { steps, nbrs } = useMemo(() => knnSteps(train, classes, opts, query.x, query.y), [train, classes, opts, query]);
  const player = usePlayer<KnnStep>(steps, 3);
  useEffect(() => player.load(steps, false, dragging), [steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const field = useMemo(() => (client ? computeField((x, y) => knnProba(train, classes, opts, x, y), classes, 64) : null), [client, train, classes, opts]);
  const curve = useMemo(
    () => kCurve(train, test, classes, { metric, weighting, xScale: opts.xScale }, K_MAX),
    [train, test, classes, metric, weighting, opts.xScale],
  );

  const step = player.step;
  const taken = nbrs.slice(0, step.taken);
  const takenSet = new Set(taken.map((t) => t.i));
  const radius = taken.length ? taken[taken.length - 1]!.d : 0;
  const total = step.votes.reduce((a, b) => a + b, 0) || 1;
  const winner = step.phase === 'vote' ? step.votes.indexOf(Math.max(...step.votes)) : -1;
  const predicted = knnPredict(train, classes, opts, query.x, query.y);

  const onPlot = (kindOf: 'down' | 'move' | 'up', at: { x: number; y: number }) => {
    if (kindOf === 'down') {
      setDragging(true);
      setQuery(at);
    } else if (kindOf === 'move' && dragging) setQuery(at);
    else if (kindOf === 'up') setDragging(false);
  };

  return (
    <div className="viz" aria-label="k-nearest neighbours visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Data</span>
          <select className="input" value={kind} onChange={(e) => setKind(e.target.value as ClassDataset)}>
            {DATASETS.map((d) => (
              <option key={d} value={d}>
                {CLASS_DATASETS[d].name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Neighbours k <output>{k}</output>
          </span>
          <input type="range" min={1} max={K_MAX} value={k} onChange={(e) => setK(+e.target.value)} />
        </label>
        <label className="viz-field">
          <span>Distance</span>
          <select className="input" value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
            {METRICS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-field">
          <span>Votes</span>
          <select className="input" value={weighting} onChange={(e) => setWeighting(e.target.value as Weighting)}>
            <option value="uniform">one each</option>
            <option value="distance">weighted by 1/distance</option>
          </select>
        </label>
      </div>
      <div className="viz-row">
        <label className="viz-check">
          <input type="checkbox" checked={unscaled} onChange={(e) => setUnscaled(e.target.checked)} />
          Forget to standardize: measure x in units {X_SCALE}× smaller
        </label>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          New data
        </button>
      </div>

      <p className="viz-note">
        {CLASS_DATASETS[kind].blurb} Click or drag anywhere on the plot to move the query point.
      </p>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-split">
        <div>
          <DecisionMap
            field={field}
            points={train}
            test={test}
            onPlot={onPlot}
            label={`k-NN decision regions with k = ${k}; the query at (${query.x.toFixed(0)}, ${query.y.toFixed(0)}) is predicted class ${CLASS_NAME[predicted]}`}
            marker={(_, i, held) => (!held && takenSet.has(i) ? { ring: 'var(--text)', r: 1.7 } : held ? { dim: step.phase !== 'query' } : undefined)}
            under={
              <>
                {step.phase === 'dist' &&
                  train.map((p, i) => (
                    <line
                      key={i}
                      x1={r1(query.x)}
                      y1={sy(query.y)}
                      x2={r1(p.x)}
                      y2={sy(p.y)}
                      style={{ stroke: 'var(--text-3)', strokeWidth: 0.15, opacity: 0.6 }}
                    />
                  ))}
                {radius > 0 && (
                  <path
                    d={ballPath(query.x, query.y, radius, metric, opts.xScale)}
                    style={{ fill: 'var(--text)', fillOpacity: 0.05, stroke: 'var(--text)', strokeWidth: 0.4, strokeDasharray: '1.5 1' }}
                  />
                )}
                {taken.map((t) => {
                  const p = train[t.i]!;
                  return (
                    <line
                      key={`n${t.i}`}
                      x1={r1(query.x)}
                      y1={sy(query.y)}
                      x2={r1(p.x)}
                      y2={sy(p.y)}
                      style={{ stroke: CLASS_COLOR[p.label], strokeWidth: 0.55 }}
                    />
                  );
                })}
              </>
            }
          >
            <g>
              <circle cx={r1(query.x)} cy={sy(query.y)} r={2.6} style={{ fill: 'var(--surface)', stroke: 'var(--text)', strokeWidth: 0.8 }} />
              <circle
                cx={r1(query.x)}
                cy={sy(query.y)}
                r={1.2}
                style={{ fill: winner >= 0 ? CLASS_COLOR[winner] : 'var(--text)' }}
              />
              <title>query point</title>
            </g>
          </DecisionMap>
          <div style={{ marginTop: '0.5rem' }}>
            <ClassLegend
              classes={classes}
              extra={
                <>
                  <span>◎ query</span>
                  <span>ringed = counted neighbour</span>
                  <span>dashed shape = every point within that distance</span>
                </>
              }
            />
          </div>
        </div>

        <CodePanel listing={KNN_CODE} cursor={step.code} previous={player.steps[player.index - 1]?.code} title="k-NN prediction" />
      </div>

      <div>
        <p className="viz-label">The vote</p>
        <ul className="viz-votes">
          {step.votes.map((v, c) => (
            <li key={c} className={c === winner ? 'win' : undefined}>
              <span>class {CLASS_NAME[c]}</span>
              <span className="bar">
                <span style={{ width: `${((v / total) * 100).toFixed(1)}%`, background: CLASS_COLOR[c] }} />
              </span>
              <span className="v">{weighting === 'distance' ? v.toFixed(2) : v}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="viz-stats">
        <Stat k="Training accuracy" v={`${curve.trainAcc[k]!.toFixed(1)}%`} />
        <Stat k="Held-out accuracy" v={`${curve.testAcc[k]!.toFixed(1)}%`} />
        <Stat k="Distances per prediction" v={String(train.length)} />
        <Stat k="Parameters learned" v="0" />
      </div>

      <figure style={{ margin: 0 }}>
        <figcaption className="viz-label">Accuracy vs k — small k memorizes, large k blurs</figcaption>
        <LineChart
          series={[
            { id: 'train', name: 'training', color: 'var(--cat-1)', values: curve.trainAcc },
            { id: 'test', name: 'held out', color: 'var(--cat-2)', values: curve.testAcc },
          ]}
          xLabel={`k   (currently ${k})`}
          yLabel="accuracy %"
          height={190}
          format={(v) => `${v.toFixed(0)}%`}
        />
      </figure>
    </div>
  );
}
