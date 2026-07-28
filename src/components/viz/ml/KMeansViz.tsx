import { useEffect, useMemo, useState } from 'react';
import {
  DATASETS,
  KMEANS_CODE,
  elbowCurve,
  kmeansSteps,
  makeDataset,
  voronoiCells,
  type DatasetKind,
  type InitKind,
  type KMeansStep,
} from '../../../lib/ml/kmeans';
import { usePlayer } from '../usePlayer';
import { PlayerControls, Swatch } from '../PlayerControls';
import { CodePanel } from '../CodePanel';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const SIZE = 100;
const MAX_K = 6;

export default function KMeansViz({ initialK = 3, n = 150 }: { initialK?: number; n?: number }) {
  const [kind, setKind] = useState<DatasetKind>('blobs');
  const [k, setK] = useState(initialK);
  const [init, setInit] = useState<InitKind>('kmeans++');
  const [dataSeed, setDataSeed] = useState(7);
  const [runSeed, setRunSeed] = useState(3);

  const points = useMemo(() => makeDataset(kind, n, dataSeed), [kind, n, dataSeed]);
  const steps = useMemo(() => kmeansSteps(points, k, init, runSeed), [points, k, init, runSeed]);
  const elbow = useMemo(() => elbowCurve(points, MAX_K, init, runSeed), [points, init, runSeed]);

  const player = usePlayer<KMeansStep>(steps, 2);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = player.step;
  const cells = useMemo(() => (step.centroids.length ? voronoiCells(step.centroids, SIZE) : []), [step.centroids]);
  const changed = new Set(step.changed);
  const maxWeight = step.weights ? Math.max(...step.weights, 1e-9) : 1;
  const dataset = DATASETS.find((d) => d.id === kind)!;

  const iterations = steps.filter((s) => s.phase === 'update').length;

  return (
    <div className="viz" aria-label="k-means clustering visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Data</span>
          <select className="input" value={kind} onChange={(e) => setKind(e.target.value as DatasetKind)}>
            {DATASETS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Clusters k <output>{k}</output>
          </span>
          <input type="range" min={2} max={MAX_K} value={k} onChange={(e) => setK(+e.target.value)} />
        </label>
        <label className="viz-field">
          <span>Start</span>
          <select className="input" value={init} onChange={(e) => setInit(e.target.value as InitKind)}>
            <option value="kmeans++">k-means++</option>
            <option value="random">Random points</option>
          </select>
        </label>
        <button type="button" className="btn" onClick={() => setRunSeed((s) => s + 1)}>
          Restart
        </button>
        <button type="button" className="btn" onClick={() => setDataSeed((s) => s + 1)}>
          New points
        </button>
      </div>

      <p className="viz-note">{dataset.blurb}</p>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-split">
        <div>
          <svg className="viz-svg" viewBox={`-2 -2 ${SIZE + 4} ${SIZE + 4}`} role="img" aria-label={`${k} clusters over ${n} points`}>
            {/* Voronoi cells: the assignment rule made visible. A point belongs to the cell it sits in. */}
            {cells.map((poly, j) => (
              <polygon
                key={`c${j}`}
                points={poly.map((p) => `${r1(p.x)},${r1(p.y)}`).join(' ')}
                style={{ fill: 'var(--surface-2)', fillOpacity: j % 2 ? 0.55 : 0.25, stroke: 'var(--border)', strokeWidth: 0.4 }}
              />
            ))}

            {/* Leader lines for points that just changed hands. */}
            {step.phase === 'assign' &&
              step.changed.map((i) => {
                const p = points[i]!;
                const c = step.centroids[step.assignment[i]!];
                if (!c) return null;
                return (
                  <line
                    key={`l${i}`}
                    x1={r1(p.x)}
                    y1={r1(p.y)}
                    x2={r1(c.x)}
                    y2={r1(c.y)}
                    style={{ stroke: 'var(--role-change)', strokeWidth: 0.35, opacity: 0.55 }}
                  />
                );
              })}

            {/* Points. During k-means++ init the fill is the D² sampling weight (a magnitude). */}
            {points.map((p, i) => {
              const w = step.weights ? step.weights[i]! / maxWeight : 0;
              const fill = step.weights
                ? `color-mix(in oklab, var(--seq-7) ${(w * 100).toFixed(0)}%, var(--seq-0))`
                : changed.has(i)
                  ? 'var(--role-change)'
                  : 'var(--role-idle)';
              return (
                <circle key={i} cx={r1(p.x)} cy={r1(p.y)} r={changed.has(i) ? 1.5 : 1.2} style={{ fill }}>
                  <title>{`point ${i}: (${r1(p.x)}, ${r1(p.y)})${step.assignment[i]! >= 0 ? ` → cluster ${step.assignment[i]! + 1}` : ''}`}</title>
                </circle>
              );
            })}

            {/* Centroids, numbered: identity comes from the label, not from a hue. */}
            {step.centroids.map((c, j) => (
              <g key={`m${j}`}>
                <circle cx={r1(c.x)} cy={r1(c.y)} r={3.1} style={{ fill: 'var(--surface)', stroke: 'var(--text)', strokeWidth: 1 }} />
                <text x={r1(c.x)} y={r1(c.y) + 1.1} className="viz-svg-pin">
                  {j + 1}
                </text>
                <title>{`centroid ${j + 1}`}</title>
              </g>
            ))}
          </svg>

          <div className="viz-legend" style={{ marginTop: '0.5rem' }}>
            <span>
              <span className="sw" style={{ background: 'var(--surface-2)', border: '1px solid var(--border)' }} /> Voronoi cell: a point
              belongs to the cell it sits in
            </span>
            <Swatch color="var(--role-change)" label="changed cluster this step" />
            <span>① ② = centroids</span>
          </div>
        </div>

        <CodePanel listing={KMEANS_CODE} cursor={step.code} previous={player.steps[player.index - 1]?.code} title="Lloyd's algorithm" />
      </div>

      <div className="viz-stats">
        <Stat k="Iteration" v={`${step.iteration} / ${iterations}`} />
        <Stat k="Inertia (within-cluster SS)" v={step.inertia ? String(Math.round(step.inertia)) : '–'} />
        <Stat k="Points reassigned" v={step.phase === 'assign' ? String(step.changed.length) : '–'} />
        <Stat k="Largest centroid move" v={step.phase === 'update' ? step.shift.toFixed(1) : '–'} />
      </div>

      <div className="viz-pair">
        <figure>
          <figcaption className="viz-label">Inertia per half-step — it can only go down</figcaption>
          <LineChart
            series={[{ id: 'inertia', name: 'inertia', color: 'var(--cat-1)', values: step.history }]}
            xLabel="half-step (assign, update, assign, …)"
            yLabel="inertia"
            height={170}
            xMax={Math.max(steps[steps.length - 1]!.history.length, 2)}
            format={(v) => String(Math.round(v))}
          />
        </figure>
        <figure>
          <figcaption className="viz-label">Choosing k: inertia always falls, so look for the elbow</figcaption>
          <LineChart
            series={[{ id: 'elbow', name: 'final inertia', color: 'var(--cat-2)', values: elbow }]}
            xLabel={`k = 1 … ${MAX_K}   (currently k = ${k})`}
            yLabel="inertia"
            height={170}
            format={(v) => String(Math.round(v))}
          />
        </figure>
      </div>
    </div>
  );
}
