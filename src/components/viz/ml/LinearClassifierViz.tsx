import { useEffect, useMemo, useState } from 'react';
import { CLASS_DATASETS, computeField, makeClassData, trainTest, type ClassDataset } from '../../../lib/ml/classify';
import {
  FEATURE_NAMES,
  LOGISTIC_CODE,
  PERCEPTRON_CODE,
  lineInBox,
  logisticSteps,
  perceptronSteps,
  score,
  sigmoid,
  type FeatureMap,
  type LinearMethod,
  type LinearStep,
} from '../../../lib/ml/linear';
import { usePlayer } from '../usePlayer';
import { PlayerControls } from '../PlayerControls';
import { CodePanel } from '../CodePanel';
import { Stat } from '../Stat';
import { Heatmap } from './Heatmap';
import { LineChart } from './LineChart';
import { ClassLegend, ClassMarker, DecisionMap, sy, useClient } from './DecisionMap';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

type Data = Extract<ClassDataset, 'separable' | 'blobs' | 'moons' | 'circles' | 'xor'>;
const DATA: { id: Data; name: string; blurb: string }[] = [
  { id: 'separable', name: 'Separable', blurb: CLASS_DATASETS.separable.blurb },
  { id: 'blobs', name: 'Overlapping blobs', blurb: CLASS_DATASETS.blobs.blurb },
  { id: 'moons', name: 'Two moons', blurb: CLASS_DATASETS.moons.blurb },
  { id: 'circles', name: 'Circles', blurb: `${CLASS_DATASETS.circles.blurb} Switch features to quadratic.` },
  { id: 'xor', name: 'XOR', blurb: `${CLASS_DATASETS.xor.blurb} The xy feature is exactly what's missing.` },
];
const LRS = [0.03, 0.1, 0.3, 1, 3, 10];
const LAMBDAS = [0, 0.001, 0.01, 0.03, 0.1];

export default function LinearClassifierViz({ n = 140 }: { n?: number }) {
  const [method, setMethod] = useState<LinearMethod>('logistic');
  const [data, setData] = useState<Data>('separable');
  const [map, setMap] = useState<FeatureMap>('linear');
  const [lrIdx, setLrIdx] = useState(3);
  const [lamIdx, setLamIdx] = useState(0);
  const [seed, setSeed] = useState(5);
  const client = useClient();

  const { train, test } = useMemo(() => trainTest(makeClassData(data, n, seed)), [data, n, seed]);
  const steps = useMemo(() => {
    const o = { map, lr: LRS[lrIdx]!, lambda: LAMBDAS[lamIdx]!, epochs: method === 'logistic' ? 120 : 40 };
    return method === 'logistic' ? logisticSteps(train, test, o) : perceptronSteps(train, test, o);
  }, [method, train, test, map, lrIdx, lamIdx]);

  const player = usePlayer<LinearStep>(steps, method === 'logistic' ? 5 : 3);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps
  const step = player.step;
  const w = step.w;

  const field = useMemo(() => {
    if (!client) return null;
    const norm = Math.hypot(...w.slice(1)) || 1;
    return computeField((x, y) => {
      const s = score(w, x, y, map);
      // The perceptron has no probabilities; shade by distance from the boundary instead.
      const p = method === 'logistic' ? sigmoid(s) : sigmoid((3 * s) / norm);
      return [1 - p, p];
    }, 2);
  }, [client, w, map, method]);

  const line = map === 'linear' ? lineInBox(w) : null;
  const prevLine = map === 'linear' && step.prev ? lineInBox(step.prev) : null;
  const focus = step.focus !== undefined ? train[step.focus] : undefined;
  const arrow = (() => {
    if (!line) return null;
    const mx = (line[0] + line[2]) / 2;
    const my = (line[1] + line[3]) / 2;
    const len = Math.hypot(w[1]!, w[2]!) || 1;
    return { x1: mx, y1: my, x2: mx + (w[1]! / len) * 11, y2: my + (w[2]! / len) * 11 };
  })();

  const info = DATA.find((d) => d.id === data)!;
  const names = FEATURE_NAMES[map];
  const wMax = Math.max(0.5, ...w.map(Math.abs));

  return (
    <div className="viz" aria-label="Logistic regression and perceptron visualization">
      <div className="viz-row">
        <div className="viz-seg" role="radiogroup" aria-label="Method">
          {(['logistic', 'perceptron'] as const).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={method === m} className={`btn${method === m ? ' on' : ''}`} onClick={() => setMethod(m)}>
              {m === 'logistic' ? 'Logistic regression' : 'Perceptron'}
            </button>
          ))}
        </div>
        <label className="viz-field grow">
          <span>Data</span>
          <select className="input" value={data} onChange={(e) => setData(e.target.value as Data)}>
            {DATA.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-field">
          <span>Features φ(x, y)</span>
          <select className="input" value={map} onChange={(e) => setMap(e.target.value as FeatureMap)}>
            <option value="linear">linear: 1, x, y</option>
            <option value="quadratic">quadratic: + x², y², xy</option>
          </select>
        </label>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          New data
        </button>
      </div>
      {method === 'logistic' && (
        <div className="viz-row">
          <label className="viz-slider">
            <span>
              Learning rate <output>{LRS[lrIdx]}</output>
            </span>
            <input type="range" min={0} max={LRS.length - 1} value={lrIdx} onChange={(e) => setLrIdx(+e.target.value)} />
          </label>
          <label className="viz-slider">
            <span>
              L2 penalty λ <output>{LAMBDAS[lamIdx]}</output>
            </span>
            <input type="range" min={0} max={LAMBDAS.length - 1} value={lamIdx} onChange={(e) => setLamIdx(+e.target.value)} />
          </label>
        </div>
      )}

      <p className="viz-note">{info.blurb}</p>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-split">
        <div>
          <DecisionMap
            field={field}
            points={train}
            test={test}
            boundary={map !== 'linear'}
            label={`${method === 'logistic' ? 'Logistic regression' : 'Perceptron'} decision boundary at epoch ${step.epoch}`}
            marker={(_, i, held) => (!held && i === step.focus ? { ring: 'var(--role-change)', r: 2 } : undefined)}
          >
            {prevLine && (
              <line
                x1={r1(prevLine[0])}
                y1={sy(prevLine[1])}
                x2={r1(prevLine[2])}
                y2={sy(prevLine[3])}
                style={{ stroke: 'var(--text-3)', strokeWidth: 0.5, strokeDasharray: '1.5 1.2' }}
              />
            )}
            {line && (
              <line x1={r1(line[0])} y1={sy(line[1])} x2={r1(line[2])} y2={sy(line[3])} style={{ stroke: 'var(--text)', strokeWidth: 0.7 }} />
            )}
            {arrow && (
              <g style={{ stroke: 'var(--text)', strokeWidth: 0.6, fill: 'var(--text)' }}>
                <line x1={r1(arrow.x1)} y1={sy(arrow.y1)} x2={r1(arrow.x2)} y2={sy(arrow.y2)} />
                <circle cx={r1(arrow.x2)} cy={sy(arrow.y2)} r={0.9} />
                <text x={r1(arrow.x2 + (arrow.x2 - arrow.x1) * 0.35)} y={sy(arrow.y2 + (arrow.y2 - arrow.y1) * 0.35) + 1} style={{ fontSize: 3.2, stroke: 'none' }}>
                  w
                </text>
              </g>
            )}
            {focus && (
              <circle cx={r1(focus.x)} cy={sy(focus.y)} r={3.4} style={{ fill: 'none', stroke: 'var(--role-change)', strokeWidth: 0.6 }} />
            )}
          </DecisionMap>
          <div style={{ marginTop: '0.5rem' }}>
            <ClassLegend
              classes={2}
              extra={
                <span>
                  {method === 'logistic' ? 'shade = P(class)' : 'shade = distance from the boundary'}
                  {map === 'linear' ? '; arrow w points towards class B' : ''}
                </span>
              }
            />
          </div>
        </div>
        <CodePanel
          listing={method === 'logistic' ? LOGISTIC_CODE : PERCEPTRON_CODE}
          cursor={step.code}
          previous={player.steps[player.index - 1]?.code}
          title={method === 'logistic' ? 'Logistic regression (gradient descent)' : 'Perceptron learning rule'}
        />
      </div>

      <div>
        <p className="viz-label">Weights — one per feature; the boundary is where Σ wᵢφᵢ = 0</p>
        <div className="viz-scroll">
          <Heatmap
            data={[w]}
            scale="diverging"
            domain={wMax}
            colLabels={names}
            showValues
            cell="3.2rem"
            format={(v) => v.toFixed(2)}
            tip={(_, j, v) => `w for ${names[j]}: ${v.toFixed(3)}`}
            label="Weight vector"
          />
        </div>
      </div>

      {method === 'logistic' && <SigmoidPanel w={w} map={map} train={train} />}

      <div className="viz-stats">
        <Stat k="Epoch" v={String(step.epoch)} />
        <Stat k={method === 'logistic' ? 'Log loss' : 'Mistakes last epoch'} v={method === 'logistic' ? step.loss.toFixed(3) : step.history.length ? String(step.history.at(-1)) : '–'} />
        <Stat k="Training accuracy" v={`${(step.trainAcc * 100).toFixed(1)}%`} />
        <Stat k="Held-out accuracy" v={`${(step.testAcc * 100).toFixed(1)}%`} />
        <Stat k="|w|" v={Math.hypot(...w.slice(1)).toFixed(2)} />
      </div>

      <figure style={{ margin: 0 }}>
        <figcaption className="viz-label">{method === 'logistic' ? 'Training log loss per epoch' : 'Mistakes per epoch — zero means it has stopped'}</figcaption>
        <LineChart
          series={[{ id: 'h', name: method === 'logistic' ? 'log loss' : 'mistakes', color: 'var(--cat-1)', values: step.history }]}
          xLabel="epoch"
          yLabel={method === 'logistic' ? 'loss' : 'mistakes'}
          height={170}
          xMax={Math.max(steps.at(-1)!.history.length, 2)}
          format={method === 'logistic' ? undefined : (v) => v.toFixed(0)}
        />
      </figure>
    </div>
  );
}

/** Every training sample placed at its score z = w·φ, lifted onto the sigmoid. */
function SigmoidPanel({ w, map, train }: { w: number[]; map: FeatureMap; train: { x: number; y: number; label: number }[] }) {
  const Z = 8;
  const W = 400;
  const H = 64;
  const X = (z: number) => r1(((Math.max(-Z, Math.min(Z, z)) + Z) / (2 * Z)) * W);
  const Y = (p: number) => r1(4 + (1 - p) * (H - 8));
  const curve = Array.from({ length: 81 }, (_, i) => {
    const z = -Z + (i / 80) * 2 * Z;
    return `${X(z)},${Y(sigmoid(z))}`;
  }).join(' ');
  return (
    <figure style={{ margin: 0 }}>
      <figcaption className="viz-label">Each sample at its score z = w·φ — the sigmoid turns it into P(B)</figcaption>
      <svg viewBox={`-4 0 ${W + 8} ${H + 8}`} className="viz-svg" role="img" aria-label="Samples placed on the sigmoid curve by their score">
        <line x1={0} y1={Y(0.5)} x2={W} y2={Y(0.5)} style={{ stroke: 'var(--border)', strokeWidth: 0.4, strokeDasharray: '2 2' }} />
        <line x1={X(0)} y1={2} x2={X(0)} y2={H - 2} style={{ stroke: 'var(--border)', strokeWidth: 0.4 }} />
        <polyline points={curve} style={{ fill: 'none', stroke: 'var(--text-3)', strokeWidth: 0.8 }} />
        {train.map((p, i) => {
          const z = score(w, p.x, p.y, map);
          // ClassMarker draws in plot units, where (50, 0) lands at (50, 100); shift that to the origin.
          return (
            <g key={i} transform={`translate(${X(z)} ${Y(sigmoid(z))}) translate(-50 -100)`}>
              <ClassMarker x={50} y={0} label={p.label} r={2.2} />
            </g>
          );
        })}
        <text x={2} y={H + 7} className="viz-svg-mini" style={{ fontSize: 6, textAnchor: 'start' }}>
          z = −{Z}
        </text>
        <text x={X(0)} y={H + 7} className="viz-svg-mini" style={{ fontSize: 6 }}>
          0
        </text>
        <text x={W - 2} y={H + 7} className="viz-svg-mini" style={{ fontSize: 6, textAnchor: 'end' }}>
          +{Z}
        </text>
      </svg>
    </figure>
  );
}
