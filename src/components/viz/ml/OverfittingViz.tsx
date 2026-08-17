import { useMemo, useState } from 'react';
import {
  FIT_CODE,
  TARGETS,
  biasVariance,
  degreeCurve,
  fit,
  makeSample,
  mse,
  predict,
  targetFn,
  type Penalty,
  type Target,
} from '../../../lib/ml/regression';
import { CodePanel } from '../CodePanel';
import { Swatch } from '../PlayerControls';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const MAX_DEGREE = 12;
const GRID = Array.from({ length: 121 }, (_, i) => -1 + (2 * i) / 120);
const Y_LO = -2.1;
const Y_HI = 2.1;

// λ slider is logarithmic: 0, then 1e-6 … 1e0.
const LAMBDAS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1];
const fmtLambda = (l: number) => (l === 0 ? '0' : l >= 0.01 ? l.toFixed(2) : l.toExponential(0));

const sx = (x: number) => ((x + 1) / 2) * 100;
const sy = (y: number) => ((Y_HI - y) / (Y_HI - Y_LO)) * 100;
const path = (ys: number[]) =>
  ys.map((y, i) => `${i ? 'L' : 'M'}${r1(sx(GRID[i]!))},${r1(sy(Math.max(Y_LO - 1, Math.min(Y_HI + 1, y))))}`).join(' ');

export default function OverfittingViz() {
  const [target, setTarget] = useState<Target>('sine');
  const [degree, setDegree] = useState(9);
  const [lambdaIdx, setLambdaIdx] = useState(0);
  const [penalty, setPenalty] = useState<Penalty>('none');
  const [n, setN] = useState(15);
  const [noise, setNoise] = useState(0.2);
  const [seed, setSeed] = useState(4);
  const [showRefits, setShowRefits] = useState(false);

  const lambda = LAMBDAS[lambdaIdx]!;
  const effective = penalty === 'none' ? 0 : lambda;
  const f = targetFn(target);

  const train = useMemo(() => makeSample(target, n, noise, seed), [target, n, noise, seed]);
  // A big, low-noise test set: the closest thing to measuring against the truth.
  const test = useMemo(() => makeSample(target, 200, noise, seed + 9999), [target, noise, seed]);

  const w = useMemo(() => fit(train, degree, effective, penalty), [train, degree, effective, penalty]);
  const curve = useMemo(() => degreeCurve(train, test, MAX_DEGREE, effective, penalty), [train, test, effective, penalty]);
  const bv = useMemo(
    () => (showRefits ? biasVariance(target, n, noise, degree, effective, penalty, GRID) : null),
    [showRefits, target, n, noise, degree, effective, penalty],
  );

  const fitted = GRID.map((x) => predict(w, x));
  const truth = GRID.map(f);
  const trainErr = mse(w, train);
  const testErr = mse(w, test);
  const nonZero = w.filter((c) => Math.abs(c) > 1e-6).length;
  const maxCoef = Math.max(1e-9, ...w.map(Math.abs));

  return (
    <div className="viz" aria-label="Overfitting and regularization visualization">
      <div className="viz-row">
        <label className="viz-field">
          <span>True function</span>
          <select className="input" value={target} onChange={(e) => setTarget(e.target.value as Target)}>
            {TARGETS.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Polynomial degree <output>{degree}</output>
          </span>
          <input type="range" min={0} max={MAX_DEGREE} value={degree} onChange={(e) => setDegree(+e.target.value)} />
        </label>
        <label className="viz-slider">
          <span>
            Training points <output>{n}</output>
          </span>
          <input type="range" min={5} max={60} value={n} onChange={(e) => setN(+e.target.value)} />
        </label>
        <label className="viz-slider">
          <span>
            Noise <output>{noise.toFixed(2)}</output>
          </span>
          <input type="range" min={0} max={60} value={Math.round(noise * 100)} onChange={(e) => setNoise(+e.target.value / 100)} />
        </label>
      </div>

      <div className="viz-row">
        <label className="viz-field">
          <span>Penalty</span>
          <select className="input" value={penalty} onChange={(e) => setPenalty(e.target.value as Penalty)}>
            <option value="none">None</option>
            <option value="l2">L2 · ridge</option>
            <option value="l1">L1 · lasso</option>
          </select>
        </label>
        <label className="viz-slider">
          <span>
            λ <output>{penalty === 'none' ? 'off' : fmtLambda(lambda)}</output>
          </span>
          <input
            type="range"
            min={0}
            max={LAMBDAS.length - 1}
            value={lambdaIdx}
            disabled={penalty === 'none'}
            onChange={(e) => setLambdaIdx(+e.target.value)}
          />
        </label>
        <label className="viz-check">
          <input type="checkbox" checked={showRefits} onChange={(e) => setShowRefits(e.target.checked)} /> Refit on 24 fresh samples
        </label>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          New sample
        </button>
      </div>

      <div className="viz-split">
        <div>
          <svg className="viz-svg" viewBox="-4 -4 108 108" role="img" aria-label="Model fit against the true function">
            <rect x={0} y={0} width={100} height={100} style={{ fill: 'var(--surface-2)', opacity: 0.5 }} />
            <line x1={0} y1={r1(sy(0))} x2={100} y2={r1(sy(0))} style={{ stroke: 'var(--border)', strokeWidth: 0.4 }} />

            {/* One faint curve per resampled training set: the spread is variance. */}
            {bv?.curves.map((c, i) => (
              <path key={i} d={path(c)} style={{ fill: 'none', stroke: 'var(--cat-2)', strokeWidth: 0.5, opacity: 0.3 }} />
            ))}
            {/* Their average: how far this sits from the truth is bias. */}
            {bv && <path d={path(bv.mean)} style={{ fill: 'none', stroke: 'var(--cat-2)', strokeWidth: 1.6, strokeDasharray: '3 2' }} />}

            <path d={path(truth)} style={{ fill: 'none', stroke: 'var(--text-3)', strokeWidth: 1.2, strokeDasharray: '4 2.5' }} />
            <path d={path(fitted)} style={{ fill: 'none', stroke: 'var(--cat-1)', strokeWidth: 1.8 }} />

            {train.xs.map((x, i) => (
              <circle key={i} cx={r1(sx(x))} cy={r1(sy(train.ys[i]!))} r={1.5} style={{ fill: 'var(--text)' }}>
                <title>{`training point (${x.toFixed(2)}, ${train.ys[i]!.toFixed(2)})`}</title>
              </circle>
            ))}

            <text x={50} y={106} className="viz-svg-mini" style={{ fontSize: 3.4 }}>
              x
            </text>
          </svg>

          <div className="viz-legend" style={{ marginTop: '0.5rem' }}>
            <Swatch color="var(--cat-1)" label="fitted model" />
            <span>
              <span className="sw" style={{ background: 'var(--text-3)' }} /> true function
            </span>
            <span>
              <span className="sw" style={{ background: 'var(--text)', borderRadius: '50%' }} /> training points
            </span>
            {showRefits && <Swatch color="var(--cat-2)" label="24 refits + their average" />}
          </div>
        </div>

        <CodePanel
          listing={FIT_CODE}
          cursor={{
            line: penalty === 'none' ? 'none' : penalty,
            vars: { degree, weights: degree + 1, λ: penalty === 'none' ? 'off' : fmtLambda(lambda) },
          }}
          title={penalty === 'none' ? 'Least squares' : penalty === 'l2' ? 'Ridge regression' : 'Lasso regression'}
        />
      </div>

      <div className="viz-stats">
        <Stat k="Training error" v={trainErr.toFixed(4)} />
        <Stat k="Test error" v={testErr.toFixed(4)} />
        <Stat k="Bias²" v={bv ? bv.bias2.toFixed(4) : '—'} />
        <Stat k="Variance" v={bv ? bv.variance.toFixed(4) : '—'} />
        <Stat k="Non-zero weights" v={`${nonZero} / ${degree + 1}`} />
      </div>

      <div className="viz-pair">
        <figure>
          <figcaption className="viz-label">Error vs degree — training always falls, test turns back up</figcaption>
          <LineChart
            series={[
              { id: 'train', name: 'training', color: 'var(--cat-1)', values: curve.trainErr },
              { id: 'test', name: 'test', color: 'var(--cat-2)', values: curve.testErr },
            ]}
            xLabel={`degree 0 … ${MAX_DEGREE}   (currently ${degree})`}
            yLabel="mean squared error"
            height={190}
            logY
          />
        </figure>
        <figure>
          <figcaption className="viz-label">Weights — regularization pulls them towards zero</figcaption>
          <div className="viz-coefs">
            {w.map((c, i) => (
              <div key={i} className="viz-coef" title={`w${i} = ${c.toPrecision(4)}`}>
                <span
                  className="viz-coef-bar"
                  style={{
                    height: `${Math.max(1, (Math.abs(c) / maxCoef) * 100)}%`,
                    background: c === 0 ? 'var(--role-idle)' : 'var(--cat-1)',
                  }}
                />
                <span className="viz-coef-label">{i}</span>
              </div>
            ))}
          </div>
          <p className="viz-note">Largest |w| = {maxCoef.toPrecision(3)}. Heights are relative, so watch the number, not the bars.</p>
        </figure>
      </div>
    </div>
  );
}
