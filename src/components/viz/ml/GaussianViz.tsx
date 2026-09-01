import { useMemo, useState } from 'react';
import { CLASS_DATASETS, accuracy, argmax, computeField, makeClassData, trainTest, type ClassDataset } from '../../../lib/ml/classify';
import { GAUSS_KINDS, ellipse, fitGaussians, logDensity, normal1d, paramCount, posterior, type ClassGaussian, type GaussKind } from '../../../lib/ml/bayes';
import { Stat } from '../Stat';
import { CLASS_COLOR, CLASS_NAME, ClassLegend, DecisionMap, sy, useClient } from './DecisionMap';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const DATASETS: ClassDataset[] = ['three', 'correlated', 'circles', 'blobs', 'moons'];

const sci = (v: number) => (v === 0 ? '0' : v >= 0.01 ? v.toPrecision(3) : v.toExponential(2));

export default function GaussianViz({ n = 180 }: { n?: number }) {
  const [kind, setKind] = useState<GaussKind>('nb');
  const [data, setData] = useState<ClassDataset>('three');
  const [seed, setSeed] = useState(2);
  const [probe, setProbe] = useState({ x: 50, y: 55 });
  const [dragging, setDragging] = useState(false);
  const client = useClient();

  const classes = CLASS_DATASETS[data].classes;
  const { train, test } = useMemo(() => trainTest(makeClassData(data, n, seed)), [data, n, seed]);
  const gs = useMemo(() => fitGaussians(train, classes, kind), [train, classes, kind]);
  const field = useMemo(() => (client ? computeField((x, y) => posterior(gs, x, y), classes, 72) : null), [client, gs, classes]);

  const predict = (x: number, y: number) => argmax(posterior(gs, x, y));
  const post = posterior(gs, probe.x, probe.y);
  const info = GAUSS_KINDS.find((g) => g.id === kind)!;

  const onPlot = (k: 'down' | 'move' | 'up', at: { x: number; y: number }) => {
    if (k === 'down') {
      setDragging(true);
      setProbe(at);
    } else if (k === 'move' && dragging) setProbe(at);
    else if (k === 'up') setDragging(false);
  };

  return (
    <div className="viz" aria-label="Gaussian Naive Bayes, LDA and QDA visualization">
      <div className="viz-row">
        <div className="viz-seg" role="radiogroup" aria-label="Model">
          {GAUSS_KINDS.map((g) => (
            <button key={g.id} type="button" role="radio" aria-checked={kind === g.id} className={`btn${kind === g.id ? ' on' : ''}`} onClick={() => setKind(g.id)}>
              {g.name}
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

      <p className="viz-note">
        {CLASS_DATASETS[data].blurb} <strong>{info.name}</strong>: covariance {info.covariance}, so the boundary is {info.boundary}. Drag on the
        plot to move the probe.
      </p>

      <div className="viz-split">
        <div>
          <DecisionMap
            field={field}
            points={train}
            test={test}
            onPlot={onPlot}
            label={`${info.name} decision regions; the probe is classified as ${CLASS_NAME[argmax(post)]}`}
            under={gs.map((g, c) => (
              <GaussEllipses key={c} g={g} color={CLASS_COLOR[c]!} />
            ))}
          >
            {gs.map((g, c) => (
              <g key={`m${c}`} style={{ stroke: 'var(--text)', strokeWidth: 0.7 }}>
                <line x1={r1(g.mx - 2.2)} y1={sy(g.my)} x2={r1(g.mx + 2.2)} y2={sy(g.my)} />
                <line x1={r1(g.mx)} y1={sy(g.my - 2.2)} x2={r1(g.mx)} y2={sy(g.my + 2.2)} />
                <title>{`class ${CLASS_NAME[c]} mean (${g.mx.toFixed(1)}, ${g.my.toFixed(1)})`}</title>
              </g>
            ))}
            <circle cx={r1(probe.x)} cy={sy(probe.y)} r={2.4} style={{ fill: 'var(--surface)', stroke: 'var(--text)', strokeWidth: 0.8 }} />
            <circle cx={r1(probe.x)} cy={sy(probe.y)} r={1.1} style={{ fill: CLASS_COLOR[argmax(post)] }} />
          </DecisionMap>
          <div style={{ marginTop: '0.5rem' }}>
            <ClassLegend classes={classes} extra={<span>ellipses = fitted 1σ and 2σ contours; + = class mean; ◎ = probe</span>} />
          </div>
        </div>

        <div style={{ display: 'grid', gap: '1rem', alignContent: 'start', minWidth: 0 }}>
          <BayesTable gs={gs} kind={kind} x={probe.x} y={probe.y} />
          {kind === 'nb' && <Marginals gs={gs} x={probe.x} y={probe.y} />}
        </div>
      </div>

      <div className="viz-stats">
        <Stat k="Training accuracy" v={`${(accuracy(predict, train) * 100).toFixed(1)}%`} />
        <Stat k="Held-out accuracy" v={`${(accuracy(predict, test) * 100).toFixed(1)}%`} />
        <Stat k="Parameters" v={String(paramCount(kind, classes))} />
        <Stat k="Training" v="one pass: means + covariances" />
      </div>
    </div>
  );
}

function GaussEllipses({ g, color }: { g: ClassGaussian; color: string }) {
  const { a, b, angle } = ellipse(g.cov);
  const deg = r1((-angle * 180) / Math.PI); // SVG rotates clockwise, and its y is flipped
  return (
    <g transform={`rotate(${deg} ${r1(g.mx)} ${sy(g.my)})`} style={{ fill: 'none', stroke: color }}>
      <ellipse cx={r1(g.mx)} cy={sy(g.my)} rx={r1(a)} ry={r1(b)} style={{ strokeWidth: 0.6 }} />
      <ellipse cx={r1(g.mx)} cy={sy(g.my)} rx={r1(2 * a)} ry={r1(2 * b)} style={{ strokeWidth: 0.4, strokeDasharray: '1.6 1.2' }} />
    </g>
  );
}

/** Bayes' rule for the probe, one row per class. */
function BayesTable({ gs, kind, x, y }: { gs: ClassGaussian[]; kind: GaussKind; x: number; y: number }) {
  const lik = gs.map((g) => Math.exp(logDensity(g, x, y)));
  const joint = gs.map((g, c) => g.prior * lik[c]!);
  const total = joint.reduce((a, b) => a + b, 0);
  const win = argmax(joint);
  return (
    <div>
      <p className="viz-label">
        Bayes' rule at the probe ({x.toFixed(0)}, {y.toFixed(0)})
      </p>
      <div className="viz-scroll">
        <table className="viz-dist">
          <thead>
            <tr>
              <th>class</th>
              <th className="num">prior</th>
              <th className="num">{kind === 'nb' ? 'p(x|c)·p(y|c)' : 'p(x, y | c)'}</th>
              <th className="num">prior × lik.</th>
              <th className="num">posterior</th>
            </tr>
          </thead>
          <tbody>
            {gs.map((g, c) => (
              <tr key={c} style={c === win ? { fontWeight: 700 } : undefined}>
                <td>
                  <span className="sw" style={{ display: 'inline-block', width: '0.7rem', height: '0.7rem', borderRadius: 3, background: CLASS_COLOR[c], marginRight: 6 }} />
                  {CLASS_NAME[c]}
                </td>
                <td className="num mono">{g.prior.toFixed(2)}</td>
                <td className="num mono">{sci(lik[c]!)}</td>
                <td className="num mono">{sci(joint[c]!)}</td>
                <td className="num mono">{total > 0 ? `${((joint[c]! / total) * 100).toFixed(1)}%` : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="viz-note" style={{ marginTop: '0.4rem' }}>
        The posterior column is the middle product divided by its column sum ({sci(total)}), the evidence p(x, y).
      </p>
    </div>
  );
}

/** The naive factorization, drawn: one 1-D Gaussian per class per feature, and the probe's reading on each. */
function Marginals({ gs, x, y }: { gs: ClassGaussian[]; x: number; y: number }) {
  const W = 200;
  const H = 54;
  const panel = (axis: 'x' | 'y') => {
    const at = axis === 'x' ? x : y;
    const curves = gs.map((g) => {
      const m = axis === 'x' ? g.mx : g.my;
      const v = axis === 'x' ? g.cov.xx : g.cov.yy;
      return { m, v, pts: Array.from({ length: 101 }, (_, i) => normal1d(i, m, v)) };
    });
    const max = Math.max(...curves.flatMap((c) => c.pts));
    const Y = (d: number) => r1(H - 4 - (d / max) * (H - 10));
    const X = (v: number) => r1((v / 100) * W);
    return (
      <figure style={{ margin: 0 }}>
        <figcaption className="viz-axis-y">p({axis} | class) — feature {axis} on its own</figcaption>
        <svg viewBox={`0 0 ${W} ${H}`} className="viz-svg" role="img" aria-label={`Per-class Gaussian over feature ${axis}`}>
          <line x1={0} y1={H - 4} x2={W} y2={H - 4} style={{ stroke: 'var(--border)', strokeWidth: 0.5 }} />
          {curves.map((c, k) => (
            <g key={k}>
              <polyline points={c.pts.map((d, i) => `${X(i)},${Y(d)}`).join(' ')} style={{ fill: 'none', stroke: CLASS_COLOR[k], strokeWidth: 1 }} />
              <circle cx={X(at)} cy={Y(normal1d(at, c.m, c.v))} r={1.6} style={{ fill: CLASS_COLOR[k], stroke: 'var(--surface)', strokeWidth: 0.5 }} />
            </g>
          ))}
          <line x1={X(at)} y1={2} x2={X(at)} y2={H - 4} style={{ stroke: 'var(--text)', strokeWidth: 0.5, strokeDasharray: '2 1.5' }} />
        </svg>
      </figure>
    );
  };
  return (
    <div>
      <p className="viz-label">The naive assumption: multiply these</p>
      <div style={{ display: 'grid', gap: '0.5rem' }}>
        {panel('x')}
        {panel('y')}
      </div>
    </div>
  );
}
