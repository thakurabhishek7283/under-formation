import { useEffect, useMemo, useRef, useState } from 'react';
import { CLASS_DATASETS, fieldFromScore, makeClassData, type ClassDataset, type LPoint } from '../../../lib/ml/classify';
import { decision, svRole, trainSvm, weightVector, type KernelKind } from '../../../lib/ml/svm';
import { Swatch } from '../PlayerControls';
import { Stat } from '../Stat';
import { CLASS_COLOR, CLASS_NAME, ClassLegend, DecisionMap, useClient } from './DecisionMap';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const DATASETS: ClassDataset[] = ['separable', 'blobs', 'moons', 'circles', 'xor'];
const CS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000];
const GAMMAS = [0.1, 0.3, 1, 3, 10, 30];
const KERNELS: { id: KernelKind; name: string; formula: string }[] = [
  { id: 'linear', name: 'Linear', formula: 'K(a, b) = a·b' },
  { id: 'poly', name: 'Polynomial', formula: 'K(a, b) = (a·b + 1)²' },
  { id: 'rbf', name: 'RBF', formula: 'K(a, b) = exp(−γ‖a − b‖²)' },
];

export default function SvmViz({ n = 70 }: { n?: number }) {
  const [data, setData] = useState<ClassDataset>('separable');
  const [seed, setSeed] = useState(6);
  const [points, setPoints] = useState<LPoint[]>(() => makeClassData('separable', n, 6));
  const [kernelKind, setKernelKind] = useState<KernelKind>('linear');
  const [cIdx, setCIdx] = useState(6);
  const [gIdx, setGIdx] = useState(2);
  const [addClass, setAddClass] = useState(0);
  const drag = useRef<number | null>(null);
  const client = useClient();

  useEffect(() => setPoints(makeClassData(data, n, seed)), [data, n, seed]);

  const options = useMemo(() => ({ kernel: kernelKind, C: CS[cIdx]!, gamma: GAMMAS[gIdx]! }), [kernelKind, cIdx, gIdx]);
  const model = useMemo(() => trainSvm(points, options), [points, options]);
  const shaded = useMemo(() => (client ? fieldFromScore((x, y) => decision(model, x, y), 72, 2) : null), [client, model]);
  const lines = useMemo(
    () =>
      shaded
        ? [
            { values: shaded.values, level: 1, dashed: true },
            { values: shaded.values, level: -1, dashed: true },
          ]
        : undefined,
    [shaded],
  );

  const roles = points.map((_, i) => svRole(model, i));
  const svCount = roles.filter((r) => r !== 'none').length;
  const boundCount = roles.filter((r) => r === 'bound').length;
  const trainAcc = points.filter((p) => (decision(model, p.x, p.y) > 0 ? 1 : 0) === p.label).length / Math.max(points.length, 1);
  const w = kernelKind === 'linear' ? weightVector(model) : null;
  const margin = w ? (2 / Math.hypot(w[0], w[1])) * 25 : null; // back from model units to plot units

  const onPlot = (kind: 'down' | 'move' | 'up', at: { x: number; y: number }) => {
    if (kind === 'down') {
      let best = -1;
      let bestD = 3.2;
      points.forEach((p, i) => {
        const d = Math.hypot(p.x - at.x, p.y - at.y);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      });
      if (best >= 0) drag.current = best;
      else setPoints((ps) => [...ps, { x: at.x, y: at.y, label: addClass }]);
    } else if (kind === 'move' && drag.current !== null) {
      const i = drag.current;
      setPoints((ps) => ps.map((p, j) => (j === i ? { ...p, x: at.x, y: at.y } : p)));
    } else if (kind === 'up') drag.current = null;
  };

  const alphas = model.alpha.map((a, i) => ({ a, i })).sort((p, q) => q.a - p.a);
  const aMax = Math.max(...model.alpha, 1e-9);

  return (
    <div className="viz" aria-label="Support vector machine visualization">
      <div className="viz-row">
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
        <div className="viz-seg" role="radiogroup" aria-label="Kernel">
          {KERNELS.map((k) => (
            <button key={k.id} type="button" role="radio" aria-checked={kernelKind === k.id} className={`btn${kernelKind === k.id ? ' on' : ''}`} onClick={() => setKernelKind(k.id)}>
              {k.name}
            </button>
          ))}
        </div>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          New data
        </button>
      </div>
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            C (penalty per margin violation) <output>{CS[cIdx]}</output>
          </span>
          <input type="range" min={0} max={CS.length - 1} value={cIdx} onChange={(e) => setCIdx(+e.target.value)} />
        </label>
        {kernelKind === 'rbf' && (
          <label className="viz-slider">
            <span>
              RBF γ <output>{GAMMAS[gIdx]}</output>
            </span>
            <input type="range" min={0} max={GAMMAS.length - 1} value={gIdx} onChange={(e) => setGIdx(+e.target.value)} />
          </label>
        )}
        <div className="viz-seg" role="radiogroup" aria-label="Class for new points">
          {[0, 1].map((c) => (
            <button key={c} type="button" role="radio" aria-checked={addClass === c} className={`btn${addClass === c ? ' on' : ''}`} onClick={() => setAddClass(c)}>
              Click adds {CLASS_NAME[c]}
            </button>
          ))}
        </div>
      </div>

      <p className="viz-note">
        {KERNELS.find((k) => k.id === kernelKind)!.formula}. Drag any point to move it; click empty space to add one. Drag a point that isn't
        ringed and watch the boundary ignore it.
      </p>

      <div className="viz-split">
        <div>
          <DecisionMap
            field={shaded?.field ?? null}
            lines={lines}
            points={points}
            onPlot={onPlot}
            cursor="pointer"
            label={`SVM with ${kernelKind} kernel: ${svCount} support vectors of ${points.length} points`}
            marker={(_, i) =>
              roles[i] === 'margin'
                ? { ring: 'var(--text)', r: 1.7 }
                : roles[i] === 'bound'
                  ? { ring: 'var(--role-change)', r: 1.7 }
                  : undefined
            }
          />
          <div style={{ marginTop: '0.5rem' }}>
            <ClassLegend
              classes={2}
              held={false}
              extra={
                <>
                  <span>solid line f = 0, dashed f = ±1 (the margin)</span>
                  <Swatch color="var(--text)" label="support vector on the margin" outline />
                  <Swatch color="var(--role-change)" label="support vector at α = C (inside the margin or wrong)" outline />
                </>
              }
            />
          </div>
        </div>

        <div style={{ display: 'grid', gap: '1rem', alignContent: 'start', minWidth: 0 }}>
          <figure style={{ margin: 0 }}>
            <figcaption className="viz-label">α for every training point, largest first</figcaption>
            <svg viewBox={`0 0 ${alphas.length * 3} 40`} className="viz-svg" role="img" aria-label={`${svCount} of ${points.length} points have non-zero alpha`}>
              <line x1={0} y1={38} x2={alphas.length * 3} y2={38} style={{ stroke: 'var(--border)', strokeWidth: 0.4 }} />
              {alphas.map(({ a, i }, k) => {
                const h = r1((a / aMax) * 34);
                return (
                  <rect key={i} x={k * 3 + 0.4} y={r1(38 - h)} width={2.2} height={Math.max(h, 0)} style={{ fill: CLASS_COLOR[points[i]!.label] }}>
                    <title>{`point ${i}: α = ${a.toFixed(4)}`}</title>
                  </rect>
                );
              })}
            </svg>
            <p className="viz-note" style={{ marginTop: '0.4rem' }}>
              {svCount} of {points.length} bars are non-zero. The other {points.length - svCount} points could be deleted without changing
              anything.
            </p>
          </figure>
          <div className="viz-stats">
            <Stat k="Support vectors" v={`${svCount} / ${points.length}`} />
            <Stat k="At the bound α = C" v={String(boundCount)} />
            <Stat k={margin !== null ? 'Margin width' : 'Training accuracy'} v={margin !== null ? margin.toFixed(1) : `${(trainAcc * 100).toFixed(1)}%`} />
            <Stat k="SMO iterations" v={`${model.iterations}${model.converged ? '' : '+'}`} />
          </div>
          {margin !== null && (
            <p className="viz-note" style={{ margin: 0 }}>
              Training accuracy {(trainAcc * 100).toFixed(1)}%. Margin = 2 / ‖w‖ with w = Σ αᵢyᵢxᵢ.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
