import { useMemo, useState } from 'react';
import { bestF1, confusionAt, curves, histogram, makeScores, metricsOf, type CurvePoint } from '../../../lib/ml/metrics';
import { Swatch } from '../PlayerControls';
import { Stat } from '../Stat';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const N = 600;
const BINS = 44;
const LO = -3.4;
const HI = 6.2;
const POS = 'var(--cat-2)';
const NEG = 'var(--cat-1)';

export default function MetricsViz() {
  const [separation, setSeparation] = useState(2);
  const [prevalence, setPrevalence] = useState(0.35);
  const [threshold, setThreshold] = useState(1);
  const [seed, setSeed] = useState(21);

  const data = useMemo(() => makeScores(N, separation, prevalence, seed), [separation, prevalence, seed]);
  const { roc, auc, ap } = useMemo(() => curves(data), [data]);
  const hist = useMemo(() => histogram(data, BINS, LO, HI), [data]);
  const best = useMemo(() => bestF1(data), [data]);

  const c = confusionAt(data, threshold);
  const m = metricsOf(c);
  const maxBin = Math.max(1, ...hist.neg, ...hist.pos);
  const positives = c.tp + c.fn;

  // The point on each curve for the threshold currently selected.
  const here: CurvePoint = roc.reduce((acc, p) => (p.threshold >= threshold && p.threshold < acc.threshold ? p : acc), {
    threshold: Infinity,
    fpr: 0,
    tpr: 0,
    precision: 1,
  });

  return (
    <div className="viz" aria-label="Classification metrics visualization">
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Class separation <output>{separation.toFixed(1)}σ</output>
          </span>
          <input type="range" min={0} max={40} value={Math.round(separation * 10)} onChange={(e) => setSeparation(+e.target.value / 10)} />
        </label>
        <label className="viz-slider">
          <span>
            Positives in the data <output>{(prevalence * 100).toFixed(0)}%</output>
          </span>
          <input
            type="range"
            min={1}
            max={90}
            value={Math.round(prevalence * 100)}
            onChange={(e) => setPrevalence(+e.target.value / 100)}
          />
        </label>
        <button type="button" className="btn" onClick={() => setThreshold(best.threshold)}>
          Best F1 threshold
        </button>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          Resample
        </button>
      </div>

      <label className="viz-slider">
        <span>
          Decision threshold <output>{threshold.toFixed(2)}</output> — everything to the right is predicted positive
        </span>
        <input
          type="range"
          min={-340}
          max={620}
          value={Math.round(threshold * 100)}
          onChange={(e) => setThreshold(+e.target.value / 100)}
        />
      </label>

      {/* Score distributions: the whole problem in one picture. */}
      <figure style={{ margin: 0 }}>
        <figcaption className="viz-label">Model scores, by true class</figcaption>
        {/* Wide and short: a 100×24 viewBox stretched to the container width. Axis labels live
            in HTML underneath rather than in the SVG, where they would scale with the viewBox. */}
        <svg
          className="viz-svg"
          viewBox="0 0 100 24"
          role="img"
          aria-label="Score distributions for both classes with the decision threshold"
        >
          {(['neg', 'pos'] as const).map((side) =>
            hist[side].map((v, i) => (
              <rect
                key={`${side}${i}`}
                x={r1((i / BINS) * 100 + 0.25)}
                y={r1(24 - (v / maxBin) * 24)}
                width={r1(100 / BINS - 0.5)}
                height={r1((v / maxBin) * 24)}
                style={{ fill: side === 'pos' ? POS : NEG, opacity: 0.78 }}
              >
                <title>{`${v} ${side === 'pos' ? 'true positives' : 'true negatives'} near score ${(LO + ((i + 0.5) / BINS) * (HI - LO)).toFixed(1)}`}</title>
              </rect>
            )),
          )}
          <line
            x1={r1(((threshold - LO) / (HI - LO)) * 100)}
            y1={0}
            x2={r1(((threshold - LO) / (HI - LO)) * 100)}
            y2={24}
            style={{ stroke: 'var(--text)', strokeWidth: 0.4 }}
          />
        </svg>
        <div className="viz-axis">
          <span>← predicted negative</span>
          <span>threshold at {threshold.toFixed(2)}</span>
          <span>predicted positive →</span>
        </div>
        <div className="viz-legend">
          <Swatch color={NEG} label="actually negative" />
          <Swatch color={POS} label="actually positive" />
          <span>overlap = the errors no threshold can remove</span>
        </div>
      </figure>

      {/* Three panels in one row: the matrix and both curves are the same story at one threshold
          and at all of them. Nesting a second grid inside one column collapses them, so don't. */}
      <div className="viz-trio">
        <figure>
          <figcaption className="viz-label">Confusion matrix at this threshold</figcaption>
          <div className="viz-cm">
            <span />
            <span className="viz-cm-head">predicted −</span>
            <span className="viz-cm-head">predicted +</span>
            <span className="viz-cm-head side">actually +</span>
            <Cell n={c.fn} total={N} label="false negative" note="missed" tone="bad" />
            <Cell n={c.tp} total={N} label="true positive" note="caught" tone="good" />
            <span className="viz-cm-head side">actually −</span>
            <Cell n={c.tn} total={N} label="true negative" note="correctly ignored" tone="good" />
            <Cell n={c.fp} total={N} label="false positive" note="false alarm" tone="bad" />
          </div>
          <p className="viz-note">
            Recall reads across the top row ({c.tp}/{positives}); precision reads down the right column ({c.tp}/{c.tp + c.fp}).
          </p>
        </figure>

        <Curve
          title={`ROC — AUC ${auc.toFixed(3)}`}
          points={roc.map((p) => [p.fpr, p.tpr] as [number, number])}
          here={[here.fpr, here.tpr]}
          xLabel="false positive rate →"
          yLabel="↑ recall"
          diagonal
        />
        <Curve
          title={`Precision–recall — AP ${ap.toFixed(3)}`}
          points={roc.filter((p) => Number.isFinite(p.threshold)).map((p) => [p.tpr, p.precision] as [number, number])}
          here={[here.tpr, here.precision]}
          xLabel="recall →"
          yLabel="↑ precision"
          baseline={prevalence}
        />
      </div>

      <div className="viz-stats">
        <Stat k="Accuracy" v={`${(m.accuracy * 100).toFixed(1)}%`} />
        <Stat k="Always-majority baseline" v={`${(m.baseline * 100).toFixed(1)}%`} />
        <Stat k="Precision" v={`${(m.precision * 100).toFixed(1)}%`} />
        <Stat k="Recall" v={`${(m.recall * 100).toFixed(1)}%`} />
        <Stat k="F1" v={m.f1.toFixed(3)} />
      </div>
    </div>
  );
}

function Cell({ n, total, label, note, tone }: { n: number; total: number; label: string; note: string; tone: 'good' | 'bad' }) {
  return (
    <span className={`viz-cm-cell ${tone}`} title={`${label}: ${n}`}>
      <strong>{n}</strong>
      <em>{note}</em>
      <span className="viz-cm-pct">{((n / total) * 100).toFixed(1)}%</span>
    </span>
  );
}

/** A unit-square curve with the current operating point marked. */
function Curve({
  title,
  points,
  here,
  xLabel,
  yLabel,
  diagonal = false,
  baseline,
}: {
  title: string;
  points: [number, number][];
  here: [number, number];
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  baseline?: number;
}) {
  const d = points.map(([x, y], i) => `${i ? 'L' : 'M'}${r1(x * 100)},${r1(100 - y * 100)}`).join(' ');
  return (
    <figure style={{ margin: 0 }}>
      <figcaption className="viz-label" style={{ marginBottom: '0.25rem' }}>
        {title}
      </figcaption>
      <p className="viz-axis-y">{yLabel}</p>
      <svg className="viz-svg" viewBox="-1 -1 102 102" role="img" aria-label={title}>
        <rect x={0} y={0} width={100} height={100} style={{ fill: 'var(--surface-2)', opacity: 0.6 }} />
        {diagonal && <line x1={0} y1={100} x2={100} y2={0} style={{ stroke: 'var(--text-3)', strokeWidth: 0.5, strokeDasharray: '3 2' }} />}
        {baseline !== undefined && (
          <line
            x1={0}
            y1={r1(100 - baseline * 100)}
            x2={100}
            y2={r1(100 - baseline * 100)}
            style={{ stroke: 'var(--text-3)', strokeWidth: 0.5, strokeDasharray: '3 2' }}
          />
        )}
        <path d={d} style={{ fill: 'none', stroke: 'var(--cat-3)', strokeWidth: 1.6 }} />
        <circle
          cx={r1(here[0] * 100)}
          cy={r1(100 - here[1] * 100)}
          r={2.4}
          style={{ fill: 'var(--text)', stroke: 'var(--surface)', strokeWidth: 1 }}
        />
      </svg>
      <p className="viz-axis-x">{xLabel}</p>
    </figure>
  );
}
