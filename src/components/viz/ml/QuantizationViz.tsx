import { useMemo, useState } from 'react';
import { demoWeights, modelGB, quantize } from '../../../lib/ml/quantization';
import { Stat } from '../Stat';
import { pct, r1 } from './num';
import '../viz.css';
import './ml.css';

const N = 512;
const GROUPS = [
  { label: 'Per-tensor (one scale)', size: 0 },
  { label: 'Per-group of 128', size: 128 },
  { label: 'Per-group of 64', size: 64 },
  { label: 'Per-group of 32', size: 32 },
];
const BINS = 48;

function histogram(values: number[], lo: number, hi: number): number[] {
  const h = new Array<number>(BINS).fill(0);
  for (const v of values) h[Math.max(0, Math.min(BINS - 1, Math.floor(((v - lo) / (hi - lo)) * BINS)))]!++;
  return h;
}

function Histogram({ counts, color, label, max }: { counts: number[]; color: string; label: string; max: number }) {
  const W = 480;
  const H = 90;
  const bw = W / counts.length;
  return (
    <figure className="viz-hist">
      <figcaption className="viz-label">{label}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {counts.map((c, i) => {
          const h = c === 0 ? 0 : Math.max(2, (Math.sqrt(c) / Math.sqrt(max)) * (H - 4));
          return (
            <rect key={i} x={r1(i * bw + 1)} y={r1(H - h)} width={r1(bw - 2)} height={r1(h)} rx={2} fill={color}>
              <title>{`${c} weight${c === 1 ? '' : 's'}`}</title>
            </rect>
          );
        })}
        <line x1={W / 2} x2={W / 2} y1={0} y2={H} className="viz-hist-zero" />
      </svg>
    </figure>
  );
}

export default function QuantizationViz() {
  const [bits, setBits] = useState(4);
  const [groupIdx, setGroupIdx] = useState(0);
  const [outliers, setOutliers] = useState(true);

  const weights = useMemo(() => demoWeights(N, outliers), [outliers]);
  const res = useMemo(() => quantize(weights, bits, GROUPS[groupIdx]!.size), [weights, bits, groupIdx]);
  const range = Math.max(...weights.map(Math.abs)) * 1.02;
  const orig = histogram(weights, -range, range);
  const deq = histogram(res.dequant, -range, range);
  const hMax = Math.max(...orig, ...deq);

  // Staircase plot: original (x) vs dequantized (y).
  const S = 260;
  const sp = (v: number) => ((v + range) / (2 * range)) * S;

  return (
    <div className="viz" aria-label="Quantization visualization">
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Bits per weight <output>{bits}</output> → {2 ** bits - 1} levels (−{2 ** (bits - 1) - 1} … +{2 ** (bits - 1) - 1})
          </span>
          <input type="range" min={2} max={8} value={bits} onChange={(e) => setBits(+e.target.value)} />
        </label>
        <label className="viz-field">
          <span>Scale granularity</span>
          <select className="input" value={groupIdx} onChange={(e) => setGroupIdx(+e.target.value)}>
            {GROUPS.map((g, i) => (
              <option key={g.label} value={i}>
                {g.label}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-check">
          <input type="checkbox" checked={outliers} onChange={(e) => setOutliers(e.target.checked)} /> Include 3 outlier weights
        </label>
      </div>

      <div className="viz-quant">
        <div>
          <Histogram counts={orig} color="var(--role-idle)" label={`Original fp16 weights (${N}, std 0.02)`} max={hMax} />
          <Histogram counts={deq} color="var(--role-compare)" label={`After int${bits} round-trip: values snap to ${res.levelsUsed} levels`} max={hMax} />
          <p className="muted" style={{ fontSize: '0.8rem', margin: 0 }}>
            x-axis: weight value from −{range.toFixed(2)} to +{range.toFixed(2)}. Bar height ∝ √count, so small bins stay visible.
          </p>
        </div>
        <figure className="viz-stair">
          <figcaption className="viz-label">Original → dequantized</figcaption>
          <svg viewBox={`0 0 ${S} ${S}`} role="img" aria-label="Original versus dequantized weights">
            <line x1={0} y1={S} x2={S} y2={0} className="viz-hist-zero" />
            {weights.map((w, i) => (
              <circle key={i} cx={r1(sp(w))} cy={r1(S - sp(res.dequant[i]!))} r={2.2} fill="var(--role-compare)" opacity={0.6}>
                <title>{`${w.toFixed(4)} → ${res.dequant[i]!.toFixed(4)} (q = ${res.q[i]})`}</title>
              </circle>
            ))}
          </svg>
          <p className="muted" style={{ fontSize: '0.8rem', margin: 0 }}>
            Perfect storage would lie on the diagonal. Each flat step is one integer level.
          </p>
        </figure>
      </div>

      <div className="viz-stats">
        <Stat k="Signal-to-noise (SQNR)" v={`${res.sqnr.toFixed(1)} dB`} />
        <Stat k="Mean squared error" v={res.mse.toExponential(2)} />
        <Stat k="Levels actually used" v={`${res.levelsUsed} / ${2 ** bits - 1}`} />
        <Stat k="Bits per weight incl. scales" v={res.bitsPerWeight.toFixed(2)} />
      </div>

      <div className="viz-ops">
        <p className="viz-label" style={{ margin: 0 }}>
          Weight memory at this bit-width
        </p>
        <ul className="viz-bars-h">
          {[
            { k: 'Llama 3 8B', p: 8.03e9 },
            { k: 'Llama 3 70B', p: 70.6e9 },
          ].flatMap((m) =>
            [
              { k: `${m.k} · fp16`, v: modelGB(m.p, 16) },
              { k: `${m.k} · int${bits}`, v: modelGB(m.p, res.bitsPerWeight) },
            ].map((row) => (
              <li key={row.k}>
                <span className="k">{row.k}</span>
                <span className="bar">
                  <span style={{ width: pct((row.v / modelGB(70.6e9, 16))), background: row.k.endsWith('fp16') ? 'var(--role-idle)' : 'var(--role-compare)' }} />
                </span>
                <span className="v">{row.v.toFixed(1)} GB</span>
              </li>
            )),
          )}
        </ul>
      </div>
    </div>
  );
}
