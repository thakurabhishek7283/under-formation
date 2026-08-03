import { useMemo, useState } from 'react';
import {
  POS_MODES,
  encodingTable,
  frequencies,
  offsetCurves,
  similarityMatrix,
  wavelengths,
  type PosMode,
} from '../../../lib/ml/positional';
import { Stat } from '../Stat';
import { Heatmap, ScaleLegend } from './Heatmap';
import { LineChart } from './LineChart';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const BASES = [0, 6, 14];
const SERIES = ['var(--cat-1)', 'var(--cat-2)', 'var(--cat-3)'];

export default function PositionalViz() {
  const [mode, setMode] = useState<PosMode>('sinusoidal');
  const [positions, setPositions] = useState(24);
  const [dim, setDim] = useState(16);
  const [baseExp, setBaseExp] = useState(4); // base = 10^baseExp
  const [probe, setProbe] = useState(5);

  const base = 10 ** baseExp;
  const table = useMemo(() => encodingTable(mode, positions, dim, base), [mode, positions, dim, base]);
  const sim = useMemo(() => similarityMatrix(mode, positions, dim, base), [mode, positions, dim, base]);
  const curves = useMemo(() => offsetCurves(mode, positions, dim, base, BASES), [mode, positions, dim, base]);
  const theta = useMemo(() => frequencies(dim, base), [dim, base]);
  const waves = useMemo(() => wavelengths(dim, base), [dim, base]);

  const info = POS_MODES.find((m) => m.id === mode)!;
  const simDomain = Math.max(0.05, ...sim.flat().map(Math.abs));
  const tableDomain = Math.max(0.05, ...table.flat().map(Math.abs));
  const posLabels = Array.from({ length: positions }, (_, i) => String(i));

  return (
    <div className="viz" aria-label="Positional encoding visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Scheme</span>
          <select className="input" value={mode} onChange={(e) => setMode(e.target.value as PosMode)}>
            {POS_MODES.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Positions <output>{positions}</output>
          </span>
          <input type="range" min={8} max={40} value={positions} onChange={(e) => setPositions(+e.target.value)} />
        </label>
        <label className="viz-slider">
          <span>
            Dimensions <output>{dim}</output>
          </span>
          <input type="range" min={4} max={32} step={2} value={dim} onChange={(e) => setDim(+e.target.value)} />
        </label>
        <label className="viz-slider">
          <span>
            Base θ <output>10^{baseExp}</output>
          </span>
          <input type="range" min={2} max={6} value={baseExp} disabled={mode === 'learned'} onChange={(e) => setBaseExp(+e.target.value)} />
        </label>
      </div>

      <p className="viz-note">{info.blurb}</p>

      <div className="viz-pair">
        <figure>
          <figcaption className="viz-label">
            {mode === 'rope' ? 'Rotation angle per dimension pair (cos)' : 'The encoding itself — one row per position'}
          </figcaption>
          <Heatmap
            data={table}
            scale="diverging"
            domain={tableDomain}
            cell="0.72rem"
            rowLabels={posLabels}
            label="positional encoding table"
            tip={(p, d, v) => `position ${p}, ${mode === 'rope' ? `pair ${d}` : `dim ${d}`} = ${v.toFixed(3)}`}
          />
          <ScaleLegend scale="diverging" domain={tableDomain} />
          <p className="viz-note">Left columns turn fast, right columns barely move — a positional binary code with continuous digits.</p>
        </figure>

        <figure>
          <figcaption className="viz-label">Similarity between every pair of positions</figcaption>
          <Heatmap
            data={sim}
            scale="diverging"
            domain={simDomain}
            cell="0.72rem"
            rowLabels={posLabels}
            highlightRow={probe}
            onRowClick={setProbe}
            label="position-to-position similarity"
            tip={(m, n, v) => `position ${m} · position ${n} = ${v.toFixed(3)}  (offset ${n - m})`}
          />
          <ScaleLegend scale="diverging" domain={simDomain} />
          <p className="viz-note">
            {mode === 'rope'
              ? 'Constant along every diagonal: the score depends on the gap between positions, not on where they are.'
              : mode === 'sinusoidal'
                ? 'Brightest on the diagonal and fading outwards — but the pattern differs from row to row.'
                : 'No structure at all. Nothing about position 5 makes it resemble position 6.'}
          </p>
        </figure>
      </div>

      {/* The rotation, drawn. One wheel per frequency pair, all at the probe position. */}
      <div>
        <p className="viz-label">Position {probe} as rotations — each wheel is one dimension pair, turning at its own speed</p>
        <div className="viz-wheels">
          {theta.slice(0, 8).map((t, i) => {
            const ang = probe * t;
            return (
              <figure key={i} title={`pair ${i}: θ = ${t.toExponential(2)}, wavelength ≈ ${waves[i]!.toFixed(1)} tokens`}>
                <svg
                  viewBox="-12 -12 24 24"
                  width="100%"
                  role="img"
                  aria-label={`dimension pair ${i} rotated by ${(ang * 57.3).toFixed(0)} degrees`}
                >
                  <circle cx={0} cy={0} r={10} style={{ fill: 'var(--surface-2)', stroke: 'var(--border)', strokeWidth: 0.7 }} />
                  <line x1={0} y1={0} x2={10} y2={0} style={{ stroke: 'var(--border)', strokeWidth: 0.7 }} />
                  <line
                    x1={0}
                    y1={0}
                    x2={r1(Math.cos(ang) * 9)}
                    y2={r1(-Math.sin(ang) * 9)}
                    style={{ stroke: 'var(--cat-1)', strokeWidth: 1.6, strokeLinecap: 'round' }}
                  />
                  <circle cx={r1(Math.cos(ang) * 9)} cy={r1(-Math.sin(ang) * 9)} r={1.6} style={{ fill: 'var(--cat-1)' }} />
                </svg>
                <figcaption>
                  {i} · λ≈{waves[i]! > 999 ? `${(waves[i]! / 1000).toFixed(0)}k` : waves[i]!.toFixed(0)}
                </figcaption>
              </figure>
            );
          })}
        </div>
        <label className="viz-slider">
          <span>
            Probe position <output>{probe}</output>
          </span>
          <input
            type="range"
            min={0}
            max={positions - 1}
            value={Math.min(probe, positions - 1)}
            onChange={(e) => setProbe(+e.target.value)}
          />
        </label>
      </div>

      <figure style={{ margin: 0 }}>
        <figcaption className="viz-label">Similarity vs offset, measured from three different starting positions</figcaption>
        <LineChart
          series={curves.map((values, i) => ({
            id: `from${BASES[i]}`,
            name: `from position ${BASES[i]}`,
            color: SERIES[i]!,
            values,
          }))}
          xLabel="offset between the two positions"
          yLabel="similarity"
          height={200}
          format={(v) => v.toFixed(2)}
        />
        <p className="viz-note">
          {mode === 'rope'
            ? 'All three lines sit exactly on top of each other — that is what "relative" means, made literal.'
            : 'The lines separate, so the same gap scores differently depending on where in the sequence it occurs.'}
        </p>
      </figure>

      <div className="viz-stats">
        <Stat k="Fastest pair repeats every" v={`${waves[0]!.toFixed(1)} tokens`} />
        <Stat
          k="Slowest pair repeats every"
          v={`${waves[waves.length - 1]! > 9999 ? `${(waves[waves.length - 1]! / 1000).toFixed(0)}k` : waves[waves.length - 1]!.toFixed(0)} tokens`}
        />
        <Stat k="Dimension pairs" v={String(theta.length)} />
        <Stat k="Extra parameters" v={mode === 'learned' ? `${positions * dim} trained` : '0 — it is a formula'} />
      </div>
    </div>
  );
}
