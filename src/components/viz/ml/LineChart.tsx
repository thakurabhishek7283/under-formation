import { useState, type PointerEvent } from 'react';

export interface Series {
  id: string;
  name: string;
  color: string;
  values: number[];
}

interface Props {
  series: Series[];
  logY?: boolean;
  xLabel: string;
  yLabel: string;
  height?: number;
  format?: (v: number) => string;
  /** Fix the x extent (e.g. max steps) so lines grow into a stable frame while animating. */
  xMax?: number;
}

const W = 900;
const PAD = { l: 76, r: 110, t: 12, b: 34 };

const defaultFmt = (v: number) =>
  v === 0 ? '0' : Math.abs(v) < 0.01 || Math.abs(v) >= 1e4 ? v.toExponential(1) : v.toPrecision(3).replace(/\.?0+$/, '');

/** Small SVG line chart with a hover crosshair, direct end labels, and a single y axis. */
export function LineChart({ series, logY = false, xLabel, yLabel, height = 220, format = defaultFmt, xMax }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const H = height;
  const n = Math.max(xMax ?? 0, ...series.map((s) => s.values.length), 2);
  const tf = (v: number) => (logY ? Math.log10(Math.max(v, 1e-12)) : v);
  const all = series.flatMap((s) => s.values.filter(Number.isFinite)).map(tf);
  let lo = all.length ? Math.min(...all) : 0;
  let hi = all.length ? Math.max(...all) : 1;
  if (!logY) lo = Math.min(lo, 0);
  if (hi - lo < 1e-9) hi = lo + 1;
  const x = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (tf(v) - lo) / (hi - lo)) * (H - PAD.t - PAD.b);

  const ticks = logY
    ? Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, k) => 10 ** (Math.ceil(lo) + k)).filter((_, k, a) => a.length <= 6 || k % 2 === 0)
    : Array.from({ length: 5 }, (_, k) => lo + ((hi - lo) * k) / 4);

  const onMove = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const i = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (n - 1));
    setHover(i >= 0 && i < n ? i : null);
  };

  // Direct labels at line ends, nudged apart so they don't collide.
  const ends = series
    .filter((s) => s.values.length)
    .map((s) => ({ s, y: y(s.values[s.values.length - 1]!) }))
    .sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) ends[k]!.y = Math.max(ends[k]!.y, ends[k - 1]!.y + 14);

  return (
    <div className="viz-line">
      {series.length > 1 && (
        <div className="viz-legend">
          {series.map((s) => (
            <span key={s.id}>
              <span className="sw" style={{ background: s.color, height: 3, borderRadius: 2 }} /> {s.name}
            </span>
          ))}
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="viz-line-svg" onPointerMove={onMove} onPointerLeave={() => setHover(null)} role="img" aria-label={`${yLabel} vs ${xLabel}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={PAD.l - 6} y={y(t) + 4} className="tick" textAnchor="end">
              {logY ? logTick(t) : format(t)}
            </text>
          </g>
        ))}
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} className="axis" />
        <text x={(PAD.l + W - PAD.r) / 2} y={H - 6} className="tick" textAnchor="middle">
          {xLabel}
        </text>
        <text x={14} y={H / 2} className="tick" textAnchor="middle" transform={`rotate(-90 14 ${H / 2})`}>
          {yLabel}
        </text>

        {series.map((s) => (
          <polyline
            key={s.id}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeLinejoin="round"
            points={s.values.map((v, i) => (Number.isFinite(v) ? `${x(i).toFixed(1)},${y(v).toFixed(1)}` : '')).join(' ')}
          />
        ))}
        {ends.map(({ s, y: ly }) => (
          <text key={s.id} x={W - PAD.r + 6} y={ly + 4} className="end-label">
            {s.name}
          </text>
        ))}

        {hover !== null && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} className="crosshair" />
            {series.map((s) =>
              Number.isFinite(s.values[hover]) ? (
                <circle key={s.id} cx={x(hover)} cy={y(s.values[hover]!)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
              ) : null,
            )}
          </g>
        )}
      </svg>
      <div className="viz-line-tip" aria-live="off">
        {hover !== null ? (
          <>
            <strong>
              {xLabel} {hover}
            </strong>
            {series.map((s) =>
              Number.isFinite(s.values[hover]) ? (
                <span key={s.id}>
                  <span className="sw" style={{ background: s.color }} /> {s.name}: {format(s.values[hover]!)}
                </span>
              ) : null,
            )}
          </>
        ) : (
          <span className="muted">Hover the chart to read values.</span>
        )}
      </div>
    </div>
  );
}

/** 1, 10, 100, 1000 as plain numbers; otherwise 1e−k style. */
function logTick(t: number): string {
  const k = Math.round(Math.log10(t));
  return k >= 0 && k <= 3 ? String(10 ** k) : `1e${k}`;
}
