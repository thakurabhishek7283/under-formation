import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { OPTIMIZERS, SURFACES, optimize, type OptimizerName, type Vec2 } from '../../../lib/ml/optimizers';
import { LineChart } from './LineChart';
import { cssColor, useThemeTick } from './useThemeTick';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const COLORS: Record<OptimizerName, string> = {
  sgd: 'var(--role-compare)',
  momentum: 'var(--role-change)',
  adam: 'var(--role-done)',
};
const PW = 480;
const PH = 360;
const STEPS = 400;
const BANDS = 14;
const LR_SCALES = [0.25, 0.5, 1, 1.5, 2, 3];

export default function OptimizerViz() {
  const [surfaceIdx, setSurfaceIdx] = useState(0);
  const surface = SURFACES[surfaceIdx]!;
  const [start, setStart] = useState<Vec2>(surface.start);
  const [lrIdx, setLrIdx] = useState(2);
  const [t, setT] = useState(0);
  const [running, setRunning] = useState(false);
  const canvas = useRef<HTMLCanvasElement>(null);
  const themeTick = useThemeTick();

  const { x: [x0, x1], y: [y0, y1] } = surface.domain;
  const toPx = ([x, y]: Vec2): Vec2 => [r1(((x - x0) / (x1 - x0)) * PW), r1((1 - (y - y0) / (y1 - y0)) * PH)];

  const runs = useMemo(
    () => OPTIMIZERS.map((o) => ({ ...o, traj: optimize(surface, o.id, start, LR_SCALES[lrIdx]!, STEPS) })),
    [surface, start, lrIdx],
  );

  // Draw the loss surface as log-spaced gray bands.
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = PW * dpr;
    c.height = PH * dpr;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(c.width, c.height);
    const lo = cssColor('--contour-lo');
    const hi = cssColor('--contour-hi');
    const vals = new Float64Array(c.width * c.height);
    let mn = Infinity;
    let mx = -Infinity;
    for (let py = 0; py < c.height; py++)
      for (let px = 0; px < c.width; px++) {
        const x = x0 + (px / c.width) * (x1 - x0);
        const y = y1 - (py / c.height) * (y1 - y0);
        const v = Math.log(surface.f(x, y) + 1e-3);
        vals[py * c.width + px] = v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    for (let i = 0; i < vals.length; i++) {
      const band = Math.min(BANDS - 1, Math.floor(((vals[i]! - mn) / (mx - mn)) * BANDS));
      const k = band / (BANDS - 1);
      img.data[i * 4] = lo[0] + (hi[0] - lo[0]) * k;
      img.data[i * 4 + 1] = lo[1] + (hi[1] - lo[1]) * k;
      img.data[i * 4 + 2] = lo[2] + (hi[2] - lo[2]) * k;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [surface, themeTick, x0, x1, y0, y1]);

  // Animate.
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const adv = Math.floor((now - last) / 25);
      if (adv > 0) {
        last = now;
        setT((cur) => {
          const next = Math.min(cur + adv, STEPS);
          if (next >= STEPS) setRunning(false);
          return next;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  const restart = () => {
    setT(0);
    setRunning(true);
  };
  const pickSurface = (i: number) => {
    setSurfaceIdx(i);
    setStart(SURFACES[i]!.start);
    setT(0);
    setRunning(false);
  };
  const onPlotClick = (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = x0 + ((e.clientX - r.left) / r.width) * (x1 - x0);
    const y = y1 - ((e.clientY - r.top) / r.height) * (y1 - y0);
    setStart([x, y]);
    setT(0);
    setRunning(true);
  };

  return (
    <div className="viz" aria-label="Gradient descent optimizers visualization">
      <div className="viz-row">
        <div className="viz-seg" role="radiogroup" aria-label="Loss surface">
          {SURFACES.map((s, i) => (
            <button key={s.id} type="button" role="radio" aria-checked={i === surfaceIdx} className={`btn${i === surfaceIdx ? ' on' : ''}`} onClick={() => pickSurface(i)}>
              {s.name}
            </button>
          ))}
        </div>
        <label className="viz-slider">
          <span>
            Learning rate <output>{LR_SCALES[lrIdx]}×</output> tuned default
          </span>
          <input type="range" min={0} max={LR_SCALES.length - 1} value={lrIdx} onChange={(e) => { setLrIdx(+e.target.value); setT(0); }} />
        </label>
      </div>
      <p className="viz-caption" style={{ minHeight: 0 }}>
        <span>{surface.blurb} Click anywhere on the surface to start all three from there.</span>
      </p>

      <div className="viz-opt">
        <div className="viz-opt-plot">
          <canvas ref={canvas} style={{ width: '100%', height: '100%' }} aria-hidden="true" />
          <svg viewBox={`0 0 ${PW} ${PH}`} onClick={onPlotClick} role="img" aria-label={`${surface.name} loss surface with optimizer paths`}>
            {surface.minima.map((m, i) => {
              const [px, py] = toPx(m);
              return (
                <g key={i} className="viz-opt-min">
                  <path d={`M${px - 6},${py - 6} L${px + 6},${py + 6} M${px - 6},${py + 6} L${px + 6},${py - 6}`} />
                </g>
              );
            })}
            {runs.map((r) => {
              const pts = r.traj.path.slice(0, t + 1).map(toPx);
              const d = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
              const cur = pts[pts.length - 1]!;
              return (
                <g key={r.id}>
                  <polyline points={d} fill="none" stroke="var(--surface)" strokeWidth={5} strokeLinejoin="round" opacity={0.8} />
                  <polyline points={d} fill="none" stroke={COLORS[r.id]} strokeWidth={2} strokeLinejoin="round" />
                  <circle cx={cur[0]} cy={cur[1]} r={6} fill={COLORS[r.id]} stroke="var(--surface)" strokeWidth={2}>
                    <title>{`${r.name} at step ${Math.min(t, r.traj.path.length - 1)}`}</title>
                  </circle>
                </g>
              );
            })}
            {(() => {
              const [sx, sy] = toPx(start);
              return <circle cx={sx} cy={sy} r={4} fill="var(--text)" />;
            })()}
          </svg>
        </div>

        <div className="viz-opt-side">
          <div className="viz-row">
            <button type="button" className="btn primary" onClick={() => (t >= STEPS ? restart() : setRunning((r) => !r))}>
              {running ? '❚❚ Pause' : t >= STEPS ? '↻ Replay' : t > 0 ? '▶ Resume' : '▶ Run'}
            </button>
            <button type="button" className="btn" onClick={() => { setT(0); setRunning(false); }}>
              Reset
            </button>
            <span className="muted mono" style={{ fontSize: '0.85rem' }}>
              step {t}/{STEPS}
            </span>
          </div>
          <ul className="viz-opt-list">
            {runs.map((r) => {
              const i = Math.min(t, r.traj.losses.length - 1);
              const diverged = r.traj.diverged && t >= r.traj.losses.length - 1;
              return (
                <li key={r.id}>
                  <span className="sw" style={{ background: COLORS[r.id] }} />
                  <div>
                    <strong>{r.name}</strong> <code>{r.rule}</code>
                    <div className="muted mono">
                      {diverged ? `diverged at step ${r.traj.losses.length}` : `loss ${r.traj.losses[i]!.toExponential(2)}`}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      </div>

      <LineChart
        series={runs.map((r) => ({ id: r.id, name: r.name, color: COLORS[r.id], values: r.traj.losses.slice(0, t + 1) }))}
        logY
        xMax={STEPS + 1}
        xLabel="step"
        yLabel="loss (log)"
        height={200}
      />
    </div>
  );
}
