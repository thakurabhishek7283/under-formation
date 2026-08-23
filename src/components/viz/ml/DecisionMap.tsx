import { useEffect, useRef, useState, type PointerEvent, type ReactNode } from 'react';
import { boundarySegments, contourSegments, type Field, type LPoint } from '../../../lib/ml/classify';
import { cssColor, useThemeTick } from './useThemeTick';
import { r1 } from './num';

export const CLASS_COLOR = ['var(--cat-1)', 'var(--cat-2)', 'var(--cat-3)'];
export const CLASS_NAME = ['A', 'B', 'C'];
const CLASS_VAR = ['--cat-1', '--cat-2', '--cat-3'];

/** Plot y points up; SVG y points down. */
export const sy = (y: number) => r1(100 - y);

/** True after hydration. Gate client-only work (sampling a model over the plane) on it. */
export function useClient(): boolean {
  const [on, setOn] = useState(false);
  useEffect(() => setOn(true), []);
  return on;
}

/**
 * One sample, drawn with a shape per class (circle, square, triangle) so classes stay
 * distinguishable without colour. Held-out samples are hollow.
 */
export function ClassMarker({
  x,
  y,
  label,
  r = 1.5,
  hollow = false,
  dim = false,
  ring,
  title,
}: {
  x: number;
  y: number;
  label: number;
  r?: number;
  hollow?: boolean;
  dim?: boolean;
  /** Outline colour for emphasis (e.g. a neighbour, a support vector). */
  ring?: string;
  title?: string;
}) {
  const color = CLASS_COLOR[label] ?? 'var(--role-idle)';
  const style = {
    fill: hollow ? 'var(--surface)' : color,
    stroke: hollow ? color : 'var(--surface)',
    strokeWidth: hollow ? 0.55 : 0.35,
    opacity: dim ? 0.3 : 1,
  };
  const cx = r1(x);
  const cy = sy(y);
  const shape =
    label === 1 ? (
      <rect x={r1(cx - r * 0.88)} y={r1(cy - r * 0.88)} width={r1(r * 1.76)} height={r1(r * 1.76)} style={style} />
    ) : label === 2 ? (
      <polygon
        points={`${cx},${r1(cy - r * 1.15)} ${r1(cx + r * 1.05)},${r1(cy + r * 0.75)} ${r1(cx - r * 1.05)},${r1(cy + r * 0.75)}`}
        style={style}
      />
    ) : (
      <circle cx={cx} cy={cy} r={r} style={style} />
    );
  return (
    <g>
      {ring && <circle cx={cx} cy={cy} r={r1(r + 1.1)} style={{ fill: 'none', stroke: ring, strokeWidth: 0.6 }} />}
      {shape}
      {title && <title>{title}</title>}
    </g>
  );
}

/** Legend entries that show the same shape as the plot. */
export function ClassLegend({ classes, extra, held = true }: { classes: number; extra?: ReactNode; held?: boolean }) {
  return (
    <div className="viz-legend">
      {Array.from({ length: classes }, (_, c) => (
        <span key={c}>
          <svg viewBox="0 0 6 6" width="12" height="12" style={{ verticalAlign: 'middle', overflow: 'visible' }} aria-hidden="true">
            <g transform="translate(-47 -97)">
              <ClassMarker x={50} y={0} label={c} r={2.3} />
            </g>
          </svg>{' '}
          class {CLASS_NAME[c]}
        </span>
      ))}
      {held && (
        <span>
          <svg viewBox="0 0 6 6" width="12" height="12" style={{ verticalAlign: 'middle', overflow: 'visible' }} aria-hidden="true">
            <circle cx={3} cy={3} r={2.2} style={{ fill: 'var(--surface)', stroke: 'var(--text-3)', strokeWidth: 0.9 }} />
          </svg>{' '}
          hollow = held out
        </span>
      )}
      {extra}
    </div>
  );
}

export interface ContourLine {
  values: Float32Array;
  level: number;
  dashed?: boolean;
}

interface Props {
  /** What the model predicts everywhere. Null while it hasn't been computed (e.g. before hydration). */
  field: Field | null;
  points: LPoint[];
  test?: LPoint[];
  /** Draw the line where the predicted class changes. */
  boundary?: boolean;
  /** Extra level sets of a scalar field on the same grid, e.g. an SVM's margins. */
  lines?: ContourLine[];
  /** SVG drawn under the points, in plot units (use `sy` for y). */
  under?: ReactNode;
  /** SVG drawn over the points. */
  children?: ReactNode;
  marker?: (p: LPoint, i: number, held: boolean) => { r?: number; ring?: string; dim?: boolean } | undefined;
  onPlot?: (kind: 'down' | 'move' | 'up', at: { x: number; y: number }, e: PointerEvent<SVGSVGElement>) => void;
  label: string;
  cursor?: string;
}

/**
 * A classifier's view of the plane. The background is the predicted class, tinted more strongly
 * where the model is more confident; the dark line is the decision boundary.
 */
export function DecisionMap({ field, points, test = [], boundary = true, lines, under, children, marker, onPlot, label, cursor }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const theme = useThemeTick();
  const [px, setPx] = useState(0);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setPx(Math.round(el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !px) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = Math.round(px * dpr);
    if (c.width !== size) {
      c.width = size;
      c.height = size;
    }
    const ctx = c.getContext('2d')!;
    const bg = cssColor('--surface');
    ctx.fillStyle = `rgb(${bg.join(',')})`;
    ctx.fillRect(0, 0, size, size);
    if (!field) return;

    const { res, classes, p } = field;
    const cols = CLASS_VAR.slice(0, classes).map(cssColor);
    const off = document.createElement('canvas');
    off.width = res;
    off.height = res;
    const octx = off.getContext('2d')!;
    const img = octx.createImageData(res, res);
    for (let k = 0; k < res * res; k++) {
      let best = 0;
      for (let cl = 1; cl < classes; cl++) if (p[k * classes + cl]! > p[k * classes + best]!) best = cl;
      // Confidence: 0 at a uniform guess, 1 at certainty.
      const conf = Math.max(0, Math.min(1, (p[k * classes + best]! - 1 / classes) / (1 - 1 / classes)));
      const a = 0.08 + 0.34 * conf;
      const col = cols[best]!;
      img.data[k * 4] = bg[0] + (col[0] - bg[0]) * a;
      img.data[k * 4 + 1] = bg[1] + (col[1] - bg[1]) * a;
      img.data[k * 4 + 2] = bg[2] + (col[2] - bg[2]) * a;
      img.data[k * 4 + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, size, size);

    const ink = cssColor('--text');
    const k = size / 100;
    const stroke = (segs: [number, number, number, number][], width: number, alpha: number, dash: number[] = []) => {
      ctx.beginPath();
      for (const [x1, y1, x2, y2] of segs) {
        ctx.moveTo(x1 * k, y1 * k);
        ctx.lineTo(x2 * k, y2 * k);
      }
      ctx.setLineDash(dash.map((d) => d * dpr));
      ctx.lineWidth = width * dpr;
      ctx.lineCap = 'round';
      ctx.strokeStyle = `rgba(${ink.join(',')},${alpha})`;
      ctx.stroke();
    };
    for (const l of lines ?? []) stroke(contourSegments(l.values, res, l.level), 1.2, 0.55, l.dashed ? [5, 4] : []);
    if (boundary) stroke(boundarySegments(field), 1.8, 0.8);
  }, [field, lines, boundary, px, theme]);

  const at = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100)),
      y: Math.max(0, Math.min(100, 100 - ((e.clientY - r.top) / r.height) * 100)),
    };
  };

  return (
    <div ref={wrap} className="viz-map">
      <canvas ref={canvas} aria-hidden="true" />
      <svg
        viewBox="0 0 100 100"
        role="img"
        aria-label={label}
        style={{ cursor: cursor ?? (onPlot ? 'crosshair' : undefined) }}
        onPointerDown={
          onPlot &&
          ((e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            onPlot('down', at(e), e);
          })
        }
        onPointerMove={onPlot && ((e) => onPlot('move', at(e), e))}
        onPointerUp={onPlot && ((e) => onPlot('up', at(e), e))}
      >
        {under}
        {points.map((p, i) => (
          <ClassMarker key={`p${i}`} x={p.x} y={p.y} label={p.label} {...marker?.(p, i, false)} />
        ))}
        {test.map((p, i) => (
          <ClassMarker key={`t${i}`} x={p.x} y={p.y} label={p.label} hollow {...marker?.(p, i, true)} />
        ))}
        {children}
      </svg>
    </div>
  );
}
