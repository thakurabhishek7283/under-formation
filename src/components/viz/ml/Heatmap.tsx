import type { CSSProperties, ReactNode } from 'react';

export type HeatScale = 'sequential' | 'diverging';

/** Color for a value: one-hue ramp for magnitudes, red–gray–blue for signed values. */
export function heatColor(v: number, scale: HeatScale, domain: number): { background: string; strong: boolean } {
  if (scale === 'sequential') {
    const t = Math.max(0, Math.min(1, v / (domain || 1)));
    return { background: `color-mix(in oklab, var(--seq-7) ${(t * 100).toFixed(1)}%, var(--seq-0))`, strong: t > 0.55 };
  }
  const t = Math.max(-1, Math.min(1, v / (domain || 1)));
  const pole = t < 0 ? 'var(--div-neg)' : 'var(--div-pos)';
  return { background: `color-mix(in oklab, ${pole} ${(Math.abs(t) * 100).toFixed(1)}%, var(--div-mid))`, strong: Math.abs(t) > 0.6 };
}

export const fmt2 = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

interface Props {
  data: number[][];
  scale: HeatScale;
  /** Max |value| (diverging) or max value (sequential). Defaults to the data's own extent. */
  domain?: number;
  rowLabels?: string[];
  colLabels?: string[];
  showValues?: boolean;
  format?: (v: number) => string;
  /** CSS size of a cell. */
  cell?: string;
  highlightRow?: number | null;
  masked?: (i: number, j: number) => boolean;
  /** Tooltip for masked cells. Masking means "no value here", which is a future token in
   *  attention but an uncomputed cell elsewhere, so callers can say which. */
  maskedTip?: string;
  onRowClick?: (i: number) => void;
  tip?: (i: number, j: number, v: number) => string;
  cellStyle?: (i: number, j: number) => CSSProperties | undefined;
  label: string;
}

export function Heatmap({
  data,
  scale,
  domain,
  rowLabels,
  colLabels,
  showValues = false,
  format = fmt2,
  cell = '1.6rem',
  highlightRow = null,
  masked,
  maskedTip = 'masked (future token)',
  onRowClick,
  tip,
  cellStyle,
  label,
}: Props) {
  const cols = data[0]?.length ?? 0;
  const extent =
    domain ??
    Math.max(
      1e-9,
      ...data.flatMap((row, i) => row.filter((_, j) => !masked?.(i, j)).map((v) => (scale === 'diverging' ? Math.abs(v) : v))),
    );
  const template = `${rowLabels ? 'max-content ' : ''}repeat(${cols}, ${cell})`;

  return (
    <div className="viz-heat" role="img" aria-label={label} style={{ gridTemplateColumns: template }}>
      {colLabels && (
        <>
          {rowLabels && <span />}
          {colLabels.map((c, j) => (
            <span key={`c${j}`} className="viz-heat-col" title={c}>
              {c}
            </span>
          ))}
        </>
      )}
      {data.map((row, i) => (
        <HeatRow
          key={i}
          i={i}
          row={row}
          rowLabel={rowLabels?.[i]}
          selected={highlightRow === i}
          onClick={onRowClick ? () => onRowClick(i) : undefined}
          render={(v, j) => {
            const isMasked = masked?.(i, j) ?? false;
            const { background, strong } = heatColor(v, scale, extent);
            return (
              <span
                key={j}
                className={`viz-heat-cell${isMasked ? ' masked' : ''}`}
                data-tip={isMasked ? maskedTip : (tip?.(i, j, v) ?? format(v))}
                tabIndex={-1}
                style={{
                  ...(isMasked ? {} : { background, color: strong ? 'var(--heat-ink-strong)' : 'var(--text)' }),
                  ...cellStyle?.(i, j),
                }}
              >
                {showValues && !isMasked ? format(v) : ''}
              </span>
            );
          }}
        />
      ))}
    </div>
  );
}

function HeatRow({
  i,
  row,
  rowLabel,
  selected,
  onClick,
  render,
}: {
  i: number;
  row: number[];
  rowLabel?: string;
  selected: boolean;
  onClick?: () => void;
  render: (v: number, j: number) => ReactNode;
}) {
  return (
    <>
      {rowLabel !== undefined &&
        (onClick ? (
          <button type="button" className={`viz-heat-row${selected ? ' selected' : ''}`} onClick={onClick} aria-pressed={selected}>
            {rowLabel}
          </button>
        ) : (
          <span className={`viz-heat-row${selected ? ' selected' : ''}`} data-row={i}>
            {rowLabel}
          </span>
        ))}
      {row.map((v, j) => render(v, j))}
    </>
  );
}

/** Gradient key for a heatmap scale. */
export function ScaleLegend({
  scale,
  domain,
  format = fmt2,
  low,
  high,
}: {
  scale: HeatScale;
  domain: number;
  format?: (v: number) => string;
  low?: string;
  high?: string;
}) {
  const gradient =
    scale === 'sequential'
      ? 'linear-gradient(90deg, var(--seq-0), var(--seq-7))'
      : 'linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos))';
  return (
    <div className="viz-scale">
      <span>{low ?? (scale === 'sequential' ? format(0) : `−${format(domain)}`)}</span>
      <span className="viz-scale-bar" style={{ background: gradient }} />
      <span>{high ?? format(domain)}</span>
    </div>
  );
}
