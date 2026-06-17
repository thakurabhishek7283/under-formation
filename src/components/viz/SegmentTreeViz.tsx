import { useEffect, useMemo, useState } from 'react';
import {
  AGGREGATES,
  BUILD_CODE,
  QUERY_CODE,
  SegmentTree,
  UPDATE_CODE,
  type Aggregate,
  type SegRole,
  type SegStep,
} from '../../lib/algorithms/segmentTree';
import { usePlayer } from './usePlayer';
import { PlayerControls, Swatch } from './PlayerControls';
import { CodePanel } from './CodePanel';
import { Stat } from './Stat';
import './viz.css';

const CELL = 64;
const NODE_W = 54;
const NODE_H = 40;
const LEVEL_H = 70;

const ROLE_COLOR: Record<SegRole, string> = {
  partial: 'var(--role-compare)',
  full: 'var(--role-done)',
  changed: 'var(--role-change)',
  outside: 'var(--role-idle)',
};

const fmt = (v: number | null | undefined) =>
  v === null || v === undefined ? '?' : v === Infinity ? '∞' : v === -Infinity ? '−∞' : String(v);

const CODE_TITLE: Record<string, string> = {
  build: 'Build · O(n), once',
  query: 'Query · O(log n)',
  update: 'Update · O(log n)',
};

const initialValues = (n: number) => Array.from({ length: n }, (_, i) => ((i * 7 + 3) % 10) + 1);
const clampIdx = (v: number, n: number) => Math.max(0, Math.min(n - 1, Math.trunc(v) || 0));

export default function SegmentTreeViz({ initialSize = 8 }: { initialSize?: number }) {
  const [aggregate, setAggregate] = useState<Aggregate>('sum');
  const [values, setValues] = useState(() => initialValues(initialSize));
  const [ql, setQl] = useState(2);
  const [qr, setQr] = useState(5);
  const [ui, setUi] = useState(3);
  const [uv, setUv] = useState(10);

  // Updates mutate this instance; each step carries its own snapshot, so rendering stays consistent.
  const tree = useMemo(() => new SegmentTree(values, aggregate), [values, aggregate]);
  const player = usePlayer<SegStep>(tree.buildTrace, 3, true);
  useEffect(() => player.load(tree.buildTrace, false, true), [tree]); // eslint-disable-line react-hooks/exhaustive-deps

  const n = tree.n;
  const step = player.step;
  const label = AGGREGATES[aggregate].label;

  const runQuery = () => {
    const l = clampIdx(Math.min(ql, qr), n);
    const r = clampIdx(Math.max(ql, qr), n);
    player.load(tree.query(l, r).steps, true);
  };
  const runUpdate = () => player.load(tree.update(clampIdx(ui, n), uv), true);

  const W = n * CELL;
  const treeH = (tree.depth + 1) * LEVEL_H;
  const H = treeH + 44;
  const pos = new Map(tree.nodes.map((nd) => [nd.id, { x: ((nd.l + nd.r + 1) / 2) * CELL, y: nd.depth * LEVEL_H + 6 }]));
  const leafOf = new Map(tree.nodes.filter((nd) => nd.l === nd.r).map((nd) => [nd.l, nd.id]));
  const visited = Object.keys(step.roles).length;

  return (
    <div className="viz" aria-label="Segment tree visualization">
      <div className="viz-row">
        <label className="viz-field">
          <span>Each node stores the</span>
          <select className="input" value={aggregate} onChange={(e) => setAggregate(e.target.value as Aggregate)}>
            <option value="sum">sum</option>
            <option value="min">min</option>
            <option value="max">max</option>
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Array size <output>{n}</output>
          </span>
          <input type="range" min={2} max={16} value={n} onChange={(e) => setValues(initialValues(+e.target.value))} />
        </label>
        <button type="button" className="btn" onClick={() => setValues(values.map(() => 1 + Math.floor(Math.random() * 20)))}>
          Random values
        </button>
        <button type="button" className="btn" onClick={() => player.load(new SegmentTree(tree.array(), aggregate).buildTrace, true)}>
          Replay build
        </button>
      </div>

      <div className="viz-ops">
        <div className="viz-row">
          <span className="viz-op-label">Query</span>
          <code>
            {label}(a[
            <NumInput value={ql} onChange={setQl} min={0} max={n - 1} label="Query start" />
            ..
            <NumInput value={qr} onChange={setQr} min={0} max={n - 1} label="Query end" />
            ])
          </code>
          <button type="button" className="btn primary" onClick={runQuery}>
            Run query
          </button>
        </div>
        <div className="viz-row">
          <span className="viz-op-label">Update</span>
          <code>
            a[
            <NumInput value={ui} onChange={setUi} min={0} max={n - 1} label="Index to update" />] =
            <NumInput value={uv} onChange={setUv} min={-99} max={99} label="New value" />
          </code>
          <button type="button" className="btn primary" onClick={runUpdate}>
            Run update
          </button>
        </div>
      </div>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-split">
        <div>
          <div className="viz-scroll">
            <svg
              className="viz-svg"
              viewBox={`0 0 ${W} ${H}`}
              style={{ minWidth: n * 40, maxWidth: W * 1.1 }}
              role="img"
              aria-label={`Segment tree over ${n} values`}
            >
              {/* Query / update target band */}
              {step.target && (
                <rect
                  x={step.target[0] * CELL + 2}
                  y={treeH}
                  width={(step.target[1] - step.target[0] + 1) * CELL - 4}
                  height={38}
                  rx={6}
                  style={{ fill: 'color-mix(in srgb, var(--role-compare) 14%, transparent)' }}
                />
              )}

              {/* Edges */}
              {tree.nodes.map((nd) => {
                if (nd.id === 1) return null;
                const p = pos.get(nd.id >> 1)!;
                const c = pos.get(nd.id)!;
                const role = step.roles[nd.id];
                return (
                  <line
                    key={`e${nd.id}`}
                    x1={p.x}
                    y1={p.y + NODE_H}
                    x2={c.x}
                    y2={c.y}
                    style={{
                      stroke: role && role !== 'outside' ? ROLE_COLOR[role] : 'var(--border)',
                      strokeWidth: role && role !== 'outside' ? 2 : 1.5,
                    }}
                  />
                );
              })}

              {/* Leaf-to-array guides */}
              {tree.nodes
                .filter((nd) => nd.l === nd.r)
                .map((nd) => {
                  const c = pos.get(nd.id)!;
                  return (
                    <line
                      key={`g${nd.id}`}
                      x1={c.x}
                      y1={c.y + NODE_H}
                      x2={c.x}
                      y2={treeH + 4}
                      style={{ stroke: 'var(--border)', strokeDasharray: '2 4' }}
                    />
                  );
                })}

              {/* Nodes */}
              {tree.nodes.map((nd) => {
                const { x, y } = pos.get(nd.id)!;
                const role = step.roles[nd.id];
                const v = step.values[nd.id];
                const isCurrent = step.current === nd.id;
                const color = role ? ROLE_COLOR[role] : 'var(--border)';
                return (
                  <g key={nd.id} style={{ opacity: role === 'outside' ? 0.45 : 1 }}>
                    <title>{`node ${nd.id}: ${label} of a[${nd.l}..${nd.r}] = ${fmt(v)}`}</title>
                    {isCurrent && (
                      <rect
                        x={x - NODE_W / 2 - 4}
                        y={y - 4}
                        width={NODE_W + 8}
                        height={NODE_H + 8}
                        rx={10}
                        style={{ fill: 'none', stroke: 'var(--text)', strokeWidth: 2 }}
                      />
                    )}
                    <rect
                      x={x - NODE_W / 2}
                      y={y}
                      width={NODE_W}
                      height={NODE_H}
                      rx={7}
                      style={{
                        fill: role && role !== 'outside' ? `color-mix(in srgb, ${color} 20%, var(--surface))` : 'var(--surface-2)',
                        stroke: color,
                        strokeWidth: role ? 2 : 1,
                        strokeDasharray: v === null || role === 'outside' ? '4 3' : undefined,
                      }}
                    />
                    <text x={x} y={y + 14} className="viz-svg-range">
                      {nd.l === nd.r ? `[${nd.l}]` : `[${nd.l}..${nd.r}]`}
                    </text>
                    <text x={x} y={y + 32} className="viz-svg-value">
                      {fmt(v)}
                    </text>
                  </g>
                );
              })}

              {/* Array row */}
              {Array.from({ length: n }, (_, i) => {
                const leaf = leafOf.get(i)!;
                return (
                  <g key={`a${i}`}>
                    <text x={(i + 0.5) * CELL} y={treeH + 16} className="viz-svg-range">
                      a[{i}]
                    </text>
                    <text x={(i + 0.5) * CELL} y={treeH + 33} className="viz-svg-value">
                      {fmt(step.values[leaf])}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="viz-legend" style={{ marginTop: '0.75rem' }}>
            <Swatch color="var(--role-compare)" label="partial overlap: recurse" />
            <Swatch color="var(--role-done)" label="fully inside: use whole node" />
            <Swatch color="var(--role-change)" label="written / recomputed" />
            <span style={{ opacity: 0.6 }}>dashed = skipped (no overlap)</span>
            <Swatch color="var(--text)" label="current node" outline />
          </div>
        </div>

        <CodePanel
          listing={{ build: BUILD_CODE, query: QUERY_CODE, update: UPDATE_CODE }}
          cursor={step.code}
          previous={player.steps[player.index - 1]?.code}
          title={CODE_TITLE[step.code.listing ?? 'build']}
        />
      </div>

      <div className="viz-stats">
        <Stat k={step.acc !== undefined ? `Running ${label}` : `Root (${label} of all)`} v={fmt(step.acc ?? step.values[1])} />
        <Stat k="Nodes touched" v={`${visited} / ${tree.nodes.length}`} />
        <Stat k="Tree height" v={String(tree.depth + 1)} />
        <Stat k="Cost" v="O(log n)" />
      </div>
    </div>
  );
}

function NumInput({
  value,
  onChange,
  min,
  max,
  label,
}: {
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  label: string;
}) {
  return (
    <input
      className="input viz-num"
      type="number"
      value={value}
      min={min}
      max={max}
      aria-label={label}
      onChange={(e) => onChange(Math.max(min, Math.min(max, +e.target.value)))}
    />
  );
}
