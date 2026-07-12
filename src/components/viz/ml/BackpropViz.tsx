import { useEffect, useMemo, useState } from 'react';
import { NEURON, PARAMS, loss, run, sgdStep, type GradStep, type Values } from '../../../lib/ml/autograd';
import { usePlayer } from '../usePlayer';
import { PlayerControls, Swatch } from '../PlayerControls';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

// Grid positions (column, row) for each node of the neuron graph.
const LAYOUT: Record<string, [number, number]> = {
  x1: [0, 0], w1: [0, 1], x2: [0, 2.3], w2: [0, 3.3],
  a: [1, 0.5], c: [1, 2.8], b: [1.9, 4.1],
  s: [2, 1.65], n: [3, 2.2], o: [4, 2.2], y: [4, 3.6], d: [5, 2.6], L: [6, 2.6],
};
const COL = 132;
const ROW = 68;
const NW = 104;
const NH = 50;

const INITIAL: Values = { x1: 2, x2: -1, w1: -0.5, w2: 0.8, b: 0.3, y: 1 };
const r3 = (v: number | undefined) => (v === undefined ? '·' : String(Math.round(v * 1000) / 1000));

export default function BackpropViz() {
  const [leaves, setLeaves] = useState<Values>(INITIAL);
  const [lr, setLr] = useState(0.1);
  const [history, setHistory] = useState<number[]>(() => [loss(NEURON, INITIAL)]);

  const { steps } = useMemo(() => run(NEURON, leaves), [leaves]);
  const player = usePlayer<GradStep>(steps, 2);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps
  const step = player.step;

  const setLeaf = (id: string, v: number) => {
    const next = { ...leaves, [id]: v };
    setLeaves(next);
    setHistory([loss(NEURON, next)]);
  };
  const train = (n: number) => {
    let v = leaves;
    const h = [...history];
    for (let i = 0; i < n; i++) {
      v = sgdStep(NEURON, v, lr);
      h.push(loss(NEURON, v));
    }
    setLeaves(v);
    setHistory(h);
  };
  const reset = () => {
    setLeaves(INITIAL);
    setHistory([loss(NEURON, INITIAL)]);
  };

  const pos = (id: string) => {
    const [c, r] = LAYOUT[id]!;
    return { x: r1(12 + c * COL), y: r1(12 + r * ROW) };
  };
  const W = 12 + 6 * COL + NW + 12;
  const H = 12 + 4.1 * ROW + NH + 12;
  const touched = new Set(step.touched);
  const backward = step.phase !== 'forward';

  return (
    <div className="viz" aria-label="Backpropagation visualization">
      <div className="viz-row">
        {(['x1', 'x2', 'y'] as const).map((id) => (
          <label key={id} className="viz-field">
            <span>{id === 'y' ? 'target y' : id === 'x1' ? 'input x₁' : 'input x₂'}</span>
            <input className="input viz-num" type="number" step={0.5} value={leaves[id]} onChange={(e) => setLeaf(id, +e.target.value || 0)} />
          </label>
        ))}
        <label className="viz-slider">
          <span>
            Learning rate η <output>{lr.toFixed(2)}</output>
          </span>
          <input type="range" min={0.01} max={1} step={0.01} value={lr} onChange={(e) => setLr(+e.target.value)} />
        </label>
      </div>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-scroll">
        <svg className="viz-svg" viewBox={`0 0 ${W} ${H}`} style={{ minWidth: 720 }} role="img" aria-label="Computational graph of one neuron">
          <defs>
            <marker id="bp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--text-3)' }} />
            </marker>
          </defs>
          {NEURON.flatMap((node) =>
            node.inputs.map((inp) => {
              const a = pos(inp);
              const b = pos(node.id);
              const active = step.current === node.id && (backward ? touched.has(inp) : true);
              return (
                <line
                  key={`${inp}-${node.id}`}
                  x1={a.x + NW}
                  y1={a.y + NH / 2}
                  x2={b.x}
                  y2={b.y + NH / 2}
                  markerEnd="url(#bp-arrow)"
                  style={{
                    stroke: active ? (backward ? 'var(--role-change)' : 'var(--role-compare)') : 'var(--border)',
                    strokeWidth: active ? 2.5 : 1.5,
                  }}
                />
              );
            }),
          )}
          {NEURON.map((node) => {
            const { x, y } = pos(node.id);
            const value = step.data[node.id];
            const grad = step.grad[node.id];
            const isCurrent = step.current === node.id;
            const isTouched = touched.has(node.id);
            const accent = isTouched ? (backward ? 'var(--role-change)' : 'var(--role-compare)') : node.kind === 'param' ? 'var(--role-done)' : 'var(--border)';
            return (
              <g key={node.id}>
                <title>{`${node.label}: value ${r3(value)}, gradient ${r3(grad)}`}</title>
                {isCurrent && <rect x={x - 4} y={y - 4} width={NW + 8} height={NH + 8} rx={11} style={{ fill: 'none', stroke: 'var(--text)', strokeWidth: 2 }} />}
                <rect
                  x={x}
                  y={y}
                  width={NW}
                  height={NH}
                  rx={8}
                  style={{
                    fill: isTouched ? `color-mix(in srgb, ${accent} 18%, var(--surface))` : 'var(--surface-2)',
                    stroke: accent,
                    strokeWidth: isTouched || node.kind === 'param' ? 2 : 1,
                  }}
                />
                <text x={x + NW / 2} y={y + 15} className="viz-svg-range" style={{ fontSize: 11 }}>
                  {node.label}
                </text>
                <text x={x + NW / 2} y={y + 31} className="viz-svg-value" style={{ fontSize: 13 }}>
                  {value === undefined ? '?' : r3(value)}
                </text>
                <text x={x + NW / 2} y={y + 45} className="viz-svg-range" style={{ fontSize: 10, fill: grad !== undefined && step.phase !== 'forward' ? 'var(--role-change)' : 'var(--text-3)' }}>
                  grad {grad === undefined || step.phase === 'forward' ? '·' : r3(grad)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="viz-legend">
        <Swatch color="var(--role-compare)" label="forward: value computed" />
        <Swatch color="var(--role-change)" label="backward: gradient pushed to inputs" />
        <Swatch color="var(--role-done)" label="trainable parameter" outline />
        <Swatch color="var(--text)" label="current node" outline />
      </div>

      <div className="viz-row">
        <button type="button" className="btn primary" onClick={() => train(1)}>
          Take a gradient step
        </button>
        <button type="button" className="btn" onClick={() => train(20)}>
          Train 20 steps
        </button>
        <button type="button" className="btn" onClick={reset}>
          Reset
        </button>
        <span className="muted" style={{ fontSize: '0.85rem' }}>
          Each step: w ← w − η · ∂L/∂w for w₁, w₂, b
        </span>
      </div>

      <div className="viz-stats">
        <Stat k="Loss" v={r3(loss(NEURON, leaves))} />
        {PARAMS.map((p) => (
          <Stat key={p} k={p === 'b' ? 'b' : p === 'w1' ? 'w₁' : 'w₂'} v={r3(leaves[p])} />
        ))}
        <Stat k="Steps taken" v={String(history.length - 1)} />
      </div>

      {history.length > 1 && (
        <LineChart
          series={[{ id: 'loss', name: 'loss', color: 'var(--role-compare)', values: history }]}
          xLabel="step"
          yLabel="loss"
          height={180}
        />
      )}
    </div>
  );
}
