import { useEffect, useMemo, useState } from 'react';
import { ADD_CODE, BUILD_CODE, FenwickTree, PREFIX_CODE, lowbit, type FenRole, type FenStep } from '../../lib/algorithms/fenwickTree';
import { usePlayer } from './usePlayer';
import { PlayerControls, Swatch } from './PlayerControls';
import { CodePanel } from './CodePanel';
import { Stat } from './Stat';
import './viz.css';

const ROLE_COLOR: Record<FenRole, string> = {
  visit: 'var(--role-compare)',
  use: 'var(--role-done)',
  changed: 'var(--role-change)',
};

const LISTINGS = { build: BUILD_CODE, prefix: PREFIX_CODE, add: ADD_CODE };

const CODE_TITLE: Record<string, string> = {
  build: 'Construction · O(n)',
  prefix: 'Prefix sum · O(log n)',
  add: 'Point add · O(log n)',
};

const initialValues = (n: number) => Array.from({ length: n }, (_, i) => ((i * 5 + 2) % 9) + 1);
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.trunc(v) || lo));

function idleStep(ft: FenwickTree, message: string): FenStep {
  const values = [0];
  const tree = [0];
  for (let i = 1; i <= ft.n; i++) {
    values.push(ft.value(i));
    tree.push(ft.cell(i));
  }
  return {
    values,
    tree,
    roles: {},
    current: null,
    message,
    target: null,
    code: { listing: 'build', line: 'body', vars: { n: ft.n } },
  };
}

export default function FenwickTreeViz({ initialSize = 16 }: { initialSize?: number }) {
  const [values, setValues] = useState(() => initialValues(initialSize));
  const [pi, setPi] = useState(11);
  const [rl, setRl] = useState(5);
  const [rr, setRr] = useState(13);
  const [ai, setAi] = useState(6);
  const [ad, setAd] = useState(3);

  const ft = useMemo(() => new FenwickTree(values), [values]);
  const intro = useMemo(
    () => [
      idleStep(
        ft,
        'Each T[i] stores the sum of a range ending at i, of length lowbit(i), the lowest set bit of i. Building all of them costs O(n), as the code shows. Run an operation below.',
      ),
    ],
    [ft],
  );
  const player = usePlayer<FenStep>(intro);
  useEffect(() => player.load(intro), [intro]); // eslint-disable-line react-hooks/exhaustive-deps

  const n = ft.n;
  const step = player.step;
  const bits = Math.floor(Math.log2(n)) + 1;
  const levels = Math.floor(Math.log2(n));
  const cols = { gridTemplateColumns: `repeat(${n}, minmax(3.4rem, 1fr))` };
  const inTarget = (i: number) => step.target && i >= step.target[0] && i <= step.target[1];

  return (
    <div className="viz" aria-label="Fenwick tree visualization">
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Array size <output>{n}</output>
          </span>
          <input type="range" min={4} max={16} value={n} onChange={(e) => setValues(initialValues(+e.target.value))} />
        </label>
        <button type="button" className="btn" onClick={() => setValues(values.map(() => 1 + Math.floor(Math.random() * 9)))}>
          Random values
        </button>
      </div>

      <div className="viz-ops">
        <div className="viz-row">
          <span className="viz-op-label">Prefix sum</span>
          <code>
            sum(a[1..
            <Num value={pi} set={setPi} min={1} max={n} label="Prefix end" />
            ])
          </code>
          <button type="button" className="btn primary" onClick={() => player.load(prefixSteps(ft, clamp(pi, 1, n)), true)}>
            Run
          </button>
        </div>
        <div className="viz-row">
          <span className="viz-op-label">Range sum</span>
          <code>
            sum(a[
            <Num value={rl} set={setRl} min={1} max={n} label="Range start" />
            ..
            <Num value={rr} set={setRr} min={1} max={n} label="Range end" />
            ])
          </code>
          <button
            type="button"
            className="btn primary"
            onClick={() => player.load(ft.rangeSum(clamp(Math.min(rl, rr), 1, n), clamp(Math.max(rl, rr), 1, n)).steps, true)}
          >
            Run
          </button>
        </div>
        <div className="viz-row">
          <span className="viz-op-label">Point add</span>
          <code>
            a[
            <Num value={ai} set={setAi} min={1} max={n} label="Index" />] +=
            <Num value={ad} set={setAd} min={-9} max={9} label="Delta" />
          </code>
          <button type="button" className="btn primary" onClick={() => player.load(ft.add(clamp(ai, 1, n), ad), true)}>
            Run
          </button>
        </div>
      </div>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-split">
        <div>
          <div className="viz-scroll">
            <div className="viz-fen" style={cols}>
              {/* Index + binary */}
              {Array.from({ length: n }, (_, k) => {
                const i = k + 1;
                const b = i.toString(2).padStart(bits, '0');
                const lbPos = bits - 1 - Math.log2(lowbit(i));
                return (
                  <div key={`i${i}`} className="viz-fen-idx" style={{ gridColumn: i, gridRow: 1 }}>
                    <strong>{i}</strong>
                    <span className="viz-fen-bin">
                      {b.slice(0, lbPos)}
                      <u>{b[lbPos]}</u>
                      {b.slice(lbPos + 1)}
                    </span>
                  </div>
                );
              })}

              {/* Underlying array */}
              {Array.from({ length: n }, (_, k) => {
                const i = k + 1;
                return (
                  <div
                    key={`a${i}`}
                    className="viz-chip-cell"
                    title={`a[${i}] = ${step.values[i]}`}
                    style={{
                      gridColumn: i,
                      gridRow: 2,
                      boxShadow: inTarget(i) ? 'inset 0 0 0 2px var(--role-compare)' : undefined,
                    }}
                  >
                    {step.values[i]}
                  </div>
                );
              })}

              {/* Tree cells: one row per lowbit size, biggest ranges on top */}
              {Array.from({ length: n }, (_, k) => {
                const i = k + 1;
                const lb = lowbit(i);
                const level = Math.log2(lb);
                const role = step.roles[i];
                const current = step.current === i;
                const color = role ? ROLE_COLOR[role] : 'var(--border)';
                return (
                  <div
                    key={`t${i}`}
                    className="viz-fen-cell"
                    title={`T[${i}] = sum of a[${i - lb + 1}..${i}] = ${step.tree[i]}`}
                    style={{
                      gridColumn: `${i - lb + 1} / ${i + 1}`,
                      gridRow: 3 + (levels - level),
                      borderColor: color,
                      background: role ? `color-mix(in srgb, ${color} 20%, var(--surface))` : 'var(--surface-2)',
                      outline: current ? '2px solid var(--text)' : undefined,
                      outlineOffset: 2,
                    }}
                  >
                    <span className="muted">T[{i}]</span> <strong>{step.tree[i]}</strong>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="viz-legend" style={{ marginTop: '0.75rem' }}>
            <span>
              Binary index, <u>lowest set bit</u> underlined. It sets how many items T[i] covers.
            </span>
            <Swatch color="var(--role-done)" label="added to the sum" />
            <Swatch color="var(--role-change)" label="updated" />
            <Swatch color="var(--text)" label="current i" outline />
          </div>
        </div>

        <CodePanel
          listing={LISTINGS}
          cursor={step.code}
          previous={player.steps[player.index - 1]?.code}
          title={CODE_TITLE[step.code.listing ?? 'build']}
        />
      </div>

      <div className="viz-stats">
        <Stat k="Running total" v={step.acc !== undefined ? String(step.acc) : '–'} />
        <Stat k="Cells touched" v={String(Object.keys(step.roles).length)} />
        <Stat k="Max cells per op" v={`⌊log₂ ${n}⌋+1 = ${levels + 1}`} />
        <Stat k="Memory" v={`${n} numbers`} />
      </div>
    </div>
  );
}

function prefixSteps(ft: FenwickTree, i: number): FenStep[] {
  const steps: FenStep[] = [];
  ft.prefixSum(i, steps);
  return steps;
}

function Num({ value, set, min, max, label }: { value: number; set: (v: number) => void; min: number; max: number; label: string }) {
  return (
    <input
      className="input viz-num"
      type="number"
      value={value}
      min={min}
      max={max}
      aria-label={label}
      onChange={(e) => set(Math.max(min, Math.min(max, +e.target.value)))}
    />
  );
}
