import { useEffect, useMemo, useState } from 'react';
import {
  MERGE_SORT_CODE,
  QUICK_SORT_CODE,
  mergeSortSteps,
  quickSortSteps,
  type PivotStrategy,
  type SortStep,
} from '../../lib/algorithms/sorting';
import { usePlayer } from './usePlayer';
import { PlayerControls, Swatch } from './PlayerControls';
import { CodePanel } from './CodePanel';
import { Stat } from './Stat';
import './viz.css';

type Algorithm = 'merge' | 'quick';
type Preset = 'random' | 'nearly-sorted' | 'reversed' | 'few-unique';

const PRESETS: Record<Preset, string> = {
  random: 'Random',
  'nearly-sorted': 'Nearly sorted',
  reversed: 'Reversed',
  'few-unique': 'Few unique values',
};

function makeArray(n: number, preset: Preset): number[] {
  const rand = (lo: number, hi: number) => lo + Math.floor(Math.random() * (hi - lo + 1));
  switch (preset) {
    case 'reversed':
      return Array.from({ length: n }, (_, i) => Math.round(8 + ((n - i) / n) * 90));
    case 'nearly-sorted': {
      const a = Array.from({ length: n }, (_, i) => Math.round(8 + ((i + 1) / n) * 90));
      for (let s = 0; s < Math.max(1, Math.floor(n / 10)); s++) {
        const i = rand(0, n - 2);
        [a[i], a[i + 1]] = [a[i + 1]!, a[i]!];
      }
      return a;
    }
    case 'few-unique':
      return Array.from({ length: n }, () => [20, 45, 70, 95][rand(0, 3)]!);
    default:
      return Array.from({ length: n }, () => rand(5, 99));
  }
}

function trace(algorithm: Algorithm, input: number[], pivot: PivotStrategy): SortStep[] {
  return algorithm === 'merge' ? mergeSortSteps(input) : quickSortSteps(input, pivot);
}

interface Props {
  algorithm: Algorithm;
  initialSize?: number;
}

export default function SortingViz({ algorithm, initialSize = 16 }: Props) {
  const [size, setSize] = useState(initialSize);
  const [preset, setPreset] = useState<Preset>('random');
  const [pivot, setPivot] = useState<PivotStrategy>('last');
  // A fixed first array keeps server and client renders identical; "New array" randomizes.
  const [input, setInput] = useState(() => Array.from({ length: initialSize }, (_, i) => ((i * 37 + 11) % 89) + 8));

  const steps = useMemo(() => trace(algorithm, input, pivot), [algorithm, input, pivot]);
  const player = usePlayer<SortStep>(steps);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const regenerate = (n = size, p = preset) => setInput(makeArray(n, p));

  const step = player.step;
  const n = step.array.length;
  const max = Math.max(...input, 1);
  const final = new Set(step.final);
  const showLabels = n <= 32;
  const log2n = Math.log2(n);

  const barColor = (i: number) => {
    const role = step.roles[i];
    if (role === 'pivot') return 'var(--text)';
    if (role === 'change') return 'var(--role-change)';
    if (role === 'compare') return 'var(--role-compare)';
    if (final.has(i)) return 'var(--role-done)';
    return 'var(--role-idle)';
  };
  const inRange = (i: number) => !step.range || (i >= step.range[0] && i <= step.range[1]);
  const cols = { gridTemplateColumns: `repeat(${n}, minmax(0, 1fr))` };

  return (
    <div className="viz" aria-label={`${algorithm === 'merge' ? 'Merge' : 'Quick'} sort visualization`}>
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Items <output>{size}</output>
          </span>
          <input
            type="range"
            min={6}
            max={48}
            value={size}
            onChange={(e) => {
              setSize(+e.target.value);
              regenerate(+e.target.value);
            }}
          />
        </label>
        <label className="viz-field">
          <span>Input</span>
          <select
            className="input"
            value={preset}
            onChange={(e) => {
              const p = e.target.value as Preset;
              setPreset(p);
              regenerate(size, p);
            }}
          >
            {Object.entries(PRESETS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>
        {algorithm === 'quick' && (
          <label className="viz-field">
            <span>Pivot</span>
            <select className="input" value={pivot} onChange={(e) => setPivot(e.target.value as PivotStrategy)}>
              <option value="last">Last element</option>
              <option value="middle">Middle element</option>
              <option value="median-of-three">Median of three</option>
              <option value="random">Random</option>
            </select>
          </label>
        )}
        <button type="button" className="btn" onClick={() => regenerate()}>
          New array
        </button>
      </div>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-split">
        {/* Bars */}
        <div>
          <div className="viz-bars" style={cols}>
            {step.array.map((v, i) => (
              <div key={i} className="viz-bar-slot" style={{ opacity: inRange(i) ? 1 : 0.3 }}>
                {showLabels && <span className="viz-bar-label">{v}</span>}
                <div className="viz-bar" title={`a[${i}] = ${v}`} style={{ height: `${(v / max) * 100}%`, background: barColor(i) }} />
              </div>
            ))}
          </div>

          {/* Quick sort pointers */}
          {step.partition && (
            <div className="viz-markers" style={cols} aria-hidden="true">
              {step.array.map((_, i) => {
                const { i: pi, j, pivot: pv } = step.partition!;
                const labels = [i === pv && 'P', i === pi && 'i', i === j && 'j'].filter(Boolean).join(' ');
                return <span key={i}>{labels}</span>;
              })}
            </div>
          )}

          {/* Merge sort buffer */}
          {step.buffer && (
            <div className="viz-buffer">
              <p className="viz-label">
                Buffer: copy of a[{step.buffer.lo}..{step.buffer.lo + step.buffer.values.length - 1}] being merged
              </p>
              <div className="viz-cells-row" style={cols}>
                {step.array.map((_, i) => {
                  const b = step.buffer!;
                  const off = i - b.lo;
                  if (off < 0 || off >= b.values.length) return <span key={i} />;
                  const isLeft = i <= b.mid;
                  const consumed = isLeft ? i < b.i : i < b.j;
                  const head = (i === b.i && b.i <= b.mid) || i === b.j;
                  const taken = b.taken === i;
                  return (
                    <span
                      key={i}
                      className="viz-chip-cell"
                      data-side={isLeft ? 'L' : 'R'}
                      style={{
                        opacity: consumed ? 0.3 : 1,
                        boxShadow: taken ? 'inset 0 0 0 2px var(--role-change)' : head ? 'inset 0 0 0 2px var(--role-compare)' : undefined,
                      }}
                    >
                      {b.values[off]}
                    </span>
                  );
                })}
              </div>
              <div className="viz-markers" style={cols} aria-hidden="true">
                {step.array.map((_, i) => (
                  <span key={i}>{bufferMarker(step.buffer!, i)}</span>
                ))}
              </div>
            </div>
          )}

          <div className="viz-legend" style={{ marginTop: '0.75rem' }}>
            <Swatch color="var(--role-compare)" label={algorithm === 'merge' ? 'read head' : 'compared'} />
            <Swatch color="var(--role-change)" label={algorithm === 'merge' ? 'written' : 'swapped'} />
            {algorithm === 'quick' && <Swatch color="var(--text)" label="pivot (P)" />}
            <Swatch color="var(--role-done)" label="in final position" />
            <span>faded = outside the current subarray</span>
          </div>
        </div>

        <CodePanel
          listing={algorithm === 'merge' ? MERGE_SORT_CODE : QUICK_SORT_CODE}
          cursor={step.code}
          previous={player.steps[player.index - 1]?.code}
          title={algorithm === 'merge' ? 'Merge sort · O(n log n)' : 'Quick sort · Lomuto partition'}
        />
      </div>

      <div className="viz-stats">
        <Stat k="Comparisons" v={String(step.comparisons)} />
        <Stat k={algorithm === 'merge' ? 'Writes' : 'Swaps'} v={String(step.moves)} />
        {/* Merge sort's depth is exactly ⌈log₂ n⌉; quick sort's depends on how the pivots split. */}
        <Stat k="Recursion depth" v={algorithm === 'merge' ? `${step.depth} / ${Math.ceil(log2n)}` : String(step.depth)} />
        <Stat k="n log₂ n" v={String(Math.round(n * log2n))} />
        {algorithm === 'quick' && <Stat k="Worst case n(n−1)/2" v={String((n * (n - 1)) / 2)} />}
      </div>
    </div>
  );
}

/** "L"/"R" under the next unread item of each half; empty once a half is used up. */
function bufferMarker(b: NonNullable<SortStep['buffer']>, i: number): string {
  const hi = b.lo + b.values.length - 1;
  if (i === b.i && b.i <= b.mid) return 'L';
  if (i === b.j && b.j <= hi) return 'R';
  return '';
}
