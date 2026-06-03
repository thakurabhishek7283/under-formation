import { useRef, useState } from 'react';
import { HyperLogLog, type HllStep } from '../../lib/algorithms/hyperLogLog';
import { toBinary32 } from '../../lib/algorithms/hash';
import { Stat } from './Stat';
import './viz.css';

interface Props {
  initialPrecision?: number;
}

const fmt = (n: number) => Math.round(n).toLocaleString('en-US');
const itemName = (i: number) => `user-${i}`;

/** Map a register value onto the 7-step sequential ramp, relative to the current max. */
function level(value: number, max: number): number {
  if (value === 0) return 0;
  return Math.max(1, Math.round((value / Math.max(max, 1)) * 7));
}

export default function HyperLogLogViz({ initialPrecision = 6 }: Props) {
  const [p, setP] = useState(initialPrecision);
  const hll = useRef(new HyperLogLog(initialPrecision));
  const [distinct, setDistinct] = useState(0);
  const [totalAdds, setTotalAdds] = useState(0);
  const [last, setLast] = useState<HllStep | null>(null);

  const addMany = (count: number, nameAt: (i: number) => string) => {
    let step: HllStep | null = null;
    for (let i = 0; i < count; i++) step = hll.current.add(nameAt(i));
    setTotalAdds((t) => t + count);
    setLast(step);
  };

  const addDistinct = (count: number) => {
    addMany(count, (i) => itemName(distinct + i));
    setDistinct((d) => d + count);
  };

  // Re-stream items already seen: the estimate should not move.
  const addDuplicates = (count: number) => {
    if (distinct > 0) addMany(count, () => itemName(Math.floor(Math.random() * distinct)));
  };

  const changePrecision = (next: number) => {
    // Replay the same distinct items into a sketch of the new size.
    const fresh = new HyperLogLog(next);
    for (let i = 0; i < distinct; i++) fresh.add(itemName(i));
    hll.current = fresh;
    setP(next);
    setLast(null);
  };

  const reset = () => {
    hll.current = new HyperLogLog(p);
    setDistinct(0);
    setTotalAdds(0);
    setLast(null);
  };

  const sketch = hll.current;
  const estimate = distinct === 0 ? 0 : sketch.estimate();
  const error = distinct === 0 ? 0 : (estimate - distinct) / distinct;
  const max = Math.max(...sketch.registers);
  const bits = last ? toBinary32(last.hash) : '';
  const zeros = last ? last.rank - 1 : 0;

  return (
    <div className="viz wide" aria-label="Interactive HyperLogLog">
      {/* Parameters */}
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Precision <i>p</i> = <output>{p}</output> → <output>{sketch.m}</output> registers
          </span>
          <input type="range" min={4} max={8} value={p} onChange={(e) => changePrecision(+e.target.value)} />
        </label>
      </div>

      <div className="viz-row">
        <button className="btn primary" onClick={() => addDistinct(1)}>+1 distinct</button>
        <button className="btn" onClick={() => addDistinct(100)}>+100</button>
        <button className="btn" onClick={() => addDistinct(1000)}>+1,000</button>
        <button className="btn" onClick={() => addDistinct(10000)}>+10,000</button>
        <button className="btn" onClick={() => addDuplicates(1000)} disabled={distinct === 0}>
          +1,000 duplicates
        </button>
        <button className="btn" onClick={reset} disabled={totalAdds === 0}>Reset</button>
      </div>

      {/* Stats */}
      <div className="viz-stats">
        <Stat k="Items streamed" v={fmt(totalAdds)} />
        <Stat k="True distinct count" v={fmt(distinct)} />
        <Stat k="HLL estimate" v={fmt(estimate)} />
        <Stat k="Error" v={distinct ? `${error >= 0 ? '+' : ''}${(error * 100).toFixed(1)}%` : '–'} />
        <Stat k="Expected std. error" v={`±${(sketch.standardError * 100).toFixed(1)}%`} />
        <Stat k="Sketch memory" v={`${Math.ceil((sketch.m * 5) / 8)} B`} />
      </div>

      {/* Last hash breakdown */}
      <div>
        <p className="viz-label">Last item's hash</p>
        {last ? (
          <div className="viz-mono">
            <div>
              h(“{last.item}”) =&nbsp;
              <span style={{ background: 'var(--accent-soft)', color: 'var(--accent-ink)', borderRadius: 4 }}>
                {bits.slice(0, p)}
              </span>
              <strong style={{ color: 'var(--critical)' }}>{bits.slice(p, p + zeros)}</strong>
              <strong>{bits.slice(p + zeros, p + zeros + 1)}</strong>
              <span className="muted">{bits.slice(p + zeros + 1)}</span>
            </div>
            <ul className="viz-list" style={{ marginTop: '0.5rem' }}>
              <li>
                first {p} bits <span style={{ color: 'var(--accent-ink)' }}>{bits.slice(0, p)}</span> → register{' '}
                <strong>#{last.register}</strong>
              </li>
              <li>
                {zeros} leading zero{zeros === 1 ? '' : 's'} in the rest → rank ρ = <strong>{last.rank}</strong>
              </li>
              <li>
                register #{last.register}: {last.previous} →{' '}
                <strong>{Math.max(last.previous, last.rank)}</strong>{' '}
                <span className="muted">
                  {last.rank > last.previous ? '(new maximum)' : '(unchanged: not a new maximum)'}
                </span>
              </li>
            </ul>
          </div>
        ) : (
          <p className="muted" style={{ margin: 0 }}>
            Stream some items to see how each one is hashed and bucketed.
          </p>
        )}
      </div>

      {/* Registers */}
      <div>
        <p className="viz-label">Registers · max leading-zero rank seen per bucket</p>
        <div className="viz-cells" style={{ ['--cell' as string]: sketch.m > 128 ? '1.6rem' : '2.2rem' }}>
          {Array.from(sketch.registers, (v, i) => {
            const lvl = level(v, max);
            return (
              <div
                key={i}
                tabIndex={0}
                className={`viz-cell${last?.register === i ? ' hl' : ''}`}
                data-tip={`register ${i} = ${v}`}
                style={{
                  background: `var(--seq-${lvl})`,
                  color: lvl >= 4 ? 'var(--seq-ink-dark)' : 'var(--seq-ink-light)',
                }}
              >
                {v || ''}
              </div>
            );
          })}
        </div>
        <div className="viz-legend" style={{ marginTop: '0.6rem' }}>
          <span>empty</span>
          {[0, 1, 2, 3, 4, 5, 6, 7].map((l) => (
            <span key={l} className="sw" style={{ background: `var(--seq-${l})` }} />
          ))}
          <span>max ({max})</span>
        </div>
      </div>
    </div>
  );
}
