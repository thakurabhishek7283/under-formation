import { useMemo, useState } from 'react';
import { BloomFilter, optimalHashCount } from '../../lib/algorithms/bloomFilter';
import { Stat } from './Stat';
import './viz.css';

const WORDS = [
  'apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew',
  'kiwi', 'lemon', 'mango', 'nectarine', 'orange', 'papaya', 'quince', 'raspberry',
  'strawberry', 'tangerine', 'ugli', 'vanilla', 'watermelon', 'yuzu', 'zucchini',
];

type LastOp =
  | { kind: 'add'; item: string; positions: number[]; duplicate: boolean }
  | { kind: 'check'; item: string; positions: number[]; result: boolean; present: boolean }
  | null;

interface Props {
  initialSize?: number;
  initialHashes?: number;
}

export default function BloomFilterViz({ initialSize = 64, initialHashes = 3 }: Props) {
  const [m, setM] = useState(initialSize);
  const [k, setK] = useState(initialHashes);
  const [items, setItems] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const [last, setLast] = useState<LastOp>(null);
  const [searchNote, setSearchNote] = useState('');

  // Derived: rebuilding is cheap, and it lets m/k changes re-hash the same items live.
  const filter = useMemo(() => {
    const f = new BloomFilter(m, k);
    for (const item of items) f.add(item);
    return f;
  }, [m, k, items]);

  const add = (raw: string) => {
    const item = raw.trim();
    if (!item) return;
    const duplicate = items.includes(item);
    if (!duplicate) setItems((prev) => [...prev, item]);
    setLast({ kind: 'add', item, positions: filter.positions(item), duplicate });
    setSearchNote('');
  };

  const check = (raw: string) => {
    const item = raw.trim();
    if (!item) return;
    const { result, positions } = filter.mightContain(item);
    setLast({ kind: 'check', item, positions, result, present: items.includes(item) });
    setSearchNote('');
  };

  const addRandom = () => {
    const unused = WORDS.filter((w) => !items.includes(w));
    add(unused.length ? unused[Math.floor(Math.random() * unused.length)]! : `item-${items.length}`);
  };

  const findFalsePositive = () => {
    if (items.length === 0) return;
    for (let i = 0; i < 20000; i++) {
      const candidate = `probe-${Math.floor(Math.random() * 1e6)}`;
      if (items.includes(candidate)) continue;
      const { result, positions } = filter.mightContain(candidate);
      if (result) {
        setLast({ kind: 'check', item: candidate, positions, result, present: false });
        setSearchNote(`Found after ${i + 1} random probe${i ? 's' : ''}.`);
        return;
      }
    }
    setLast(null);
    setSearchNote('No false positive in 20,000 random probes. The filter is doing well!');
  };

  const onSubmit = (e: { preventDefault(): void }) => {
    e.preventDefault();
    add(input);
    setInput('');
  };

  const reset = () => {
    setItems([]);
    setLast(null);
    setSearchNote('');
  };

  const n = items.length;
  const setBits = filter.setBits;
  const fpRate = filter.expectedFalsePositiveRate(n);
  const hl = new Set(last?.positions ?? []);

  return (
    <div className="viz wide" aria-label="Interactive Bloom filter">
      {/* Parameters */}
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Bit array size <i>m</i> = <output>{m}</output>
          </span>
          <input type="range" min={16} max={256} step={8} value={m} onChange={(e) => { setM(+e.target.value); setLast(null); }} />
        </label>
        <label className="viz-slider">
          <span>
            Hash functions <i>k</i> = <output>{k}</output>
          </span>
          <input type="range" min={1} max={8} value={k} onChange={(e) => { setK(+e.target.value); setLast(null); }} />
        </label>
      </div>

      {/* Input */}
      <form className="viz-row" onSubmit={onSubmit}>
        <input
          className="input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a word…"
          aria-label="Item"
        />
        <button className="btn primary" type="submit" disabled={!input.trim()}>
          Add
        </button>
        <button className="btn" type="button" disabled={!input.trim()} onClick={() => check(input)}>
          Check
        </button>
        <button className="btn" type="button" onClick={addRandom}>
          Add random fruit
        </button>
        <button className="btn" type="button" onClick={findFalsePositive} disabled={n === 0}>
          Find a false positive
        </button>
        <button className="btn" type="button" onClick={reset} disabled={n === 0}>
          Reset
        </button>
      </form>

      {/* Verdict */}
      <Verdict last={last} note={searchNote} />

      {/* Bit array */}
      <div>
        <p className="viz-label">Bit array · {setBits} of {m} bits set</p>
        <div className="viz-cells" style={{ ['--cell' as string]: m > 128 ? '1.5rem' : '2rem' }}>
          {Array.from(filter.bits, (bit, i) => {
            const isHl = hl.has(i);
            const miss = isHl && last?.kind === 'check' && bit === 0;
            return (
              <div
                key={i}
                tabIndex={0}
                className={`viz-cell${isHl ? (miss ? ' miss' : ' hl') : ''}`}
                data-tip={`bit ${i} = ${bit}`}
                style={
                  bit
                    ? { background: 'var(--seq-4)', color: 'var(--seq-ink-dark)' }
                    : undefined
                }
              >
                {bit}
              </div>
            );
          })}
        </div>
        <div className="viz-legend" style={{ marginTop: '0.6rem' }}>
          <span><span className="sw" style={{ background: 'var(--seq-0)' }} /> 0</span>
          <span><span className="sw" style={{ background: 'var(--seq-4)' }} /> 1</span>
          <span><span className="sw" style={{ boxShadow: '0 0 0 2px var(--accent)' }} /> probed by last operation</span>
          <span><span className="sw" style={{ boxShadow: '0 0 0 2px var(--critical)' }} /> probed and 0: proves absence</span>
        </div>
      </div>

      {/* Hash breakdown */}
      {last && (
        <div>
          <p className="viz-label">Hashes for “{last.item}”</p>
          <ul className="viz-list viz-mono">
            {last.positions.map((p, i) => (
              <li key={i}>
                g<sub>{i}</sub>(“{last.item}”) mod {m} = <strong>{p}</strong>
                <span className="muted"> → bit is {filter.bits[p]}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Stats */}
      <div className="viz-stats">
        <Stat k="Items inserted" v={String(n)} />
        <Stat k="Fill ratio" v={`${((setBits / m) * 100).toFixed(0)}%`} />
        <Stat k="Expected false-positive rate" v={`${(fpRate * 100).toFixed(2)}%`} />
        <Stat k={`Optimal k for n=${n}`} v={n ? String(optimalHashCount(m, n)) : "–"} />
      </div>

      {n > 0 && (
        <div>
          <p className="viz-label">Actually in the set (the filter doesn't store these)</p>
          <div className="viz-chips">
            {items.map((it) => (
              <button key={it} type="button" className="tag" onClick={() => check(it)} title="Check this item">
                {it}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Verdict({ last, note }: { last: LastOp; note: string }) {
  if (!last) {
    return (
      <div className="viz-verdict neutral" role="status">
        <span className="icon">i</span>
        <span>{note || 'Add a few words, then check some that you did and did not add.'}</span>
      </div>
    );
  }
  if (last.kind === 'add') {
    return (
      <div className="viz-verdict neutral" role="status">
        <span className="icon">+</span>
        <span>
          {last.duplicate ? 'Re-added' : 'Added'} “{last.item}”: set {last.positions.length} bit
          {last.positions.length === 1 ? '' : 's'} to 1.
        </span>
      </div>
    );
  }
  if (!last.result) {
    return (
      <div className="viz-verdict good" role="status">
        <span className="icon">✓</span>
        <span>
          “{last.item}” is <strong>definitely not</strong> in the set: at least one probed bit is 0.
        </span>
      </div>
    );
  }
  if (last.present) {
    return (
      <div className="viz-verdict good" role="status">
        <span className="icon">✓</span>
        <span>
          “{last.item}” is <strong>probably</strong> in the set, and it really is.
        </span>
      </div>
    );
  }
  return (
    <div className="viz-verdict bad" role="status">
      <span className="icon">⚠</span>
      <span>
        <strong>False positive!</strong> Every probed bit for “{last.item}” is 1, but it was never added.
        Other items set those bits. {note}
      </span>
    </div>
  );
}
