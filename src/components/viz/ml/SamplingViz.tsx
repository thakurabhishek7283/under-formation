import { useMemo, useState } from 'react';
import { PROMPTS, applySampling, entropy, sample } from '../../../lib/ml/sampling';
import { Stat } from '../Stat';
import { pct } from './num';
import '../viz.css';
import './ml.css';

const PRESETS = [
  { label: 'Greedy', t: 0.01, k: 0, p: 1 },
  { label: 'Balanced (T 0.7, top-p 0.9)', t: 0.7, k: 0, p: 0.9 },
  { label: 'Creative (T 1.2, top-k 8)', t: 1.2, k: 8, p: 1 },
  { label: 'Raw (T 1, no filters)', t: 1, k: 0, p: 1 },
];

export default function SamplingViz() {
  const [promptIdx, setPromptIdx] = useState(0);
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState(0);
  const [topP, setTopP] = useState(1);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [lastPick, setLastPick] = useState<string | null>(null);

  const prompt = PROMPTS[promptIdx]!;
  const rows = useMemo(
    () => applySampling(prompt.candidates, { temperature, topK, topP }),
    [prompt, temperature, topK, topP],
  );
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const maxP = Math.max(...rows.map((r) => Math.max(r.p, r.base)));

  const draw = (n: number) => {
    const next = { ...counts };
    let pick = '';
    for (let i = 0; i < n; i++) {
      pick = rows[sample(rows)]!.token;
      next[pick] = (next[pick] ?? 0) + 1;
    }
    setCounts(next);
    setLastPick(pick);
  };
  const reset = () => {
    setCounts({});
    setLastPick(null);
  };
  const change = (f: () => void) => {
    f();
    reset();
  };

  const kept = rows.filter((r) => r.p > 0).length;

  return (
    <div className="viz" aria-label="Sampling visualization">
      <div className="viz-row">
        <label className="viz-field">
          <span>Prompt</span>
          <select className="input" value={promptIdx} onChange={(e) => change(() => setPromptIdx(+e.target.value))}>
            {PROMPTS.map((p, i) => (
              <option key={p.prompt} value={i}>
                “{p.prompt} …”
              </option>
            ))}
          </select>
        </label>
        <label className="viz-field">
          <span>Preset</span>
          <select
            className="input"
            value=""
            onChange={(e) => {
              const p = PRESETS[+e.target.value];
              if (p) change(() => { setTemperature(p.t); setTopK(p.k); setTopP(p.p); });
            }}
          >
            <option value="" disabled>
              Choose…
            </option>
            {PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Temperature <output>{temperature.toFixed(2)}</output>
          </span>
          <input type="range" min={0.01} max={2} step={0.01} value={temperature} onChange={(e) => change(() => setTemperature(+e.target.value))} />
        </label>
        <label className="viz-slider">
          <span>
            Top-k <output>{topK === 0 ? 'off' : topK}</output>
          </span>
          <input type="range" min={0} max={prompt.candidates.length} value={topK} onChange={(e) => change(() => setTopK(+e.target.value))} />
        </label>
        <label className="viz-slider">
          <span>
            Top-p <output>{topP >= 1 ? 'off' : topP.toFixed(2)}</output>
          </span>
          <input type="range" min={0.05} max={1} step={0.01} value={topP} onChange={(e) => change(() => setTopP(+e.target.value))} />
        </label>
      </div>

      <div className="viz-prompt">
        <span className="muted">{prompt.prompt}</span> <span className="viz-tok new">{lastPick ?? '___'}</span>
      </div>

      <div className="viz-scroll">
        <table className="viz-dist">
          <thead>
            <tr>
              <th>Token</th>
              <th>Logit</th>
              <th className="bar-col">Probability</th>
              <th>Cumulative</th>
              {total > 0 && <th>Drawn</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.token} className={r.cut ? 'cut' : ''} title={r.cut ? `removed by ${r.cut}` : `p = ${(r.p * 100).toFixed(2)}%`}>
                <td className="mono">{r.token}</td>
                <td className="mono num">{r.logit.toFixed(1)}</td>
                <td className="bar-col">
                  <span className="viz-dist-bar">
                    <span className="base" style={{ width: pct((r.base / maxP)) }} />
                    <span className="final" style={{ width: pct((r.p / maxP)) }} />
                  </span>
                  <span className="viz-dist-val">{r.cut ? `cut by ${r.cut}` : `${(r.p * 100).toFixed(1)}%`}</span>
                </td>
                <td className="mono num">{Number.isNaN(r.cumulative) ? '–' : `${(r.cumulative * 100).toFixed(0)}%`}</td>
                {total > 0 && <td className="mono num">{counts[r.token] ? `${counts[r.token]} (${(((counts[r.token] ?? 0) / total) * 100).toFixed(0)}%)` : '–'}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="viz-legend">
        <span>
          <span className="sw" style={{ background: 'var(--role-compare)' }} /> final probability
        </span>
        <span>
          <span className="sw" style={{ background: 'var(--border)' }} /> original (T = 1, no filters)
        </span>
        <span>faded rows = filtered out</span>
      </div>

      <div className="viz-row">
        <button type="button" className="btn primary" onClick={() => draw(1)}>
          Sample 1 token
        </button>
        <button type="button" className="btn" onClick={() => draw(100)}>
          Sample 100
        </button>
        <button type="button" className="btn" onClick={() => draw(1000)}>
          Sample 1,000
        </button>
        <button type="button" className="btn" onClick={reset} disabled={total === 0}>
          Clear
        </button>
      </div>

      <div className="viz-stats">
        <Stat k="Candidates kept" v={`${kept} / ${rows.length}`} />
        <Stat k="Top token probability" v={`${(rows[0]!.p * 100).toFixed(1)}%`} />
        <Stat k="Entropy" v={`${entropy(rows.map((r) => r.p)).toFixed(2)} bits`} />
        <Stat k="Samples drawn" v={String(total)} />
      </div>
    </div>
  );
}
