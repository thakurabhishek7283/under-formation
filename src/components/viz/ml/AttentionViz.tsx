import { useMemo, useState } from 'react';
import { HEADS, WORD_CLASSES, selfAttention, splitWords, wordClass } from '../../../lib/ml/attention';
import { Heatmap, ScaleLegend, fmt2 } from './Heatmap';
import { pct } from './num';
import '../viz.css';
import './ml.css';

const SENTENCES = [
  'The cat sat on the mat because the cat was tired',
  'A big dog chased the small bird into the park',
  'She saw the man with the red ball and he saw her',
];

export default function AttentionViz() {
  const [text, setText] = useState(SENTENCES[0]!);
  const [headIdx, setHeadIdx] = useState(0);
  const [causal, setCausal] = useState(true);
  const [query, setQuery] = useState(4);
  const [view, setView] = useState<'one' | 'all'>('one');

  const tokens = useMemo(() => splitWords(text), [text]);
  const results = useMemo(() => HEADS.map((h) => selfAttention(tokens, h, causal)), [tokens, causal]);
  const res = results[headIdx]!;
  const q = Math.min(query, tokens.length - 1);
  const labels = tokens.map((t, i) => `${i} ${t}`);
  const scoreDomain = Math.max(...res.scores.flat().map(Math.abs), 1e-9);

  if (tokens.length === 0) {
    return (
      <div className="viz">
        <input className="input" value={text} onChange={(e) => setText(e.target.value)} />
        <p className="muted">Type a sentence.</p>
      </div>
    );
  }

  const row = res.weights[q]!;
  const ranked = row.map((w, j) => ({ j, w, s: res.scores[q]![j]! })).sort((a, b) => b.w - a.w);
  const mix = res.output[q]!.slice(0, WORD_CLASSES.length);

  return (
    <div className="viz" aria-label="Self-attention visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Sentence (up to 12 words)</span>
          <input className="input" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
        </label>
        <label className="viz-field">
          <span>Examples</span>
          <select className="input" value={SENTENCES.indexOf(text)} onChange={(e) => setText(SENTENCES[+e.target.value] ?? text)}>
            <option value={-1} disabled>
              Custom
            </option>
            {SENTENCES.map((s, i) => (
              <option key={s} value={i}>
                {s.slice(0, 32)}…
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="viz-row">
        <div className="viz-seg" role="radiogroup" aria-label="Attention head">
          {HEADS.map((h, i) => (
            <button key={h.id} type="button" role="radio" aria-checked={i === headIdx} className={`btn${i === headIdx ? ' on' : ''}`} onClick={() => setHeadIdx(i)}>
              {h.name.replace(' head', '')}
            </button>
          ))}
        </div>
        <label className="viz-check">
          <input type="checkbox" checked={causal} onChange={(e) => setCausal(e.target.checked)} /> Causal mask (GPT-style)
        </label>
        <div className="viz-seg" role="radiogroup" aria-label="View">
          <button type="button" role="radio" aria-checked={view === 'one'} className={`btn${view === 'one' ? ' on' : ''}`} onClick={() => setView('one')}>
            One head
          </button>
          <button type="button" role="radio" aria-checked={view === 'all'} className={`btn${view === 'all' ? ' on' : ''}`} onClick={() => setView('all')}>
            All heads
          </button>
        </div>
      </div>

      <p className="viz-caption" style={{ minHeight: 0 }}>
        <span>
          <strong>{res.head.name}.</strong> {res.head.description}
        </span>
      </p>

      {view === 'all' ? (
        <div className="viz-multiples">
          {results.map((r, h) => (
            <figure key={r.head.id} className={h === headIdx ? 'on' : ''}>
              <button type="button" className="viz-multiple-title" onClick={() => { setHeadIdx(h); setView('one'); }}>
                {r.head.name}
              </button>
              <Heatmap data={r.weights} scale="sequential" domain={1} masked={r.masked} cell="0.95rem" label={`${r.head.name} attention weights`} tip={(i, j, v) => `${tokens[i]} → ${tokens[j]}: ${(v * 100).toFixed(0)}%`} />
            </figure>
          ))}
          <p className="muted" style={{ gridColumn: '1 / -1', margin: 0, fontSize: '0.85rem' }}>
            Multi-head attention runs these side by side on the same tokens and concatenates the results, so one layer can track several relationships at once.
          </p>
        </div>
      ) : (
        <>
          <div className="viz-attn-grid">
            <figure>
              <figcaption>
                <strong>1 · Scores</strong> QKᵀ / √d
              </figcaption>
              <div className="viz-scroll">
                <Heatmap
                  data={res.scores}
                  scale="diverging"
                  domain={scoreDomain}
                  rowLabels={labels}
                  colLabels={tokens}
                  highlightRow={q}
                  onRowClick={setQuery}
                  masked={res.masked}
                  label="Attention scores"
                  tip={(i, j, v) => `${tokens[i]} · ${tokens[j]} = ${fmt2(v)}`}
                />
              </div>
              <ScaleLegend scale="diverging" domain={scoreDomain} />
            </figure>
            <figure>
              <figcaption>
                <strong>2 · Weights</strong> softmax per row{causal ? ', future masked' : ''}
              </figcaption>
              <div className="viz-scroll">
                <Heatmap
                  data={res.weights}
                  scale="sequential"
                  domain={1}
                  rowLabels={labels}
                  colLabels={tokens}
                  highlightRow={q}
                  onRowClick={setQuery}
                  masked={res.masked}
                  label="Attention weights"
                  tip={(i, j, v) => `${tokens[i]} attends to ${tokens[j]}: ${(v * 100).toFixed(1)}%`}
                />
              </div>
              <ScaleLegend scale="sequential" domain={1} format={(v) => `${Math.round(v * 100)}%`} />
            </figure>
          </div>

          <div className="viz-attn-detail">
            <div>
              <p className="viz-label">
                Query “{tokens[q]}” (row {q}). Click a row label to change it.
              </p>
              <ul className="viz-bars-h">
                {ranked
                  .filter((r) => !res.masked(q, r.j))
                  .slice(0, 6)
                  .map((r) => (
                    <li key={r.j}>
                      <span className="k">
                        {r.j} {tokens[r.j]}
                      </span>
                      <span className="bar">
                        <span style={{ width: pct(r.w) }} />
                      </span>
                      <span className="v">
                        {(r.w * 100).toFixed(1)}% <span className="muted">(score {fmt2(r.s)})</span>
                      </span>
                    </li>
                  ))}
              </ul>
            </div>
            <div>
              <p className="viz-label">3 · Output = weights · V (what “{tokens[q]}” gathered)</p>
              <ul className="viz-bars-h">
                {WORD_CLASSES.map((c, k) => (
                  <li key={c}>
                    <span className="k">{c}</span>
                    <span className="bar">
                      <span style={{ width: pct(Math.max(0, mix[k]!)) }} />
                    </span>
                    <span className="v">{(mix[k]! * 100).toFixed(0)}%</span>
                  </li>
                ))}
              </ul>
              <p className="muted" style={{ fontSize: '0.8rem', margin: '0.4rem 0 0' }}>
                Values here carry each word's class ({tokens.map((t) => `${t}=${wordClass(t)}`).slice(0, 3).join(', ')}…), so the output shows which kinds of words this token pulled information from.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
