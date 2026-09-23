// Per-ticket results of scripts/readout-experiment.mjs: what the model wrote, next to three
// ways of reading its answer straight out of the forward pass. Used in "The Answer Comes Before the Words".
import { useState } from 'react';
import data from '../../lib/ml/readoutExperiment.json';
import { Stat } from '../viz/Stat';
import '../viz/viz.css';
import './readout.css';

type Row = (typeof data.models)[number]['rows'][number];
const LABELS = data.labels;
const COLOR: Record<string, string> = { bug: 'var(--cat-1)', feature: 'var(--cat-2)', billing: 'var(--cat-3)' };

const pct = (p: number) => `${Math.round(p * 100)}%`;
// Make a token's leading space visible, since it's part of the token.
const showToken = (t: string) => JSON.stringify(t.replace(/^ /, '␣').replace(/\n/g, '⏎')).slice(1, -1);

function Verdict({ pick, label }: { pick: string | null; label: string }) {
  const ok = pick === label;
  return (
    <span className={`ro-verdict ${ok ? 'ok' : 'bad'}`}>
      {ok ? '✓' : '✗'} {pick ?? 'no label found'}
    </span>
  );
}

function LabelBars({ probs }: { probs: number[] }) {
  return (
    <ul className="ro-bars">
      {LABELS.map((l, i) => (
        <li key={l}>
          <code>{l}</code>
          <span className="ro-track">
            <span className="ro-fill" style={{ width: pct(probs[i]), background: COLOR[l] }} />
          </span>
          <span className="ro-p">{probs[i].toFixed(2)}</span>
        </li>
      ))}
    </ul>
  );
}

function Panel({ title, sub, children }: { title: string; sub: string; children: React.ReactNode }) {
  return (
    <section className="ro-panel">
      <p className="viz-label">{title}</p>
      <p className="ro-sub">{sub}</p>
      {children}
    </section>
  );
}

export default function ReadoutExplorer() {
  const [sel, setSel] = useState(0);
  const [m, setM] = useState(data.models.length - 1);
  const { summary: S, rows } = data.models[m];
  const r: Row = rows[sel];
  return (
    <div className="viz">
      <div className="viz-row" role="radiogroup" aria-label="Model">
        {data.models.map((x, i) => (
          <button key={i} role="radio" aria-checked={i === m} className={`ro-model${i === m ? ' selected' : ''}`} onClick={() => setM(i)}>
            {x.summary.model}
          </button>
        ))}
      </div>
      <div className="viz-stats">
        <Stat k="Wrote an answer" v={`${S.accuracy.generated}%`} />
        <Stat k="One pass, one-word prompt" v={`${S.accuracy.constrained}%`} />
        <Stat k="Scored whole answers" v={`${S.accuracy.sentences}%`} />
        <Stat k="Naive first token" v={`${S.accuracy.freeFirst}%`} />
      </div>
      <p className="ro-note">
        Accuracy against my labels on {S.tickets} tickets. {S.source}. Machine: {data.machine}.
      </p>

      <div>
        <p className="viz-label">Pick a ticket</p>
        <div className="ro-tickets" role="listbox" aria-label="Tickets">
          {rows.map((row, i) => (
            <button
              key={i}
              role="option"
              aria-selected={i === sel}
              className={`ro-ticket${i === sel ? ' selected' : ''}`}
              onClick={() => setSel(i)}
              title={row.text}
            >
              <span className="ro-dot" style={{ background: COLOR[row.label] }} aria-hidden="true" />
              {row.text}
              {row.ambiguous && <span className="ro-tag">ambiguous</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="ro-selected">
        <span className="ro-dot" style={{ background: COLOR[r.label] }} aria-hidden="true" />
        <span>“{r.text}”</span>
        <span className="ro-truth">labelled {r.label}</span>
      </div>

      <div className="ro-grid">
        <Panel title="What it wrote" sub={`Free-form prompt. ${r.generated.tokens} tokens, ${r.generated.ms} ms, then a parser looks for the first label word.`}>
          <blockquote className="ro-gen">{r.generated.text}</blockquote>
          <Verdict pick={r.generated.parsed} label={r.label} />
        </Panel>

        <Panel title="One pass, one-word prompt" sub={`First-token probabilities, renormalised over the three labels. ${pct(r.constrained.mass)} of all probability landed on a label token. ${r.constrained.ms} ms.`}>
          <LabelBars probs={r.constrained.probs} />
          <Verdict pick={r.constrained.pick} label={r.label} />
        </Panel>

        <Panel title="Score the whole answers" sub={`Same free-form prompt. Probability of “This is a bug report.” and the other two sentences, one pass each. ${r.sentences.ms} ms.`}>
          <LabelBars probs={r.sentences.probs} />
          <Verdict pick={r.sentences.pick} label={r.label} />
        </Panel>

        <Panel title="Naive: first token, free-form prompt" sub={`What the model actually wants to say first. Only ${pct(r.freeFirst.mass)} of the probability is on a label token, so renormalising it reads noise.`}>
          <ul className="ro-bars ro-tokens">
            {r.freeFirst.top.map((t, i) => (
              <li key={i}>
                <code>{showToken(t.token)}</code>
                <span className="ro-track">
                  <span className="ro-fill" style={{ width: pct(t.p), background: 'var(--role-idle)' }} />
                </span>
                <span className="ro-p">{t.p.toFixed(2)}</span>
              </li>
            ))}
          </ul>
          <Verdict pick={r.freeFirst.pick} label={r.label} />
        </Panel>
      </div>
    </div>
  );
}
