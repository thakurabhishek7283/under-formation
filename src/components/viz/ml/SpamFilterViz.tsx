import { useMemo, useState } from 'react';
import { CORPUS, classify, fitWords } from '../../../lib/ml/bayes';
import { Stat } from '../Stat';
import '../viz.css';
import './ml.css';

const EXAMPLES = [
  'Claim your free prize now',
  'Can you call me when you get home',
  'Free lunch at the meeting tomorrow',
  'Urgent: click the link to verify your cash reward',
];
const ALPHAS = [0, 0.1, 0.5, 1, 2];
const BAR = 6; // |log-likelihood ratio| that fills a bar

const fmt = (v: number) => (v === Infinity ? '+∞' : v === -Infinity ? '−∞' : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`);

export default function SpamFilterViz() {
  const [text, setText] = useState(EXAMPLES[2]!);
  const [alphaIdx, setAlphaIdx] = useState(3);
  const model = useMemo(() => fitWords(CORPUS), []);
  const alpha = ALPHAS[alphaIdx]!;
  const v = classify(model, text, alpha);
  const known = v.words.filter((w) => w.known);
  const verdict = Number.isNaN(v.pSpam) ? 'undefined' : v.pSpam >= 0.5 ? 'spam' : 'not spam';

  return (
    <div className="viz" aria-label="Naive Bayes spam filter">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Message</span>
          <input className="input" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
        </label>
      </div>
      <div className="viz-row">
        {EXAMPLES.map((e) => (
          <button key={e} type="button" className={`btn${e === text ? ' on' : ''}`} onClick={() => setText(e)} style={{ fontSize: '0.8rem' }}>
            {e}
          </button>
        ))}
      </div>
      <div className="viz-row">
        <label className="viz-slider">
          <span>
            Laplace smoothing α <output>{alpha}</output> {alpha === 0 ? '— off: unseen words are "impossible"' : ''}
          </span>
          <input type="range" min={0} max={ALPHAS.length - 1} value={alphaIdx} onChange={(e) => setAlphaIdx(+e.target.value)} />
        </label>
      </div>

      <div>
        <p className="viz-label">Evidence, word by word: log P(word | spam) − log P(word | ham)</p>
        <ul className="viz-bars-h viz-llr">
          <li>
            <span className="k">prior</span>
            <Bar v={v.priorLogOdds} />
            <span className="v">
              {fmt(v.priorLogOdds)} <small className="muted">{model.docs[1]} spam / {model.docs[0]} ham</small>
            </span>
          </li>
          {v.words.map((w, i) => (
            <li key={i} style={w.known ? undefined : { opacity: 0.5 }}>
              <span className="k">{w.word}</span>
              {w.known ? <Bar v={w.llr} /> : <span className="muted" style={{ fontSize: '0.75rem' }}>never seen in training — ignored</span>}
              <span className="v">
                {w.known ? fmt(w.llr) : ''}{' '}
                {w.known && (
                  <small className="muted">
                    {w.counts[1]}× spam, {w.counts[0]}× ham
                  </small>
                )}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <div className={`viz-verdict ${verdict === 'spam' ? 'bad' : verdict === 'not spam' ? 'good' : 'neutral'}`}>
        <span className="icon" aria-hidden="true">
          {verdict === 'spam' ? '✗' : verdict === 'not spam' ? '✓' : '?'}
        </span>
        <span>
          {Number.isNaN(v.pSpam)
            ? 'One word rules out spam and another rules out ham: with α = 0 the model has no answer at all.'
            : `Total log-odds ${fmt(v.logOdds)} → P(spam) = ${(v.pSpam * 100).toFixed(v.pSpam > 0.999 || v.pSpam < 0.001 ? 3 : 1)}%: ${verdict}.`}
        </span>
      </div>

      <div className="viz-stats">
        <Stat k="Training messages" v={String(CORPUS.length)} />
        <Stat k="Vocabulary" v={`${model.vocab} words`} />
        <Stat k="Words used here" v={`${known.length} of ${v.words.length}`} />
        <Stat k="Word order used" v="none" />
      </div>
    </div>
  );
}

/** A diverging bar around a centre line: right = evidence for spam, left = for ham. */
function Bar({ v }: { v: number }) {
  const t = Number.isFinite(v) ? Math.max(-1, Math.min(1, v / BAR)) : Math.sign(v);
  const w = `${(Math.abs(t) * 50).toFixed(1)}%`;
  return (
    <span className="bar llr">
      <span
        style={{
          width: w,
          marginLeft: t < 0 ? `${(50 - Math.abs(t) * 50).toFixed(1)}%` : '50%',
          background: t < 0 ? 'var(--cat-1)' : 'var(--cat-2)',
          borderRadius: 2,
          backgroundImage: Number.isFinite(v) ? undefined : 'repeating-linear-gradient(45deg, transparent 0 3px, var(--surface) 3px 5px)',
        }}
      />
    </span>
  );
}
