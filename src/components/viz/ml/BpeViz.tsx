import { useEffect, useMemo, useState } from 'react';
import { encode, trainBpe, type BpeStep } from '../../../lib/ml/bpe';
import { usePlayer } from '../usePlayer';
import { PlayerControls } from '../PlayerControls';
import { Stat } from '../Stat';
import '../viz.css';
import './ml.css';

const CORPORA = [
  {
    label: 'Word families',
    text: 'low low low low low lower lower lower newest newest newest newest newest newest widest widest widest lowest lowest newer newer',
  },
  {
    label: 'English sentence',
    text: 'the cat sat on the mat and the dog sat on the log then the cat and the dog ran to the other mat and sat there together',
  },
  {
    label: 'Code',
    text: 'def add(a, b): return a + b def sub(a, b): return a - b def mul(a, b): return a * b def div(a, b): return a / b',
  },
];

export default function BpeViz() {
  const [corpus, setCorpus] = useState(CORPORA[0]!.text);
  const [maxMerges, setMaxMerges] = useState(20);
  const [probe, setProbe] = useState('lowest newer widest slowest');

  const { merges, steps } = useMemo(() => trainBpe(corpus, maxMerges), [corpus, maxMerges]);
  const player = usePlayer<BpeStep>(steps, 2);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = player.step;
  // Tokenize the probe with the merges learned *so far*, so it tracks the player.
  const mergesSoFar = merges.slice(0, step.vocab.length - steps[0]!.vocab.length);
  const encoded = useMemo(() => encode(probe, mergesSoFar, step.vocab), [probe, mergesSoFar, step.vocab]);
  const probeChars = probe.replace(/\s+/g, '').length + probe.split(/\s+/).filter(Boolean).length;
  const newToken = step.merge?.token;
  const originalTokens = steps[0]!.corpusTokens;

  return (
    <div className="viz" aria-label="Byte-pair encoding visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Training corpus</span>
          <textarea className="input mono" rows={2} value={corpus} onChange={(e) => setCorpus(e.target.value)} spellCheck={false} />
        </label>
      </div>
      <div className="viz-row">
        <label className="viz-field">
          <span>Preset</span>
          <select
            className="input"
            value={CORPORA.findIndex((c) => c.text === corpus)}
            onChange={(e) => setCorpus(CORPORA[+e.target.value]?.text ?? corpus)}
          >
            <option value={-1} disabled>
              Custom
            </option>
            {CORPORA.map((c, i) => (
              <option key={c.label} value={i}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Max merges <output>{maxMerges}</output>
          </span>
          <input type="range" min={1} max={60} value={maxMerges} onChange={(e) => setMaxMerges(+e.target.value)} />
        </label>
      </div>

      <PlayerControls player={player} message={step.message} />

      <div>
        <p className="viz-label">Corpus, segmented with the current vocabulary (×n = word count)</p>
        <div className="viz-bpe-words">
          {step.words.map((w, i) => (
            <span key={i} className="viz-bpe-word">
              {w.symbols.map((s, k) => (
                <span key={k} className={`viz-tok${s === newToken ? ' new' : ''}`}>
                  {s}
                </span>
              ))}
              <small>×{w.freq}</small>
            </span>
          ))}
        </div>
      </div>

      <div>
        <p className="viz-label">
          Vocabulary · {step.vocab.length} tokens ({steps[0]!.vocab.length} characters + {step.vocab.length - steps[0]!.vocab.length} merges)
        </p>
        <div className="viz-chips">
          {step.vocab.map((t, i) => (
            <span key={t} className={`viz-tok small${t === newToken ? ' new' : ''}${i >= steps[0]!.vocab.length ? ' merged' : ''}`} title={`id ${i}`}>
              {t === ' ' ? '␠' : t}
            </span>
          ))}
        </div>
      </div>

      <div className="viz-stats">
        <Stat k="Corpus length (tokens)" v={`${step.corpusTokens} / ${originalTokens}`} />
        <Stat k="Compression" v={`${(originalTokens / step.corpusTokens).toFixed(2)}×`} />
        <Stat k="Vocabulary size" v={String(step.vocab.length)} />
      </div>

      <div className="viz-ops">
        <label className="viz-field grow">
          <span>Tokenize new text with the merges learned so far</span>
          <input className="input mono" value={probe} onChange={(e) => setProbe(e.target.value)} spellCheck={false} />
        </label>
        <div className="viz-chips">
          {encoded.map((t, i) => (
            <span key={i} className={`viz-tok${t.id < 0 ? ' unk' : ''}`} title={t.id < 0 ? 'character never seen in training' : `token id ${t.id}`}>
              {t.text}
              <sub>{t.id < 0 ? '?' : t.id}</sub>
            </span>
          ))}
        </div>
        <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>
          {probeChars} characters → {encoded.length} tokens. Words it has never seen still tokenize, just into smaller pieces.
          {encoded.some((t) => t.id < 0) && ' Characters marked ? were never in the training data; byte-level BPE (GPT-2 onward) avoids this by starting from the 256 byte values.'}
        </p>
      </div>
    </div>
  );
}
