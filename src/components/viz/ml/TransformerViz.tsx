import { useEffect, useMemo, useState } from 'react';
import { BLOCK_DIAGRAM, VOCAB, forward, type Stage } from '../../../lib/ml/transformer';
import { usePlayer } from '../usePlayer';
import { PlayerControls } from '../PlayerControls';
import { Heatmap, ScaleLegend, fmt2 } from './Heatmap';
import { pct } from './num';
import '../viz.css';
import './ml.css';

const KIND_COLOR: Record<string, string> = {
  attn: 'var(--role-compare)',
  mlp: 'var(--role-change)',
  norm: 'var(--role-idle)',
  add: 'var(--role-done)',
  io: 'var(--text-3)',
};

export default function TransformerViz() {
  const [text, setText] = useState('the cat sat on the mat');
  const stages = useMemo(() => forward(text), [text]);
  const player = usePlayer<Stage>(stages, 1);
  useEffect(() => player.load(stages), [stages]); // eslint-disable-line react-hooks/exhaustive-deps

  const stage = player.step as Stage | undefined;
  const currentIdx = stage ? BLOCK_DIAGRAM.findIndex((b) => b.id === stage.id) : -1;

  return (
    <div className="viz" aria-label="Transformer forward pass visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Input text (vocabulary: {VOCAB.slice(1).join(' ')})</span>
          <input className="input" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} />
        </label>
      </div>

      {!stage ? (
        <p className="muted">Type a few words.</p>
      ) : (
        <>
          <PlayerControls player={player} message={stage.message} />
          <div className="viz-tf">
            <ol className="viz-tf-diagram" aria-label="Transformer block">
              {BLOCK_DIAGRAM.map((b, i) => {
                const inBlock = i >= 3 && i <= 10;
                return (
                  <li
                    key={b.id + i}
                    className={`${i === currentIdx ? 'on' : ''}${i < currentIdx ? ' done' : ''}${inBlock ? ' in-block' : ''}`}
                    style={{ ['--kind' as string]: KIND_COLOR[b.kind] }}
                  >
                    <button type="button" onClick={() => player.seek(i)}>
                      {b.label}
                    </button>
                  </li>
                );
              })}
            </ol>

            <div className="viz-tf-body">
              <div className="viz-tf-head">
                <h4>{stage.title}</h4>
                <code>{stage.shape}</code>
              </div>

              {stage.tokens && (
                <div className="viz-chips">
                  {stage.tokens.map((t, i) => (
                    <span key={i} className={`viz-tok${t.id === 0 ? ' unk' : ''}`}>
                      {t.text}
                      <sub>{t.id}</sub>
                    </span>
                  ))}
                </div>
              )}

              {stage.tensors.map((t) => {
                const domain = t.scale === 'sequential' ? 1 : Math.max(...t.data.flat().map(Math.abs), 1e-9);
                const wide = (t.data[0]?.length ?? 0) > 24;
                return (
                  <figure key={t.label} className="viz-tf-tensor">
                    <figcaption>
                      <strong>{t.label}</strong> <span className="muted">· columns: {t.colLabel}</span>
                    </figcaption>
                    <div className="viz-scroll">
                      <Heatmap
                        data={t.data}
                        scale={t.scale}
                        domain={domain}
                        rowLabels={t.rowLabels}
                        masked={t.scale === 'sequential' && stage.id === 'attn' ? (i, j) => j > i : undefined}
                        cell={wide ? '0.62rem' : '1.35rem'}
                        label={t.label}
                        tip={(i, j, v) => `${t.rowLabels[i]}, dim ${j}: ${fmt2(v)}`}
                      />
                    </div>
                    <ScaleLegend scale={t.scale} domain={domain} format={t.scale === 'sequential' ? (v) => `${Math.round(v * 100)}%` : fmt2} />
                  </figure>
                );
              })}

              {stage.next && (
                <figure className="viz-tf-tensor">
                  <figcaption>
                    <strong>Next-token probabilities</strong> <span className="muted">· top 6 of {VOCAB.length}</span>
                  </figcaption>
                  <ul className="viz-bars-h">
                    {stage.next.map((n) => (
                      <li key={n.token}>
                        <span className="k">{n.token}</span>
                        <span className="bar">
                          <span style={{ width: pct(n.p) }} />
                        </span>
                        <span className="v">{(n.p * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                  </ul>
                </figure>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
