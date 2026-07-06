import { useEffect, useMemo, useState } from 'react';
import { MODELS, formatBytes, generationTrace, kvCacheBytes, type KvStep } from '../../../lib/ml/kvCache';
import { usePlayer } from '../usePlayer';
import { PlayerControls, Swatch } from '../PlayerControls';
import { Stat } from '../Stat';
import { Heatmap, fmt2 } from './Heatmap';
import { pct } from './num';
import '../viz.css';
import './ml.css';

const PROMPT = ['The', 'cat', 'sat'];
const GENERATED = ['on', 'the', 'warm', 'mat', 'and', 'fell', 'asleep', '.'];
const CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

export default function KvCacheViz() {
  const [useCache, setUseCache] = useState(true);
  const steps = useMemo(() => generationTrace(PROMPT, GENERATED, useCache), [useCache]);
  const other = useMemo(() => generationTrace(PROMPT, GENERATED, !useCache), [useCache]);
  const player = usePlayer<KvStep>(steps, 1);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps
  const step = player.step;
  const computed = new Set(step.computed);
  const rowLabels = step.tokens.map((t, i) => `${i} ${t}`);
  const withCache = useCache ? step.work : other[player.index]!.work;
  const withoutCache = useCache ? other[player.index]!.work : step.work;

  const [modelIdx, setModelIdx] = useState(2);
  const [ctxIdx, setCtxIdx] = useState(3);
  const [batch, setBatch] = useState(1);
  const [bytes, setBytes] = useState(2);
  const model = MODELS[modelIdx]!;
  const ctx = CONTEXTS[ctxIdx]!;
  const mem = kvCacheBytes(model, ctx, batch, bytes);
  const memMha = kvCacheBytes({ ...model, kvHeads: model.heads }, ctx, batch, bytes);

  const rowStyle = (i: number) =>
    computed.has(i) ? { boxShadow: 'inset 0 0 0 2px var(--role-change)' } : undefined;

  return (
    <div className="viz" aria-label="KV cache visualization">
      <div className="viz-row">
        <div className="viz-seg" role="radiogroup" aria-label="Cache mode">
          <button type="button" role="radio" aria-checked={useCache} className={`btn${useCache ? ' on' : ''}`} onClick={() => setUseCache(true)}>
            With KV cache
          </button>
          <button type="button" role="radio" aria-checked={!useCache} className={`btn${!useCache ? ' on' : ''}`} onClick={() => setUseCache(false)}>
            Without cache
          </button>
        </div>
      </div>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-prompt">
        {step.tokens.map((t, i) => (
          <span key={i} className={`viz-tok${i < PROMPT.length ? '' : ' merged'}${i === step.tokens.length - 1 ? ' new' : ''}`}>
            {t}
          </span>
        ))}
      </div>

      <div className="viz-kv">
        {(['k', 'v'] as const).map((which) => (
          <figure key={which}>
            <figcaption>
              <strong>{which === 'k' ? 'K cache' : 'V cache'}</strong> <span className="muted">· one row per token</span>
            </figcaption>
            <Heatmap
              data={step[which]}
              scale="diverging"
              domain={2.5}
              rowLabels={rowLabels}
              cell="1.5rem"
              cellStyle={(i) => rowStyle(i)}
              label={`${which.toUpperCase()} cache`}
              tip={(i, j, v) => `${step.tokens[i]}, dim ${j}: ${fmt2(v)}${computed.has(i) ? ' (computed this step)' : ' (reused)'}`}
            />
          </figure>
        ))}
        <figure>
          <figcaption>
            <strong>Attention of “{step.tokens.at(-1)}”</strong> <span className="muted">· over every cached key</span>
          </figcaption>
          <ul className="viz-bars-h compact">
            {step.attention.map((w, i) => (
              <li key={i}>
                <span className="k">{step.tokens[i]}</span>
                <span className="bar">
                  <span style={{ width: pct(w) }} />
                </span>
                <span className="v">{(w * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
        </figure>
      </div>

      <div className="viz-legend">
        <Swatch color="var(--role-change)" label="K/V computed this step" outline />
        <span>no outline = read from the cache</span>
      </div>

      <div className="viz-stats">
        <Stat k="K/V rows computed this step" v={String(step.computed.length)} />
        <Stat k="Total so far, with cache" v={String(withCache)} />
        <Stat k="Total so far, without cache" v={String(withoutCache)} />
        <Stat k="Growth" v={useCache ? 'O(n): one row per token' : 'O(n²): re-do the prefix'} />
      </div>

      <div className="viz-ops">
        <p className="viz-label" style={{ margin: 0 }}>
          The price: memory. Real model KV cache size
        </p>
        <div className="viz-row">
          <label className="viz-field">
            <span>Model</span>
            <select className="input" value={modelIdx} onChange={(e) => setModelIdx(+e.target.value)}>
              {MODELS.map((m, i) => (
                <option key={m.name} value={i}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label className="viz-slider">
            <span>
              Context <output>{ctx.toLocaleString('en-US')}</output> tokens
            </span>
            <input type="range" min={0} max={CONTEXTS.length - 1} value={ctxIdx} onChange={(e) => setCtxIdx(+e.target.value)} />
          </label>
          <label className="viz-slider">
            <span>
              Batch <output>{batch}</output>
            </span>
            <input type="range" min={1} max={64} value={batch} onChange={(e) => setBatch(+e.target.value)} />
          </label>
          <label className="viz-field">
            <span>Precision</span>
            <select className="input" value={bytes} onChange={(e) => setBytes(+e.target.value)}>
              <option value={2}>fp16 / bf16 (2 bytes)</option>
              <option value={1}>fp8 / int8 (1 byte)</option>
            </select>
          </label>
        </div>
        <div className="viz-stats">
          <Stat k="KV cache" v={formatBytes(mem)} />
          <Stat k="Per token" v={formatBytes(mem / ctx / batch)} />
          <Stat
            k={model.kvHeads < model.heads ? `Without GQA (${model.heads} KV heads)` : 'KV heads'}
            v={model.kvHeads < model.heads ? formatBytes(memMha) : String(model.kvHeads)}
          />
        </div>
        <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>
          2 (K and V) × {model.layers} layers × {model.kvHeads} KV heads × {model.headDim} dims × {bytes} bytes × {ctx.toLocaleString('en-US')} tokens × {batch}.
          {model.kvHeads < model.heads && ` Grouped-query attention shares each K/V head across ${model.heads / model.kvHeads} query heads, cutting the cache ${model.heads / model.kvHeads}×.`}
        </p>
      </div>
    </div>
  );
}
