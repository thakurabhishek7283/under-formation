import { useMemo, useState } from 'react';
import { DEMO_D, LLMS, TARGETS, demoUpdate, factor, formatCount, loraParams, singularValues, type LoraTargets } from '../../../lib/ml/lora';
import { Stat } from '../Stat';
import { Heatmap, ScaleLegend, fmt2 } from './Heatmap';
import { pct } from './num';
import '../viz.css';
import './ml.css';

const RANKS = [4, 8, 16, 32, 64];

export default function LoraViz() {
  const [rank, setRank] = useState(2);
  const target = useMemo(() => demoUpdate(), []);
  const sigmas = useMemo(() => singularValues(target), [target]);
  const { a, b, product, relError } = useMemo(() => factor(target, rank), [target, rank]);
  const residual = target.map((row, i) => row.map((v, j) => v - product[i]![j]!));
  const domain = Math.max(...target.flat().map(Math.abs));
  const factorDomain = Math.max(...b.flat().map(Math.abs), ...a.flat().map(Math.abs), 1e-9);
  const energy = sigmas.slice(0, rank).reduce((s, x) => s + x * x, 0) / sigmas.reduce((s, x) => s + x * x, 0);

  const [modelIdx, setModelIdx] = useState(0);
  const [targets, setTargets] = useState<LoraTargets>('qv');
  const [rIdx, setRIdx] = useState(1);
  const model = LLMS[modelIdx]!;
  const r = RANKS[rIdx]!;
  const trainable = loraParams(model, targets, r);
  // Rough training memory, excluding activations: fp16 weights + fp16 grads + fp32 Adam state and master copy.
  const fullGB = (model.params * 16) / 1e9;
  const loraGB = (model.params * 2 + trainable * 16) / 1e9;
  const qloraGB = (model.params * 0.55 + trainable * 16) / 1e9;
  const maxGB = fullGB;

  return (
    <div className="viz" aria-label="LoRA visualization">
      <label className="viz-slider">
        <span>
          Rank r = <output>{rank}</output> · trainable numbers: {2 * DEMO_D * rank} vs {DEMO_D * DEMO_D} for the full ΔW
        </span>
        <input type="range" min={1} max={DEMO_D} value={rank} onChange={(e) => setRank(+e.target.value)} />
      </label>

      <div className="viz-lora">
        <figure>
          <figcaption>
            <strong>ΔW</strong> <span className="muted">{DEMO_D}×{DEMO_D} update we want</span>
          </figcaption>
          <Heatmap data={target} scale="diverging" domain={domain} cell="0.95rem" label="Target update" tip={(i, j, v) => `ΔW[${i},${j}] = ${fmt2(v)}`} />
        </figure>
        <span className="viz-lora-op">≈</span>
        <figure>
          <figcaption>
            <strong>B</strong> <span className="muted">{DEMO_D}×{rank}</span>
          </figcaption>
          <Heatmap data={b} scale="diverging" domain={factorDomain} cell="0.95rem" label="B factor" tip={(i, j, v) => `B[${i},${j}] = ${fmt2(v)}`} />
        </figure>
        <span className="viz-lora-op">×</span>
        <figure>
          <figcaption>
            <strong>A</strong> <span className="muted">{rank}×{DEMO_D}</span>
          </figcaption>
          <Heatmap data={a} scale="diverging" domain={factorDomain} cell="0.95rem" label="A factor" tip={(i, j, v) => `A[${i},${j}] = ${fmt2(v)}`} />
        </figure>
        <span className="viz-lora-op">=</span>
        <figure>
          <figcaption>
            <strong>BA</strong> <span className="muted">rank {rank}</span>
          </figcaption>
          <Heatmap data={product} scale="diverging" domain={domain} cell="0.95rem" label="Low-rank product" tip={(i, j, v) => `BA[${i},${j}] = ${fmt2(v)}`} />
        </figure>
        <figure>
          <figcaption>
            <strong>Error</strong> <span className="muted">ΔW − BA</span>
          </figcaption>
          <Heatmap data={residual} scale="diverging" domain={domain} cell="0.95rem" label="Residual error" tip={(_i, _j, v) => `error = ${fmt2(v)}`} />
        </figure>
      </div>
      <ScaleLegend scale="diverging" domain={domain} />

      <div className="viz-lora-bottom">
        <div>
          <p className="viz-label">Singular values of ΔW: how much of the update lies in each direction</p>
          <ul className="viz-bars-h compact">
            {sigmas.map((s, i) => (
              <li key={i}>
                <span className="k">σ{i + 1}</span>
                <span className="bar">
                  <span style={{ width: pct((s / sigmas[0]!)), background: i < rank ? 'var(--role-compare)' : 'var(--role-idle)' }} />
                </span>
                <span className="v">{s.toFixed(2)}</span>
              </li>
            ))}
          </ul>
          <p className="muted" style={{ fontSize: '0.8rem', margin: '0.3rem 0 0' }}>
            Blue = kept by rank {rank}. Most of the energy sits in the first 3 directions, so a small r captures it.
          </p>
        </div>
        <div className="viz-stats" style={{ alignContent: 'start' }}>
          <Stat k="Relative error ‖ΔW − BA‖/‖ΔW‖" v={`${(relError * 100).toFixed(1)}%`} />
          <Stat k="Energy captured" v={`${(energy * 100).toFixed(1)}%`} />
          <Stat k="Parameters" v={`${2 * DEMO_D * rank} / ${DEMO_D * DEMO_D}`} />
        </div>
      </div>

      <div className="viz-ops">
        <p className="viz-label" style={{ margin: 0 }}>
          At real scale
        </p>
        <div className="viz-row">
          <label className="viz-field">
            <span>Model</span>
            <select className="input" value={modelIdx} onChange={(e) => setModelIdx(+e.target.value)}>
              {LLMS.map((m, i) => (
                <option key={m.name} value={i}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label className="viz-field">
            <span>Adapt</span>
            <select className="input" value={targets} onChange={(e) => setTargets(e.target.value as LoraTargets)}>
              {Object.entries(TARGETS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <label className="viz-slider">
            <span>
              Rank <output>{r}</output>
            </span>
            <input type="range" min={0} max={RANKS.length - 1} value={rIdx} onChange={(e) => setRIdx(+e.target.value)} />
          </label>
        </div>
        <div className="viz-stats">
          <Stat k="Trainable parameters" v={formatCount(trainable)} />
          <Stat k="Share of the model" v={`${((trainable / model.params) * 100).toFixed(trainable / model.params < 0.001 ? 3 : 2)}%`} />
          <Stat k="Adapter file (fp16)" v={`${((trainable * 2) / 1e6).toFixed(0)} MB`} />
        </div>
        <p className="viz-label" style={{ margin: '0.5rem 0 0' }}>
          Rough training memory (weights + gradients + Adam state, excluding activations)
        </p>
        <ul className="viz-bars-h">
          {[
            { k: 'Full fine-tune', v: fullGB },
            { k: 'LoRA (fp16 base)', v: loraGB },
            { k: 'QLoRA (4-bit base)', v: qloraGB },
          ].map((row) => (
            <li key={row.k}>
              <span className="k">{row.k}</span>
              <span className="bar">
                <span style={{ width: pct((row.v / maxGB)) }} />
              </span>
              <span className="v">{row.v.toFixed(0)} GB</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
