import { useEffect, useMemo, useState } from 'react';
import {
  CONV_CODE,
  IMAGES,
  KERNELS,
  convSteps,
  makeImage,
  outputSize,
  paramCount,
  pool,
  type ConvStep,
  type ImageKind,
  type Padding,
  type PoolKind,
} from '../../../lib/ml/convolution';
import { usePlayer } from '../usePlayer';
import { PlayerControls } from '../PlayerControls';
import { CodePanel } from '../CodePanel';
import { Stat } from '../Stat';
import { Heatmap, ScaleLegend } from './Heatmap';
import '../viz.css';
import './ml.css';

const N = 14;
const fmt = (v: number) => (Math.abs(v) < 0.005 ? '0' : v.toFixed(Math.abs(v) >= 10 ? 0 : 1));

export default function ConvolutionViz() {
  const [imageKind, setImageKind] = useState<ImageKind>('digit');
  const [kernelId, setKernelId] = useState('sobel-x');
  const [weights, setWeights] = useState(() => KERNELS.find((k) => k.id === 'sobel-x')!.weights.map((r) => [...r]));
  const [custom, setCustom] = useState(false);
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState<Padding>('valid');
  const [poolKind, setPoolKind] = useState<PoolKind>('max');

  const preset = KERNELS.find((k) => k.id === kernelId)!;
  const kernel = useMemo(() => ({ ...preset, weights, divisor: custom ? 1 : preset.divisor }), [preset, weights, custom]);
  const img = useMemo(() => makeImage(imageKind, N), [imageKind]);
  const steps = useMemo(() => convSteps(img, kernel, stride, padding), [img, kernel, stride, padding]);

  const player = usePlayer<ConvStep>(steps, 4);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = player.step;
  const k = weights.length;
  const outN = outputSize(N, k, stride, padding);
  const params = paramCount(N, k, outN);

  const chooseKernel = (id: string) => {
    setKernelId(id);
    setWeights(KERNELS.find((kk) => kk.id === id)!.weights.map((r) => [...r]));
    setCustom(false);
  };
  const editWeight = (y: number, x: number, v: number) => {
    setWeights((w) => w.map((row, i) => row.map((c, j) => (i === y && j === x ? v : c))));
    setCustom(true);
  };

  // The completed feature map, for the pooling panel.
  const finalMap = steps[steps.length - 1]!.output.map((r) => r.map((v) => v ?? 0));
  const pooled = useMemo(() => (outN >= 2 ? pool(finalMap, 2, poolKind) : []), [finalMap, outN, poolKind]);

  const outDomain = Math.max(0.3, ...finalMap.flat().map(Math.abs));
  const inWindow = (y: number, x: number) => y >= step.iy && y < step.iy + k && x >= step.ix && x < step.ix + k;

  return (
    <div className="viz" aria-label="Convolution visualization">
      <div className="viz-row">
        <label className="viz-field">
          <span>Input</span>
          <select className="input" value={imageKind} onChange={(e) => setImageKind(e.target.value as ImageKind)}>
            {IMAGES.map((i) => (
              <option key={i.id} value={i.id}>
                {i.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-field grow">
          <span>Kernel</span>
          <select className="input" value={kernelId} onChange={(e) => chooseKernel(e.target.value)}>
            {KERNELS.map((kk) => (
              <option key={kk.id} value={kk.id}>
                {kk.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-field">
          <span>Stride</span>
          <select className="input" value={stride} onChange={(e) => setStride(+e.target.value)}>
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </label>
        <label className="viz-field">
          <span>Padding</span>
          <select className="input" value={padding} onChange={(e) => setPadding(e.target.value as Padding)}>
            <option value="valid">valid (none)</option>
            <option value="same">same (zeros)</option>
          </select>
        </label>
      </div>

      <p className="viz-note">{custom ? 'Custom kernel — edit any weight and the feature map re-computes.' : preset.blurb}</p>

      <PlayerControls player={player} message={step.message} />

      {/* Full width: input, kernel and feature map need to sit side by side to read as a pipeline,
          and there isn't room for all three next to the code panel. */}
      <div className="viz-conv">
        <figure>
          <figcaption className="viz-label">
            Input {N}×{N}
          </figcaption>
          <Heatmap
            data={img}
            scale="sequential"
            domain={1}
            cell="1.15rem"
            label={`input image, ${N} by ${N}`}
            tip={(y, x, v) => `input[${y}][${x}] = ${v.toFixed(2)}`}
            cellStyle={(y, x) =>
              inWindow(y, x) ? { boxShadow: 'inset 0 0 0 2px var(--role-compare)', zIndex: 1, position: 'relative' } : undefined
            }
          />
        </figure>

        <figure className="viz-conv-kernel">
          <figcaption className="viz-label">
            Kernel {k}×{k}
          </figcaption>
          <div className="viz-kernel-grid" style={{ gridTemplateColumns: `repeat(${k}, 2.6rem)` }}>
            {weights.map((row, y) =>
              row.map((v, x) => (
                <input
                  key={`${y}-${x}`}
                  className="input viz-kernel-cell"
                  type="number"
                  step={1}
                  value={v}
                  aria-label={`kernel weight row ${y + 1} column ${x + 1}`}
                  onChange={(e) => editWeight(y, x, Number(e.target.value) || 0)}
                />
              )),
            )}
          </div>
          {kernel.divisor && kernel.divisor !== 1 && <p className="viz-note">÷ {kernel.divisor}</p>}
        </figure>

        <figure>
          <figcaption className="viz-label">
            Feature map {outN}×{outN}
          </figcaption>
          <Heatmap
            data={step.output.map((r) => r.map((v) => v ?? 0))}
            scale="diverging"
            domain={outDomain}
            cell="1.15rem"
            label={`feature map, ${outN} by ${outN}`}
            masked={(y, x) => step.output[y]![x] === null}
            maskedTip="not computed yet"
            tip={(y, x, v) => `out[${y}][${x}] = ${v.toFixed(2)}`}
            cellStyle={(y, x) =>
              y === step.oy && x === step.ox
                ? { boxShadow: 'inset 0 0 0 2px var(--role-change)', zIndex: 1, position: 'relative' }
                : undefined
            }
          />
          <ScaleLegend scale="diverging" domain={outDomain} format={fmt} />
        </figure>
      </div>

      <div className="viz-split">
        {/* The arithmetic behind the single highlighted output cell. */}
        <div>
          <p className="viz-label">This one output cell, in full — patch × kernel, summed</p>
          <div className="viz-mac">
            {step.window.map((row, y) =>
              row.map((v, x) => (
                <span key={`${y}-${x}`} className="viz-mac-term" data-null={v === null ? '' : undefined}>
                  {v === null ? '0*' : v.toFixed(2)} × {weights[y]![x]}
                  <em>{v === null ? '0.00' : (v * weights[y]![x]!).toFixed(2)}</em>
                </span>
              )),
            )}
            <span className="viz-mac-sum">
              = {step.sum.toFixed(2)}
              <em>
                out[{step.oy}][{step.ox}]
              </em>
            </span>
          </div>
          <p className="viz-note">
            0* = outside the image, counted as zero by <code>same</code> padding.
          </p>
        </div>

        <CodePanel listing={CONV_CODE} cursor={step.code} previous={player.steps[player.index - 1]?.code} title="One conv layer" />
      </div>

      <div className="viz-stats">
        <Stat k="Output size" v={`${outN}×${outN}`} />
        <Stat k="Weights in this layer" v={String(params.conv)} />
        <Stat k="Dense layer, same shape" v={params.dense.toLocaleString()} />
        <Stat k="Receptive field" v={`${k}×${k} inputs`} />
      </div>

      {pooled.length > 0 && (
        <div className="viz-conv">
          <figure>
            <figcaption className="viz-label">
              …then {poolKind === 'max' ? 'max' : 'average'}-pool 2×2 → {pooled.length}×{pooled.length}
            </figcaption>
            <Heatmap
              data={pooled}
              scale="diverging"
              domain={outDomain}
              cell="1.6rem"
              showValues={pooled.length <= 8}
              format={fmt}
              label={`pooled map, ${pooled.length} by ${pooled.length}`}
              tip={(y, x, v) => `pooled[${y}][${x}] = ${v.toFixed(2)}`}
            />
          </figure>
          <label className="viz-field">
            <span>Pooling</span>
            <select className="input" value={poolKind} onChange={(e) => setPoolKind(e.target.value as PoolKind)}>
              <option value="max">max</option>
              <option value="avg">average</option>
            </select>
          </label>
        </div>
      )}
    </div>
  );
}
