import { useEffect, useMemo, useRef, useState } from 'react';
import { CLASS_DATASETS, cellCentre, computeField, makeClassData, trainTest, type ClassDataset } from '../../../lib/ml/classify';
import { evaluate, forward, initNet, makeTrainer, paramCount, predictProba, trainEpoch, type Activation, type Net, type Trainer } from '../../../lib/ml/mlp';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { ClassLegend, DecisionMap, useClient } from './DecisionMap';
import { cssColor, useThemeTick } from './useThemeTick';
import '../viz.css';
import './ml.css';

const DATASETS: ClassDataset[] = ['circles', 'moons', 'xor', 'spiral', 'three', 'blobs'];
const LRS = [0.001, 0.003, 0.01, 0.03, 0.1];
const RES = 20; // neuron thumbnails
const MAX_POINTS = 400; // chart history before it is thinned

export default function NeuralNetViz({ n = 200 }: { n?: number }) {
  const [data, setData] = useState<ClassDataset>('circles');
  const [act, setAct] = useState<Activation>('tanh');
  const [layers, setLayers] = useState(1);
  const [width, setWidth] = useState(4);
  const [lrIdx, setLrIdx] = useState(2);
  const [seed, setSeed] = useState(1);
  const [running, setRunning] = useState(false);
  const [tick, setTick] = useState(0);
  const client = useClient();

  const classes = CLASS_DATASETS[data].classes;
  const sizes = useMemo(() => [2, ...Array.from({ length: layers }, () => width), classes], [layers, width, classes]);
  const { train, test } = useMemo(() => trainTest(makeClassData(data, n, 4)), [data, n]);

  const trainer = useRef<Trainer | null>(null);
  const history = useRef({ stride: 1, train: [] as number[], test: [] as number[] });
  const net = useMemo<Net>(() => {
    const fresh = initNet(sizes, act, seed);
    trainer.current = makeTrainer(fresh, LRS[lrIdx]!, seed + 100);
    const e = [evaluate(fresh, train).loss, evaluate(fresh, test).loss];
    history.current = { stride: 1, train: [e[0]!], test: [e[1]!] };
    return fresh;
  }, [sizes, act, seed, train, test]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (trainer.current) trainer.current.lr = LRS[lrIdx]!;
  }, [lrIdx]);

  const epochOnce = () => {
    const tr = trainer.current;
    if (!tr) return;
    trainEpoch(tr, train);
    const h = history.current;
    if (tr.epoch % h.stride === 0) {
      h.train.push(evaluate(tr.net, train).loss);
      h.test.push(evaluate(tr.net, test).loss);
      if (h.train.length > MAX_POINTS) {
        // Keep the chart light on long runs: drop every other point and record half as often.
        h.train = h.train.filter((_, i) => i % 2 === 0);
        h.test = h.test.filter((_, i) => i % 2 === 0);
        h.stride *= 2;
      }
    }
  };

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const loop = () => {
      epochOnce();
      setTick((t) => t + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, net, train, test]); // eslint-disable-line react-hooks/exhaustive-deps

  const field = useMemo(() => (client ? computeField((x, y) => predictProba(net, x, y), classes, 56) : null), [client, net, classes, tick]); // eslint-disable-line react-hooks/exhaustive-deps
  const evTrain = evaluate(net, train);
  const evTest = evaluate(net, test);
  const epoch = trainer.current?.epoch ?? 0;
  const reset = () => {
    setRunning(false);
    setSeed((s) => s + 1);
  };

  return (
    <div className="viz" aria-label="Neural network classifier visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Data</span>
          <select className="input" value={data} onChange={(e) => setData(e.target.value as ClassDataset)}>
            {DATASETS.map((d) => (
              <option key={d} value={d}>
                {CLASS_DATASETS[d].name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Hidden layers <output>{layers}</output>
            {layers === 0 ? ' — plain logistic regression' : ''}
          </span>
          <input type="range" min={0} max={3} value={layers} onChange={(e) => setLayers(+e.target.value)} />
        </label>
        <label className="viz-slider">
          <span>
            Neurons per layer <output>{width}</output>
          </span>
          <input type="range" min={1} max={8} value={width} onChange={(e) => setWidth(+e.target.value)} disabled={layers === 0} />
        </label>
        <label className="viz-field">
          <span>Activation</span>
          <select className="input" value={act} onChange={(e) => setAct(e.target.value as Activation)}>
            <option value="tanh">tanh</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">sigmoid</option>
          </select>
        </label>
      </div>
      <div className="viz-row">
        <button type="button" className="btn primary" onClick={() => setRunning((r) => !r)}>
          {running ? '❚❚ Pause' : epoch > 0 ? '▶ Resume training' : '▶ Train'}
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => {
            setRunning(false);
            epochOnce();
            setTick((t) => t + 1);
          }}
        >
          +1 epoch
        </button>
        <button type="button" className="btn" onClick={reset}>
          New weights
        </button>
        <label className="viz-field">
          <span>Learning rate (Adam)</span>
          <select className="input" value={lrIdx} onChange={(e) => setLrIdx(+e.target.value)}>
            {LRS.map((lr, i) => (
              <option key={lr} value={i}>
                {lr}
              </option>
            ))}
          </select>
        </label>
        <span className="muted mono" style={{ fontSize: '0.85rem' }}>
          epoch {epoch}
        </span>
      </div>

      <p className="viz-note">{CLASS_DATASETS[data].blurb}</p>

      <div className="viz-nn">
        <figure>
          <figcaption className="viz-label">The network — each tile is what that neuron computes over the whole plane</figcaption>
          <NetworkCanvas net={net} tick={tick} />
          <div className="viz-legend" style={{ marginTop: '0.5rem' }}>
            <span>
              <span className="sw" style={{ background: 'var(--div-pos)', height: 3 }} /> positive weight
            </span>
            <span>
              <span className="sw" style={{ background: 'var(--div-neg)', height: 3 }} /> negative weight
            </span>
            <span>thicker = larger |w|</span>
            <span>
              <span className="sw" style={{ background: 'linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos))', width: '1.6rem' }} /> tile: neuron
              output low · 0 · high
            </span>
          </div>
        </figure>
        <figure>
          <figcaption className="viz-label">Output: the predicted class everywhere</figcaption>
          <DecisionMap field={field} points={train} test={test} label={`Neural network decision regions after ${epoch} epochs`} />
          <div style={{ marginTop: '0.5rem' }}>
            <ClassLegend classes={classes} />
          </div>
        </figure>
      </div>

      <div className="viz-stats">
        <Stat k="Training loss" v={evTrain.loss.toFixed(3)} />
        <Stat k="Held-out loss" v={evTest.loss.toFixed(3)} />
        <Stat k="Training accuracy" v={`${(evTrain.acc * 100).toFixed(1)}%`} />
        <Stat k="Held-out accuracy" v={`${(evTest.acc * 100).toFixed(1)}%`} />
        <Stat k="Parameters" v={String(paramCount(sizes))} />
      </div>

      <figure style={{ margin: 0 }}>
        <figcaption className="viz-label">Cross-entropy loss while training</figcaption>
        <LineChart
          series={[
            { id: 'train', name: 'training', color: 'var(--cat-1)', values: history.current.train },
            { id: 'test', name: 'held out', color: 'var(--cat-2)', values: history.current.test },
          ]}
          xLabel={history.current.stride === 1 ? 'epoch' : `epoch ÷ ${history.current.stride}`}
          yLabel="loss"
          height={180}
          xMax={Math.max(history.current.train.length, 50)}
        />
      </figure>
    </div>
  );
}

/**
 * The whole network on one canvas: a thumbnail per neuron showing its activation at every point
 * of the plane, and a line per weight coloured by sign and thickened by magnitude.
 */
function NetworkCanvas({ net, tick }: { net: Net; tick: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const theme = useThemeTick();
  const [px, setPx] = useState(0);
  const tallest = Math.max(...net.sizes);
  const TILE = 46;
  const GAP = 12;
  const height = tallest * (TILE + GAP) + GAP + 18;

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setPx(Math.round(el.clientWidth)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !px) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = Math.round(px * dpr);
    c.height = Math.round(height * dpr);
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, px, height);

    const L = net.sizes.length;
    const colX = (l: number) => (L === 1 ? px / 2 : 8 + (l / (L - 1)) * (px - 16 - TILE));
    const rowY = (l: number, i: number) => {
      const count = net.sizes[l]!;
      const total = count * TILE + (count - 1) * GAP;
      return (height - 18 - total) / 2 + i * (TILE + GAP);
    };

    // Every neuron's output over a RES × RES grid of the plane.
    const maps = net.sizes.map((s) => Array.from({ length: s }, () => new Float32Array(RES * RES)));
    for (let row = 0; row < RES; row++)
      for (let col = 0; col < RES; col++) {
        const acts = forward(net, cellCentre(col, RES), 100 - cellCentre(row, RES));
        acts.forEach((a, l) => a.forEach((v, i) => (maps[l]![i]![row * RES + col] = v)));
      }

    const neg = cssColor('--div-neg');
    const mid = cssColor('--div-mid');
    const pos = cssColor('--div-pos');
    const border = cssColor('--border');
    const text3 = cssColor('--text-3');

    // Edges first, underneath the tiles.
    let wMax = 1e-6;
    for (const W of net.W) for (const w of W) wMax = Math.max(wMax, Math.abs(w));
    for (let l = 0; l < L - 1; l++) {
      const nIn = net.sizes[l]!;
      const nOut = net.sizes[l + 1]!;
      for (let o = 0; o < nOut; o++)
        for (let i = 0; i < nIn; i++) {
          const w = net.W[l]![o * nIn + i]!;
          const t = Math.abs(w) / wMax;
          const col = w >= 0 ? pos : neg;
          ctx.strokeStyle = `rgba(${col.join(',')},${(0.15 + 0.75 * t).toFixed(3)})`;
          ctx.lineWidth = 0.5 + 3.5 * t;
          const x1 = colX(l) + TILE;
          const y1 = rowY(l, i) + TILE / 2;
          const x2 = colX(l + 1);
          const y2 = rowY(l + 1, o) + TILE / 2;
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.bezierCurveTo((x1 + x2) / 2, y1, (x1 + x2) / 2, y2, x2, y2);
          ctx.stroke();
        }
    }

    // Tiles. Inputs span about ±2; tanh ±1; sigmoid and softmax 0…1 (centred on ½); ReLU 0…its max.
    const off = document.createElement('canvas');
    off.width = RES;
    off.height = RES;
    const octx = off.getContext('2d')!;
    for (let l = 0; l < L; l++)
      for (let i = 0; i < net.sizes[l]!; i++) {
        const m = maps[l]![i]!;
        const last = l === L - 1;
        const hidden = l > 0 && !last;
        let centre = 0;
        let span = 2;
        if (last || (hidden && net.act === 'sigmoid')) {
          centre = 0.5;
          span = 0.5;
        } else if (hidden && net.act === 'tanh') span = 1;
        else if (hidden && net.act === 'relu') {
          let mx = 1e-6;
          for (const v of m) mx = Math.max(mx, v);
          centre = mx / 2;
          span = mx / 2;
        }
        const img = octx.createImageData(RES, RES);
        for (let k = 0; k < RES * RES; k++) {
          const t = Math.max(-1, Math.min(1, (m[k]! - centre) / span));
          const pole = t < 0 ? neg : pos;
          const a = Math.abs(t);
          img.data[k * 4] = mid[0] + (pole[0] - mid[0]) * a;
          img.data[k * 4 + 1] = mid[1] + (pole[1] - mid[1]) * a;
          img.data[k * 4 + 2] = mid[2] + (pole[2] - mid[2]) * a;
          img.data[k * 4 + 3] = 255;
        }
        octx.putImageData(img, 0, 0);
        const x = colX(l);
        const y = rowY(l, i);
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(off, x, y, TILE, TILE);
        ctx.strokeStyle = `rgb(${border.join(',')})`;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, TILE - 1, TILE - 1);
      }

    ctx.fillStyle = `rgb(${text3.join(',')})`;
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.textAlign = 'center';
    for (let l = 0; l < L; l++) {
      const label = l === 0 ? 'input x, y' : l === L - 1 ? 'softmax' : `hidden ${l}`;
      ctx.fillText(label, colX(l) + TILE / 2, height - 4);
    }
  }, [net, tick, px, height, theme]);

  return (
    <div ref={wrap} style={{ width: '100%' }}>
      <canvas
        ref={canvas}
        style={{ width: '100%', height, display: 'block' }}
        role="img"
        aria-label={`Network with layers ${net.sizes.join(' → ')}; each neuron shows its activation over the input plane`}
      />
    </div>
  );
}
