import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react';
import {
  CNN_PARAMS,
  CONV,
  DIGITS,
  FILTERS,
  K,
  POOL,
  RECIPE,
  SIZE,
  TEST_SET,
  cnnAccuracy,
  cnnForward,
  cnnPredict,
  cnnStep,
  cnnTrainer,
  deserialize,
  epochOrder,
  initCnn,
  makeDigits,
  rasterize,
  sampleStrokes,
  type CnnTrainer,
  type CnnWeights,
  type DigitSet,
  type Stroke,
} from '../../../lib/ml/cnn';
import { rng } from '../../../lib/ml/math';
import weightsJson from '../../../lib/ml/cnnWeights.json';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { cssColor, useThemeTick } from './useThemeTick';
import '../viz.css';
import './ml.css';

const SEQ = ['--seq-0', '--seq-1', '--seq-2', '--seq-3', '--seq-4', '--seq-5', '--seq-6', '--seq-7'];
const DIV = ['--div-neg', '--div-mid', '--div-pos'];

/** Fit strokes into the middle of the pad so a generated digit is drawn the way a person would. */
function fitToPad(strokes: Stroke[]): Stroke[] {
  const pts = strokes.flat();
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const s = 0.64 / Math.max(x1 - x0, y1 - y0, 1e-6);
  return strokes.map((st) => st.map(([x, y]) => [0.5 + (x - (x0 + x1) / 2) * s, 0.5 + (y - (y0 + y1) / 2) * s] as [number, number]));
}

export default function CnnViz() {
  const [weights, setWeights] = useState<CnnWeights>(() => deserialize(weightsJson));
  const [version, setVersion] = useState(0);
  const [strokes, setStrokes] = useState<Stroke[]>(() => fitToPad(sampleStrokes(5, rng(8))));
  const [trained, setTrained] = useState<'shipped' | 'scratch' | 'training'>('shipped');
  const [progress, setProgress] = useState({ phase: '', done: 0, total: 1, losses: [] as number[] });
  const [sampleSeed, setSampleSeed] = useState(1);
  const stop = useRef(false);

  const input = useMemo(() => rasterize(strokes), [strokes]);
  const acts = useMemo(() => cnnForward(weights, input), [weights, input, version]); // eslint-disable-line react-hooks/exhaustive-deps
  const empty = strokes.length === 0;
  const pred = acts.probs.indexOf(Math.max(...acts.probs));

  const test = useMemo<DigitSet | null>(() => (typeof window === 'undefined' ? null : makeDigits(TEST_SET.samples, TEST_SET.seed, TEST_SET.wild)), []);
  const [testAcc, setTestAcc] = useState<number | null>(null);
  useEffect(() => {
    if (test && trained !== 'training') setTestAcc(cnnAccuracy(weights, test));
  }, [test, weights, version, trained]);

  // A strip of held-out samples, each with the network's guess.
  const samples = useMemo(() => {
    const r = rng(1000 + sampleSeed);
    return Array.from({ length: 10 }, (_, d) => fitToPad(sampleStrokes(d, r, TEST_SET.wild)));
  }, [sampleSeed]);

  const retrain = () => {
    if (trained === 'training') {
      stop.current = true;
      return;
    }
    stop.current = false;
    setTrained('training');
    const m = initCnn(Date.now() % 100000);
    setWeights(m);
    const images: Float32Array[] = [];
    const labels: number[] = [];
    const r = rng(RECIPE.dataSeed);
    const losses: number[] = [];
    let tr: CnnTrainer | null = null;
    let epoch = 0;
    let order: number[] = [];
    let pos = 0;
    let acc = 0;
    let accN = 0;
    const totalSteps = RECIPE.epochs * Math.ceil(RECIPE.samples / RECIPE.batch);
    let steps = 0;

    const frame = () => {
      if (stop.current) {
        setTrained('scratch');
        setVersion((v) => v + 1);
        return;
      }
      const t0 = performance.now();
      if (images.length < RECIPE.samples) {
        // Generating the training set takes a moment too: do it in slices.
        while (images.length < RECIPE.samples && performance.now() - t0 < 12) {
          const d = images.length % DIGITS;
          images.push(rasterize(sampleStrokes(d, r, RECIPE.wild), 1.2 + r() * 1.1));
          labels.push(d);
        }
        setProgress({ phase: 'Drawing training digits', done: images.length, total: RECIPE.samples, losses: [] });
      } else {
        tr ??= cnnTrainer(m, RECIPE.lr, RECIPE.orderSeed);
        const data = { images, labels };
        while (performance.now() - t0 < 14 && epoch < RECIPE.epochs) {
          if (pos === 0) order = epochOrder(tr, RECIPE.samples);
          acc += cnnStep(tr, data, order.slice(pos, pos + RECIPE.batch));
          accN++;
          steps++;
          pos += RECIPE.batch;
          if (accN === 25) {
            losses.push(acc / accN);
            acc = 0;
            accN = 0;
          }
          if (pos >= RECIPE.samples) {
            pos = 0;
            epoch++;
          }
        }
        setProgress({ phase: `Training, epoch ${Math.min(epoch + 1, RECIPE.epochs)} of ${RECIPE.epochs}`, done: steps, total: totalSteps, losses: [...losses] });
        setVersion((v) => v + 1);
        if (epoch >= RECIPE.epochs) {
          setTrained('scratch');
          return;
        }
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };
  useEffect(() => () => void (stop.current = true), []);

  return (
    <div className="viz" aria-label="Convolutional neural network digit recognizer">
      <div className="viz-cnn-top">
        <div>
          <p className="viz-label">Draw a digit</p>
          <DrawPad strokes={strokes} onChange={setStrokes} />
          <div className="viz-row" style={{ marginTop: '0.5rem' }}>
            <button type="button" className="btn" onClick={() => setStrokes([])}>
              Clear
            </button>
            <button type="button" className="btn" onClick={() => setStrokes(fitToPad(sampleStrokes(Math.floor(Math.random() * 10), rng(Date.now() % 1e6), TEST_SET.wild)))}>
              Random digit
            </button>
          </div>
        </div>

        <div className="viz-cnn-io">
          <figure>
            <figcaption className="viz-label">What the network sees: {SIZE}×{SIZE}</figcaption>
            <MatrixCanvas data={input} w={SIZE} h={SIZE} scale="seq" domain={1} size="8rem" label="Normalized 16 by 16 input image" />
          </figure>
          <div>
            <p className="viz-label">Output: softmax over 10 digits</p>
            <ul className="viz-votes viz-digits">
              {Array.from(acts.probs).map((p, d) => (
                <li key={d} className={!empty && d === pred ? 'win' : undefined}>
                  <span>{d}</span>
                  <span className="bar">
                    <span style={{ width: `${(p * 100).toFixed(1)}%`, background: !empty && d === pred ? 'var(--role-done)' : 'var(--role-compare)' }} />
                  </span>
                  <span className="v">{(p * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className={`viz-verdict ${empty ? 'neutral' : 'good'}`}>
        <span className="icon" aria-hidden="true">
          {empty ? '✎' : pred}
        </span>
        <span>
          {empty
            ? 'Draw a digit in the box — one stroke or several.'
            : `The network reads ${pred === 8 ? 'an' : 'a'} ${pred}, with ${(acts.probs[pred]! * 100).toFixed(1)}% of the probability.`}
        </span>
      </div>

      <div>
        <p className="viz-label">Inside the convolution layer: each learned 3×3 filter, slid over the image, then max-pooled</p>
        <div className="viz-cnn-filters">
          {Array.from({ length: FILTERS }, (_, f) => {
            const kern = weights.k.subarray(f * K * K, (f + 1) * K * K);
            const kmax = Math.max(1e-6, ...Array.from(kern, Math.abs));
            const fmap = acts.conv.subarray(f * CONV * CONV, (f + 1) * CONV * CONV);
            const pmap = acts.pooled.subarray(f * POOL * POOL, (f + 1) * POOL * POOL);
            const fmax = Math.max(1e-6, ...Array.from(fmap));
            return (
              <figure key={f}>
                <figcaption>filter {f + 1}</figcaption>
                <div className="viz-cnn-chain">
                  <MatrixCanvas data={kern} w={K} h={K} scale="div" domain={kmax} size="2.4rem" label={`filter ${f + 1} weights`} />
                  <span aria-hidden="true">→</span>
                  <MatrixCanvas data={fmap} w={CONV} h={CONV} scale="seq" domain={fmax} size="4.4rem" label={`filter ${f + 1} feature map`} />
                  <span aria-hidden="true">→</span>
                  <MatrixCanvas data={pmap} w={POOL} h={POOL} scale="seq" domain={fmax} size="3.2rem" label={`filter ${f + 1} after max-pooling`} />
                </div>
              </figure>
            );
          })}
        </div>
        <div className="viz-legend" style={{ marginTop: '0.6rem' }}>
          <span>
            <span className="sw" style={{ background: 'linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos))', width: '1.6rem' }} /> filter
            weight: negative · 0 · positive
          </span>
          <span>
            <span className="sw" style={{ background: 'linear-gradient(90deg, var(--seq-0), var(--seq-7))', width: '1.6rem' }} /> activation after ReLU: 0 →
            strongest
          </span>
        </div>
      </div>

      <div className="viz-stats">
        <Stat k="Held-out accuracy" v={testAcc === null ? '…' : `${(testAcc * 100).toFixed(1)}%`} />
        <Stat k="Parameters" v={CNN_PARAMS.toLocaleString('en-US')} />
        <Stat k="…of which convolution" v={String(FILTERS * K * K + FILTERS)} />
        <Stat k="Weights" v={trained === 'shipped' ? 'pre-trained' : trained === 'training' ? 'training…' : 'trained here'} />
      </div>

      <div>
        <div className="viz-row" style={{ justifyContent: 'space-between' }}>
          <p className="viz-label" style={{ margin: 0 }}>
            Held-out digits (never trained on) and what the network calls them
          </p>
          <button type="button" className="btn" onClick={() => setSampleSeed((s) => s + 1)}>
            More samples
          </button>
        </div>
        <div className="viz-cnn-samples">
          {samples.map((st, d) => {
            const img = rasterize(st);
            const guess = cnnPredict(weights, img);
            return (
              <button key={`${sampleSeed}-${d}`} type="button" onClick={() => setStrokes(st)} title="Load into the network">
                <MatrixCanvas data={img} w={SIZE} h={SIZE} scale="seq" domain={1} size="3rem" label={`sample ${d}`} />
                <span className={guess === d ? 'ok' : 'bad'}>
                  {guess === d ? '✓' : '✗'} {guess}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="viz-cnn-train">
        <div className="viz-row">
          <button type="button" className="btn primary" onClick={retrain}>
            {trained === 'training' ? '■ Stop' : '↻ Retrain from scratch in your browser'}
          </button>
          <span className="muted" style={{ fontSize: '0.85rem' }}>
            {trained === 'training'
              ? `${progress.phase}: ${Math.round((progress.done / progress.total) * 100)}%`
              : `${RECIPE.samples.toLocaleString('en-US')} generated digits × ${RECIPE.epochs} epochs, Adam, batch ${RECIPE.batch}. Watch the filters form.`}
          </span>
        </div>
        {progress.losses.length > 0 && (
          <LineChart
            series={[{ id: 'loss', name: 'training loss', color: 'var(--cat-1)', values: progress.losses }]}
            xMax={Math.ceil((RECIPE.epochs * RECIPE.samples) / RECIPE.batch / 25)}
            xLabel="× 25 mini-batches"
            yLabel="cross-entropy"
            height={160}
            logY
          />
        )}
      </div>
    </div>
  );
}

/** Freehand drawing surface. Strokes are kept as polylines in [0, 1]², y down. */
function DrawPad({ strokes, onChange }: { strokes: Stroke[]; onChange: (s: Stroke[]) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  // Pointer events can arrive faster than React re-renders; always extend the latest strokes.
  const current = useRef(strokes);
  current.current = strokes;
  const set = (next: Stroke[]) => {
    current.current = next;
    onChange(next);
  };
  const theme = useThemeTick();

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = Math.round(c.clientWidth * dpr);
    c.width = size;
    c.height = size;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, size, size);
    ctx.strokeStyle = `rgb(${cssColor('--text').join(',')})`;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = size * 0.06;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const st of strokes) {
      if (st.length === 1) {
        ctx.beginPath();
        ctx.arc(st[0]![0] * size, st[0]![1] * size, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      st.forEach(([x, y], i) => (i ? ctx.lineTo(x * size, y * size) : ctx.moveTo(x * size, y * size)));
      ctx.stroke();
    }
  }, [strokes, theme]);

  const at = (e: PointerEvent<HTMLCanvasElement>): [number, number] => {
    const r = e.currentTarget.getBoundingClientRect();
    return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  };

  return (
    <canvas
      ref={canvas}
      className="viz-pad"
      aria-label="Drawing pad: draw a digit with the mouse, a finger or a pen"
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        drawing.current = true;
        set([...current.current, [at(e)]]);
      }}
      onPointerMove={(e) => {
        if (!drawing.current) return;
        const p = at(e);
        const last = current.current.at(-1);
        if (!last) return;
        const q = last.at(-1)!;
        if (Math.hypot(p[0] - q[0], p[1] - q[1]) < 0.012) return;
        set([...current.current.slice(0, -1), [...last, p]]);
      }}
      onPointerUp={() => (drawing.current = false)}
      onPointerCancel={() => (drawing.current = false)}
    />
  );
}

/** A small matrix drawn as pixels: one-hue ramp for magnitudes, red–gray–blue for signed values. */
function MatrixCanvas({
  data,
  w,
  h,
  scale,
  domain,
  size,
  label,
}: {
  data: ArrayLike<number>;
  w: number;
  h: number;
  scale: 'seq' | 'div';
  domain: number;
  size: string;
  label: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const theme = useThemeTick();
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(w, h);
    const stops = (scale === 'seq' ? SEQ : DIV).map(cssColor);
    for (let i = 0; i < w * h; i++) {
      const v = data[i]!;
      // Position along the ramp, 0…1.
      const t = scale === 'seq' ? Math.max(0, Math.min(1, v / domain)) : Math.max(0, Math.min(1, 0.5 + v / (2 * domain)));
      const x = t * (stops.length - 1);
      const k = Math.min(Math.floor(x), stops.length - 2);
      const f = x - k;
      const a = stops[k]!;
      const b = stops[k + 1]!;
      img.data[i * 4] = a[0] + (b[0] - a[0]) * f;
      img.data[i * 4 + 1] = a[1] + (b[1] - a[1]) * f;
      img.data[i * 4 + 2] = a[2] + (b[2] - a[2]) * f;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, w, h, scale, domain, theme]);
  return <canvas ref={canvas} className="viz-matrix" style={{ width: size, height: `calc(${size} * ${h / w})` }} role="img" aria-label={label} />;
}
