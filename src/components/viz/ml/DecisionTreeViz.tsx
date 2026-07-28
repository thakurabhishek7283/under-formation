import { useEffect, useMemo, useState } from 'react';
import {
  TREE_CODE,
  TREE_DATASETS,
  depthCurve,
  makeTreeData,
  treeSteps,
  type Criterion,
  type TreeDataset,
  type TreeNode,
  type TreeStep,
} from '../../../lib/ml/decisionTree';
import { usePlayer } from '../usePlayer';
import { PlayerControls, Swatch } from '../PlayerControls';
import { CodePanel } from '../CodePanel';
import { Stat } from '../Stat';
import { LineChart } from './LineChart';
import { r1 } from './num';
import '../viz.css';
import './ml.css';

const PLOT = 100;
const STRIP = 16; // gain curve gutters, in the same units as the plot
const MAX_DEPTH = 5;
const CLASS_COLOR = ['var(--cat-1)', 'var(--cat-2)'];
const CLASS_NAME = ['A', 'B'];

export default function DecisionTreeViz({ n = 160 }: { n?: number }) {
  const [kind, setKind] = useState<TreeDataset>('blobs');
  const [criterion, setCriterion] = useState<Criterion>('gini');
  const [maxDepth, setMaxDepth] = useState(3);
  const [minSamples, setMinSamples] = useState(8);
  const [seed, setSeed] = useState(11);

  // A fixed 70/30 split, so "held out" means something.
  const { train, test } = useMemo(() => {
    const all = makeTreeData(kind, n, seed);
    const cut = Math.round(all.length * 0.7);
    return { train: all.slice(0, cut), test: all.slice(cut) };
  }, [kind, n, seed]);

  const steps = useMemo(() => treeSteps(train, test, { maxDepth, minSamples, criterion }), [train, test, maxDepth, minSamples, criterion]);
  const curve = useMemo(() => depthCurve(train, test, MAX_DEPTH, { minSamples, criterion }), [train, test, minSamples, criterion]);

  const player = usePlayer<TreeStep>(steps, 2);
  useEffect(() => player.load(steps), [steps]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = player.step;
  const nodes = step.nodes;
  const current = step.current === null ? null : nodes[step.current];
  const dataset = TREE_DATASETS.find((d) => d.id === kind)!;
  const maxGain = Math.max(1e-9, ...(step.candidates ?? []).map((c) => c.gain));
  const leaves = nodes.filter((nd) => nd.leaf);

  return (
    <div className="viz" aria-label="Decision tree visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Data</span>
          <select className="input" value={kind} onChange={(e) => setKind(e.target.value as TreeDataset)}>
            {TREE_DATASETS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.name}
              </option>
            ))}
          </select>
        </label>
        <label className="viz-field">
          <span>Impurity</span>
          <select className="input" value={criterion} onChange={(e) => setCriterion(e.target.value as Criterion)}>
            <option value="gini">Gini</option>
            <option value="entropy">Entropy</option>
          </select>
        </label>
        <label className="viz-slider">
          <span>
            Max depth <output>{maxDepth}</output>
          </span>
          <input type="range" min={1} max={MAX_DEPTH} value={maxDepth} onChange={(e) => setMaxDepth(+e.target.value)} />
        </label>
        <label className="viz-slider">
          <span>
            Min samples <output>{minSamples}</output>
          </span>
          <input type="range" min={2} max={30} value={minSamples} onChange={(e) => setMinSamples(+e.target.value)} />
        </label>
        <button type="button" className="btn" onClick={() => setSeed((s) => s + 1)}>
          New data
        </button>
      </div>

      <p className="viz-note">{dataset.blurb}</p>

      <PlayerControls player={player} message={step.message} />

      <div className="viz-split">
        <div>
          <svg
            className="viz-svg"
            viewBox={`-3 -3 ${PLOT + STRIP + 8} ${PLOT + STRIP + 8}`}
            role="img"
            aria-label="Feature space with the tree's decision regions"
          >
            {/* Leaf regions: what the tree predicts everywhere. */}
            {leaves.map((nd) => (
              <rect
                key={`r${nd.id}`}
                x={r1(nd.bounds.x0)}
                y={r1(PLOT - nd.bounds.y1)}
                width={r1(nd.bounds.x1 - nd.bounds.x0)}
                height={r1(nd.bounds.y1 - nd.bounds.y0)}
                style={{ fill: CLASS_COLOR[nd.prediction], fillOpacity: 0.16 }}
              >
                <title>{`leaf ${nd.id}: predicts ${CLASS_NAME[nd.prediction]} (${nd.counts[0]}A / ${nd.counts[1]}B)`}</title>
              </rect>
            ))}

            {/* Cuts made so far. */}
            {nodes
              .filter((nd) => nd.split)
              .map((nd) => {
                const s = nd.split!;
                const b = nd.bounds;
                const [x1, y1, x2, y2] =
                  s.feature === 0
                    ? [s.threshold, PLOT - b.y0, s.threshold, PLOT - b.y1]
                    : [b.x0, PLOT - s.threshold, b.x1, PLOT - s.threshold];
                return (
                  <line
                    key={`s${nd.id}`}
                    x1={r1(x1)}
                    y1={r1(y1)}
                    x2={r1(x2)}
                    y2={r1(y2)}
                    style={{
                      stroke: 'var(--text)',
                      strokeWidth: nd.id === step.current ? 1.2 : 0.7,
                      opacity: nd.id === step.current ? 1 : 0.5,
                    }}
                  />
                );
              })}

            {/* The node being worked on. */}
            {current && (
              <rect
                x={r1(current.bounds.x0)}
                y={r1(PLOT - current.bounds.y1)}
                width={r1(current.bounds.x1 - current.bounds.x0)}
                height={r1(current.bounds.y1 - current.bounds.y0)}
                style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 1, strokeDasharray: '2 1.5' }}
              />
            )}

            {/* Training samples; the current node's are full strength. */}
            {train.map((s, i) => {
              const inNode = !current || current.indices.includes(i);
              return (
                <circle key={i} cx={r1(s.x)} cy={r1(PLOT - s.y)} r={1.3} style={{ fill: CLASS_COLOR[s.label], opacity: inNode ? 1 : 0.22 }}>
                  <title>{`(${r1(s.x)}, ${r1(s.y)}) → class ${CLASS_NAME[s.label]}`}</title>
                </circle>
              );
            })}

            {/* Gain curves, drawn against the axis they split on: below for x, right for y.
                The peak lines up with the cut the tree is about to make. */}
            <line x1={0} y1={PLOT + 2} x2={PLOT} y2={PLOT + 2} style={{ stroke: 'var(--border)', strokeWidth: 0.4 }} />
            <line x1={PLOT + 2} y1={0} x2={PLOT + 2} y2={PLOT} style={{ stroke: 'var(--border)', strokeWidth: 0.4 }} />
            {step.candidates && (
              <>
                <polyline
                  points={step.candidates
                    .filter((c) => c.feature === 0)
                    .map((c) => `${r1(c.threshold)},${r1(PLOT + 2 + STRIP - (c.gain / maxGain) * STRIP)}`)
                    .join(' ')}
                  style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 0.8 }}
                />
                <polyline
                  points={step.candidates
                    .filter((c) => c.feature === 1)
                    .map((c) => `${r1(PLOT + 2 + (c.gain / maxGain) * STRIP)},${r1(PLOT - c.threshold)}`)
                    .join(' ')}
                  style={{ fill: 'none', stroke: 'var(--accent)', strokeWidth: 0.8 }}
                />
              </>
            )}
            <text x={PLOT / 2} y={PLOT + STRIP + 5} className="viz-svg-mini" style={{ fontSize: 3 }}>
              feature x → {step.candidates ? 'gain per threshold' : ''}
            </text>
            <text
              x={PLOT + 2 + STRIP / 2}
              y={PLOT / 2}
              className="viz-svg-mini"
              style={{ fontSize: 3 }}
              transform={`rotate(-90 ${PLOT + 2 + STRIP / 2} ${PLOT / 2})`}
            >
              feature y →
            </text>
          </svg>

          <div className="viz-legend" style={{ marginTop: '0.5rem' }}>
            <Swatch color={CLASS_COLOR[0]!} label="class A" />
            <Swatch color={CLASS_COLOR[1]!} label="class B" />
            <span>tinted region = what the tree predicts there</span>
            <Swatch color="var(--accent)" label="impurity gain per candidate cut" />
          </div>
        </div>

        <CodePanel
          listing={TREE_CODE}
          cursor={step.code}
          previous={player.steps[player.index - 1]?.code}
          title="Greedy tree growing (CART)"
        />
      </div>

      <div>
        <p className="viz-label">The tree so far</p>
        <div className="viz-scroll">
          <TreeDiagram nodes={nodes} current={step.current} criterion={criterion} />
        </div>
      </div>

      <div className="viz-stats">
        <Stat k="Nodes / leaves" v={`${nodes.length} / ${leaves.length}`} />
        <Stat k="Training accuracy" v={`${(step.trainAcc * 100).toFixed(1)}%`} />
        <Stat k="Held-out accuracy" v={`${(step.testAcc * 100).toFixed(1)}%`} />
        <Stat k={criterion === 'gini' ? 'Gini at this node' : 'Entropy at this node'} v={current ? current.impurity.toFixed(3) : '–'} />
      </div>

      <figure style={{ margin: 0 }}>
        <figcaption className="viz-label">Accuracy vs depth — where fitting turns into memorizing</figcaption>
        <LineChart
          series={[
            { id: 'train', name: 'training', color: 'var(--cat-1)', values: curve.trainAcc },
            { id: 'test', name: 'held out', color: 'var(--cat-2)', values: curve.testAcc },
          ]}
          xLabel={`max depth 1 … ${MAX_DEPTH}   (currently ${maxDepth})`}
          yLabel="accuracy %"
          height={190}
          format={(v) => `${v.toFixed(0)}%`}
        />
      </figure>
    </div>
  );
}

/** Node-link diagram of the tree built so far. Leaves are laid out left to right, parents centred. */
function TreeDiagram({ nodes, current, criterion }: { nodes: TreeNode[]; current: number | null; criterion: Criterion }) {
  if (!nodes.length) return null;
  const W = 96;
  const H = 46;
  const GAP_X = 14;
  const GAP_Y = 34;

  // In-order walk: each leaf takes the next slot, each internal node sits above its children.
  const xs = new Map<number, number>();
  let slot = 0;
  const place = (id: number): number => {
    const nd = nodes[id]!;
    if (nd.leaf || nd.left === null || nd.right === null || !nodes[nd.left] || !nodes[nd.right]) {
      const x = slot++ * (W + GAP_X);
      xs.set(id, x);
      return x;
    }
    const l = place(nd.left);
    const r = place(nd.right);
    const x = (l + r) / 2;
    xs.set(id, x);
    return x;
  };
  place(0);

  const depth = Math.max(...nodes.map((n) => n.depth));
  const width = slot * (W + GAP_X);
  const height = (depth + 1) * (H + GAP_Y);
  const y = (d: number) => d * (H + GAP_Y);

  return (
    <svg
      viewBox={`-6 -6 ${width + 12} ${height + 12}`}
      style={{ minWidth: Math.min(width, 420), width: '100%', maxWidth: width + 12, height: 'auto', display: 'block' }}
      role="img"
      aria-label="The decision tree built so far"
    >
      {nodes.map((nd) => {
        if (nd.parent === null) return null;
        const p = nodes[nd.parent]!;
        const px = xs.get(p.id)! + W / 2;
        const cx = xs.get(nd.id)! + W / 2;
        return (
          <g key={`e${nd.id}`}>
            <line x1={px} y1={y(p.depth) + H} x2={cx} y2={y(nd.depth)} style={{ stroke: 'var(--border)', strokeWidth: 1.5 }} />
            <text x={(px + cx) / 2} y={y(nd.depth) - 5} className="viz-svg-mini" style={{ fontSize: 11 }}>
              {p.left === nd.id ? 'yes' : 'no'}
            </text>
          </g>
        );
      })}
      {nodes.map((nd) => {
        const x = xs.get(nd.id)!;
        const on = nd.id === current;
        return (
          <g key={nd.id}>
            <title>{`node ${nd.id}: ${nd.indices.length} samples, ${criterion} ${nd.impurity.toFixed(3)}`}</title>
            <rect
              x={x}
              y={y(nd.depth)}
              width={W}
              height={H}
              rx={7}
              style={{
                fill: nd.leaf ? `color-mix(in srgb, ${CLASS_COLOR[nd.prediction]} 16%, var(--surface))` : 'var(--surface-2)',
                stroke: on ? 'var(--accent)' : 'var(--border)',
                strokeWidth: on ? 2.5 : 1,
              }}
            />
            <text x={x + W / 2} y={y(nd.depth) + 17} className="viz-svg-mini" style={{ fontSize: 13, fill: 'var(--text)' }}>
              {nd.split
                ? `${nd.split.feature === 0 ? 'x' : 'y'} ≤ ${nd.split.threshold.toFixed(0)}`
                : `predict ${CLASS_NAME[nd.prediction]}`}
            </text>
            <text x={x + W / 2} y={y(nd.depth) + 33} className="viz-svg-mini" style={{ fontSize: 11 }}>
              {nd.counts[0]}A · {nd.counts[1]}B · {nd.impurity.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
