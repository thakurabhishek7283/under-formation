// Reverse-mode automatic differentiation on a tiny computational graph (micrograd-style):
// one tanh neuron with a squared-error loss. Records every forward and backward step.

export type Op = 'leaf' | 'add' | 'mul' | 'sub' | 'tanh' | 'square';

export interface GraphNode {
  id: string;
  label: string;
  op: Op;
  inputs: string[];
  kind: 'input' | 'param' | 'op' | 'target';
}

/** Topologically ordered: every node comes after its inputs. */
export const NEURON: GraphNode[] = [
  { id: 'x1', label: 'x₁', op: 'leaf', inputs: [], kind: 'input' },
  { id: 'w1', label: 'w₁', op: 'leaf', inputs: [], kind: 'param' },
  { id: 'x2', label: 'x₂', op: 'leaf', inputs: [], kind: 'input' },
  { id: 'w2', label: 'w₂', op: 'leaf', inputs: [], kind: 'param' },
  { id: 'a', label: 'x₁·w₁', op: 'mul', inputs: ['x1', 'w1'], kind: 'op' },
  { id: 'c', label: 'x₂·w₂', op: 'mul', inputs: ['x2', 'w2'], kind: 'op' },
  { id: 's', label: 'sum', op: 'add', inputs: ['a', 'c'], kind: 'op' },
  { id: 'b', label: 'b', op: 'leaf', inputs: [], kind: 'param' },
  { id: 'n', label: 'n', op: 'add', inputs: ['s', 'b'], kind: 'op' },
  { id: 'o', label: 'tanh(n)', op: 'tanh', inputs: ['n'], kind: 'op' },
  { id: 'y', label: 'target y', op: 'leaf', inputs: [], kind: 'target' },
  { id: 'd', label: 'o − y', op: 'sub', inputs: ['o', 'y'], kind: 'op' },
  { id: 'L', label: 'loss', op: 'square', inputs: ['d'], kind: 'op' },
];

export type Values = Record<string, number>;

export interface GradStep {
  phase: 'forward' | 'backward' | 'done';
  data: Values;
  /** Undefined = not computed yet. */
  grad: Partial<Values>;
  /** Node being computed / whose gradient is being pushed back. */
  current: string | null;
  /** Nodes whose value (forward) or gradient (backward) changed this step. */
  touched: string[];
  message: string;
}

const f = (x: number) => (Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(1) : String(Math.round(x * 1000) / 1000));
const paren = (x: number) => (x < 0 ? `(${f(x)})` : f(x));

function evalOp(node: GraphNode, v: Values): number {
  const [p, q] = node.inputs.map((id) => v[id]!);
  switch (node.op) {
    case 'add': return p! + q!;
    case 'mul': return p! * q!;
    case 'sub': return p! - q!;
    case 'tanh': return Math.tanh(p!);
    case 'square': return p! * p!;
    default: return v[node.id]!;
  }
}

/** ∂node/∂input for each input, given current values. */
function localGrads(node: GraphNode, v: Values): number[] {
  const [p, q] = node.inputs.map((id) => v[id]!);
  switch (node.op) {
    case 'add': return [1, 1];
    case 'mul': return [q!, p!];
    case 'sub': return [1, -1];
    case 'tanh': return [1 - Math.tanh(p!) ** 2];
    case 'square': return [2 * p!];
    default: return [];
  }
}

function formula(node: GraphNode, v: Values, byId: Map<string, GraphNode>): string {
  const [p, q] = node.inputs.map((id) => byId.get(id)!.label);
  const [pv, qv] = node.inputs.map((id) => v[id]!);
  switch (node.op) {
    case 'add': return `${p} + ${q} = ${paren(pv!)} + ${paren(qv!)}`;
    case 'mul': return `${p} × ${q} = ${paren(pv!)} × ${paren(qv!)}`;
    case 'sub': return `${paren(pv!)} − ${paren(qv!)}`;
    case 'tanh': return `tanh(${paren(pv!)})`;
    case 'square': return `(${paren(pv!)})²`;
    default: return '';
  }
}

const LOCAL_NAMES: Record<Op, string[]> = {
  leaf: [],
  add: ['1', '1'],
  mul: [],
  sub: ['1', '−1'],
  tanh: ['1 − tanh²(n)'],
  square: ['2·(o − y)'],
};

export function run(graph: GraphNode[], leaves: Values): { steps: GradStep[]; data: Values; grad: Values } {
  const byId = new Map(graph.map((n) => [n.id, n]));
  const data: Values = { ...leaves };
  const grad: Values = {};
  const steps: GradStep[] = [];
  const push = (phase: GradStep['phase'], current: string | null, touched: string[], message: string) =>
    steps.push({ phase, data: { ...data }, grad: { ...grad }, current, touched, message });

  push('forward', null, [], 'Forward pass: compute every node from its inputs, left to right.');
  for (const node of graph) {
    if (node.op === 'leaf') continue;
    const expr = formula(node, data, byId);
    data[node.id] = evalOp(node, data);
    push('forward', node.id, [node.id], `${node.label} = ${expr} = ${f(data[node.id]!)}`);
  }

  const out = graph[graph.length - 1]!;
  for (const node of graph) grad[node.id] = 0;
  grad[out.id] = 1;
  push('backward', out.id, [out.id], `Backward pass. Start at the loss: ∂L/∂L = 1. Now walk the graph in reverse, applying the chain rule at each node.`);

  for (let k = graph.length - 1; k >= 0; k--) {
    const node = graph[k]!;
    if (node.op === 'leaf') continue;
    const locals = localGrads(node, data);
    const parts = node.inputs.map((id, i) => {
      const contribution = locals[i]! * grad[node.id]!;
      grad[id]! += contribution;
      const child = byId.get(id)!;
      const localName = node.op === 'mul' ? byId.get(node.inputs[1 - i]!)!.label : LOCAL_NAMES[node.op][i];
      return `∂L/∂${child.label} += ∂L/∂${node.label} × ${localName} = ${paren(grad[node.id]!)} × ${paren(locals[i]!)} = ${f(contribution)}`;
    });
    push('backward', node.id, node.inputs, parts.join(';  '));
  }

  push(
    'done',
    null,
    graph.filter((n) => n.kind === 'param').map((n) => n.id),
    `Done. Each parameter's gradient says how the loss changes if it is nudged up: w₁ ${f(grad.w1!)}, w₂ ${f(grad.w2!)}, b ${f(grad.b!)}. Gradient descent moves each one the opposite way.`,
  );
  return { steps, data, grad };
}

/** Loss only, for training loops and numerical gradient checks. */
export function loss(graph: GraphNode[], leaves: Values): number {
  const v: Values = { ...leaves };
  for (const node of graph) if (node.op !== 'leaf') v[node.id] = evalOp(node, v);
  return v[graph[graph.length - 1]!.id]!;
}

export const PARAMS = ['w1', 'w2', 'b'] as const;

/** One step of gradient descent on the parameters. */
export function sgdStep(graph: GraphNode[], leaves: Values, lr: number): Values {
  const { grad } = run(graph, leaves);
  const next = { ...leaves };
  for (const p of PARAMS) next[p] = leaves[p]! - lr * grad[p]!;
  return next;
}
