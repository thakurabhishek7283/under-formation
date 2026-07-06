// A tiny decoder-only transformer block (GPT-style, pre-LayerNorm) with seeded random weights.
// Every intermediate tensor is recorded so the forward pass can be stepped through.

import { add, gelu, layerNorm, mapM, matmul, positionalEncoding, randomMatrix, rng, softmax, transpose, type Matrix } from './math';

export const VOCAB = [
  '<unk>', 'the', 'a', 'cat', 'dog', 'sat', 'ran', 'on', 'in', 'mat', 'park', 'big', 'small',
  'bird', 'saw', 'tree', 'and', 'it', 'was', 'happy', 'red', 'ball', 'under', '.',
];
export const D_MODEL = 16;
export const N_HEADS = 2;
export const D_HEAD = D_MODEL / N_HEADS;
export const D_FF = 4 * D_MODEL;

export type StageId =
  | 'tokens' | 'embed' | 'pos' | 'ln1' | 'qkv' | 'attn' | 'attnout' | 'res1'
  | 'ln2' | 'mlp' | 'res2' | 'lnf' | 'logits';

export interface Tensor {
  label: string;
  data: Matrix;
  rowLabels: string[];
  colLabel: string;
  /** diverging for signed activations, sequential for probabilities in [0, 1]. */
  scale: 'diverging' | 'sequential';
}

export interface Stage {
  id: StageId;
  title: string;
  message: string;
  shape: string;
  tensors: Tensor[];
  next?: { token: string; p: number }[];
  tokens?: { text: string; id: number }[];
}

const r = rng(2026);
const W = {
  embed: randomMatrix(VOCAB.length, D_MODEL, 1, r),
  q: Array.from({ length: N_HEADS }, () => randomMatrix(D_MODEL, D_HEAD, 1 / Math.sqrt(D_MODEL), r)),
  k: Array.from({ length: N_HEADS }, () => randomMatrix(D_MODEL, D_HEAD, 1 / Math.sqrt(D_MODEL), r)),
  v: Array.from({ length: N_HEADS }, () => randomMatrix(D_MODEL, D_HEAD, 1 / Math.sqrt(D_MODEL), r)),
  o: randomMatrix(D_MODEL, D_MODEL, 1 / Math.sqrt(D_MODEL), r),
  up: randomMatrix(D_MODEL, D_FF, 1 / Math.sqrt(D_MODEL), r),
  down: randomMatrix(D_FF, D_MODEL, 1 / Math.sqrt(D_FF), r),
};

export function tokenize(text: string): { tokens: string[]; ids: number[] } {
  const words = (text.toLowerCase().match(/[a-z]+|\./g) ?? []).slice(0, 10);
  const ids = words.map((w) => Math.max(0, VOCAB.indexOf(w)));
  return { tokens: ids.map((id, i) => (id === 0 ? `${words[i]}?` : VOCAB[id]!)), ids };
}

export function forward(text: string): Stage[] {
  const { tokens, ids } = tokenize(text);
  if (ids.length === 0) return [];
  const T = ids.length;
  const rows = tokens.map((t, i) => `${i}:${t}`);
  const div = (label: string, data: Matrix, colLabel = `d_model = ${D_MODEL}`): Tensor => ({
    label, data, rowLabels: rows, colLabel, scale: 'diverging',
  });
  const stages: Stage[] = [];

  stages.push({
    id: 'tokens',
    title: 'Tokens → IDs',
    shape: `[${T}]`,
    message: `The tokenizer turns text into ${T} integer IDs from a ${VOCAB.length}-word vocabulary. Words outside it become <unk> (shown with "?"). Real models use BPE with 50k–200k tokens.`,
    tensors: [],
    tokens: tokens.map((text, i) => ({ text, id: ids[i]! })),
  });

  const emb = ids.map((id) => [...W.embed[id]!]);
  stages.push({
    id: 'embed',
    title: 'Token embedding',
    shape: `[${T} × ${D_MODEL}]`,
    message: `Each ID selects one row of the embedding matrix (${VOCAB.length} × ${D_MODEL}). From here on, every token is a vector of ${D_MODEL} numbers. GPT-3 uses 12,288.`,
    tensors: [div('X = E[ids]', emb)],
  });

  const pe = positionalEncoding(T, D_MODEL);
  const x0 = add(emb, pe);
  stages.push({
    id: 'pos',
    title: '+ Positional encoding',
    shape: `[${T} × ${D_MODEL}]`,
    message: 'Attention by itself ignores word order, so a position signal is added. Here that\'s sinusoids of different frequencies; Llama-style models rotate Q and K instead (RoPE). The result starts the residual stream.',
    tensors: [div('PE (position signal)', pe), div('x₀ = X + PE (residual stream)', x0)],
  });

  const ln1 = layerNorm(x0);
  stages.push({
    id: 'ln1',
    title: 'LayerNorm',
    shape: `[${T} × ${D_MODEL}]`,
    message: 'Each row is rescaled to mean 0 and variance 1. Normalizing before each sub-layer ("pre-LN") keeps activations in a stable range, so deep stacks train reliably.',
    tensors: [div('LN(x₀)', ln1)],
  });

  const qs = W.q.map((wq) => matmul(ln1, wq));
  const ks = W.k.map((wk) => matmul(ln1, wk));
  const vs = W.v.map((wv) => matmul(ln1, wv));
  stages.push({
    id: 'qkv',
    title: 'Q, K, V projections',
    shape: `${N_HEADS} heads × [${T} × ${D_HEAD}]`,
    message: `Three learned matrices project every token into a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what do I pass on?"). With ${N_HEADS} heads, each gets its own ${D_HEAD}-dim slice.`,
    tensors: [div('Q (head 1)', qs[0]!, `d_head = ${D_HEAD}`), div('K (head 1)', ks[0]!, `d_head = ${D_HEAD}`), div('V (head 1)', vs[0]!, `d_head = ${D_HEAD}`)],
  });

  const weights = qs.map((q, h) =>
    matmul(q, transpose(ks[h]!)).map((row, i) => softmax(row.map((s, j) => (j > i ? -Infinity : s / Math.sqrt(D_HEAD))))),
  );
  stages.push({
    id: 'attn',
    title: 'Masked attention',
    shape: `${N_HEADS} × [${T} × ${T}]`,
    message: 'Scores = QKᵀ/√d. The causal mask hides future tokens (upper triangle), then softmax turns each row into weights that sum to 1. Row i shows where token i looks.',
    tensors: weights.map((w, h) => ({ label: `head ${h + 1} weights`, data: w, rowLabels: rows, colLabel: 'key position →', scale: 'sequential' as const })),
  });

  const heads = weights.map((w, h) => matmul(w, vs[h]!));
  const concat = heads[0]!.map((_, i) => heads.flatMap((hd) => hd[i]!));
  const attnOut = matmul(concat, W.o);
  stages.push({
    id: 'attnout',
    title: 'Mix values, combine heads',
    shape: `[${T} × ${D_MODEL}]`,
    message: 'Each head outputs a weighted average of its V rows. The heads are concatenated back to d_model and mixed by the output projection W_o.',
    tensors: [div('concat(heads)', concat), div('attention output = concat · W_o', attnOut)],
  });

  const x1 = add(x0, attnOut);
  stages.push({
    id: 'res1',
    title: '+ Residual',
    shape: `[${T} × ${D_MODEL}]`,
    message: 'The attention output is added to the residual stream rather than replacing it. Each layer writes a small edit, and gradients flow straight through the additions during training.',
    tensors: [div('x₁ = x₀ + attention', x1)],
  });

  const ln2 = layerNorm(x1);
  const hidden = mapM(matmul(ln2, W.up), gelu);
  stages.push({
    id: 'ln2',
    title: 'LayerNorm',
    shape: `[${T} × ${D_MODEL}]`,
    message: 'Normalize again before the second sub-layer.',
    tensors: [div('LN(x₁)', ln2)],
  });

  const mlpOut = matmul(hidden, W.down);
  stages.push({
    id: 'mlp',
    title: 'MLP (feed-forward)',
    shape: `[${T} × ${D_FF}] → [${T} × ${D_MODEL}]`,
    message: `Each token, independently, is expanded to ${D_FF} dims, passed through GELU, and projected back. Attention moves information between tokens; the MLP processes it within each token. It holds about two-thirds of a model's parameters.`,
    tensors: [div(`GELU(LN(x₁) · W_up)`, hidden, `d_ff = ${D_FF}`), div('MLP output = hidden · W_down', mlpOut)],
  });

  const x2 = add(x1, mlpOut);
  stages.push({
    id: 'res2',
    title: '+ Residual',
    shape: `[${T} × ${D_MODEL}]`,
    message: 'Add the MLP\'s edit to the stream. That completes one transformer block. A real model repeats it: 12 layers in GPT-2 small, 32 in Llama 3 8B, 96 in GPT-3.',
    tensors: [div('x₂ = x₁ + MLP', x2)],
  });

  const lnf = layerNorm(x2);
  stages.push({
    id: 'lnf',
    title: 'Final LayerNorm',
    shape: `[${T} × ${D_MODEL}]`,
    message: 'After the last block, one more LayerNorm. Only the last row matters for predicting the next token.',
    tensors: [div('LN(x₂)', lnf)],
  });

  const logits = matmul([lnf[T - 1]!], transpose(W.embed))[0]!;
  const probs = softmax(logits);
  const next = probs
    .map((p, i) => ({ token: VOCAB[i]!, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 6);
  stages.push({
    id: 'logits',
    title: 'Unembed → next-token probabilities',
    shape: `[${VOCAB.length}]`,
    message: `The last token's vector is multiplied by the (tied) embedding matrix, giving one logit per vocabulary word, and softmax turns logits into probabilities. These weights are untrained, so the prediction is meaningless. With tied embeddings, an untrained model mostly echoes the current token back. The flow and the shapes are the real thing; training is what makes the numbers mean something.`,
    tensors: [{ label: 'logits (last token)', data: [logits], rowLabels: [rows[T - 1]!], colLabel: `vocabulary (${VOCAB.length})`, scale: 'diverging' }],
    next,
  });

  return stages;
}

export const BLOCK_DIAGRAM: { id: StageId; label: string; kind: 'io' | 'norm' | 'attn' | 'mlp' | 'add' }[] = [
  { id: 'tokens', label: 'Tokens', kind: 'io' },
  { id: 'embed', label: 'Embedding', kind: 'io' },
  { id: 'pos', label: '+ Position', kind: 'add' },
  { id: 'ln1', label: 'LayerNorm', kind: 'norm' },
  { id: 'qkv', label: 'Q · K · V', kind: 'attn' },
  { id: 'attn', label: 'Masked attention', kind: 'attn' },
  { id: 'attnout', label: 'Heads → W_o', kind: 'attn' },
  { id: 'res1', label: '+ Residual', kind: 'add' },
  { id: 'ln2', label: 'LayerNorm', kind: 'norm' },
  { id: 'mlp', label: 'MLP (GELU)', kind: 'mlp' },
  { id: 'res2', label: '+ Residual', kind: 'add' },
  { id: 'lnf', label: 'Final LayerNorm', kind: 'norm' },
  { id: 'logits', label: 'Unembed + softmax', kind: 'io' },
];
