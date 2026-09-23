// The experiment behind the blog post "The Answer Comes Before the Words". Runs two small
// open models over 36 labelled support tickets and compares four ways of getting a label out
// of each. Writes src/lib/ml/readoutExperiment.json.
//
//   npm i --no-save @huggingface/transformers
//   node scripts/readout-experiment.mjs [out.json]
//
// Set MODEL_DIR=<dir> to load models from a local folder instead of the Hugging Face Hub.
// Greedy decoding throughout, so reruns give the same labels; timings depend on the machine.
// Avoid dtype 'q8' for these models: 8-bit dynamic quantisation made the 0.5B model worse
// than chance on this task, and slower than fp32 on CPU.

import { AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers';
import { writeFileSync } from 'node:fs';
import os from 'node:os';

const MODELS = [
  { id: 'onnx-community/Qwen2.5-0.5B-Instruct', dtype: 'fp32', name: 'Qwen2.5 0.5B' },
  { id: 'onnx-community/Qwen2.5-1.5B-Instruct', dtype: 'q4', name: 'Qwen2.5 1.5B (4-bit)' },
];
const MAX_NEW_TOKENS = 80;
// ONNX Runtime's default spreads over every logical core; on a hybrid Intel CPU pinning to the
// performance cores was ~2.7x faster. Override with THREADS=<n>.
const THREADS = Number(process.env.THREADS ?? 8);
if (process.env.MODEL_DIR) {
  env.localModelPath = process.env.MODEL_DIR;
  env.allowRemoteModels = false;
}

const LABELS = ['bug', 'feature', 'billing'];
// Full answers scored under the free-form prompt (readout D).
const SENTENCES = { bug: 'This is a bug report.', feature: 'This is a feature request.', billing: 'This is a billing issue.' };

// `ambiguous` marks tickets a reasonable person could file under two labels; the label is my call.
const TICKETS = [
  { label: 'bug', text: 'Export to PDF crashes the app in Safari. Works fine in Chrome.' },
  { label: 'bug', text: "The search bar returns no results for anything with an apostrophe, like O'Brien." },
  { label: 'bug', text: "Since yesterday's update, dark mode resets to light every time I log in." },
  { label: 'bug', text: "Clicking 'Save' on the settings page does nothing. No error, nothing saved." },
  { label: 'bug', text: 'Uploading a PNG over 10MB shows a spinner forever.' },
  { label: 'bug', text: "The dashboard shows yesterday's numbers even after I hit refresh." },
  { label: 'bug', text: 'Notifications arrive twice on Android, once on iOS.' },
  { label: 'bug', text: 'Password reset emails link to a page that says 404.' },
  { label: 'bug', text: 'Dates in the report are off by one day for users in Sydney.' },
  { label: 'bug', text: 'The mobile app logs me out every few minutes.' },
  { label: 'bug', text: 'Pasting a table from Excel scrambles the column order.' },
  { label: 'bug', ambiguous: true, text: "Would be nice if the export didn't crash every time I include images." },
  { label: 'feature', text: 'Can you add a dark mode to the mobile app?' },
  { label: 'feature', text: 'It would be great to export reports as CSV, not just PDF.' },
  { label: 'feature', text: 'Please let us schedule posts for a later date.' },
  { label: 'feature', text: 'We need single sign-on with Okta for our team.' },
  { label: 'feature', text: 'Could the dashboard show week-over-week change?' },
  { label: 'feature', text: 'Add keyboard shortcuts for switching between projects.' },
  { label: 'feature', text: "I'd love an API endpoint for bulk-deleting old records." },
  { label: 'feature', text: 'Can you support two-factor authentication with hardware keys?' },
  { label: 'feature', text: 'Please add a way to mute notifications on weekends.' },
  { label: 'feature', text: 'Let me pin my favourite reports to the top of the sidebar.' },
  { label: 'feature', text: 'Support for Portuguese would help our Brazil office a lot.' },
  { label: 'feature', ambiguous: true, text: 'Search is too slow on big workspaces. Can you add filters to narrow it down?' },
  { label: 'billing', text: 'I was charged twice for order A-104.' },
  { label: 'billing', text: 'Please send me a VAT invoice for last month.' },
  { label: 'billing', text: 'How do I switch from monthly to annual billing?' },
  { label: 'billing', text: "My card was declined but my bank says there's no problem." },
  { label: 'billing', text: 'I cancelled in March but was still charged in April.' },
  { label: 'billing', text: 'Can I get a refund for the unused months on my plan?' },
  { label: 'billing', text: 'Our invoice lists 40 seats but we only have 25 users.' },
  { label: 'billing', text: 'Please update the billing email to finance@example.com.' },
  { label: 'billing', text: 'Why did my price go up from $12 to $15 this month?' },
  { label: 'billing', text: 'Do you offer a discount for nonprofits?' },
  { label: 'billing', ambiguous: true, text: "The student discount code isn't being applied at checkout." },
  { label: 'billing', ambiguous: true, text: 'The checkout page froze and now I see two pending charges.' },
];

const constrainedPrompt = (t, order = LABELS) =>
  `Classify this support ticket as exactly one of: ${order.join(', ')}.\nTicket: "${t}"\nAnswer with one word.`;
const freePrompt = (t) => `Ticket: "${t}"\nIs this a bug report, a feature request, or a billing issue?`;

// The model under test. Set by run() for each entry in MODELS.
let tok, model, variantIds;

const chat = (content) => tok.apply_chat_template([{ role: 'user', content }], { tokenize: false, add_generation_prompt: true });
const ids = (text) => Array.from(tok.encode(text, { add_special_tokens: false }), Number);

function logSoftmax(row) {
  let m = -Infinity;
  for (const x of row) m = Math.max(m, x);
  let z = 0;
  for (const x of row) z += Math.exp(x - m);
  const lz = m + Math.log(z);
  return Float64Array.from(row, (x) => x - lz);
}

/** One forward pass. Returns the log-probabilities at every position. */
async function forward(text) {
  const inputs = tok(text, { add_special_tokens: false });
  const out = await model(inputs);
  const [, L, V] = out.logits.dims;
  const rows = [];
  for (let i = 0; i < L; i++) rows.push(out.logits.data.subarray(i * V, (i + 1) * V));
  return { n: L, logp: (i) => logSoftmax(rows[i]) };
}

// Every casing/spacing variant of a label's first token counts towards that label.
const labelVariants = () =>
  Object.fromEntries(
    LABELS.map((l) => {
      const forms = [l, l[0].toUpperCase() + l.slice(1), ' ' + l, ' ' + l[0].toUpperCase() + l.slice(1)];
      return [l, [...new Set(forms.map((f) => ids(f)[0]))]];
    }),
  );

const r3 = (x) => Math.round(x * 1000) / 1000;
const top = (lp, k) =>
  [...lp.keys()]
    .sort((a, b) => lp[b] - lp[a])
    .slice(0, k)
    .map((i) => ({ token: tok.decode([i]), p: r3(Math.exp(lp[i])) }));

/** Readouts A and C: probability on each label's first token, then renormalised over the labels. */
async function firstToken(prompt) {
  const t0 = performance.now();
  const f = await forward(prompt);
  const lp = f.logp(f.n - 1);
  const ms = performance.now() - t0;
  const raw = LABELS.map((l) => variantIds[l].reduce((s, i) => s + Math.exp(lp[i]), 0));
  const mass = raw.reduce((a, b) => a + b, 0);
  return { ms, mass, probs: raw.map((x) => x / mass), top: top(lp, 5) };
}

/** Readout D: log P(whole answer sentence | prompt) for each label. One pass per candidate. */
async function sentenceScores(prompt) {
  const n0 = ids(prompt).length;
  const t0 = performance.now();
  const scores = [];
  for (const l of LABELS) {
    const full = ids(prompt + SENTENCES[l]);
    if (ids(prompt).some((x, i) => x !== full[i])) throw new Error('prompt tokens changed when the answer was appended');
    const f = await forward(prompt + SENTENCES[l]);
    let s = 0;
    for (let i = n0; i < full.length; i++) s += f.logp(i - 1)[full[i]];
    scores.push(s);
  }
  const ms = performance.now() - t0;
  const m = Math.max(...scores);
  const e = scores.map((s) => Math.exp(s - m));
  const z = e.reduce((a, b) => a + b, 0);
  return { ms, probs: e.map((x) => x / z) };
}

/** Readout B: let the model write its answer, then parse the first label it mentions. */
async function generateAndParse(prompt) {
  const inputs = tok(prompt, { add_special_tokens: false });
  const n0 = inputs.input_ids.dims[1];
  const t0 = performance.now();
  const out = await model.generate({ ...inputs, max_new_tokens: MAX_NEW_TOKENS, do_sample: false });
  const ms = performance.now() - t0;
  const newIds = Array.from(out.data, Number).slice(n0);
  const text = tok.decode(newIds, { skip_special_tokens: true }).trim();
  const hits = LABELS.map((l) => ({ l, at: text.toLowerCase().indexOf(l) })).filter((h) => h.at >= 0);
  hits.sort((a, b) => a.at - b.at);
  return { ms, tokens: newIds.length, text, parsed: hits[0]?.l ?? null };
}

const argmax = (p) => LABELS[p.indexOf(Math.max(...p))];
const r1 = (x) => Math.round(x * 10) / 10;
const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

async function run({ id, dtype, name }) {
  tok = await AutoTokenizer.from_pretrained(id);
  model = await AutoModelForCausalLM.from_pretrained(id, {
    dtype,
    device: 'cpu',
    session_options: { intraOpNumThreads: THREADS, interOpNumThreads: 1 },
  });
  variantIds = labelVariants();
  await forward(chat('warm up')); // the first run pays for graph initialisation
  console.log(`\n== ${name}`);

  const rows = [];
  for (const [i, t] of TICKETS.entries()) {
    const cp = chat(constrainedPrompt(t.text));
    const fp = chat(freePrompt(t.text));
    const A = await firstToken(cp);
    // Control: same question with the labels listed in reverse order.
    const Ar = await firstToken(chat(constrainedPrompt(t.text, [...LABELS].reverse())));
    const C = await firstToken(fp);
    const D = await sentenceScores(fp);
    const B = await generateAndParse(fp);
    const row = {
      text: t.text,
      label: t.label,
      ambiguous: !!t.ambiguous,
      constrained: { probs: A.probs.map(r3), mass: r3(A.mass), top: A.top, pick: argmax(A.probs), ms: Math.round(A.ms) },
      reversed: { probs: Ar.probs.map(r3), pick: argmax(Ar.probs) },
      freeFirst: { probs: C.probs.map(r3), mass: r3(C.mass), top: C.top, pick: argmax(C.probs) },
      sentences: { probs: D.probs.map(r3), pick: argmax(D.probs), ms: Math.round(D.ms) },
      generated: { text: B.text, parsed: B.parsed, tokens: B.tokens, ms: Math.round(B.ms) },
    };
    rows.push(row);
    console.log(
      `${String(i + 1).padStart(2)} ${t.label.padEnd(7)} A=${row.constrained.pick.padEnd(7)} (mass ${row.constrained.mass}) ` +
        `C=${row.freeFirst.pick.padEnd(7)} (mass ${row.freeFirst.mass}) D=${row.sentences.pick.padEnd(7)} B=${String(B.parsed).padEnd(7)} ` +
        `${B.tokens} tok ${row.generated.ms} ms | ${JSON.stringify(B.text.slice(0, 90))}`,
    );
  }

  const pct = (n) => r1((n / rows.length) * 100);
  const count = (f) => rows.filter(f).length;
  const conf = (r) => Math.max(...r.constrained.probs);
  const right = rows.filter((r) => r.constrained.pick === r.label);
  const wrong = rows.filter((r) => r.constrained.pick !== r.label);
  const summary = {
    model: name,
    source: `${id} (${dtype}, CPU)`,
    tickets: rows.length,
    accuracy: {
      constrained: pct(count((r) => r.constrained.pick === r.label)),
      reversed: pct(count((r) => r.reversed.pick === r.label)),
      freeFirst: pct(count((r) => r.freeFirst.pick === r.label)),
      sentences: pct(count((r) => r.sentences.pick === r.label)),
      generated: pct(count((r) => r.generated.parsed === r.label)),
    },
    // Tickets whose one-pass answer changed when only the order of the labels in the prompt changed.
    flippedByOrder: count((r) => r.constrained.pick !== r.reversed.pick),
    picks: Object.fromEntries(
      ['constrained', 'reversed', 'generated'].map((k) => [
        k,
        Object.fromEntries(LABELS.map((l) => [l, count((r) => (k === 'generated' ? r[k].parsed : r[k].pick) === l)])),
      ]),
    ),
    agreeWithGenerated: {
      constrained: pct(count((r) => r.constrained.pick === r.generated.parsed)),
      freeFirst: pct(count((r) => r.freeFirst.pick === r.generated.parsed)),
      sentences: pct(count((r) => r.sentences.pick === r.generated.parsed)),
    },
    // How sure the one-pass readout was, split by whether it turned out right.
    meanConfidence: { right: r3(right.length ? mean(right.map(conf)) : 0), wrong: r3(wrong.length ? mean(wrong.map(conf)) : 0) },
    medianMass: { constrained: r3(median(rows.map((r) => r.constrained.mass))), freeFirst: r3(median(rows.map((r) => r.freeFirst.mass))) },
    medianMs: {
      constrained: median(rows.map((r) => r.constrained.ms)),
      sentences: median(rows.map((r) => r.sentences.ms)),
      generated: median(rows.map((r) => r.generated.ms)),
    },
    medianGeneratedTokens: median(rows.map((r) => r.generated.tokens)),
    unparsed: count((r) => r.generated.parsed === null),
  };
  console.log(JSON.stringify(summary, null, 2));
  await model.dispose();
  return { summary, rows };
}

const results = [];
for (const m of MODELS) results.push(await run(m));

const outPath = process.argv[2] ?? new URL('../src/lib/ml/readoutExperiment.json', import.meta.url);
const machine = `${os.cpus()[0].model.trim()}, ${THREADS} threads`;
writeFileSync(outPath, JSON.stringify({ labels: LABELS, machine, models: results }, null, 1) + '\n');
console.log(`wrote ${outPath}`);
