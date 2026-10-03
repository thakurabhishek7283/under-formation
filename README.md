# Under Formation

[![Deploy](https://github.com/thakurabhishek7283/under-formation/actions/workflows/deploy.yml/badge.svg)](https://github.com/thakurabhishek7283/under-formation/actions/workflows/deploy.yml)

My personal site, live at **[thakurabhishek7283.github.io/under-formation](https://thakurabhishek7283.github.io/under-formation/)**.
It has the projects I've built and 30 interactive visualizations of things I've learned, built with
[Astro](https://astro.build). The site is fully static, and the interactive parts are React islands
that only load JavaScript on the pages that use them.

| Section | What's there |
| --- | --- |
| [Projects](https://thakurabhishek7283.github.io/under-formation/projects/) | Tessera, Circuit Forge and Chase, with a live demo embedded where one exists |
| [Visualize](https://thakurabhishek7283.github.io/under-formation/visualize/) | How LLMs work (tokenization, positional encoding, self-attention, the transformer, sampling, the KV cache), efficient LLMs (LoRA, quantization), training and classical ML (backpropagation, gradient descent, k-NN, SVM, decision trees, random forests, a CNN digit recognizer), and data structures and algorithms (Bloom filter, HyperLogLog, segment and Fenwick trees, KMP, sorting) |
| [About](https://thakurabhishek7283.github.io/under-formation/about/) | Experience and skills |

Some details that took the most care:

- **Algorithms are plain TypeScript, separate from the UI.** Step-by-step pages record a trace of
  snapshots, and a code panel highlights the line that produced each step along with the variables in
  scope.
- **Server and browser render identical output.** Random data comes from a seeded generator and every
  number in markup is rounded, and anything that branches on floating-point results is computed after
  hydration, because `Math.exp` and friends can differ in the last bit between Node and a browser.
- **The CNN ships pre-trained weights** that `npm run train:cnn` regenerates deterministically.
- **Accessible charts.** Colour ramps are checked for colour-blind safety, and class is shown by shape as
  well as colour.

## Commands

| Command           | What it does                                   |
| ----------------- | ---------------------------------------------- |
| `npm run dev`     | Dev server at `http://localhost:4321/under-formation/` (drafts visible) |
| `npm run build`   | Production build to `dist/`                    |
| `npm run preview` | Serve the production build locally             |
| `npm run check`   | Type-check `.astro`, `.ts`, `.tsx` and content |
| `npm run train:cnn` | Retrain the digit CNN's shipped weights    |

Site-wide settings (name, tagline, social links, whether the blog is shown) live in `src/consts.ts`.

## Structure

```
src/
  content/
    blog/             *.md / *.mdx  → /blog/<file-name>/
    projects/         *.md / *.mdx  → /projects/<file-name>/
    visualizations/   *.mdx         → /visualize/<file-name>/
  content.config.ts   frontmatter schemas for the three collections
  lib/algorithms/     pure TypeScript algorithms (no UI code)
  components/viz/     React islands that visualize them
  components/         Card, DemoFrame (live iframe demos), etc.
  layouts/            BaseLayout, ArticleLayout
  pages/              routes, incl. rss.xml.ts
  styles/global.css   design tokens (light + dark) and shared styles
```

## Adding a blog post

Create `src/content/blog/my-post.mdx`:

```mdx
---
title: My post
description: One-sentence summary.
pubDate: 2026-10-01
tags: [go, databases]
draft: true   # visible in dev, hidden in production until you remove this
---

Markdown here. You can also import and render components.
```

## Adding a project with a live demo

Create `src/content/projects/my-app.mdx`:

```yaml
---
title: My App
description: What it does.
date: 2026-10-01
tech: [Go, React]
status: live          # live | wip | archived
featured: true        # sorts to the top
repo: https://github.com/you/my-app
demo:
  url: https://my-app.vercel.app
  embed: true         # show it live in an iframe on the page
  height: 600
---
```

The deployed app must allow iframing (no `X-Frame-Options: DENY` / `frame-ancestors 'none'`).
If it can't be framed, set `embed: false` and visitors get an "Open live demo" button instead.

For a demo that runs entirely in the browser, write it as a component and embed it in the MDX
body with `<MyDemo client:visible />`, the same as the visualizations.

## Adding a visualization

Three steps, following the Bloom filter / HyperLogLog examples:

1. **Algorithm**: `src/lib/algorithms/countMinSketch.ts`. Plain TypeScript, no React, so it's
   easy to test and reuse. Put shared helpers (e.g. `murmur3`) in `hash.ts`.
2. **Island**: `src/components/viz/CountMinSketchViz.tsx`. Import `./viz.css` and reuse its
   classes (`viz`, `viz-row`, `viz-slider`, `viz-cells`/`viz-cell`, `viz-stats`, `viz-verdict`)
   and the `<Stat>` tile so every visualization looks consistent. Use the `--seq-0` … `--seq-7`
   color ramp for magnitudes; it already adapts to dark mode.
3. **Page**: `src/content/visualizations/count-min-sketch.mdx`:

   ```mdx
   ---
   title: Count-Min Sketch
   description: Frequency estimates for a stream in sub-linear space.
   pubDate: 2026-10-01
   category: Probabilistic data structures   # groups cards on /visualize
   tags: [hashing, streaming]
   ---

   import CountMinSketchViz from '../../components/viz/CountMinSketchViz.tsx';

   Intro text…

   <CountMinSketchViz client:visible />

   Explanation…
   ```

`client:visible` means the island's JavaScript only loads when it scrolls into view.

### ML / LLM visualizations

These live in `src/lib/ml/` (math, BPE, attention, transformer, sampling, KV cache, autograd,
optimizers, LoRA, quantization) and `src/components/viz/ml/`. Shared building blocks:

- `Heatmap` + `ScaleLegend`: matrices with a `sequential` (one blue ramp) or `diverging`
  (red · gray · blue) scale, hover tooltips, masking and clickable rows.
- `LineChart`: loss curves with hover crosshair, end labels, optional log scale.
- `num.ts`: `pct()` / `r1()`. Round every computed number that goes into markup (widths, SVG
  coordinates). Unrounded floats serialize differently on the server and in the browser and break
  hydration.

Anything "random" (weights, samples shown on first render) must come from the seeded `rng()` in
`math.ts` so the server and browser render the same thing.

Seeding isn't quite enough on its own. `Math.log`, `Math.exp`, `Math.cos`, `Math.tanh` and friends
can differ in the last bit between Node and a browser (Node 24 and Edge already disagree). After
rounding that rarely matters, but an algorithm that *branches* on floats can take a different path.
The SVM solver did exactly that, and hydration failed. Two defences:

- Generate first-render data with plain arithmetic, as the classification datasets do (see
  `makeClassData` in `classify.ts`).
- Compute anything chaotic after hydration: gate it on `useClient()` from `DecisionMap.tsx` and
  render a placeholder on the server.

### Classifiers

The classification pages (k-NN, logistic regression, SVM, Naive Bayes/LDA/QDA, ensembles, neural
network, CNN, comparison) share:

- `classify.ts`: the labelled 2-D datasets (`makeClassData`, identical on every page),
  `trainTest`, `computeField` (a model's class probabilities over a grid) and marching-squares
  contours.
- `DecisionMap`: canvas shading (predicted class, tinted by confidence), the decision boundary,
  and the samples as circle / square / triangle per class, so class never depends on colour
  alone. `ClassLegend` matches it.
- `zoo.ts`: every classifier behind one `fit → proba` interface, for the comparison grid.

The CNN ships pre-trained weights in `src/lib/ml/cnnWeights.json`. `npm run train:cnn` regenerates
them deterministically from `scripts/train-cnn.mjs`; rerun it if you change the architecture or the
digit generator in `cnn.ts`.

### Step-by-step animations

Sorting, the trees and KMP use a **trace** pattern: the algorithm in `src/lib/algorithms/` records a
snapshot plus a one-line `message` at every step, and the component just renders `steps[index]`.

```tsx
const player = usePlayer<MyStep>(steps);       // play / pause / seek / speed
player.load(newSteps, true);                    // after an operation: replace trace and autoplay
<PlayerControls player={player} message={player.step.message} />
```

### The code panel

Sorting, KMP and the trees show the algorithm's source beside the animation, with the executing
line highlighted and the variables in scope underneath. It's what makes these pages teach rather
than just play: the reader can see *which* line produced the change.

Write the listing in the algorithm file with `listing()` from `src/lib/code.ts`, tagging lines with
a trailing `//@ name` marker, then have each step carry a `CodeCursor` naming the line it is on:

```ts
export const MY_CODE = listing(`
  function f(a, lo, hi) {                     //@ fn
    if (lo >= hi) return;   // 1 item: done   //@ base
  }
`);

rec.push('…', { line: 'base', vars: { lo, hi } });
```

```tsx
<div className="viz-split">
  <div>{/* the visual */}</div>
  <CodePanel listing={MY_CODE} cursor={step.code} previous={player.steps[player.index - 1]?.code} title="…" />
</div>
```

Rules that keep it readable:

- **Markers, not line numbers**, so the trace survives reformatting. Repeat a name on two lines to
  highlight a wrapped statement as one unit.
- **Keep display lines to 46 columns.** The panel is a fixed column; longer lines get clipped.
- An algorithm with several operations (build / query / update) exports one listing per operation
  and passes a map; the cursor's `listing` field picks one, and `vars` are only diffed within the
  same listing.
- The listing is *display* code: write the clearest version, not the traced implementation. It may
  differ in structure as long as it doesn't lie about what the step does.

Keep colors consistent with the existing role tokens: `--role-compare` (blue, looking at),
`--role-change` (orange, writing / mismatch), `--role-done` (aqua, settled / accepted). Always pair
a color with a legend entry or text; the colors are validated for color-blind safety, not for
standing alone.

## Deploying

Every push to `main` builds the site and publishes it to GitHub Pages
([`deploy.yml`](.github/workflows/deploy.yml)). The site is served under `/under-formation/`;
`astro.config.mjs` sets that base path and prefixes root-relative links written in Markdown.
