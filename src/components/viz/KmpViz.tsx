import { useEffect, useMemo, useState } from 'react';
import { PREFIX_CODE, SEARCH_CODE, kmpSearch, naiveComparisons, type KmpStep } from '../../lib/algorithms/kmp';
import { usePlayer } from './usePlayer';
import { PlayerControls, Swatch } from './PlayerControls';
import { CodePanel } from './CodePanel';
import { Stat } from './Stat';
import './viz.css';

const PRESETS = [
  { label: 'Classic example', text: 'ABABDABACDABABCABAB', pattern: 'ABABCABAB' },
  { label: 'Repetitive (brute force worst case)', text: 'AAAAAAAAAAAAAAAAAAAAAB', pattern: 'AAAAB' },
  { label: 'Overlapping matches', text: 'ABABABABABA', pattern: 'ABA' },
  { label: 'DNA', text: 'GATTAGATTACAGATTACGATTACA', pattern: 'GATTACA' },
];

const MAX_TEXT = 40;
const MAX_PATTERN = 12;

type CellTone = 'match' | 'mismatch' | 'compare' | 'matched' | null;

const toneStyle = (tone: CellTone) => {
  switch (tone) {
    case 'match':
      return { boxShadow: 'inset 0 0 0 2px var(--role-done)', background: 'color-mix(in srgb, var(--role-done) 22%, var(--surface))' };
    case 'mismatch':
      return { boxShadow: 'inset 0 0 0 2px var(--role-change)', background: 'color-mix(in srgb, var(--role-change) 22%, var(--surface))' };
    case 'compare':
      return { boxShadow: 'inset 0 0 0 2px var(--role-compare)' };
    case 'matched':
      return { background: 'color-mix(in srgb, var(--role-done) 16%, var(--surface))' };
    default:
      return undefined;
  }
};

const outcomeTone = (o: KmpStep['outcome']): CellTone => (o === 'mismatch' ? 'mismatch' : o ? 'match' : 'compare');

export default function KmpViz() {
  const [text, setText] = useState(PRESETS[0]!.text);
  const [pattern, setPattern] = useState(PRESETS[0]!.pattern);

  const result = useMemo(() => (pattern && text ? kmpSearch(text, pattern) : null), [text, pattern]);
  const steps = result?.steps ?? [];
  const player = usePlayer<KmpStep>(steps);
  useEffect(() => player.load(steps), [result]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = player.step as KmpStep | undefined;
  const tableComparisons = steps.filter((s) => s.phase === 'table').at(-1)?.comparisons ?? 0;
  const naive = useMemo(() => naiveComparisons(text, pattern), [text, pattern]);

  return (
    <div className="viz" aria-label="KMP string search visualization">
      <div className="viz-row">
        <label className="viz-field grow">
          <span>Text</span>
          <input className="input mono" value={text} maxLength={MAX_TEXT} onChange={(e) => setText(e.target.value)} spellCheck={false} />
        </label>
        <label className="viz-field">
          <span>Pattern</span>
          <input
            className="input mono"
            value={pattern}
            maxLength={MAX_PATTERN}
            onChange={(e) => setPattern(e.target.value)}
            spellCheck={false}
          />
        </label>
        <label className="viz-field">
          <span>Preset</span>
          <select
            className="input"
            value=""
            onChange={(e) => {
              const p = PRESETS[+e.target.value];
              if (p) {
                setText(p.text);
                setPattern(p.pattern);
              }
            }}
          >
            <option value="" disabled>
              Choose…
            </option>
            {PRESETS.map((p, i) => (
              <option key={p.label} value={i}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {!step ? (
        <p className="muted">Enter some text and a pattern.</p>
      ) : (
        <>
          <PlayerControls player={player} message={step.message} />
          <p className="viz-label" style={{ margin: 0 }}>
            {step.phase === 'table' ? 'Phase 1 · build the prefix table π from the pattern' : 'Phase 2 · scan the text'}
          </p>
          <div className="viz-split">
            <div>
              <div className="viz-scroll">
                {step.phase === 'table' ? (
                  <TableView step={step} pattern={pattern} />
                ) : (
                  <SearchView step={step} text={text} pattern={pattern} />
                )}
              </div>

              <div className="viz-legend" style={{ marginTop: '0.75rem' }}>
                <Swatch color="var(--role-done)" label="characters match" />
                <Swatch color="var(--role-change)" label="mismatch" />
                <span>
                  <span className="sw" style={{ background: 'color-mix(in srgb, var(--role-done) 16%, var(--surface))' }} /> already matched
                </span>
                <Swatch color="var(--role-compare)" label="current position" outline />
              </div>
            </div>

            <CodePanel
              listing={{ table: PREFIX_CODE, search: SEARCH_CODE }}
              cursor={step.code}
              previous={player.steps[player.index - 1]?.code}
              title={step.phase === 'table' ? 'Phase 1 · prefix table' : 'Phase 2 · search'}
            />
          </div>

          <div className="viz-stats">
            <Stat k="Matches" v={step.matches.length ? step.matches.join(', ') : '–'} />
            <Stat k="KMP comparisons" v={`${tableComparisons} + ${step.phase === 'search' ? step.comparisons : 0}`} />
            <Stat k="Brute-force comparisons" v={String(naive)} />
            <Stat k="Bound" v={`≤ 2(n+m) = ${2 * (text.length + pattern.length)}`} />
          </div>
        </>
      )}
    </div>
  );
}

function CharRow({
  label,
  cells,
  offset = 0,
  total,
}: {
  label: string;
  cells: { ch: string; tone: CellTone; title?: string }[];
  offset?: number;
  total: number;
}) {
  return (
    <div className="viz-kmp-row">
      <span className="viz-kmp-label">{label}</span>
      <div className="viz-kmp-cells" style={{ gridTemplateColumns: `repeat(${total}, 1.9rem)` }}>
        {cells.map((c, k) => (
          <span key={k} className="viz-kmp-cell" title={c.title} style={{ gridColumn: offset + k + 1, ...toneStyle(c.tone) }}>
            {c.ch}
          </span>
        ))}
      </div>
    </div>
  );
}

function IndexRow({ total }: { total: number }) {
  return (
    <div className="viz-kmp-row">
      <span className="viz-kmp-label" />
      <div className="viz-kmp-cells idx" style={{ gridTemplateColumns: `repeat(${total}, 1.9rem)` }}>
        {Array.from({ length: total }, (_, k) => (
          <span key={k}>{k}</span>
        ))}
      </div>
    </div>
  );
}

function PiRow({ pi, current, total }: { pi: (number | null)[]; current: number | null; total: number }) {
  return (
    <CharRow
      label="π"
      total={total}
      cells={pi.map((v, k) => ({
        ch: v === null ? '·' : String(v),
        tone: k === current ? 'compare' : null,
        title: v === null ? undefined : `π[${k}] = ${v}`,
      }))}
    />
  );
}

/** Phase 1: the pattern against a shifted copy of itself. */
function TableView({ step, pattern }: { step: KmpStep; pattern: string }) {
  const { i, j, outcome } = step;
  const offset = i - j;
  const total = pattern.length;
  const compared = outcome !== null;
  return (
    <div className="viz-kmp">
      <IndexRow total={total} />
      <CharRow
        label="p"
        total={total}
        cells={[...pattern].map((ch, k) => ({
          ch,
          tone: compared && k === i ? outcomeTone(outcome) : k > offset - 1 && k < i && compared ? 'matched' : null,
        }))}
      />
      {compared && (
        <CharRow
          label="prefix"
          total={total}
          offset={offset}
          cells={[...pattern.slice(0, j + 1)].map((ch, k) => ({ ch, tone: k === j ? outcomeTone(outcome) : 'matched' }))}
        />
      )}
      <PiRow pi={step.pi} current={i} total={total} />
    </div>
  );
}

/** Phase 2: the pattern sliding along the text. */
function SearchView({ step, text, pattern }: { step: KmpStep; text: string; pattern: string }) {
  const { i, j, outcome, matches } = step;
  const offset = Math.max(0, i - j);
  const total = Math.max(text.length, offset + pattern.length);
  const inMatch = (k: number) => matches.some((s) => k >= s && k < s + pattern.length);
  return (
    <div className="viz-kmp">
      <IndexRow total={total} />
      <CharRow
        label="text"
        total={total}
        cells={[...text].map((ch, k) => ({
          ch,
          tone: outcome && k === i ? outcomeTone(outcome) : k >= offset && k < i ? 'matched' : null,
          title: inMatch(k) ? 'part of a match' : undefined,
        }))}
      />
      <div className="viz-kmp-row">
        <span className="viz-kmp-label" />
        <div className="viz-kmp-cells" style={{ gridTemplateColumns: `repeat(${total}, 1.9rem)` }}>
          {matches.map((s) => (
            <span key={s} className="viz-kmp-match" style={{ gridColumn: `${s + 1} / span ${pattern.length}` }} title={`match at ${s}`} />
          ))}
        </div>
      </div>
      <CharRow
        label="pattern"
        total={total}
        offset={offset}
        cells={[...pattern].map((ch, k) => ({
          ch,
          tone: outcome && k === j ? outcomeTone(outcome) : k < j ? 'matched' : null,
        }))}
      />
      <CharRow label="π" total={total} offset={offset} cells={step.pi.map((v) => ({ ch: String(v), tone: null }))} />
    </div>
  );
}
