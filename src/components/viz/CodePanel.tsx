import { useEffect, useRef } from 'react';
import { tokenize, type CodeCursor, type CodeListing } from '../../lib/code';
import './viz.css';

interface Props {
  /** The listing to show, or a map of listings when the cursor names one. */
  listing: CodeListing | Record<string, CodeListing>;
  /** Position of the current step, or null before anything has run. */
  cursor?: CodeCursor | null;
  /** The previous step's cursor, so values that just changed can be called out. */
  previous?: CodeCursor | null;
  /** Heading above the code, e.g. "merge sort". */
  title?: string;
}

const isListing = (v: Props['listing']): v is CodeListing => Array.isArray((v as CodeListing).lines);

/**
 * The algorithm's source with the executing line highlighted and its variables underneath.
 * Pairs with a `usePlayer` trace: every step says which line it is on (see `src/lib/code.ts`).
 */
export function CodePanel({ listing, cursor, previous, title }: Props) {
  const code = isListing(listing) ? listing : listing[cursor?.listing ?? ''];
  const activeRef = useRef<HTMLLIElement>(null);
  const active = code && cursor ? (code.at[cursor.line] ?? []) : [];
  const first = active[0];

  // Follow the cursor when a long listing scrolls, without moving the page itself.
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [first, cursor?.listing]);

  if (!code) return null;
  const vars = cursor?.vars ? Object.entries(cursor.vars) : [];
  // Only diff against a step in the same listing; across listings every value is "new".
  const before = previous && previous.listing === cursor?.listing ? previous.vars : undefined;

  return (
    <div className="viz-code">
      {title && <p className="viz-label">{title}</p>}
      <ol className="viz-code-lines">
        {code.lines.map((line, i) => {
          const on = active.includes(i);
          return (
            <li key={i} ref={on && i === first ? activeRef : null} className={on ? 'on' : undefined} aria-current={on ? 'step' : undefined}>
              <span className="n" aria-hidden="true">
                {i + 1}
              </span>
              <code>
                {tokenize(line).map((t, k) => (
                  <span key={k} className={`t-${t.kind}`}>
                    {t.text}
                  </span>
                ))}
                {line === '' && '\u00a0'}
              </code>
            </li>
          );
        })}
      </ol>
      {vars.length > 0 && (
        <dl className="viz-vars">
          {vars.map(([k, v]) => (
            <div key={k} className={before && k in before && before[k] !== v ? 'changed' : undefined}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
