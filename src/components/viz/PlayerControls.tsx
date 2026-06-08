import type { KeyboardEvent } from 'react';
import { SPEEDS, type PlayerState } from './usePlayer';

interface Props {
  player: Pick<PlayerState, 'index' | 'last' | 'playing' | 'speedIdx' | 'setSpeedIdx' | 'toggle' | 'seek'>;
  /** Narration for the current step. */
  message: string;
}

/** Transport controls + step narration. Arrow keys step, space plays/pauses (when focused). */
export function PlayerControls({ player, message }: Props) {
  const { index, last, playing, speedIdx, setSpeedIdx, toggle, seek } = player;

  const onKeyDown = (e: KeyboardEvent) => {
    if ((e.target as HTMLElement).tagName === 'INPUT') return;
    if (e.key === 'ArrowRight') seek(index + 1);
    else if (e.key === 'ArrowLeft') seek(index - 1);
    else if (e.key === ' ') toggle();
    else return;
    e.preventDefault();
  };

  return (
    <div className="viz-player" onKeyDown={onKeyDown}>
      <div className="viz-caption" role="status" aria-live="polite">
        <span className="viz-step-no">
          {index + 1}/{last + 1}
        </span>
        <span>{message}</span>
      </div>
      <div className="viz-row">
        <button type="button" className="btn icon" onClick={() => seek(0)} disabled={index === 0} aria-label="First step">
          ⏮
        </button>
        <button type="button" className="btn icon" onClick={() => seek(index - 1)} disabled={index === 0} aria-label="Previous step">
          ◀
        </button>
        <button type="button" className="btn primary play" onClick={toggle} disabled={last === 0}>
          {playing ? '❚❚ Pause' : index >= last && last > 0 ? '↻ Replay' : '▶ Play'}
        </button>
        <button type="button" className="btn icon" onClick={() => seek(index + 1)} disabled={index >= last} aria-label="Next step">
          ▶
        </button>
        <button type="button" className="btn icon" onClick={() => seek(last)} disabled={index >= last} aria-label="Last step">
          ⏭
        </button>
        <input
          className="viz-scrub"
          type="range"
          min={0}
          max={last}
          value={index}
          onChange={(e) => seek(+e.target.value)}
          aria-label="Scrub through steps"
        />
        <label className="viz-speed">
          <span className="visually-hidden">Speed</span>
          <select className="input" value={speedIdx} onChange={(e) => setSpeedIdx(+e.target.value)}>
            {SPEEDS.map((s, i) => (
              <option key={s} value={i}>
                {s}× /s
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}

/** Legend swatch for a role color; identity is always also given in words. */
export function Swatch({ color, label, outline = false }: { color: string; label: string; outline?: boolean }) {
  return (
    <span>
      <span className="sw" style={outline ? { boxShadow: `inset 0 0 0 2px ${color}` } : { background: color }} /> {label}
    </span>
  );
}
