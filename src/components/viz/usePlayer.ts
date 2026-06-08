import { useCallback, useEffect, useState } from 'react';

/** Playback speeds in steps per second. */
export const SPEEDS = [0.5, 1, 2, 4, 8, 16, 32, 64];

/** Steps through a precomputed algorithm trace: play/pause, step, scrub, speed. */
export function usePlayer<T>(initial: T[], initialSpeed = 3, startAtEnd = false) {
  const [steps, setSteps] = useState<T[]>(initial);
  const [index, setIndex] = useState(startAtEnd ? Math.max(initial.length - 1, 0) : 0);
  const [playing, setPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(initialSpeed);

  const last = Math.max(steps.length - 1, 0);

  useEffect(() => {
    if (!playing) return;
    if (index >= last) {
      setPlaying(false);
      return;
    }
    const t = window.setTimeout(() => setIndex((i) => Math.min(i + 1, last)), 1000 / SPEEDS[speedIdx]!);
    return () => window.clearTimeout(t);
  }, [playing, index, last, speedIdx]);

  /** Replace the trace (e.g. a new operation was run); optionally start playing, or jump to the end. */
  const load = useCallback((next: T[], autoplay = false, atEnd = false) => {
    setSteps(next);
    setIndex(atEnd ? Math.max(next.length - 1, 0) : 0);
    setPlaying(autoplay && next.length > 1);
  }, []);

  const toggle = () => {
    if (!playing && index >= last) setIndex(0);
    setPlaying((p) => !p);
  };

  return {
    steps,
    step: steps[Math.min(index, last)]!,
    index: Math.min(index, last),
    last,
    playing,
    speedIdx,
    setSpeedIdx,
    load,
    toggle,
    seek: (i: number) => {
      setPlaying(false);
      setIndex(Math.max(0, Math.min(i, last)));
    },
  };
}

export type PlayerState = ReturnType<typeof usePlayer<unknown>>;
