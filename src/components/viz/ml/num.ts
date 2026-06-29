// Rounded values for markup. Unrounded floats can serialize differently on the server and in the
// browser (e.g. 127.371147280916 vs 127.37114728091598), which breaks hydration.

/** A fraction as a CSS percentage, e.g. 0.4273 → "42.73%". */
export const pct = (x: number) => `${(x * 100).toFixed(2)}%`;

/** Round an SVG coordinate to 0.1 px. */
export const r1 = (x: number) => Math.round(x * 10) / 10;
