import { murmur3 } from './hash';

/**
 * A Bloom filter over strings.
 *
 * Uses the Kirsch–Mitzenmacher trick: k indexes are derived from two base hashes
 * as g_i(x) = h1(x) + i * h2(x) (mod m), which behaves like k independent hashes.
 * Bits are stored one-per-byte so the visualization can read them directly.
 */
export class BloomFilter {
  readonly bits: Uint8Array;
  private inserted = 0;

  constructor(
    readonly size: number,
    readonly hashCount: number,
  ) {
    this.bits = new Uint8Array(size);
  }

  /** The k bit positions an item maps to (may contain repeats for tiny m). */
  positions(item: string): number[] {
    const h1 = murmur3(item, 0);
    const h2 = (murmur3(item, 0x9747b28c) | 1) >>> 0; // odd, so it never degenerates to one slot
    return Array.from({ length: this.hashCount }, (_, i) => (h1 + i * h2) % this.size);
  }

  add(item: string): number[] {
    const pos = this.positions(item);
    for (const p of pos) this.bits[p] = 1;
    this.inserted++;
    return pos;
  }

  /** `true` means "probably in the set"; `false` means "definitely not". */
  mightContain(item: string): { result: boolean; positions: number[] } {
    const positions = this.positions(item);
    return { result: positions.every((p) => this.bits[p] === 1), positions };
  }

  get count(): number {
    return this.inserted;
  }

  get setBits(): number {
    return this.bits.reduce((sum, b) => sum + b, 0);
  }

  /** Theoretical false-positive probability after n insertions: (1 - e^(-kn/m))^k. */
  expectedFalsePositiveRate(n = this.inserted): number {
    return Math.pow(1 - Math.exp((-this.hashCount * n) / this.size), this.hashCount);
  }
}

/** Optimal number of hash functions for m bits and n expected items: (m/n) ln 2. */
export function optimalHashCount(m: number, n: number): number {
  return Math.max(1, Math.round((m / n) * Math.LN2));
}
