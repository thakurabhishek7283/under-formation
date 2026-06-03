import { murmur3 } from './hash';

export interface HllStep {
  item: string;
  hash: number;
  register: number;
  /** Position of the leftmost 1-bit in the remaining (32 - p) bits, 1-based. */
  rank: number;
  /** The register's value before this item. */
  previous: number;
  /** Whether this item raised the register's stored maximum. */
  updated: boolean;
}

/**
 * HyperLogLog cardinality estimator (Flajolet et al., 2007) with a 32-bit hash.
 *
 * The first p bits of the hash pick one of m = 2^p registers; each register keeps
 * the largest "leading zeros + 1" seen in the remaining bits. Seeing a long run of
 * zeros is rare, so the maxima tell us roughly how many distinct items went by.
 */
export class HyperLogLog {
  readonly m: number;
  readonly registers: Uint8Array;

  constructor(readonly p: number) {
    if (p < 4 || p > 16) throw new RangeError('p must be between 4 and 16');
    this.m = 1 << p;
    this.registers = new Uint8Array(this.m);
  }

  add(item: string): HllStep {
    const hash = murmur3(item);
    const register = hash >>> (32 - this.p);
    const rest = (hash << this.p) >>> 0;
    const maxRank = 32 - this.p + 1;
    const rank = Math.min(Math.clz32(rest) + 1, maxRank);
    const previous = this.registers[register]!;
    const updated = rank > previous;
    if (updated) this.registers[register] = rank;
    return { item, hash, register, rank, previous, updated };
  }

  estimate(): number {
    const { m } = this;
    let sum = 0;
    let zeros = 0;
    for (const r of this.registers) {
      sum += 2 ** -r;
      if (r === 0) zeros++;
    }
    const raw = (alpha(m) * m * m) / sum;

    // Small-range correction: fall back to linear counting while registers are sparse.
    if (raw <= 2.5 * m && zeros > 0) return m * Math.log(m / zeros);

    // Large-range correction for a 32-bit hash space.
    const two32 = 2 ** 32;
    if (raw > two32 / 30) return -two32 * Math.log(1 - raw / two32);

    return raw;
  }

  /** Expected relative standard error: 1.04 / sqrt(m). */
  get standardError(): number {
    return 1.04 / Math.sqrt(this.m);
  }
}

function alpha(m: number): number {
  if (m === 16) return 0.673;
  if (m === 32) return 0.697;
  if (m === 64) return 0.709;
  return 0.7213 / (1 + 1.079 / m);
}
