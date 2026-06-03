// Framework-agnostic hashing helpers shared by the visualizations.

const encoder = new TextEncoder();

/** MurmurHash3 (x86, 32-bit). Returns an unsigned 32-bit integer. */
export function murmur3(key: string, seed = 0): number {
  const bytes = encoder.encode(key);
  const len = bytes.length;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;
  let h = seed >>> 0;

  const blocks = len >> 2;
  for (let i = 0; i < blocks; i++) {
    const o = i * 4;
    let k = bytes[o]! | (bytes[o + 1]! << 8) | (bytes[o + 2]! << 16) | (bytes[o + 3]! << 24);
    k = Math.imul(k, c1);
    k = (k << 15) | (k >>> 17);
    k = Math.imul(k, c2);
    h ^= k;
    h = (h << 13) | (h >>> 19);
    h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }

  const tail = blocks * 4;
  const rem = len & 3;
  let k = 0;
  if (rem === 3) k ^= bytes[tail + 2]! << 16;
  if (rem >= 2) k ^= bytes[tail + 1]! << 8;
  if (rem >= 1) {
    k ^= bytes[tail]!;
    k = Math.imul(k, c1);
    k = (k << 15) | (k >>> 17);
    k = Math.imul(k, c2);
    h ^= k;
  }

  h ^= len;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** 32-bit binary string, zero-padded. */
export function toBinary32(n: number): string {
  return (n >>> 0).toString(2).padStart(32, '0');
}
