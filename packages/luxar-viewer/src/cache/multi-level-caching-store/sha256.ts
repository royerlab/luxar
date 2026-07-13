/**
 * SHA-256 with a secure-context-independent fallback.
 *
 * The cache layer hashes small inputs — a dataset URL, the raw `.zattrs`
 * bytes — to derive OPFS cache-bucket ids and validation tokens. These are
 * cache keys, NOT a security mechanism.
 *
 * `crypto.subtle` (Web Crypto / SubtleCrypto) is only defined in a *secure
 * context* — HTTPS, `localhost`, or `file:`. Over plain HTTP (a LAN box, a
 * Tailscale IP, an embedded host) it is `undefined`, so calling
 * `crypto.subtle.digest` directly throws `Cannot read properties of
 * undefined (reading 'digest')` and blocks the whole scene load. This also
 * bites inside Web Workers, which inherit the page's (non-)secure context.
 *
 * {@link sha256Hex} uses `crypto.subtle` when it exists and falls back to a
 * vendored pure-JS SHA-256 otherwise, producing BYTE-IDENTICAL output either
 * way so cache-bucket ids stay stable regardless of origin.
 *
 * Why pure JS and not a WASM hash library: the inputs here are tens of bytes
 * to a few KB, hashed a handful of times per session. Pure JS runs at
 * ~50-150 MB/s, i.e. microseconds per call. A WASM path's module fetch +
 * instantiation (~ms) and per-call FFI copy would be strictly slower at these
 * sizes; WASM/SIMD hashing only wins when streaming hundreds of KB to GB,
 * which the viewer never does on this path.
 */

/** Round constants: first 32 bits of the fractional parts of the cube roots
 *  of the first 64 primes. */
const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x: number, n: number): number => (x >>> n) | (x << (32 - n));

/** Lowercase-hex encode a byte array. */
function bytesToHex(bytes: Uint8Array): string {
  let hex = '';
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  return hex;
}

/**
 * Pure-JS SHA-256 of `msg`, returned as 32 raw bytes. Hashes exactly the
 * `msg` view (respecting byteOffset/byteLength). Big-endian throughout, per
 * FIPS 180-4.
 */
function sha256Bytes(msg: Uint8Array): Uint8Array {
  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const len = msg.length;
  // Pad to a multiple of 64 bytes: append 0x80, then zeros, then the 64-bit
  // big-endian bit length. Room needed = len + 1 + 8 bytes.
  const blocks = Math.ceil((len + 9) / 64);
  const total = blocks * 64;
  const buf = new Uint8Array(total);
  buf.set(msg);
  buf[len] = 0x80;

  const dv = new DataView(buf.buffer);
  const bitLen = len * 8;
  // bitLen can exceed 32 bits for large inputs; split via Number math
  // (exact for byte lengths well under 2^53).
  dv.setUint32(total - 8, Math.floor(bitLen / 0x100000000));
  dv.setUint32(total - 4, bitLen >>> 0);

  const w = new Uint32Array(64);
  for (let off = 0; off < total; off += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = dv.getUint32(off + i * 4);
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + K[i] + w[i]) | 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) | 0;
      h = g;
      g = f;
      f = e;
      e = (d + t1) | 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) | 0;
    }

    h0 = (h0 + a) | 0;
    h1 = (h1 + b) | 0;
    h2 = (h2 + c) | 0;
    h3 = (h3 + d) | 0;
    h4 = (h4 + e) | 0;
    h5 = (h5 + f) | 0;
    h6 = (h6 + g) | 0;
    h7 = (h7 + h) | 0;
  }

  const out = new Uint8Array(32);
  const odv = new DataView(out.buffer);
  [h0, h1, h2, h3, h4, h5, h6, h7].forEach((hh, i) => odv.setUint32(i * 4, hh >>> 0));
  return out;
}

/**
 * SHA-256 of `data`, lowercase hex. Uses `crypto.subtle` in a secure context,
 * a vendored pure-JS implementation otherwise. Output is identical in both.
 */
export async function sha256Hex(data: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    // Copy into a fresh Uint8Array (plain ArrayBuffer-backed, exact bytes of
    // the input view). This both satisfies `digest`'s BufferSource type — a
    // view over a possibly-shared ArrayBufferLike is rejected — and hashes
    // exactly `data`'s logical bytes regardless of byteOffset/byteLength.
    const buf = await subtle.digest('SHA-256', new Uint8Array(data));
    return bytesToHex(new Uint8Array(buf));
  }
  return bytesToHex(sha256Bytes(data));
}

/** Exposed for tests: the pure-JS path, independent of `crypto.subtle`. */
export function sha256HexPureJs(data: Uint8Array): string {
  return bytesToHex(sha256Bytes(data));
}
