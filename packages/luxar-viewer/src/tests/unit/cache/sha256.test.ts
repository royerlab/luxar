import { describe, it, expect, afterEach } from 'vitest';
import { createHash } from 'node:crypto';
import { sha256Hex, sha256HexPureJs } from '../../../cache/multi-level-caching-store/sha256';

/**
 * The cache layer hashes small inputs (dataset URL, `.zattrs` bytes) to derive
 * cache keys. `crypto.subtle` only exists in a secure context, so these tests
 * pin down two things:
 *   1. the vendored pure-JS SHA-256 is byte-correct (known FIPS 180-4 vectors
 *      + parity with Node's native SHA-256 over random inputs), and
 *   2. `sha256Hex` returns identical output whether or not `crypto.subtle` is
 *      available — i.e. the secure-context fallback is transparent.
 */

const enc = (s: string): Uint8Array => new TextEncoder().encode(s);

// FIPS 180-4 / standard test vectors.
const VECTORS: Array<[string, string]> = [
  ['', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'],
  ['abc', 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'],
  [
    'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
    '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
  ],
];

describe('sha256HexPureJs (vendored fallback)', () => {
  it('matches known SHA-256 test vectors', () => {
    for (const [input, expected] of VECTORS) {
      expect(sha256HexPureJs(enc(input))).toBe(expected);
    }
  });

  it('crosses the 64-byte block boundary correctly', () => {
    // Lengths that exercise single-block, exactly-full, and multi-block padding.
    for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 200, 1000]) {
      const bytes = new Uint8Array(n);
      for (let i = 0; i < n; i++) bytes[i] = (i * 31 + 7) & 0xff;
      const expected = createHash('sha256').update(Buffer.from(bytes)).digest('hex');
      expect(sha256HexPureJs(bytes)).toBe(expected);
    }
  });

  it('hashes exactly the view (respects byteOffset/byteLength)', () => {
    const backing = new Uint8Array([0xde, 0xad, 0x01, 0x02, 0x03, 0xbe, 0xef]);
    const view = backing.subarray(2, 5); // bytes [0x01, 0x02, 0x03]
    const expected = createHash('sha256')
      .update(Buffer.from([0x01, 0x02, 0x03]))
      .digest('hex');
    expect(sha256HexPureJs(view)).toBe(expected);
  });
});

describe('sha256Hex (subtle fast-path + fallback)', () => {
  const original = globalThis.crypto;

  afterEach(() => {
    // Restore whatever the environment provided.
    Object.defineProperty(globalThis, 'crypto', {
      value: original,
      configurable: true,
      writable: true,
    });
  });

  it('produces correct output when crypto.subtle is unavailable (plain HTTP)', async () => {
    Object.defineProperty(globalThis, 'crypto', {
      value: undefined,
      configurable: true,
      writable: true,
    });
    for (const [input, expected] of VECTORS) {
      expect(await sha256Hex(enc(input))).toBe(expected);
    }
  });

  it('produces identical output with and without crypto.subtle', async () => {
    const inputs = ['https://example.com/scene.luxar.zarr', 'abc', '{"timestamp":42}'];
    // With whatever the env provides (jsdom/node may or may not expose subtle).
    const withEnv = await Promise.all(inputs.map((s) => sha256Hex(enc(s))));

    // Forced fallback path.
    Object.defineProperty(globalThis, 'crypto', {
      value: undefined,
      configurable: true,
      writable: true,
    });
    const withFallback = await Promise.all(inputs.map((s) => sha256Hex(enc(s))));

    expect(withFallback).toEqual(withEnv);
    // And both agree with the pure-JS reference.
    expect(withFallback).toEqual(inputs.map((s) => sha256HexPureJs(enc(s))));
  });
});
