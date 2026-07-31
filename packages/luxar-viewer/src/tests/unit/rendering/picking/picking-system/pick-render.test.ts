/**
 * Direct unit tests for the brightness-weighted majority-voting helper.
 * Pure over its (pixels, pickSize, scratch) inputs — no PickingSystem
 * instance, no GPU, no renderer.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  VOTE_KEY_STRIDE,
  voteWinner,
  type VoteEntry,
} from '../../../../../rendering/picking/picking-system/pick-render';
import {
  configureElementTextureLayout,
  getElementTextureWidth,
  getMaxElementCapacityPerNode,
  resetElementTextureLayoutForTests,
  SPLAT_TEXTURE_LAYOUT,
  POINT_TEXTURE_LAYOUT,
  LINE_TEXTURE_LAYOUT,
} from '../../../../../rendering/element-texture-layout';

const PICK_SIZE = 5;

/** Build a 5×5 RGBA Float32Array, all background by default. */
function buildPixels(): Float32Array {
  return new Float32Array(PICK_SIZE * PICK_SIZE * 4);
}

/**
 * Write raw (r, g, b, a) into pixel slot (x, y). `a` defaults to 0 because
 * it carries the element-id HIGH half — a stray 1.0 there would decode as
 * elementId + 65536. Prefer {@link setHit} for id-based cases.
 */
function setPixel(
  pixels: Float32Array,
  x: number,
  y: number,
  r: number,
  g: number,
  b: number,
  a: number = 0.0
): void {
  const i = (y * PICK_SIZE + x) * 4;
  pixels[i] = r;
  pixels[i + 1] = g;
  pixels[i + 2] = b;
  pixels[i + 3] = a;
}

/**
 * Write a hit the way the pick shaders do: element index split into two
 * 16-bit halves, LOW in `g` and HIGH in `a` (see `luxarElementIdParts`).
 */
function setHit(
  pixels: Float32Array,
  x: number,
  y: number,
  nodeId: number,
  elementId: number,
  brightness: number
): void {
  const hi = Math.floor(elementId / 65536);
  const lo = elementId - hi * 65536;
  setPixel(pixels, x, y, nodeId, lo, brightness, hi);
}

/** The key `voteWinner` uses internally, for scratch-map assertions. */
const voteKey = (nodeId: number, elementId: number): number => nodeId * VOTE_KEY_STRIDE + elementId;

describe('voteWinner', () => {
  let scratch: Map<number, VoteEntry>;

  beforeEach(() => {
    scratch = new Map();
  });

  it('returns null for an all-background buffer (r < 0.5 everywhere)', () => {
    const pixels = buildPixels();
    expect(voteWinner(pixels, PICK_SIZE, scratch)).toBeNull();
  });

  it('returns null when every pixel has r=0 exactly (background sentinel)', () => {
    const pixels = buildPixels();
    // explicit fill of zeros — the constructor already does this, but be explicit
    pixels.fill(0);
    expect(voteWinner(pixels, PICK_SIZE, scratch)).toBeNull();
  });

  it('treats 0.4 as background but 0.5 as a real hit (r >= 0.5 threshold)', () => {
    const belowThreshold = buildPixels();
    setPixel(belowThreshold, 0, 0, 0.4, 7, 0.9);
    expect(voteWinner(belowThreshold, PICK_SIZE, scratch)).toBeNull();

    const atThreshold = buildPixels();
    // r=0.5 → Math.round(0.5) === 1 (banker's rounding away from zero in JS)
    setPixel(atThreshold, 0, 0, 0.5, 7, 0.9);
    const winner = voteWinner(atThreshold, PICK_SIZE, new Map());
    expect(winner).not.toBeNull();
    expect(winner!.nodeId).toBe(1);
  });

  it('a single hit wins with that pixel as the winner', () => {
    const pixels = buildPixels();
    setHit(pixels, 2, 2, 7, 42, 0.8);
    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(winner!.nodeId).toBe(7);
    expect(winner!.elementId).toBe(42);
    // Float32Array narrows brightness; compare with float32-grade tolerance.
    expect(winner!.weight).toBeCloseTo(0.8, 5);
  });

  it('sums brightness weights across pixels with the same (nodeId, elementId)', () => {
    const pixels = buildPixels();
    setHit(pixels, 0, 0, 3, 9, 0.4);
    setHit(pixels, 1, 0, 3, 9, 0.3);
    setHit(pixels, 2, 0, 3, 9, 0.2);
    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(winner!.nodeId).toBe(3);
    expect(winner!.elementId).toBe(9);
    expect(winner!.weight).toBeCloseTo(0.4 + 0.3 + 0.2, 5);
  });

  it('picks the highest summed-brightness ID when multiple IDs are present', () => {
    const pixels = buildPixels();
    // ID (1, 1) gets two pixels totaling 0.5
    setHit(pixels, 0, 0, 1, 1, 0.3);
    setHit(pixels, 1, 0, 1, 1, 0.2);
    // ID (2, 2) gets one pixel of 0.7 → wins
    setHit(pixels, 0, 1, 2, 2, 0.7);
    // ID (3, 3) gets two pixels totaling 0.4
    setHit(pixels, 0, 2, 3, 3, 0.25);
    setHit(pixels, 1, 2, 3, 3, 0.15);

    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(winner!.nodeId).toBe(2);
    expect(winner!.elementId).toBe(2);
    expect(winner!.weight).toBeCloseTo(0.7, 5);
  });

  it('disambiguates by (nodeId, elementId) — same nodeId, different elementId are distinct entries', () => {
    const pixels = buildPixels();
    setHit(pixels, 0, 0, 5, 100, 0.6);
    setHit(pixels, 1, 0, 5, 101, 0.4);
    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(winner!.nodeId).toBe(5);
    expect(winner!.elementId).toBe(100); // 0.6 > 0.4
    expect(winner!.weight).toBeCloseTo(0.6, 5);
    // Scratch should hold both entries
    expect(scratch.size).toBe(2);
  });

  it('clears the votes scratch map at the start (reused across calls)', () => {
    // Pre-populate scratch with stale data
    scratch.set(999_999_999, { nodeId: 99, elementId: 99, weight: 99 });
    expect(scratch.size).toBe(1);

    const pixels = buildPixels();
    setHit(pixels, 0, 0, 1, 1, 0.5);
    const winner = voteWinner(pixels, PICK_SIZE, scratch);

    // Stale entry MUST be gone; only the fresh (1,1) remains
    expect(scratch.size).toBe(1);
    expect(scratch.has(999_999_999)).toBe(false);
    expect(winner!.nodeId).toBe(1);
  });

  it('keys distinct (nodeId, elementId) pairs without collision', () => {
    const pixels = buildPixels();
    setHit(pixels, 0, 0, 1, 0, 0.5);
    setHit(pixels, 1, 0, 2, 0, 0.7);

    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(scratch.size).toBe(2);
    // Same elementId, different nodeId — must stay distinct.
    expect(scratch.has(voteKey(1, 0))).toBe(true);
    expect(scratch.has(voteKey(2, 0))).toBe(true);
    expect(winner!.nodeId).toBe(2);
  });

  it('is DECODE-compatible with the old one-channel encoding below 2^16', () => {
    // The split only engages above 65535: under it the high half is 0, so
    // `g` still holds the whole index and the decode reduces to
    // `0 * 65536 + g` — exactly what the single-channel path produced.
    // DECODE-compatible, not byte-identical: alpha went from a constant
    // 1.0 to the (zero) high half, so a raw buffer dump does differ. What
    // matters is that no decoded id moves, which is why the change cannot
    // perturb ordinary scenes — and why a cross-backend pick difference at
    // small ids has to come from rasterization, not from this encoding.
    for (const elementId of [0, 1, 42, 65_535]) {
      const pixels = buildPixels();
      setHit(pixels, 1, 1, 4, elementId, 0.7);
      // The high channel really is zero — the old decoder read `g` alone.
      const i = (1 * PICK_SIZE + 1) * 4;
      expect(pixels[i + 3], `high half should be 0 for ${elementId}`).toBe(0);
      expect(pixels[i + 1]).toBe(elementId);

      const winner = voteWinner(pixels, PICK_SIZE, new Map());
      expect(winner!.elementId).toBe(elementId);
    }
  });

  it('round-trips an element index ABOVE 2^24 exactly', () => {
    // REGRESSION GUARD. The index used to ride a single f32 channel, whose
    // 24-bit mantissa stops representing consecutive integers at
    // 16,777,216 — yet a node's capacity reaches 2^25 on a 32768-texel
    // device, so picking there resolved to the wrong element. It is now
    // split across g (low 16 bits) and a (high 16 bits).
    for (const elementId of [16_777_216, 16_777_217, 20_000_001, 33_554_431]) {
      const pixels = buildPixels();
      setHit(pixels, 3, 3, 6, elementId, 0.9);
      const winner = voteWinner(pixels, PICK_SIZE, new Map());
      expect(winner!.nodeId).toBe(6);
      expect(winner!.elementId, `elementId ${elementId} did not round-trip`).toBe(elementId);
    }
  });

  it('does not alias across nodes when the element index exceeds 2^24', () => {
    // With the old `nodeId * 2^24 + elementId` key, (1, 2^24) and (2, 0)
    // both hash to 2^25 — two different elements would merge their votes
    // and the winner could be reported under the wrong node.
    const pixels = buildPixels();
    setHit(pixels, 0, 0, 1, 16_777_216, 0.4);
    setHit(pixels, 1, 0, 2, 0, 0.6);

    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(scratch.size, 'votes from different elements were merged').toBe(2);
    expect(winner!.nodeId).toBe(2);
    expect(winner!.elementId).toBe(0);
    expect(winner!.weight).toBeCloseTo(0.6, 5);
  });

  it('keeps the vote key alias-free AND exact across every supported layout', () => {
    // The stride has to clear the largest reachable elementId (or two
    // elements merge votes) while `nodeId * stride + elementId` stays under
    // 2^53 (or adjacent keys round together). Pinned against the LIVE
    // layout maxima so a capacity increase that outgrows the stride fails
    // here instead of silently merging votes in the field.
    try {
      let maxElementId = 0;
      for (const maxTextureSize of [4096, 8192, 16384, 32768]) {
        configureElementTextureLayout(maxTextureSize);
        for (const layout of [SPLAT_TEXTURE_LAYOUT, POINT_TEXTURE_LAYOUT, LINE_TEXTURE_LAYOUT]) {
          // Sanity: the capacity really is width x height / texels.
          expect(getElementTextureWidth(layout)).toBeGreaterThan(0);
          maxElementId = Math.max(maxElementId, getMaxElementCapacityPerNode(layout) - 1);
        }
      }
      expect(maxElementId, 'stride must exceed every reachable elementId').toBeLessThan(
        VOTE_KEY_STRIDE
      );

      // nodeId rides an f32 channel of the pick buffer, so 2^24 is its ceiling.
      const worstKey = (2 ** 24 - 1) * VOTE_KEY_STRIDE + maxElementId;
      expect(worstKey, 'worst-case key must stay exactly representable').toBeLessThan(
        Number.MAX_SAFE_INTEGER
      );
      expect(worstKey + 1).not.toBe(worstKey); // i.e. still in exact-integer territory
    } finally {
      resetElementTextureLayoutForTests();
    }
  });

  it('does not merge adjacent elements at the nodeId ceiling', () => {
    // The 2^32 stride this replaced broke exactly here: at nodeId 2^21 the
    // key crosses 2^53 and elementId 0 vs 1 round onto the same number.
    for (const nodeId of [2 ** 21, 2 ** 24 - 1]) {
      const pixels = buildPixels();
      setHit(pixels, 0, 0, nodeId, 0, 0.4);
      setHit(pixels, 1, 0, nodeId, 1, 0.6);
      const scratchLocal = new Map<number, VoteEntry>();
      const winner = voteWinner(pixels, PICK_SIZE, scratchLocal);
      expect(scratchLocal.size, `nodeId ${nodeId}: adjacent elements merged`).toBe(2);
      expect(winner!.elementId).toBe(1);
      expect(winner!.weight).toBeCloseTo(0.6, 5);
    }
  });

  it('honors pickSize parameter (3×3 block reads only the first 9 RGBA quads)', () => {
    // Build a 5×5 buffer but pass pickSize=3 — only the first 9 entries
    // (indices 0..8) should be read.
    const pixels = buildPixels();
    // Put a hit at index 9 (which is OUTSIDE the 3×3 read window)
    pixels[9 * 4] = 7;
    pixels[9 * 4 + 1] = 7;
    pixels[9 * 4 + 2] = 0.9;

    expect(voteWinner(pixels, 3, scratch)).toBeNull();
    expect(scratch.size).toBe(0);
  });
});
