/**
 * Direct unit tests for the brightness-weighted majority-voting helper.
 * Pure over its (pixels, pickSize, scratch) inputs — no PickingSystem
 * instance, no GPU, no renderer.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  voteWinner,
  type VoteEntry,
} from '../../../../../rendering/picking/picking-system/pick-render';

const PICK_SIZE = 5;

/** Build a 5×5 RGBA Float32Array, all background by default. */
function buildPixels(): Float32Array {
  return new Float32Array(PICK_SIZE * PICK_SIZE * 4);
}

/** Write (r, g, b, a) into pixel slot (x, y) of a `pickSize × pickSize` RGBA buffer. */
function setPixel(
  pixels: Float32Array,
  x: number,
  y: number,
  r: number,
  g: number,
  b: number,
  a: number = 1.0
): void {
  const i = (y * PICK_SIZE + x) * 4;
  pixels[i] = r;
  pixels[i + 1] = g;
  pixels[i + 2] = b;
  pixels[i + 3] = a;
}

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
    setPixel(pixels, 2, 2, 7, 42, 0.8);
    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(winner!.nodeId).toBe(7);
    expect(winner!.elementId).toBe(42);
    // Float32Array narrows brightness; compare with float32-grade tolerance.
    expect(winner!.weight).toBeCloseTo(0.8, 5);
  });

  it('sums brightness weights across pixels with the same (nodeId, elementId)', () => {
    const pixels = buildPixels();
    setPixel(pixels, 0, 0, 3, 9, 0.4);
    setPixel(pixels, 1, 0, 3, 9, 0.3);
    setPixel(pixels, 2, 0, 3, 9, 0.2);
    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(winner!.nodeId).toBe(3);
    expect(winner!.elementId).toBe(9);
    expect(winner!.weight).toBeCloseTo(0.4 + 0.3 + 0.2, 5);
  });

  it('picks the highest summed-brightness ID when multiple IDs are present', () => {
    const pixels = buildPixels();
    // ID (1, 1) gets two pixels totaling 0.5
    setPixel(pixels, 0, 0, 1, 1, 0.3);
    setPixel(pixels, 1, 0, 1, 1, 0.2);
    // ID (2, 2) gets one pixel of 0.7 → wins
    setPixel(pixels, 0, 1, 2, 2, 0.7);
    // ID (3, 3) gets two pixels totaling 0.4
    setPixel(pixels, 0, 2, 3, 3, 0.25);
    setPixel(pixels, 1, 2, 3, 3, 0.15);

    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(winner!.nodeId).toBe(2);
    expect(winner!.elementId).toBe(2);
    expect(winner!.weight).toBeCloseTo(0.7, 5);
  });

  it('disambiguates by (nodeId, elementId) — same nodeId, different elementId are distinct entries', () => {
    const pixels = buildPixels();
    setPixel(pixels, 0, 0, 5, 100, 0.6);
    setPixel(pixels, 1, 0, 5, 101, 0.4);
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
    setPixel(pixels, 0, 0, 1, 1, 0.5);
    const winner = voteWinner(pixels, PICK_SIZE, scratch);

    // Stale entry MUST be gone; only the fresh (1,1) remains
    expect(scratch.size).toBe(1);
    expect(scratch.has(999_999_999)).toBe(false);
    expect(winner!.nodeId).toBe(1);
  });

  it('encodes the nodeId/elementId composite key losslessly for 24-bit IDs', () => {
    const pixels = buildPixels();
    // nodeId * 2^24 + elementId must not collide for IDs within [0, 2^24)
    setPixel(pixels, 0, 0, 1, 0, 0.5);
    setPixel(pixels, 1, 0, 0, 1, 0.7); // r=0 → background, but g=1 alone shouldn't form a vote
    // Re-place: keep both as hits — (1, 0) and (2, 0)
    setPixel(pixels, 1, 0, 2, 0, 0.7);

    const winner = voteWinner(pixels, PICK_SIZE, scratch);
    expect(scratch.size).toBe(2);
    // (1, 0) and (2, 0) — different nodeIds, same elementId — should be distinct
    expect(scratch.has(1 * 16777216 + 0)).toBe(true);
    expect(scratch.has(2 * 16777216 + 0)).toBe(true);
    expect(winner!.nodeId).toBe(2);
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
