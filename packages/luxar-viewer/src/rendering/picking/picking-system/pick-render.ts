/**
 * Pick-buffer voting for PickingSystem.
 *
 * The pick render itself stays on the orchestrator (binds the renderer,
 * sets the target, etc.). This module owns the brightness-weighted
 * majority-voting pass over the 5×5 readback pixel block.
 *
 * @module rendering/picking/picking-system/pick-render
 */

/** Per-cell vote bucket: brightness-weighted accumulator. */
export interface VoteEntry {
  nodeId: number;
  elementId: number;
  weight: number;
}

/**
 * Tally votes from a 5×5 pixel block of the pick buffer and return the
 * winning (nodeId, elementId, accumulated brightness). Pixels with
 * `r < 0.5` are skipped (background = no hit).
 *
 * **Channel layout** (RGBA32F): `r` = nodeId, `g` = element index LOW
 * 16 bits, `b` = brightness, `a` = element index HIGH 16 bits. The
 * element index is split because float32 carries a 24-bit mantissa, so a
 * single channel stops representing consecutive integers at 16,777,216 —
 * while a node's capacity reaches 2^25 on a 32768-texel device, where a
 * one-channel id silently resolved to the wrong element. The shaders
 * split it in int space (`luxarElementIdParts` / its TSL twin); both
 * halves are <= 65535 and exact, so the recombination below is exact for
 * the whole uint32 range.
 *
 * The vote key must not alias across nodes now that `elementId` can
 * exceed 2^24, so it scales by 2^32 rather than 2^24. That is exact in a
 * double while `nodeId < 2^21` — and nodeId rides an f32 channel, which
 * caps it at 2^24 anyway, so the binding limit is the 2^21 here.
 *
 * `votesScratch` is reused across calls (cleared here) to keep the
 * hot path allocation-free.
 */
export function voteWinner(
  pixels: Float32Array,
  pickSize: number,
  votesScratch: Map<number, VoteEntry>
): VoteEntry | null {
  votesScratch.clear();
  for (let i = 0; i < pickSize * pickSize; i++) {
    const r = pixels[i * 4];
    const g = pixels[i * 4 + 1];
    const b = pixels[i * 4 + 2];
    const a = pixels[i * 4 + 3];
    if (r < 0.5) continue;
    const nodeId = Math.round(r);
    const elementId = Math.round(a) * 65536 + Math.round(g);
    const key = nodeId * 4294967296 + elementId;
    const existing = votesScratch.get(key);
    if (existing) existing.weight += b;
    else votesScratch.set(key, { nodeId, elementId, weight: b });
  }

  let winner: VoteEntry | null = null;
  for (const entry of votesScratch.values()) {
    if (!winner || entry.weight > winner.weight) winner = entry;
  }
  return winner;
}
