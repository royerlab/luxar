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
 * `r < 0.5` are skipped (background = no hit). The numeric key
 * `nodeId * 2^24 + elementId` is lossless for 24-bit IDs.
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
    if (r < 0.5) continue;
    const nodeId = Math.round(r);
    const elementId = Math.round(g);
    const key = nodeId * 16777216 + elementId;
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
