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
 * `votesScratch` is reused across calls (cleared here) to keep the
 * hot path allocation-free — which is why the key stays a NUMBER rather
 * than a string or a nested map.
 *
 * See {@link VOTE_KEY_STRIDE} for why that is safe.
 */
/**
 * Multiplier packing `(nodeId, elementId)` into one numeric vote key.
 *
 * Two conditions must hold simultaneously, and 2^27 is the value that
 * satisfies both with room to spare:
 *
 * 1. **Alias-free** — the stride must exceed every reachable `elementId`,
 *    or `(node n, element stride)` collides with `(node n+1, element 0)`
 *    and two different elements merge their brightness votes. The largest
 *    reachable index is a node at max capacity, which is
 *    `getMaxElementCapacityPerNode` = `width x maxTextureSize /
 *    texelsPerElement`: 44,728,319 for points on a 32768-texel device.
 *    2^27 = 134,217,728 clears that ~3x over, so even a future
 *    `maxTextureSize` of 65536 (~89M) still fits.
 * 2. **Exactly representable** — `nodeId * stride + elementId` must stay
 *    under 2^53 or adjacent keys round onto each other. `nodeId` rides an
 *    f32 channel of the pick buffer, so it cannot exceed 2^24; the worst
 *    case is `(2^24 - 1) * 2^27 + 44,728,319` ~ 2.25e15, comfortably
 *    inside 9.007e15.
 *
 * A stride of 2^32 would satisfy (1) but violate (2) past `nodeId` 2^21 —
 * an unenforced bound is not a bound, hence 2^27. `assertVoteKeyHeadroom`
 * in the unit tests pins both conditions against the live layout maxima,
 * so a capacity increase that outgrows this stride fails there rather
 * than silently merging votes.
 */
export const VOTE_KEY_STRIDE = 134217728; // 2^27

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
    const key = nodeId * VOTE_KEY_STRIDE + elementId;
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
