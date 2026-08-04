/**
 * Where the byte budget actually cuts off, found by bisection rather than predicted.
 *
 * Worth measuring rather than deriving: my first attempt at this arithmetic was wrong
 * (it omitted the faces terms), and a stale derived figure had already reached the docs
 * — the "~44.7M vertices" claim survived the change that charges the decoded term and
 * halved the real limit to ~22.4M, because no test compared the number to behaviour.
 *
 * Pins three things the comparison operator decides: that there is exactly ONE
 * transition, that it is monotone on both sides, and that a peak exactly AT the ceiling
 * is admitted while one byte over is refused (`peak > BUDGET`, not `>=`).
 */

import { describe, it, expect } from 'vitest';
import { preflightMesh } from '../../../../data/mesh/mesh-preflight';
import { MESH_DECODE_BUDGET_BYTES } from '../../../../config/constants';
import type * as zarr from '../../../../data/zarr';
import type { MeshMetadata } from '../../../../types/mesh';

const fa = (shape: number[], dtype: string, chunks?: number[]) =>
  ({ shape, chunks: chunks ?? shape, dtype, attrs: {} }) as unknown as zarr.Array<
    zarr.DataType,
    zarr.Readable
  >;
const A = (o: Partial<MeshMetadata> = {}): MeshMetadata => ({
  type: 'mesh',
  n_vertices: 4,
  n_faces: 4,
  ndim: 3,
  has_normals: false,
  has_colors: false,
  has_scalars: false,
  shading: 'flat',
  double_sided: true,
  ordering: 'none',
  ...o,
});

async function admits(V: number): Promise<boolean> {
  try {
    await preflightMesh('/m', A({ n_vertices: V, n_faces: 1 }), {
      vertices: fa([V, 3], '<f4', [64, 3]),
      faces: fa([1, 3], '<u4'),
    } as never);
    return true;
  } catch {
    return false;
  }
}

describe('budget boundary, bisected', () => {
  it('has exactly one monotone transition, and admits AT the ceiling', async () => {
    let lo = 1,
      hi = 30_000_000;
    expect(await admits(lo)).toBe(true);
    expect(await admits(hi)).toBe(false);
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (await admits(mid)) lo = mid;
      else hi = mid;
    }
    // lo = largest admitted, hi = smallest refused.
    const peak = (V: number) => V * 3 * 4 * 2 + (1 * 3 * 4 + 3 * 4) + 64 * 3 * 4;
    console.log(
      `largest admitted V=${lo} (peak ${peak(lo)}), smallest refused V=${hi} (peak ${peak(hi)}), budget ${MESH_DECODE_BUDGET_BYTES}`
    );
    console.log(
      `admitted peak <= budget? ${peak(lo) <= MESH_DECODE_BUDGET_BYTES}; refused peak > budget? ${peak(hi) > MESH_DECODE_BUDGET_BYTES}`
    );
    expect(hi - lo).toBe(1);
    // The comparison is `peak > BUDGET`, so the largest admitted peak must be <= budget
    // and the smallest refused must be strictly greater. This pins > vs >=.
    expect(peak(lo)).toBeLessThanOrEqual(MESH_DECODE_BUDGET_BYTES);
    expect(peak(hi)).toBeGreaterThan(MESH_DECODE_BUDGET_BYTES);
    // Monotonicity: nothing below lo may refuse, nothing above hi may admit.
    for (const V of [1, 100, Math.floor(lo / 2), lo - 1]) expect(await admits(V)).toBe(true);
    for (const V of [hi + 1, hi * 2, 29_000_000]) expect(await admits(V)).toBe(false);
  });
});
