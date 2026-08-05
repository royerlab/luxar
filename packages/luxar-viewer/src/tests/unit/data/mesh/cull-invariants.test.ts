/**
 * Seven laws the whole-triangle cull must obey, over 1200 randomized nD meshes.
 *
 * The unit tests check named cases; this checks the LAWS, which is what survives a
 * reimplementation. Law 3 is the load-bearing one — every emitted triangle must be a
 * triangle that was in the input, in ORIGINAL vertex numbering — because that is the
 * no-compaction contract, and a remap bug would satisfy every count-based assertion
 * while silently rewiring topology.
 */

import { describe, it, expect } from 'vitest';
import { TypeScriptFallback } from '../../../../wasm/typescript';
import { projectMeshTo3D } from '../../../../data/mesh/projection';
import type { LoadedMeshData, MeshViewState } from '../../../../types/mesh';

const backend = new TypeScriptFallback();

/** Deterministic pseudo-random mesh in `ndim` dims. */
function makeMesh(seed: number, nv: number, nf: number, ndim: number): LoadedMeshData {
  const vertices = new Float32Array(nv * ndim);
  for (let i = 0; i < nv * ndim; i++) {
    const d = i % ndim;
    vertices[i] = Math.sin(i * 0.037 * (d + 1) + seed * 1.7 + d * 1.23) * 3;
  }
  const faces = new Uint32Array(nf * 3);
  for (let f = 0; f < nf; f++)
    for (let c = 0; c < 3; c++)
      faces[f * 3 + c] = Math.abs(Math.floor(Math.sin(f * 2.1 + c * 0.9 + seed) * 1e4)) % nv;
  return {
    vertices,
    faces,
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: nv,
    faceCount: nf,
    ndim,
  };
}
const vs = (ndim: number, tol: number): MeshViewState =>
  ({
    displayDims: [0, 1, 2],
    slicePosition: new Array(ndim).fill(0),
    tolerance: [1e10, 1e10, 1e10, ...new Array(Math.max(0, ndim - 3)).fill(tol)],
  }) as never;

describe('cull invariants over 1000 random meshes', () => {
  it('holds every law', () => {
    let totalTrials = 0,
      sawPartial = 0,
      sawGrowth = 0;
    for (let seed = 0; seed < 200; seed++) {
      const ndim = 4 + (seed % 3); // 4..6 dims -> at least one hidden
      const nv = 5 + (seed % 17);
      const nf = 3 + (seed % 11);
      const mesh = makeMesh(seed, nv, nf, ndim);
      const inFaces = new Set<string>();
      for (let f = 0; f < nf; f++)
        inFaces.add([0, 1, 2].map((c) => mesh.faces[f * 3 + c]).join(','));

      let prevCount = -1;
      for (const tol of [0.01, 0.1, 0.5, 1, 3, 10]) {
        const r = projectMeshTo3D(mesh, vs(ndim, tol), undefined, true, backend);
        totalTrials++;

        // LAW 1: every emitted index is a valid vertex ordinal.
        for (const idx of r.indices) expect(idx).toBeLessThan(nv);

        // LAW 2: emitted count is a multiple of 3 and matches the reported face count.
        expect(r.indices.length).toBe(r.visibleFaceCount * 3);

        // LAW 3: every emitted TRIANGLE was present in the input (a subsequence, in
        // original vertex numbering — nothing is invented or remapped).
        for (let f = 0; f < r.visibleFaceCount; f++) {
          const key = [0, 1, 2].map((c) => r.indices[f * 3 + c]).join(',');
          expect(inFaces.has(key), `emitted face ${key} not in input`).toBe(true);
        }

        // LAW 4: visible count never exceeds the input count.
        expect(r.visibleFaceCount).toBeLessThanOrEqual(nf);

        // LAW 5: MONOTONE in tolerance — a wider slab can never show fewer faces.
        if (prevCount >= 0) {
          expect(r.visibleFaceCount, `tol=${tol} shrank vs previous`).toBeGreaterThanOrEqual(
            prevCount
          );
          if (r.visibleFaceCount > prevCount) sawGrowth++;
        }
        prevCount = r.visibleFaceCount;
        if (r.visibleFaceCount > 0 && r.visibleFaceCount < nf) sawPartial++;

        // LAW 6: position buffer is always full-length (no compaction).
        expect(r.position.length).toBe(nv * 3);

        // LAW 7: the reported bounds are EXACTLY the AABB of the indexed vertices —
        // null iff nothing is drawn. Named cases can show the box excludes a culled
        // triangle; only the sweep shows it never drifts, over every partial cull the
        // 1200 trials produce. Recomputed here independently of the implementation.
        if (r.visibleFaceCount === 0) {
          expect(r.bounds, 'nothing drawn must report null bounds').toBeNull();
        } else {
          const min = [Infinity, Infinity, Infinity];
          const max = [-Infinity, -Infinity, -Infinity];
          for (const idx of r.indices) {
            for (let c = 0; c < 3; c++) {
              const v = r.position[idx * 3 + c];
              if (!Number.isFinite(v)) continue;
              if (v < min[c]) min[c] = v;
              if (v > max[c]) max[c] = v;
            }
          }
          expect(r.bounds, 'drawn geometry must report bounds').not.toBeNull();
          expect(r.bounds!.min).toEqual(min);
          expect(r.bounds!.max).toEqual(max);
        }
      }
    }
    // Anti-vacuity: the cull must actually discriminate, or every law above is
    // satisfied trivially by "emit everything" / "emit nothing".
    //
    // The thresholds are the MEASURED counts with margin, not aspirations. Partial
    // culls (0 < visible < nf) are the rarer outcome with this fixture — 9 of 1200 —
    // because randomly-indexed faces scatter their three vertices across the hidden
    // axis, so a slab tends to admit all or none. The 207 growth steps are what carry
    // the monotonicity law's weight.
    expect(totalTrials, 'the sweep actually ran').toBe(200 * 6);
    expect(sawPartial, 'partial culls observed').toBeGreaterThanOrEqual(5);
    expect(sawGrowth, 'tolerance-growth steps observed').toBeGreaterThan(100);
  });
});
