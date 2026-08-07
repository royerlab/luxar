/**
 * Lock-in tests for line geometry bounding-box expansion.
 *
 * `createInstancedLinesMesh` expands `geometry.boundingBox` outward by the
 * max half-width (`line-geometry.ts::computeLineBounds`). This margin is
 * what keeps the GPU-picking ray-AABB cull from clipping the pick footprint
 * of a line segment sitting on the scene's bounding-box surface — the pick
 * cull (`picking/picking-system/ray-aabb.ts`) trusts `boundingBox` directly
 * for lines/gsplats (it only adds its own margin for points, whose mesh has
 * frustum culling disabled). If the width expansion here is ever removed,
 * line hover-picking silently regresses for edge segments, so pin it.
 *
 * Pure buffer + Box3 math — no GL context needed (jsdom-safe).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  createInstancedLinesMesh,
  clampJointCode,
  MAX_EXACT_JOINT_SLOT,
  type InstancedLinesMeshConfig,
} from '../../../rendering/line-geometry';

/** One segment from (0,0,0)→(1,0,0) with a uniform half-width on both ends. */
function makeSingleSegmentConfig(width: number): InstancedLinesMeshConfig {
  return {
    startPositions: new Float32Array([0, 0, 0]),
    endPositions: new Float32Array([1, 0, 0]),
    startColors: new Float32Array([1, 1, 1]),
    endColors: new Float32Array([1, 1, 1]),
    startWidths: new Float32Array([width]),
    endWidths: new Float32Array([width]),
    startSharpness: new Float32Array([2]),
    endSharpness: new Float32Array([2]),
    segmentLengths: new Float32Array([1]),
    startJointCode: new Float32Array([0]),
    endJointCode: new Float32Array([0]),
    segmentCount: 1,
  };
}

describe('createInstancedLinesMesh — bounding-box footprint expansion', () => {
  it('expands boundingBox outward by the max half-width', () => {
    const width = 0.5;
    const mesh = createInstancedLinesMesh(
      makeSingleSegmentConfig(width),
      new THREE.MeshBasicMaterial()
    );
    const box = mesh.geometry.boundingBox!;

    expect(box).not.toBeNull();
    // Vertex span is x:[0,1], y:0, z:0; expanded by width on every axis.
    expect(box.min.x).toBeCloseTo(-width, 5);
    expect(box.max.x).toBeCloseTo(1 + width, 5);
    expect(box.min.y).toBeCloseTo(-width, 5);
    expect(box.max.y).toBeCloseTo(width, 5);
    expect(box.min.z).toBeCloseTo(-width, 5);
    expect(box.max.z).toBeCloseTo(width, 5);
    // Mesh is frustum-culled, so this box doubles as the frustum-cull box.
    expect(mesh.frustumCulled).toBe(true);
  });

  it('does not expand when all widths are zero', () => {
    const mesh = createInstancedLinesMesh(
      makeSingleSegmentConfig(0),
      new THREE.MeshBasicMaterial()
    );
    const box = mesh.geometry.boundingBox!;
    // No width → box is exactly the vertex span.
    expect(box.min.x).toBeCloseTo(0, 5);
    expect(box.max.x).toBeCloseTo(1, 5);
    expect(box.min.y).toBeCloseTo(0, 5);
    expect(box.max.y).toBeCloseTo(0, 5);
  });
});

describe('computeLineBounds — precomputed projection bounds fast path', () => {
  it('bounds-present and scan-fallback configs produce identical cull bounds', async () => {
    const { computeLinesProjectionBounds } =
      await import('../../../workers/data-worker/projection/lines');
    // Two segments with negative coordinates and distinct end widths.
    const base: InstancedLinesMeshConfig = {
      startPositions: new Float32Array([-4, -5, -6, 1, 2, 3]),
      endPositions: new Float32Array([7, 8, 9, -1, -2, -3]),
      startColors: new Float32Array(6).fill(1),
      endColors: new Float32Array(6).fill(1),
      startWidths: new Float32Array([0.25, 2.0]),
      endWidths: new Float32Array([1.0, 0.5]),
      startSharpness: new Float32Array([2, 2]),
      endSharpness: new Float32Array([2, 2]),
      segmentLengths: new Float32Array([1, 1]),
      startJointCode: new Float32Array([0, 0]),
      endJointCode: new Float32Array([0, 0]),
      segmentCount: 2,
    };

    // Fallback: no bounds metadata → computeLineBounds scans.
    const scanMesh = createInstancedLinesMesh(base, new THREE.MeshBasicMaterial());

    // Fast path: fused-scan metadata supplied → the scan is skipped.
    const fastMesh = createInstancedLinesMesh(
      {
        ...base,
        bounds: computeLinesProjectionBounds(
          base.startPositions,
          base.endPositions,
          base.startWidths,
          base.endWidths,
          base.segmentCount
        ),
      },
      new THREE.MeshBasicMaterial()
    );

    const scanBox = scanMesh.geometry.boundingBox!;
    const fastBox = fastMesh.geometry.boundingBox!;
    // Bit-exact equality (same float ops in the fused scan).
    expect(fastBox.min.toArray()).toEqual(scanBox.min.toArray());
    expect(fastBox.max.toArray()).toEqual(scanBox.max.toArray());
    expect(fastMesh.geometry.boundingSphere!.radius).toBe(scanMesh.geometry.boundingSphere!.radius);
  });
});

describe('clampJointCode — the two independent write-time guards', () => {
  // Driven directly rather than through writeLineTexels: the two rules bind at
  // wildly different scales, and the prefix rule rejects any large slot before
  // the representability rule is ever reached, so a writer-level test can only
  // exercise the first.
  const enc = { atStart: (slot: number) => slot + 1, atEnd: (slot: number) => -(slot + 3) };

  it('passes every sentinel through untouched', () => {
    // 0 / -1 / -2 carry no slot and must never be reinterpreted as one — a
    // naive `code > 0 ? code - 1 : -code - 3` would read -1 as slot -2.
    for (const sentinel of [0, -1, -2]) {
      expect(clampJointCode(sentinel, 0)).toBe(sentinel);
      expect(clampJointCode(sentinel, 1_000_000)).toBe(sentinel);
    }
  });

  it('rule 1: drops a code naming a slot outside the written prefix', () => {
    expect(clampJointCode(enc.atStart(5), 6)).toBe(enc.atStart(5)); // slot 5 < 6
    expect(clampJointCode(enc.atStart(6), 6)).toBe(0); // slot 6 is the first unwritten
    expect(clampJointCode(enc.atEnd(5), 6)).toBe(enc.atEnd(5));
    expect(clampJointCode(enc.atEnd(6), 6)).toBe(0);
  });

  it('rule 2: drops a code whose slot is not float32-exact (> 2^24)', () => {
    // The code rides an RGBA32F texel and float32 spaces consecutive integers
    // by 1 only to 2^24; past that an ODD value rounds to a neighbour and
    // decodes to a DIFFERENT slot. `-(slot + 3)` has the larger magnitude, so
    // it binds first. Reachable on a 32768-class device, where the per-node
    // line capacity reaches 22.35M and a measured 12.5% of codes mis-decode —
    // mitring against an unrelated segment, the exact failure this design
    // exists to prevent.
    const written = Number.MAX_SAFE_INTEGER; // isolate rule 2 from rule 1
    expect(clampJointCode(enc.atEnd(MAX_EXACT_JOINT_SLOT), written)).toBe(
      enc.atEnd(MAX_EXACT_JOINT_SLOT)
    );
    expect(clampJointCode(enc.atEnd(MAX_EXACT_JOINT_SLOT + 1), written)).toBe(0);
    expect(clampJointCode(enc.atStart(MAX_EXACT_JOINT_SLOT + 1), written)).toBe(0);
  });

  it('rule 3: drops a malformed code that decodes to a negative or fractional slot', () => {
    // The two bounds above are both UPPER bounds, so on their own they accept
    // 0.75: it decodes to slot -0.25, passes `slot < written` and `slot <=
    // MAX_EXACT_JOINT_SLOT`, and reaches the shader — which reads it as
    // partner-bearing (> 0.5) and truncates to partnerSlot = int(0.75) - 1 =
    // -1, an out-of-range texelFetch. The kernel never emits a fractional
    // code, but the kernel is not the only producer (the TSL parity harness
    // hand-authors them), which is precisely why this guard exists.
    expect(clampJointCode(0.75, 1_000_000)).toBe(0);
    expect(clampJointCode(0.25, 1_000_000)).toBe(0);
    expect(clampJointCode(-2.75, 1_000_000)).toBe(0); // decodes to slot -0.25
    expect(clampJointCode(enc.atStart(3) + 0.5, 1_000_000)).toBe(0);
    expect(clampJointCode(NaN, 1_000_000)).toBe(0);
    expect(clampJointCode(Infinity, 1_000_000)).toBe(0);
    expect(clampJointCode(-Infinity, 1_000_000)).toBe(0);
    // The well-formed neighbours of those values still ride through.
    expect(clampJointCode(enc.atStart(0), 1)).toBe(enc.atStart(0));
    expect(clampJointCode(enc.atEnd(0), 1)).toBe(enc.atEnd(0));
  });

  it('the boundary slot really does survive a float32 round-trip, and the next one does not', () => {
    // Pins WHY MAX_EXACT_JOINT_SLOT is where it is, so the constant cannot be
    // nudged without this failing.
    const decode = (c: number) => (c > 0 ? Math.trunc(c) - 1 : Math.trunc(-c) - 3);
    for (const e of [enc.atStart, enc.atEnd]) {
      expect(decode(Math.fround(e(MAX_EXACT_JOINT_SLOT)))).toBe(MAX_EXACT_JOINT_SLOT);
    }
    // One past the bound, the END encoding is the one that loses exactness.
    expect(decode(Math.fround(enc.atEnd(MAX_EXACT_JOINT_SLOT + 1)))).not.toBe(
      MAX_EXACT_JOINT_SLOT + 1
    );
  });
});
