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
    startCapSuppression: new Float32Array([0]),
    endCapSuppression: new Float32Array([0]),
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
      startCapSuppression: new Float32Array([0, 0]),
      endCapSuppression: new Float32Array([0, 0]),
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
