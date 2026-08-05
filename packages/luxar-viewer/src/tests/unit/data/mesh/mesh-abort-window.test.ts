/**
 * The superseded-commit abort window at the `projectMeshTo3D` + `updateMeshGeometry`
 * seam (#1245 follow-up).
 *
 * `projectMeshTo3D` decides re-extraction at PROJECTION time: it extracts the new
 * `displayDims` into the loader-owned scratch and advances the loader's
 * `displayDimsKey` there, then reports `positionChanged`. But a commit can be
 * aborted or superseded AFTER its projection ran and BEFORE `updateMeshGeometry`
 * uploads (`atomic-commit.ts` skips a staged-but-aborted commit). The loader key
 * has already advanced, so the NEXT projection at that same `displayDims` sees the
 * key match and reports `positionChanged === false` — even though the geometry on
 * the GPU still holds the OLD axes. A projection-time flag cannot see that gap.
 *
 * The fix decides the upload commit-side too: the geometry stamps the key it last
 * uploaded (`userData.meshUploadedKey`) and re-uploads whenever the projection's
 * `positionKey` differs from it, which is exactly the aborted-commit case. This
 * test reproduces the window against the real `projectMeshTo3D` +
 * `updateMeshGeometry` (no mocks, `TypeScriptFallback` backend) and pins the
 * repair: after the abort, the next same-key commit must still signal a re-upload
 * (the position attribute's `version` advances), not silently keep the stale frame.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { TypeScriptFallback } from '../../../../wasm/typescript';
import { projectMeshTo3D } from '../../../../data/mesh/projection';
import { createMeshGeometry, updateMeshGeometry } from '../../../../rendering/mesh-geometry';
import type { LoadedMeshData, MeshViewState } from '../../../../types/mesh';

const backend = new TypeScriptFallback();

function viewState(displayDims: number[], slicePosition: number[]): MeshViewState {
  // Hidden dim is w (index 3); the triangle sits at w = 0 and the tolerance covers
  // it, so both slice positions below keep it visible — they differ ONLY on w.
  return { displayDims, slicePosition, tolerance: [1e10, 1e10, 1e10, 1] } as MeshViewState;
}

/**
 * One triangle in 4D with a loader-owned projection buffer. Vertex 1 = (1,0,0,w):
 * displaying [0,1,2] extracts it to (1,0,0); displaying [1,0,2] to (0,1,0), so the
 * two axis triples are trivially distinguishable in the position buffer.
 */
function oneTriangleIn4D(): LoadedMeshData {
  return {
    vertices: new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
    faces: new Uint32Array([0, 1, 2]),
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: 3,
    faceCount: 1,
    ndim: 4,
    projection: { position: new Float32Array(3 * 3), displayDimsKey: null },
  };
}

/** Position attribute of `geom` as a concrete `BufferAttribute`. */
function positionAttr(geom: THREE.BufferGeometry): THREE.BufferAttribute {
  return geom.getAttribute('position') as THREE.BufferAttribute;
}

describe('mesh commit — the superseded-commit abort window (#1245 follow-up)', () => {
  it('re-uploads at the aborted key even when positionChanged reads false', () => {
    const data = oneTriangleIn4D();
    const slice0 = [0, 0, 0, 0];
    const sliceOther = [0, 0, 0, 0.5]; // differs from slice0 only on hidden w

    // A 1-vertex placeholder, exactly as the node factory builds it.
    const geom = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
    });

    // 1. Commit at k1 = [0,1,2]: the placeholder grows to the real buffer and the
    //    geometry stamps meshUploadedKey = '0,1,2'.
    const pA = projectMeshTo3D(data, viewState([0, 1, 2], slice0), undefined, true, backend);
    updateMeshGeometry(geom, {
      position: pA.position,
      positionChanged: pA.positionChanged,
      positionKey: pA.positionKey,
      indices: pA.indices,
      colors: null,
      vertexCount: 3,
      faceCount: 1,
      bounds: pA.bounds,
    });
    // A k1 slice move: same axes, so nothing re-extracts or re-uploads — it just lets
    // the position attribute settle to a stable version we can baseline against.
    const pB = projectMeshTo3D(data, viewState([0, 1, 2], sliceOther), undefined, true, backend);
    updateMeshGeometry(geom, {
      position: pB.position,
      positionChanged: pB.positionChanged,
      positionKey: pB.positionKey,
      indices: pB.indices,
      colors: null,
      vertexCount: 3,
      faceCount: 1,
      bounds: pB.bounds,
    });
    const v1 = positionAttr(geom).version;

    // 2. A SUPERSEDED k2 = [1,0,2] projection whose commit is aborted: it extracts k2
    //    into the shared buffer and advances the loader key to '1,0,2', but we
    //    deliberately never call updateMeshGeometry — so the GPU still holds k1 while
    //    the loader now thinks '1,0,2' is resident. This arms the trap.
    projectMeshTo3D(data, viewState([1, 0, 2], slice0), undefined, true, backend);

    // 3. A k2 slice move that DOES commit. The reused buffer already holds '1,0,2', so
    //    the projection reports positionChanged === false — the projection-time flag is
    //    now blind to the missed upload. Only the commit-side key guard can catch it.
    const pC = projectMeshTo3D(data, viewState([1, 0, 2], sliceOther), undefined, true, backend);
    expect(pC.positionChanged).toBe(false); // the abort set up the trap
    updateMeshGeometry(geom, {
      position: pC.position,
      positionChanged: pC.positionChanged,
      positionKey: pC.positionKey,
      indices: pC.indices,
      colors: null,
      vertexCount: 3,
      faceCount: 1,
      bounds: pC.bounds,
    });

    // 4. The geometry now shows k2 AND signalled a re-upload. The value check alone is
    //    not enough — the shared buffer already held k2 from step 2 — so the version
    //    bump is the load-bearing assertion: without the key guard the upload is
    //    skipped, version stays at v1, and the GPU keeps the stale k1 frame.
    const attr = positionAttr(geom);
    // Vertex 1 = (1,0,0,0); displaying [1,0,2] puts it at (0,1,0).
    expect(Array.from(attr.array.slice(3, 6))).toEqual([0, 1, 0]);
    expect(attr.version).toBeGreaterThan(v1);
  });
});
