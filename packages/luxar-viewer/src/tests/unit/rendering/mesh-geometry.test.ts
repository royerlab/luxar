/**
 * Mesh geometry assembly.
 *
 * The centre of gravity is the §6.1.1 vertex-attribute dtype rules, which are
 * WebGPU-validity requirements rather than preferences: three r184's WebGPU backend
 * exposes no 3-component 8/16-bit vertex format, and requires `arrayStride` to be a
 * multiple of 4. A size-3 `uint8` colour attribute has a 3-byte stride and fails
 * `createRenderPipeline`, so the mesh renders NOTHING on WebGPU while looking
 * correct on WebGL — the kind of asymmetry that only shows up on the backend the
 * test suite doesn't run.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { log } from '../../../utils/log';
import {
  createMeshColorAttribute,
  createMeshDefaultColorAttribute,
  createMeshIndexAttribute,
  createMeshGeometry,
  updateMeshGeometry,
} from '../../../rendering/mesh-geometry';

describe('createMeshColorAttribute — the WebGPU dtype rules', () => {
  it('pads uint8 RGB to a 4-component normalized attribute', () => {
    const attr = createMeshColorAttribute(new Uint8Array([255, 0, 0, 0, 255, 0]), 3, 2);
    expect(attr.itemSize).toBe(4);
    expect(attr.normalized).toBe(true);
    expect(attr.array).toBeInstanceOf(Uint8Array);
    // Opaque pad alpha: 255 normalizes to exactly 1.0.
    expect(Array.from(attr.array)).toEqual([255, 0, 0, 255, 0, 255, 0, 255]);
  });

  it('pads uint16 RGB with 65535, which also normalizes to 1.0', () => {
    const attr = createMeshColorAttribute(new Uint16Array([65535, 0, 0]), 3, 1);
    expect(attr.itemSize).toBe(4);
    expect(attr.array).toBeInstanceOf(Uint16Array);
    expect(Array.from(attr.array)).toEqual([65535, 0, 0, 65535]);
  });

  it('gives the padded attribute a stride that is a multiple of 4', () => {
    // The actual WebGPU constraint. uint8 x4 = 4 bytes, uint16 x4 = 8 bytes; the
    // unpadded size-3 forms would be 3 and 6, neither of which is legal.
    for (const [colors, bytesPerElement] of [
      [new Uint8Array(6), 1],
      [new Uint16Array(6), 2],
    ] as const) {
      const attr = createMeshColorAttribute(colors, 3, 2);
      expect((attr.itemSize * bytesPerElement) % 4).toBe(0);
    }
  });

  it('leaves uint8 RGBA alone — already a valid 4-byte-stride format', () => {
    const src = new Uint8Array([1, 2, 3, 4]);
    const attr = createMeshColorAttribute(src, 4, 1);
    expect(attr.itemSize).toBe(4);
    expect(attr.array).toBe(src); // no copy
    expect(attr.normalized).toBe(true);
  });

  it('leaves float32 at its native width and does NOT normalize it', () => {
    // float32x3 is a valid WebGPU format with a 12-byte stride, so there is nothing
    // to pad. Normalizing would also be wrong: float colours are authored in [0, 1]
    // already, and HDR values legitimately exceed 1.
    const src = new Float32Array([0.5, 0.25, 0.125]);
    const attr = createMeshColorAttribute(src, 3, 1);
    expect(attr.itemSize).toBe(3);
    expect(attr.normalized).toBe(false);
    expect(attr.array).toBe(src);
  });
});

describe('createMeshDefaultColorAttribute', () => {
  it('fills opaque white, not zeros', () => {
    // An UNBOUND color attribute reads the GL default (0, 0, 0, 1), so a bare
    // add_mesh(vertices, faces) surface would come out solid black the moment any
    // multiplicative shade term lands. Binding white is what keeps it visible.
    const attr = createMeshDefaultColorAttribute(3);
    expect(attr.itemSize).toBe(3);
    expect(Array.from(attr.array)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
  });
});

describe('createMeshIndexAttribute', () => {
  it('uses Uint16Array below 65536 vertices', () => {
    expect(createMeshIndexAttribute(new Uint32Array([0, 1, 2]), 100, 1).array).toBeInstanceOf(
      Uint16Array
    );
  });

  it('uses Uint32Array at and above 65536 vertices', () => {
    expect(createMeshIndexAttribute(new Uint32Array([0, 1, 2]), 65536, 1).array).toBeInstanceOf(
      Uint32Array
    );
  });

  it('keys the dtype on vertexCount, NOT on the largest index present', () => {
    // vertexCount is fixed for the node; the largest index actually drawn changes with
    // the slice. Keying off the observed maximum would let the dtype differ between
    // epochs, defeating the buffer reuse — and re-binding a drawn geometry's index with
    // a different dtype is the attribute-identity change WebGPU does not tolerate.
    const sparse = new Uint32Array([0, 1, 2]); // max index 2, but a big node
    expect(createMeshIndexAttribute(sparse, 200_000, 1).array).toBeInstanceOf(Uint32Array);
    const dense = new Uint32Array([60000, 60001, 60002]); // large indices, small node
    expect(createMeshIndexAttribute(dense, 65535, 1).array).toBeInstanceOf(Uint16Array);
  });
});

describe('createMeshGeometry', () => {
  const input = () => ({
    position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    positionChanged: true,
    indices: new Uint32Array([0, 1, 2]),
    colors: null,
    vertexCount: 3,
    faceCount: 1,
  });

  it('binds position, color and an index, and computes bounds', () => {
    const g = createMeshGeometry(input());
    expect(g.getAttribute('position').itemSize).toBe(3);
    expect(g.getAttribute('color')).toBeDefined();
    expect(g.index?.count).toBe(3);
    expect(g.boundingBox).not.toBeNull();
    expect(g.boundingSphere).not.toBeNull();
  });

  it('always binds color, even with no colors array', () => {
    // Bound-not-omitted is the contract; see createMeshDefaultColorAttribute.
    expect(createMeshGeometry(input()).getAttribute('color')).toBeDefined();
  });

  it('does NOT bind normal or aScalar in this phase', () => {
    // Nothing reads them until the material pair lands, and binding an attribute
    // with no consumer would mean picking its WebGPU-safe dtype with no shader to
    // validate against. They join the set at CREATION time in that phase, which is a
    // new build rather than a runtime mutation of a live geometry.
    const g = createMeshGeometry(input());
    expect(g.getAttribute('normal')).toBeUndefined();
    expect(g.getAttribute('aScalar')).toBeUndefined();
  });
});

describe('updateMeshGeometry', () => {
  function seeded(): THREE.BufferGeometry {
    return createMeshGeometry({
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
  }

  // The real production placeholder from `createEmptyMeshNode`: one vertex, no
  // indices, no colors. This is the state `updateMeshGeometry` first sees, so the
  // color-install guard must fire against it (unlike `seeded()`, which is already
  // 3-vertex).
  function placeholder(): THREE.BufferGeometry {
    return createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
    });
  }

  it('replaces the index without touching the position buffer on a slice move', () => {
    // The whole point of the no-compaction design: a slice change rewrites the index
    // buffer alone, and the vertex buffers stay uploaded.
    const g = seeded();
    const positionBefore = g.getAttribute('position');
    updateMeshGeometry(g, {
      position: positionBefore.array as Float32Array,
      positionChanged: true,
      indices: new Uint32Array([]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(g.getAttribute('position')).toBe(positionBefore);
    // Nothing drawn is an empty DRAW RANGE over the capacity buffer, not a
    // zero-length index — the index is allocated once at the node's face count so that
    // replacing it per epoch (which leaks its GPU buffer) never happens.
    expect(g.drawRange.count).toBe(0);
    expect(g.index?.count).toBe(3); // capacity, unchanged
  });

  it('recomputes bounds when position is replaced', () => {
    // Re-uploading position does NOT invalidate three's cached bounds, which
    // `frustumCulled` and the raycaster broad phase both consult — and the
    // display-space AABB genuinely changes under an axis permutation. Skipping the
    // recompute makes a permuted mesh vanish from the frustum test while still being
    // "loaded".
    const g = seeded();
    const before = g.boundingSphere!.radius;
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 100, 0, 0, 0, 100, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(g.boundingSphere!.radius).toBeGreaterThan(before);
  });

  it('does not recompute bounds when position is unchanged', () => {
    // Identity alone cannot prove this: three's `computeBoundingSphere()` mutates
    // the existing `Sphere` in place, so `toBe(sphere)` would hold even if the
    // recompute ran. Spying on the compute calls is the real assertion.
    const g = seeded();
    // Prime the geometry's upload stamps with one real update first — a
    // freshly-created geometry never stamped `meshUploadedVertexCount`, so the very
    // next call would see `countChanged` fire regardless of `positionChanged`.
    updateMeshGeometry(g, {
      position: g.getAttribute('position').array as Float32Array,
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    const computeBox = vi.spyOn(g, 'computeBoundingBox');
    const computeSphere = vi.spyOn(g, 'computeBoundingSphere');
    updateMeshGeometry(g, {
      position: g.getAttribute('position').array as Float32Array,
      positionChanged: false,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(computeBox).not.toHaveBeenCalled();
    expect(computeSphere).not.toHaveBeenCalled();
  });

  it('bounds the position in-place upload to the written prefix', () => {
    // Mirrors the index-buffer range assertion in `applyMeshIndices` — a byte-count
    // vs. element-count mistake here is a real GL `INVALID_VALUE` at draw time,
    // invisible in jsdom without asserting the range itself.
    const g = seeded();
    const positionAttr = g.getAttribute('position') as THREE.BufferAttribute;
    const next = new Float32Array([0, 0, 0, 5, 0, 0, 0, 5, 0]);
    updateMeshGeometry(g, {
      position: next,
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(positionAttr.updateRanges).toEqual([{ start: 0, count: next.length }]);

    // Two commits without an intervening render must not stack ranges — a real
    // renderer clears them after upload, so this is hygiene rather than a
    // correctness bug, but unbounded growth between frames is not left to chance.
    const third = new Float32Array([0, 0, 0, 9, 0, 0, 0, 9, 0]);
    updateMeshGeometry(g, {
      position: third,
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(positionAttr.updateRanges).toHaveLength(1);
  });

  it('rebinds position once at ladder level 0 from the real 1-vertex placeholder, then writes level 1 in place', () => {
    // Every other ladder test in this file seeds `createMeshGeometry` WITH
    // `capacityVertexCount` already set. Production never does that:
    // `create-mesh-node.ts` builds the real placeholder — `placeholder()` above —
    // with NO capacity, so the real level-0 commit goes through the REBIND branch
    // instead of the in-place-copy path every other ladder test exercises from its
    // second commit on.
    const g = placeholder();
    const totals = { vertices: 20, faces: 10 };
    const level0Position = new Float32Array(10 * 3).fill(1);
    const rebuiltLevel0 = updateMeshGeometry(g, {
      position: level0Position,
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 10,
      faceCount: 5,
      vertexCountGrows: true,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
    expect(rebuiltLevel0).toBe(true); // the placeholder → real-buffer grow-rebind
    const positionAttr = g.getAttribute('position') as THREE.BufferAttribute;
    expect(positionAttr.count).toBe(totals.vertices); // capacity-sized, not 10

    const level1Position = new Float32Array(20 * 3).fill(1);
    level1Position[15 * 3] = 42; // a distinctive value in the newly-revealed tail
    const rebuiltLevel1 = updateMeshGeometry(g, {
      position: level1Position,
      positionChanged: true,
      indices: new Uint32Array(Array.from({ length: 30 }, (_, i) => i % 20)),
      colors: null,
      vertexCount: 20,
      faceCount: 10,
      vertexCountGrows: true,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
    expect(rebuiltLevel1).toBe(false); // written in place, not rebound
    expect(g.getAttribute('position')).toBe(positionAttr);
    expect((g.getAttribute('position').array as Float32Array)[15 * 3]).toBe(42);
  });

  it('keeps the length-change warning for live nodes but not the placeholder grow', () => {
    // Every mesh's first commit grows the 1-vertex placeholder to the real buffer —
    // that is the designed path, not an anomaly, and must not log a warning per
    // mesh. A LIVE node changing vertex count is what the warning exists for.
    const warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const g = placeholder();
      updateMeshGeometry(g, {
        position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        positionChanged: true,
        indices: new Uint32Array([0, 1, 2]),
        colors: null,
        vertexCount: 3,
        faceCount: 1,
      });
      expect(warn).not.toHaveBeenCalled();

      updateMeshGeometry(g, {
        position: new Float32Array(18),
        positionChanged: true,
        indices: new Uint32Array([0, 1, 2]),
        colors: null,
        vertexCount: 6,
        faceCount: 2,
      });
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      warn.mockRestore();
    }
  });

  it('installs authored colors on the first commit, growing off the placeholder', () => {
    // The node is born with the 1-vertex placeholder color; the first real commit
    // has to bind the authored per-vertex colors here or they never reach the
    // shader. Before the fix `color` stayed the 1-vertex placeholder.
    const g = placeholder();
    const rebuilt = updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]),
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    const color = g.getAttribute('color');
    expect(color.count).toBe(3);
    expect(color.itemSize).toBe(4); // uint8 RGB padded to RGBA
    expect(color.normalized).toBe(true);
    expect(Array.from(color.array).slice(0, 4)).toEqual([255, 0, 0, 255]);
    // A vertex-attribute rebind happened (position grow + color install), so the
    // commit must evict three's stale WebGPU RenderObject cache.
    expect(rebuilt).toBe(true);
  });

  it('installs the default-white attribute for null colors from the placeholder', () => {
    const g = placeholder();
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    const color = g.getAttribute('color');
    expect(color.count).toBe(3);
    expect(color.itemSize).toBe(3);
    expect(Array.from(color.array)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1]);
  });

  it('installs authored colors for a one-vertex mesh, where counts cannot tell', () => {
    // A 1-vertex mesh has the SAME vertex count as the placeholder, so the
    // count-mismatch test can never fire for it — first-commit detection has to
    // come from the explicit installed marker, or the authored color silently
    // stays placeholder white.
    const g = placeholder();
    const rebuilt = updateMeshGeometry(g, {
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array([0, 0, 0]),
      colors: new Uint8Array([255, 0, 0]),
      colorComponents: 3,
      vertexCount: 1,
      faceCount: 1,
    });
    const color = g.getAttribute('color') as THREE.BufferAttribute;
    expect(color.itemSize).toBe(4); // uint8 RGB padded, i.e. the AUTHORED attribute
    expect(Array.from(color.array).slice(0, 4)).toEqual([255, 0, 0, 255]);
    expect(rebuilt).toBe(true);

    // Installed once: the marker keeps the next commit from re-binding it.
    updateMeshGeometry(g, {
      position: new Float32Array(3),
      positionChanged: false,
      indices: new Uint32Array([0, 0, 0]),
      colors: new Uint8Array([255, 0, 0]),
      colorComponents: 3,
      vertexCount: 1,
      faceCount: 1,
    });
    expect(g.getAttribute('color')).toBe(color);
  });

  it('leaves the color buffer untouched on a subsequent slice move', () => {
    // The guard is keyed off the capacity plus the `meshColorsInstalled` marker
    // (see the color guard in `updateMeshGeometry`), and currency past that is
    // tracked by the SOURCE array's identity (`isAttributeCurrent`, #1522), not by
    // vertexCount alone — a later slice move at the SAME vertexCount must NOT
    // re-create/re-upload the buffer, only the index rebuilds. Tying color to
    // position identity would fail this differently: `projectMeshTo3D` REUSES one
    // loader-owned position buffer across epochs (`MeshGeometryConfig.position`),
    // so position identity alone cannot even signal a slice move, let alone gate
    // color off it correctly.
    //
    // The SAME `colors` array is passed both times, matching production: the
    // whole-node loader serves one cached `LoadedMeshData` for the node's
    // lifetime, so `data.colors` is identity-stable across epochs exactly like
    // `data.normals`/`data.scalars`. A fresh array with equal content, unlike
    // here, legitimately re-uploads — see the next test, the colour mirror of
    // `replaceVertexAttribute`'s "new array, same length" case for
    // `normal`/`aScalar`.
    const g = placeholder();
    const colors = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]);
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    const colorBefore = g.getAttribute('color') as THREE.BufferAttribute;
    const versionBefore = colorBefore.version;
    const rebuilt = updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0]),
      positionChanged: true,
      indices: new Uint32Array([2, 1, 0]),
      colors,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(g.getAttribute('color')).toBe(colorBefore); // same object, not re-created
    // No re-upload either: an unchanged `version` proves the buffer wasn't dirtied.
    expect((g.getAttribute('color') as THREE.BufferAttribute).version).toBe(versionBefore);
    // Nothing rebound → the commit skips the WebGPU RenderObject eviction.
    expect(rebuilt).toBe(false);
  });

  it('re-uploads in place when a NEW colours array with the same content arrives', () => {
    // The colour mirror of the `normal`/`aScalar` "copies in place when a NEW
    // array of the same length arrives" case above. Dropping the
    // `meshColorsSource === colors` term from the currency check would report
    // this as already-current (since count and content are unchanged), leaving
    // the version un-bumped — currently passes the whole suite without this test.
    const g = placeholder();
    const first = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]);
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: first,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    const colorAttr = g.getAttribute('color') as THREE.BufferAttribute;
    const versionBefore = colorAttr.version;
    const second = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]); // same content, new array
    const rebuilt = updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: false,
      indices: new Uint32Array([0, 1, 2]),
      colors: second,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(g.getAttribute('color')).toBe(colorAttr); // same object, not rebound
    expect(colorAttr.version).toBeGreaterThan(versionBefore);
    expect(rebuilt).toBe(false);
  });
});

describe('applyMeshIndices — the index buffer is allocated once per node', () => {
  /** The one-vertex placeholder `createEmptyMeshNode` attaches. */
  function placeholder(): THREE.BufferGeometry {
    return createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
    });
  }

  const real = {
    position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    // These cases are about the INDEX buffer; position is grown once on the first
    // call, so the flag stays true throughout rather than modelling an epoch.
    positionChanged: true,
    colors: null,
    vertexCount: 3,
    faceCount: 1,
  };

  it('keeps the SAME index attribute across epochs', () => {
    // The leak guard. Three caches attribute buffers in a WeakMap keyed by the
    // attribute and only deletes a GPU buffer via `WebGLAttributes.remove()`, which
    // nothing calls when `geometry.index` is replaced — so a per-epoch `setIndex`
    // orphans one index buffer per slice move, unfreed for the tab's lifetime. Mesh is
    // the only geometry type that rewrites its index per epoch.
    const g = placeholder();
    updateMeshGeometry(g, { ...real, indices: new Uint32Array([0, 1, 2]) });
    const indexAttr = g.index;
    expect(indexAttr).not.toBeNull();

    for (const indices of [new Uint32Array(0), new Uint32Array([0, 1, 2]), new Uint32Array(0)]) {
      updateMeshGeometry(g, { ...real, indices });
      expect(g.index).toBe(indexAttr);
      expect(g.drawRange.count).toBe(indices.length);
    }
  });

  it('sizes capacity from faceCount even when the FIRST epoch is culled', () => {
    // The realistic scrub a visible-count-sized capacity gets wrong: a mesh loaded at
    // a timepoint where nothing is visible, then scrubbed to where it is. Sizing from
    // the first epoch's `indices.length` (0) means the next epoch cannot fit and calls
    // `setIndex`, orphaning a buffer per move. Caught by mutation — without this case
    // `capacity = indices.length` passed the whole suite.
    const g = placeholder();
    updateMeshGeometry(g, { ...real, indices: new Uint32Array(0) });
    const indexAttr = g.index;
    expect(indexAttr?.count).toBe(3); // faceCount * 3, not 0

    updateMeshGeometry(g, { ...real, indices: new Uint32Array([0, 1, 2]) });
    expect(g.index).toBe(indexAttr);
    expect(g.drawRange.count).toBe(3);
  });

  it('bounds the index upload to the rewritten prefix', () => {
    // Reusing the buffer must not mean re-uploading all of it: a large mesh with a
    // small visible set would then move far more bytes per slice change than the old
    // reallocating path did, trading the leak for a bandwidth regression. The classic
    // WebGL backend honours update ranges (the WebGPU ones re-upload in full).
    const g = placeholder();
    updateMeshGeometry(g, { ...real, indices: new Uint32Array(0) });
    updateMeshGeometry(g, { ...real, indices: new Uint32Array([0, 1, 2]) });
    const index = g.index!;
    // `needsUpdate` is setter-only in three (it just bumps `version`), so the
    // observable is the version counter, not a readable flag.
    expect(index.version).toBeGreaterThan(0);
    expect(index.updateRanges).toEqual([{ start: 0, count: 3 }]);

    // Two epochs in a row without a render in between must not stack duplicate ranges.
    // A real renderer clears them after each upload, so this is hygiene rather than a
    // correctness bug — but unbounded growth between frames is not left to chance.
    updateMeshGeometry(g, { ...real, indices: new Uint32Array([0, 1, 2]) });
    expect(index.updateRanges).toHaveLength(1);
  });

  it('does not dirty the reused index buffer on a fully-culled epoch', () => {
    // An empty epoch rewrites zero indices, so it must not set `needsUpdate` at
    // all: with an EMPTY update-range list the WebGL backend uploads the WHOLE
    // attribute, so flagging an update here would re-upload the full
    // capacity-sized buffer on every scrub through empty slices. Only the draw
    // range has to change.
    const g = placeholder();
    updateMeshGeometry(g, { ...real, indices: new Uint32Array([0, 1, 2]) });
    const index = g.index!;
    const versionBefore = index.version;
    updateMeshGeometry(g, { ...real, indices: new Uint32Array(0) });
    expect(index.version).toBe(versionBefore); // not dirtied → nothing uploads
    expect(g.drawRange.count).toBe(0); // but nothing draws either
  });

  it('does not report an index rebind as a vertex-attribute rebuild', () => {
    // `attributesRebuilt` drives the WebGPU RenderObject eviction and is about VERTEX
    // attributes. Once the index buffer is stable, a slice move rebinds nothing, so a
    // scrub must not keep evicting three's render-object cache.
    const g = placeholder();
    updateMeshGeometry(g, { ...real, indices: new Uint32Array([0, 1, 2]) });
    const rebuilt = updateMeshGeometry(g, { ...real, indices: new Uint32Array(0) });
    expect(rebuilt).toBe(false);
  });
});

describe('the `normal` / `aScalar` attributes — replaced, never added or removed', () => {
  /** A minimal 3-vertex config; `normals`/`scalars` are supplied per case. */
  const cfg = (over: Record<string, unknown> = {}) => ({
    position: new Float32Array(9),
    positionChanged: true,
    indices: new Uint32Array([0, 1, 2]),
    colors: null,
    vertexCount: 3,
    faceCount: 1,
    ...over,
  });

  it('binds them at CREATION when the arrays are supplied', () => {
    const geometry = createMeshGeometry(
      cfg({ normals: new Float32Array(9), scalars: new Float32Array(3) })
    );
    expect(geometry.getAttribute('normal').itemSize).toBe(3);
    expect(geometry.getAttribute('aScalar').itemSize).toBe(1);
    expect(geometry.userData.hasScalars).toBe(true);
  });

  it('never ADDS one that creation left out', () => {
    // The WebGPU vertex layout is cached from the attribute set at first draw and
    // never rebuilt, so growing the set on a later commit renders the node black on
    // that backend only. Existence is a per-node constant (`has_normals` /
    // `has_scalars`); a commit that finds the attribute missing must leave it missing.
    const geometry = createMeshGeometry(cfg());
    expect(geometry.getAttribute('normal')).toBeUndefined();
    updateMeshGeometry(geometry, cfg({ normals: new Float32Array(9) }));
    expect(geometry.getAttribute('normal')).toBeUndefined();
  });

  it('never REMOVES one when an epoch supplies no data', () => {
    // The mirror case, and it is the one a `displayDims` change hits: a frame
    // mismatch makes stored normals meaningless, but the fix is the shader VARIANT,
    // not unbinding the attribute (§3.4).
    const geometry = createMeshGeometry(cfg({ normals: new Float32Array(9) }));
    updateMeshGeometry(geometry, cfg({ normals: null }));
    expect(geometry.getAttribute('normal')).toBeDefined();
  });

  it('grows the 1-vertex placeholder stub on the first real commit, and reports it', () => {
    // The expected first commit of every normal-bearing mesh: the placeholder is
    // 1-vertex, the real buffer is N. A `setAttribute` rebind must report
    // `attributesRebuilt` so the commit evicts three's cached WebGPU RenderObject.
    const geometry = createMeshGeometry(
      cfg({ position: new Float32Array(3), vertexCount: 1, normals: new Float32Array(3) })
    );
    const real = new Float32Array([0, 0, 1, 0, 1, 0, 1, 0, 0]);
    const rebuilt = updateMeshGeometry(geometry, cfg({ normals: real }));
    expect(rebuilt).toBe(true);
    expect(geometry.getAttribute('normal').count).toBe(3);
    expect(Array.from(geometry.getAttribute('normal').array as Float32Array)).toEqual([
      0, 0, 1, 0, 1, 0, 1, 0, 0,
    ]);
  });

  it('copies in place when a NEW array of the same length arrives — no rebind', () => {
    // The re-fetch case (dispose/reload hands over fresh buffers): content genuinely
    // changed, so it must copy and flag, but without rebinding the attribute object.
    const geometry = createMeshGeometry(cfg({ normals: new Float32Array(9) }));
    const attr = geometry.getAttribute('normal') as THREE.BufferAttribute;
    const before = attr.version;
    const next = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const rebuilt = updateMeshGeometry(geometry, cfg({ normals: next, positionChanged: false }));
    expect(rebuilt).toBe(false);
    expect(geometry.getAttribute('normal')).toBe(attr); // same object
    expect(Array.from(attr.array as Float32Array)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(attr.version).toBeGreaterThan(before);
  });

  it('bounds the normal in-place upload to the written prefix', () => {
    // Mirrors the index-buffer range assertion in `applyMeshIndices`.
    const geometry = createMeshGeometry(cfg({ normals: new Float32Array(9) }));
    const attr = geometry.getAttribute('normal') as THREE.BufferAttribute;
    const next = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    updateMeshGeometry(geometry, cfg({ normals: next, positionChanged: false }));
    expect(attr.updateRanges).toEqual([{ start: 0, count: next.length }]);

    // Two commits without an intervening render must not stack ranges.
    const third = new Float32Array([0, 1, 0, 1, 0, 0, 0, 0, 1]);
    updateMeshGeometry(geometry, cfg({ normals: third, positionChanged: false }));
    expect(attr.updateRanges).toHaveLength(1);
  });

  it('re-uploads NOTHING when the same array arrives again — the steady state', () => {
    // The whole-node loader serves one cached `LoadedMeshData` for the node's
    // lifetime and the commit passes `data.normals` / `data.scalars` every epoch, so
    // after the first commit the identity is ALWAYS equal. Flagging `needsUpdate`
    // there would re-upload the entire normal (V·12 B) and scalar (V·4 B) buffers on
    // every slice move, with no update ranges — real bandwidth during a scrub, and a
    // contradiction of the no-compaction rule that only the index rebuilds.
    //
    // Observed through `attribute.version`, since `needsUpdate` is a write-only
    // setter in three (it has no getter and just bumps `version`).
    const normals = new Float32Array([0, 0, 1, 0, 1, 0, 1, 0, 0]);
    const scalars = new Float32Array([0.25, 0.5, 0.75]);
    const geometry = createMeshGeometry(cfg({ normals, scalars }));
    const normalAttr = geometry.getAttribute('normal') as THREE.BufferAttribute;
    const scalarAttr = geometry.getAttribute('aScalar') as THREE.BufferAttribute;
    const normalVersion = normalAttr.version;
    const scalarVersion = scalarAttr.version;

    // Three consecutive slice moves: same arrays, only the visible index changes.
    for (const indices of [
      new Uint32Array([0, 1, 2]),
      new Uint32Array([]),
      new Uint32Array([0, 1, 2]),
    ]) {
      updateMeshGeometry(geometry, cfg({ normals, scalars, indices, positionChanged: false }));
    }

    expect(normalAttr.version, 'normal buffer re-uploaded on a slice move').toBe(normalVersion);
    expect(scalarAttr.version, 'scalar buffer re-uploaded on a slice move').toBe(scalarVersion);
  });

  it('on a LADDER, re-uploads nothing within a level but bumps on a new one (regression for #1522 fix A)', () => {
    // Before the fix, currency for `normal`/`aScalar` was tracked by comparing
    // `existing.array` (the capacity-sized bound buffer) against `data` directly.
    // Once a ladder's first level rebinds that buffer to a capacity-sized COPY,
    // `existing.array !== data` is permanently true — including for the level
    // that produced it — so every slice move re-uploaded the WHOLE capacity
    // buffer with no update ranges, on every level, forever. `isAttributeCurrent`
    // fixes that by tracking the SOURCE array's own identity instead.
    const totals = { vertices: 20, faces: 10 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      normals: new Float32Array(3),
      scalars: new Float32Array(1),
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    function commitLevel(vertexCount: number, normals: Float32Array, scalars: Float32Array): void {
      const faceCount = Math.floor((vertexCount / totals.vertices) * totals.faces);
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      updateMeshGeometry(geometry, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors: null,
        normals,
        scalars,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }

    const level0Normals = new Float32Array(10 * 3).fill(1);
    const level0Scalars = new Float32Array(10).fill(0.5);
    commitLevel(10, level0Normals, level0Scalars);
    const normalAttr = geometry.getAttribute('normal') as THREE.BufferAttribute;
    const scalarAttr = geometry.getAttribute('aScalar') as THREE.BufferAttribute;
    const normalVersionLevel0 = normalAttr.version;
    const scalarVersionLevel0 = scalarAttr.version;

    // Several sweeps at the SAME level, with the SAME arrays (matching production:
    // the whole-node loader's memoized `LoadedMeshData` hands back identical
    // `data.normals`/`data.scalars` every sweep) — no version change expected.
    commitLevel(10, level0Normals, level0Scalars);
    commitLevel(10, level0Normals, level0Scalars);
    expect(normalAttr.version).toBe(normalVersionLevel0);
    expect(scalarAttr.version).toBe(scalarVersionLevel0);

    // A new level — a longer prefix, fresh arrays — must bump both.
    const level1Normals = new Float32Array(20 * 3).fill(2);
    const level1Scalars = new Float32Array(20).fill(0.75);
    commitLevel(20, level1Normals, level1Scalars);
    expect(normalAttr.version).toBeGreaterThan(normalVersionLevel0);
    expect(scalarAttr.version).toBeGreaterThan(scalarVersionLevel0);
    // Same attribute objects throughout — never orphaned (#1521).
    expect(geometry.getAttribute('normal')).toBe(normalAttr);
    expect(geometry.getAttribute('aScalar')).toBe(scalarAttr);
  });
});

describe('capacity sizing — a reveal ladder must not orphan GPU buffers (#1521)', () => {
  /** One commit of a growing ladder, against a node whose TOTAL is fixed. */
  function commit(
    geometry: THREE.BufferGeometry,
    vertexCount: number,
    faceCount: number,
    totals: { vertices: number; faces: number }
  ): void {
    const indices = new Uint32Array(faceCount * 3);
    for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
    updateMeshGeometry(geometry, {
      position: new Float32Array(vertexCount * 3),
      positionChanged: true,
      indices,
      colors: new Uint8Array(vertexCount * 3),
      colorComponents: 3,
      normals: new Float32Array(vertexCount * 3),
      scalars: new Float32Array(vertexCount),
      vertexCount,
      faceCount,
      vertexCountGrows: true,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
  }

  function ids(geometry: THREE.BufferGeometry): Record<string, unknown> {
    return {
      index: geometry.index,
      position: geometry.getAttribute('position'),
      color: geometry.getAttribute('color'),
      normal: geometry.getAttribute('normal'),
      aScalar: geometry.getAttribute('aScalar'),
    };
  }

  it('keeps every attribute object identical as the revealed prefix grows', () => {
    // The defect: three frees a replaced attribute's GL buffer from NOWHERE — not
    // on replacement, and not on dispose (only what is still bound is freed). So a
    // rebind per level orphans the previous level's buffers for the session.
    // Identity is therefore the thing to assert: an object that never changes is a
    // buffer that is never orphaned.
    const totals = { vertices: 140_000, faces: 70_000 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      normals: new Float32Array(3),
      scalars: new Float32Array(1),
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    commit(geometry, 40_000, 20_000, totals); // level 0 — installs the real buffers
    const afterFirst = ids(geometry);
    commit(geometry, 90_000, 45_000, totals); // level 1
    commit(geometry, 140_000, 70_000, totals); // level 2 — the full ladder

    for (const [name, attr] of Object.entries(afterFirst)) {
      expect(geometry.getAttribute(name) ?? geometry.index).toBe(attr);
    }
  });

  it('sizes those buffers to the ladder TOTAL, not the first committed prefix', () => {
    // The identity assertion above passes trivially if the buffers are too small
    // and the later levels are silently truncated. This is its control.
    const totals = { vertices: 140_000, faces: 70_000 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      normals: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
    commit(geometry, 40_000, 20_000, totals);

    expect(geometry.getAttribute('position').count).toBe(140_000);
    expect(geometry.index!.count).toBe(70_000 * 3);
    // And the dtype is chosen from the TOTAL, so it cannot flip Uint16 → Uint32 on
    // an already-drawn geometry when the reveal crosses 65,536 vertices — the
    // attribute-identity change the WebGPU backend does not tolerate.
    expect(geometry.index!.array).toBeInstanceOf(Uint32Array);
  });

  it('leaves an UNLADDERED mesh byte-identical: no capacity, no copy', () => {
    // The no-op guarantee. An ordinary mesh passes no capacity, so every path must
    // behave exactly as before — including the zero-copy colour wrap, where the
    // bound buffer IS the caller's array.
    const colors = new Float32Array(6);
    const attr = createMeshColorAttribute(colors, 3, 2);
    expect(attr.array).toBe(colors);
  });
});

describe('a superseded ladder commit must not strand the newly revealed vertices (#1522 fix B)', () => {
  it('uploads a grown prefix even when positionChanged is false and the key is unchanged', () => {
    // The bug: an aborted commit lets `projectMeshTo3D` stamp the loader-owned
    // scratch's `displayDimsKey` (so the NEXT sweep reads `positionChanged ===
    // false`) while the geometry itself never received that level's data. The key
    // is also unchanged (same `displayDims` throughout), so `keyChanged` cannot
    // see the gap either — only a stamped vertex COUNT can. This reproduces the
    // second sweep directly: same key, `positionChanged: false`, but a longer
    // prefix than the geometry has ever uploaded.
    const totals = { vertices: 20, faces: 10 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    // Level 0: a real commit, the normal way — positionChanged true, a key stamped.
    const level0Position = new Float32Array(10 * 3).fill(1);
    updateMeshGeometry(geometry, {
      position: level0Position,
      positionChanged: true,
      positionKey: '0,1,2',
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 10,
      faceCount: 5,
      vertexCountGrows: true,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
    const positionAttr = geometry.getAttribute('position') as THREE.BufferAttribute;

    // Level 1: the superseded-then-resumed sweep. Vertex 15 (index 5 of the newly
    // revealed 10-vertex tail) carries a distinctive nonzero value; a stranded
    // commit would leave it at the zero `atCapacity` filled the buffer with.
    const level1Position = new Float32Array(20 * 3).fill(1);
    level1Position[15 * 3] = 42;
    updateMeshGeometry(geometry, {
      position: level1Position,
      positionChanged: false, // the projection saw no displayDims change
      positionKey: '0,1,2', // same key — keyChanged cannot see the gap either
      indices: new Uint32Array(Array.from({ length: 30 }, (_, i) => i % 20)),
      colors: null,
      vertexCount: 20,
      faceCount: 10,
      vertexCountGrows: true,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    // The attribute object is never rebound — only its contents change.
    expect(geometry.getAttribute('position')).toBe(positionAttr);
    // The newly revealed vertex reached the buffer instead of staying at the
    // origin-collapsing zero fill.
    expect((geometry.getAttribute('position').array as Float32Array)[15 * 3]).toBe(42);
  });
});

describe('color refresh — a reveal ladder must update its colour prefix, alpha included (#1522)', () => {
  /**
   * Colours for a ladder committed up through `counts[i]` vertices, `values[i]`
   * per component (RGB, or RGBA when `alphas` is given). Mirrors how a real
   * ladder's levels concatenate: an earlier level's bytes are carried forward
   * unchanged, and only the newly revealed tail gets the new level's fill —
   * unlike a naive test double that re-fills the whole array every commit, which
   * would pass even if the "revealed" range were never touched.
   */
  function ladderColors<A extends Uint8Array | Uint16Array | Float32Array>(
    ctor: new (n: number) => A,
    counts: number[],
    values: number[],
    components: 3 | 4,
    alphas?: number[]
  ): A {
    const total = counts[counts.length - 1]!;
    const arr = new ctor(total * components);
    let start = 0;
    for (let i = 0; i < counts.length; i++) {
      for (let v = start; v < counts[i]!; v++) {
        const base = v * components;
        arr[base] = values[i]!;
        arr[base + 1] = values[i]!;
        arr[base + 2] = values[i]!;
        if (components === 4) arr[base + 3] = alphas ? alphas[i]! : 255;
      }
      start = counts[i]!;
    }
    return arr;
  }

  it('writes a growing uint8 RGB ladder into the SAME buffer, alpha included', () => {
    // The bug: once `color` is bound at the node's capacity, every level binds at
    // the same count, so the old count-only guard treated every level after the
    // first as "already installed" and never touched the buffer again — a
    // revealed vertex kept the zero-filled slot it was born with. Alpha is the
    // load-bearing component here: `vAlpha = sanitizeAlpha(color.a)` is the
    // mesh's ENTIRE coverage term (shader-glsl.ts), so a zero there makes the
    // revealed surface invisible, not merely the wrong colour.
    const totals = { vertices: 300, faces: 100 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    function commit(vertexCount: number, counts: number[], values: number[]): void {
      const faceCount = Math.floor((vertexCount / totals.vertices) * totals.faces);
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      updateMeshGeometry(geometry, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors: ladderColors(Uint8Array, counts, values, 3),
        colorComponents: 3,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }

    commit(100, [100], [11]);
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    commit(200, [100, 200], [11, 22]);
    commit(300, [100, 200, 300], [11, 22, 33]);

    // The attribute OBJECT never changed — the buffer was never orphaned (#1521).
    expect(geometry.getAttribute('color')).toBe(colorAttr);
    const color = geometry.getAttribute('color') as THREE.BufferAttribute;
    const at = (v: number): number[] => Array.from(color.array).slice(v * 4, v * 4 + 4);
    expect(at(0)).toEqual([11, 11, 11, 255]);
    expect(at(150)).toEqual([22, 22, 22, 255]);
    expect(at(250)).toEqual([33, 33, 33, 255]);
  });

  it('bounds the colour refresh upload to the committed prefix', () => {
    // Mirrors the index-buffer range assertion in `applyMeshIndices` — a
    // byte-count vs. element-count mistake here is a real GL `INVALID_VALUE` at
    // draw time, invisible in jsdom without asserting the range itself.
    const g = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
    });
    const colors = new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]);
    // The install (placeholder → real buffer) rebinds via `setAttribute` and needs
    // no range of its own — the whole new buffer uploads regardless. The REFRESH
    // this test is about only starts on the second commit.
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    const colorAttr = g.getAttribute('color') as THREE.BufferAttribute;

    const second = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: false,
      indices: new Uint32Array([0, 1, 2]),
      colors: second,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(colorAttr.updateRanges).toEqual([{ start: 0, count: 3 * colorAttr.itemSize }]);

    // A THIRD commit with no intervening render must not stack a second range.
    const third = new Uint8Array([9, 8, 7, 6, 5, 4, 3, 2, 1]);
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: false,
      indices: new Uint32Array([0, 1, 2]),
      colors: third,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    expect(colorAttr.updateRanges).toEqual([{ start: 0, count: 3 * colorAttr.itemSize }]);
  });

  it('carries the AUTHORED alpha for a later-revealed vertex in an RGBA ladder', () => {
    // The RGBA sibling of the case above: no padding, so the guard's format
    // check is exercised at itemSize 4 natively rather than via the pad path.
    const totals = { vertices: 200, faces: 100 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    function commit(
      vertexCount: number,
      counts: number[],
      values: number[],
      alphas: number[]
    ): void {
      const faceCount = Math.floor((vertexCount / totals.vertices) * totals.faces);
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      updateMeshGeometry(geometry, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors: ladderColors(Uint8Array, counts, values, 4, alphas),
        colorComponents: 4,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }

    commit(100, [100], [40], [128]);
    commit(200, [100, 200], [40, 80], [128, 200]);

    const color = geometry.getAttribute('color') as THREE.BufferAttribute;
    // Vertex 150 was only revealed by the second commit — its alpha must be the
    // AUTHORED 200, not the 0 a never-updated slot would carry.
    expect(color.array[150 * 4 + 3]).toBe(200);
  });

  it('carries the AUTHORED channel values for a later-revealed vertex in a float32 RGB mesh', () => {
    // The float32 case is black-but-opaque when broken (the size-3 attribute's
    // `w = 1.0` default hides the alpha symptom), so this asserts RGB directly.
    const totals = { vertices: 200, faces: 100 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    function commit(vertexCount: number, counts: number[], values: number[]): void {
      const faceCount = Math.floor((vertexCount / totals.vertices) * totals.faces);
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      updateMeshGeometry(geometry, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors: ladderColors(Float32Array, counts, values, 3),
        colorComponents: 3,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }

    // 0.25 / 0.5 are exactly representable in float32, so the equality check
    // below isn't fighting binary rounding on top of the thing under test.
    commit(100, [100], [0.25]);
    commit(200, [100, 200], [0.25, 0.5]);

    const color = geometry.getAttribute('color') as THREE.BufferAttribute;
    expect(Array.from(color.array).slice(150 * 3, 150 * 3 + 3)).toEqual([0.5, 0.5, 0.5]);
  });

  it('re-uploads nothing for an unladdered mesh across slice moves, but bumps per ladder level', () => {
    // The steady-state control mirroring the `normal`/`aScalar` version check: an
    // UNLADDERED padded-RGB mesh driven through several slice moves with the
    // SAME colours array must not dirty the buffer, while a ladder — whose
    // committed prefix genuinely grows — must dirty it once per level.
    const colors = new Uint8Array([10, 10, 10, 20, 20, 20, 30, 30, 30]);
    const geometry = createMeshGeometry({
      position: new Float32Array(9),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors,
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    const versionAfterCreate = colorAttr.version;

    for (const indices of [
      new Uint32Array([0, 1, 2]),
      new Uint32Array([]),
      new Uint32Array([0, 1, 2]),
    ]) {
      updateMeshGeometry(geometry, {
        position: new Float32Array(9),
        positionChanged: false,
        indices,
        colors,
        colorComponents: 3,
        vertexCount: 3,
        faceCount: 1,
      });
    }
    expect(colorAttr.version, 'colour buffer re-uploaded on a slice move').toBe(versionAfterCreate);

    // Now grow a ladder from a fresh placeholder: each level's PREFIX genuinely
    // changes, so the version must advance every time.
    const totals = { vertices: 30, faces: 10 };
    const ladder = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
    function commitLevel(vertexCount: number, faceCount: number, byte: number): void {
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      updateMeshGeometry(ladder, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors: new Uint8Array(vertexCount * 3).fill(byte),
        colorComponents: 3,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }
    commitLevel(10, 3, 1);
    const ladderAttr = ladder.getAttribute('color') as THREE.BufferAttribute;
    const versionLevel0 = ladderAttr.version;
    commitLevel(20, 6, 2);
    expect(ladderAttr.version).toBeGreaterThan(versionLevel0);
    const versionLevel1 = ladderAttr.version;
    commitLevel(30, 10, 3);
    expect(ladderAttr.version).toBeGreaterThan(versionLevel1);
  });

  it("re-commits nothing when the SAME colours array is committed again after the ladder's last level", () => {
    // `refreshMeshColors` stamps its OWN currency (`meshColorsSource`/
    // `meshColorsCount`) after every write. Deleting those two stamp lines
    // passes the whole rest of the suite, because no other ladder test commits
    // the identical array object twice at the same level — every `commitLevel`
    // helper above allocates a fresh array per call. Without the stamp, the
    // pad loop and the version bump would re-run on every repeat commit too.
    const totals = { vertices: 300, faces: 100 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    function commit(vertexCount: number, colors: Uint8Array): void {
      const faceCount = Math.floor((vertexCount / totals.vertices) * totals.faces);
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      updateMeshGeometry(geometry, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors,
        colorComponents: 3,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }

    commit(100, ladderColors(Uint8Array, [100], [11], 3));
    const lastLevelColors = ladderColors(Uint8Array, [100, 300], [11, 33], 3);
    commit(300, lastLevelColors);
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    const versionAfterLastLevel = colorAttr.version;

    // The SAME array object, same vertexCount — a slice move at the ladder's
    // completed level, not a new one.
    commit(300, lastLevelColors);
    expect(colorAttr.version).toBe(versionAfterLastLevel);
    // And the padded bytes are still exactly what the last real write produced —
    // proof the pad loop did not silently re-run and (say) re-derive them wrong.
    expect(Array.from(colorAttr.array).slice(0, 4)).toEqual([11, 11, 11, 255]);
  });

  it('pads a growing Uint16 RGB ladder with the 65535 opaque alpha, mirroring the Uint8 case', () => {
    // The `uint16` pad path is otherwise untested for the ladder in-place writer —
    // only `createMeshColorAttribute`'s unit tests exercise it, and those never go
    // through `refreshMeshColors`.
    const totals = { vertices: 300, faces: 100 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    function commit(vertexCount: number, counts: number[], values: number[]): void {
      const faceCount = Math.floor((vertexCount / totals.vertices) * totals.faces);
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      updateMeshGeometry(geometry, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors: ladderColors(Uint16Array, counts, values, 3),
        colorComponents: 3,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }

    commit(100, [100], [111]);
    const colorAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    commit(300, [100, 300], [111, 222]);

    expect(geometry.getAttribute('color')).toBe(colorAttr); // never orphaned
    const at = (v: number): number[] => Array.from(colorAttr.array).slice(v * 4, v * 4 + 4);
    expect(at(0)).toEqual([111, 111, 111, 65535]);
    expect(at(150)).toEqual([222, 222, 222, 65535]);
  });

  it('leaves a not-yet-revealed vertex all zeros — the pad loop stops at vertexCount, not the capacity', () => {
    // The pad loop's bound is `v < vertexCount`; if it instead ran to the
    // capacity (or if it wrote past the committed prefix for any other reason)
    // this would fail. Regression for a mutation on the loop bound.
    const totals = { vertices: 300, faces: 100 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
    updateMeshGeometry(geometry, {
      position: new Float32Array(100 * 3),
      positionChanged: true,
      indices: new Uint32Array(30),
      colors: ladderColors(Uint8Array, [100], [11], 3),
      colorComponents: 3,
      vertexCount: 100,
      faceCount: 10,
      vertexCountGrows: true,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });
    const color = geometry.getAttribute('color') as THREE.BufferAttribute;
    // Vertex 150 is well past the 100-vertex committed prefix.
    expect(Array.from(color.array).slice(150 * 4, 150 * 4 + 4)).toEqual([0, 0, 0, 0]);
  });

  it('falls back to a rebind when the bound format cannot hold a new dtype, and reports true', () => {
    // The format-mismatch path: an installed `float32` RGB attribute cannot hold a
    // later `Uint8Array` RGB commit in place (different `constructor`, different
    // itemSize post-pad), so `refreshMeshColors` must report `false` and the
    // caller rebinds — the ONLY way a genuinely authored dtype change reaches the
    // shader instead of being silently dropped.
    const geometry = createMeshGeometry({
      position: new Float32Array(9),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    const before = geometry.getAttribute('color');
    const rebuilt = updateMeshGeometry(geometry, {
      position: new Float32Array(9),
      positionChanged: false,
      indices: new Uint32Array([0, 1, 2]),
      colors: new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80, 90]),
      colorComponents: 3,
      vertexCount: 3,
      faceCount: 1,
    });
    const after = geometry.getAttribute('color') as THREE.BufferAttribute;
    expect(after).not.toBe(before); // a NEW attribute object
    expect(after.itemSize).toBe(4); // uint8 RGB pads to RGBA
    expect(Array.from(after.array).slice(0, 4)).toEqual([10, 20, 30, 255]);
    expect(rebuilt).toBe(true);
  });

  it("returns false for a writing refresh — a commit must not evict three's RenderObject cache", () => {
    // `attributesRebuilt` (surfaced as this function's return value) drives the
    // WebGPU RenderObject eviction. A ladder level that WRITES into the
    // already-bound buffer (as opposed to installing it for the first time) must
    // report `false`, or every level would needlessly evict a live render object.
    const totals = { vertices: 300, faces: 100 };
    const geometry = createMeshGeometry({
      position: new Float32Array(3),
      positionChanged: true,
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
      faceCount: 0,
      capacityVertexCount: totals.vertices,
      capacityFaceCount: totals.faces,
    });

    function commit(vertexCount: number, counts: number[], values: number[]): boolean {
      const faceCount = Math.floor((vertexCount / totals.vertices) * totals.faces);
      const indices = new Uint32Array(faceCount * 3);
      for (let i = 0; i < indices.length; i++) indices[i] = i % vertexCount;
      return updateMeshGeometry(geometry, {
        position: new Float32Array(vertexCount * 3),
        positionChanged: true,
        indices,
        colors: ladderColors(Uint8Array, counts, values, 3),
        colorComponents: 3,
        vertexCount,
        faceCount,
        vertexCountGrows: true,
        capacityVertexCount: totals.vertices,
        capacityFaceCount: totals.faces,
      });
    }

    commit(100, [100], [11]); // level 0 — the install, may legitimately rebind
    const rebuiltOnWrite = commit(200, [100, 200], [11, 22]); // level 1 — a write
    expect(rebuiltOnWrite).toBe(false);
  });
});
