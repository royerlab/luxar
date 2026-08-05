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
    const g = seeded();
    const sphere = g.boundingSphere;
    updateMeshGeometry(g, {
      position: g.getAttribute('position').array as Float32Array,
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
      faceCount: 1,
    });
    // Same object identity — computeBoundingSphere() would have replaced it.
    expect(g.boundingSphere).toBe(sphere);
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

  it('leaves the color buffer untouched on a subsequent slice move', () => {
    // The guard is keyed off vertexCount, so once colors are installed (count 3) a
    // later slice move at the SAME vertexCount must NOT re-create/re-upload the
    // buffer — only the index rebuilds. Tying color to position identity would fail
    // this, since `projectMeshTo3D` reallocates position every call.
    const g = placeholder();
    updateMeshGeometry(g, {
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      positionChanged: true,
      indices: new Uint32Array([0, 1, 2]),
      colors: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]),
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
      colors: new Uint8Array([255, 0, 0, 0, 255, 0, 0, 0, 255]),
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
