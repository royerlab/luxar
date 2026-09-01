/**
 * The mesh reveal ladder's monitor surface.
 *
 * A progressive node must look like ONE loader to the data-loading monitor,
 * keyed by the node path — the `additive_<i>` levels are an implementation
 * detail. This file pins the roll-up and the one counter that is deliberately
 * NOT rolled up.
 */

import { describe, expect, it, vi } from 'vitest';
import {
  concatenateMeshData,
  MeshProgressiveLoader,
} from '../../../../data/mesh/mesh-progressive-loader';
import {
  meshPayloadBytes,
  meshProjectionBytes,
  type MeshWholeNodeLoader,
} from '../../../../data/mesh/mesh-whole-node-loader';
import type { LoaderMetrics } from '../../../../types/data-monitor-types';
import type { LoadedMeshData } from '../../../../types/mesh';

/** A level loader that reports the given telemetry and nothing else. */
function level(path: string, over: Partial<LoaderMetrics> = {}) {
  return {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    getActiveQueries: vi.fn(() => []),
    dispose: vi.fn(),
    getMetrics: vi.fn((): LoaderMetrics => ({
      type: 'mesh-whole-node',
      path,
      queries: 0,
      loads: 1,
      errors: 0,
      elementsLoaded: 100,
      bytesLoaded: 1000,
      visibleElements: 0,
      avgQueryTime: 0,
      avgLoadTime: 10,
      memoryUsed: 2000,
      ...over,
    })),
  } as unknown as MeshWholeNodeLoader;
}

function ladder(levels: MeshWholeNodeLoader[]): MeshProgressiveLoader {
  return new MeshProgressiveLoader(levels, levels.length, '/surface');
}

function meshData(vertexOffset = 0): LoadedMeshData {
  return {
    vertices: new Float32Array([vertexOffset, 0, 0, vertexOffset + 1, 0, 0, vertexOffset, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
    normals: null,
    colors: null,
    vertexCount: 3,
    faceCount: 1,
    ndim: 3,
    projection: {
      position: new Float32Array(9),
      displayDimsKey: null,
      mask: new Uint8Array(3),
      faceScratch: new Uint32Array(3),
      fastPathBounds: null,
      fastPathBoundsKey: null,
    },
  };
}

describe('meshPayloadBytes — texture arms', () => {
  // The encoded arms must charge the same decoded surface `preflight.ts`
  // budgets them at, or the panel and the admission gate disagree about what a
  // texture expands to — a three-fold gap between the bitmap and KTX2 arms.
  const W = 256;
  const H = 128;

  /** Bytes the payload attributes to the texture, isolated from the geometry. */
  function textureCharge(texture: LoadedMeshData['texture']): number {
    const bare = meshData();
    return meshPayloadBytes({ ...bare, texture }) - meshPayloadBytes(bare);
  }

  it('charges a raw texture its materialized surface', () => {
    const pixels = new Uint8Array(W * H * 3);
    expect(textureCharge({ kind: 'raw', pixels, width: W, height: H, channels: 3 })).toBe(
      pixels.byteLength
    );
  });

  it('charges a bitmap 4 bytes per pixel regardless of source channels', () => {
    // An ImageBitmap is always 4-channel 8-bit once decoded, so a 3-channel
    // WebP still costs the full RGBA surface.
    expect(
      textureCharge({
        kind: 'bitmap',
        bitmap: {} as ImageBitmap,
        width: W,
        height: H,
        channels: 3,
      })
    ).toBe(W * H * 4);
  });

  it('charges a compressed texture ~1 byte per texel plus its mip chain', () => {
    const charge = textureCharge({
      kind: 'compressed',
      texture: {} as never,
      width: W,
      height: H,
      channels: 4,
    });
    // MIRROR: the `decode === 'ktx2'` term in `preflight.ts`.
    expect(charge).toBe(Math.ceil((W * H * 4) / 3));
    // The point of KTX2: it must NOT be charged as if it expanded to RGBA8.
    expect(charge).toBeLessThan(W * H * 4);
  });
});

describe('MeshProgressiveLoader — monitor telemetry', () => {
  it('reports one aggregate keyed by the NODE path, not per level', () => {
    const loader = ladder([level('/surface/additive_0'), level('/surface/additive_1')]);
    const m = loader.getMetrics();

    expect(m.path).toBe('/surface');
    expect(m.type).toBe('mesh-whole-node');
    // Summed across the revealed levels: each level is its own whole-node fetch.
    expect(m.loads).toBe(2);
    expect(m.elementsLoaded).toBe(200);
    expect(m.bytesLoaded).toBe(2000);
    expect(m.memoryUsed).toBe(4000);
  });

  it('reports the retained cumulative payload after child payloads are released', () => {
    const first = meshData();
    const second = meshData(2);
    const firstBytes = meshPayloadBytes(first) + meshProjectionBytes(first);
    const single = ladder([level('/surface/additive_0', { memoryUsed: 0 })]);
    const singleInternals = single as unknown as {
      loadedLODs: LoadedMeshData[];
      _concatCache: { lodCount: number; result: LoadedMeshData } | null;
    };

    singleInternals.loadedLODs = [first];
    singleInternals._concatCache = { lodCount: 1, result: first };
    expect(single.getMetrics().memoryUsed).toBe(firstBytes);

    const concatenated = concatenateMeshData([first, second]);
    const folded = ladder([
      level('/surface/additive_0', { memoryUsed: 0 }),
      level('/surface/additive_1', { memoryUsed: 0 }),
    ]);
    const foldedInternals = folded as unknown as {
      loadedLODs: LoadedMeshData[];
      _concatCache: { lodCount: number; result: LoadedMeshData } | null;
    };
    foldedInternals.loadedLODs = [concatenated];
    foldedInternals._concatCache = { lodCount: 2, result: concatenated };
    expect(folded.getMetrics().memoryUsed).toBe(
      meshPayloadBytes(concatenated) + meshProjectionBytes(concatenated)
    );
  });

  it('re-paths a level event to the node path', () => {
    // Otherwise the monitor records one metrics entry per `additive_<i>`
    // sub-path and double-counts the node's throughput.
    const a = level('/surface/additive_0');
    const loader = ladder([a]);
    const seen: { data: { path?: string } }[] = [];
    loader.addEventListener((e) => seen.push(e));

    const registered = vi.mocked(a.addEventListener!).mock.calls[0][0];
    registered({
      type: 'load',
      loader: 'mesh-whole-node',
      timestamp: 0,
      data: { path: '/surface/additive_0', elements: 100 },
    });

    expect(seen).toHaveLength(1);
    expect(seen[0].data.path).toBe('/surface');
  });

  it('takes visibleElements from the commit, NOT from the level sum', () => {
    // The committed surface is the revealed prefix's concatenation — a
    // node-level fact. Summing the levels would report 0 (no level is told)
    // or, if levels ever were told, double-count the shared prefix.
    const loader = ladder([
      level('/surface/additive_0', { visibleElements: 7 }),
      level('/surface/additive_1', { visibleElements: 7 }),
    ]);
    expect(loader.getMetrics().visibleElements).toBe(0);

    loader.recordVisibleElements(150);
    expect(loader.getMetrics().visibleElements).toBe(150);
    // The roll-up of everything else is untouched by the override.
    expect(loader.getMetrics().loads).toBe(2);
  });

  it('still names itself mesh after dispose clears its levels', () => {
    // With no levels left there is nothing to read the type from; the adapter
    // used to hard-code a points fallback, which showed a disposed mesh node as
    // a points loader.
    const loader = ladder([level('/surface/additive_0')]);
    loader.recordVisibleElements(150);
    loader.dispose();

    const m = loader.getMetrics();
    expect(m.type).toBe('mesh-whole-node');
    expect(m.path).toBe('/surface');
    expect(m.visibleElements).toBe(0);
  });
});
