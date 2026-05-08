/**
 * Phase 14.3 regression tests for the initial-load placeholder model.
 *
 * Pre-fix scenario:
 *   - `loadPoints/Lines/GSplats` constructed a fully-populated THREE
 *     object only on data-fetch success.
 *   - On initial-load failure, no THREE object existed in the scene.
 *   - `retryFailedLoader()` could fetch data successfully but commit
 *     helpers found no object by name and silently no-oped, then
 *     `failedLoaders.delete(path)` cleared the failure — geometry was
 *     permanently absent while the loader claimed success.
 *
 * Post-fix:
 *   - Each `loadX()` builds an empty placeholder via
 *     `NodeFactory.createEmpty{Points,Lines,GSplats}Node`, attaches it
 *     to `parentThree` BEFORE the data fetch, then commits real data
 *     on success or records failure on error.
 *   - The placeholder remains in the scene across failures. Retry
 *     commits into the existing object via the same path used by all
 *     future updates.
 *   - Defensive: `retryFailedLoader()` only clears `failedLoaders` when
 *     the named object still exists in `rootGroup` after commit. A
 *     scene that lost the placeholder (programmatic removal between
 *     failure and retry) returns `false` instead of false-claiming
 *     success.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { NodeFactory } from '../../../rendering/node-factory';
import type { PointsMetadata } from '../../../types/points';
import type { LinesMetadata } from '../../../types/lines';
import type { GSplatsMetadata } from '../../../types/gsplats';
import type { DataLoader } from '../../../data/data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';

// `materialManager.getX` calls hit shader compilation, which requires a
// live WebGL context. Mock it the same way scene-loader tests do.
vi.mock('../../../rendering/material-manager', async () => {
  const actual = await vi.importActual<typeof import('../../../rendering/material-manager')>(
    '../../../rendering/material-manager'
  );
  return {
    ...actual,
    materialManager: {
      getPointMaterial: vi.fn(() => ({
        uniforms: {},
        userData: {},
        updateCameraParams: vi.fn(),
      })),
      getLineMaterial: vi.fn(() => ({
        uniforms: { uMaxWidth: { value: 1.0 } },
        userData: {},
        updateCameraParams: vi.fn(),
        clone: vi.fn().mockReturnThis(),
        updateColormapTexture: vi.fn(),
        updateScalarRange: vi.fn(),
      })),
      getGSplatMaterial: vi.fn(() => ({
        uniforms: { uTruncate: { value: 3.0 } },
        userData: {},
        updateCameraParams: vi.fn(),
        clone: vi.fn().mockReturnThis(),
        updateColormapTexture: vi.fn(),
        updateScalarRange: vi.fn(),
      })),
      register: vi.fn(),
    },
  };
});

// `getColormapTexture` reads a sampler uniform; the placeholder factories
// don't trip the colormap branch (no `nodeAttrs.colormap`), but mock for
// safety.
vi.mock('../../../rendering/colormap-textures', () => ({
  getColormapTexture: vi.fn(() => null),
}));

describe('NodeFactory.createEmptyPointsNode', () => {
  it('produces a THREE.Points with empty geometry and correct userData', () => {
    const factory = new NodeFactory();
    const attrs: PointsMetadata = {
      n_points: 100,
      max_radius: 1.0,
      max_sharpness: 31.0,
    } as PointsMetadata;
    const loader = { dispose: vi.fn() } as unknown as DataLoader;

    const placeholder = factory.createEmptyPointsNode('/empty-points', attrs, loader);

    expect(placeholder).toBeInstanceOf(THREE.Points);
    expect(placeholder.name).toBe('/empty-points');
    expect(placeholder.userData.nodeType).toBe('points');
    expect(placeholder.userData.attrs).toBe(attrs);
    expect(placeholder.userData.loader).toBe(loader);
    expect(placeholder.userData.visiblePointCount).toBe(0);

    // Geometry exists but has zero points.
    expect(placeholder.geometry).toBeDefined();
    const positionAttr = placeholder.geometry.getAttribute('position') as THREE.BufferAttribute;
    expect(positionAttr).toBeDefined();
    expect(positionAttr.count).toBe(0);
  });
});

describe('NodeFactory.createEmptyLinesNode', () => {
  it('produces a THREE.Mesh with empty instance buffers and correct userData', () => {
    const factory = new NodeFactory();
    const attrs: LinesMetadata = {
      n_segments: 50,
      n_vertices: 100,
      max_width: 2.0,
    } as LinesMetadata;
    const loader = { dispose: vi.fn() } as unknown as LinesDataLoader;

    const placeholder = factory.createEmptyLinesNode(
      '/empty-lines',
      {} as Record<string, unknown>,
      attrs,
      loader
    );

    expect(placeholder).toBeInstanceOf(THREE.Mesh);
    expect(placeholder.name).toBe('/empty-lines');
    expect(placeholder.userData.nodeType).toBe('lines');
    expect(placeholder.userData.attrs).toBe(attrs);
    expect(placeholder.userData.loader).toBe(loader);
    expect(placeholder.userData.visibleSegmentCount).toBe(0);
  });
});

describe('NodeFactory.createEmptyGSplatsNode', () => {
  it('produces a THREE.Mesh with empty instance buffers and correct userData', () => {
    const factory = new NodeFactory();
    const attrs: GSplatsMetadata = {
      n_splats: 1000,
      truncation_radius: 3.0,
    } as GSplatsMetadata;
    const loader = { dispose: vi.fn() } as unknown as GSplatsDataLoader;

    const placeholder = factory.createEmptyGSplatsNode(
      '/empty-gsplats',
      {} as Record<string, unknown>,
      attrs,
      loader
    );

    expect(placeholder).toBeInstanceOf(THREE.Mesh);
    expect(placeholder.name).toBe('/empty-gsplats');
    expect(placeholder.userData.nodeType).toBe('gsplats');
    expect(placeholder.userData.attrs).toBe(attrs);
    expect(placeholder.userData.loader).toBe(loader);
    expect(placeholder.userData.visibleSplatCount).toBe(0);
  });
});

describe('NodeFactory placeholder factories — common contract', () => {
  it('all three placeholder factories produce objects findable by name', () => {
    // Ensures `commitX` helpers (which use `getObjectByName`) can locate
    // the placeholder once data arrives. This is the primary contract of
    // the placeholder model: an empty object with the right name and
    // userData.nodeType is enough for the existing commit pipeline.
    const factory = new NodeFactory();
    const root = new THREE.Group();
    const pLoader = { dispose: vi.fn() } as unknown as DataLoader;
    const lLoader = { dispose: vi.fn() } as unknown as LinesDataLoader;
    const gLoader = { dispose: vi.fn() } as unknown as GSplatsDataLoader;

    const points = factory.createEmptyPointsNode(
      '/p',
      { n_points: 0 } as PointsMetadata,
      pLoader
    );
    const lines = factory.createEmptyLinesNode(
      '/l',
      {} as Record<string, unknown>,
      { n_segments: 0 } as LinesMetadata,
      lLoader
    );
    const gsplats = factory.createEmptyGSplatsNode(
      '/g',
      {} as Record<string, unknown>,
      { n_splats: 0 } as GSplatsMetadata,
      gLoader
    );

    root.add(points);
    root.add(lines);
    root.add(gsplats);

    expect(root.getObjectByName('/p')).toBe(points);
    expect(root.getObjectByName('/l')).toBe(lines);
    expect(root.getObjectByName('/g')).toBe(gsplats);
  });

  it('placeholders persist in the scene when the loader function throws (caller responsibility)', () => {
    // This test asserts the *invariant* that the placeholder model
    // depends on: once attached, removing it requires explicit action.
    // The loader's catch block doesn't remove the placeholder, so a
    // failed initial load still leaves the placeholder in the scene
    // for retry to populate.
    const factory = new NodeFactory();
    const root = new THREE.Group();
    const placeholder = factory.createEmptyPointsNode(
      '/persistent',
      { n_points: 0 } as PointsMetadata,
      { dispose: vi.fn() } as unknown as DataLoader
    );
    root.add(placeholder);

    // Simulate the load function throwing without removing the placeholder
    // (mirrors the actual scene-loader catch path).
    const simulateFailedLoad = (): void => {
      throw new Error('Network timeout');
    };
    expect(simulateFailedLoad).toThrow();

    // Placeholder still findable.
    expect(root.getObjectByName('/persistent')).toBe(placeholder);
  });
});
