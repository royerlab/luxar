// @vitest-environment jsdom
/**
 * `LoaderConfig.onSceneMetadata`: the scene loader hands the root group to the
 * hook once its root metadata is attached and before the nodes load, so the
 * scene manager can frame the opening camera first. Without it, every
 * load-time view decision (B4 ranks a partition's parts against the camera and
 * the displayed dims) ran against the reset pose and no dimensions, and so
 * gated nothing: an authored close-up of `sp64` still initialised all 64 parts.
 *
 * Harness: the mocked-zarrita `SceneLoader` scaffolding of
 * `scene-graph-lookups.test.ts`, with root attrs carrying a `viewer_config`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type * as THREE from 'three';
import * as zarr from 'zarrita';
import { SceneLoader } from '../../../../data';

vi.mock('zarrita', () => ({
  FetchStore: vi.fn(),
  withMaybeConsolidatedMetadata: vi.fn(),
  registry: {},
  root: vi.fn(),
  open: (() => {
    const openMock = vi.fn();
    return Object.assign(openMock, { v2: openMock, v3: openMock });
  })(),
  NotFoundError: class NotFoundError extends Error {},
  InvalidMetadataError: class InvalidMetadataError extends Error {},
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

vi.mock('../../../../utils/cross-layer/notifier', () => ({
  notifier: {
    toast: vi.fn(),
    error: vi.fn(),
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    clearError: vi.fn(),
    showSceneIdentityBanner: vi.fn(),
    hideSceneIdentityBanner: vi.fn(),
  },
}));

const CAMERA = { position: [3, 3, 8], target: [3, 3, 0], up: [0, 1, 0], fov: 40 };
const BOUNDS = { min: [0, 0, 0], max: [40, 40, 40] };

describe('SceneLoader — onSceneMetadata hook', () => {
  let sceneLoader: SceneLoader | null = null;

  beforeEach(() => {
    vi.clearAllMocks();
    const mockStore = { contents: vi.fn().mockResolvedValue([{ path: '/', kind: 'group' }]) };
    const mockRootLoc = {
      resolve: vi.fn().mockImplementation(() => ({
        resolve: vi.fn().mockImplementation(() => ({ resolve: vi.fn() })),
      })),
    };
    const mockZarrGroup = {
      attrs: {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 40], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 40], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 40], display: true, step: 1 },
          ],
        },
        viewer_config: { camera: CAMERA },
        position_bounds: BOUNDS,
      },
    };
    vi.mocked(zarr.FetchStore).mockImplementation(() => mockStore as unknown as zarr.FetchStore);
    vi.mocked(zarr.withMaybeConsolidatedMetadata).mockResolvedValue(mockStore as never);
    vi.mocked(zarr.root).mockReturnValue(mockRootLoc as never);
    vi.mocked(zarr.open).mockResolvedValue(mockZarrGroup as never);
  });

  afterEach(async () => {
    await sceneLoader?.dispose();
    sceneLoader = null;
    vi.restoreAllMocks();
  });

  it('hands the root, with its viewer_config and bounds, to the hook before the load resolves', async () => {
    const seen: { viewerConfig: unknown; positionBounds: unknown; sceneDimensions: unknown }[] = [];
    const onSceneMetadata = vi.fn((root: THREE.Group) => {
      seen.push({
        viewerConfig: root.userData.viewerConfig,
        positionBounds: root.userData.positionBounds,
        sceneDimensions: root.userData.sceneDimensions,
      });
    });
    // `onSceneMetadata` is read structurally so this test compiles before the hook exists.
    sceneLoader = new SceneLoader({ onSceneMetadata } as ConstructorParameters<
      typeof SceneLoader
    >[0]);

    const root = await sceneLoader.loadScene('http://localhost:8000/test.zarr');

    expect(onSceneMetadata).toHaveBeenCalledTimes(1);
    expect(onSceneMetadata.mock.calls[0][0]).toBe(root);
    expect(seen[0].viewerConfig).toEqual({ camera: CAMERA });
    expect(seen[0].positionBounds).toEqual(BOUNDS);
    expect(seen[0].sceneDimensions).toBeDefined();
  });
});
