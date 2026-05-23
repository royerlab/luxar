/**
 * Unit tests for core/app/overlays/init-overlays.ts.
 *
 * The helper always constructs a fresh OverlayManager (so the
 * downstream input-handler + recording-panel + __luxarDebug probe
 * see a stable, non-null reference) and only hydrates it from zarr
 * metadata when both `overlayConfigs` and `zarrBaseUrl` are present
 * on the LuxarScene root's userData.
 */

import * as THREE from 'three';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  initOverlays,
  type InitOverlaysPorts,
} from '../../../../../core/app/overlays/init-overlays';
import { OverlayManager } from '../../../../../ui/overlay-manager';
import type { SceneManager } from '../../../../../scene/scene-manager';

// AUDIT NOTE (core.md W9): this file vi.mocks the entire OverlayManager
// — the principal collaborator. The "construction" test on its own
// asserts only that the constructor was called, which exercises the
// test's own mock harness more than initOverlays. The other tests in
// this file DO exercise meaningful contracts (loadOverlays argument
// shapes, wire-up order between disposePrevious and construct, port
// fan-out into inputHandler.setOverlayManager / recordingPanel.set*,
// happy-path / null-root branches). The proper strengthening for the
// pure-construction test is to drop the mock entirely and assert that
// initOverlays returns an OverlayManager instance that satisfies the
// downstream contract (e.g. `manager.dispose()` is callable, the
// __luxarDebug probe sees a real reference). That's a multi-hour
// refactor because OverlayManager pulls in zarr loaders + UI DOM
// helpers; deferred behind a comment so a follow-up PR can address
// it once OverlayManager has a simpler constructor seam.

// Mock OverlayManager so we can intercept construction + loadOverlays.
vi.mock('../../../../../ui/overlay-manager', () => {
  const loadOverlays = vi.fn().mockResolvedValue(undefined);
  return {
    OverlayManager: vi.fn().mockImplementation(() => ({
      loadOverlays,
      dispose: vi.fn(),
    })),
  };
});

interface InputStub {
  setOverlayManager: ReturnType<typeof vi.fn>;
}
interface RecordingStub {
  setOverlayManager: ReturnType<typeof vi.fn>;
}

function makeInput(): InputStub {
  return { setOverlayManager: vi.fn() };
}
function makeRecording(): RecordingStub {
  return { setOverlayManager: vi.fn() };
}

function makeSceneManager(root: THREE.Object3D | null): SceneManager {
  const scene = new THREE.Scene();
  if (root) scene.add(root);
  return { scene } as unknown as SceneManager;
}

function makeLuxarRoot(userData?: Record<string, unknown>): THREE.Group {
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  if (userData) root.userData = userData;
  return root;
}

describe('initOverlays', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('calls disposePrevious BEFORE constructing a new OverlayManager', async () => {
    const order: string[] = [];
    const disposePrevious = vi.fn(() => order.push('disposePrevious'));
    (OverlayManager as unknown as ReturnType<typeof vi.fn>).mockImplementationOnce(() => {
      order.push('construct');
      return { loadOverlays: vi.fn(), dispose: vi.fn() };
    });

    const ports: InitOverlaysPorts = {
      disposePrevious,
      sceneManager: makeSceneManager(makeLuxarRoot()),
      inputHandler: makeInput() as never,
      recordingPanel: makeRecording() as never,
    };

    await initOverlays(ports);

    expect(order).toEqual(['disposePrevious', 'construct']);
  });

  it('constructs an OverlayManager even when LuxarScene root is missing', async () => {
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(null),
      inputHandler: makeInput() as never,
      recordingPanel: makeRecording() as never,
    };

    const manager = await initOverlays(ports);

    // Helper still returns a manager — the downstream probe
    // (`__luxarDebug.getOverlayManager()`) needs a stable reference.
    expect(manager).toBeDefined();
    expect(OverlayManager).toHaveBeenCalledOnce();
  });

  it('skips loadOverlays when LuxarScene root has no overlayConfigs', async () => {
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(makeLuxarRoot()),
      inputHandler: makeInput() as never,
      recordingPanel: makeRecording() as never,
    };

    const manager = await initOverlays(ports);

    // Manager constructed but not hydrated.
    expect(manager.loadOverlays).not.toHaveBeenCalled();
  });

  it('skips loadOverlays when overlayConfigs is empty', async () => {
    const root = makeLuxarRoot({
      overlayConfigs: [],
      zarrBaseUrl: 'http://example.com/dataset.zarr',
    });
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(root),
      inputHandler: makeInput() as never,
      recordingPanel: makeRecording() as never,
    };

    const manager = await initOverlays(ports);

    expect(manager.loadOverlays).not.toHaveBeenCalled();
  });

  it('skips loadOverlays when zarrBaseUrl is missing', async () => {
    const root = makeLuxarRoot({
      overlayConfigs: [{ name: 'foo' }],
      // zarrBaseUrl omitted
    });
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(root),
      inputHandler: makeInput() as never,
      recordingPanel: makeRecording() as never,
    };

    const manager = await initOverlays(ports);

    expect(manager.loadOverlays).not.toHaveBeenCalled();
  });

  it('calls loadOverlays(configs, baseUrl) when both are present and non-empty', async () => {
    const configs = [{ name: 'foo' }, { name: 'bar' }];
    const baseUrl = 'http://example.com/dataset.zarr';
    const root = makeLuxarRoot({ overlayConfigs: configs, zarrBaseUrl: baseUrl });
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(root),
      inputHandler: makeInput() as never,
      recordingPanel: makeRecording() as never,
    };

    const manager = await initOverlays(ports);

    expect(manager.loadOverlays).toHaveBeenCalledExactlyOnceWith(configs, baseUrl);
  });

  it('wires the new manager into inputHandler.setOverlayManager', async () => {
    const input = makeInput();
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(makeLuxarRoot()),
      inputHandler: input as never,
      recordingPanel: makeRecording() as never,
    };

    const manager = await initOverlays(ports);

    expect(input.setOverlayManager).toHaveBeenCalledExactlyOnceWith(manager);
  });

  it('wires the new manager into recordingPanel.setOverlayManager when present', async () => {
    const recording = makeRecording();
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(makeLuxarRoot()),
      inputHandler: makeInput() as never,
      recordingPanel: recording as never,
    };

    const manager = await initOverlays(ports);

    expect(recording.setOverlayManager).toHaveBeenCalledExactlyOnceWith(manager);
  });

  it('skips recordingPanel wiring when recordingPanel is undefined (no throw)', async () => {
    const ports: InitOverlaysPorts = {
      disposePrevious: vi.fn(),
      sceneManager: makeSceneManager(makeLuxarRoot()),
      inputHandler: makeInput() as never,
      recordingPanel: undefined,
    };

    await expect(initOverlays(ports)).resolves.toBeDefined();
  });
});
