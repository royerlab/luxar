import { describe, expect, it, vi } from 'vitest';
import {
  buildBookmarkUrl,
  captureBookmark,
  parseBookmark,
  restoreBookmark,
} from '../../../../core/app/bookmark-state';
import type { LuxarApp } from '../../../../core/app';
import * as THREE from 'three';
import { sceneDimsManager } from '../../../../scene/scene-dims-manager';
import { WaypointDriver, resolveWaypointPose } from '../../../../core/app/camera/waypoint-driver';
import { CameraFlight } from '../../../../core/app/camera/camera-flight';
import {
  captureSnapshot as captureViewerSnapshot,
  restoreCamera,
  restoreSnapshot as restoreViewerSnapshot,
} from '../../../../core/app/snapshot/viewer-snapshot';

const pose = {
  position: [1, 2, 3] as const,
  target: [0, 0, 0] as const,
  up: [0, 1, 0] as const,
  isOrtho: false,
  fov: 45,
  near: 0.1,
  far: 100,
};

function makeApp() {
  const app = {
    getViewerState: vi.fn(() => ({ src: '/sample.zarr', camera: pose })),
    captureSnapshot: vi.fn(() => ({
      version: 1,
      camera: pose,
      dims: { ndim: 4, displayed: [0, 1, 2], currentStep: [0, 0, 0, 7] },
    })),
    getRenderingSettings: vi.fn(() => ({
      exposure: 1.5,
      controlType: 'orbit',
      dynamicClippingEnabled: false,
      near: 0.1,
      far: 100,
    })),
    getLayers: vi.fn(() => [
      {
        path: 'cells',
        name: 'Cells',
        type: 'points',
        visible: false,
        opacity: 0.4,
        gamma: 1.2,
        displayRange: [2, 8],
        dataRange: [0, 10],
        colormap: 'viridis',
        supportsColormap: true,
        blendingMode: 'normal',
        absorption: 0,
        layerOrder: 2,
      },
    ]),
    restoreSnapshot: vi.fn(),
    flyTo: vi.fn().mockResolvedValue({ completed: true }),
    setRenderingSettings: vi.fn(),
    setLayer: vi.fn(),
    switchDataset: vi.fn().mockResolvedValue(undefined),
  };
  return app;
}

describe('view bookmarks', () => {
  it('round-trips the current view through a share URL and restores settings and layers', async () => {
    const app = makeApp();
    const bookmark = captureBookmark(app as unknown as LuxarApp);
    const url = buildBookmarkUrl(
      'https://example.org/viewer?debug&src=old&control=ws%3A%2F%2Fhost&controlToken=secret#section',
      bookmark
    );
    const parsed = parseBookmark(new URLSearchParams(new URL(url).hash.slice(1)).get('view'));
    expect(parsed).toEqual(bookmark);
    expect(new URL(url).searchParams.get('src')).toBe('/sample.zarr');
    expect(new URL(url).searchParams.has('view')).toBe(false);
    expect(new URL(url).searchParams.has('controlToken')).toBe(false);
    expect(new URL(url).searchParams.has('control')).toBe(false);
    await restoreBookmark(app as unknown as LuxarApp, parsed!);
    expect(app.restoreSnapshot).toHaveBeenCalledWith(bookmark.snapshot);
    expect(app.flyTo).toHaveBeenCalledWith(bookmark.snapshot.camera, { durationMs: 0 });
    expect(app.setRenderingSettings).toHaveBeenCalledWith(bookmark.rendering);
    expect(app.setLayer).toHaveBeenCalledWith(
      'cells',
      expect.objectContaining({ visible: false, opacity: 0.4, displayRange: [2, 8] })
    );
    expect(app.switchDataset).not.toHaveBeenCalled();
  });

  it('switches datasets before applying a bookmark', async () => {
    const app = makeApp();
    const bookmark = captureBookmark(app as unknown as LuxarApp);
    app.getViewerState.mockReturnValue({ src: '/other.zarr', camera: pose });
    await restoreBookmark(app as unknown as LuxarApp, bookmark);
    expect(app.switchDataset).toHaveBeenCalledWith('/sample.zarr');
    expect(app.restoreSnapshot).toHaveBeenCalledTimes(1);
  });

  it('leaves dynamic clipping planes to the camera updater', () => {
    const app = makeApp();
    app.getRenderingSettings.mockReturnValue({
      exposure: 1.5,
      controlType: 'orbit',
      dynamicClippingEnabled: true,
      near: 0.1,
      far: 100,
    });
    const bookmark = captureBookmark(app as unknown as LuxarApp);
    expect(bookmark.rendering).not.toHaveProperty('near');
    expect(bookmark.rendering).not.toHaveProperty('far');
    expect(bookmark.rendering).toHaveProperty('exposure', 1.5);
  });

  it('drops device and input settings from captured and hand-edited links', () => {
    const app = makeApp();
    app.getRenderingSettings.mockReturnValue({
      exposure: 1.5,
      controlType: 'orbit',
      dynamicClippingEnabled: false,
      near: 0.1,
      far: 100,
      ssaaEnabled: true,
      allowHighDPR: true,
      orbitZoomSpeed: 4,
    } as never);
    const bookmark = captureBookmark(app as unknown as LuxarApp);
    expect(bookmark.rendering).not.toHaveProperty('ssaaEnabled');
    expect(bookmark.rendering).not.toHaveProperty('allowHighDPR');
    expect(bookmark.rendering).not.toHaveProperty('orbitZoomSpeed');
    const parsed = parseBookmark(
      JSON.stringify({
        ...bookmark,
        rendering: { ...bookmark.rendering, ssaaEnabled: true, allowHighDPR: true, exposure: 500 },
      })
    );
    expect(parsed?.rendering).not.toHaveProperty('ssaaEnabled');
    expect(parsed?.rendering).not.toHaveProperty('allowHighDPR');
    expect(parsed?.rendering).not.toHaveProperty('exposure');
  });

  it('cancels waypoint flights and restores the saved pose after dimension changes', async () => {
    const app = makeApp();
    const scene = new THREE.Scene();
    scene.userData.sceneDimensions = {
      dimensions: ['x', 'y', 'z', 'story'].map((name, i) => ({
        name,
        unit: '',
        range: [0, 10],
        step: 1,
        display: i < 3,
      })),
    };
    sceneDimsManager.reset();
    sceneDimsManager.initFromScene(scene);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 10);
    const target = new THREE.Vector3();
    const sceneManager = {
      camera,
      controls: {
        getFocusTarget: () => target.clone(),
        setTarget: (next: THREE.Vector3) => target.copy(next),
        reinitialize: () => {},
        dispatchEvent: () => {},
      },
      commitCameraChange: (write?: () => void) => {
        write?.();
        camera.updateMatrixWorld();
      },
    } as unknown as ConstructorParameters<typeof CameraFlight>[0]['sceneManager'];
    const flight = new CameraFlight({
      sceneManager,
      animationController: {
        addPerFrameCallback: vi.fn(),
        removePerFrameCallback: vi.fn(),
        startAnimation: vi.fn(),
      },
      inputElement: null,
    });
    const driver = new WaypointDriver(
      [
        { when: { story: 0 }, camera: { position: [0, 0, 10] } },
        { when: { story: 3 }, camera: { position: [50, 0, 0] }, rendering: { exposure: 0.1 } },
      ],
      {
        getDims: () => sceneDimsManager.getDims(),
        getLivePose: () => captureViewerSnapshot(sceneManager).camera,
        resolvePose: (authored, live) =>
          resolveWaypointPose(authored, live, {
            resolveNodeCenter: () => null,
            fovPresets: {},
          }),
        snapTo: (saved) => restoreCamera(sceneManager, saved),
        flyTo: (saved, opts) => flight.flyTo(saved, opts),
        applyRendering: (settings) => app.setRenderingSettings(settings),
        autoRotateActive: () => false,
      }
    );
    const listener = () => {
      driver.evaluate('fly');
    };
    sceneDimsManager.addListener(listener);
    driver.evaluate('snap');
    app.restoreSnapshot.mockImplementation((saved) => restoreViewerSnapshot(sceneManager, saved));
    app.flyTo.mockImplementation((saved, opts) => flight.flyTo(saved, opts));
    app.captureSnapshot.mockReturnValue({
      version: 1,
      camera: pose,
      dims: { ndim: 4, displayed: [0, 1, 2], currentStep: [0, 0, 0, 3] },
    });
    try {
      const bookmark = captureBookmark(app as unknown as LuxarApp);
      await restoreBookmark(app as unknown as LuxarApp, bookmark);
      expect(sceneDimsManager.getDims()?.currentStep[3]).toBe(3);
      expect(driver.currentIndex).toBe(1);
      expect(flight.isActive).toBe(false);
      expect(captureViewerSnapshot(sceneManager).camera.position).toEqual(pose.position);
      expect(app.setRenderingSettings).toHaveBeenLastCalledWith(bookmark.rendering);
    } finally {
      sceneDimsManager.removeListener(listener);
      sceneDimsManager.reset();
      flight.dispose();
    }
  });

  it('accepts an orthographic camera with zoom instead of perspective FOV', () => {
    const app = makeApp();
    const ortho = { ...pose, isOrtho: true, fov: undefined, zoom: 2 };
    app.captureSnapshot.mockReturnValue({
      version: 1,
      camera: ortho,
      dims: { ndim: 4, displayed: [0, 1, 2], currentStep: [0, 0, 0, 7] },
    } as never);
    const bookmark = captureBookmark(app as unknown as LuxarApp);
    expect(parseBookmark(JSON.stringify(bookmark))).toEqual(bookmark);
  });

  it('requires a loaded scene even if a source was configured', () => {
    const app = makeApp();
    app.captureSnapshot.mockReturnValue({ version: 1, camera: pose, dims: undefined } as never);
    expect(() => captureBookmark(app as unknown as LuxarApp)).toThrow('Load a dataset');
  });

  it('rejects malformed, oversized, and non-finite URL state', () => {
    expect(parseBookmark('invalid')).toBeNull();
    expect(parseBookmark('x'.repeat(100_001))).toBeNull();
    const bookmark = captureBookmark(makeApp() as unknown as LuxarApp);
    expect(parseBookmark(JSON.stringify({ ...bookmark, src: 'javascript:alert(1)' }))).toBeNull();
    expect(
      parseBookmark(
        JSON.stringify({
          ...bookmark,
          snapshot: { ...bookmark.snapshot, camera: { ...pose, fov: 'bad' } },
        })
      )
    ).toBeNull();
    expect(
      parseBookmark(
        JSON.stringify({
          ...bookmark,
          snapshot: {
            ...bookmark.snapshot,
            dims: { ndim: 4, displayed: [0, 1, 2], currentStep: [0, 0, 0] },
          },
        })
      )
    ).toBeNull();
  });
});
