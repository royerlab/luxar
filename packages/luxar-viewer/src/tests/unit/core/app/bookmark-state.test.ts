import { describe, expect, it, vi } from 'vitest';
import {
  buildBookmarkUrl,
  captureBookmark,
  parseBookmark,
  restoreBookmark,
} from '../../../../core/app/bookmark-state';
import type { LuxarApp } from '../../../../core/app';

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
    const parsed = parseBookmark(new URL(url).searchParams.get('view'));
    expect(parsed).toEqual(bookmark);
    expect(new URL(url).searchParams.get('src')).toBe('/sample.zarr');
    expect(new URL(url).hash).toBe('#section');
    expect(new URL(url).searchParams.has('controlToken')).toBe(false);
    expect(new URL(url).searchParams.has('control')).toBe(false);
    await restoreBookmark(app as unknown as LuxarApp, parsed!);
    expect(app.restoreSnapshot).toHaveBeenCalledWith(bookmark.snapshot);
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
