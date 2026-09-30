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
    getRenderingSettings: vi.fn(() => ({ exposure: 1.5, controlType: 'orbit' })),
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
    const url = buildBookmarkUrl('https://example.org/viewer?debug&src=old#section', bookmark);
    const parsed = parseBookmark(new URL(url).searchParams.get('view'));
    expect(parsed).toEqual(bookmark);
    expect(new URL(url).searchParams.get('src')).toBe('/sample.zarr');
    expect(new URL(url).hash).toBe('#section');
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

  it('rejects malformed, oversized, and non-finite URL state', () => {
    expect(parseBookmark('invalid')).toBeNull();
    expect(parseBookmark('x'.repeat(100_001))).toBeNull();
    const bookmark = captureBookmark(makeApp() as unknown as LuxarApp);
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
