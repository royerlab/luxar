/**
 * Unit tests for core/pick-result-handler.ts.
 *
 * Pure factory. Tests pass stub ports so each branch in the async
 * handler (null result, label-only, image-only, both, neither,
 * fetch reject, missing loaders) is observable without a real
 * `PickingSystem`, zarr-backed label loaders, or OverlayManager
 * instance.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { buildPickResultHandler } from '../../../core/app/picking/pick-result-handler';
import type { PickResult } from '../../../rendering/picking/picking-system';

type GetLabelFn = (path: string, idx: number) => Promise<string | null>;
type GetImageUrlFn = (path: string, idx: number) => Promise<string | null>;
type UpdateHoverFn = (r: unknown) => void;

interface Stubs {
  getLabel: ReturnType<typeof vi.fn<GetLabelFn>>;
  getImageUrl: ReturnType<typeof vi.fn<GetImageUrlFn>>;
  updateHoverContent: ReturnType<typeof vi.fn<UpdateHoverFn>>;
}

function makeStubs(): Stubs {
  return {
    getLabel: vi.fn<GetLabelFn>(),
    getImageUrl: vi.fn<GetImageUrlFn>(),
    updateHoverContent: vi.fn<UpdateHoverFn>(),
  };
}

/**
 * Build a fake PickResult. The handler reads only `mainNode.name` and
 * `elementId`, so a bare Object3D suffices. The other fields are
 * required by the type but go unread.
 */
function makeResult(nodeName: string, elementId: number): PickResult {
  const mainNode = new THREE.Object3D();
  mainNode.name = nodeName;
  return { nodeId: 1, elementId, brightness: 1.0, mainNode };
}

describe('buildPickResultHandler', () => {
  it('clears hover when result is null and does not call loaders', async () => {
    const s = makeStubs();
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(null);

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith(null);
    expect(s.getLabel).not.toHaveBeenCalled();
    expect(s.getImageUrl).not.toHaveBeenCalled();
  });

  it('forwards a label-only payload', async () => {
    const s = makeStubs();
    s.getLabel.mockResolvedValue('Cell 42');
    s.getImageUrl.mockResolvedValue(null);
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(makeResult('/Cells', 42));

    expect(s.getLabel).toHaveBeenCalledWith('/Cells', 42);
    expect(s.getImageUrl).toHaveBeenCalledWith('/Cells', 42);
    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: 'Cell 42',
      imageUrl: null,
      nodeName: '/Cells',
      elementIndex: 42,
    });
  });

  it('forwards an image-only payload', async () => {
    const s = makeStubs();
    s.getLabel.mockResolvedValue(null);
    s.getImageUrl.mockResolvedValue('thumbs/42.png');
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(makeResult('/Cells', 42));

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: null,
      imageUrl: 'thumbs/42.png',
      nodeName: '/Cells',
      elementIndex: 42,
    });
  });

  it('forwards both label and image when both are present', async () => {
    const s = makeStubs();
    s.getLabel.mockResolvedValue('Cell 42');
    s.getImageUrl.mockResolvedValue('thumbs/42.png');
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(makeResult('/Cells', 42));

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: 'Cell 42',
      imageUrl: 'thumbs/42.png',
      nodeName: '/Cells',
      elementIndex: 42,
    });
  });

  it('clears hover when neither label nor image is available', async () => {
    const s = makeStubs();
    s.getLabel.mockResolvedValue(null);
    s.getImageUrl.mockResolvedValue(null);
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(makeResult('/Cells', 42));

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith(null);
  });

  it('treats an empty-string label as no content', async () => {
    // Defense-in-depth: LabelLoader already normalizes empty labels to
    // null, but the handler's `label || imageUrl` gate must also
    // exclude '' in case a different loader implementation returns it.
    const s = makeStubs();
    s.getLabel.mockResolvedValue('');
    s.getImageUrl.mockResolvedValue(null);
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(makeResult('/Cells', 7));

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith(null);
  });

  it('works when labelLoader is undefined and image loader returns a URL', async () => {
    const s = makeStubs();
    s.getImageUrl.mockResolvedValue('thumbs/0.png');
    const handle = buildPickResultHandler({
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(makeResult('/Cells', 0));

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: null,
      imageUrl: 'thumbs/0.png',
      nodeName: '/Cells',
      elementIndex: 0,
    });
  });

  it('clears hover when both loaders are undefined', async () => {
    const s = makeStubs();
    const handle = buildPickResultHandler({
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(makeResult('/Cells', 0));

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith(null);
  });

  it('clears hover and does not throw when a loader rejects', async () => {
    const s = makeStubs();
    s.getLabel.mockRejectedValue(new Error('zarr 404'));
    s.getImageUrl.mockResolvedValue(null);
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await expect(handle(makeResult('/Cells', 0))).resolves.toBeUndefined();
    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith(null);
  });

  it('is a no-op when overlayManager port is undefined', async () => {
    const s = makeStubs();
    s.getLabel.mockResolvedValue('hello');
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
    });

    await expect(handle(makeResult('/Cells', 0))).resolves.toBeUndefined();
    expect(s.updateHoverContent).not.toHaveBeenCalled();
  });
});
