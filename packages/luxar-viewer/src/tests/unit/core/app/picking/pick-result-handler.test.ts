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
import { buildPickResultHandler } from '../../../../../core/app/picking/pick-result-handler';
import { log } from '../../../../../utils/log';
import type { PickResult } from '../../../../../rendering/picking/picking-system';

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

/** A promise plus its external `resolve` — lets a test gate when a fetch completes. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
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

  it('clears hover and logs a warning when a loader rejects', async () => {
    // core.md W15 strengthening: previously this test only asserted
    // hover was cleared. The handler's contract explicitly logs a
    // warning so debugging hover failures isn't silent. Pin the
    // warning + the underlying error message so a regression that
    // dropped log.warning (or wrapped the wrong scope) would fail.
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
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

    // Warning was logged with the picking-callback context + the
    // underlying error string.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = String(warnSpy.mock.calls[0][1]);
    expect(msg).toMatch(/Picking callback error/);
    expect(msg).toMatch(/zarr 404/);
    warnSpy.mockRestore();
  });

  it('reports the kind=partition wrapper path when the hit sits under one', async () => {
    // Partition-aware picking: a hit on an inner ``part_<i>`` leaf must
    // surface the wrapper's path as ``nodeName`` (and as the path used
    // to look up labels). Matches the layers-panel's outermost-as-layer
    // convention.
    const s = makeStubs();
    s.getLabel.mockResolvedValue('Cell 42');
    const wrapper = new THREE.Group();
    wrapper.name = '/Splat';
    wrapper.userData.kind = 'partition';
    const part = new THREE.Object3D();
    part.name = '/Splat/part_3';
    wrapper.add(part);

    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle({ nodeId: 1, elementId: 42, brightness: 1.0, mainNode: part });
    expect(s.getLabel).toHaveBeenCalledWith('/Splat', 42);
    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: 'Cell 42',
      imageUrl: null,
      nodeName: '/Splat',
      elementIndex: 42,
    });
  });

  it('picks the OUTERMOST kind=partition when nested', async () => {
    const s = makeStubs();
    s.getLabel.mockResolvedValue('Cell 42');
    const outer = new THREE.Group();
    outer.name = '/Outer';
    outer.userData.kind = 'partition';
    const inner = new THREE.Group();
    inner.name = '/Outer/part_1';
    inner.userData.kind = 'partition';
    outer.add(inner);
    const leaf = new THREE.Object3D();
    leaf.name = '/Outer/part_1/part_0';
    inner.add(leaf);

    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle({ nodeId: 1, elementId: 42, brightness: 1.0, mainNode: leaf });
    expect(s.getLabel).toHaveBeenCalledWith('/Outer', 42);
  });

  it('falls back to the leaf name when no kind=partition ancestor exists', async () => {
    // Plain group ancestors (kind=undefined) must not affect picking.
    const s = makeStubs();
    s.getLabel.mockResolvedValue('hi');
    const plainGroup = new THREE.Group();
    plainGroup.name = '/Plain';
    const leaf = new THREE.Object3D();
    leaf.name = '/Plain/leaf';
    plainGroup.add(leaf);

    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle({ nodeId: 1, elementId: 7, brightness: 1.0, mainNode: leaf });
    expect(s.getLabel).toHaveBeenCalledWith('/Plain/leaf', 7);
  });

  it('drops a superseded content result when a newer null arrives mid-fetch', async () => {
    // Ordering guard: a slow label/image fetch from an older hover must
    // not re-show a tooltip after a newer mousemove already faded it.
    const s = makeStubs();
    const labelGate = deferred<string | null>();
    s.getLabel.mockReturnValue(labelGate.promise);
    s.getImageUrl.mockResolvedValue(null);
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    const stale = handle(makeResult('/Cells', 1)); // parks on the label fetch
    await handle(null); // newer fade — emits null, bumps the token
    labelGate.resolve('Cell 1'); // older fetch finally resolves
    await stale; // …and must be dropped, not emitted

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith(null);
  });

  it('emits only the latest result when two fetches resolve out of order', async () => {
    // Even if the OLDER fetch resolves first, only the newest invocation's
    // result reaches the overlay.
    const s = makeStubs();
    const gateA = deferred<string | null>();
    const gateB = deferred<string | null>();
    s.getLabel.mockReturnValueOnce(gateA.promise).mockReturnValueOnce(gateB.promise);
    s.getImageUrl.mockResolvedValue(null);
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    const first = handle(makeResult('/Cells', 1));
    const second = handle(makeResult('/Cells', 2));
    gateA.resolve('Cell 1'); // older resolves first → must be dropped
    gateB.resolve('Cell 2'); // newer wins
    await Promise.all([first, second]);

    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: 'Cell 2',
      imageUrl: null,
      nodeName: '/Cells',
      elementIndex: 2,
    });
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

  describe('onSelection (public selection event sink)', () => {
    it('emits the picked element regardless of tooltip content', async () => {
      const s = makeStubs();
      s.getLabel.mockResolvedValue(null);
      s.getImageUrl.mockResolvedValue(null); // no tooltip content...
      const onSelection = vi.fn();
      const handle = buildPickResultHandler({
        labelLoader: { getLabel: s.getLabel },
        imageLabelLoader: { getImageUrl: s.getImageUrl },
        overlayManager: { updateHoverContent: s.updateHoverContent },
        onSelection,
      });

      await handle(makeResult('/Cells', 9));

      // ...but selection still reports the picked element.
      expect(onSelection).toHaveBeenCalledExactlyOnceWith({ nodeName: '/Cells', elementIndex: 9 });
      expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith(null);
    });

    it('emits null when the hover clears', async () => {
      const onSelection = vi.fn();
      const handle = buildPickResultHandler({
        overlayManager: { updateHoverContent: vi.fn() },
        onSelection,
      });

      await handle(null);

      expect(onSelection).toHaveBeenCalledExactlyOnceWith(null);
    });

    it('reports the outermost partition wrapper as the selection node', async () => {
      const s = makeStubs();
      s.getLabel.mockResolvedValue('Cell 42');
      const onSelection = vi.fn();
      const wrapper = new THREE.Group();
      wrapper.name = '/Splat';
      wrapper.userData.kind = 'partition';
      const part = new THREE.Object3D();
      part.name = '/Splat/part_3';
      wrapper.add(part);

      const handle = buildPickResultHandler({
        labelLoader: { getLabel: s.getLabel },
        overlayManager: { updateHoverContent: s.updateHoverContent },
        onSelection,
      });

      await handle({ nodeId: 1, elementId: 42, brightness: 1.0, mainNode: part });

      expect(onSelection).toHaveBeenCalledExactlyOnceWith({ nodeName: '/Splat', elementIndex: 42 });
    });

    it('does not emit a superseded selection', async () => {
      const s = makeStubs();
      const labelGate = deferred<string | null>();
      s.getLabel.mockReturnValue(labelGate.promise);
      s.getImageUrl.mockResolvedValue(null);
      const onSelection = vi.fn();
      const handle = buildPickResultHandler({
        labelLoader: { getLabel: s.getLabel },
        imageLabelLoader: { getImageUrl: s.getImageUrl },
        overlayManager: { updateHoverContent: s.updateHoverContent },
        onSelection,
      });

      const stale = handle(makeResult('/Cells', 1)); // parks on label fetch
      await handle(null); // newer fade → onSelection(null), bumps token
      labelGate.resolve('Cell 1'); // stale resolves...
      await stale;

      // Only the null fade emitted; the superseded content pick was dropped.
      expect(onSelection).toHaveBeenCalledExactlyOnceWith(null);
    });
  });
});
