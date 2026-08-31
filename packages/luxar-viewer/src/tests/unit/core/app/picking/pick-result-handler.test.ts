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
  return {
    nodeId: 1,
    elementId,
    storageElementId: elementId,
    brightness: 1.0,
    mainNode,
    screenX: 0,
    screenY: 0,
  };
}

describe('buildPickResultHandler', () => {
  it('surfaces the exact gsplat categorical id when no text label is baked', async () => {
    const s = makeStubs();
    s.getLabel.mockResolvedValue(null);
    const result = makeResult('/Cells', 9);
    result.storageElementId = 0;
    result.mainNode.userData = {
      nodeType: 'gsplats',
      labelIndices: new Uint32Array([1]),
      labelVocabulary: [{ id: '9007199254740993', name: 'rare class' }],
    };
    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle(result);

    expect(s.updateHoverContent).toHaveBeenCalledWith({
      label: 'rare class (9007199254740993)',
      key: null,
      imageUrl: null,
      nodeName: '/Cells',
      elementIndex: 9,
    });
  });
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
      key: null,
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
      key: null,
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
      key: null,
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
      key: null,
      imageUrl: 'thumbs/0.png',
      nodeName: '/Cells',
      elementIndex: 0,
    });
  });

  it('shows a key-only hover payload for {hover_key} overlays', async () => {
    const updateHoverContent = vi.fn();
    const getKey = vi.fn().mockResolvedValue('P04637');
    const handle = buildPickResultHandler({
      keyLoader: { getLabel: getKey },
      overlayManager: { updateHoverContent },
    });

    await handle(makeResult('/Proteins', 7));

    expect(getKey).toHaveBeenCalledExactlyOnceWith('/Proteins', 7);
    expect(updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: null,
      key: 'P04637',
      imageUrl: null,
      nodeName: '/Proteins',
      elementIndex: 7,
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

  it('reports the kind=partition wrapper path but QUERIES the leaf', async () => {
    // Partition-aware picking (#1415): a hit on an inner ``part_<i>`` leaf
    // surfaces the wrapper's path as ``nodeName`` (layers-panel
    // outermost-as-layer convention), while the label/image lookup goes to
    // the LEAF — the wrapper is a bare group with no label CSR, and
    // ``elementId`` is part-local.
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

    await handle({
      nodeId: 1,
      elementId: 42,
      storageElementId: 42,
      brightness: 1.0,
      mainNode: part,
      screenX: 0,
      screenY: 0,
    });
    expect(s.getLabel).toHaveBeenCalledWith('/Splat/part_3', 42);
    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: 'Cell 42',
      key: null,
      imageUrl: null,
      nodeName: '/Splat',
      elementIndex: 42,
    });
  });

  it('resolves a tooltip whose label only exists under the leaf path', async () => {
    // Regression pin for #1415. Before the reported/queried split, both
    // loaders were handed the wrapper path. A realistic store answers only
    // for the leaf (that is where the writer put ``label_offsets`` /
    // ``label_bytes``), so the wrapper query returned null and the tooltip
    // was silently empty on EVERY hover of a partitioned layer. This test
    // fails with the pre-fix handler.
    const s = makeStubs();
    s.getLabel.mockImplementation(async (path) => (path === '/Splat/part_3' ? 'Cell 42' : null));
    s.getImageUrl.mockImplementation(async (path) =>
      path === '/Splat/part_3' ? 'thumbs/42.png' : null
    );
    const wrapper = new THREE.Group();
    wrapper.name = '/Splat';
    wrapper.userData.kind = 'partition';
    const part = new THREE.Object3D();
    part.name = '/Splat/part_3';
    wrapper.add(part);

    const handle = buildPickResultHandler({
      labelLoader: { getLabel: s.getLabel },
      imageLabelLoader: { getImageUrl: s.getImageUrl },
      overlayManager: { updateHoverContent: s.updateHoverContent },
    });

    await handle({
      nodeId: 1,
      elementId: 42,
      storageElementId: 42,
      brightness: 1.0,
      mainNode: part,
      screenX: 0,
      screenY: 0,
    });

    expect(s.getImageUrl).toHaveBeenCalledWith('/Splat/part_3', 42);
    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: 'Cell 42',
      key: null,
      imageUrl: 'thumbs/42.png',
      nodeName: '/Splat',
      elementIndex: 42,
    });
  });

  it('picks the OUTERMOST kind=partition to report, still querying the leaf', async () => {
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

    await handle({
      nodeId: 1,
      elementId: 42,
      storageElementId: 42,
      brightness: 1.0,
      mainNode: leaf,
      screenX: 0,
      screenY: 0,
    });
    // Queried on the innermost leaf (the only node that owns a label CSR)...
    expect(s.getLabel).toHaveBeenCalledWith('/Outer/part_1/part_0', 42);
    // ...reported as the outermost wrapper.
    expect(s.updateHoverContent).toHaveBeenCalledExactlyOnceWith({
      label: 'Cell 42',
      key: null,
      imageUrl: null,
      nodeName: '/Outer',
      elementIndex: 42,
    });
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

    await handle({
      nodeId: 1,
      elementId: 7,
      storageElementId: 7,
      brightness: 1.0,
      mainNode: leaf,
      screenX: 0,
      screenY: 0,
    });
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
      key: null,
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

      // ...but selection still reports the picked element. With no partition
      // wrapper the reported and hit nodes coincide.
      expect(onSelection).toHaveBeenCalledExactlyOnceWith({
        nodeName: '/Cells',
        elementIndex: 9,
        hitNodeName: '/Cells',
      });
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

    it('reports the outermost partition wrapper as the selection node, and the hit leaf as hitNodeName', async () => {
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

      await handle({
        nodeId: 1,
        elementId: 42,
        storageElementId: 42,
        brightness: 1.0,
        mainNode: part,
        screenX: 0,
        screenY: 0,
      });

      // The reported/queried split (#1415) is visible in the payload itself:
      // `nodeName` is the user-facing layer, `hitNodeName` is the leaf that
      // `elementIndex` is local to and that an embedder must index against.
      expect(onSelection).toHaveBeenCalledExactlyOnceWith({
        nodeName: '/Splat',
        elementIndex: 42,
        hitNodeName: '/Splat/part_3',
      });
      // The loader is still asked about the leaf.
      expect(s.getLabel).toHaveBeenCalledWith('/Splat/part_3', 42);
    });

    it('carries the innermost leaf as hitNodeName under nested partitions', async () => {
      // `nodeName` climbs to the OUTERMOST kind=partition wrapper, but the
      // element index belongs to the innermost leaf — so a two-level partition
      // is where the two fields are furthest apart, and the only place a
      // `hitNodeName` that merely copied `nodeName` would go unnoticed.
      const s = makeStubs();
      s.getLabel.mockResolvedValue('Cell 42');
      const onSelection = vi.fn();
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
        onSelection,
      });

      await handle({
        nodeId: 1,
        elementId: 42,
        storageElementId: 42,
        brightness: 1.0,
        mainNode: leaf,
        screenX: 0,
        screenY: 0,
      });

      expect(onSelection).toHaveBeenCalledExactlyOnceWith({
        nodeName: '/Outer',
        elementIndex: 42,
        hitNodeName: '/Outer/part_1/part_0',
      });
    });

    it('sets hitNodeName equal to nodeName when no partition wrapper exists', async () => {
      // Plain (non-partition) group ancestors must not split the two paths —
      // the additive field is a no-op for the ordinary case.
      const s = makeStubs();
      s.getLabel.mockResolvedValue('hi');
      const onSelection = vi.fn();
      const plainGroup = new THREE.Group();
      plainGroup.name = '/Plain';
      const leaf = new THREE.Object3D();
      leaf.name = '/Plain/leaf';
      plainGroup.add(leaf);

      const handle = buildPickResultHandler({
        labelLoader: { getLabel: s.getLabel },
        overlayManager: { updateHoverContent: s.updateHoverContent },
        onSelection,
      });

      await handle({
        nodeId: 1,
        elementId: 7,
        storageElementId: 7,
        brightness: 1.0,
        mainNode: leaf,
        screenX: 0,
        screenY: 0,
      });

      const sel = onSelection.mock.calls[0][0] as { nodeName: string; hitNodeName: string };
      expect(sel.hitNodeName).toBe('/Plain/leaf');
      expect(sel.hitNodeName).toBe(sel.nodeName);
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
