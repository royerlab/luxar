/**
 * Unit tests for core/app/picking/init-picking.ts.
 *
 * Five early-return branches (no root / no labels / no scene loader /
 * no zarr store / success) plus listener wiring through the supplied
 * EventGroup.
 */

import * as THREE from 'three';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { initPicking, type InitPickingResult } from '../../../../../core/app/picking/init-picking';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';

// All heavy collaborators are module-mocked so the test never touches
// WebGL / zarr stores / GPU picking pipeline.
vi.mock('../../../../../rendering/picking/picking-system', () => ({
  PickingSystem: vi.fn().mockImplementation(() => ({
    dispose: vi.fn(),
    onMouseMove: vi.fn(),
    onMouseLeave: vi.fn(),
    markDirty: vi.fn(),
    setShouldPick: vi.fn(),
    setPostProcessing: vi.fn(),
    setCamera: vi.fn(),
    suppress: vi.fn(),
  })),
}));
vi.mock('../../../../../data/loaders/picking/label-loader', () => ({
  LabelLoader: vi.fn().mockImplementation(() => ({ dispose: vi.fn() })),
}));
vi.mock('../../../../../data/loaders/picking/image-label-loader', () => ({
  ImageLabelLoader: vi.fn().mockImplementation(() => ({ dispose: vi.fn() })),
}));
vi.mock('../../../../../data/zarr', () => ({
  root: vi.fn(() => ({ kind: 'zarr-root-loc' })),
}));
vi.mock('../../../../../data/scene-loader-manager', () => ({
  getSceneLoader: vi.fn(),
}));

import { PickingSystem } from '../../../../../rendering/picking/picking-system';
import { LabelLoader, ImageLabelLoader } from '../../../../../data/loaders';
import { getSceneLoader } from '../../../../../data/scene-loader-manager';

interface SceneManagerStub {
  scene: THREE.Scene;
  renderer: { domElement: HTMLCanvasElement };
  capabilities: unknown;
  camera: unknown;
  postProcessing: unknown;
  controls: {
    addEventListener: ReturnType<typeof vi.fn>;
    removeEventListener: ReturnType<typeof vi.fn>;
  };
  addEventListener: ReturnType<typeof vi.fn>;
  removeEventListener: ReturnType<typeof vi.fn>;
}

function makeSceneManager(scene: THREE.Scene): SceneManagerStub {
  return {
    scene,
    renderer: { domElement: document.createElement('canvas') },
    capabilities: { foo: 1 },
    camera: { kind: 'camera' },
    postProcessing: { kind: 'pp' },
    controls: {
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    },
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  };
}

function makeLuxarRoot(opts: { hasLabels?: boolean; hasImageLabels?: boolean } = {}): THREE.Group {
  const root = new THREE.Group();
  root.name = 'LuxarScene';
  const child = new THREE.Group();
  child.userData = {
    attrs: {
      ...(opts.hasLabels ? { has_labels: true } : {}),
      ...(opts.hasImageLabels ? { has_image_labels: true } : {}),
    },
  };
  root.add(child);
  return root;
}

function makeSceneLoader(opts: { hasStore?: boolean } = { hasStore: true }) {
  return {
    zarrStore: opts.hasStore ? { kind: 'store' } : null,
    nodeFactory: {
      setPickingSystem: vi.fn(),
      registerExistingSceneNodes: vi.fn(),
    },
  };
}

function makePreviousEmpty(): InitPickingResult {
  return {
    pickingSystem: undefined,
    labelLoader: undefined,
    imageLabelLoader: undefined,
  };
}

describe('initPicking', () => {
  let pickingEvents: EventGroup;

  beforeEach(() => {
    vi.clearAllMocks();
    pickingEvents = new EventGroup();
  });

  describe('previous-session teardown', () => {
    it('disposes pickingEvents + previous pickingSystem + label loaders before doing anything', async () => {
      const order: string[] = [];
      const prevPicking = { dispose: vi.fn(() => order.push('picking')) };
      const prevLabel = { dispose: vi.fn(() => order.push('label')) };
      const prevImage = { dispose: vi.fn(() => order.push('image')) };
      const events = new EventGroup();
      const eventsSpy = vi.spyOn(events, 'dispose').mockImplementation(() => order.push('events'));

      await initPicking({
        sceneManager: makeSceneManager(new THREE.Scene()) as never,
        pickingEvents: events,
        previous: {
          pickingSystem: prevPicking as never,
          labelLoader: prevLabel as never,
          imageLabelLoader: prevImage as never,
        },
        getOverlayManager: () => undefined,
      });

      expect(order).toEqual(['events', 'picking', 'label', 'image']);
      eventsSpy.mockRestore();
    });

    it('handles all-undefined previous (no throw)', async () => {
      await expect(
        initPicking({
          sceneManager: makeSceneManager(new THREE.Scene()) as never,
          pickingEvents,
          previous: makePreviousEmpty(),
          getOverlayManager: () => undefined,
        })
      ).resolves.toBeDefined();
    });
  });

  describe('early-return branches', () => {
    it('returns all-undefined when LuxarScene root is missing', async () => {
      const result = await initPicking({
        sceneManager: makeSceneManager(new THREE.Scene()) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(result).toEqual({
        pickingSystem: undefined,
        labelLoader: undefined,
        imageLabelLoader: undefined,
      });
      expect(PickingSystem).not.toHaveBeenCalled();
    });

    it('returns all-undefined when no node declares has_labels or has_image_labels', async () => {
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot()); // no labels

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(result.pickingSystem).toBeUndefined();
      expect(PickingSystem).not.toHaveBeenCalled();
      expect(LabelLoader).not.toHaveBeenCalled();
      expect(ImageLabelLoader).not.toHaveBeenCalled();
    });

    it('returns all-undefined when getSceneLoader returns null', async () => {
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true }));
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(null);

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(result.pickingSystem).toBeUndefined();
      expect(PickingSystem).not.toHaveBeenCalled();
    });

    it('returns all-undefined when sceneLoader has no zarr store', async () => {
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true }));
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: false })
      );

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(result.pickingSystem).toBeUndefined();
      expect(PickingSystem).not.toHaveBeenCalled();
    });
  });

  describe('success path — label-only scene', () => {
    let sceneLoader: ReturnType<typeof makeSceneLoader>;
    let scene: THREE.Scene;

    beforeEach(() => {
      sceneLoader = makeSceneLoader({ hasStore: true });
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(sceneLoader);
      scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true }));
    });

    it('constructs LabelLoader but NOT ImageLabelLoader when only has_labels is set', async () => {
      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(LabelLoader).toHaveBeenCalledOnce();
      expect(ImageLabelLoader).not.toHaveBeenCalled();
      expect(result.labelLoader).toBeDefined();
      expect(result.imageLabelLoader).toBeUndefined();
    });

    it('returns a PickingSystem and wires it into nodeFactory', async () => {
      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(result.pickingSystem).toBeDefined();
      expect(sceneLoader.nodeFactory.setPickingSystem).toHaveBeenCalledExactlyOnceWith(
        result.pickingSystem
      );
    });

    it('retroactively registers existing scene nodes', async () => {
      await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(sceneLoader.nodeFactory.registerExistingSceneNodes).toHaveBeenCalledOnce();
    });
  });

  describe('success path — image-label-only scene', () => {
    it('constructs ImageLabelLoader but NOT LabelLoader when only has_image_labels is set', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasImageLabels: true }));

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(LabelLoader).not.toHaveBeenCalled();
      expect(ImageLabelLoader).toHaveBeenCalledOnce();
      expect(result.labelLoader).toBeUndefined();
      expect(result.imageLabelLoader).toBeDefined();
    });
  });

  describe('success path — both has_labels AND has_image_labels (core.md G15)', () => {
    it('constructs BOTH loaders when a scene declares has_labels and has_image_labels', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true, hasImageLabels: true }));

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      // Both loaders constructed (mutation guard: a regression that
      // swapped `||` for `else if` between the two if-blocks would
      // leave imageLabelLoader undefined in this case).
      expect(LabelLoader).toHaveBeenCalledOnce();
      expect(ImageLabelLoader).toHaveBeenCalledOnce();
      expect(result.labelLoader).toBeDefined();
      expect(result.imageLabelLoader).toBeDefined();
      // PickingSystem still constructed once for the combined scene.
      expect(PickingSystem).toHaveBeenCalledOnce();
    });

    it('constructs BOTH loaders when separate nodes declare has_labels and has_image_labels', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      // Scene with one labels-node and a separate image-labels-node.
      const root = new THREE.Group();
      root.name = 'LuxarScene';
      const labelChild = new THREE.Group();
      labelChild.userData = { attrs: { has_labels: true } };
      const imageChild = new THREE.Group();
      imageChild.userData = { attrs: { has_image_labels: true } };
      root.add(labelChild);
      root.add(imageChild);
      const scene = new THREE.Scene();
      scene.add(root);

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      expect(LabelLoader).toHaveBeenCalledOnce();
      expect(ImageLabelLoader).toHaveBeenCalledOnce();
      expect(result.labelLoader).toBeDefined();
      expect(result.imageLabelLoader).toBeDefined();
    });
  });

  describe('shouldPick predicate wiring (core.md G16)', () => {
    it('wires setShouldPick on the PickingSystem with an overlay-aware predicate', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true }));
      const hasVisibleHoverOverlay = vi.fn().mockReturnValue(false);
      const overlayManager = { hasVisibleHoverOverlay };

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => overlayManager as never,
      });

      // setShouldPick was called with a predicate function.
      const ps = result.pickingSystem as unknown as {
        setShouldPick: ReturnType<typeof vi.fn>;
      };
      expect(ps.setShouldPick).toHaveBeenCalledOnce();
      const predicate = ps.setShouldPick.mock.calls[0][0] as () => boolean;
      expect(typeof predicate).toBe('function');

      // The predicate reads through the live overlayManager accessor.
      // Initially no hover overlay is visible → shouldPick=false
      // (picking system skips the GPU work).
      expect(predicate()).toBe(false);
      expect(hasVisibleHoverOverlay).toHaveBeenCalledOnce();

      // Toggle the overlay state to "visible" — predicate sees the new
      // value (live read, not snapshot) and returns true.
      hasVisibleHoverOverlay.mockReturnValue(true);
      expect(predicate()).toBe(true);
    });

    it('predicate returns false when getOverlayManager() returns undefined', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true }));

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      const ps = result.pickingSystem as unknown as {
        setShouldPick: ReturnType<typeof vi.fn>;
      };
      const predicate = ps.setShouldPick.mock.calls[0][0] as () => boolean;
      // No overlay manager → ?? false → shouldPick=false. The picking
      // system skips the GPU work because there's no consumer for the
      // pick result.
      expect(predicate()).toBe(false);
    });
  });

  describe('selection-consumer provisioning (embedder API)', () => {
    it('initializes picking for a LABEL-LESS scene when a selection consumer exists', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({})); // no has_labels / has_image_labels

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
        hasSelectionConsumer: () => true,
      });

      // Picking pipeline provisioned for the embedder, no tooltip loaders.
      expect(result.pickingSystem).toBeDefined();
      expect(result.labelLoader).toBeUndefined();
      expect(result.imageLabelLoader).toBeUndefined();
      expect(LabelLoader).not.toHaveBeenCalled();
      expect(ImageLabelLoader).not.toHaveBeenCalled();
    });

    it('still skips picking for a label-less scene when nobody consumes selection', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({}));

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
        hasSelectionConsumer: () => false,
      });

      expect(result.pickingSystem).toBeUndefined();
    });

    it('shouldPick predicate ORs the LIVE selection-consumer state with overlay visibility', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({}));
      let consuming = true;

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined, // no tooltip overlay at all
        hasSelectionConsumer: () => consuming,
      });

      const ps = result.pickingSystem as unknown as { setShouldPick: ReturnType<typeof vi.fn> };
      const predicate = ps.setShouldPick.mock.calls[0][0] as () => boolean;

      expect(predicate()).toBe(true); // listener present → picks run
      consuming = false; // embedder unsubscribed...
      expect(predicate()).toBe(false); // ...pick renders stop (live read)
    });

    it('proceeds without tooltip loaders when the store is missing but selection is wanted', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: false })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true })); // labels declared, but no store

      const result = await initPicking({
        sceneManager: makeSceneManager(scene) as never,
        pickingEvents,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
        hasSelectionConsumer: () => true,
      });

      // Selection still works; only the (store-backed) tooltip loaders are skipped.
      expect(result.pickingSystem).toBeDefined();
      expect(result.labelLoader).toBeUndefined();
      expect(result.imageLabelLoader).toBeUndefined();
    });
  });

  describe('success path — listener wiring', () => {
    it('registers mousemove + mouseleave on canvas + resize on window via pickingEvents', async () => {
      (getSceneLoader as unknown as ReturnType<typeof vi.fn>).mockReturnValue(
        makeSceneLoader({ hasStore: true })
      );
      const scene = new THREE.Scene();
      scene.add(makeLuxarRoot({ hasLabels: true }));
      const sm = makeSceneManager(scene);

      // Spy on the EventGroup.on / .add methods to verify wiring.
      const events = new EventGroup();
      const onSpy = vi.spyOn(events, 'on');
      const addSpy = vi.spyOn(events, 'add');

      await initPicking({
        sceneManager: sm as never,
        pickingEvents: events,
        previous: makePreviousEmpty(),
        getOverlayManager: () => undefined,
      });

      // .on covers mousemove + mouseleave + window.resize + window.scroll
      // (scroll busts the cached canvas rect when the canvas moves on screen).
      const targets = onSpy.mock.calls.map((c) => c[1]);
      expect(targets).toContain('mousemove');
      expect(targets).toContain('mouseleave');
      expect(targets).toContain('resize');
      expect(targets).toContain('scroll');

      // core.md W10 strengthening: previously `>=4`. Pin to EXACTLY 4 so
      // a regression that double-registered a cleanup (or added a 5th
      // listener without considering teardown) gets flagged. The four
      // are: controls.removeEventListener('change' | 'start' | 'end')
      // and sceneManager.removeEventListener('camera-changed') — see
      // core/app/picking/init-picking.ts lines 144, 161, 162, 169.
      expect(addSpy.mock.calls.length).toBe(4);
      // Each registered cleanup is a function (not a value / object).
      for (const call of addSpy.mock.calls) {
        expect(typeof call[0]).toBe('function');
      }

      // Direct addEventListener on controls + sceneManager (Three.js
      // EventDispatcher doesn't satisfy the EventTarget type).
      expect(sm.controls.addEventListener).toHaveBeenCalledWith('change', expect.any(Function));
      expect(sm.controls.addEventListener).toHaveBeenCalledWith('start', expect.any(Function));
      expect(sm.controls.addEventListener).toHaveBeenCalledWith('end', expect.any(Function));
      expect(sm.addEventListener).toHaveBeenCalledWith('camera-changed', expect.any(Function));
    });
  });
});
