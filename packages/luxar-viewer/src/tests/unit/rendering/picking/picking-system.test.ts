/**
 * Unit tests for PickingSystem.
 *
 * The class wraps a WebGLRenderTarget and depends on
 * THREE.WebGLRenderer for the actual readRenderTargetPixels
 * / getDrawingBufferSize / domElement.getBoundingClientRect calls. Under
 * jsdom we can:
 *   - stub the renderer with the bare-minimum methods used outside
 *     performPick()
 *   - construct a real WebGLRenderTarget (its constructor is pure JS;
 *     no GL context needed)
 *
 * Tests focus on the registration/lifecycle surface and the exported
 * pure helpers (computePickBufferSize / MAX_PICK_BUFFER_DIM). The
 * actual pick-render path needs a real WebGL context — out of scope.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';
import {
  PickingSystem,
  computePickBufferSize,
  MAX_PICK_BUFFER_DIM,
} from '../../../../rendering/picking/picking-system';

function makeStubRenderer(): THREE.WebGLRenderer {
  // PickingSystem ctor + the lifecycle methods we test only need
  // `renderer` as an opaque reference. domElement is read by
  // onMouseMove via getBoundingClientRect; not exercised here.
  return {
    domElement: document.createElement('canvas'),
    getDrawingBufferSize: vi.fn(),
    readRenderTargetPixels: vi.fn(),
  } as unknown as THREE.WebGLRenderer;
}

function makeStubCapabilities(): import('../../../../rendering/renderer-capabilities').RendererCapabilities {
  // Minimal capabilities for unit tests — only `apiSurface` is read by
  // the picking-system body (to discriminate readback signatures);
  // `framebufferYDown` is added for completeness (the readback
  // primitive consults it, but no readback path is exercised here).
  return {
    apiSurface: 'webgl2',
    framebufferYDown: false,
    hdr: {
      hdr: false,
      p3Gamut: false,
      rec2020Gamut: false,
      recommendedColorSpace: THREE.SRGBColorSpace,
      floatTextures: true,
      colorDepth: { red: 8, green: 8, blue: 8 },
    },
    maxMSAASamples: 0,
    pointSizeRange: [1, 64] as const,
    readBackbufferPixels: vi.fn(),
  } as unknown as import('../../../../rendering/renderer-capabilities').RendererCapabilities;
}

function makeCamera(): THREE.PerspectiveCamera {
  return new THREE.PerspectiveCamera(60, 1.0, 0.1, 1000);
}

describe('computePickBufferSize', () => {
  it('returns half the drawing-buffer size', () => {
    expect(computePickBufferSize(1920, 1080)).toEqual({ w: 960, h: 540 });
  });

  it('clamps both axes to MAX_PICK_BUFFER_DIM', () => {
    // 4K wide → /2 = 1920 → clamp to MAX (1024)
    expect(computePickBufferSize(3840, 2160)).toEqual({
      w: MAX_PICK_BUFFER_DIM,
      h: MAX_PICK_BUFFER_DIM,
    });
  });

  it('floors odd input to integer pixels', () => {
    expect(computePickBufferSize(101, 99)).toEqual({ w: 50, h: 49 });
  });

  it('returns at least 1 pixel per axis even for tiny inputs', () => {
    expect(computePickBufferSize(0, 0)).toEqual({ w: 1, h: 1 });
    expect(computePickBufferSize(1, 1)).toEqual({ w: 1, h: 1 });
  });

  it('handles asymmetric clamping (one axis under cap, other over)', () => {
    expect(computePickBufferSize(1500, 5000)).toEqual({
      w: 750,
      h: MAX_PICK_BUFFER_DIM,
    });
  });
});

describe('PickingSystem — registration', () => {
  let system: PickingSystem;
  const onPickResult = vi.fn();

  beforeEach(() => {
    system = new PickingSystem(
      makeStubRenderer(),
      makeStubCapabilities(),
      makeCamera(),
      onPickResult
    );
    onPickResult.mockClear();
  });

  it('starts with zero registered nodes', () => {
    expect(system.registeredNodeCount).toBe(0);
  });

  it('allocatePickId returns sequentially starting at 1', () => {
    expect(system.allocatePickId()).toBe(1);
    expect(system.allocatePickId()).toBe(2);
    expect(system.allocatePickId()).toBe(3);
  });

  it('registerNode increments registeredNodeCount', () => {
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id);
    expect(system.registeredNodeCount).toBe(1);
  });

  it('registerNode disables matrixAutoUpdate on the pick node', () => {
    const pickNode = new THREE.Object3D();
    system.registerNode(new THREE.Object3D(), pickNode, system.allocatePickId());
    expect(pickNode.matrixAutoUpdate).toBe(false);
    expect(pickNode.matrixWorldAutoUpdate).toBe(false);
  });

  it('unregisterNode decrements registeredNodeCount', () => {
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id);
    expect(system.registeredNodeCount).toBe(1);

    system.unregisterNode(id);
    expect(system.registeredNodeCount).toBe(0);
  });

  it('unregisterNode disposes the pick mesh material', () => {
    const dispose = vi.fn();
    const pickMesh = new THREE.Mesh(new THREE.BufferGeometry(), {
      dispose,
    } as unknown as THREE.Material);
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), pickMesh, id);

    system.unregisterNode(id);
    expect(dispose).toHaveBeenCalled();
  });

  it('unregisterNode disposes every entry in a material array', () => {
    const a = vi.fn();
    const b = vi.fn();
    const pickMesh = new THREE.Mesh(new THREE.BufferGeometry(), [
      { dispose: a },
      { dispose: b },
    ] as unknown as THREE.Material[]);
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), pickMesh, id);

    system.unregisterNode(id);
    expect(a).toHaveBeenCalled();
    expect(b).toHaveBeenCalled();
  });

  it('unregisterNode for an unknown ID is a no-op', () => {
    expect(() => system.unregisterNode(9999)).not.toThrow();
    expect(system.registeredNodeCount).toBe(0);
  });

  // ---------------------------------------------------------------------
  // Cache-invalidation wiring: PickingSystem owns the *lifecycle* of the
  // world-AABB cache (populate on first scan, invalidate on register/
  // unregister/geometry-commit). The cache-logic semantics themselves
  // are unit-tested in `picking-system/ray-aabb.test.ts`; the tests
  // below pin that the orchestrator's public methods reach the cache
  // correctly. Private-field reach-in is the only option here under
  // jsdom — driving a real pick would require a WebGL context.
  // ---------------------------------------------------------------------

  it('invalidateBoxes(pickId) drops just that entry; invalidateBoxes() drops all', () => {
    const cache = (system as unknown as { _worldBoxCache: Map<number, THREE.Box3> })
      ._worldBoxCache;
    cache.set(1, new THREE.Box3());
    cache.set(2, new THREE.Box3());

    system.invalidateBoxes(1);
    expect(cache.has(1)).toBe(false);
    expect(cache.has(2)).toBe(true);

    system.invalidateBoxes();
    expect(cache.size).toBe(0);
  });

  it('unregisterNode drops the corresponding cached world AABB', () => {
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id);
    const cache = (system as unknown as { _worldBoxCache: Map<number, THREE.Box3> })
      ._worldBoxCache;
    cache.set(id, new THREE.Box3());

    system.unregisterNode(id);
    expect(cache.has(id)).toBe(false);
  });
});

describe('PickingSystem — context-restore registration drop', () => {
  let system: PickingSystem;

  beforeEach(() => {
    system = new PickingSystem(makeStubRenderer(), makeStubCapabilities(), makeCamera(), vi.fn());
  });

  it('clearRegistrationsForRebuild empties the node map without disposing materials', () => {
    const dispose = vi.fn();
    const pickMesh = new THREE.Mesh(new THREE.BufferGeometry(), {
      dispose,
    } as unknown as THREE.Material);
    system.registerNode(new THREE.Object3D(), pickMesh, system.allocatePickId());
    expect(system.registeredNodeCount).toBe(1);

    system.clearRegistrationsForRebuild();
    expect(system.registeredNodeCount).toBe(0);
    // Material dispose must NOT be called — its shader is already
    // invalid in a context-loss scenario; calling dispose can throw
    // on some drivers.
    expect(dispose).not.toHaveBeenCalled();
  });

  it('clearRegistrationsForRebuild unregisters CameraAwareMaterial from materialManager', async () => {
    // Pre-fix: pick materials registered with materialManager via
    // node-factory's `materialManager.register(pickMaterial)`. After
    // context-restore, clearRegistrationsForRebuild dropped the
    // node map but left the registry entries. Repeated cycles
    // accumulated stale references; camera-uniform updates would
    // target dead materials and `getCacheStats().totalRegistered`
    // would grow unboundedly.
    const { materialManager } = await import('../../../../rendering/material-manager');

    // Use a real THREE.Material so EventDispatcher is wired (the
    // materialManager subscribes to the 'dispose' event), then graft
    // `updateCameraParams` onto it so isCameraAwareMaterial returns
    // true.
    const baseMaterial = new THREE.MeshBasicMaterial();
    (baseMaterial as unknown as { updateCameraParams: () => void }).updateCameraParams = vi.fn();
    const disposeSpy = vi.spyOn(baseMaterial, 'dispose');
    const pickMaterial = baseMaterial as unknown as THREE.Material;

    // Match what node-factory does: register the pick material with
    // materialManager at construction.
    const before = materialManager.getCacheStats().totalRegistered;
    materialManager.register(
      pickMaterial as unknown as Parameters<typeof materialManager.register>[0]
    );
    expect(materialManager.getCacheStats().totalRegistered).toBe(before + 1);

    const pickMesh = new THREE.Mesh(new THREE.BufferGeometry(), pickMaterial);
    system.registerNode(new THREE.Object3D(), pickMesh, system.allocatePickId());

    system.clearRegistrationsForRebuild();

    // pick material is now unregistered from
    // materialManager — count returns to baseline.
    expect(materialManager.getCacheStats().totalRegistered).toBe(before);
    // Material's own dispose() was NOT called — the shader is
    // already invalid in a context-restore scenario.
    expect(disposeSpy).not.toHaveBeenCalled();
    disposeSpy.mockRestore();
  });

  it('round-trip: clearRegistrationsForRebuild followed by re-register restores node tracking', () => {
    // After context restore, `NodeFactory.rebuildAfterContextRestore`
    // calls `clearRegistrationsForRebuild` then re-registers every
    // scene node with fresh pick materials. This test locks in that
    // the system accepts new registrations cleanly after a clear,
    // with no stale state leaking between rounds.
    const pickMesh1 = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
    const pickMesh2 = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());

    // Initial registration before "context loss".
    system.registerNode(new THREE.Object3D(), pickMesh1, system.allocatePickId());
    expect(system.registeredNodeCount).toBe(1);

    // Simulate context-loss clear.
    system.clearRegistrationsForRebuild();
    expect(system.registeredNodeCount).toBe(0);

    // Simulate post-restore re-registration with a fresh pick mesh.
    const newId = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), pickMesh2, newId);
    expect(system.registeredNodeCount).toBe(1);
    // The new pick ID is unique relative to the original allocation
    // stream (allocatePickId is monotonic — it does not reset across
    // a clear, which prevents the ambiguity of two materials sharing
    // an ID across a restore).
    expect(newId).toBeGreaterThan(1);
  });
});

describe('PickingSystem — camera + suppression', () => {
  let system: PickingSystem;

  beforeEach(() => {
    system = new PickingSystem(makeStubRenderer(), makeStubCapabilities(), makeCamera(), vi.fn());
  });

  it('setCamera replaces the camera reference (no throw)', () => {
    expect(() => system.setCamera(makeCamera())).not.toThrow();
  });

  it('setPostProcessing accepts null (no throw)', () => {
    expect(() => system.setPostProcessing(null)).not.toThrow();
  });

  it('markDirty does not throw when called outside a frame', () => {
    expect(() => system.markDirty()).not.toThrow();
  });

  it('suppress(true) clears any pending debounce timer', () => {
    expect(() => system.suppress(true)).not.toThrow();
    expect(() => system.suppress(false)).not.toThrow();
  });

  it('getDiagnostics reports fresh-construction defaults', () => {
    const d = system.getDiagnostics();
    expect(d.lastPickFiredTime).toBe(0);
    expect(d.lastMouseMoveTime).toBe(0);
    expect(d.lastDirtyTime).toBe(0);
    expect(d.registeredNodeCount).toBe(0);
    expect(d.suppressed).toBe(false);
  });

  it('getDiagnostics reflects suppress + registration state changes', () => {
    system.suppress(true);
    system.registerNode(
      new THREE.Object3D(),
      new THREE.Object3D(),
      system.allocatePickId()
    );
    const d = system.getDiagnostics();
    expect(d.suppressed).toBe(true);
    expect(d.registeredNodeCount).toBe(1);
  });
});

// =============================================================================
// Settle scheduler + caches
// =============================================================================

/**
 * Test harness for the rAF-driven settle scheduler. Controls
 * `performance.now()` so settle windows can be advanced deterministically
 * without real time elapsing, and queues `requestAnimationFrame`
 * callbacks so the rAF tick fires only when `flushRaf()` is called.
 */
function setupSchedulerHarness(): {
  advanceTime: (ms: number) => void;
  flushRaf: () => void;
  setNow: (ms: number) => void;
  restore: () => void;
} {
  let nowMs = 1000;
  let rafQueue: Array<() => void> = [];
  const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => nowMs);
  const rafSpy = vi
    .spyOn(globalThis, 'requestAnimationFrame')
    .mockImplementation((cb: FrameRequestCallback) => {
      rafQueue.push(() => cb(nowMs));
      return rafQueue.length;
    });
  const cafSpy = vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation(() => {});
  return {
    advanceTime: (ms) => {
      nowMs += ms;
    },
    setNow: (ms) => {
      nowMs = ms;
    },
    flushRaf: () => {
      const todo = rafQueue;
      rafQueue = [];
      todo.forEach((fn) => fn());
    },
    restore: () => {
      nowSpy.mockRestore();
      rafSpy.mockRestore();
      cafSpy.mockRestore();
    },
  };
}

function makeMouseEvent(x: number, y: number): MouseEvent {
  return { clientX: x, clientY: y } as MouseEvent;
}

describe('PickingSystem — settle scheduler', () => {
  let system: PickingSystem;
  let onPickResult: ReturnType<typeof vi.fn>;
  let harness: ReturnType<typeof setupSchedulerHarness>;
  let performPick: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    harness = setupSchedulerHarness();
    onPickResult = vi.fn();
    const cb = onPickResult as unknown as ConstructorParameters<typeof PickingSystem>[3];
    system = new PickingSystem(makeStubRenderer(), makeStubCapabilities(), makeCamera(), cb);
    // Stub performPick so the rAF tick is observable without touching GL.
    performPick = vi.fn().mockResolvedValue(undefined);
    (system as unknown as { performPick: typeof performPick }).performPick = performPick;
  });

  afterEach(() => {
    harness.restore();
  });

  it('does not fire a pick before the settle window elapses', () => {
    system.onMouseMove(makeMouseEvent(100, 100));
    harness.advanceTime(50);
    harness.flushRaf();
    harness.advanceTime(50);
    harness.flushRaf();
    expect(performPick).not.toHaveBeenCalled();
  });

  it('fires a pick after the mouse settle window elapses', () => {
    system.onMouseMove(makeMouseEvent(100, 100));
    harness.advanceTime(130);
    // First tick re-checks settle and (likely) reschedules; flush again to
    // catch the tick scheduled past the settle threshold.
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledTimes(1);
  });

  it('camera-axis settle: markDirty within the window keeps pick suppressed', () => {
    system.onMouseMove(makeMouseEvent(100, 100));
    harness.advanceTime(60);
    system.markDirty();
    harness.advanceTime(70); // total 130ms since mousemove, 70ms since markDirty
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).not.toHaveBeenCalled();

    harness.advanceTime(60); // 130ms since markDirty
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledTimes(1);
  });

  it('a new mousemove resets the settle timer', () => {
    system.onMouseMove(makeMouseEvent(100, 100));
    harness.advanceTime(100);
    system.onMouseMove(makeMouseEvent(110, 110));
    harness.advanceTime(100);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).not.toHaveBeenCalled();
    harness.advanceTime(30);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledTimes(1);
  });

  it('shouldPick predicate gates the pick', () => {
    system.setShouldPick(() => false);
    system.onMouseMove(makeMouseEvent(100, 100));
    harness.advanceTime(130);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).not.toHaveBeenCalled();
  });

  it('re-pick on camera-settle: markDirty after a pick fires another pick once camera settles', () => {
    // First settle + pick
    system.onMouseMove(makeMouseEvent(100, 100));
    harness.advanceTime(130);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledTimes(1);

    // Mouse stays still; camera dirties (advance time first so the
    // markDirty timestamp is strictly greater than the pick timestamp —
    // mirrors real rAF-driven camera events arriving on a later frame).
    harness.advanceTime(10);
    system.markDirty();
    harness.advanceTime(130);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledTimes(2);
  });

  it('onMouseLeave clears the pending cursor so no pick fires', () => {
    system.onMouseMove(makeMouseEvent(100, 100));
    system.onMouseLeave();
    harness.advanceTime(200);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).not.toHaveBeenCalled();
  });

  it('suppress(true) cancels any pending rAF and blocks future picks', () => {
    system.onMouseMove(makeMouseEvent(100, 100));
    system.suppress(true);
    harness.advanceTime(200);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).not.toHaveBeenCalled();
  });

  it('suppress(false) re-arms the rAF so orbit-and-release picks once camera settles', () => {
    // Simulate: cursor over a target, user starts orbiting, camera moves
    // every frame, user releases. Without a fresh mousemove, the system
    // should still fire one pick HOVER_SETTLE_MS after the camera goes quiet.
    system.onMouseMove(makeMouseEvent(100, 100));
    system.suppress(true);
    // Camera ticks during the suppressed window
    system.markDirty();
    harness.advanceTime(30);
    system.markDirty();
    harness.advanceTime(30);
    // User releases — camera goes quiet from here on, but no new mousemove
    system.suppress(false);
    harness.advanceTime(130);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledTimes(1);
  });

  it('onMouseMove fades the existing overlay (onPickResult called with null)', () => {
    system.onMouseMove(makeMouseEvent(100, 100));
    expect(onPickResult).toHaveBeenCalledWith(null);
  });

  it('markDirty fades the existing overlay (onPickResult called with null)', () => {
    onPickResult.mockClear();
    system.markDirty();
    expect(onPickResult).toHaveBeenCalledWith(null);
  });
});

// NOTE: pure-function tests for the world-AABB cache and votes-map reuse live
// in `picking-system/ray-aabb.test.ts` and `picking-system/pick-render.test.ts`
// respectively. The orchestrator-level wiring checks that touched private
// fields here previously have been recycled into the registration describe
// block above (cache-invalidation API + unregister side-effect). The
// votes-map-instance check was retired — `pick-render.test.ts` exercises the
// actual reuse semantics with real pixel inputs, which is the more meaningful
// guarantee.

describe('PickingSystem.dispose', () => {
  it('disposes the underlying pick render target', () => {
    const system = new PickingSystem(
      makeStubRenderer(),
      makeStubCapabilities(),
      makeCamera(),
      vi.fn()
    );
    expect(() => system.dispose()).not.toThrow();
  });

  it('double dispose is safe', () => {
    const system = new PickingSystem(
      makeStubRenderer(),
      makeStubCapabilities(),
      makeCamera(),
      vi.fn()
    );
    system.dispose();
    expect(() => system.dispose()).not.toThrow();
  });
});
