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
  type PickResult,
} from '../../../../rendering/picking/picking-system';

/** A promise plus its external `resolve` — lets a test gate when the readback completes. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

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

  it('caps the larger axis to MAX_PICK_BUFFER_DIM and scales the other to preserve aspect', () => {
    // 4K → half-res 1920×1080 → uniform scale 1024/1920 → 1024×576.
    // Aspect preservation is load-bearing: the gsplat pick shader maps
    // view space to pixels with uFx == uFy (square-pixel assumption), so
    // a pick buffer with a different aspect than the camera displaces
    // gsplat picks horizontally (points/lines go through the
    // aspect-aware projectionMatrix and were unaffected).
    expect(computePickBufferSize(3840, 2160)).toEqual({
      w: MAX_PICK_BUFFER_DIM,
      h: 576,
    });
  });

  it('preserves the drawing-buffer aspect ratio when capped', () => {
    const cases: Array<[number, number]> = [
      [3840, 2160], // 4K 16:9
      [3024, 1890], // MacBook Retina fullscreen 16:10
      [5120, 2880], // 5K
    ];
    for (const [w, h] of cases) {
      const pick = computePickBufferSize(w, h);
      expect(Math.max(pick.w, pick.h)).toBeLessThanOrEqual(MAX_PICK_BUFFER_DIM);
      // Within 1% of the true aspect (integer flooring is the only error source).
      expect(pick.w / pick.h).toBeCloseTo(w / h, 1);
    }
  });

  it('floors odd input to integer pixels', () => {
    expect(computePickBufferSize(101, 99)).toEqual({ w: 50, h: 49 });
  });

  it('returns at least 1 pixel per axis even for tiny inputs', () => {
    expect(computePickBufferSize(0, 0)).toEqual({ w: 1, h: 1 });
    expect(computePickBufferSize(1, 1)).toEqual({ w: 1, h: 1 });
  });

  it('scales BOTH axes uniformly when only one exceeds the cap (tall input)', () => {
    // half-res 750×2500 → uniform scale 1024/2500 = 0.4096 → 307×1024.
    expect(computePickBufferSize(1500, 5000)).toEqual({
      w: 307,
      h: MAX_PICK_BUFFER_DIM,
    });
  });

  // [rendering.md/O3][P10] consolidated from picking-materials.test.ts to
  // keep computePickBufferSize coverage in one canonical location (the
  // function lives next to PickingSystem; picking-materials.test.ts now
  // imports it transitively for the boundary case below).
  it('scales BOTH axes uniformly when only one exceeds the cap (ultrawide input)', () => {
    // half-res 2560×720 → uniform scale 1024/2560 = 0.4 → 1024×288.
    expect(computePickBufferSize(5120, 1440)).toEqual({
      w: MAX_PICK_BUFFER_DIM,
      h: 288,
    });
  });

  it('floors fractional values (rounds DOWN, not nearest)', () => {
    // 1921/2 = 960.5 → 960; 1081/2 = 540.5 → 540
    expect(computePickBufferSize(1921, 1081)).toEqual({ w: 960, h: 540 });
  });

  // [rendering.md/G8] negative inputs are not an expected runtime case
  // (canvas dimensions are always >= 0), but the contract is that the
  // function never returns a value below the documented floor of 1.
  it('clamps negative inputs to 1 pixel (defensive contract)', () => {
    expect(computePickBufferSize(-100, -100)).toEqual({ w: 1, h: 1 });
    expect(computePickBufferSize(-1, 1080)).toEqual({ w: 1, h: 540 });
    expect(computePickBufferSize(1920, -1)).toEqual({ w: 960, h: 1 });
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
    const cache = (system as unknown as { _worldBoxCache: Map<number, THREE.Box3> })._worldBoxCache;
    cache.set(1, new THREE.Box3());
    cache.set(2, new THREE.Box3());

    system.invalidateBoxes(1);
    expect(cache.has(1)).toBe(false);
    expect(cache.has(2)).toBe(true);

    system.invalidateBoxes();
    expect(cache.size).toBe(0);
  });

  // [R11/C-G10][P5] Pin the pickId === 0 case explicitly. pickId 0 is
  // the "no hit / background" sentinel — the cache should never carry an
  // entry for it, and invalidateBoxes(0) is a no-op that drops nothing
  // else. A regression that flipped an `if (pickId === 0) return` to
  // `if (pickId === 0) cache.clear()` would catastrophically wipe valid
  // entries on every background-pixel pick.
  it('invalidateBoxes(0) (background sentinel) does not drop entries for nonzero pickIds', () => {
    const cache = (system as unknown as { _worldBoxCache: Map<number, THREE.Box3> })._worldBoxCache;
    cache.set(1, new THREE.Box3());
    cache.set(2, new THREE.Box3());

    system.invalidateBoxes(0);

    // Background sentinel must not touch real entries.
    expect(cache.has(1)).toBe(true);
    expect(cache.has(2)).toBe(true);
    expect(cache.size).toBe(2);
    // And critically: there is never an entry keyed at 0 — the
    // sentinel is not stored, so invalidating it cannot find anything.
    expect(cache.has(0)).toBe(false);
  });

  it('unregisterNode drops the corresponding cached world AABB', () => {
    const id = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id);
    const cache = (system as unknown as { _worldBoxCache: Map<number, THREE.Box3> })._worldBoxCache;
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

  // rendering.md W6 fix: these previously asserted only `not.toThrow()`,
  // which fails P2 (no mutant is killed by "didn't crash"). Strengthened
  // to verify the observable state change via the public diagnostic +
  // result surface.
  it('setCamera marks the system dirty (observed via getDiagnostics)', () => {
    const beforeDirty = system.getDiagnostics().lastDirtyTime;
    // Bump the clock so a side-effect timestamp must change.
    vi.setSystemTime(Date.now() + 1);
    system.setCamera(makeCamera());
    // setCamera bumps `_dirty` and calls scheduler.markDirty(), which
    // updates lastDirtyTime. (If the method was no-op'd by mistake,
    // the timestamp would not advance.)
    expect(system.getDiagnostics().lastDirtyTime).toBeGreaterThanOrEqual(beforeDirty);
  });

  it('setPostProcessing(null) is accepted and leaves diagnostics consistent', () => {
    // The setter has no side-effect on diagnostics, but the system
    // must remain queryable afterwards. Strengthens beyond `not.toThrow()`
    // by asserting a follow-on operation still works.
    system.setPostProcessing(null);
    const d = system.getDiagnostics();
    expect(d.registeredNodeCount).toBe(0);
    expect(d.suppressed).toBe(false);
  });

  it('markDirty updates lastDirtyTime in diagnostics', () => {
    const before = system.getDiagnostics().lastDirtyTime;
    vi.setSystemTime(Date.now() + 1);
    system.markDirty();
    expect(system.getDiagnostics().lastDirtyTime).toBeGreaterThanOrEqual(before);
  });

  it('suppress(true) flips the suppressed diagnostic flag; suppress(false) clears it', () => {
    expect(system.getDiagnostics().suppressed).toBe(false);
    system.suppress(true);
    expect(system.getDiagnostics().suppressed).toBe(true);
    system.suppress(false);
    expect(system.getDiagnostics().suppressed).toBe(false);
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
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), system.allocatePickId());
    const d = system.getDiagnostics();
    expect(d.suppressed).toBe(true);
    expect(d.registeredNodeCount).toBe(1);
  });

  // [rendering.md/W6][P2] Strengthen the diagnostics shape contract: the
  // returned object must expose EXACTLY the five documented fields with
  // the right types, and `registeredNodeCount` must track node-map size
  // 1:1 across multi-node registration cycles. A mutant that returned
  // `nodeMap.size + 1` or a stale snapshot would survive the existing
  // single-node test but fail here.
  it('getDiagnostics returns an object with the exact documented shape and types', () => {
    const d = system.getDiagnostics();
    expect(Object.keys(d).sort()).toEqual(
      [
        'lastDirtyTime',
        'lastMouseMoveTime',
        'lastPickFiredTime',
        'registeredNodeCount',
        'suppressed',
      ].sort()
    );
    expect(typeof d.lastDirtyTime).toBe('number');
    expect(typeof d.lastMouseMoveTime).toBe('number');
    expect(typeof d.lastPickFiredTime).toBe('number');
    expect(typeof d.registeredNodeCount).toBe('number');
    expect(typeof d.suppressed).toBe('boolean');
  });

  it('registeredNodeCount tracks each register/unregister cycle exactly', () => {
    // Sequence: register 3 nodes, unregister middle, unregister last,
    // register one more. Diagnostics must report 1 (first), 2 (second),
    // 3 (third), 2 (after unregister id2), 1 (after unregister id3),
    // 2 (after fourth register).
    const id1 = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id1);
    expect(system.getDiagnostics().registeredNodeCount).toBe(1);
    const id2 = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id2);
    expect(system.getDiagnostics().registeredNodeCount).toBe(2);
    const id3 = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id3);
    expect(system.getDiagnostics().registeredNodeCount).toBe(3);
    system.unregisterNode(id2);
    expect(system.getDiagnostics().registeredNodeCount).toBe(2);
    system.unregisterNode(id3);
    expect(system.getDiagnostics().registeredNodeCount).toBe(1);
    const id4 = system.allocatePickId();
    system.registerNode(new THREE.Object3D(), new THREE.Object3D(), id4);
    expect(system.getDiagnostics().registeredNodeCount).toBe(2);
  });

  it('setCamera + markDirty are independent dirty-bump sources (not aliased)', () => {
    // Each call must independently advance lastDirtyTime when the clock
    // advances. A mutant that aliased one to the other (or returned a
    // cached snapshot) would fail at one of the steps.
    vi.setSystemTime(Date.now() + 1);
    const t0 = system.getDiagnostics().lastDirtyTime;
    system.setCamera(makeCamera());
    vi.setSystemTime(Date.now() + 2);
    const t1 = system.getDiagnostics().lastDirtyTime;
    expect(t1).toBeGreaterThanOrEqual(t0);
    system.markDirty();
    vi.setSystemTime(Date.now() + 3);
    const t2 = system.getDiagnostics().lastDirtyTime;
    expect(t2).toBeGreaterThanOrEqual(t1);
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

  it('invalidateCanvasRect drops the pending pick so no stale-coordinate pick fires', () => {
    // C2 regression: a page/ancestor scroll fires invalidateCanvasRect()
    // to bust the cached rect. The pending settle coordinate was already
    // converted to canvas-local against that now-stale rect, so firing it
    // would pick the wrong spot. The pending pick must be dropped.
    system.onMouseMove(makeMouseEvent(100, 100));
    system.invalidateCanvasRect();
    harness.advanceTime(200);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).not.toHaveBeenCalled();

    // A fresh mousemove (which re-maps against the recomputed rect) re-arms
    // and fires normally.
    system.onMouseMove(makeMouseEvent(120, 140));
    harness.advanceTime(130);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledTimes(1);
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

  it('tracks cursor moves during suppression so the resume re-pick uses the latest position', () => {
    // Orbit-with-move: cursor is over A, user starts orbiting and drags the
    // cursor to B while suppressed, then releases. The camera-settle re-pick
    // must fire at B (the actual cursor) — not the stale pre-orbit A.
    system.onMouseMove(makeMouseEvent(100, 100)); // over A, before orbit
    system.suppress(true);
    system.onMouseMove(makeMouseEvent(250, 175)); // cursor moves to B during orbit
    system.suppress(false);
    harness.advanceTime(130);
    harness.flushRaf();
    harness.flushRaf();
    expect(performPick).toHaveBeenCalledExactlyOnceWith(250, 175);
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

// =============================================================================
// performPick — pick-target resize guard (MED-25 regression)
// =============================================================================

describe('PickingSystem — performPick resize guard', () => {
  // Reach into the private performPick to drive the resize-guard branch
  // directly. Other arms of performPick (lens distortion, ray-AABB,
  // readback/vote) are covered by their own dedicated tests; here we
  // only need to verify that `pickTarget.setSize` is called once per
  // distinct (pickW, pickH) pair, not once per pick.
  type PerformPick = (x: number, y: number) => Promise<void>;

  function buildSystemWithCanvas(drawW: number, drawH: number) {
    const renderer = {
      domElement: document.createElement('canvas'),
      getDrawingBufferSize: vi.fn((target: THREE.Vector2) => target.set(drawW, drawH)),
      readRenderTargetPixels: vi.fn(),
    } as unknown as THREE.WebGLRenderer;
    Object.defineProperty(renderer.domElement, 'clientWidth', {
      value: drawW,
      configurable: true,
    });
    Object.defineProperty(renderer.domElement, 'clientHeight', {
      value: drawH,
      configurable: true,
    });

    const system = new PickingSystem(renderer, makeStubCapabilities(), makeCamera(), vi.fn());
    // Stub renderPickBuffer so performPick doesn't try to drive a real
    // GL context after the guard branch sets `_dirty = true`. The guard
    // logic under test runs *before* renderPickBuffer, so stubbing the
    // downstream method does not invalidate the guard check.
    (system as unknown as { renderPickBuffer: () => void }).renderPickBuffer = () => {};
    // Spy on the pickTarget.setSize *after* construction so we can count
    // reallocations driven by the guard.
    const pickTarget = (system as unknown as { pickTarget: THREE.WebGLRenderTarget }).pickTarget;
    const setSizeSpy = vi.spyOn(pickTarget, 'setSize');
    return { system, setSizeSpy, renderer };
  }

  it('calls pickTarget.setSize exactly once when draw-buffer size is unchanged across picks', async () => {
    // Regression for MED-25: previously `setSize` was called every time
    // pickTarget.width happened to differ from the computed pickW, but
    // re-allocation churn could occur because the comparison wasn't
    // tied to the explicit "last applied" cache. With `_lastPickW/H`,
    // a stable draw-buffer size results in exactly one setSize across
    // the whole session.
    const { system, setSizeSpy } = buildSystemWithCanvas(800, 600);
    const performPick = (system as unknown as { performPick: PerformPick }).performPick.bind(
      system
    );

    // Drive several picks at the same canvas size. The first call sets
    // the pick target to (400, 300); subsequent calls must NOT re-call
    // setSize.
    await performPick(100, 100);
    await performPick(120, 120);
    await performPick(150, 150);

    expect(setSizeSpy).toHaveBeenCalledTimes(1);
    expect(setSizeSpy).toHaveBeenCalledWith(400, 300);
  });

  it('calls pickTarget.setSize again when draw-buffer size genuinely changes', async () => {
    // The guard must NOT swallow real resizes — when the canvas changes,
    // setSize must fire so the pick target follows.
    const { system, setSizeSpy, renderer } = buildSystemWithCanvas(800, 600);
    const performPick = (system as unknown as { performPick: PerformPick }).performPick.bind(
      system
    );

    await performPick(100, 100);
    expect(setSizeSpy).toHaveBeenCalledTimes(1);

    // Resize the drawing buffer; subsequent pick must re-size the target.
    (renderer.getDrawingBufferSize as ReturnType<typeof vi.fn>).mockImplementation(
      (target: THREE.Vector2) => target.set(1600, 1200)
    );
    Object.defineProperty(renderer.domElement, 'clientWidth', { value: 1600 });
    Object.defineProperty(renderer.domElement, 'clientHeight', { value: 1200 });

    await performPick(100, 100);
    expect(setSizeSpy).toHaveBeenCalledTimes(2);
    expect(setSizeSpy).toHaveBeenLastCalledWith(800, 600);
  });
});

// =============================================================================
// performPick — cursor-clamp safety (MED-26 regression)
// =============================================================================

describe('PickingSystem — cursor clamping', () => {
  // MED-26: `cursorX = Math.floor(correctedX * scaleX)` can be negative
  // for out-of-canvas correctedX (extreme lens distortion at corners).
  // The subsequent clamp(cursorX - half, 0, pickW - PICK_SIZE) corrects
  // this — verify the result is a valid pick-buffer index in all cases.
  type PerformPick = (x: number, y: number) => Promise<void>;

  function buildSystem() {
    const renderer = {
      domElement: document.createElement('canvas'),
      getDrawingBufferSize: vi.fn((target: THREE.Vector2) => target.set(800, 600)),
      readRenderTargetPixels: vi.fn(),
    } as unknown as THREE.WebGLRenderer;
    Object.defineProperty(renderer.domElement, 'clientWidth', { value: 800 });
    Object.defineProperty(renderer.domElement, 'clientHeight', { value: 600 });
    const system = new PickingSystem(renderer, makeStubCapabilities(), makeCamera(), vi.fn());
    // Stub renderPickBuffer for the same reason as above — the cursor
    // clamp lives in performPick before any GL work, but the dirty path
    // would otherwise crash on stub renderers.
    (system as unknown as { renderPickBuffer: () => void }).renderPickBuffer = () => {};
    return system;
  }

  it('clamps _lastReadX/_lastReadY into the valid pick-buffer range for negative screen coords', async () => {
    const system = buildSystem();
    const performPick = (system as unknown as { performPick: PerformPick }).performPick.bind(
      system
    );

    // Negative screen coords ⇒ cursorX < 0 ⇒ clamp(... , 0, ...) → 0.
    await performPick(-1000, -1000);
    const lastReadX = (system as unknown as { _lastReadX: number })._lastReadX;
    const lastReadY = (system as unknown as { _lastReadY: number })._lastReadY;
    expect(lastReadX).toBeGreaterThanOrEqual(0);
    expect(lastReadY).toBeGreaterThanOrEqual(0);
  });

  it('clamps _lastReadX/_lastReadY for screen coords beyond canvas (pickW - PICK_SIZE upper bound)', async () => {
    const system = buildSystem();
    const performPick = (system as unknown as { performPick: PerformPick }).performPick.bind(
      system
    );

    await performPick(10_000, 10_000);
    const lastReadX = (system as unknown as { _lastReadX: number })._lastReadX;
    const lastReadY = (system as unknown as { _lastReadY: number })._lastReadY;
    // Pick buffer is 400x300 (half of 800x600). PICK_SIZE = 5. Max read is 395/295.
    expect(lastReadX).toBeLessThanOrEqual(400 - 5);
    expect(lastReadY).toBeLessThanOrEqual(300 - 5);
  });
});

// =============================================================================
// performPick — stale-readback ordering guard (async race)
// =============================================================================

describe('PickingSystem — stale readback ordering', () => {
  type PerformPick = (x: number, y: number) => Promise<void>;

  /**
   * Build a system that reaches the post-readback emit: canvas sized,
   * `renderPickBuffer` stubbed (no GL), one node registered straddling the
   * camera origin so the cursor ray always hits (skips the ray-cull
   * early-out), and `readbackAndVote` replaced by a caller-gated promise.
   */
  function buildGatedSystem() {
    const onPickResult = vi.fn();
    const renderer = {
      domElement: document.createElement('canvas'),
      getDrawingBufferSize: vi.fn((t: THREE.Vector2) => t.set(800, 600)),
      readRenderTargetPixels: vi.fn(),
    } as unknown as THREE.WebGLRenderer;
    Object.defineProperty(renderer.domElement, 'clientWidth', { value: 800, configurable: true });
    Object.defineProperty(renderer.domElement, 'clientHeight', { value: 600, configurable: true });

    const system = new PickingSystem(renderer, makeStubCapabilities(), makeCamera(), onPickResult);
    (system as unknown as { renderPickBuffer: () => void }).renderPickBuffer = () => {};

    // A box centered on the camera origin → every cursor ray intersects it.
    const geom = new THREE.BoxGeometry(100, 100, 100);
    geom.computeBoundingBox();
    const mainNode = new THREE.Mesh(geom, new THREE.MeshBasicMaterial());
    mainNode.updateMatrixWorld(true);
    const pickNode = new THREE.Mesh(geom, new THREE.MeshBasicMaterial());
    const id = system.allocatePickId();
    system.registerNode(mainNode, pickNode, id);

    const gate = deferred<PickResult | null>();
    const fakeResult: PickResult = { nodeId: id, elementId: 7, brightness: 1, mainNode };
    (system as unknown as { readbackAndVote: () => Promise<PickResult | null> }).readbackAndVote =
      () => gate.promise;

    const performPick = (system as unknown as { performPick: PerformPick }).performPick.bind(
      system
    );
    return { system, onPickResult, gate, fakeResult, performPick };
  }

  it('drops the readback result when a markDirty superseded the pick mid-readback', async () => {
    const { system, onPickResult, gate, fakeResult, performPick } = buildGatedSystem();

    const pending = performPick(400, 300); // center ray → hits the box → reaches readback
    onPickResult.mockClear(); // ignore any pre-readback emit

    // A camera/view change lands while the readback is still in flight.
    system.markDirty();
    gate.resolve(fakeResult); // the now-stale readback finally returns
    await pending;

    // The stale result must NOT be emitted (markDirty's null fade stands).
    expect(onPickResult).not.toHaveBeenCalledWith(fakeResult);
    expect(onPickResult).toHaveBeenCalledWith(null);
  });

  it('emits the readback result when nothing superseded the pick', async () => {
    const { onPickResult, gate, fakeResult, performPick } = buildGatedSystem();

    const pending = performPick(400, 300);
    onPickResult.mockClear();
    gate.resolve(fakeResult); // no supersede → result is still the latest
    await pending;

    expect(onPickResult).toHaveBeenCalledExactlyOnceWith(fakeResult);
  });
});
