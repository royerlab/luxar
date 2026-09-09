// @vitest-environment jsdom
/**
 * Unit tests for the ResizeOrchestrator extracted from SceneManager.
 *
 * Pin the scheduling contract (rAF coalescing, resizeLocked toggle,
 * dispose cleanup) and the doResize pipeline (renderer setPixelRatio,
 * post-processing resize + DPR scale sync, material refresh).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  ResizeOrchestrator,
  type ResizeCtx,
} from '../../../../../scene/scene-manager/viewport/resize-orchestrator';
import type { Renderer } from '../../../../../rendering/renderer-capabilities';
import type { PostProcessingManager } from '../../../../../rendering';
import * as THREE from 'three';
import { allowHighDPR, setNativeDPR } from '../../../../helpers/device-pixel-ratio';
import {
  DEFAULT_MAX_PIXEL_RATIO,
  setMaxPixelRatioCap,
} from '../../../../../rendering/pixel-ratio-cap';

function makeRenderer() {
  const setPixelRatio = vi.fn();
  const setSize = vi.fn();
  return {
    renderer: {
      setPixelRatio,
      setSize,
    } as unknown as Renderer,
    setPixelRatio,
    setSize,
  };
}

function makePostProcessing() {
  const resize = vi.fn();
  const setDPRScale = vi.fn();
  return {
    pp: { resize, setDPRScale } as unknown as PostProcessingManager,
    resize,
    setDPRScale,
  };
}

function makeCtx(
  opts: {
    withPostProcessing?: boolean;
    withCamera?: boolean;
    pixelRatioOverride?: number | null;
  } = {}
) {
  const { renderer, setPixelRatio, setSize } = makeRenderer();
  const { pp, resize: ppResize, setDPRScale } = makePostProcessing();
  const camera = opts.withCamera === false ? null : new THREE.PerspectiveCamera();
  const updateMaterialsForCurrentCamera = vi.fn();

  const ctx: ResizeCtx = {
    renderer,
    camera,
    postProcessing: opts.withPostProcessing === false ? null : pp,
    pixelRatioOverride: opts.pixelRatioOverride ?? null,
    updateMaterialsForCurrentCamera,
  };

  return {
    ctx,
    setPixelRatio,
    setSize,
    ppResize,
    setDPRScale,
    updateMaterialsForCurrentCamera,
  };
}

describe('ResizeOrchestrator.resizeNow', () => {
  // A 2x display with the cap LIFTED, so the resize pipeline's own
  // assertions read against the display's DPR as they always have. The
  // capped default gets its own test at the end of this block — it is
  // the one place the orchestrator can observe the ceiling, since it
  // resolves the ratio through dpr-policy rather than owning it.
  let restoreCap: () => void;
  let restoreNative: () => void;
  beforeEach(() => {
    restoreNative = setNativeDPR(2);
    restoreCap = allowHighDPR();
  });
  afterEach(() => {
    restoreCap();
    restoreNative();
  });

  it('applies pixelRatio + postProcessing.resize + DPR-scale sync + material refresh', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx();

    orchestrator.resizeNow(1024, 768, harness.ctx);

    expect(harness.setPixelRatio).toHaveBeenCalledWith(2); // native DPR
    expect(harness.ppResize).toHaveBeenCalledWith(1024, 768);
    expect(harness.setDPRScale).toHaveBeenCalledTimes(1);
    expect(harness.updateMaterialsForCurrentCamera).toHaveBeenCalledTimes(1);
    // setSize is NOT called when postProcessing owns sizing.
    expect(harness.setSize).not.toHaveBeenCalled();

    // M4: pin the SEQUENCE, not just the presence, of side effects. Pixel
    // ratio must be set before post-processing resizes, the DPR scale synced
    // after the resize, and the material refresh must run last. A mutant that
    // reordered these (e.g. refreshed materials before resizing) would survive
    // presence-only assertions.
    const order = [
      harness.setPixelRatio.mock.invocationCallOrder[0],
      harness.ppResize.mock.invocationCallOrder[0],
      harness.setDPRScale.mock.invocationCallOrder[0],
      harness.updateMaterialsForCurrentCamera.mock.invocationCallOrder[0],
    ];
    for (let i = 1; i < order.length; i++) {
      expect(order[i]).toBeGreaterThan(order[i - 1]);
    }
  });

  // G3: a zero-size resize (minimized window / detached canvas) must not crash
  // and must still forward the 0×0 dimensions to post-processing.
  it('handles a zero-size resize without throwing', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx();

    expect(() => orchestrator.resizeNow(0, 0, harness.ctx)).not.toThrow();
    expect(harness.ppResize).toHaveBeenCalledWith(0, 0);
  });

  it('falls back to renderer.setSize when postProcessing has not yet been wired', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx({ withPostProcessing: false });

    orchestrator.resizeNow(800, 600, harness.ctx);

    expect(harness.setPixelRatio).toHaveBeenCalled();
    expect(harness.setSize).toHaveBeenCalledWith(800, 600);
    expect(harness.updateMaterialsForCurrentCamera).toHaveBeenCalledTimes(1);
  });

  it('skips material refresh when ctx.camera is null', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx({ withCamera: false });

    orchestrator.resizeNow(800, 600, harness.ctx);

    expect(harness.ppResize).toHaveBeenCalled();
    expect(harness.updateMaterialsForCurrentCamera).not.toHaveBeenCalled();
  });

  it('uses the override DPR when ctx.pixelRatioOverride is set', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx();
    const ctx: ResizeCtx = { ...harness.ctx, pixelRatioOverride: 0.5 };

    orchestrator.resizeNow(800, 600, ctx);

    expect(harness.setPixelRatio).toHaveBeenCalledWith(0.5);
  });

  it('keeps the orbit target at the same screen position across DPR probes', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx();
    const camera = harness.ctx.camera as THREE.PerspectiveCamera;
    const target = new THREE.Vector3(4, -2, 1);
    camera.position.set(10, 5, 12);
    camera.lookAt(target);
    camera.updateMatrixWorld(true);

    const projectedBefore = target.clone().project(camera);
    orchestrator.resizeNow(2509, 1328, { ...harness.ctx, pixelRatioOverride: 1.31 });
    camera.updateMatrixWorld(true);
    const projectedDuringProbe = target.clone().project(camera);
    orchestrator.resizeNow(2509, 1328, { ...harness.ctx, pixelRatioOverride: 0.81 });
    camera.updateMatrixWorld(true);
    const projectedAfterProbe = target.clone().project(camera);

    expect(projectedDuringProbe.x).toBeCloseTo(projectedBefore.x, 12);
    expect(projectedDuringProbe.y).toBeCloseTo(projectedBefore.y, 12);
    expect(projectedAfterProbe.x).toBeCloseTo(projectedBefore.x, 12);
    expect(projectedAfterProbe.y).toBeCloseTo(projectedBefore.y, 12);
  });
});

describe('ResizeOrchestrator.scheduleResize', () => {
  let rafCallback: FrameRequestCallback | null = null;
  let rafId = 0;
  let mockRAF: ReturnType<typeof vi.spyOn>;
  let mockCancelRAF: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    rafCallback = null;
    rafId = 0;
    mockRAF = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallback = cb;
      return ++rafId;
    });
    mockCancelRAF = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
    Object.defineProperty(window, 'innerWidth', { value: 1280, configurable: true });
    Object.defineProperty(window, 'innerHeight', { value: 720, configurable: true });
  });

  afterEach(() => {
    mockRAF.mockRestore();
    mockCancelRAF.mockRestore();
  });

  it('coalesces multiple synchronous resize events into one rAF callback', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx();

    orchestrator.scheduleResize(() => harness.ctx);
    orchestrator.scheduleResize(() => harness.ctx);
    orchestrator.scheduleResize(() => harness.ctx);

    // Three schedules, but the last one cancels the prior two.
    expect(mockRAF).toHaveBeenCalledTimes(3);
    expect(mockCancelRAF).toHaveBeenCalledTimes(2);

    // Fire the surviving rAF.
    rafCallback?.(0);
    expect(harness.ppResize).toHaveBeenCalledTimes(1);
    expect(harness.ppResize).toHaveBeenCalledWith(1280, 720);
  });

  it('is suppressed when resizeLocked=true (recording mode)', () => {
    const orchestrator = new ResizeOrchestrator();
    orchestrator.resizeLocked = true;
    const harness = makeCtx();

    orchestrator.scheduleResize(() => harness.ctx);

    expect(mockRAF).not.toHaveBeenCalled();
  });
});

describe('ResizeOrchestrator.dispose', () => {
  it('cancels any in-flight rAF and clears pending state', () => {
    const orchestrator = new ResizeOrchestrator();
    let rafCallback: FrameRequestCallback | null = null;
    const mockRAF = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallback = cb;
      return 42;
    });
    const mockCancelRAF = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});

    const harness = makeCtx();
    orchestrator.scheduleResize(() => harness.ctx);

    orchestrator.dispose();

    expect(mockCancelRAF).toHaveBeenCalledWith(42);
    // Calling rAF after dispose must not crash even if the orchestrator was disposed.
    expect(() => rafCallback?.(0)).not.toThrow();

    mockRAF.mockRestore();
    mockCancelRAF.mockRestore();
  });
});

/**
 * The orchestrator does not own the ceiling — it resolves the ratio
 * through `dpr-policy.getActivePixelRatio`, which is exactly why the cap
 * lives there. This block is the proof that the seam actually binds on
 * the path every window resize takes, including the very first one,
 * before AdaptiveDPRManager has evaluated a single frame.
 */
describe('ResizeOrchestrator honours the pixel-ratio cap', () => {
  let restoreNative: () => void;
  beforeEach(() => {
    restoreNative = setNativeDPR(2);
    setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);
  });
  afterEach(() => restoreNative());

  it('sizes a null-override resize at the cap, not the display DPR', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx();

    orchestrator.resizeNow(1024, 768, harness.ctx);

    expect(harness.setPixelRatio).toHaveBeenCalledWith(1);
  });

  it('still honours a reduction below the cap', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx({ pixelRatioOverride: 0.5 });

    orchestrator.resizeNow(1024, 768, harness.ctx);

    expect(harness.setPixelRatio).toHaveBeenCalledWith(0.5);
  });

  it('clamps an above-cap override rather than trusting it', () => {
    const orchestrator = new ResizeOrchestrator();
    const harness = makeCtx({ pixelRatioOverride: 2 });

    orchestrator.resizeNow(1024, 768, harness.ctx);

    expect(harness.setPixelRatio).toHaveBeenCalledWith(1);
  });
});
