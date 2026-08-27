// @vitest-environment jsdom
/**
 * Tests for LiveOverlayCompositor — the mirror canvas that lets the
 * REAL-TIME (MediaRecorder) capture path carry DOM overlays.
 *
 * The behaviours worth pinning are the ones a plausible refactor breaks
 * silently: that the blit happens on `frame-end` (the only moment the
 * `preserveDrawingBuffer: false` canvas still has pixels), that `detach`
 * really unsubscribes (a per-frame listener outliving its recording is a
 * permanent tax on every later frame), and that a throwing overlay draw
 * cannot escape into the animation loop.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../../utils/log', () => ({
  log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
  Modules: { RECORDING: 'Recording' },
}));

const compositeOverlaysMock = vi.fn();
vi.mock('../../../../ui/recording-panel/overlay-compositor', () => ({
  compositeOverlays: (...args: unknown[]) => compositeOverlaysMock(...args),
}));

import {
  LiveOverlayCompositor,
  createLiveOverlayCompositor,
} from '../../../../ui/recording-panel/live-overlay-compositor';
import { eventBus } from '../../../../utils/cross-layer/event-bus';
import { log } from '../../../../utils/log';
import type { OverlayManager } from '../../../../ui/overlay-manager';

/** A 2D context stub — jsdom implements none of these. */
interface FakeCtx {
  clearRect: ReturnType<typeof vi.fn>;
  drawImage: ReturnType<typeof vi.fn>;
}

let fakeCtx: FakeCtx;
let origGetContext: typeof HTMLCanvasElement.prototype.getContext;
let contextAvailable = true;

function makeGlCanvas(width = 1600, height = 900): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/** An OverlayManager double reporting `count` visible overlays. */
function makeOverlayManager(count: number): OverlayManager {
  return {
    getVisibleOverlays: vi.fn(() => Array.from({ length: count }, () => ({}))),
  } as unknown as OverlayManager;
}

beforeEach(() => {
  compositeOverlaysMock.mockReset();
  vi.mocked(log.warning).mockReset();
  contextAvailable = true;
  fakeCtx = { clearRect: vi.fn(), drawImage: vi.fn() };
  origGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = vi.fn(() =>
    contextAvailable ? fakeCtx : null
  ) as unknown as typeof HTMLCanvasElement.prototype.getContext;
});

afterEach(() => {
  HTMLCanvasElement.prototype.getContext = origGetContext;
});

describe('createLiveOverlayCompositor', () => {
  it('returns null when the user did not ask for overlays', () => {
    expect(createLiveOverlayCompositor(false, makeOverlayManager(3), makeGlCanvas())).toBeNull();
  });

  it('returns null when there is no overlay manager', () => {
    expect(createLiveOverlayCompositor(true, null, makeGlCanvas())).toBeNull();
  });

  it('returns null when the scene has no visible overlays', () => {
    // The whole point of the null case: a recording with nothing to
    // composite must pay no per-frame cost at all.
    expect(createLiveOverlayCompositor(true, makeOverlayManager(0), makeGlCanvas())).toBeNull();
    expect(eventBus.hasListeners('frame-end')).toBe(false);
  });

  it('returns an attached compositor sized to the GL canvas when overlays exist', () => {
    const gl = makeGlCanvas(2942, 1602);
    const compositor = createLiveOverlayCompositor(true, makeOverlayManager(4), gl);

    expect(compositor).not.toBeNull();
    expect(compositor!.canvas).not.toBe(gl);
    expect(compositor!.canvas.width).toBe(2942);
    expect(compositor!.canvas.height).toBe(1602);

    compositor!.detach();
  });

  it('degrades to null (not a throw) when a 2D context is unavailable', () => {
    contextAvailable = false;
    expect(createLiveOverlayCompositor(true, makeOverlayManager(2), makeGlCanvas())).toBeNull();
    expect(log.warning).toHaveBeenCalled();
    expect(eventBus.hasListeners('frame-end')).toBe(false);
  });
});

describe('LiveOverlayCompositor per-frame compositing', () => {
  it('blits the GL canvas and draws overlays on every frame-end', () => {
    const gl = makeGlCanvas(1600, 900);
    const manager = makeOverlayManager(2);
    const compositor = createLiveOverlayCompositor(true, manager, gl)!;

    // attach() paints once eagerly so the stream's first frame is not blank.
    expect(fakeCtx.drawImage).toHaveBeenCalledTimes(1);

    eventBus.emit('frame-end', {});
    eventBus.emit('frame-end', {});

    expect(fakeCtx.drawImage).toHaveBeenCalledTimes(3);
    // Cleared first: a transparent-background recording would otherwise
    // show earlier frames through the current one.
    expect(fakeCtx.clearRect).toHaveBeenCalledTimes(3);
    expect(fakeCtx.clearRect).toHaveBeenLastCalledWith(0, 0, 1600, 900);
    expect(compositeOverlaysMock).toHaveBeenCalledTimes(3);
    expect(compositeOverlaysMock).toHaveBeenLastCalledWith(compositor.canvas, fakeCtx, manager, gl);

    compositor.detach();
  });

  it('holds the frame geometry fixed when the GL canvas changes size', () => {
    // MediaRecorder is fed one frame size for the whole recording, so a
    // GL canvas that resizes anyway must be SCALED into the mirror, never
    // allowed to change the stream's geometry.
    const gl = makeGlCanvas(1600, 900);
    const compositor = createLiveOverlayCompositor(true, makeOverlayManager(1), gl)!;

    gl.width = 800;
    gl.height = 450;
    eventBus.emit('frame-end', {});

    expect(compositor.canvas.width).toBe(1600);
    expect(compositor.canvas.height).toBe(900);
    expect(fakeCtx.drawImage).toHaveBeenLastCalledWith(gl, 0, 0, 1600, 900);

    compositor.detach();
  });

  it('stops compositing after detach', () => {
    const compositor = createLiveOverlayCompositor(true, makeOverlayManager(1), makeGlCanvas())!;
    compositor.detach();

    const before = fakeCtx.drawImage.mock.calls.length;
    eventBus.emit('frame-end', {});

    expect(fakeCtx.drawImage).toHaveBeenCalledTimes(before);
    expect(eventBus.hasListeners('frame-end')).toBe(false);
  });

  it('detach is idempotent', () => {
    const compositor = createLiveOverlayCompositor(true, makeOverlayManager(1), makeGlCanvas())!;
    compositor.detach();
    expect(() => compositor.detach()).not.toThrow();
  });

  it('attaching twice composites once per frame, not twice', () => {
    const compositor = new LiveOverlayCompositor(makeGlCanvas(), makeOverlayManager(1));
    compositor.attach();
    compositor.attach();
    fakeCtx.drawImage.mockClear();

    eventBus.emit('frame-end', {});

    expect(fakeCtx.drawImage).toHaveBeenCalledTimes(1);
    compositor.detach();
  });

  it('swallows a per-frame failure and warns once, not every frame', () => {
    // A throw escaping here would propagate out of eventBus.emit into
    // AnimationController.animate() on every single frame.
    compositeOverlaysMock.mockImplementation(() => {
      throw new Error('overlay boom');
    });
    const compositor = createLiveOverlayCompositor(true, makeOverlayManager(1), makeGlCanvas())!;

    expect(() => {
      eventBus.emit('frame-end', {});
      eventBus.emit('frame-end', {});
    }).not.toThrow();

    expect(log.warning).toHaveBeenCalledTimes(1);
    expect(vi.mocked(log.warning).mock.calls[0][1]).toContain('overlay boom');

    compositor.detach();
  });
});
