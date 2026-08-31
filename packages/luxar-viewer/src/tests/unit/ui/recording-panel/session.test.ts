// @vitest-environment jsdom
/**
 * Tests for RecordingSession — the shared scaffolding owned by Panel
 * and used by all three strategies.
 *
 * Covers: confirmation dialog (DOM structure, keyboard / mouse routing,
 * ARIA semantics, focus trap, focus restore), recording indicator
 * widget (creation, removal, interval cleanup, click listener cleanup),
 * dispose-during-dialog cleanup, slider-sync teardown on dispose.
 *
 * The Session is exercised through `panel.session.X(...)` since the
 * Panel wires the SceneManager / AnimationController / hooks into
 * Session via the constructor; standalone construction would just
 * duplicate that wiring.
 *
 * AUDIT NOTE (ui.md C2): the `panel.session.X(...)` accessor is itself a
 * private surface the tests reach into. Renaming `panel.session` to
 * something else breaks every test in this file — even when the public
 * RecordingSession contract is unchanged. The trade-off is conscious:
 * RecordingSession is wired by the Panel and isn't independently
 * constructible without duplicating the wiring. Follow-up: extract a
 * public `panel.getSession()` accessor (or expose the strategies a
 * `Session` interface explicitly) so the test contract doesn't depend
 * on a private field name.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';

if (typeof globalThis.ImageData === 'undefined') {
  (globalThis as any).ImageData = class ImageData {
    data: Uint8ClampedArray;
    width: number;
    height: number;
    constructor(widthOrData: number | Uint8ClampedArray, heightOrWidth: number, height?: number) {
      if (widthOrData instanceof Uint8ClampedArray) {
        this.data = widthOrData;
        this.width = heightOrWidth;
        this.height = height ?? widthOrData.length / (4 * heightOrWidth);
      } else {
        this.width = widthOrData;
        this.height = heightOrWidth;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  };
}

// ui.md C1 fix: drop the `ui/gui` mock and exercise the real GUI library
// under jsdom. `ui/gui/core/*.test.ts` already proves the library runs
// cleanly in jsdom; the previous mock left panel ↔ GUI wiring regressions
// invisible because the mock always returned a controller-like object.
vi.mock('../../../../config', () => ({ config: { ui: { zIndex: { recordingPanel: 1500 } } } }));
vi.mock('../../../../ui/toast', () => ({ showToast: vi.fn() }));
vi.mock('../../../../utils/log', () => ({
  log: { info: vi.fn(), warning: vi.fn(), error: vi.fn() },
  Modules: { RECORDING: 'Recording' },
}));
vi.mock('../../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: vi.fn().mockReturnValue({ ndim: 4, displayed: [0, 1, 2] }),
    getDimensionNames: vi.fn().mockReturnValue(['x', 'y', 'z', 'time']),
    getDimensionRanges: vi.fn().mockReturnValue([
      [0, 100],
      [0, 100],
      [0, 100],
      [0, 50],
    ]),
    setDimensionValue: vi.fn(),
    hasNonDisplayedDimensions: vi.fn().mockReturnValue(true),
  },
}));

import { RecordingPanel } from '../../../../ui/recording-panel';
import { createMockSceneManager, createMockAnimationController } from './_helpers';
import {
  DEFAULT_MAX_PIXEL_RATIO,
  getMaxPixelRatioCap,
  setMaxPixelRatioCap,
} from '../../../../rendering/pixel-ratio-cap';

describe('RecordingSession', () => {
  let panel: RecordingPanel;
  let mockSceneManager: any;
  let mockAnimController: any;

  beforeEach(() => {
    document.body.innerHTML = '';
    mockSceneManager = createMockSceneManager();
    mockAnimController = createMockAnimationController();
    panel = new RecordingPanel(mockSceneManager, mockAnimController);
  });

  afterEach(() => {
    panel.dispose();
    document.body.innerHTML = '';
  });

  describe('DPR save/restore around a capture', () => {
    function makeDPRManager(active: boolean, currentDPR: number) {
      return {
        isActive: vi.fn().mockReturnValue(active),
        getCurrentDPR: vi.fn().mockReturnValue(currentDPR),
        getNativeDPR: vi.fn().mockReturnValue(2.0),
        setEnabled: vi.fn(),
      };
    }

    it('adaptive-enabled: freezes adaptation and pins the capture DPR, restore re-enables', () => {
      const manager = makeDPRManager(true, 1.4);
      panel.setAdaptiveDPRManager(manager as any);

      panel.session.saveRecordingState({ captureDPR: 1.0 });
      expect(manager.setEnabled).toHaveBeenCalledWith(false);
      // Applied UNCONDITIONALLY now. It used to lean on setEnabled(false)
      // resetting to native, but that reset lands on the CEILING, which
      // is not necessarily the ratio the capture asked for.
      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenCalledWith(1.0);

      panel.session.restoreRecordingState();
      expect(manager.setEnabled).toHaveBeenLastCalledWith(true);
    });

    it('manual-DPR mode: pins the capture DPR, restore reapplies the manual value', () => {
      const manager = makeDPRManager(false, 0.5);
      panel.setAdaptiveDPRManager(manager as any);

      panel.session.saveRecordingState({ captureDPR: 1.0 });
      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenCalledWith(1.0);

      panel.session.restoreRecordingState();
      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenLastCalledWith(0.5);
      expect(manager.setEnabled).not.toHaveBeenCalledWith(true);
    });

    it('without captureDPR the session leaves DPR entirely alone', () => {
      const manager = makeDPRManager(true, 1.4);
      panel.setAdaptiveDPRManager(manager as any);

      panel.session.saveRecordingState({});
      panel.session.restoreRecordingState();

      expect(manager.setEnabled).not.toHaveBeenCalledWith(false);
      expect(mockSceneManager.setAdaptivePixelRatio).not.toHaveBeenCalled();
    });

    /**
     * The capture override: an export ABOVE the on-screen ceiling must
     * actually render at that ratio. The renderer boundary clamps to the
     * cap, so the session has to lift the cap for the duration — and put
     * it back, or the whole session would keep rendering at the export's
     * resolution afterwards.
     */
    it('lifts the pixel-ratio cap for an above-ceiling capture and restores it', () => {
      const manager = makeDPRManager(false, 1.0);
      panel.setAdaptiveDPRManager(manager as any);
      setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);

      panel.session.saveRecordingState({ captureDPR: 2.0 });
      expect(getMaxPixelRatioCap()).toBe(2.0);
      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenCalledWith(2.0);

      panel.session.restoreRecordingState();
      expect(getMaxPixelRatioCap()).toBe(DEFAULT_MAX_PIXEL_RATIO);
    });

    it('leaves the cap alone for a WYSIWYG capture at the ceiling', () => {
      const manager = makeDPRManager(false, 1.0);
      panel.setAdaptiveDPRManager(manager as any);
      setMaxPixelRatioCap(DEFAULT_MAX_PIXEL_RATIO);

      panel.session.saveRecordingState({ captureDPR: 1.0 });
      expect(getMaxPixelRatioCap()).toBe(DEFAULT_MAX_PIXEL_RATIO);

      panel.session.restoreRecordingState();
      expect(getMaxPixelRatioCap()).toBe(DEFAULT_MAX_PIXEL_RATIO);
    });
  });

  describe('capture-resolution alignment', () => {
    /**
     * Point the mock at a viewport of the given DISPLAY size and give it
     * the two collaborators the scale-resolution branch touches: a real
     * PerspectiveCamera (the branch is gated on `instanceof`) and the
     * material refresh.
     *
     * `renderer.getSize()` is set to the SSAA-multiplied size the real
     * renderer would report, so a session that reads the renderer
     * instead of the post-processing display size is visible here rather
     * than passing on a coincidence.
     */
    function withCanvas(width: number, height: number): THREE.PerspectiveCamera {
      const scale = mockSceneManager.postProcessing.getEffectiveRenderScale();
      mockSceneManager.postProcessing.getDisplaySize = vi.fn().mockReturnValue({ width, height });
      mockSceneManager.renderer.getSize = vi
        .fn()
        .mockReturnValue({ x: Math.round(width * scale), y: Math.round(height * scale) });
      mockSceneManager.updateMaterialsForCurrentCamera = vi.fn();
      const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
      mockSceneManager.camera = camera;
      return camera;
    }

    it('records the 1080p preset at exactly 1920x1080', () => {
      // Multiples of 16 are not an encoder requirement — H.264/H.265 with
      // yuv420p need EVEN dimensions and encoders pad internally to their
      // own macroblock size. Truncating to 16 recorded 1072 lines from the
      // menu entry labelled 1080p, while the real-time path (which passes
      // no alignment at all) recorded 1080 from the same entry.
      withCanvas(1920, 1080);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1080, alignEven: true },
      });
      expect(mockSceneManager.postProcessing.resize).toHaveBeenCalledWith(1920, 1080);
    });

    it('keeps a Native odd height, and the canvas aspect with it', () => {
      // "Native" is documented as the canvas's own size: a 1512×850 CSS
      // canvas at DPR 2 is 3024×1700, not 3024×1696. And the width comes
      // from the ALIGNED height, so the output aspect tracks the source
      // instead of widening the horizontal FOV the user framed.
      const camera = withCanvas(3024, 1700);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1700, alignEven: true },
      });
      expect(mockSceneManager.postProcessing.resize).toHaveBeenCalledWith(3024, 1700);
      expect(camera.aspect).toBeCloseTo(3024 / 1700, 6);
    });

    it('never resizes a tiny canvas to zero', () => {
      // A canvas smaller than the alignment used to truncate to 0, giving
      // `resize(0, 0)` and `camera.aspect = 0/0`. Unreachable while every
      // target height was a preset ≥ 1080; Native makes it reachable.
      const camera = withCanvas(2, 1);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1, alignEven: true },
      });
      const [w, h] = mockSceneManager.postProcessing.resize.mock.calls[0];
      expect(w).toBeGreaterThanOrEqual(2);
      expect(h).toBeGreaterThanOrEqual(2);
      expect(Number.isFinite(camera.aspect)).toBe(true);
    });

    it('never resizes to a NaN aspect when the canvas has no height yet', () => {
      // A hidden or not-yet-laid-out container reports height 0, so the
      // aspect is Infinity. The old 16-alignment masked that with a
      // bitwise AND (`Infinity & ~15` is 0), giving `resize(0, 1072)` and
      // a zero camera aspect; arithmetic alignment carries the Infinity
      // instead, so the aspect has to fall back on its own.
      const camera = withCanvas(1920, 0);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1080, alignEven: true },
      });
      const [w, h] = mockSceneManager.postProcessing.resize.mock.calls[0];
      expect(Number.isFinite(w)).toBe(true);
      expect(Number.isFinite(h)).toBe(true);
      expect(w).toBeGreaterThanOrEqual(2);
      expect(h).toBeGreaterThanOrEqual(2);
      expect(Number.isFinite(camera.aspect)).toBe(true);
      // Square, not a two-pixel-wide sliver: the aspect itself falls back
      // rather than being caught downstream by the alignment's own guard.
      expect(camera.aspect).toBe(1);
    });

    it('keeps the SSAA-multiplied frame size even, and close to what was asked for', () => {
      // The encoder sees the PHYSICAL frame: post-processing renders at
      // ssaaMultiplier × the size handed to `resize`, and that is what
      // lands in the ZIP. A 3024×1698 Native target at 1.5× captures at
      // 4536×2547 — x265 rejects the odd height outright and writes a
      // 0-byte file after the whole archive has been downloaded.
      //
      // Only the PRODUCT has to be even; the requested size never
      // reaches an encoder. Requiring both made the walk step by the
      // multiplier's denominator and give up on perfectly legal scales.
      mockSceneManager.postProcessing.getEffectiveRenderScale = vi.fn().mockReturnValue(1.5);
      withCanvas(3024, 1698);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1698, alignEven: true },
      });
      const [w, h] = mockSceneManager.postProcessing.resize.mock.calls[0];
      expect(Math.round(w * 1.5) % 2).toBe(0);
      expect(Math.round(h * 1.5) % 2).toBe(0);
      // Alignment shrinks, never grows, and only by a pixel or two —
      // parity alone would also be satisfied by a frame 1.5× too big.
      expect(h).toBeLessThanOrEqual(1698);
      expect(h).toBeGreaterThan(1698 - 6);
      expect(w).toBeLessThanOrEqual(3024);
      expect(w).toBeGreaterThan(3024 - 6);
    });

    it('finds an even physical frame at a scale where even-only candidates cannot', () => {
      // At a 1.05× multiplier the nearest even height whose product is
      // also even is six even steps down from 1700 — past the cap — so a
      // walk restricted to even candidates gives up and ships 1700 →
      // 1785, an odd physical height x265 refuses. Stepping by one finds
      // 1699 → 1784 on the first try.
      mockSceneManager.postProcessing.getEffectiveRenderScale = vi.fn().mockReturnValue(1.05);
      withCanvas(3024, 1700);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1700, alignEven: true },
      });
      const [w, h] = mockSceneManager.postProcessing.resize.mock.calls[0];
      expect(Math.round(h * 1.05) % 2).toBe(0);
      expect(Math.round(w * 1.05) % 2).toBe(0);
      expect(h).toBe(1699);
    });

    it('aligns the DERIVED width too, not just the height', () => {
      // The width comes from the aligned height × the source aspect, so
      // it can land odd even when the height is even: a 3024×1698
      // viewport recording the 1080p preset derives 1923.
      withCanvas(3024, 1698);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1080, alignEven: true },
      });
      const [w, h] = mockSceneManager.postProcessing.resize.mock.calls[0];
      expect(h).toBe(1080);
      expect(w).toBe(1922);
    });

    it('restores the DISPLAY size after a capture, not the SSAA-multiplied one', () => {
      // `renderer.getSize()` reports the SSAA-multiplied size (post-
      // processing hands the renderer `display × multiplier` and puts
      // the display size on the canvas CSS), while `resize()` takes the
      // display size and applies the multiplier itself. Snapshotting the
      // renderer's number therefore grew the viewport by the multiplier
      // on every capture, and compounded on the next one.
      mockSceneManager.postProcessing.getEffectiveRenderScale = vi.fn().mockReturnValue(2);
      withCanvas(1512, 850);
      panel.session.saveRecordingState({
        scaleResolution: { targetH: 1700, alignEven: true },
      });
      panel.session.restoreRecordingState();

      const calls = mockSceneManager.postProcessing.resize.mock.calls;
      expect(calls.at(-1)).toEqual([1512, 850]);
    });

    it('leaves the height alone when no alignment is asked for', () => {
      // The real-time path passes `{ targetH }` only.
      withCanvas(1920, 1080);
      panel.session.saveRecordingState({ scaleResolution: { targetH: 1081 } });
      expect(mockSceneManager.postProcessing.resize).toHaveBeenCalledWith(1922, 1081);
    });
  });

  describe('confirmation dialog', () => {
    it('creates dialog with correct structure', () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });

      const dialog = document.querySelector('.luxar-recording-confirm');
      expect(dialog).toBeTruthy();
      expect(dialog?.querySelector('.luxar-recording-confirm__title')?.textContent).toBe(
        'Start Video Recording'
      );
      expect(dialog?.querySelector('[data-action="start"]')).toBeTruthy();
      expect(dialog?.querySelector('[data-action="cancel"]')).toBeTruthy();

      (dialog?.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      return promise.then((result: boolean) => {
        expect(result).toBe(false);
      });
    });

    it('resolves true when Start is clicked', () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });

      const startBtn = document.querySelector('[data-action="start"]') as HTMLElement;
      startBtn?.click();

      return promise.then((result: boolean) => {
        expect(result).toBe(true);
      });
    });

    it('resolves false when Escape is pressed', () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });

      const overlay = document.querySelector('.luxar-recording-confirm') as HTMLElement;
      overlay?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

      return promise.then((result: boolean) => {
        expect(result).toBe(false);
      });
    });

    it('resolves true when Enter is pressed', () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });

      const overlay = document.querySelector('.luxar-recording-confirm') as HTMLElement;
      overlay?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));

      return promise.then((result: boolean) => {
        expect(result).toBe(true);
      });
    });

    it('removes dialog after resolution', async () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });

      const cancelBtn = document.querySelector('[data-action="cancel"]') as HTMLElement;
      cancelBtn?.click();

      await promise;
      expect(document.querySelector('.luxar-recording-confirm')).toBeNull();
    });

    it('sets modal-dialog ARIA attributes', async () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });

      const overlay = document.querySelector('.luxar-recording-confirm') as HTMLElement;
      expect(overlay.getAttribute('role')).toBe('dialog');
      expect(overlay.getAttribute('aria-modal')).toBe('true');
      expect(overlay.getAttribute('aria-labelledby')).toBe('luxar-recording-confirm-title');
      expect(overlay.getAttribute('aria-describedby')).toBe('luxar-recording-confirm-message');
      expect(overlay.querySelector('#luxar-recording-confirm-title')).toBeTruthy();
      expect(overlay.querySelector('#luxar-recording-confirm-message')).toBeTruthy();

      (overlay.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      await promise;
    });

    it('focuses the primary Start button on open', async () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });
      const startBtn = document.querySelector(
        '.luxar-recording-confirm__btn--primary'
      ) as HTMLElement;
      expect(document.activeElement).toBe(startBtn);

      (document.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      await promise;
    });

    it('restores focus to the previously-focused element on close', async () => {
      const sentinel = document.createElement('button');
      sentinel.id = 'before-dialog';
      document.body.appendChild(sentinel);
      sentinel.focus();
      expect(document.activeElement).toBe(sentinel);

      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });
      expect(document.activeElement).not.toBe(sentinel);

      (document.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      await promise;

      expect(document.activeElement).toBe(sentinel);
      sentinel.remove();
    });

    // The dialog's promised frame count must match what the capture loop
    // actually produces (`offline-capture-strategy` ceil(360/speed * fps))
    // and what the panel's own Output field shows (`getTurntableInfo`).
    // Rounding the duration BEFORE the multiply broke both: at 7°/s and
    // 30 FPS it promised 1530 frames / 51s against 1543 / 51.4s.
    it.each([
      [7, 30, '1543 frames at 30 FPS (51.4s video)'],
      [36, 30, '300 frames at 30 FPS (10.0s video)'],
    ])(
      'turntable dialog at %i°/s, %i FPS promises the count the capture produces',
      async (turntableSpeed, videoFPS, expected) => {
        const promise = (panel as any).session.showConfirmationDialog({
          mode: 'turntable',
          options: { ...(panel as any).options, turntableSpeed, videoFPS },
        });

        const message = document.querySelector('#luxar-recording-confirm-message');
        expect(message?.textContent).toContain(expected);

        (document.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
        await promise;
      }
    );

    // The overlay warning is now EXR-only: the real-time WebM path
    // composites overlays onto a mirror canvas (see
    // `live-overlay-compositor.ts`), so warning about it there would be
    // a lie that pushes the user off a mode that works.
    it.each([
      ['webm', 'video', false],
      ['webm', 'turntable', false],
      ['mp4', 'turntable', false],
      ['exr', 'turntable', true],
    ] as const)(
      '%s output in %s mode warns about dropped overlays: %s',
      async (outputFormat, mode, expected) => {
        (panel as any).session.overlayManager = {
          getVisibleOverlays: () => [{}],
        };
        const promise = (panel as any).session.showConfirmationDialog({
          mode,
          options: {
            ...(panel as any).options,
            outputFormat,
            includeOverlays: true,
            frameByFrame: false,
          },
        });

        const message = document.querySelector('#luxar-recording-confirm-message');
        expect(message?.textContent?.includes('Overlays will NOT be included')).toBe(expected);

        (document.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
        await promise;
      }
    );

    it('does not warn about overlays when the scene has none', async () => {
      (panel as any).session.overlayManager = { getVisibleOverlays: () => [] };
      const promise = (panel as any).session.showConfirmationDialog({
        mode: 'turntable',
        options: { ...(panel as any).options, outputFormat: 'exr', includeOverlays: true },
      });

      const message = document.querySelector('#luxar-recording-confirm-message');
      expect(message?.textContent).not.toContain('Overlays will NOT be included');

      (document.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      await promise;
    });

    it('resolves to false when panel disposes mid-dialog', async () => {
      const promise = (panel as any).session.showConfirmationDialog({
        mode: (panel as any).mode,
        options: (panel as any).options,
      });
      expect(document.querySelector('.luxar-recording-confirm')).toBeTruthy();

      panel.dispose();

      await expect(promise).resolves.toBe(false);
      expect(document.querySelector('.luxar-recording-confirm')).toBeNull();
    });
  });

  describe('recording indicator widget', () => {
    it('creates indicator element when shown', () => {
      (panel as any).session.showRecordingIndicator();

      const indicator = document.querySelector('.luxar-recording-indicator');
      expect(indicator).toBeTruthy();
      expect(indicator?.querySelector('.luxar-recording-indicator__dot')).toBeTruthy();
      expect(indicator?.querySelector('.luxar-recording-indicator__text')?.textContent).toBe('REC');
    });

    it('removes indicator when hidden', () => {
      (panel as any).session.showRecordingIndicator();
      expect(document.querySelector('.luxar-recording-indicator')).toBeTruthy();

      (panel as any).session.hideRecordingIndicator();
      expect(document.querySelector('.luxar-recording-indicator')).toBeNull();
    });

    // [ui.md/C7 / Phase F] Verifies the OBSERVABLE timer-cleanup
    // contract rather than spying on `clearInterval`. The previous
    // assertion (`expect(clearIntervalSpy).toHaveBeenCalled()`) would
    // silently pass under any future refactor that switched from
    // `setInterval` → `setTimeout`-recursion / `requestAnimationFrame`,
    // because vitest's own jsdom teardown also invokes `clearInterval`.
    //
    // The observable: the indicator's `__time` element receives a fresh
    // textContent value on every interval tick. After hide, that timer
    // must NOT fire again — captured by checking textContent remains
    // unchanged after additional simulated time passes (the detached
    // DOM node is still alive in memory; a leaked interval would still
    // mutate its textContent).
    it('stops the per-second time updates when hiding (observable contract)', () => {
      vi.useFakeTimers();
      try {
        // Anchor recordingStartTime to "now" so the elapsed-time
        // formatting produces a sensible MM:SS value.
        (panel as any).session.recordingStartTime = Date.now();
        (panel as any).session.showRecordingIndicator();
        const timeEl = document.querySelector('.luxar-recording-indicator__time') as HTMLElement;
        expect(timeEl).toBeTruthy();

        // [P2] Pin the exact MM:SS formatting, not just "!= 00:00". After 1s
        // the label reads 00:01; after a full minute it must roll over to
        // 01:05 — catching a broken padStart or minutes/seconds calculation.
        vi.advanceTimersByTime(1100);
        expect(timeEl.textContent).toBe('00:01');
        vi.advanceTimersByTime(64000); // ~65.1s total elapsed
        expect(timeEl.textContent).toBe('01:05');

        (panel as any).session.hideRecordingIndicator();

        // The indicator DOM node has been detached, but the timeEl
        // reference is still live. If `hideRecordingIndicator()` failed
        // to clear the interval, subsequent ticks would still mutate
        // timeEl.textContent.
        const textRightAfterHide = timeEl.textContent;
        vi.advanceTimersByTime(5000);
        expect(timeEl.textContent).toBe(textRightAfterHide);
      } finally {
        vi.useRealTimers();
      }
    });

    it('removes the click listener when hiding', () => {
      (panel as any).session.showRecordingIndicator();
      const indicator = document.querySelector('.luxar-recording-indicator') as HTMLElement;
      const removeSpy = vi.spyOn(indicator, 'removeEventListener');

      (panel as any).session.hideRecordingIndicator();

      expect(removeSpy).toHaveBeenCalledWith('click', expect.any(Function));
    });
  });

  describe('slider sync', () => {
    it('clears delayed slider playback on dispose', () => {
      vi.useFakeTimers();
      const mockAnimManager = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        play: vi.fn(),
      };
      panel.setAnimationManager(mockAnimManager as any);
      (panel as any).options.syncDimensionIndex = 3;

      (panel as any).session.startSliderSync(3, () => panel.stopVideoRecording());
      panel.dispose();
      vi.advanceTimersByTime(150);

      expect(mockAnimManager.play).not.toHaveBeenCalled();
      expect(mockAnimManager.removeEventListener).toHaveBeenCalledWith(
        'complete',
        expect.any(Function)
      );

      vi.useRealTimers();
    });
  });
});
