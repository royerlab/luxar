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
