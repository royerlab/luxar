// @vitest-environment jsdom
/**
 * Tests for ScreenshotStrategy — the screenshot capture pipeline.
 *
 * Covers: SDR capture, EXR capture, debounce, format fallbacks
 * (JPEG-no-alpha auto-switch), transparent background restore, max-DPR
 * mode, panel-state save/restore, refusal during active recording.
 *
 * The strategy is exercised through `panel.captureScreenshot()` because
 * the strategy is constructed by the Panel with all its hooks wired up;
 * standalone construction would duplicate that wiring without
 * additional coverage.
 *
 * AUDIT NOTE (ui.md C2): several tests reach into private state via
 * `(panel as any).session.X` and write to it (e.g.
 * `session.isRecording = true`). That pins the INTERNAL field shape of
 * session — a rename or restructuring of session state will break these
 * tests even when behavior is unchanged. The "refuses screenshot while
 * recording is active" assertion in particular could be driven through
 * `startVideoRecording()` instead of mutating `session.isRecording`
 * directly. Follow-up: route through public APIs where possible.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// jsdom polyfill for ImageData (not provided by default).
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

// Mock canvas 2D context for offscreen canvases (jsdom doesn't support getContext('2d')).
let canvasToBlobOverride: ((cb: any) => void) | null = null;
const origCreateElement = document.createElement.bind(document);
vi.spyOn(document, 'createElement').mockImplementation((tag: string, options?: any) => {
  const el = origCreateElement(tag, options);
  if (tag === 'canvas') {
    const canvasEl = el as HTMLCanvasElement;
    const origGetContext = canvasEl.getContext.bind(canvasEl);
    (canvasEl as any).getContext = (type: string, ...args: any[]) => {
      if (type === '2d') {
        return { putImageData: vi.fn(), drawImage: vi.fn() };
      }
      return origGetContext(type, ...args);
    };
    (el as HTMLCanvasElement).toBlob = vi.fn((cb: any) => {
      if (canvasToBlobOverride) {
        canvasToBlobOverride(cb);
      } else {
        cb(new Blob(['test'], { type: 'image/png' }));
      }
    });
  }
  return el;
});

// ui.md C1 fix: drop the `ui/gui` mock and exercise the real GUI library
// under jsdom.
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
import { showToast } from '../../../../ui/toast';
import { createMockSceneManager, createMockAnimationController } from './_helpers';

URL.createObjectURL = vi.fn().mockReturnValue('blob:mock-url');
URL.revokeObjectURL = vi.fn();

describe('ScreenshotStrategy', () => {
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

  describe('SDR capture', () => {
    it('calls renderToImageData for non-EXR formats', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      await panel.captureScreenshot();

      expect(mockSceneManager.postProcessing.renderToImageData).toHaveBeenCalled();
      expect(showToast).toHaveBeenCalledWith('Screenshot saved');

      rafSpy.mockRestore();
    });

    it('debounces concurrent screenshot requests', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      const p1 = panel.captureScreenshot();
      // [P11] The inProgress lock is acquired synchronously, before the
      // first await — so it must already be set here. Observing it directly
      // kills a mutation that drops the guard (which a "called once" check
      // alone could survive if the second call happened to be coalesced).
      expect((panel as any).screenshotStrategy.isInProgress()).toBe(true);
      const p2 = panel.captureScreenshot();
      await Promise.all([p1, p2]);

      // Second call is blocked by the inProgress lock.
      expect(mockSceneManager.postProcessing.renderToImageData).toHaveBeenCalledTimes(1);
      // Lock released after both settle.
      expect((panel as any).screenshotStrategy.isInProgress()).toBe(false);

      rafSpy.mockRestore();
    });

    it('surfaces a "Screenshot failed" toast when encoding returns null', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      canvasToBlobOverride = (cb: any) => cb(null);

      await panel.captureScreenshot();

      expect(showToast).toHaveBeenCalledWith('Screenshot failed');

      canvasToBlobOverride = null;
      rafSpy.mockRestore();
    });

    it('saves and restores panel states around capture', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      const mockStates = new Map([['renderingControls', true]]);
      const getStates = vi.fn().mockReturnValue(mockStates);
      const restoreStates = vi.fn();

      panel.setPanelStateCallbacks(getStates, restoreStates);

      await panel.captureScreenshot();

      expect(getStates).toHaveBeenCalled();
      // Once to hide-all, once to restore.
      expect(restoreStates).toHaveBeenCalledTimes(2);

      rafSpy.mockRestore();
    });
  });

  describe('mutual exclusion', () => {
    it('refuses screenshot while real-time recording is active and preserves savedRecordingState', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      const recordingSavedState = {
        dprEnabled: true,
        dpr: 1,
        rendererSize: null,
        resizeLocked: false,
      };
      (panel as any).session.isRecording = true;
      (panel as any).session.savedRecordingState = recordingSavedState;

      const saveStateSpy = vi.spyOn((panel as any).session, 'saveRecordingState');
      const restoreStateSpy = vi.spyOn((panel as any).session, 'restoreRecordingState');

      vi.mocked(showToast).mockClear();
      try {
        await panel.captureScreenshot();

        expect(showToast).toHaveBeenCalledWith('Stop recording before taking a screenshot');
        expect((panel as any).session.savedRecordingState).toBe(recordingSavedState);
        expect(saveStateSpy).not.toHaveBeenCalled();
        expect(restoreStateSpy).not.toHaveBeenCalled();
      } finally {
        (panel as any).session.isRecording = false;
        (panel as any).session.savedRecordingState = null;
        rafSpy.mockRestore();
      }
    });

    it('refuses screenshot while offline capture is active', async () => {
      const recordingSavedState = {
        dprEnabled: false,
        dpr: 2,
        rendererSize: { width: 1920, height: 1080 },
        resizeLocked: true,
      };
      (panel as any).session.isOfflineCaptureActive = true;
      (panel as any).session.savedRecordingState = recordingSavedState;

      const saveStateSpy = vi.spyOn((panel as any).session, 'saveRecordingState');

      vi.mocked(showToast).mockClear();
      try {
        await panel.captureScreenshot();

        expect(showToast).toHaveBeenCalledWith('Stop recording before taking a screenshot');
        expect((panel as any).session.savedRecordingState).toBe(recordingSavedState);
        expect(saveStateSpy).not.toHaveBeenCalled();
      } finally {
        (panel as any).session.isOfflineCaptureActive = false;
        (panel as any).session.savedRecordingState = null;
      }
    });
  });

  describe('transparent background', () => {
    it('sets scene.background to null during capture then restores', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      (panel as any).options.transparentBackground = true;
      const originalBg = mockSceneManager.scene.background;

      // [P2] Capture the background value AT the moment the frame is rendered
      // (mid-capture), not only afterward. A regression that nulls too late,
      // or restores before the render, would pass a restore-only assertion
      // but fail here.
      let bgDuringCapture: unknown = 'unset';
      mockSceneManager.postProcessing.renderToImageData.mockImplementation(async () => {
        bgDuringCapture = mockSceneManager.scene.background;
        return new ImageData(4, 4);
      });

      await panel.captureScreenshot();

      expect(bgDuringCapture).toBeNull();
      expect(mockSceneManager.scene.background).toBe(originalBg);

      rafSpy.mockRestore();
    });

    it('restores background + recording state and clears the lock when capture throws', async () => {
      // [P2/C1] run() has no catch — only a finally — so a mid-capture
      // rejection propagates. The finally MUST still restore the background,
      // restore recording state, and release the debounce lock; otherwise
      // the panel is wedged for every subsequent screenshot.
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      (panel as any).options.transparentBackground = true;
      const originalBg = mockSceneManager.scene.background;
      mockSceneManager.postProcessing.renderToImageData.mockRejectedValue(
        new Error('render failed')
      );
      const restoreSpy = vi.spyOn((panel as any).session, 'restoreRecordingState');

      await expect(panel.captureScreenshot()).rejects.toThrow('render failed');

      expect(mockSceneManager.scene.background).toBe(originalBg);
      expect(restoreSpy).toHaveBeenCalled();
      expect((panel as any).screenshotStrategy.isInProgress()).toBe(false);

      rafSpy.mockRestore();
    });

    it('auto-switches from JPEG to PNG when transparent', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      (panel as any).options.transparentBackground = true;
      (panel as any).options.outputFormat = 'jpeg';

      await panel.captureScreenshot();

      expect(showToast).toHaveBeenCalledWith('Switched to PNG (JPEG has no alpha)');

      rafSpy.mockRestore();
    });
  });

  describe('capture DPR', () => {
    it('freezes adaptation at the capture DPR during the shot, then re-enables', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      const mockDPRManager = {
        isActive: vi.fn().mockReturnValue(true),
        getCurrentDPR: vi.fn().mockReturnValue(1.0),
        getNativeDPR: vi.fn().mockReturnValue(2.0),
        setEnabled: vi.fn(),
      };
      panel.setAdaptiveDPRManager(mockDPRManager as any);
      (panel as any).options.captureDPR = 1.0;

      await panel.captureScreenshot();

      // Adaptation off for the shot, and the ratio pinned explicitly —
      // otherwise the adaptive loop could move the resolution mid-capture.
      expect(mockDPRManager.setEnabled).toHaveBeenCalledWith(false);
      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenCalledWith(1.0);
      expect(mockDPRManager.setEnabled).toHaveBeenCalledWith(true);

      rafSpy.mockRestore();
    });

    /**
     * WYSIWYG is the default, so the capture must NOT reach for the
     * display's DPR of its own accord — a 2x export from a 1x viewport
     * does not match what the user framed, and because thin lines and
     * small points have a minimum size in DEVICE pixels it would not even
     * be a clean upscale.
     */
    it('captures at the requested DPR, not the display DPR (manual-DPR mode)', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      const mockDPRManager = {
        isActive: vi.fn().mockReturnValue(false), // manual-DPR mode
        getCurrentDPR: vi.fn().mockReturnValue(0.5),
        getNativeDPR: vi.fn().mockReturnValue(2.0),
        setEnabled: vi.fn(),
      };
      panel.setAdaptiveDPRManager(mockDPRManager as any);
      (panel as any).options.captureDPR = 1.0;

      await panel.captureScreenshot();

      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenCalledWith(1.0);
      expect(mockSceneManager.setAdaptivePixelRatio).not.toHaveBeenCalledWith(2.0);
      // Restore reapplies the saved manual DPR.
      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenLastCalledWith(0.5);
      expect(mockDPRManager.setEnabled).not.toHaveBeenCalledWith(true);

      rafSpy.mockRestore();
    });

    it('honours an explicitly raised capture DPR', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      const mockDPRManager = {
        isActive: vi.fn().mockReturnValue(false),
        getCurrentDPR: vi.fn().mockReturnValue(1.0),
        getNativeDPR: vi.fn().mockReturnValue(2.0),
        setEnabled: vi.fn(),
      };
      panel.setAdaptiveDPRManager(mockDPRManager as any);
      (panel as any).options.captureDPR = 2.0;

      await panel.captureScreenshot();

      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenCalledWith(2.0);

      rafSpy.mockRestore();
    });
  });

  describe('EXR HDR path', () => {
    it('calls captureHDRAsEXR instead of toBlob for EXR format', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      (panel as any).options.outputFormat = 'exr';

      await panel.captureScreenshot();

      expect(mockSceneManager.postProcessing.captureHDRAsEXR).toHaveBeenCalled();
      expect(mockSceneManager.postProcessing.renderToImageData).not.toHaveBeenCalled();
      expect(showToast).toHaveBeenCalledWith('HDR screenshot saved (EXR)');

      rafSpy.mockRestore();
    });

    it('generates filename with .exr extension', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      (panel as any).options.outputFormat = 'exr';

      const downloadSpy = vi.spyOn(panel as any, 'downloadBlob');

      await panel.captureScreenshot();

      expect(downloadSpy).toHaveBeenCalledWith(expect.any(Blob), expect.stringMatching(/\.exr$/));

      rafSpy.mockRestore();
    });
  });

  // [P8] CaptureStrategy contract parity. Video & Offline strategies have
  // explicit canRun / abort / dispose coverage; Screenshot only had these
  // exercised implicitly. Pin them so the three strategies stay symmetric.
  describe('CaptureStrategy contract (parity with video / offline)', () => {
    it('declares kind "screenshot"', () => {
      expect((panel as any).screenshotStrategy.kind).toBe('screenshot');
    });

    it('canRun is true when idle and false during any active/in-flight capture', () => {
      const s = (panel as any).screenshotStrategy;
      expect(
        s.canRun({ isRecording: false, isOfflineCaptureActive: false, isCaptureInProgress: false })
      ).toBe(true);
      expect(
        s.canRun({ isRecording: true, isOfflineCaptureActive: false, isCaptureInProgress: false })
      ).toBe(false);
      expect(
        s.canRun({ isRecording: false, isOfflineCaptureActive: true, isCaptureInProgress: false })
      ).toBe(false);
    });

    it('abort() is a safe no-op that leaves the in-progress lock clear', () => {
      const s = (panel as any).screenshotStrategy;
      expect(() => s.abort()).not.toThrow();
      expect(s.isInProgress()).toBe(false);
    });
  });
});
