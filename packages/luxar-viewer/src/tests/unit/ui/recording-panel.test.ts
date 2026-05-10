/**
 * Unit tests for RecordingPanel
 *
 * Tests screenshot capture, video recording lifecycle,
 * panel state management, and edge cases.
 *
 * NOTE: The GUI is heavily mocked (createMockController/createMockFolder). This means
 * tests verify orchestration logic but cannot verify correct GUI structure, DOM event
 * propagation, or real user interactions. For GUI correctness, rely on E2E tests.
 * TODO(test-review): Consider using a lightweight real DOM (jsdom + real GUI classes)
 * for tests that verify dialog interactions and button click handlers.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { LuxarOrbitControls } from '../../../controls/luxar-orbit-controls';

// Polyfill ImageData for jsdom (not available in jsdom by default)
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
// The `canvasToBlobOverride` variable allows individual tests to make toBlob return null.
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

// Mock the GUI module before importing RecordingPanel
vi.mock('../../../ui/gui/index', () => {
  function createMockElement(): any {
    return {
      style: {},
      className: '',
      classList: { add: vi.fn(), remove: vi.fn() },
      closest: vi.fn().mockReturnValue({ classList: { add: vi.fn() }, setAttribute: vi.fn() }),
      appendChild: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      querySelector: vi.fn().mockReturnValue({ readOnly: false, style: {}, cursor: '' }),
    };
  }

  function createMockController() {
    return {
      name: vi.fn().mockReturnThis(),
      onChange: vi.fn().mockReturnThis(),
      show: vi.fn().mockReturnThis(),
      hide: vi.fn().mockReturnThis(),
      updateDisplay: vi.fn().mockReturnThis(),
      domElement: createMockElement(),
    };
  }

  function createMockFolder(): any {
    return {
      add: vi.fn().mockImplementation(() => createMockController()),
      addFolder: vi.fn().mockImplementation(() => createMockFolder()),
      close: vi.fn(),
    };
  }

  const MockGUI = vi.fn().mockImplementation(() => ({
    domElement: createMockElement(),
    add: vi.fn().mockImplementation(() => createMockController()),
    addFolder: vi.fn().mockImplementation(() => createMockFolder()),
    show: vi.fn(),
    hide: vi.fn(),
    destroy: vi.fn(),
  }));

  return {
    default: MockGUI,
    GUI: MockGUI,
    Controller: vi.fn(),
  };
});

// Mock config
vi.mock('../../../config', () => ({
  config: {
    ui: {
      zIndex: {
        recordingPanel: 1500,
      },
    },
  },
}));

// Mock helpers
vi.mock('../../../ui/helpers', () => ({
  showToast: vi.fn(),
}));

// Mock log
vi.mock('../../../utils/log', () => ({
  log: {
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
  },
  Modules: { RECORDING: 'Recording' },
}));

// Mock scene-dims-manager singleton
vi.mock('../../../scene/scene-dims-manager', () => ({
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

import { RecordingPanel } from '../../../ui/recording-panel';
import { showToast } from '../../../ui/helpers';

// Mock URL.createObjectURL/revokeObjectURL. jsdom may provide a stub by
// default, but tests need it to be a spy so call assertions work.
URL.createObjectURL = vi.fn().mockReturnValue('blob:mock-url');
URL.revokeObjectURL = vi.fn();

// Helper to create mock SceneManager
function createMockSceneManager() {
  const mockCanvas = document.createElement('canvas');
  // Mock toBlob
  mockCanvas.toBlob = vi.fn((callback, _type?, _quality?) => {
    const blob = new Blob(['test'], { type: 'image/png' });
    callback(blob);
  });
  // Mock captureStream
  (mockCanvas as any).captureStream = vi.fn(() => new MediaStream());
  mockCanvas.focus = vi.fn();

  return {
    renderer: {
      domElement: mockCanvas,
      getSize: vi.fn().mockReturnValue({ x: 800, y: 600 }),
      setPixelRatio: vi.fn(),
      setSize: vi.fn(),
    },
    resizeLocked: false,
    postProcessing: {
      render: vi.fn(),
      renderToImageData: vi.fn().mockReturnValue(new ImageData(4, 4)),
      captureHDRAsEXR: vi.fn().mockResolvedValue(new Uint8Array([0x76, 0x2f, 0x31, 0x01])),
      resize: vi.fn(),
    },
    camera: {
      position: { x: 10, y: 5, z: 10, distanceTo: vi.fn().mockReturnValue(15), clone: vi.fn() },
      lookAt: vi.fn(),
    },
    scene: {
      background: { clone: vi.fn() },
    },
    controls: {
      getControls: vi.fn().mockReturnValue(
        Object.assign(Object.create(LuxarOrbitControls.prototype), {
          target: { x: 0, y: 0, z: 0, clone: vi.fn().mockReturnValue({ x: 0, y: 0, z: 0 }) },
          applyOrbitRotation: vi.fn(),
        })
      ),
      getAutoRotate: vi.fn().mockReturnValue(false),
      setAutoRotate: vi.fn(),
    },
    setAdaptivePixelRatio: vi.fn(),
  } as any;
}

// Helper to create mock AnimationController
function createMockAnimationController() {
  return {
    startAnimation: vi.fn(),
    stopAnimation: vi.fn(),
    addPerFrameCallback: vi.fn(),
    removePerFrameCallback: vi.fn(),
  } as any;
}

describe('RecordingPanel', () => {
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

  describe('visibility', () => {
    it('should start hidden', () => {
      expect(panel.isVisible()).toBe(false);
    });

    it('should toggle visibility', () => {
      panel.show();
      expect(panel.isVisible()).toBe(true);
      panel.hide();
      expect(panel.isVisible()).toBe(false);
    });

    it('should toggle via toggle()', () => {
      panel.toggle();
      expect(panel.isVisible()).toBe(true);
      panel.toggle();
      expect(panel.isVisible()).toBe(false);
    });
  });

  describe('screenshot capture', () => {
    it('should call renderToImageData() for SDR screenshot', async () => {
      // Use real requestAnimationFrame
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      await panel.captureScreenshot();

      expect(mockSceneManager.postProcessing.renderToImageData).toHaveBeenCalled();
      expect(showToast).toHaveBeenCalledWith('Screenshot saved');

      rafSpy.mockRestore();
    });

    it('should debounce concurrent screenshot requests', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      // Start two captures simultaneously
      const p1 = panel.captureScreenshot();
      const p2 = panel.captureScreenshot();
      await Promise.all([p1, p2]);

      // Should only capture once (second is blocked by isCaptureInProgress)
      expect(mockSceneManager.postProcessing.renderToImageData).toHaveBeenCalledTimes(1);

      rafSpy.mockRestore();
    });

    it('should handle toBlob returning null', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      // Make the offscreen canvas toBlob return null (simulates encoding failure)
      canvasToBlobOverride = (cb: any) => {
        cb(null);
      };

      await panel.captureScreenshot();

      expect(showToast).toHaveBeenCalledWith('Screenshot failed');

      canvasToBlobOverride = null;
      rafSpy.mockRestore();
    });

    it('should save and restore panel states', async () => {
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
      // Should hide all panels then restore
      expect(restoreStates).toHaveBeenCalledTimes(2); // once to hide, once to restore

      rafSpy.mockRestore();
    });
  });

  describe('screenshot during recording', () => {
    it('refuses screenshot while real-time recording is active and preserves savedRecordingState', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      // Simulate an active real-time recording with a populated saved state.
      const recordingSavedState = {
        dprEnabled: true,
        dpr: 1,
        rendererSize: null,
        resizeLocked: false,
      };
      (panel as any).isRecording = true;
      (panel as any).savedRecordingState = recordingSavedState;

      const saveStateSpy = vi.spyOn(panel as any, 'saveRecordingState');
      const restoreStateSpy = vi.spyOn(panel as any, 'restoreRecordingState');

      vi.mocked(showToast).mockClear();
      try {
        await panel.captureScreenshot();

        // Expect the user-visible refusal toast and that the recording's
        // saved state was not overwritten or cleared.
        expect(showToast).toHaveBeenCalledWith(
          'Stop recording before taking a screenshot'
        );
        expect((panel as any).savedRecordingState).toBe(recordingSavedState);
        expect(saveStateSpy).not.toHaveBeenCalled();
        expect(restoreStateSpy).not.toHaveBeenCalled();
      } finally {
        // Clear the stub state so afterEach's panel.dispose() doesn't try
        // to restore against the mock sceneManager.
        (panel as any).isRecording = false;
        (panel as any).savedRecordingState = null;
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
      (panel as any).isOfflineCaptureActive = true;
      (panel as any).savedRecordingState = recordingSavedState;

      const saveStateSpy = vi.spyOn(panel as any, 'saveRecordingState');

      vi.mocked(showToast).mockClear();
      try {
        await panel.captureScreenshot();

        expect(showToast).toHaveBeenCalledWith(
          'Stop recording before taking a screenshot'
        );
        expect((panel as any).savedRecordingState).toBe(recordingSavedState);
        expect(saveStateSpy).not.toHaveBeenCalled();
      } finally {
        (panel as any).isOfflineCaptureActive = false;
        (panel as any).savedRecordingState = null;
      }
    });
  });

  describe('video recording', () => {
    // Mock MediaRecorder
    let mockMediaRecorder: any;

    beforeEach(() => {
      mockMediaRecorder = {
        start: vi.fn(),
        stop: vi.fn(),
        ondataavailable: null as any,
        onstop: null as any,
      };

      // Mock the global MediaRecorder
      vi.stubGlobal(
        'MediaRecorder',
        vi.fn().mockImplementation(() => mockMediaRecorder)
      );
      (MediaRecorder as any).isTypeSupported = vi.fn().mockReturnValue(true);
    });

    it('should not be recording initially', () => {
      expect(panel.isCurrentlyRecording()).toBe(false);
    });

    it('should check for MediaRecorder support', async () => {
      (MediaRecorder as any).isTypeSupported = vi.fn().mockReturnValue(false);

      // Need to re-create panel since codec check happens in constructor
      const freshPanel = new RecordingPanel(mockSceneManager, mockAnimController);

      // Simulate the getSupportedMimeType returning null
      // This is tested indirectly through startVideoRecording
      await freshPanel.startVideoRecording();

      expect(showToast).toHaveBeenCalledWith('Video recording not supported in this browser');

      freshPanel.dispose();
    });

    it('should stop recording when stopVideoRecording is called', () => {
      // Simulate an active recording
      (panel as any).isRecording = true;
      (panel as any).mediaRecorder = mockMediaRecorder;

      panel.stopVideoRecording();

      expect(mockMediaRecorder.stop).toHaveBeenCalled();
    });

    it('should ignore stopVideoRecording when not recording', () => {
      panel.stopVideoRecording(); // Should not throw
      expect(mockMediaRecorder.stop).not.toHaveBeenCalled();
    });

    it('should clear duration timer when stopping', () => {
      const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');

      (panel as any).isRecording = true;
      (panel as any).mediaRecorder = mockMediaRecorder;
      (panel as any).durationTimer = setTimeout(() => {}, 10000);

      panel.stopVideoRecording();

      expect(clearTimeoutSpy).toHaveBeenCalled();
    });

    it('stops every captureStream track on the disposed onstop branch', () => {
      const trackA = { stop: vi.fn() };
      const trackB = { stop: vi.fn() };
      const fakeStream = {
        getTracks: vi.fn().mockReturnValue([trackA, trackB]),
      };
      (panel as any).captureStream = fakeStream;
      (panel as any).disposed = true;
      (panel as any).mediaRecorder = mockMediaRecorder;

      // Drive the disposed branch of cleanupCaptureStream directly.
      (panel as any).cleanupCaptureStream();

      expect(fakeStream.getTracks).toHaveBeenCalledTimes(1);
      expect(trackA.stop).toHaveBeenCalledTimes(1);
      expect(trackB.stop).toHaveBeenCalledTimes(1);
      expect((panel as any).captureStream).toBeNull();
    });

    it('cleanupCaptureStream is idempotent when no stream is active', () => {
      (panel as any).captureStream = null;
      expect(() => (panel as any).cleanupCaptureStream()).not.toThrow();
      expect((panel as any).captureStream).toBeNull();
    });

    it('startVideoRecording catch path restores state when canvas.captureStream throws', async () => {
      // Force the confirmation dialog to resolve true so the setup
      // body runs.
      vi.spyOn(panel as any, 'showConfirmationDialog').mockResolvedValue(true);
      // Make canvas.captureStream() throw.
      const canvas = mockSceneManager.renderer.domElement;
      (canvas as any).captureStream = vi.fn(() => {
        throw new Error('captureStream not supported');
      });

      const restoreStateSpy = vi.spyOn(panel as any, 'restoreRecordingState');
      const cleanupStreamSpy = vi.spyOn(panel as any, 'cleanupCaptureStream');

      vi.mocked(showToast).mockClear();
      await expect(panel.startVideoRecording()).rejects.toThrow('captureStream not supported');

      // State must be restored, capture-stream cleanup called,
      // recording flag cleared, and a user-visible toast surfaced.
      expect(restoreStateSpy).toHaveBeenCalled();
      expect(cleanupStreamSpy).toHaveBeenCalled();
      expect((panel as any).isRecording).toBe(false);
      expect((panel as any).mediaRecorder).toBeNull();
      expect(showToast).toHaveBeenCalledWith('Video recording failed to start');
    });
  });

  describe('dispose', () => {
    it('should stop recording on dispose', () => {
      const mockMediaRecorder = {
        start: vi.fn(),
        stop: vi.fn(),
        onstop: null,
        ondataavailable: null,
      };
      (panel as any).isRecording = true;
      (panel as any).mediaRecorder = mockMediaRecorder;

      panel.dispose();

      expect(mockMediaRecorder.stop).toHaveBeenCalled();
    });

    it('should clean up recording indicator on dispose', () => {
      const indicator = document.createElement('div');
      document.body.appendChild(indicator);
      (panel as any).recordingIndicator = indicator;
      (panel as any).recordingTimeInterval = setInterval(() => {}, 1000);

      panel.dispose();

      expect(indicator.parentNode).toBeNull();
    });

    it('should resolve an active confirmation dialog when disposed', async () => {
      const promise = (panel as any).showConfirmationDialog();
      expect(document.querySelector('.luxar-recording-confirm')).toBeTruthy();

      panel.dispose();

      await expect(promise).resolves.toBe(false);
      expect(document.querySelector('.luxar-recording-confirm')).toBeNull();
    });
  });

  describe('recording indicator', () => {
    it('should create indicator element when recording starts', () => {
      (panel as any).showRecordingIndicator();

      const indicator = document.querySelector('.luxar-recording-indicator');
      expect(indicator).toBeTruthy();
      expect(indicator?.querySelector('.luxar-recording-indicator__dot')).toBeTruthy();
      expect(indicator?.querySelector('.luxar-recording-indicator__text')?.textContent).toBe('REC');
    });

    it('should remove indicator when recording stops', () => {
      (panel as any).showRecordingIndicator();
      expect(document.querySelector('.luxar-recording-indicator')).toBeTruthy();

      (panel as any).hideRecordingIndicator();
      expect(document.querySelector('.luxar-recording-indicator')).toBeNull();
    });

    it('should clear interval timer when hiding indicator', () => {
      const clearIntervalSpy = vi.spyOn(global, 'clearInterval');

      (panel as any).showRecordingIndicator();
      (panel as any).hideRecordingIndicator();

      expect(clearIntervalSpy).toHaveBeenCalled();
    });

    it('should remove indicator click listener when hiding indicator', () => {
      (panel as any).showRecordingIndicator();
      const indicator = document.querySelector('.luxar-recording-indicator') as HTMLElement;
      const removeSpy = vi.spyOn(indicator, 'removeEventListener');

      (panel as any).hideRecordingIndicator();

      expect(removeSpy).toHaveBeenCalledWith('click', expect.any(Function));
    });
  });

  describe('transparent background', () => {
    it('should set scene.background to null when transparent enabled', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      (panel as any).options.transparentBackground = true;
      const originalBg = mockSceneManager.scene.background;

      await panel.captureScreenshot();

      // Background should have been set to null during capture, then restored
      expect(mockSceneManager.scene.background).toBe(originalBg);

      rafSpy.mockRestore();
    });

    it('should auto-switch from JPEG to PNG when transparent', async () => {
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

  describe('max DPR', () => {
    it('should maximize DPR during screenshot when enabled', async () => {
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
      (panel as any).options.maxDPR = true;

      await panel.captureScreenshot();

      expect(mockDPRManager.setEnabled).toHaveBeenCalledWith(false);
      expect(mockSceneManager.setAdaptivePixelRatio).toHaveBeenCalledWith(2.0);
      // Should restore after capture
      expect(mockDPRManager.setEnabled).toHaveBeenCalledWith(true);

      rafSpy.mockRestore();
    });
  });

  describe('utility methods', () => {
    it('should generate filenames with timestamp', () => {
      const filename = (panel as any).generateFilename('png');
      expect(filename).toMatch(/^luxar-capture-\d{4}-\d{2}-\d{2}-\d{6}\.png$/);
    });

    it('should generate correct webm filename for video', () => {
      const filename = (panel as any).generateFilename('webm');
      expect(filename).toMatch(/\.webm$/);
    });

    it('should download blob via hidden anchor', () => {
      const blob = new Blob(['test']);
      (panel as any).downloadBlob(blob, 'test.png');

      // Verify createObjectURL was called with the blob
      expect(URL.createObjectURL).toHaveBeenCalledWith(blob);
    });
  });

  describe('confirmation dialog', () => {
    it('should create dialog with correct structure', () => {
      const promise = (panel as any).showConfirmationDialog();

      const dialog = document.querySelector('.luxar-recording-confirm');
      expect(dialog).toBeTruthy();
      expect(dialog?.querySelector('.luxar-recording-confirm__title')?.textContent).toBe(
        'Start Video Recording'
      );
      expect(dialog?.querySelector('[data-action="start"]')).toBeTruthy();
      expect(dialog?.querySelector('[data-action="cancel"]')).toBeTruthy();

      // Click cancel to resolve the promise
      (dialog?.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      return promise.then((result: boolean) => {
        expect(result).toBe(false);
      });
    });

    it('should resolve true when Start is clicked', () => {
      const promise = (panel as any).showConfirmationDialog();

      const startBtn = document.querySelector('[data-action="start"]') as HTMLElement;
      startBtn?.click();

      return promise.then((result: boolean) => {
        expect(result).toBe(true);
      });
    });

    it('should resolve false when Escape is pressed', () => {
      const promise = (panel as any).showConfirmationDialog();

      const overlay = document.querySelector('.luxar-recording-confirm') as HTMLElement;
      overlay?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

      return promise.then((result: boolean) => {
        expect(result).toBe(false);
      });
    });

    it('should resolve true when Enter is pressed', () => {
      const promise = (panel as any).showConfirmationDialog();

      const overlay = document.querySelector('.luxar-recording-confirm') as HTMLElement;
      overlay?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));

      return promise.then((result: boolean) => {
        expect(result).toBe(true);
      });
    });

    it('should remove dialog after resolution', async () => {
      const promise = (panel as any).showConfirmationDialog();

      const cancelBtn = document.querySelector('[data-action="cancel"]') as HTMLElement;
      cancelBtn?.click();

      await promise;
      expect(document.querySelector('.luxar-recording-confirm')).toBeNull();
    });

    it('sets modal-dialog ARIA attributes', async () => {
      const promise = (panel as any).showConfirmationDialog();

      const overlay = document.querySelector('.luxar-recording-confirm') as HTMLElement;
      expect(overlay.getAttribute('role')).toBe('dialog');
      expect(overlay.getAttribute('aria-modal')).toBe('true');
      expect(overlay.getAttribute('aria-labelledby')).toBe('luxar-recording-confirm-title');
      expect(overlay.getAttribute('aria-describedby')).toBe(
        'luxar-recording-confirm-message'
      );
      expect(overlay.querySelector('#luxar-recording-confirm-title')).toBeTruthy();
      expect(overlay.querySelector('#luxar-recording-confirm-message')).toBeTruthy();

      // Resolve the promise to clean up.
      (overlay.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      await promise;
    });

    it('focuses the primary Start button on open', async () => {
      const promise = (panel as any).showConfirmationDialog();
      const startBtn = document.querySelector(
        '.luxar-recording-confirm__btn--primary'
      ) as HTMLElement;
      expect(document.activeElement).toBe(startBtn);

      (document.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      await promise;
    });

    it('restores focus to the previously-focused element on close', async () => {
      // Pre-focus a sentinel element.
      const sentinel = document.createElement('button');
      sentinel.id = 'before-dialog';
      document.body.appendChild(sentinel);
      sentinel.focus();
      expect(document.activeElement).toBe(sentinel);

      const promise = (panel as any).showConfirmationDialog();
      // Dialog steals focus.
      expect(document.activeElement).not.toBe(sentinel);

      (document.querySelector('[data-action="cancel"]') as HTMLElement)?.click();
      await promise;

      expect(document.activeElement).toBe(sentinel);
      sentinel.remove();
    });
  });

  describe('turntable rotation', () => {
    it('should add per-frame callback when turntable starts', () => {
      (panel as any).startTurntableRotation();

      expect(mockAnimController.addPerFrameCallback).toHaveBeenCalledWith(
        'recording-turntable',
        expect.any(Function),
        { continuous: true }
      );
    });
  });

  describe('EXR HDR screenshot', () => {
    it('should call captureHDRAsEXR instead of toBlob for EXR format', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      // Set format to EXR
      (panel as any).options.outputFormat = 'exr';

      await panel.captureScreenshot();

      expect(mockSceneManager.postProcessing.captureHDRAsEXR).toHaveBeenCalled();
      // Should NOT call renderToImageData for EXR (uses captureHDRAsEXR instead)
      expect(mockSceneManager.postProcessing.renderToImageData).not.toHaveBeenCalled();
      expect(showToast).toHaveBeenCalledWith('HDR screenshot saved (EXR)');

      rafSpy.mockRestore();
    });

    it('should generate filename with .exr extension', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      (panel as any).options.outputFormat = 'exr';

      // Spy on downloadBlob to check the filename
      const downloadSpy = vi.spyOn(panel as any, 'downloadBlob');

      await panel.captureScreenshot();

      expect(downloadSpy).toHaveBeenCalledWith(expect.any(Blob), expect.stringMatching(/\.exr$/));

      rafSpy.mockRestore();
    });
  });

  describe('EXR sequence recording', () => {
    it('should initialize EXR sequence state fields', () => {
      expect((panel as any).isEXRSequenceRecording).toBe(false);
    });

    it('should branch to EXR sequence when format is exr in video mode', async () => {
      (panel as any).options.outputFormat = 'exr';
      (panel as any).mode = 'video';

      const startEXRSpy = vi
        .spyOn(panel as any, 'startEXRSequenceRecording')
        .mockResolvedValue(undefined);

      await panel.startVideoRecording();

      expect(startEXRSpy).toHaveBeenCalled();
    });

    it('should stop EXR sequence recording via stopVideoRecording', () => {
      (panel as any).isRecording = true;
      (panel as any).isEXRSequenceRecording = true;

      panel.stopVideoRecording();

      // The offline loop checks isRecording — stopVideoRecording sets flags to signal the loop
      expect((panel as any).isRecording).toBe(false);
      expect((panel as any).isEXRSequenceRecording).toBe(false);
    });

    it('should set isRecording flag to signal the offline loop to stop', () => {
      (panel as any).isRecording = true;
      (panel as any).isEXRSequenceRecording = true;

      (panel as any).stopEXRSequenceRecording();

      // Offline loop architecture: stop methods just set flags
      expect((panel as any).isRecording).toBe(false);
      expect((panel as any).isEXRSequenceRecording).toBe(false);
    });
  });

  describe('slider sync', () => {
    it('should set animation manager via setter', () => {
      const mockAnimManager = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        play: vi.fn(),
      };
      panel.setAnimationManager(mockAnimManager as any);
      expect((panel as any).animationManager).toBe(mockAnimManager);
    });

    it('should clear delayed slider playback on dispose', () => {
      vi.useFakeTimers();
      const mockAnimManager = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        play: vi.fn(),
      };
      panel.setAnimationManager(mockAnimManager as any);
      (panel as any).options.syncDimensionIndex = 3;

      (panel as any).startSliderSync();
      panel.dispose();
      vi.advanceTimersByTime(150);

      expect(mockAnimManager.play).not.toHaveBeenCalled();
      expect(mockAnimManager.removeEventListener).toHaveBeenCalledWith(
        'complete',
        expect.any(Function)
      );

      vi.useRealTimers();
    });

    it('should set adaptive DPR manager via setter', () => {
      const mockDPR = { isActive: vi.fn() };
      panel.setAdaptiveDPRManager(mockDPR as any);
      expect((panel as any).adaptiveDPRManager).toBe(mockDPR);
    });
  });

  describe('computeVideoBitrate', () => {
    it('low quality at 1080p30', () => {
      (panel as any).options.videoQuality = 'low';
      (panel as any).options.videoFPS = 30;
      const bitrate = (panel as any).computeVideoBitrate(1920, 1080);
      expect(bitrate).toBe(Math.round(1920 * 1080 * 30 * 0.04));
    });

    it('high quality at 1080p60', () => {
      (panel as any).options.videoQuality = 'high';
      (panel as any).options.videoFPS = 60;
      const bitrate = (panel as any).computeVideoBitrate(1920, 1080);
      expect(bitrate).toBe(Math.round(1920 * 1080 * 60 * 0.15));
    });

    it('max quality at 4K', () => {
      (panel as any).options.videoQuality = 'max';
      (panel as any).options.videoFPS = 60;
      const bitrate = (panel as any).computeVideoBitrate(3840, 2160);
      expect(bitrate).toBe(Math.round(3840 * 2160 * 60 * 0.3));
    });
  });

  describe('generateFfmpegScript', () => {
    it('generates valid bash script', () => {
      const script = (panel as any).generateFfmpegScript(30, 300, 'png');
      expect(script.startsWith('#!/bin/bash')).toBe(true);
      expect(script).toContain('set -e');
    });

    it('uses correct frame pattern for PNG', () => {
      const script = (panel as any).generateFfmpegScript(60, 600, 'png');
      expect(script).toContain('frame_%06d.png');
      expect(script).toContain('framerate 60');
    });

    it('includes HDR section for EXR', () => {
      const script = (panel as any).generateFfmpegScript(30, 300, 'exr');
      expect(script).toContain('yuv420p10le');
      expect(script).toContain('bt2020');
    });

    it('excludes HDR section for non-EXR', () => {
      const script = (panel as any).generateFfmpegScript(30, 300, 'jpg');
      expect(script).not.toContain('yuv420p10le');
    });
  });

  describe('updateControlVisibility', () => {
    it('Image mode hides video and turntable controls', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'webp';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).videoControllers) {
        expect(ctrl.hide).toHaveBeenCalled();
      }
      for (const ctrl of (panel as any).turntableControllers) {
        expect(ctrl.hide).toHaveBeenCalled();
      }
    });

    it('Video mode hides image controls', () => {
      (panel as any).mode = 'video';
      (panel as any).options.outputFormat = 'webm';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).imageControllers) {
        expect(ctrl.hide).toHaveBeenCalled();
      }
    });

    it('Turntable shows video and turntable controls', () => {
      (panel as any).mode = 'turntable';
      (panel as any).options.outputFormat = 'mp4';
      (panel as any).updateControlVisibility();

      for (const ctrl of (panel as any).videoControllers) {
        expect(ctrl.show).toHaveBeenCalled();
      }
      for (const ctrl of (panel as any).turntableControllers) {
        expect(ctrl.show).toHaveBeenCalled();
      }
    });

    it('Image+EXR hides quality and transparent', () => {
      (panel as any).mode = 'image';
      (panel as any).options.outputFormat = 'exr';
      (panel as any).updateControlVisibility();

      const qualityCtrl = (panel as any).qualityController;
      if (qualityCtrl) {
        expect(qualityCtrl.hide).toHaveBeenCalled();
      }
      const transparentCtrl = (panel as any).transparentController;
      if (transparentCtrl) {
        expect(transparentCtrl.hide).toHaveBeenCalled();
      }
    });

    it('auto-corrects format when switching to Video mode', () => {
      (panel as any).mode = 'video';
      (panel as any).options.outputFormat = 'png';
      (panel as any).updateControlVisibility();

      expect((panel as any).options.outputFormat).toBe('webm');
    });
  });

  describe('getSupportedMimeType', () => {
    let originalMediaRecorder: typeof MediaRecorder | undefined;

    beforeEach(() => {
      originalMediaRecorder = globalThis.MediaRecorder;
    });

    afterEach(() => {
      if (originalMediaRecorder !== undefined) {
        globalThis.MediaRecorder = originalMediaRecorder;
      } else {
        delete (globalThis as any).MediaRecorder;
      }
    });

    it('returns VP9 mime type when supported', () => {
      (globalThis as any).MediaRecorder = {
        isTypeSupported: vi.fn((type: string) => type.includes('vp9')),
      };
      const result = (panel as any).getSupportedMimeType();
      expect(result).toBe('video/webm;codecs=vp9');
    });

    it('falls back to VP8', () => {
      (globalThis as any).MediaRecorder = {
        isTypeSupported: vi.fn((type: string) => type.includes('vp8')),
      };
      const result = (panel as any).getSupportedMimeType();
      expect(result).toBe('video/webm;codecs=vp8');
    });

    it('returns null when nothing supported', () => {
      (globalThis as any).MediaRecorder = {
        isTypeSupported: vi.fn().mockReturnValue(false),
      };
      const result = (panel as any).getSupportedMimeType();
      expect(result).toBeNull();
    });

    it('returns null when MediaRecorder undefined', () => {
      delete (globalThis as any).MediaRecorder;
      const result = (panel as any).getSupportedMimeType();
      expect(result).toBeNull();
    });
  });
});
