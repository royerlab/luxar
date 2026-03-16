/**
 * Unit tests for RecordingPanel
 *
 * Tests screenshot capture, video recording lifecycle,
 * panel state management, and edge cases.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

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
    };
  }

  function createMockController() {
    return {
      name: vi.fn().mockReturnThis(),
      onChange: vi.fn().mockReturnThis(),
      show: vi.fn().mockReturnThis(),
      hide: vi.fn().mockReturnThis(),
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

// Mock URL.createObjectURL/revokeObjectURL (not available in jsdom)
if (typeof URL.createObjectURL === 'undefined') {
  URL.createObjectURL = vi.fn().mockReturnValue('blob:mock-url');
  URL.revokeObjectURL = vi.fn();
}

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
    },
    postProcessing: {
      render: vi.fn(),
    },
    camera: {
      position: { x: 10, y: 5, z: 10, distanceTo: vi.fn().mockReturnValue(15), clone: vi.fn() },
      lookAt: vi.fn(),
    },
    scene: {
      background: { clone: vi.fn() },
    },
    controls: {
      getControls: vi.fn().mockReturnValue({
        target: { x: 0, y: 0, z: 0, clone: vi.fn().mockReturnValue({ x: 0, y: 0, z: 0 }) },
      }),
    },
    setAdaptivePixelRatio: vi.fn(),
  } as any;
}

// Helper to create mock AnimationController
function createMockAnimationController() {
  return {
    startAnimation: vi.fn(),
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
    it('should call postProcessing.render() and canvas.toBlob()', async () => {
      // Use real requestAnimationFrame
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      await panel.captureScreenshot();

      expect(mockSceneManager.postProcessing.render).toHaveBeenCalled();
      expect(mockSceneManager.renderer.domElement.toBlob).toHaveBeenCalled();
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

      // Should only render once
      expect(mockSceneManager.postProcessing.render).toHaveBeenCalledTimes(1);

      rafSpy.mockRestore();
    });

    it('should handle toBlob returning null with PNG fallback', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      let callCount = 0;
      mockSceneManager.renderer.domElement.toBlob = vi.fn((callback: any) => {
        callCount++;
        if (callCount === 1) {
          callback(null); // First call fails
        } else {
          callback(new Blob(['test'], { type: 'image/png' })); // Fallback succeeds
        }
      });

      await panel.captureScreenshot();

      expect(showToast).toHaveBeenCalledWith('Screenshot saved (PNG fallback)');

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
    it('should not clobber saved panel states when screenshotting during recording', async () => {
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
        cb(0);
        return 0;
      });

      const originalStates = new Map([['renderingControls', true]]);
      const getStates = vi.fn().mockReturnValue(originalStates);
      const restoreStates = vi.fn();
      panel.setPanelStateCallbacks(getStates, restoreStates);

      // Simulate active recording
      (panel as any).isRecording = true;

      // Take screenshot during recording
      await panel.captureScreenshot();

      // savedPanelStates should be preserved (not nulled) because recording is active
      expect((panel as any).savedPanelStates).toEqual(originalStates);

      // restoreAllPanels should NOT have restored (recording still active)
      // Only the hideAllPanels call to hide other panels should have happened
      const restoreCalls = restoreStates.mock.calls;
      // All restore calls should be "hide" calls (all false), no "restore" calls
      for (const call of restoreCalls) {
        const states = call[0] as Map<string, boolean>;
        for (const [, visible] of states) {
          expect(visible).toBe(false);
        }
      }

      rafSpy.mockRestore();
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
      (panel as any).options.imageFormat = 'jpeg';

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
  });

  describe('turntable rotation', () => {
    it('should add per-frame callback when turntable starts', () => {
      (panel as any).startTurntableRotation();

      expect(mockAnimController.addPerFrameCallback).toHaveBeenCalledWith(
        'recording-turntable',
        expect.any(Function)
      );
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

    it('should set adaptive DPR manager via setter', () => {
      const mockDPR = { isActive: vi.fn() };
      panel.setAdaptiveDPRManager(mockDPR as any);
      expect((panel as any).adaptiveDPRManager).toBe(mockDPR);
    });
  });
});
