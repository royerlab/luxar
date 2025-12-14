/**
 * Unit tests for InputHandler - keyboard interaction and dimension navigation
 *
 * Tests verify keyboard shortcuts, dimension selection, navigation, and event cleanup
 * WITHOUT mocking internal modules (following TESTING_GUIDELINES.md).
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { InputHandler } from '../../../input/input-handler';
import type { SceneManager } from '../../../scene/scene-manager';

// Mock only external DOM APIs that aren't available in test environment
const createMockElement = () => ({
  className: '',
  innerHTML: '',
  style: {},
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  appendChild: vi.fn(),
  remove: vi.fn(),
  contains: vi.fn(() => false),
});

const mockDocument = {
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  createElement: vi.fn(() => createMockElement()),
  getElementById: vi.fn(() => null),
  body: {
    appendChild: vi.fn(),
    removeChild: vi.fn(),
  },
  head: {
    appendChild: vi.fn(),
    removeChild: vi.fn(),
  },
};

vi.stubGlobal('document', mockDocument);

describe('InputHandler', () => {
  let inputHandler: InputHandler;
  let mockSceneDimsManager: SceneDimsManager;
  let mockCommandHandler: (command: InputCommand) => void;
  let mockViewState: ViewState;

  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks();

    // Create mock dependencies (minimal mocking - only what InputHandler needs)
    mockViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0, 0],
      ndim: 5,
    } as ViewState;

    mockSceneDimsManager = {
      getViewState: () => mockViewState,
      updateSlicePosition: vi.fn(),
      setDisplayDims: vi.fn(),
      getDimensionInfo: vi.fn((dim: number) => ({
        name: `dim${dim}`,
        range: [0, 10],
        discrete: false,
        categories: null,
      })),
    } as unknown as SceneDimsManager;

    mockCommandHandler = vi.fn();

    // Create InputHandler with real implementation
    inputHandler = new InputHandler(mockSceneDimsManager, mockCommandHandler);
  });

  afterEach(() => {
    if (inputHandler) {
      inputHandler.dispose();
    }
  });

  describe('initialization', () => {
    it('should register keyboard event listeners on init', () => {
      inputHandler.init();

      expect(mockDocument.addEventListener).toHaveBeenCalledWith('keydown', expect.any(Function));
      expect(mockDocument.addEventListener).toHaveBeenCalledWith('keyup', expect.any(Function));
    });

    it('should not register listeners before init', () => {
      expect(mockDocument.addEventListener).not.toHaveBeenCalled();
    });

    it('should allow multiple init calls safely', () => {
      inputHandler.init();
      inputHandler.init();

      // Should only register once
      expect(mockDocument.addEventListener).toHaveBeenCalledTimes(2);
    });
  });

  describe('cleanup and disposal', () => {
    it('should remove event listeners on dispose', () => {
      inputHandler.init();
      const keydownHandler = mockDocument.addEventListener.mock.calls[0][1];
      const keyupHandler = mockDocument.addEventListener.mock.calls[1][1];

      inputHandler.dispose();

      expect(mockDocument.removeEventListener).toHaveBeenCalledWith('keydown', keydownHandler);
      expect(mockDocument.removeEventListener).toHaveBeenCalledWith('keyup', keyupHandler);
    });

    it('should handle dispose without init gracefully', () => {
      expect(() => inputHandler.dispose()).not.toThrow();
    });

    it('should handle multiple dispose calls gracefully', () => {
      inputHandler.init();
      inputHandler.dispose();
      expect(() => inputHandler.dispose()).not.toThrow();
    });
  });

  describe('dimension selection (1-9 keys)', () => {
    beforeEach(() => {
      inputHandler.init();
    });

    it('should select dimension 0 when pressing "1"', () => {
      const event = new KeyboardEvent('keydown', { key: '1' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'selectDimension',
          dimension: 0,
        })
      );
    });

    it('should select dimension 4 when pressing "5"', () => {
      const event = new KeyboardEvent('keydown', { key: '5' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'selectDimension',
          dimension: 4,
        })
      );
    });

    it('should select dimension 8 when pressing "9"', () => {
      const event = new KeyboardEvent('keydown', { key: '9' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'selectDimension',
          dimension: 8,
        })
      );
    });

    it('should not select dimension for "0" key', () => {
      const event = new KeyboardEvent('keydown', { key: '0' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).not.toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'selectDimension',
        })
      );
    });
  });

  describe('dimension navigation ([ ] keys)', () => {
    beforeEach(() => {
      inputHandler.init();
      // Simulate dimension 3 being selected
      const selectEvent = new KeyboardEvent('keydown', { key: '4' });
      mockDocument.addEventListener.mock.calls[0][1](selectEvent);
    });

    it('should navigate backward with "[" key', () => {
      const event = new KeyboardEvent('keydown', { key: '[' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockSceneDimsManager.updateSlicePosition).toHaveBeenCalled();
      const callArgs = (mockSceneDimsManager.updateSlicePosition as any).mock.calls[0];
      expect(callArgs[0]).toBe(3); // dimension
      expect(callArgs[1]).toBeLessThan(0); // negative delta
    });

    it('should navigate forward with "]" key', () => {
      const event = new KeyboardEvent('keydown', { key: ']' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockSceneDimsManager.updateSlicePosition).toHaveBeenCalled();
      const callArgs = (mockSceneDimsManager.updateSlicePosition as any).mock.calls[0];
      expect(callArgs[0]).toBe(3); // dimension
      expect(callArgs[1]).toBeGreaterThan(0); // positive delta
    });

    it('should navigate with Shift+[ for larger steps', () => {
      const event = new KeyboardEvent('keydown', { key: '[', shiftKey: true });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockSceneDimsManager.updateSlicePosition).toHaveBeenCalled();
      const callArgs = (mockSceneDimsManager.updateSlicePosition as any).mock.calls[0];
      // With shift, step should be larger (multiplied by 10)
      expect(Math.abs(callArgs[1])).toBeGreaterThan(1);
    });
  });

  describe('help overlay (H key)', () => {
    beforeEach(() => {
      inputHandler.init();
    });

    it('should toggle help overlay when pressing "h"', () => {
      const event = new KeyboardEvent('keydown', { key: 'h' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'toggleHelp',
        })
      );
    });

    it('should toggle help overlay when pressing "H" (uppercase)', () => {
      const event = new KeyboardEvent('keydown', { key: 'H' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'toggleHelp',
        })
      );
    });
  });

  describe('panel toggles', () => {
    beforeEach(() => {
      inputHandler.init();
    });

    it('should toggle performance panel when pressing "p"', () => {
      const event = new KeyboardEvent('keydown', { key: 'p' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'togglePerformance',
        })
      );
    });

    it('should toggle rendering controls when pressing "m"', () => {
      const event = new KeyboardEvent('keydown', { key: 'm' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'toggleRenderingControls',
        })
      );
    });

    it('should toggle dataset browser when pressing "r"', () => {
      const event = new KeyboardEvent('keydown', { key: 'r' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'toggleDatasetBrowser',
        })
      );
    });
  });

  describe('fullscreen toggle', () => {
    beforeEach(() => {
      inputHandler.init();
    });

    it('should toggle fullscreen when pressing Space', () => {
      const event = new KeyboardEvent('keydown', { key: ' ' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'toggleFullscreen',
        })
      );
    });

    it('should prevent default Space behavior', () => {
      const event = new KeyboardEvent('keydown', { key: ' ' });
      const preventDefaultSpy = vi.spyOn(event, 'preventDefault');
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });
  });

  describe('input context awareness', () => {
    beforeEach(() => {
      inputHandler.init();
    });

    it('should not process keys when input element is focused', () => {
      const mockInput = document.createElement('input');
      const event = new KeyboardEvent('keydown', {
        key: 'h',
        target: mockInput as any,
      });

      mockDocument.addEventListener.mock.calls[0][1](event);

      // Should not trigger help toggle
      expect(mockCommandHandler).not.toHaveBeenCalled();
    });

    it('should not process keys when textarea is focused', () => {
      const mockTextarea = document.createElement('textarea');
      const event = new KeyboardEvent('keydown', {
        key: 'h',
        target: mockTextarea as any,
      });

      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).not.toHaveBeenCalled();
    });

    it('should process keys when canvas is focused', () => {
      const mockCanvas = document.createElement('canvas');
      const event = new KeyboardEvent('keydown', {
        key: 'h',
        target: mockCanvas as any,
      });

      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalled();
    });
  });

  describe('modifier key handling', () => {
    beforeEach(() => {
      inputHandler.init();
    });

    it('should detect Shift key press', () => {
      const event = new KeyboardEvent('keydown', { key: 'Shift' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      // Shift key press should be recorded for next navigation
      const navEvent = new KeyboardEvent('keydown', { key: ']', shiftKey: true });
      mockDocument.addEventListener.mock.calls[0][1](navEvent);

      expect(mockSceneDimsManager.updateSlicePosition).toHaveBeenCalled();
    });

    it('should detect Ctrl key combinations', () => {
      const event = new KeyboardEvent('keydown', { key: 'l', ctrlKey: true });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'clearConsole',
        })
      );
    });
  });

  describe('edge cases', () => {
    beforeEach(() => {
      inputHandler.init();
    });

    it('should handle undefined key gracefully', () => {
      const event = new KeyboardEvent('keydown', { key: undefined as any });
      expect(() => mockDocument.addEventListener.mock.calls[0][1](event)).not.toThrow();
    });

    it('should handle numeric keys beyond 9', () => {
      // Keys like F1-F12 have numeric codes but shouldn't select dimensions
      const event = new KeyboardEvent('keydown', { key: 'F1' });
      mockDocument.addEventListener.mock.calls[0][1](event);

      expect(mockCommandHandler).not.toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'selectDimension',
        })
      );
    });

    it('should handle rapid key presses without errors', () => {
      for (let i = 0; i < 100; i++) {
        const event = new KeyboardEvent('keydown', { key: 'h' });
        expect(() => mockDocument.addEventListener.mock.calls[0][1](event)).not.toThrow();
      }
    });
  });

  describe('memory management', () => {
    it('should not leak event listeners after dispose', () => {
      inputHandler.init();
      const initialListenerCount = mockDocument.addEventListener.mock.calls.length;

      inputHandler.dispose();

      // Create new handler
      const newHandler = new InputHandler(mockSceneDimsManager, mockCommandHandler);
      newHandler.init();

      // Should register same number of listeners
      const newListenerCount =
        mockDocument.addEventListener.mock.calls.length - initialListenerCount;
      expect(newListenerCount).toBe(2); // keydown and keyup

      newHandler.dispose();
    });
  });
});
