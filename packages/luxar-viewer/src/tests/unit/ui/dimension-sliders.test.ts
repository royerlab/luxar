/**
 * Unit tests for Dimension Sliders
 * Tests critical fix: memory leaks from slider/dropdown event listeners
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DimensionSliders } from '../../../ui/dimension-sliders';
import type { SimpleDims } from '../../../types/dims';

// Mock scene dims manager
vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    setDimensionValue: vi.fn(),
    getDims: vi.fn(),
    getDimensionRanges: vi.fn(),
    getDimensionNames: vi.fn(),
  },
}));

beforeEach(() => {
  document.body.innerHTML = '<div id="test-container"></div>';
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('DimensionSliders - Memory Leak Prevention', () => {
  const createMockDims = (): SimpleDims => ({
    ndim: 5,
    displayed: [0, 1, 2], // X, Y, Z
    currentStep: [0, 0, 0, 5.5, 2], // Time at 5.5, Channel at 2
    metadata: [
      { name: 'X', discrete: false, step: 0.1 },
      { name: 'Y', discrete: false, step: 0.1 },
      { name: 'Z', discrete: false, step: 0.1 },
      { name: 'Time', discrete: false, step: 0.1, unit: 's' },
      {
        name: 'Channel',
        discrete: true,
        step: 1,
        categories: ['DAPI', 'GFP', 'RFP'],
      },
    ],
  });

  describe('Slider Event Listener Cleanup', () => {
    it('should properly remove all slider event listeners on dispose', () => {
      const container = document.getElementById('test-container')!;
      const dims = createMockDims();

      const sliders = new DimensionSliders({
        container,
        dims,
        dimensionRanges: [
          [0, 100],
          [0, 100],
          [0, 100],
          [0, 10],
          [0, 2],
        ],
        dimensionNames: ['X', 'Y', 'Z', 'Time', 'Channel'],
        dimensionUnits: ['μm', 'μm', 'μm', 's', ''],
      });

      // Get the created sliders
      const sliderElements = document.querySelectorAll('input[type="range"]');
      expect(sliderElements.length).toBeGreaterThan(0);

      // Track event listeners
      const removeEventSpy = vi.fn();
      sliderElements.forEach((slider) => {
        const original = slider.removeEventListener.bind(slider);
        slider.removeEventListener = (type: string, listener: any, options?: any) => {
          removeEventSpy(type);
          original(type, listener, options);
        };
      });

      // Dispose
      sliders.dispose();

      // Should have removed 'input' and 'keydown' for each slider
      const inputRemoved = removeEventSpy.mock.calls.filter((call) => call[0] === 'input').length;
      const keydownRemoved = removeEventSpy.mock.calls.filter(
        (call) => call[0] === 'keydown'
      ).length;

      expect(inputRemoved).toBeGreaterThan(0);
      expect(keydownRemoved).toBeGreaterThan(0);
    });

    it('should properly remove all dropdown event listeners on dispose', () => {
      const container = document.getElementById('test-container')!;
      const dims = createMockDims();

      const sliders = new DimensionSliders({
        container,
        dims,
        dimensionRanges: [
          [0, 100],
          [0, 100],
          [0, 100],
          [0, 10],
          [0, 2],
        ],
        dimensionNames: ['X', 'Y', 'Z', 'Time', 'Channel'],
      });

      // Get the created dropdown (Channel dimension has < 10 categories)
      const dropdown = document.querySelector('select') as HTMLSelectElement;
      expect(dropdown).toBeTruthy();

      // Track removals
      const removeEventSpy = vi.fn();
      const original = dropdown.removeEventListener.bind(dropdown);
      dropdown.removeEventListener = (type: string, listener: any, options?: any) => {
        removeEventSpy(type);
        original(type, listener, options);
      };

      // Dispose
      sliders.dispose();

      // Should remove: change, keydown, mouseenter, mouseleave, focus, blur
      const eventTypes = removeEventSpy.mock.calls.map((call) => call[0]);
      expect(eventTypes).toContain('change');
      expect(eventTypes).toContain('keydown');
      expect(eventTypes).toContain('mouseenter');
      expect(eventTypes).toContain('mouseleave');
      expect(eventTypes).toContain('focus');
      expect(eventTypes).toContain('blur');
    });

    it('should clean up event listeners when rebuilding sliders (createSliders)', () => {
      const container = document.getElementById('test-container')!;
      const dims = createMockDims();

      const sliders = new DimensionSliders({
        container,
        dims,
        dimensionRanges: [
          [0, 100],
          [0, 100],
          [0, 100],
          [0, 10],
          [0, 2],
        ],
        dimensionNames: ['X', 'Y', 'Z', 'Time', 'Channel'],
      });

      // Get initial slider count
      const initialSliders = document.querySelectorAll('input[type="range"]');
      const initialCount = initialSliders.length;

      // Access private createSliders method to test cleanup
      const sliders_any = sliders as any;

      // Track removals
      let removeCount = 0;
      initialSliders.forEach((slider) => {
        const original = slider.removeEventListener.bind(slider);
        slider.removeEventListener = (type: string, listener: any, options?: any) => {
          removeCount++;
          original(type, listener, options);
        };
      });

      // Rebuild sliders (simulates dimension change)
      sliders_any.createSliders();

      // Should have removed listeners from old sliders
      expect(removeCount).toBeGreaterThan(0);

      // Clean up
      sliders.dispose();
    });
  });

  describe('DOM Element Lifecycle', () => {
    it('should remove all DOM elements on dispose', () => {
      const debugConsole = new DebugConsole();

      expect(document.querySelector('.debug-console-panel')).toBeTruthy();
      expect(document.getElementById('debug-console-styles')).toBeTruthy();

      debugConsole.dispose();

      expect(document.querySelector('.debug-console-panel')).toBeNull();
      expect(document.getElementById('debug-console-styles')).toBeNull();
    });
  });

  describe('XSS Protection', () => {
    it('should render arguments as DOM elements, not innerHTML', () => {
      const debugConsole = new DebugConsole();
      const console_any = debugConsole as any;

      // Test malicious strings
      const malicious = '<img src=x onerror=alert(1)>';
      const element = console_any.formatArgAsDOMElement(malicious);

      expect(element.tagName).toBe('SPAN');
      expect(element.className).toBe('console-message-string');
      expect(element.textContent).toBe(`"${malicious}"`);
      expect(element.querySelector('img')).toBeNull();

      debugConsole.dispose();
    });

    it('should safely handle all argument types', () => {
      const debugConsole = new DebugConsole();
      const console_any = debugConsole as any;

      const testCases = [
        { input: undefined, expectedClass: 'console-message-undefined', expectedText: 'undefined' },
        { input: null, expectedClass: 'console-message-undefined', expectedText: 'null' },
        {
          input: '<b>bold</b>',
          expectedClass: 'console-message-string',
          expectedText: '"<b>bold</b>"',
        },
        { input: 42, expectedClass: 'console-message-number', expectedText: '42' },
        { input: true, expectedClass: 'console-message-boolean', expectedText: 'true' },
        {
          input: { key: '<script>alert(1)</script>' },
          expectedClass: 'console-message-object',
          expectedText: null, // Just check it doesn't execute
        },
      ];

      testCases.forEach(({ input, expectedClass, expectedText }) => {
        const element = console_any.formatArgAsDOMElement(input);
        expect(element.className).toBe(expectedClass);
        if (expectedText) {
          expect(element.textContent).toBe(expectedText);
        }
        // Most important: no script elements should be created
        expect(element.querySelector('script')).toBeNull();
      });

      debugConsole.dispose();
    });
  });
});
