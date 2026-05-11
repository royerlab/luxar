/**
 * Unit tests for Dimension Sliders
 * Tests critical fix: memory leaks from slider/dropdown event listeners
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DimensionSliders } from '../../../ui/panels/dimension-sliders';
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
      { name: 'X', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Y', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Z', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Time', unit: 's', scale: 1.0, discrete: false, step: 0.1 },
      {
        name: 'Channel',
        unit: '',
        scale: 1.0,
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

      // Should remove: change, keydown
      // Note: mouseenter, mouseleave, focus, blur are now handled by CSS :hover and :focus
      const eventTypes = removeEventSpy.mock.calls.map((call) => call[0]);
      expect(eventTypes).toContain('change');
      expect(eventTypes).toContain('keydown');
      // Hover/focus handlers no longer needed - CSS handles these states
      expect(eventTypes).not.toContain('mouseenter');
      expect(eventTypes).not.toContain('mouseleave');
      expect(eventTypes).not.toContain('focus');
      expect(eventTypes).not.toContain('blur');
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

      // Get initial sliders
      const initialSliders = document.querySelectorAll('input[type="range"]');

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
});

describe('DimensionSliders - Binary Toggle Controls', () => {
  const createBinaryDims = (): SimpleDims => ({
    ndim: 6,
    displayed: [0, 1, 2],
    currentStep: [0, 0, 0, 0, 1, 2],
    metadata: [
      { name: 'X', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Y', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      { name: 'Z', unit: 'μm', scale: 1.0, discrete: false, step: 0.1 },
      {
        name: 'DAPI',
        unit: '',
        scale: 1.0,
        discrete: true,
        step: 1,
        categories: ['Off', 'On'],
      },
      {
        name: 'GFP',
        unit: '',
        scale: 1.0,
        discrete: true,
        step: 1,
        categories: ['Off', 'On'],
      },
      {
        name: 'Channel',
        unit: '',
        scale: 1.0,
        discrete: true,
        step: 1,
        categories: ['DAPI', 'GFP', 'RFP'],
      },
    ],
  });

  it('should render binary categorical dimension as toggle, not dropdown', () => {
    const container = document.getElementById('test-container')!;
    const sliders = new DimensionSliders({
      container,
      dims: createBinaryDims(),
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
        [0, 1],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'DAPI', 'GFP', 'Channel'],
    });

    // Binary categoricals should be toggles
    const toggles = document.querySelectorAll('.luxar-dimension-toggle');
    expect(toggles.length).toBe(2);

    // 3-category dimension should remain a dropdown
    const dropdowns = document.querySelectorAll('select');
    expect(dropdowns.length).toBe(1);

    sliders.dispose();
  });

  it('should show current value label and correct visual state', () => {
    const container = document.getElementById('test-container')!;
    const dims = createBinaryDims();
    // DAPI=0 (Off), GFP=1 (On)

    const sliders = new DimensionSliders({
      container,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
        [0, 1],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'DAPI', 'GFP', 'Channel'],
    });

    // First toggle (DAPI, currentStep=0 → shows "Off", no --on class)
    const firstToggle = document.getElementById('luxar-dim-toggle-3')!;
    expect(firstToggle.textContent).toBe('Off');
    expect(firstToggle.classList.contains('luxar-dimension-toggle--on')).toBe(false);

    // Second toggle (GFP, currentStep=1 → shows "On", has --on class)
    const secondToggle = document.getElementById('luxar-dim-toggle-4')!;
    expect(secondToggle.textContent).toBe('On');
    expect(secondToggle.classList.contains('luxar-dimension-toggle--on')).toBe(true);

    sliders.dispose();
  });

  it('should render discrete non-categorical binary dim as toggle', () => {
    const container = document.getElementById('test-container')!;
    const dims: SimpleDims = {
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 0],
      metadata: [
        { name: 'X', unit: '', scale: 1.0 },
        { name: 'Y', unit: '', scale: 1.0 },
        { name: 'Z', unit: '', scale: 1.0 },
        { name: 'Flag', unit: '', scale: 1.0, discrete: true, step: 1 },
      ],
    };

    const sliders = new DimensionSliders({
      container,
      dims,
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'Flag'],
    });

    const toggle = document.querySelector('.luxar-dimension-toggle');
    expect(toggle).toBeTruthy();

    // Non-categorical: shows current numeric label (value=0 → "0")
    expect(toggle!.textContent).toBe('0');

    // No dropdowns should exist
    const dropdowns = document.querySelectorAll('select');
    expect(dropdowns.length).toBe(0);

    sliders.dispose();
  });

  it('should properly remove toggle event listeners on dispose', () => {
    const container = document.getElementById('test-container')!;
    const sliders = new DimensionSliders({
      container,
      dims: createBinaryDims(),
      dimensionRanges: [
        [0, 100],
        [0, 100],
        [0, 100],
        [0, 1],
        [0, 1],
        [0, 2],
      ],
      dimensionNames: ['X', 'Y', 'Z', 'DAPI', 'GFP', 'Channel'],
    });

    const toggleElements = document.querySelectorAll('.luxar-dimension-toggle');
    expect(toggleElements.length).toBe(2);

    // Track event listener removals
    const removeEventSpy = vi.fn();
    toggleElements.forEach((toggle) => {
      const original = toggle.removeEventListener.bind(toggle);
      toggle.removeEventListener = (type: string, listener: any, options?: any) => {
        removeEventSpy(type);
        original(type, listener, options);
      };
    });

    sliders.dispose();

    const eventTypes = removeEventSpy.mock.calls.map((call) => call[0]);
    expect(eventTypes).toContain('click');
    expect(eventTypes).toContain('keydown');
  });
});
