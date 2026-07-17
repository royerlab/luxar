/**
 * Unit tests for Scale Bar component
 *
 * Tests the pure math functions (computeNiceValue, formatScaleValue)
 * and the ScaleBar component rendering and update behavior.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';

// Mock ThemeManager (required by UIComponent base class) — hoisted before imports
vi.mock('../../../themes/theme-manager', () => ({
  ThemeManager: {
    getInstance: () => ({
      onChange: vi.fn(() => vi.fn()),
    }),
  },
}));

// Mock sceneDimsManager
vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDimensionUnits: vi.fn(() => ['s', 'um', 'um', 'um']),
    getDims: vi.fn(() => ({
      ndim: 4,
      displayed: [1, 2, 3],
      currentStep: [0, 0, 0, 0],
      metadata: [
        { name: 'T', unit: 's' },
        { name: 'Z', unit: 'um' },
        { name: 'Y', unit: 'um' },
        { name: 'X', unit: 'um' },
      ],
    })),
  },
}));

// Import after vi.mock declarations (mocks are hoisted)
import { computeNiceValue, formatScaleValue, ScaleBar } from '../../../ui/scale-bar';

describe('Scale Bar', () => {
  describe('computeNiceValue', () => {
    it('should return 1 for values close to 1', () => {
      expect(computeNiceValue(1)).toBe(1);
      expect(computeNiceValue(1.2)).toBe(1);
      expect(computeNiceValue(1.4)).toBe(1);
    });

    it('should return 2 for values between 1.5 and 3.5', () => {
      expect(computeNiceValue(1.5)).toBe(2);
      expect(computeNiceValue(2)).toBe(2);
      expect(computeNiceValue(3)).toBe(2);
      expect(computeNiceValue(3.4)).toBe(2);
    });

    it('should return 5 for values between 3.5 and 7.5', () => {
      expect(computeNiceValue(3.5)).toBe(5);
      expect(computeNiceValue(5)).toBe(5);
      expect(computeNiceValue(7)).toBe(5);
      expect(computeNiceValue(7.4)).toBe(5);
    });

    it('should return 10 for values between 7.5 and 10', () => {
      expect(computeNiceValue(7.5)).toBe(10);
      expect(computeNiceValue(9)).toBe(10);
      expect(computeNiceValue(10)).toBe(10);
    });

    it('should handle orders of magnitude correctly', () => {
      expect(computeNiceValue(15)).toBe(20);
      expect(computeNiceValue(42)).toBe(50);
      expect(computeNiceValue(80)).toBe(100);
      expect(computeNiceValue(120)).toBe(100);
      expect(computeNiceValue(300)).toBe(200);
      expect(computeNiceValue(500)).toBe(500);
      expect(computeNiceValue(0.3)).toBe(0.2);
      expect(computeNiceValue(0.5)).toBe(0.5);
      expect(computeNiceValue(0.8)).toBe(1);
    });

    it('should handle very small values', () => {
      expect(computeNiceValue(0.001)).toBe(0.001);
      expect(computeNiceValue(0.003)).toBe(0.002);
      expect(computeNiceValue(0.005)).toBe(0.005);
      expect(computeNiceValue(0.008)).toBe(0.01);
    });

    it('should handle very large values', () => {
      expect(computeNiceValue(1500)).toBe(2000);
      expect(computeNiceValue(50000)).toBe(50000);
      expect(computeNiceValue(120000)).toBe(100000);
    });

    it('should return 1 for zero or negative values', () => {
      expect(computeNiceValue(0)).toBe(1);
      expect(computeNiceValue(-5)).toBe(1);
    });

    it('should always return a value from {1, 2, 5} × 10^n', () => {
      const testValues = [0.003, 0.07, 0.4, 1.8, 6, 25, 90, 350, 7500];
      for (const v of testValues) {
        const nice = computeNiceValue(v);
        const mag = Math.pow(10, Math.floor(Math.log10(nice)));
        const leading = Math.round(nice / mag);
        expect([1, 2, 5, 10]).toContain(leading);
      }
    });
  });

  describe('formatScaleValue', () => {
    it('should format integers without decimals', () => {
      expect(formatScaleValue(1)).toBe('1');
      expect(formatScaleValue(2)).toBe('2');
      expect(formatScaleValue(5)).toBe('5');
      expect(formatScaleValue(10)).toBe('10');
      expect(formatScaleValue(100)).toBe('100');
      expect(formatScaleValue(5000)).toBe('5000');
    });

    it('should format sub-1 values without trailing zeros', () => {
      expect(formatScaleValue(0.5)).toBe('0.5');
      expect(formatScaleValue(0.2)).toBe('0.2');
      expect(formatScaleValue(0.1)).toBe('0.1');
      expect(formatScaleValue(0.05)).toBe('0.05');
      expect(formatScaleValue(0.02)).toBe('0.02');
      expect(formatScaleValue(0.01)).toBe('0.01');
      expect(formatScaleValue(0.005)).toBe('0.005');
      expect(formatScaleValue(0.001)).toBe('0.001');
    });

    it('should format non-integer values >= 1 cleanly', () => {
      expect(formatScaleValue(1.5)).toBe('1.5');
      expect(formatScaleValue(2.5)).toBe('2.5');
    });
  });

  describe('computeNiceValue + formatScaleValue integration', () => {
    it('should produce clean labels for typical microscopy scales', () => {
      const rawValues = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500];
      for (const raw of rawValues) {
        const nice = computeNiceValue(raw);
        const label = formatScaleValue(nice);
        if (label.includes('.')) {
          expect(label).not.toMatch(/0$/);
        }
        expect(label.length).toBeGreaterThan(0);
      }
    });
  });

  describe('ScaleBar component', () => {
    beforeEach(() => {
      document.body.innerHTML = '';
    });

    afterEach(() => {
      document.body.innerHTML = '';
    });

    function createMockConfig(): import('../../../ui/scale-bar').ScaleBarConfig & {
      camera: THREE.PerspectiveCamera;
    } {
      const camera = new THREE.PerspectiveCamera(47, 1, 0.1, 1000);
      camera.position.set(0, 0, 10);
      camera.position.distanceTo = vi.fn(() => 10);
      return {
        // getCamera is a live accessor (the app swaps cameras on
        // perspective ↔ ortho); `camera` is kept alongside so tests can
        // mutate the instance the accessor returns.
        camera,
        getCamera: () => camera as any,
        controls: {
          getFocusTarget: vi.fn(() => ({ x: 0, y: 0, z: 0 })),
        } as any,
        canvas: {
          clientHeight: 800,
        } as any,
        targetWidthPx: 150,
        position: 'bottom-left',
      };
    }

    it('should create DOM structure with correct class names', () => {
      const scaleBar = new ScaleBar(createMockConfig());
      const el = scaleBar.getElement();

      expect(el.className).toBe('luxar-scale-bar');
      expect(el.querySelector('.luxar-scale-bar__bar')).not.toBeNull();
      expect(el.querySelector('.luxar-scale-bar__label')).not.toBeNull();
    });

    it('should add bottom-right class when configured', () => {
      const config = createMockConfig();
      config.position = 'bottom-right';
      const scaleBar = new ScaleBar(config);

      expect(scaleBar.getElement().classList.contains('luxar-scale-bar--bottom-right')).toBe(true);
    });

    it('should not update DOM when not visible', () => {
      // W8 strengthening (P2): also assert the bar element wasn't sized,
      // since "no update" must mean neither label nor bar mutated.
      const scaleBar = new ScaleBar(createMockConfig());
      scaleBar.update();

      const el = scaleBar.getElement();
      const label = el.querySelector('.luxar-scale-bar__label');
      const bar = el.querySelector('.luxar-scale-bar__bar') as HTMLElement;
      expect(label?.textContent).toBe('');
      expect(bar.style.width).toBe('');
      expect(scaleBar.isVisible()).toBe(false);
    });

    it('should update bar width and label when visible', () => {
      const config = createMockConfig();
      const scaleBar = new ScaleBar(config);
      scaleBar.show();
      scaleBar.update();

      const bar = scaleBar.getElement().querySelector('.luxar-scale-bar__bar') as HTMLElement;
      const label = scaleBar.getElement().querySelector('.luxar-scale-bar__label');

      // Bar should have a pixel width set
      expect(bar.style.width).toMatch(/^\d+px$/);
      // Label should contain the unit from the first displayed dimension (index 1 = 'um')
      expect(label?.textContent).toContain('um');
    });

    it('tracks a runtime camera swap (perspective → ortho) via getCamera', () => {
      // Regression: ScaleBar used to capture the camera instance at
      // construction; after the app's perspective ↔ ortho swap it kept
      // computing from the abandoned camera, freezing the label in
      // ortho mode. getCamera must be re-read every update.
      let current: any = new THREE.PerspectiveCamera(47, 1, 0.1, 1000);
      current.position.set(0, 0, 10);
      current.position.distanceTo = vi.fn(() => 10);
      const config = { ...createMockConfig(), getCamera: () => current };
      const scaleBar = new ScaleBar(config);
      scaleBar.show();
      scaleBar.update();
      const bar = scaleBar.getElement().querySelector('.luxar-scale-bar__bar') as HTMLElement;
      const widthBefore = bar.style.width;

      // Swap to an ortho camera with a very different world-per-pixel.
      const ortho = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 1000);
      ortho.zoom = 100;
      ortho.position.set(0, 0, 10);
      ortho.position.distanceTo = vi.fn(() => 10);
      current = ortho;
      scaleBar.update();

      expect(bar.style.width).not.toBe(widthBefore);
    });

    it('should handle zero canvas height gracefully', () => {
      const config = createMockConfig();
      (config.canvas as any).clientHeight = 0;
      const scaleBar = new ScaleBar(config);
      scaleBar.show();
      expect(() => scaleBar.update()).not.toThrow();
    });

    it('should handle zero camera distance gracefully', () => {
      const config = createMockConfig();
      config.camera.position.distanceTo = vi.fn(() => 0);
      const scaleBar = new ScaleBar(config);
      scaleBar.show();
      expect(() => scaleBar.update()).not.toThrow();
    });

    it('should toggle visibility', () => {
      const scaleBar = new ScaleBar(createMockConfig());
      expect(scaleBar.isVisible()).toBe(false);

      scaleBar.toggle();
      expect(scaleBar.isVisible()).toBe(true);

      scaleBar.toggle();
      expect(scaleBar.isVisible()).toBe(false);
    });

    it('should clean up on dispose', () => {
      const scaleBar = new ScaleBar(createMockConfig());
      scaleBar.show();
      expect(scaleBar.getElement().parentNode).toBe(document.body);

      scaleBar.dispose();
      expect(document.querySelector('.luxar-scale-bar')).toBeNull();
    });
  });
});
