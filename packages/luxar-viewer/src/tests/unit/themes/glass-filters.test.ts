/**
 * Unit tests for `themes/glass-filters.ts`.
 *
 * Closes themes.md G1 (G1a-G1l): the 333-line glass-filters module had
 * no sibling test file. Each sub-finding is pinned below.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  injectGlassFilters,
  removeGlassFilters,
  injectGlassRefractionLayers,
  removeGlassRefractionLayers,
  setupGlassRefractionObserver,
  defaultGlassParams,
} from '../../../themes/glass-filters';

const GLASS_PANEL_SELECTORS = [
  '.luxar-help-overlay',
  '.luxar-error-dialog',
  '.luxar-dataset-browser',
  '.luxar-debug-console',
  '.luxar-data-monitor',
  '.luxar-dimension-sliders',
  '.luxar-layers-panel',
  '.luxar-gui',
];

describe('glass-filters', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  afterEach(() => {
    // Always clean up the injected SVG so it doesn't bleed across tests.
    removeGlassFilters();
    removeGlassRefractionLayers();
    document.body.innerHTML = '';
  });

  describe('[G1a] injectGlassFilters idempotency', () => {
    it('inserts exactly one #luxar-glass-filters SVG on first call', () => {
      injectGlassFilters();
      expect(document.querySelectorAll('#luxar-glass-filters').length).toBe(1);
    });

    it('is a no-op on the second call (still exactly one SVG)', () => {
      injectGlassFilters();
      injectGlassFilters();
      injectGlassFilters();
      expect(document.querySelectorAll('#luxar-glass-filters').length).toBe(1);
    });
  });

  describe('[G1b] injectGlassFilters parameter interpolation', () => {
    it('interpolates blurRadius into the feGaussianBlur stdDeviation', () => {
      injectGlassFilters({
        blurRadius: 42,
        refractionScale: 1,
        chromaticStrength: 1,
        specularIntensity: 0,
      });
      const svg = document.getElementById('luxar-glass-filters');
      expect(svg?.innerHTML).toContain('stdDeviation="42"');
    });

    it('interpolates refractionScale into the feDisplacementMap scale', () => {
      injectGlassFilters({
        blurRadius: 1,
        refractionScale: 73,
        chromaticStrength: 1,
        specularIntensity: 0,
      });
      const svg = document.getElementById('luxar-glass-filters');
      expect(svg?.innerHTML).toContain('scale="73"');
    });

    it('interpolates chromaticStrength into the feOffset dx values (positive and negative)', () => {
      injectGlassFilters({
        blurRadius: 1,
        refractionScale: 1,
        chromaticStrength: 7,
        specularIntensity: 0,
      });
      const html = document.getElementById('luxar-glass-filters')?.innerHTML ?? '';
      // Red channel shifts right by +chromaticStrength.
      expect(html).toContain('dx="7"');
      // Blue channel shifts left by -chromaticStrength.
      expect(html).toContain('dx="-7"');
    });

    it('interpolates specularIntensity into the specular feColorMatrix values when > 0', () => {
      injectGlassFilters({
        blurRadius: 1,
        refractionScale: 1,
        chromaticStrength: 1,
        specularIntensity: 0.42,
      });
      const html = document.getElementById('luxar-glass-filters')?.innerHTML ?? '';
      // The matrix uses 0.42 in the constant column for all three RGB rows.
      expect(html).toContain('0.42');
    });
  });

  describe('[G1c] injectGlassFilters specularIntensity truthiness', () => {
    it('omits the specular block when specularIntensity is 0', () => {
      injectGlassFilters({
        blurRadius: 1,
        refractionScale: 1,
        chromaticStrength: 1,
        specularIntensity: 0,
      });
      const html = document.getElementById('luxar-glass-filters')?.innerHTML ?? '';
      expect(html).not.toContain('specMap');
      expect(html).not.toContain('result="withSpecular"');
    });

    it('omits the specular block when specularIntensity is undefined', () => {
      injectGlassFilters({
        blurRadius: 1,
        refractionScale: 1,
        chromaticStrength: 1,
        // specularIntensity intentionally omitted.
      });
      const html = document.getElementById('luxar-glass-filters')?.innerHTML ?? '';
      expect(html).not.toContain('specMap');
      expect(html).not.toContain('result="withSpecular"');
    });

    it('includes the specular block when specularIntensity > 0', () => {
      injectGlassFilters({
        blurRadius: 1,
        refractionScale: 1,
        chromaticStrength: 1,
        specularIntensity: 0.5,
      });
      const html = document.getElementById('luxar-glass-filters')?.innerHTML ?? '';
      expect(html).toContain('result="specMap"');
      expect(html).toContain('result="withSpecular"');
    });

    it('uses defaultGlassParams when no params are supplied (includes specular at default 0.5)', () => {
      // defaultGlassParams.specularIntensity = 0.5 > 0, so specular block is on.
      injectGlassFilters();
      const html = document.getElementById('luxar-glass-filters')?.innerHTML ?? '';
      expect(html).toContain('result="specMap"');
      expect(html).toContain(`stdDeviation="${defaultGlassParams.blurRadius}"`);
    });
  });

  describe('[G1d] removeGlassFilters no-op when SVG missing', () => {
    it('does not throw when no #luxar-glass-filters element exists', () => {
      // Audit W15 fix: pin the observable contract — the SVG remains
      // absent after the no-op removeGlassFilters call.
      expect(document.getElementById('luxar-glass-filters')).toBeNull();
      expect(() => removeGlassFilters()).not.toThrow();
      expect(document.getElementById('luxar-glass-filters')).toBeNull();
    });

    it('removes the SVG when present', () => {
      injectGlassFilters();
      expect(document.getElementById('luxar-glass-filters')).not.toBeNull();
      removeGlassFilters();
      expect(document.getElementById('luxar-glass-filters')).toBeNull();
    });
  });

  describe('[G1e] injectGlassRefractionLayers per-selector iteration', () => {
    it('injects a .luxar-glass-refraction child into each of the 8 panel selectors', () => {
      // Create one panel matching each selector.
      for (const selector of GLASS_PANEL_SELECTORS) {
        const className = selector.replace(/^\./, '');
        const panel = document.createElement('div');
        panel.className = className;
        document.body.appendChild(panel);
      }

      injectGlassRefractionLayers();

      // Each panel now has exactly one refraction layer.
      for (const selector of GLASS_PANEL_SELECTORS) {
        const panel = document.querySelector(selector);
        expect(panel?.querySelectorAll('.luxar-glass-refraction').length).toBe(1);
      }
    });
  });

  describe('[G1f] injectGlassRefractionLayers skip-if-exists', () => {
    it('does not add a duplicate refraction layer on the second call', () => {
      const panel = document.createElement('div');
      panel.className = 'luxar-gui';
      document.body.appendChild(panel);

      injectGlassRefractionLayers();
      injectGlassRefractionLayers();
      injectGlassRefractionLayers();

      expect(panel.querySelectorAll('.luxar-glass-refraction').length).toBe(1);
    });
  });

  describe('[G1g] injectGlassRefractionLayers insert position', () => {
    it('inserts the refraction layer as the FIRST child of the panel, not the last', () => {
      const panel = document.createElement('div');
      panel.className = 'luxar-gui';
      // Pre-existing children — refraction must go before them.
      const child1 = document.createElement('span');
      child1.id = 'pre-existing-1';
      const child2 = document.createElement('span');
      child2.id = 'pre-existing-2';
      panel.appendChild(child1);
      panel.appendChild(child2);
      document.body.appendChild(panel);

      injectGlassRefractionLayers();

      expect(panel.firstElementChild?.className).toBe('luxar-glass-refraction');
      // Pre-existing children are still present and now follow the refraction layer.
      expect(panel.children[1]).toBe(child1);
      expect(panel.children[2]).toBe(child2);
    });

    it('sets aria-hidden="true" on the injected refraction layer', () => {
      const panel = document.createElement('div');
      panel.className = 'luxar-gui';
      document.body.appendChild(panel);

      injectGlassRefractionLayers();

      const layer = panel.querySelector('.luxar-glass-refraction');
      expect(layer?.getAttribute('aria-hidden')).toBe('true');
    });
  });

  describe('[G1h] removeGlassRefractionLayers global sweep', () => {
    it('removes EVERY .luxar-glass-refraction element, across all panels', () => {
      // Inject into many panels at once.
      for (const selector of GLASS_PANEL_SELECTORS) {
        const className = selector.replace(/^\./, '');
        const panel = document.createElement('div');
        panel.className = className;
        document.body.appendChild(panel);
      }
      injectGlassRefractionLayers();
      expect(document.querySelectorAll('.luxar-glass-refraction').length).toBe(
        GLASS_PANEL_SELECTORS.length
      );

      removeGlassRefractionLayers();
      expect(document.querySelectorAll('.luxar-glass-refraction').length).toBe(0);
    });

    it('is a no-op when no refraction layers are present', () => {
      // Audit W15 fix: pin the observable contract — layer count
      // remains zero after the no-op call.
      expect(document.querySelectorAll('.luxar-glass-refraction').length).toBe(0);
      expect(() => removeGlassRefractionLayers()).not.toThrow();
      expect(document.querySelectorAll('.luxar-glass-refraction').length).toBe(0);
    });
  });

  describe('[G1i] setupGlassRefractionObserver cleanup', () => {
    it('returns a cleanup function that disconnects the observer', async () => {
      // Set up the observer first (no panels yet — nothing to react to).
      const cleanup = setupGlassRefractionObserver();
      expect(typeof cleanup).toBe('function');

      // Disconnect immediately. Adding a glass panel afterward should NOT
      // trigger refraction injection (since the observer is gone).
      cleanup();

      // Stub rAF so any post-observer injection would be observable.
      const rafSpy = vi
        .spyOn(window, 'requestAnimationFrame')
        .mockImplementation(((_cb: FrameRequestCallback) => 1) as typeof requestAnimationFrame);

      const panel = document.createElement('div');
      panel.className = 'luxar-gui';
      document.body.appendChild(panel);

      // Wait a tick for any pending MutationObserver callbacks.
      await new Promise((resolve) => setTimeout(resolve, 0));

      // rAF was not invoked because the observer is disconnected.
      expect(rafSpy).not.toHaveBeenCalled();
      // The panel has no refraction layer (nothing injected it).
      expect(panel.querySelector('.luxar-glass-refraction')).toBeNull();

      rafSpy.mockRestore();
    });
  });

  describe('[G1j] setupGlassRefractionObserver container-scope fallback', () => {
    it('scopes the observer to .luxar-viewer when present', async () => {
      const viewer = document.createElement('div');
      viewer.className = 'luxar-viewer';
      document.body.appendChild(viewer);

      let rafFired = false;
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
        cb: FrameRequestCallback
      ) => {
        rafFired = true;
        cb(0);
        return 1;
      }) as typeof requestAnimationFrame);

      const cleanup = setupGlassRefractionObserver();

      // Adding a glass panel INSIDE .luxar-viewer must trigger the observer.
      const panel = document.createElement('div');
      panel.className = 'luxar-gui';
      viewer.appendChild(panel);

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(rafFired).toBe(true);
      // Refraction layer present (rAF callback was invoked synchronously by the spy).
      expect(panel.querySelector('.luxar-glass-refraction')).not.toBeNull();

      cleanup();
      rafSpy.mockRestore();
    });

    it('falls back to #luxar-container when .luxar-viewer is absent', async () => {
      const container = document.createElement('div');
      container.id = 'luxar-container';
      document.body.appendChild(container);

      let rafFired = false;
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
        cb: FrameRequestCallback
      ) => {
        rafFired = true;
        cb(0);
        return 1;
      }) as typeof requestAnimationFrame);

      const cleanup = setupGlassRefractionObserver();

      const panel = document.createElement('div');
      panel.className = 'luxar-gui';
      container.appendChild(panel);

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(rafFired).toBe(true);
      expect(panel.querySelector('.luxar-glass-refraction')).not.toBeNull();

      cleanup();
      rafSpy.mockRestore();
    });

    it('falls back to document.body when neither .luxar-viewer nor #luxar-container is present', async () => {
      let rafFired = false;
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
        cb: FrameRequestCallback
      ) => {
        rafFired = true;
        cb(0);
        return 1;
      }) as typeof requestAnimationFrame);

      const cleanup = setupGlassRefractionObserver();

      const panel = document.createElement('div');
      panel.className = 'luxar-gui';
      document.body.appendChild(panel);

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(rafFired).toBe(true);

      cleanup();
      rafSpy.mockRestore();
    });
  });

  describe('[G1k] MutationObserver selector matching', () => {
    it('does NOT trigger refraction injection when an unrelated element is added', async () => {
      const rafSpy = vi
        .spyOn(window, 'requestAnimationFrame')
        .mockImplementation(((_cb: FrameRequestCallback) => 1) as typeof requestAnimationFrame);

      const cleanup = setupGlassRefractionObserver();

      // Adding a non-glass element should not fire the rAF (selector mismatch).
      const unrelated = document.createElement('div');
      unrelated.className = 'some-random-class';
      document.body.appendChild(unrelated);

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(rafSpy).not.toHaveBeenCalled();

      cleanup();
      rafSpy.mockRestore();
    });

    it('triggers refraction injection when a glass-class element is added directly', async () => {
      let rafFired = false;
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
        cb: FrameRequestCallback
      ) => {
        rafFired = true;
        cb(0);
        return 1;
      }) as typeof requestAnimationFrame);

      const cleanup = setupGlassRefractionObserver();

      const panel = document.createElement('div');
      panel.className = 'luxar-debug-console'; // a glass selector
      document.body.appendChild(panel);

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(rafFired).toBe(true);
      expect(panel.querySelector('.luxar-glass-refraction')).not.toBeNull();

      cleanup();
      rafSpy.mockRestore();
    });
  });

  describe('[G1l] MutationObserver nested match', () => {
    it('triggers refraction injection when a CONTAINER element holds a glass child via querySelector', async () => {
      let rafFired = false;
      const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(((
        cb: FrameRequestCallback
      ) => {
        rafFired = true;
        cb(0);
        return 1;
      }) as typeof requestAnimationFrame);

      const cleanup = setupGlassRefractionObserver();

      // Build a non-glass wrapper that contains a glass child nested inside.
      const wrapper = document.createElement('div');
      wrapper.className = 'some-wrapper';
      const nestedGlass = document.createElement('div');
      nestedGlass.className = 'luxar-help-overlay';
      wrapper.appendChild(nestedGlass);

      // A single appendChild adds the WRAPPER node — the observer must
      // detect the nested glass child via node.querySelector.
      document.body.appendChild(wrapper);

      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(rafFired).toBe(true);
      expect(nestedGlass.querySelector('.luxar-glass-refraction')).not.toBeNull();

      cleanup();
      rafSpy.mockRestore();
    });
  });
});
