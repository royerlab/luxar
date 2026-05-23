/**
 * Unit tests for ThemeManager
 * Tests singleton pattern, theme switching, CSS variable injection, and observer pattern
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ThemeManager } from '../../../themes/theme-manager';
import { darkTheme } from '../../../themes/themes/dark.theme';
import { lightTheme } from '../../../themes/themes/light.theme';

describe('ThemeManager', () => {
  beforeEach(() => {
    // Reset singleton before each test
    ThemeManager.disposeInstance();
    // Clear localStorage
    localStorage.clear();
    // Clear any existing CSS variables
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    // Clean up
    ThemeManager.disposeInstance();
    localStorage.clear();
  });

  describe('Singleton Pattern', () => {
    it('should return the same instance on multiple calls', () => {
      const instance1 = ThemeManager.getInstance();
      const instance2 = ThemeManager.getInstance();
      expect(instance1).toBe(instance2);
    });

    it('should clear singleton when disposeInstance() is called', () => {
      const instance1 = ThemeManager.getInstance();
      ThemeManager.disposeInstance();
      const instance2 = ThemeManager.getInstance();
      expect(instance1).not.toBe(instance2);
    });
  });

  describe('Theme Registration', () => {
    it('registers all four default themes (dark, light, frosted-glass, liquid-glass) on initialization', () => {
      // themes.md W1 fix: previous version asserted `themes.length >= 3`
      // and listed only three IDs. Four built-in themes are registered;
      // a regression dropping `liquid-glass` would have slipped through.
      const manager = ThemeManager.getInstance();
      const themes = manager.getAllThemes();
      const ids = themes.map((t) => t.id);

      expect(ids).toEqual(expect.arrayContaining(['dark', 'light', 'frosted-glass', 'liquid-glass']));
      expect(themes.length).toBeGreaterThanOrEqual(4);
    });

    it('should retrieve theme by ID', () => {
      // themes.md W2 fix: the `darkTheme.toBeDefined()` line was redundant
      // — the next `darkTheme?.id` line already implies definedness.
      // Drop the redundant assertion.
      const manager = ThemeManager.getInstance();
      const darkTheme = manager.getTheme('dark');

      expect(darkTheme?.id).toBe('dark');
      expect(darkTheme?.name).toBe('Dark Theme');
    });

    it('should return undefined for invalid theme ID', () => {
      const manager = ThemeManager.getInstance();
      const invalidTheme = manager.getTheme('nonexistent');

      expect(invalidTheme).toBeUndefined();
    });
  });

  describe('Theme Switching', () => {
    it('should set data-theme attribute on root element', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('light');

      const dataTheme = document.documentElement.getAttribute('data-theme');
      expect(dataTheme).toBe('light');
    });

    it('should inject CSS variables when theme is set', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const bgPrimary = getComputedStyle(document.documentElement).getPropertyValue(
        '--luxar-bg-primary'
      );
      expect(bgPrimary.trim()).toBe('#111111');
    });

    it('should update CSS variables when switching themes', () => {
      const manager = ThemeManager.getInstance();

      // Set dark theme
      manager.setTheme('dark');
      let bgColor = getComputedStyle(document.documentElement).getPropertyValue(
        '--luxar-bg-primary'
      );
      expect(bgColor.trim()).toBe('#111111');

      // Switch to light theme
      manager.setTheme('light');
      bgColor = getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary');
      expect(bgColor.trim()).toBe('#ffffff');

      // Switch to frosted-glass theme
      manager.setTheme('frosted-glass');
      bgColor = getComputedStyle(document.documentElement).getPropertyValue('--luxar-bg-primary');
      expect(bgColor.trim()).toBe('rgba(255, 255, 255, 0.15)');
    });

    it('should throw error mentioning the missing theme ID for invalid input', () => {
      // themes.md W3 fix: previous version only asserted .toThrow() with
      // no message matcher. A mutation that threw a generic "Error" with
      // no useful message would pass. Pin the message format.
      const manager = ThemeManager.getInstance();
      expect(() => manager.setTheme('invalid')).toThrow(/invalid|not found|unknown/i);
    });

    it('should update currentTheme after switching', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('light');

      const current = manager.getCurrentTheme();
      expect(current.id).toBe('light');
    });
  });

  describe('Observer Pattern', () => {
    it('should notify observers when theme changes', () => {
      const manager = ThemeManager.getInstance();
      const callback = vi.fn();

      manager.onChange(callback);
      manager.setTheme('light');

      expect(callback).toHaveBeenCalledTimes(1);
      expect(callback).toHaveBeenCalledWith(expect.objectContaining({ id: 'light' }));
    });

    it('should support multiple observers', () => {
      const manager = ThemeManager.getInstance();
      const callback1 = vi.fn();
      const callback2 = vi.fn();

      manager.onChange(callback1);
      manager.onChange(callback2);
      manager.setTheme('light');

      expect(callback1).toHaveBeenCalledTimes(1);
      expect(callback2).toHaveBeenCalledTimes(1);
    });

    it('should unsubscribe observer when unsubscribe function is called', () => {
      const manager = ThemeManager.getInstance();
      const callback = vi.fn();

      const unsubscribe = manager.onChange(callback);
      manager.setTheme('light');
      expect(callback).toHaveBeenCalledTimes(1);

      // Unsubscribe
      unsubscribe();
      callback.mockClear();

      // Change theme again
      manager.setTheme('dark');
      expect(callback).not.toHaveBeenCalled();
    });

    it('should handle observer errors gracefully', () => {
      const manager = ThemeManager.getInstance();
      const errorCallback = vi.fn(() => {
        throw new Error('Observer error');
      });
      const normalCallback = vi.fn();

      manager.onChange(errorCallback);
      manager.onChange(normalCallback);

      // Should not throw even if one observer throws
      expect(() => manager.setTheme('light')).not.toThrow();

      // Normal callback should still be called
      expect(normalCallback).toHaveBeenCalledTimes(1);
    });
  });

  describe('Theme Persistence', () => {
    it('should save theme to localStorage', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('light');

      const savedTheme = localStorage.getItem('luxar.theme');
      expect(savedTheme).toBe('light');
    });

    it('does NOT persist (and does NOT update currentTheme) if applyTheme throws partway', () => {
      // HIGH-11 regression: prior to the fix, the contract that
      // "persisted state == currentTheme == DOM state" was only implicit.
      // If applyTheme throws after clearing/half-setting CSS variables,
      // the previous theme MUST remain the source of truth in
      // localStorage and in manager.currentTheme — otherwise a reload
      // would silently mask the failure by loading a theme that never
      // fully applied.
      const manager = ThemeManager.getInstance();

      // Establish a known-good baseline: 'light' fully applied + persisted.
      manager.setTheme('light');
      expect(localStorage.getItem('luxar.theme')) .toBe('light');
      expect(manager.getCurrentTheme().id).toBe('light');

      // Force the next applyTheme to throw. We stub the root element's
      // setAttribute (used by applyTheme to write the data-theme attribute)
      // so that the call fails partway through applyTheme — i.e. after
      // clearThemeVariables() but before setCSSVariables() completes.
      const root = document.documentElement;
      const setAttrSpy = vi.spyOn(root, 'setAttribute').mockImplementation((name: string) => {
        if (name === 'data-theme') {
          throw new Error('simulated SSR / unmount: setAttribute unavailable');
        }
        // Other attributes pass through to the real impl (none expected here).
      });

      expect(() => manager.setTheme('dark')).toThrow(/setAttribute unavailable/);

      // Persistence MUST be unchanged — we never reached saveTheme.
      expect(localStorage.getItem('luxar.theme')).toBe('light');

      // currentTheme MUST be unchanged — we never reached the assignment.
      expect(manager.getCurrentTheme().id).toBe('light');

      setAttrSpy.mockRestore();
    });

    it('should load saved theme on initialization', () => {
      // Set theme in localStorage before creating manager
      localStorage.setItem('luxar.theme', 'frosted-glass');

      const manager = ThemeManager.getInstance();
      const current = manager.getCurrentTheme();

      expect(current.id).toBe('frosted-glass');
    });

    it('should fallback to frosted-glass theme if saved theme is invalid', () => {
      localStorage.setItem('luxar.theme', 'invalid');

      const manager = ThemeManager.getInstance();
      const current = manager.getCurrentTheme();

      expect(current.id).toBe('frosted-glass');
    });

    it('should use frosted-glass theme if localStorage is empty', () => {
      const manager = ThemeManager.getInstance();
      const current = manager.getCurrentTheme();

      expect(current.id).toBe('frosted-glass');
    });
  });

  describe('CSS Variable Injection', () => {
    it('injects each color variable with the EXACT value from the active theme object', () => {
      // themes.md W4 fix: previous version asserted only `value !== ""`.
      // A mutation that swapped `darkTheme.colors.background.primary` with
      // any other non-empty string would have survived. Pin the exact
      // mapping from theme-object value → emitted CSS variable.
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const root = document.documentElement;
      const getVar = (name: string) =>
        getComputedStyle(root).getPropertyValue(name).trim();

      // background.*
      expect(getVar('--luxar-bg-primary')).toBe(darkTheme.colors.background.primary);
      expect(getVar('--luxar-bg-secondary')).toBe(darkTheme.colors.background.secondary);
      expect(getVar('--luxar-bg-tertiary')).toBe(darkTheme.colors.background.tertiary);

      // text.*
      expect(getVar('--luxar-text-primary')).toBe(darkTheme.colors.text.primary);
      expect(getVar('--luxar-text-secondary')).toBe(darkTheme.colors.text.secondary);

      // semantic.* — mutation swapping success/warning/error would now fail.
      expect(getVar('--luxar-success')).toBe(darkTheme.colors.semantic.success);
      expect(getVar('--luxar-warning')).toBe(darkTheme.colors.semantic.warning);
      expect(getVar('--luxar-error')).toBe(darkTheme.colors.semantic.error);
    });

    it('injects typography variables with the EXACT fontFamily.base from the theme', () => {
      // themes.md W5 fix: previous version only asserted
      // `fontBase.includes('apple-system')`. Pin the full string so a
      // mutant that reorders the font-stack fails the assertion.
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const fontBase = getComputedStyle(document.documentElement)
        .getPropertyValue('--luxar-font-base')
        .trim();
      // CSS may normalize whitespace; trim and compare canonical forms.
      const canonical = (s: string) => s.replace(/\s+/g, ' ').trim();
      expect(canonical(fontBase)).toBe(canonical(darkTheme.typography.fontFamily.base));
    });

    it('should inject spacing variables', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const spacing8 = getComputedStyle(document.documentElement).getPropertyValue(
        '--luxar-spacing-8'
      );
      expect(spacing8.trim()).toBe('16px');
    });

    it('overwrites each shared variable with the new theme value on switch (no leftover dark values)', () => {
      // themes.md W6 fix: the previous test claimed to verify
      // "variables are updated, not duplicated" but only sampled ONE
      // variable. A mutation that "clears nothing" would still see the
      // new value in `--luxar-bg-primary` (it gets set, just not
      // cleared). Strengthen by asserting MULTIPLE variables flipped to
      // the new theme's values AND none retain dark values.
      const manager = ThemeManager.getInstance();
      const root = document.documentElement;
      const getVar = (name: string) =>
        getComputedStyle(root).getPropertyValue(name).trim();

      manager.setTheme('dark');
      // Sanity: dark values are live before the switch.
      expect(getVar('--luxar-bg-primary')).toBe(darkTheme.colors.background.primary);
      expect(getVar('--luxar-text-primary')).toBe(darkTheme.colors.text.primary);
      expect(getVar('--luxar-success')).toBe(darkTheme.colors.semantic.success);

      manager.setTheme('light');

      // Every variable now reflects the LIGHT theme, with no leftover dark values.
      expect(getVar('--luxar-bg-primary')).toBe(lightTheme.colors.background.primary);
      expect(getVar('--luxar-text-primary')).toBe(lightTheme.colors.text.primary);
      expect(getVar('--luxar-success')).toBe(lightTheme.colors.semantic.success);

      // No variable still equals its dark-theme value (defends "duplicated, not cleared").
      // (We only assert the ones where dark and light demonstrably differ.)
      if (darkTheme.colors.background.primary !== lightTheme.colors.background.primary) {
        expect(getVar('--luxar-bg-primary')).not.toBe(darkTheme.colors.background.primary);
      }
      if (darkTheme.colors.text.primary !== lightTheme.colors.text.primary) {
        expect(getVar('--luxar-text-primary')).not.toBe(darkTheme.colors.text.primary);
      }
    });
  });

  describe('Disposal', () => {
    it('should clear all observers on dispose', () => {
      const manager = ThemeManager.getInstance();
      const callback = vi.fn();

      manager.onChange(callback);

      // Set theme first to trigger observer
      manager.setTheme('light');
      expect(callback).toHaveBeenCalledTimes(1);

      // Dispose and create new instance
      ThemeManager.disposeInstance();
      const newManager = ThemeManager.getInstance();
      newManager.setTheme('dark');

      // Old callback should not be called (only called once from before disposal)
      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('should remove data-theme attribute on dispose', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('light');

      expect(document.documentElement.getAttribute('data-theme')).toBe('light');

      manager.dispose();

      expect(document.documentElement.getAttribute('data-theme')).toBeNull();
    });

    it('should clear current theme on dispose', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('light');
      manager.dispose();

      expect(() => manager.getCurrentTheme()).toThrow();
    });

    it('removes the body-level glass filter SVG when liquid-glass theme is disposed', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('liquid-glass');
      expect(document.getElementById('luxar-glass-filters')).not.toBeNull();

      manager.dispose();

      expect(document.getElementById('luxar-glass-filters')).toBeNull();
    });

    it('cancels pending refraction-layer rAF when disposed before the frame fires', () => {
      let nextHandle = 1;
      const issuedHandles: number[] = [];
      const cancelledHandles: number[] = [];

      const rafSpy = vi
        .spyOn(window, 'requestAnimationFrame')
        .mockImplementation((_cb: FrameRequestCallback) => {
          const handle = nextHandle++;
          issuedHandles.push(handle);
          return handle;
        });
      const cafSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((h: number) => {
        cancelledHandles.push(h);
      });

      const manager = ThemeManager.getInstance();
      manager.setTheme('liquid-glass');

      // The refraction-layer rAF was scheduled but the callback hasn't run
      // yet. (Other rAFs may also be scheduled as side effects of DOM
      // mutation observers in glass-filters.ts; we only care that ours fired.)
      expect(rafSpy).toHaveBeenCalled();
      expect(issuedHandles.length).toBeGreaterThanOrEqual(1);

      manager.dispose();

      // The pending handle must have been cancelled.
      expect(cancelledHandles).toContain(issuedHandles[0]);

      // Sanity: fire the rAF callback manually post-dispose. It still runs (we
      // mocked rAF to just capture, never schedule), but dispose() removed
      // both the SVG filters and the refraction layers it would have injected.
      const capturedCallback = rafSpy.mock.calls[0][0] as FrameRequestCallback;
      capturedCallback(performance.now());
      expect(document.getElementById('luxar-glass-filters')).toBeNull();

      rafSpy.mockRestore();
      cafSpy.mockRestore();
    });

    it('cancels pending refraction-layer rAF when switching away from liquid-glass', () => {
      let nextHandle = 1;
      const issuedHandles: number[] = [];
      const cancelledHandles: number[] = [];

      const rafSpy = vi
        .spyOn(window, 'requestAnimationFrame')
        .mockImplementation((_cb: FrameRequestCallback) => {
          const handle = nextHandle++;
          issuedHandles.push(handle);
          return handle;
        });
      const cafSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation((h: number) => {
        cancelledHandles.push(h);
      });

      const manager = ThemeManager.getInstance();
      manager.setTheme('liquid-glass');
      const liquidGlassHandle = issuedHandles[0];

      // Switch to a non-glass theme before the rAF fires.
      manager.setTheme('dark');

      expect(cancelledHandles).toContain(liquidGlassHandle);

      rafSpy.mockRestore();
      cafSpy.mockRestore();
    });
  });
});
