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

    // [themes.md/O2][P4] Replaced a single test that bundled three sequential
    // theme switches into one it() with an it.each table — each row exercises
    // one (themeId → expected --luxar-bg-primary) mapping in isolation.
    it.each([
      ['dark', '#111111'],
      ['light', '#ffffff'],
      ['frosted-glass', 'rgba(255, 255, 255, 0.15)'],
    ])('sets --luxar-bg-primary correctly for theme %s', (themeId, expectedBg) => {
      const manager = ThemeManager.getInstance();
      manager.setTheme(themeId);
      const bgColor = getComputedStyle(document.documentElement).getPropertyValue(
        '--luxar-bg-primary',
      );
      expect(bgColor.trim()).toBe(expectedBg);
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
    // [themes.md/O3][P4] Replaced six expect() calls in a single it() with
    // it.each — each (cssVar → theme-object-path) pair becomes its own case
    // so individual mutations surface as a single failing row.
    // Also closes [themes.md/W4][P2]: previous version asserted only
    // `value !== ""`; now pins the exact mapping from theme-object value to
    // emitted CSS variable, so a mutant swapping any value with another
    // non-empty string still fails.
    it.each([
      ['--luxar-bg-primary', () => darkTheme.colors.background.primary],
      ['--luxar-bg-secondary', () => darkTheme.colors.background.secondary],
      ['--luxar-bg-tertiary', () => darkTheme.colors.background.tertiary],
      ['--luxar-text-primary', () => darkTheme.colors.text.primary],
      ['--luxar-text-secondary', () => darkTheme.colors.text.secondary],
      ['--luxar-success', () => darkTheme.colors.semantic.success],
      ['--luxar-warning', () => darkTheme.colors.semantic.warning],
      ['--luxar-error', () => darkTheme.colors.semantic.error],
    ])('injects %s with the EXACT value from the active theme object', (cssVar, expected) => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const root = document.documentElement;
      const actual = getComputedStyle(root).getPropertyValue(cssVar).trim();
      expect(actual).toBe(expected());
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

  // themes.md G2-G5, G7, G9 — boundary / error-path / symmetry gaps that
  // the audit flagged as carry-overs from the prior audit. These tests
  // close the gaps directly against ThemeManager's public API.
  describe('Boundary and error paths (themes.md G2-G5, G7, G9)', () => {
    it('[G2] registerTheme throws when given a theme whose id already exists', () => {
      // theme-manager.ts:144-149: duplicate-id throws with a message that
      // names the offending id. The four built-in themes are registered
      // during construction, so re-registering darkTheme triggers the
      // branch on a real-world id.
      const manager = ThemeManager.getInstance();
      expect(() => manager.registerTheme(darkTheme)).toThrow(/already registered/);
      expect(() => manager.registerTheme(darkTheme)).toThrow(/"dark"/);
    });

    it('[G3] saveTheme swallows localStorage.setItem failures and logs a warning', () => {
      // theme-manager.ts:486-492: setItem in a try/catch — the error must
      // not propagate to the caller (otherwise setTheme would fail mid-
      // pipeline). Stub setItem to throw and confirm setTheme completes
      // cleanly + log.warning fires.
      const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new Error('quota exceeded');
      });
      const manager = ThemeManager.getInstance();

      expect(() => manager.setTheme('light')).not.toThrow();
      // applyTheme still ran — data-theme attribute is on the root.
      expect(document.documentElement.getAttribute('data-theme')).toBe('light');
      // currentTheme was updated.
      expect(manager.getCurrentTheme().id).toBe('light');

      setItemSpy.mockRestore();
    });

    it('[G3] loadTheme swallows localStorage.getItem failures and falls back to default', () => {
      // theme-manager.ts:499-506: getItem in a try/catch — null fallback
      // forces frosted-glass default per theme-manager initialization.
      const getItemSpy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
        throw new Error('storage disabled');
      });

      // Construct after the stub so initializeTheme hits the throw.
      const manager = ThemeManager.getInstance();
      expect(manager.getCurrentTheme().id).toBe('frosted-glass');

      getItemSpy.mockRestore();
    });

    it('[G4] CSS variables for optional theme fields with undefined values are NOT injected', () => {
      // theme-manager.ts:459-463: themeToCSSVariables maps `undefined`
      // entries to "skip"; setCSSVariables then guards `value !== undefined`
      // before setProperty. Verify by registering a clone of darkTheme
      // with `text.disabled = undefined` and confirming the corresponding
      // CSS variable is empty (root.style.getPropertyValue returns '').
      const manager = ThemeManager.getInstance();
      const clone: typeof darkTheme = JSON.parse(JSON.stringify(darkTheme));
      clone.id = 'g4-test-theme';
      clone.name = 'G4 Test Theme';
      // text.disabled is an optional field.
      (clone.colors.text as { disabled?: string }).disabled = undefined;
      manager.registerTheme(clone);
      manager.setTheme('g4-test-theme');

      const root = document.documentElement;
      // setProperty is never called for the undefined value, so the
      // inline-style getPropertyValue is empty. (computedStyle could
      // return an inherited or fallback value, so we check the inline
      // style directly which is the contract enforced by setCSSVariables.)
      expect(root.style.getPropertyValue('--luxar-text-disabled')).toBe('');
    });

    it('[G5] re-applying the same theme is idempotent (DOM + observers unchanged)', () => {
      // theme-manager.ts:204-216: setting the same theme twice should be
      // a clean no-op for the DOM (same data-theme attribute, same CSS
      // variables) but observers still fire each time (the public contract
      // does NOT debounce).
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const root = document.documentElement;
      const snapshot = {
        dataTheme: root.getAttribute('data-theme'),
        bgPrimary: getComputedStyle(root).getPropertyValue('--luxar-bg-primary').trim(),
        textPrimary: getComputedStyle(root).getPropertyValue('--luxar-text-primary').trim(),
      };

      const observerSpy = vi.fn();
      manager.onChange(observerSpy);

      expect(() => manager.setTheme('dark')).not.toThrow();

      // DOM state is identical.
      expect(root.getAttribute('data-theme')).toBe(snapshot.dataTheme);
      expect(getComputedStyle(root).getPropertyValue('--luxar-bg-primary').trim()).toBe(
        snapshot.bgPrimary
      );
      expect(getComputedStyle(root).getPropertyValue('--luxar-text-primary').trim()).toBe(
        snapshot.textPrimary
      );

      // Observer fires (no debouncing — public contract).
      expect(observerSpy).toHaveBeenCalledTimes(1);
      expect(observerSpy).toHaveBeenCalledWith(expect.objectContaining({ id: 'dark' }));
    });

    it.each([
      ['dark'],
      ['light'],
      ['frosted-glass'],
      ['liquid-glass'],
    ])('[G7] data-theme attribute round-trips to %s', (id) => {
      // theme-manager.ts:204-209: setTheme writes the id to data-theme.
      // Prior test covered only 'light'; the audit flagged the missing
      // four-theme round-trip.
      const manager = ThemeManager.getInstance();
      manager.setTheme(id);
      expect(document.documentElement.getAttribute('data-theme')).toBe(id);
    });

    it('[G9] setTheme notifies observers even when saveTheme silently fails', () => {
      // theme-manager.ts:486-492 wraps setItem in try/catch (swallowed),
      // so saveTheme NEVER throws to setTheme — observers always fire.
      // This pins that contract: a localStorage failure cannot prevent
      // theme-change notifications.
      const setItemSpy = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new Error('quota exceeded');
      });
      const manager = ThemeManager.getInstance();
      const observerSpy = vi.fn();
      manager.onChange(observerSpy);

      manager.setTheme('light');

      expect(observerSpy).toHaveBeenCalledTimes(1);
      expect(observerSpy).toHaveBeenCalledWith(expect.objectContaining({ id: 'light' }));

      setItemSpy.mockRestore();
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
