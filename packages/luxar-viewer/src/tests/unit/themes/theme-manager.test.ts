/**
 * Unit tests for ThemeManager
 * Tests singleton pattern, theme switching, CSS variable injection, and observer pattern
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ThemeManager } from '../../../themes/theme-manager';

describe('ThemeManager', () => {
  beforeEach(() => {
    // Reset singleton before each test
    ThemeManager.resetInstance();
    // Clear localStorage
    localStorage.clear();
    // Clear any existing CSS variables
    document.documentElement.removeAttribute('data-theme');
  });

  afterEach(() => {
    // Clean up
    ThemeManager.resetInstance();
    localStorage.clear();
  });

  describe('Singleton Pattern', () => {
    it('should return the same instance on multiple calls', () => {
      const instance1 = ThemeManager.getInstance();
      const instance2 = ThemeManager.getInstance();
      expect(instance1).toBe(instance2);
    });

    it('should reset instance when resetInstance() is called', () => {
      const instance1 = ThemeManager.getInstance();
      ThemeManager.resetInstance();
      const instance2 = ThemeManager.getInstance();
      expect(instance1).not.toBe(instance2);
    });
  });

  describe('Theme Registration', () => {
    it('should register default themes on initialization', () => {
      const manager = ThemeManager.getInstance();
      const themes = manager.getAllThemes();

      expect(themes.length).toBeGreaterThanOrEqual(3);
      expect(themes.some((t) => t.id === 'dark')).toBe(true);
      expect(themes.some((t) => t.id === 'light')).toBe(true);
      expect(themes.some((t) => t.id === 'frosted-glass')).toBe(true);
    });

    it('should retrieve theme by ID', () => {
      const manager = ThemeManager.getInstance();
      const darkTheme = manager.getTheme('dark');

      expect(darkTheme).toBeDefined();
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

    it('should throw error for invalid theme ID', () => {
      const manager = ThemeManager.getInstance();
      expect(() => manager.setTheme('invalid')).toThrow();
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

      const savedTheme = localStorage.getItem('luxar-theme');
      expect(savedTheme).toBe('light');
    });

    it('should load saved theme on initialization', () => {
      // Set theme in localStorage before creating manager
      localStorage.setItem('luxar-theme', 'frosted-glass');

      const manager = ThemeManager.getInstance();
      const current = manager.getCurrentTheme();

      expect(current.id).toBe('frosted-glass');
    });

    it('should fallback to frosted-glass theme if saved theme is invalid', () => {
      localStorage.setItem('luxar-theme', 'invalid');

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
    it('should inject all color variables', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const variables = [
        '--luxar-bg-primary',
        '--luxar-bg-secondary',
        '--luxar-text-primary',
        '--luxar-success',
        '--luxar-warning',
        '--luxar-error',
      ];

      variables.forEach((varName) => {
        const value = getComputedStyle(document.documentElement).getPropertyValue(varName);
        expect(value.trim()).not.toBe('');
      });
    });

    it('should inject typography variables', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const fontBase = getComputedStyle(document.documentElement).getPropertyValue(
        '--luxar-font-base'
      );
      expect(fontBase).toContain('apple-system');
    });

    it('should inject spacing variables', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');

      const spacing8 = getComputedStyle(document.documentElement).getPropertyValue(
        '--luxar-spacing-8'
      );
      expect(spacing8.trim()).toBe('16px');
    });

    it('should clear old theme variables when switching', () => {
      const manager = ThemeManager.getInstance();
      manager.setTheme('dark');
      manager.setTheme('light');

      // Verify variables are updated, not duplicated
      const bgColor = getComputedStyle(document.documentElement).getPropertyValue(
        '--luxar-bg-primary'
      );
      expect(bgColor.trim()).toBe('#ffffff'); // Light theme value
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
      ThemeManager.resetInstance();
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
  });
});
