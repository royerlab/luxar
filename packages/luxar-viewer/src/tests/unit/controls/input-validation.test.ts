/**
 * Tests for input validation and keyboard event handling utilities
 *
 * These utilities determine when keyboard input should be blocked
 * (e.g., when typing in text fields) and handle FOV adjustments.
 */

import { describe, it, expect } from 'vitest';
import {
  isNavigationKey,
  calculateFovChange,
  shouldBlockShortcut,
} from '../../../input/input-handler-utils';

describe('Input Validation Utilities', () => {
  describe('isNavigationKey', () => {
    it('should identify navigation keys', () => {
      const navKeys = ['[', ']', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

      for (const key of navKeys) {
        const event = new KeyboardEvent('keydown', { key });
        expect(isNavigationKey(event)).toBe(true);
      }
    });

    it('should reject non-navigation keys', () => {
      const nonNavKeys = ['a', 'Enter', ' ', 'Escape', '0'];

      for (const key of nonNavKeys) {
        const event = new KeyboardEvent('keydown', { key });
        expect(isNavigationKey(event)).toBe(false);
      }
    });

    it('should reject when typing in input fields', () => {
      const input = document.createElement('input');
      const event = new KeyboardEvent('keydown', { key: '1' });
      Object.defineProperty(event, 'target', { value: input, writable: false });

      expect(isNavigationKey(event)).toBe(false);
    });

    it('should reject when typing in textareas', () => {
      const textarea = document.createElement('textarea');
      const event = new KeyboardEvent('keydown', { key: '[' });
      Object.defineProperty(event, 'target', { value: textarea, writable: false });

      expect(isNavigationKey(event)).toBe(false);
    });
  });

  describe('calculateFovChange', () => {
    it('should increase FOV with positive delta', () => {
      const newFov = calculateFovChange(60, 10);
      expect(newFov).toBe(61); // 60 + (10 * 0.1)
    });

    it('should decrease FOV with negative delta', () => {
      const newFov = calculateFovChange(60, -10);
      expect(newFov).toBe(59); // 60 + (-10 * 0.1)
    });

    it('should clamp to minimum FOV', () => {
      const newFov = calculateFovChange(15, -100);
      expect(newFov).toBe(10); // Clamped to min
    });

    it('should clamp to maximum FOV', () => {
      const newFov = calculateFovChange(165, 100);
      expect(newFov).toBe(170); // Clamped to max (config.camera.fovMax = 170)
    });

    it('should use custom sensitivity', () => {
      const newFov = calculateFovChange(60, 10, 0.5);
      expect(newFov).toBe(65); // 60 + (10 * 0.5)
    });
  });

  describe('shouldBlockShortcut', () => {
    it('should block when modal is active', () => {
      const event = new KeyboardEvent('keydown', { key: 'a' });
      expect(shouldBlockShortcut(event, ['settings-modal'])).toBe(true);
    });

    it('should block when typing in input', () => {
      const input = document.createElement('input');
      const event = new KeyboardEvent('keydown', { key: 'a' });
      Object.defineProperty(event, 'target', { value: input, writable: false });

      expect(shouldBlockShortcut(event)).toBe(true);
    });

    it('should block when typing in textarea', () => {
      const textarea = document.createElement('textarea');
      const event = new KeyboardEvent('keydown', { key: 'a' });
      Object.defineProperty(event, 'target', { value: textarea, writable: false });

      expect(shouldBlockShortcut(event)).toBe(true);
    });

    it('should not block browser shortcuts', () => {
      const event1 = new KeyboardEvent('keydown', { key: 's', ctrlKey: true });
      const event2 = new KeyboardEvent('keydown', { key: 'o', ctrlKey: true });
      const event3 = new KeyboardEvent('keydown', { key: 'p', ctrlKey: true });

      expect(shouldBlockShortcut(event1)).toBe(false);
      expect(shouldBlockShortcut(event2)).toBe(false);
      expect(shouldBlockShortcut(event3)).toBe(false);
    });

    it('should not block normal keys when no modal active', () => {
      const event = new KeyboardEvent('keydown', { key: 'a' });
      expect(shouldBlockShortcut(event, [])).toBe(false);
    });
  });
});
