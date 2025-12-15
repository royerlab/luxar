/**
 * Unit tests for UI helper functions
 * Tests critical fixes: memory leaks, race conditions, ARIA attributes
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { showHelpOverlay, hideHelpOverlay, showError, clearError } from '../../../ui/helpers';

// Mock DOM environment
beforeEach(() => {
  document.body.innerHTML = '';
  // Clean up any existing event listeners
  document.removeEventListener('click', () => {});
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('UI Helpers - Critical Fixes', () => {
  describe('showHelpOverlay - Memory Leak Prevention', () => {
    it('should not create multiple overlays when called repeatedly', () => {
      showHelpOverlay();
      const firstOverlay = document.getElementById('help-overlay');
      expect(firstOverlay).toBeTruthy();

      // Try to create another one
      showHelpOverlay();
      const allOverlays = document.querySelectorAll('#help-overlay');

      // Should still be only one
      expect(allOverlays.length).toBe(1);
      expect(document.getElementById('help-overlay')).toBe(firstOverlay);
    });

    it('should properly clean up global click listener when closed via keyboard', async () => {
      const removeEventSpy = vi.spyOn(document, 'removeEventListener');

      showHelpOverlay();
      const overlay = document.getElementById('help-overlay');
      expect(overlay).toBeTruthy();

      // Wait for the global click listener to be added
      await new Promise((resolve) => setTimeout(resolve, 150));

      // Simulate Escape key
      const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true });
      overlay?.dispatchEvent(event);

      // Check overlay is removed
      expect(document.getElementById('help-overlay')).toBeNull();

      // Verify removeEventListener was called for 'click'
      const clickRemovals = removeEventSpy.mock.calls.filter((call) => call[0] === 'click');
      expect(clickRemovals.length).toBeGreaterThanOrEqual(1);

      removeEventSpy.mockRestore();
    });

    it('should properly clean up when close button is clicked', async () => {
      const removeEventSpy = vi.spyOn(document, 'removeEventListener');

      showHelpOverlay();

      // Find the close button
      const closeBtn = document.querySelector('button[title="Close (Escape)"]') as HTMLButtonElement;
      expect(closeBtn).toBeTruthy();

      // Wait for timeout to add global listener
      await new Promise((resolve) => setTimeout(resolve, 150));

      // Click close button
      closeBtn?.click();

      // Overlay should be removed
      expect(document.getElementById('help-overlay')).toBeNull();

      // Verify removeEventListener was called for 'click'
      const clickRemovals = removeEventSpy.mock.calls.filter((call) => call[0] === 'click');
      expect(clickRemovals.length).toBeGreaterThanOrEqual(1);

      removeEventSpy.mockRestore();
    });

    it('should have proper ARIA attributes', () => {
      showHelpOverlay();
      const overlay = document.getElementById('help-overlay');

      expect(overlay?.getAttribute('role')).toBe('dialog');
      expect(overlay?.getAttribute('aria-modal')).toBe('true');
      expect(overlay?.getAttribute('aria-labelledby')).toBe('help-overlay-title');

      const title = document.getElementById('help-overlay-title');
      expect(title).toBeTruthy();
      expect(title?.textContent).toContain('Luxar Controls');
    });
  });

  describe('showError - ARIA Attributes', () => {
    it('should have proper ARIA attributes for alertdialog', () => {
      showError('Test error message');
      const errorDialog = document.getElementById('error-message');

      expect(errorDialog?.getAttribute('role')).toBe('alertdialog');
      expect(errorDialog?.getAttribute('aria-modal')).toBe('true');
      expect(errorDialog?.getAttribute('aria-labelledby')).toBe('error-title');
      expect(errorDialog?.getAttribute('aria-describedby')).toBe('error-message-text');
    });

    it('should have properly linked title and message', () => {
      showError('Custom error message');

      const title = document.getElementById('error-title');
      const message = document.getElementById('error-message-text');

      expect(title).toBeTruthy();
      expect(message).toBeTruthy();
      expect(message?.textContent).toBe('Custom error message');
    });

    it('should replace existing errors instead of stacking', () => {
      showError('First error');
      showError('Second error');

      const errors = document.querySelectorAll('#error-message');
      expect(errors.length).toBe(1);

      const messageText = document.getElementById('error-message-text');
      expect(messageText?.textContent).toBe('Second error');
    });
  });

  describe('clearError', () => {
    it('should remove error message', () => {
      showError('Test error');
      expect(document.getElementById('error-message')).toBeTruthy();

      clearError();
      expect(document.getElementById('error-message')).toBeNull();
    });

    it('should be safe to call when no error exists', () => {
      expect(() => clearError()).not.toThrow();
    });
  });

  describe('hideHelpOverlay', () => {
    it('should remove help overlay', () => {
      showHelpOverlay();
      expect(document.getElementById('help-overlay')).toBeTruthy();

      hideHelpOverlay();
      expect(document.getElementById('help-overlay')).toBeNull();
    });

    it('should be safe to call when no overlay exists', () => {
      expect(() => hideHelpOverlay()).not.toThrow();
    });
  });
});
