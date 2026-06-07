/**
 * Brief auto-dismissing toast notification (centered-bottom by CSS).
 *
 * Replaces any existing toast so multiple rapid calls collapse into one.
 */

import { getViewerContainer } from '../utils/viewer-container';

export function showToast(message: string, durationMs: number = 2000): void {
  const existing = document.getElementById('luxar-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = 'luxar-toast';
  toast.className = 'luxar-toast';
  toast.textContent = message;
  toast.style.opacity = '1';

  getViewerContainer().appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, durationMs);
}
