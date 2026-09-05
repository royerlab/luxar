/**
 * The offline capture's modal progress overlay — ARIA semantics, focus trap,
 * Escape/Cancel wiring, live preview and frame counter.
 *
 * Extracted from `OfflineCaptureStrategy.runOfflineCaptureLoop` (audit A2-01).
 * It is the one part of that method with no capture logic in it at all: pure
 * DOM plus the two small sinks (`setLabel` / `setPreview`) the drivers write
 * progress into. Pulling it out is what lets the loop read as a loop.
 */

import { getViewerContainer } from '../../utils/viewer-container';

/** The progress sinks a capture driver writes into. */
export interface CaptureProgress {
  setLabel(text: string): void;
  setPreview(canvas: HTMLCanvasElement): void;
}

/** A mounted overlay, plus the handles the capture loop drives it with. */
export interface OfflineCaptureOverlay {
  progress: CaptureProgress;
  /** `n/total` in the counter element. No-op if the element is missing. */
  setFrameCount(captured: number): void;
  /** Idempotent: unbinds listeners, removes the node, restores focus. */
  cleanup(): void;
}

/**
 * Build, wire and mount the overlay.
 *
 * @param totalFrames  denominator for the counter.
 * @param onCancel     invoked by the Cancel button and by Escape.
 * @param onCleanup    called at the end of `cleanup()` so the owner can drop
 *                     its own reference to it (the strategy nulls
 *                     `this.overlayCleanup` only if it still points here).
 */
export function createOfflineCaptureOverlay(
  totalFrames: number,
  onCancel: () => void,
  onCleanup: (self: () => void) => void
): OfflineCaptureOverlay {
  // Offline overlay — modal dialog with ARIA semantics, focus trap,
  // and explicit Escape handler that aborts the session.
  const overlay = document.createElement('div');
  overlay.className = 'luxar-recording-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-labelledby', 'luxar-recording-overlay-label');
  overlay.setAttribute('aria-describedby', 'luxar-recording-overlay-counter');
  overlay.innerHTML = `
      <div class="luxar-recording-overlay__content">
        <canvas class="luxar-recording-overlay__preview"></canvas>
        <div class="luxar-recording-overlay__progress">
          <span id="luxar-recording-overlay-label" class="luxar-recording-overlay__label">Capturing frames...</span>
          <span id="luxar-recording-overlay-counter" class="luxar-recording-overlay__counter">0/${totalFrames}</span>
        </div>
        <button class="luxar-recording-overlay__cancel">Cancel</button>
      </div>
    `;
  const previewCanvas = overlay.querySelector(
    '.luxar-recording-overlay__preview'
  ) as HTMLCanvasElement;
  const previewCtx = previewCanvas.getContext('2d');
  const cancelButton = overlay.querySelector(
    '.luxar-recording-overlay__cancel'
  ) as HTMLButtonElement | null;
  const previouslyFocused = document.activeElement as HTMLElement | null;
  const handleOverlayKeydown = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') {
      e.preventDefault();
      e.stopPropagation();
      onCancel();
      return;
    }
    if (e.key === 'Tab') {
      const focusable = Array.from(
        overlay.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        )
      ).filter((el) => !el.hasAttribute('disabled'));
      if (focusable.length > 0) {
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey && active === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && active === last) {
          e.preventDefault();
          first.focus();
        }
      }
      e.stopPropagation();
      return;
    }
    e.stopPropagation();
  };
  cancelButton?.addEventListener('click', onCancel);
  overlay.addEventListener('keydown', handleOverlayKeydown, true);

  let overlayCleaned = false;
  const cleanup = (): void => {
    if (overlayCleaned) return;
    overlayCleaned = true;
    cancelButton?.removeEventListener('click', onCancel);
    overlay.removeEventListener('keydown', handleOverlayKeydown, true);
    overlay.remove();
    if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
      previouslyFocused.focus();
    }
    onCleanup(cleanup);
  };

  getViewerContainer().appendChild(overlay);
  cancelButton?.focus();

  const counterEl = overlay.querySelector('.luxar-recording-overlay__counter');
  const labelEl = overlay.querySelector('.luxar-recording-overlay__label');

  return {
    progress: {
      setLabel: (text: string): void => {
        if (labelEl) labelEl.textContent = text;
      },
      setPreview: (canvas: HTMLCanvasElement): void => {
        if (!previewCtx) return;
        if (previewCanvas.width !== canvas.width || previewCanvas.height !== canvas.height) {
          previewCanvas.width = canvas.width;
          previewCanvas.height = canvas.height;
        }
        previewCtx.drawImage(canvas, 0, 0);
      },
    },
    setFrameCount: (captured: number): void => {
      if (counterEl) counterEl.textContent = `${captured}/${totalFrames}`;
    },
    cleanup,
  };
}
