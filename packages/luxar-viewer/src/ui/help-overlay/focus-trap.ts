/**
 * Focus-trap utility for modal dialogs.
 *
 * Keeps Tab/Shift+Tab cycling within the container's focusable elements
 * so focus can't escape to the background while a modal is open. The
 * cleanup function restores focus to the previously-focused element.
 *
 * Shared by error-overlay and help-overlay.
 */

export function trapFocus(container: HTMLElement): () => void {
  const focusableSelectors =
    'a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])';
  const previouslyFocused = document.activeElement as HTMLElement;

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;
    const focusable = Array.from(container.querySelectorAll<HTMLElement>(focusableSelectors));
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  };

  container.addEventListener('keydown', handleKeyDown);

  // Focus first focusable element. The timer id is captured so the
  // cleanup function can cancel it — otherwise a trap that's released
  // before the next tick leaks a pending timer (audit G17).
  const firstFocusable = container.querySelector<HTMLElement>(focusableSelectors);
  let focusTimerId: ReturnType<typeof setTimeout> | null = null;
  if (firstFocusable) {
    focusTimerId = setTimeout(() => {
      focusTimerId = null;
      firstFocusable.focus();
    }, 0);
  }

  return () => {
    if (focusTimerId !== null) {
      clearTimeout(focusTimerId);
      focusTimerId = null;
    }
    container.removeEventListener('keydown', handleKeyDown);
    if (previouslyFocused) previouslyFocused.focus();
  };
}
