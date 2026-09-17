// @vitest-environment jsdom
/**
 * Guard: the dataset-browser toggle stays wired end to end.
 *
 * The event name is necessarily written twice. The listener
 * (`core/app/dataset/browser-shortcut.ts`) imports `OPEN_DATASET_BROWSER_EVENT`
 * from `core/app/interaction/canvas-actions.ts`; the dispatch lives in the
 * InputHandler's shared command surface (`input/input-handler.ts`), and the
 * layer contract in `.dependency-cruiser.cjs` forbids `input/` from importing
 * `core/`, so the command hard-codes the string. A rename on either side fails
 * silently — `O` just stops opening the browser — so the two spellings are
 * pinned together here (the `luxar-open-element-menu` twin is
 * `element-menu-event-sync.test.ts`).
 */
import { describe, expect, it, vi } from 'vitest';
import { OPEN_DATASET_BROWSER_EVENT } from '../../../../../core/app/interaction/canvas-actions';
import { InputHandler } from '../../../../../input';

/** A real InputHandler over stub collaborators, so the command runs for real. */
function makeHandler(): InputHandler {
  return new InputHandler(
    {
      renderer: { domElement: document.createElement('canvas') },
      controls: { addEventListener: vi.fn(), removeEventListener: vi.fn() },
    } as never,
    { startAnimation: vi.fn() } as never,
    { toggle: vi.fn(), cycleMode: vi.fn(), visible: false } as never,
    { toggle: vi.fn(), getIsVisible: vi.fn(() => false), dispose: vi.fn() } as never
  );
}

describe('dataset-browser command ↔ browser-shortcut listener', () => {
  it('the command dispatches exactly OPEN_DATASET_BROWSER_EVENT', () => {
    expect(OPEN_DATASET_BROWSER_EVENT).toBe('luxar-open-dataset-browser');
    const handler = makeHandler();
    handler.init();
    const listener = vi.fn();
    window.addEventListener(OPEN_DATASET_BROWSER_EVENT, listener);
    try {
      handler.getUiActions().commands.toggleDatasetBrowser();
      expect(listener).toHaveBeenCalledOnce();
    } finally {
      window.removeEventListener(OPEN_DATASET_BROWSER_EVENT, listener);
      handler.dispose();
    }
  });
});
