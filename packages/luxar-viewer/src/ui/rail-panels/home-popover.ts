/**
 * Home rail popover — deeper reset actions behind the Home button.
 *
 * The rail's Home button reframes the camera to fit the whole scene on
 * left-click (the F shortcut); right-click opens this popover with the full
 * reset menu: fit scene, center on origin, reset dimension sliders, and reset
 * rendering settings. Unlike the sibling popovers this one hosts no nested
 * GUI — it is a plain vertical stack of action rows (label + hint), so there
 * is nothing to dispose beyond the host's own DOM.
 *
 * The popover stays open after an action (consistent with the Navigation
 * popover) so several resets can be fired in a row; the rail overlay's
 * document-pointer / Escape handling closes it.
 *
 * @module ui/rail-panels/home-popover
 */

export interface HomePopoverContext {
  /** Frame all visible geometry (same as the F shortcut / Home left-click). */
  fitScene: () => void;
  /** Re-target camera + controls to (0,0,0) at the current distance. */
  centerOnOrigin: () => void;
  /** All dimension sliders back to their initial defaults. */
  resetDimensions: () => void;
  /**
   * True when the scene has non-displayed (slider) dimensions to reset — a
   * 3D-only scene has nothing to reset, so the row renders grayed.
   */
  hasDimensionSliders: () => boolean;
  /** Restore default rendering settings (same as the panel's Reset button). */
  resetRendering: () => void;
  /** Request a render so the action's effect is visible immediately. */
  triggerAnimation: () => void;
}

/** One action row: label + muted hint, firing `run` then a render. */
interface HomeAction {
  label: string;
  hint: string;
  run: (ctx: HomePopoverContext) => void;
  /** When present and false at build time, the row renders disabled. */
  enabled?: (ctx: HomePopoverContext) => boolean;
  /** Tooltip explaining a disabled row. */
  disabledTitle?: string;
}

const ACTIONS: HomeAction[] = [
  {
    label: 'Fit scene',
    hint: 'Frame all visible geometry (F)',
    run: (ctx) => ctx.fitScene(),
  },
  {
    label: 'Center on origin',
    hint: 'Look at (0, 0, 0), keep distance',
    run: (ctx) => ctx.centerOnOrigin(),
  },
  {
    label: 'Reset dimensions',
    hint: 'All sliders back to their defaults',
    run: (ctx) => ctx.resetDimensions(),
    enabled: (ctx) => ctx.hasDimensionSliders(),
    disabledTitle: 'No dimension sliders in this scene',
  },
  {
    label: 'Reset rendering',
    hint: 'Restore default rendering settings',
    run: (ctx) => ctx.resetRendering(),
  },
];

/**
 * Build the Home popover into `host`. Returns a teardown for symmetry with
 * the other rail-panel builders (nothing to release — all listeners live on
 * elements inside `host` and go away with the popover's DOM).
 */
export function buildHomePopover(host: HTMLElement, ctx: HomePopoverContext): () => void {
  const stack = document.createElement('div');
  stack.className = 'luxar-control-rail__actions';
  stack.setAttribute('role', 'group');
  stack.setAttribute('aria-label', 'Home actions');

  for (const action of ACTIONS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'luxar-control-rail__action';

    const label = document.createElement('span');
    label.className = 'luxar-control-rail__action-label';
    label.textContent = action.label;
    btn.appendChild(label);

    const hint = document.createElement('span');
    hint.className = 'luxar-control-rail__action-hint';
    hint.textContent = action.hint;
    btn.appendChild(hint);

    // The popover rebuilds fresh on every open, so the disabled state is
    // evaluated live (e.g. a 3D-only scene has no slider dimensions to reset).
    if (action.enabled && !action.enabled(ctx)) {
      btn.disabled = true;
      if (action.disabledTitle) btn.title = action.disabledTitle;
    } else {
      btn.addEventListener('click', () => {
        action.run(ctx);
        ctx.triggerAnimation();
      });
    }

    stack.appendChild(btn);
  }

  host.appendChild(stack);
  return () => {};
}
