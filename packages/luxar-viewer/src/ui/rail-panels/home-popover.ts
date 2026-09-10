/**
 * Home rail popover — deeper reset actions behind the Home button.
 *
 * The rail's Home button reframes the camera to fit the whole scene on
 * left-click (the F shortcut); right-click opens this popover with the full
 * reset menu: fit scene, center on origin, reset dimension sliders, reset
 * rendering settings, and reset layer parameters.
 *
 * Visually this mirrors the View-options flyout: a horizontal row of square
 * icon chips (`.luxar-control-rail__chip`, the rail's "square button"
 * aesthetic) with a live caption strip below that names and explains the
 * hovered/focused action — instead of the flyout's floating per-chip
 * tooltips, so the explanation never overlaps the chips or the rail.
 *
 * The popover stays open after an action (consistent with the Navigation
 * popover) so several resets can be fired in a row; the rail overlay's
 * document-pointer / Escape handling closes it.
 *
 * @module ui/rail-panels/home-popover
 */

import { RAIL_ICONS } from '../control-rail/icons';
import { isTouchLikePointer } from '../../utils/input-capabilities';

export interface HomePopoverContext {
  /** Frame all visible geometry (same as the F shortcut / Home left-click). */
  fitScene: () => void;
  /** Re-target camera + controls to (0,0,0) at the current distance. */
  centerOnOrigin: () => void;
  /** All dimension sliders back to their initial defaults. */
  resetDimensions: () => void;
  /**
   * True when the scene has non-displayed (slider) dimensions to reset — a
   * 3D-only scene has nothing to reset, so the chip renders disabled.
   */
  hasDimensionSliders: () => boolean;
  /** Restore default rendering settings (same as the panel's Reset button). */
  resetRendering: () => void;
  /** All layer parameters back to their authored defaults. */
  resetLayers: () => void;
  /** True when the scene exposes layers to reset (chip disabled otherwise). */
  hasLayers: () => boolean;
  /** Request a render so the action's effect is visible immediately. */
  triggerAnimation: () => void;
}

/** One action chip: icon + caption text, firing `run` then a render. */
interface HomeAction {
  icon: string;
  label: string;
  hint: string;
  run: (ctx: HomePopoverContext) => void;
  /** When present and false at build time, the chip renders disabled. */
  enabled?: (ctx: HomePopoverContext) => boolean;
  /** Tooltip explaining a disabled chip (native title — see below). */
  disabledTitle?: string;
}

const ACTIONS: HomeAction[] = [
  {
    icon: RAIL_ICONS.fit,
    label: 'Fit scene',
    hint: 'Frame all visible geometry (F)',
    run: (ctx) => ctx.fitScene(),
  },
  {
    icon: RAIL_ICONS.origin,
    label: 'Center on origin',
    hint: 'Look at (0, 0, 0), keep distance',
    run: (ctx) => ctx.centerOnOrigin(),
  },
  {
    icon: RAIL_ICONS.dims,
    label: 'Reset dimensions',
    hint: 'All sliders back to their defaults',
    run: (ctx) => ctx.resetDimensions(),
    enabled: (ctx) => ctx.hasDimensionSliders(),
    disabledTitle: 'No dimension sliders in this scene',
  },
  {
    icon: RAIL_ICONS.render,
    label: 'Reset rendering',
    hint: 'Restore default rendering settings',
    run: (ctx) => ctx.resetRendering(),
  },
  {
    icon: RAIL_ICONS.layers,
    label: 'Reset layers',
    hint: 'All layer parameters back to defaults',
    run: (ctx) => ctx.resetLayers(),
    enabled: (ctx) => ctx.hasLayers(),
    disabledTitle: 'No layers in this scene',
  },
];

/** Caption shown while nothing is hovered/focused. */
const IDLE_LABEL = 'Home';
const IDLE_HINT = 'Hover a button · click to apply';

/**
 * Build the Home popover into `host`. Returns a teardown for symmetry with
 * the other rail-panel builders (nothing to release — all listeners live on
 * elements inside `host` and go away with the popover's DOM).
 */
export function buildHomePopover(host: HTMLElement, ctx: HomePopoverContext): () => void {
  const chips = document.createElement('div');
  chips.className = 'luxar-control-rail__home-chips';
  chips.setAttribute('role', 'group');
  chips.setAttribute('aria-label', 'Home actions');

  const caption = document.createElement('div');
  caption.className = 'luxar-control-rail__home-caption';
  // Screen readers get each chip's aria-label; the caption is the sighted
  // user's copy of the same text, so keep it out of the accessibility tree.
  caption.setAttribute('aria-hidden', 'true');

  const captionLabel = document.createElement('span');
  captionLabel.className = 'luxar-control-rail__home-caption-label';
  const captionHint = document.createElement('span');
  captionHint.className = 'luxar-control-rail__home-caption-hint';
  caption.appendChild(captionLabel);
  caption.appendChild(captionHint);

  const setCaption = (label: string, hint: string): void => {
    captionLabel.textContent = label;
    captionHint.textContent = hint;
  };
  setCaption(IDLE_LABEL, IDLE_HINT);

  for (const action of ACTIONS) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'luxar-control-rail__chip luxar-control-rail__home-chip';
    chip.setAttribute('aria-label', `${action.label} — ${action.hint}`);
    chip.innerHTML = action.icon;

    // The popover rebuilds fresh on every open, so the disabled state is
    // evaluated live (e.g. a 3D-only scene has no slider dimensions to reset).
    // Native `disabled` also suppresses hover events, so the caption stays
    // idle — the native `title` carries the explanation instead.
    if (action.enabled && !action.enabled(ctx)) {
      chip.disabled = true;
      if (action.disabledTitle) chip.title = action.disabledTitle;
    } else {
      chip.addEventListener('click', () => {
        action.run(ctx);
        ctx.triggerAnimation();
      });
      chip.addEventListener('mouseenter', () => setCaption(action.label, action.hint));
      chip.addEventListener('focus', () => setCaption(action.label, action.hint));
      // A finger cannot hover: name the action as the press begins, so the
      // caption reads before the tap commits. Mouse presses already hovered.
      chip.addEventListener('pointerdown', (e) => {
        if (isTouchLikePointer(e)) setCaption(action.label, action.hint);
      });
    }

    chips.appendChild(chip);
  }

  // Return to the idle caption when the pointer/focus leaves the chip row.
  chips.addEventListener('mouseleave', () => setCaption(IDLE_LABEL, IDLE_HINT));
  chips.addEventListener('focusout', (e) => {
    if (!chips.contains(e.relatedTarget as Node | null)) setCaption(IDLE_LABEL, IDLE_HINT);
  });

  host.appendChild(chips);
  host.appendChild(caption);
  return () => {};
}
