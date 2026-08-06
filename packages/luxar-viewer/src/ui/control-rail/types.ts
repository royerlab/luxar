/**
 * Type descriptors for the control-rail items, flyout chips, and panel
 * popovers. Kept separate so both the {@link ControlRail} class (control-rail.ts) and
 * the {@link RailOverlay} (rail-overlay.ts) can import them without a cycle.
 *
 * @module ui/control-rail/types
 */

/** One button in the rail. */
export interface ControlRailItem {
  /** Stable id (also used as a data attribute). */
  id: string;
  /** Human-readable name, shown in the tooltip + aria-label. */
  title: string;
  /** Keyboard shortcut shown in the tooltip (display only). */
  shortcut?: string;
  /** Inline SVG markup for the icon. */
  icon: string;
  /** Invoked on click — should be the same action as the shortcut. */
  activate: () => void;
  /** CSS selector whose visible presence means this item's panel is open. */
  openSelector?: string;
  /** Explicit active check (takes precedence over {@link openSelector}). */
  isActive?: () => boolean;
  /** Momentary action (e.g. screenshot) — never shows an active state. */
  momentary?: boolean;
  /** Insert a separator before this item. */
  separatorBefore?: boolean;
  /**
   * If set, this button opens a horizontal flyout of toggles instead of
   * firing {@link activate}. The parent button shows active when the flyout
   * is open or any toggle inside it is active.
   */
  flyout?: ControlRailToggle[];
  /**
   * If set, this button can open a vertical panel popover hosting rich
   * controls (sliders/dropdowns), built lazily into a host element. Unlike a
   * {@link flyout} (a row of toggle chips), a popover holds arbitrary content.
   * The trigger decides how it opens relative to {@link activate}:
   * - `'click'`: primary click opens the popover ({@link activate} unused).
   * - `'context'`: right-click opens the popover; primary click still fires
   *   {@link activate} (e.g. Performance: left-click toggles the readout,
   *   right-click opens the DPR controls).
   */
  popover?: ControlRailPopover;
  /**
   * Optional per-refresh hook to sync the button's icon/label/tooltip from live
   * state (e.g. the Navigation button reflecting the current control mode).
   * Called on every {@link ControlRail} refresh with the button element.
   */
  render?: (btn: HTMLButtonElement) => void;
  /**
   * Optional predicate for a disabled (grayed, non-interactive) state — e.g.
   * the Layers button when the scene has no layers. Re-evaluated on every
   * refresh; when true the button gets the native `disabled` attribute so it
   * can't be clicked or focused. Fire a refresh (see {@link ControlRail}
   * event listeners) when the underlying condition changes.
   */
  disabled?: () => boolean;
}

/** A rich, lazily-built panel popover anchored to a {@link ControlRailItem}. */
export interface ControlRailPopover {
  /**
   * Populate the popover body. Called each time the popover opens (rebuilt
   * fresh so it always reflects live state). May return a teardown callback
   * run when the popover closes (e.g. to clear intervals / dispose a GUI).
   */
  build: (host: HTMLElement) => void | (() => void);
  /** How the popover opens: on primary click, or on right-click. */
  trigger: 'click' | 'context';
  /** Accessible label for the popover group (defaults to the item title). */
  title?: string;
}

/** A compact icon toggle shown inside a {@link ControlRailItem.flyout}. */
export interface ControlRailToggle {
  id: string;
  title: string;
  shortcut?: string;
  icon: string;
  activate: () => void;
  openSelector?: string;
  isActive?: () => boolean;
  /**
   * Exclude this toggle from the PARENT flyout button's active state (the
   * chip itself still shows active inside the flyout). For session-long
   * ambient states like fullscreen, where lighting the parent button for
   * the whole session reads as noise rather than signal — unlike cinematic
   * mode, whose glow deliberately advertises a changed render pipeline.
   */
  excludeFromParentActive?: boolean;
}
