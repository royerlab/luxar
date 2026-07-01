/**
 * ControlRail — a slim, always-visible vertical activity rail docked to the
 * left edge of the viewer.
 *
 * Luxar's UI is otherwise entirely keyboard-triggered (H, N, R, L, P, O, …),
 * which means a first-time visitor sees a bare canvas with no hint that any
 * controls exist. The rail fixes that: one discoverable, recognizable icon per
 * panel, each wired to the *exact same* command the keyboard shortcut fires
 * (via {@link InputHandler.getUiActions}), with a tooltip showing the shortcut.
 *
 * Design notes:
 * - Matches the viewer design tokens (`--luxar-*`), so it themes for all four
 *   themes automatically.
 * - Idle-dims after a few seconds of no pointer movement so it recedes into the
 *   immersive canvas, and wakes on pointer movement / hover.
 * - Reflects live open/closed state per panel (active highlight).
 * - Shows a one-time first-run hint (localStorage-gated) pointing at itself.
 *
 * @module ui/control-rail
 */

import { getViewerContainer } from '../utils/viewer-container';

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
}

const HINT_STORAGE_KEY = 'luxar-control-rail-hint-dismissed';
const COLLAPSED_STORAGE_KEY = 'luxar-control-rail-collapsed';
const IDLE_MS = 2600;
/** Collapsed handle lingers a little longer, then fades to barely-visible. */
const COLLAPSED_IDLE_MS = 5000;
const REFRESH_MS = 400;

/** Chevron used by the collapse/expand handle (rotated via CSS when collapsed). */
const CHEVRON_ICON =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 6l-6 6 6 6"/></svg>';

export class ControlRail {
  private readonly root: HTMLDivElement;
  private readonly container: HTMLElement;
  private readonly items: ControlRailItem[];
  private readonly buttons = new Map<string, HTMLButtonElement>();
  private hint?: HTMLDivElement;
  private idleTimer?: number;
  private refreshTimer?: number;
  private disposed = false;
  private collapsed = false;
  private readonly onPointerMove = (): void => this.wake();
  private readonly onFullscreenChange = (): void => this.syncFullscreen();

  constructor(items: ControlRailItem[]) {
    this.items = items;
    this.container = getViewerContainer();

    this.root = document.createElement('div');
    this.root.className = 'luxar-control-rail is-awake';
    this.root.setAttribute('role', 'toolbar');
    this.root.setAttribute('aria-label', 'Viewer controls');
    this.root.setAttribute('aria-orientation', 'vertical');

    for (const item of items) {
      if (item.separatorBefore) {
        const sep = document.createElement('div');
        sep.className = 'luxar-control-rail__sep';
        this.root.appendChild(sep);
      }
      this.root.appendChild(this.buildButton(item));
    }

    // Collapse/expand handle — always visible so a collapsed rail can always
    // be brought back (never re-hides discoverability).
    this.root.appendChild(this.buildCollapseButton());

    // Don't steal keyboard focus from the canvas/body when the rail is
    // clicked with the mouse — otherwise global shortcuts that are gated on
    // canvas/body focus (e.g. Space = fullscreen) would stop working after
    // any rail interaction. The click still fires; keyboard Tab focus (which
    // doesn't go through mousedown) is unaffected, so the rail stays operable
    // and focus-visible for keyboard users.
    this.root.addEventListener('mousedown', (e) => e.preventDefault());

    this.container.appendChild(this.root);
    // Marker class lets left-anchored panels offset to clear the rail.
    this.container.classList.add('luxar-has-control-rail');

    // Restore persisted collapsed state.
    let startCollapsed = false;
    try {
      startCollapsed = localStorage.getItem(COLLAPSED_STORAGE_KEY) === '1';
    } catch {
      /* ignore */
    }
    this.setCollapsed(startCollapsed, false);

    // Idle-dim behaviour.
    this.container.addEventListener('pointermove', this.onPointerMove);
    this.root.addEventListener('pointerenter', this.onPointerMove);
    // In fullscreen the rail hides and only reveals on hover; restore on exit.
    document.addEventListener('fullscreenchange', this.onFullscreenChange);
    this.syncFullscreen();
    this.scheduleSleep();

    // Reflect live panel open/closed state.
    this.refresh();
    this.refreshTimer = window.setInterval(() => this.refresh(), REFRESH_MS);

    if (!startCollapsed) this.maybeShowHint();
  }

  private buildCollapseButton(): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'luxar-control-rail__btn luxar-control-rail__collapse';
    btn.setAttribute('aria-label', 'Collapse controls');
    btn.innerHTML = CHEVRON_ICON;
    const tip = document.createElement('span');
    tip.className = 'luxar-control-rail__tip';
    tip.setAttribute('role', 'tooltip');
    tip.textContent = 'Hide controls';
    btn.appendChild(tip);
    btn.addEventListener('click', () => this.setCollapsed(!this.collapsed, true));
    return btn;
  }

  /** Collapse to just the handle, or expand back to the full rail. */
  setCollapsed(collapsed: boolean, persist = true): void {
    this.collapsed = collapsed;
    this.root.classList.toggle('is-collapsed', collapsed);
    const handle = this.root.querySelector<HTMLButtonElement>('.luxar-control-rail__collapse');
    if (handle) {
      handle.setAttribute('aria-label', collapsed ? 'Show controls' : 'Collapse controls');
      const tip = handle.querySelector('.luxar-control-rail__tip');
      if (tip) tip.textContent = collapsed ? 'Show controls' : 'Hide controls';
    }
    if (collapsed) this.dismissHint();
    if (persist) {
      try {
        localStorage.setItem(COLLAPSED_STORAGE_KEY, collapsed ? '1' : '0');
      } catch {
        /* ignore */
      }
    }
    this.wake();
  }

  private buildButton(item: ControlRailItem): HTMLButtonElement {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'luxar-control-rail__btn';
    btn.dataset.railId = item.id;
    const label = item.shortcut ? `${item.title} (${item.shortcut})` : item.title;
    btn.setAttribute('aria-label', label);
    btn.innerHTML = item.icon;

    const tip = document.createElement('span');
    tip.className = 'luxar-control-rail__tip';
    tip.setAttribute('role', 'tooltip');
    tip.innerHTML = item.shortcut
      ? `${escapeHtml(item.title)}<kbd>${escapeHtml(item.shortcut)}</kbd>`
      : escapeHtml(item.title);
    btn.appendChild(tip);

    btn.addEventListener('click', () => {
      this.dismissHint();
      try {
        item.activate();
      } catch {
        /* a panel toggle throwing must not break the rail */
      }
      // The panel toggles synchronously-ish; reflect state on the next tick.
      window.setTimeout(() => this.refresh(), 30);
    });

    this.buttons.set(item.id, btn);
    return btn;
  }

  private wake(): void {
    if (this.disposed) return;
    this.root.classList.add('is-awake');
    this.scheduleSleep();
  }

  private scheduleSleep(): void {
    if (this.idleTimer) window.clearTimeout(this.idleTimer);
    const delay = this.collapsed ? COLLAPSED_IDLE_MS : IDLE_MS;
    this.idleTimer = window.setTimeout(() => {
      // Stay awake while the pointer is over the rail itself.
      if (this.root.matches(':hover')) {
        this.scheduleSleep();
        return;
      }
      this.root.classList.remove('is-awake');
    }, delay);
  }

  /** Reflect fullscreen state — the rail hides (hover-to-reveal) in fullscreen. */
  private syncFullscreen(): void {
    if (this.disposed) return;
    this.root.classList.toggle('is-fullscreen', !!document.fullscreenElement);
    this.wake();
  }

  private refresh(): void {
    if (this.disposed) return;
    for (const item of this.items) {
      if (item.momentary) continue;
      const btn = this.buttons.get(item.id);
      if (!btn) continue;
      let active = false;
      if (item.isActive) {
        try {
          active = !!item.isActive();
        } catch {
          active = false;
        }
      } else if (item.openSelector) {
        active = isPanelVisible(item.openSelector, this.container);
      }
      btn.classList.toggle('is-active', active);
    }
  }

  private maybeShowHint(): void {
    let seen = false;
    try {
      seen = localStorage.getItem(HINT_STORAGE_KEY) === '1';
    } catch {
      /* private mode / storage blocked — just show it */
    }
    if (seen) return;

    const hint = document.createElement('div');
    hint.className = 'luxar-control-rail-hint';
    hint.innerHTML =
      '<button class="luxar-control-rail-hint__close" type="button" aria-label="Dismiss">&times;</button>' +
      '<b>New here?</b><br>Hover these controls, or press <kbd>H</kbd> — dimensions, rendering, layers &amp; more.';
    hint
      .querySelector('.luxar-control-rail-hint__close')
      ?.addEventListener('click', () => this.dismissHint());
    this.container.appendChild(hint);
    this.hint = hint;
  }

  private dismissHint(): void {
    if (!this.hint) return;
    try {
      localStorage.setItem(HINT_STORAGE_KEY, '1');
    } catch {
      /* ignore */
    }
    this.hint.remove();
    this.hint = undefined;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.idleTimer) window.clearTimeout(this.idleTimer);
    if (this.refreshTimer) window.clearInterval(this.refreshTimer);
    this.container.removeEventListener('pointermove', this.onPointerMove);
    document.removeEventListener('fullscreenchange', this.onFullscreenChange);
    this.container.classList.remove('luxar-has-control-rail');
    this.hint?.remove();
    this.root.remove();
    this.buttons.clear();
  }
}

/** A panel counts as "open" when its root element is present and rendered. */
function isPanelVisible(selector: string, scope: ParentNode): boolean {
  const el = scope.querySelector(selector) ?? document.querySelector(selector);
  if (!(el instanceof HTMLElement)) return false;
  if (el.getClientRects().length === 0) return false;
  const s = getComputedStyle(el);
  return s.visibility !== 'hidden' && Number(s.opacity) > 0.01;
}

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"]/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c] ?? c
  );
}

/**
 * Default icon set (inline SVG, currentColor stroke) keyed by rail item id,
 * kept here so the pipeline that wires actions stays readable.
 */
export const RAIL_ICONS: Record<string, string> = {
  help: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.8.4-1 .9-1 1.7"/><line x1="12" y1="17" x2="12" y2="17.01"/></svg>',
  dims: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 3v18M4 7.5l8 4.5 8-4.5"/></svg>',
  render:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><line x1="4" y1="7" x2="20" y2="7"/><circle cx="9" cy="7" r="2.2"/><line x1="4" y1="17" x2="20" y2="17"/><circle cx="15" cy="17" r="2.2"/></svg>',
  layers:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
  perf: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18a8 8 0 0 1 16 0"/><line x1="12" y1="18" x2="16" y2="11"/></svg>',
  data: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h5l2 2h9v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7z"/></svg>',
  monitor:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l2 6 4-13 2 7h6"/></svg>',
  recording:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/></svg>',
  screenshot:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h3l1.5-2h7L17 8h3v11H4z"/><circle cx="12" cy="13" r="3.2"/></svg>',
  logs: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3"/><line x1="13" y1="15" x2="17" y2="15"/></svg>',
};
