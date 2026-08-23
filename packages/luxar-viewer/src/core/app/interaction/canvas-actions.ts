/**
 * Canvas pointer + keyboard actions on the picked element (issue #1917).
 *
 * Left-click opens the element's authored `link`; right-click opens the shared
 * context menu with `Copy`, `Open link in new tab` and `Copy link address`.
 * Before this, nothing in the viewer listened for `click`, `contextmenu`,
 * `mousedown` or `pointerdown` on the canvas at all — both gestures were
 * swallowed by the camera controls and replaced by nothing.
 *
 * ## Telling a click from a camera gesture
 *
 * Movement, never the button. Which button orbits and which pans is
 * platform-dependent — `naturalDrag` defaults to `isMacPlatform()`, and the
 * ortho controls map RIGHT to nothing at all — so any button-based rule would
 * be wrong on half the installs. A press that moves no further than
 * {@link CLICK_SLOP_PX} is a click, whatever gesture it would otherwise have
 * begun; anything further is left to the controls, which have been receiving
 * it all along.
 *
 * The menu opens on `pointerup`, NOT on the `contextmenu` event: on macOS that
 * event fires at press time, so a right-drag to rotate would pop a menu the
 * instant the drag began. A `contextmenu` listener is still registered, purely
 * to `preventDefault` the native menu — the controls already do this in both
 * orbit and fly mode, so it is belt-and-braces against that changing.
 *
 * ## Dependency injection
 *
 * `window.open`, the clipboard, the toast and the menu all arrive as ports.
 * `window.open` in particular is called nowhere else in the viewer and is
 * stubbed in no existing test, so injecting it is the only way the URL a click
 * would navigate to can be asserted without a real popup.
 *
 * @module core/app/interaction/canvas-actions
 */

import { log, Modules } from '../../../utils/log';
import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { openContextMenu, type ContextMenuItem } from '../../../ui/overlay-widgets/context-menu';
import { showToast } from '../../../ui/toast';
import {
  CLICK_SLOP_PX,
  type PickedElementCache,
  type PickGenerationPort,
} from './picked-element-cache';
import { resolveElementActions, type ResolvedElementActions } from './element-actions';

/**
 * Window event that asks for the element menu at the current hover, dispatched
 * by the Shift+F10 / ContextMenu keybinding.
 *
 * A `CustomEvent` rather than a threaded command because the keybinding is
 * registered once for the app's lifetime while these listeners are rebuilt on
 * every dataset load — the same decoupling `open-dataset-browser` already uses
 * (`input/input-handler/key-bindings/navigation-bindings.ts`).
 */
export const OPEN_ELEMENT_MENU_EVENT = 'luxar-open-element-menu';

/** Payload for the `element-click` / `element-contextmenu` embedder events. */
export interface ElementPointerPayload {
  /** Reported layer path — outermost `kind=partition` wrapper if any. */
  nodeName: string;
  /** Element index within the hit leaf. */
  elementIndex: number;
  /** Scene node `elementIndex` is local to. */
  hitNodeName: string;
  /** Which pointer button: 0 = primary, 2 = secondary. */
  button: number;
  /** Viewport coordinates of the gesture, in CSS pixels. */
  x: number;
  y: number;
  /**
   * The URL the built-in handler resolved, or null when the element has none
   * (or link opening is disabled). Reported so a host can mirror or override
   * the behaviour without re-implementing template resolution.
   */
  link: string | null;
}

export interface CanvasActionsPorts {
  canvas: HTMLElement;
  /** Session event group — one `dispose()` removes every listener below. */
  events: EventGroup;
  cache: PickedElementCache;
  picking: PickGenerationPort;
  /**
   * Whether links may be opened at all (`allowLinks` option / `?no-links`).
   * When false: no navigation, no link menu items, no pointer cursor — but
   * `Copy` still works, because writing to the clipboard is not navigation.
   */
  allowLinks: boolean;
  /** Injected for testability; defaults to `window.open`. */
  openUrl?: (url: string, target: string) => void;
  /** Injected for testability; defaults to `navigator.clipboard.writeText`. */
  writeText?: (text: string) => Promise<void> | undefined;
  /** Injected for testability; defaults to `showToast`. */
  notify?: (message: string) => void;
  /** Injected for testability; defaults to `openContextMenu`. */
  openMenu?: typeof openContextMenu;
  onElementClick?: (payload: ElementPointerPayload) => void;
  onElementContextMenu?: (payload: ElementPointerPayload) => void;
}

/** Handle returned by {@link installCanvasActions}. */
export interface CanvasActionsHandle {
  /**
   * Re-evaluate the canvas cursor. Called whenever the settled pick changes,
   * so the pointer affordance appears and disappears with the tooltip.
   */
  refreshCursor(): void;
}

/** How much of the copy string to show in the menu item before eliding. */
const MENU_LABEL_MAX = 40;

function elide(text: string): string {
  const oneLine = text.replace(/\s+/g, ' ').trim();
  return oneLine.length <= MENU_LABEL_MAX ? oneLine : `${oneLine.slice(0, MENU_LABEL_MAX - 1)}…`;
}

/** Default opener. `noopener,noreferrer` on every call — see the target rules. */
function defaultOpenUrl(url: string, target: string): void {
  window.open(url, target, 'noopener,noreferrer');
}

function defaultWriteText(text: string): Promise<void> | undefined {
  return navigator.clipboard?.writeText(text);
}

/**
 * Copy `text`, reporting the outcome either way.
 *
 * The Clipboard API is absent on insecure origins and its write can reject on
 * a permission denial; both must surface as feedback rather than as silence
 * and an unhandled rejection. Mirrors `ui/layers/layers-panel.ts`.
 */
function copyWithFeedback(ports: CanvasActionsPorts, text: string, what: string): void {
  const notify = ports.notify ?? showToast;
  const write = (ports.writeText ?? defaultWriteText)(text);
  if (!write) {
    notify('Clipboard unavailable (needs a secure context)');
    return;
  }
  write.then(
    () => notify(`${what} copied`),
    () => notify(`Could not copy ${what.toLowerCase()}`)
  );
}

/** Wire canvas pointer + keyboard actions. All listeners go through `events`. */
export function installCanvasActions(ports: CanvasActionsPorts): CanvasActionsHandle {
  const { canvas, events, cache, picking } = ports;

  /** Pointers currently down, so a pinch is never mistaken for a click. */
  const down = new Map<number, { button: number; x: number; y: number }>();

  /**
   * Resolve what the currently-valid pick offers, honouring `allowLinks`.
   * Returns null when there is no valid pick at `(x, y)`.
   */
  const actionsAt = (
    x: number,
    y: number
  ): (ResolvedElementActions & { pick: NonNullable<ReturnType<typeof cache.read>> }) | null => {
    const pick = cache.read(picking, x, y);
    if (!pick) return null;
    const resolved = resolveElementActions(pick.mainNode, {
      label: pick.label,
      nodeName: pick.nodeName,
      elementIndex: pick.elementIndex,
    });
    return {
      ...resolved,
      url: ports.allowLinks ? resolved.url : null,
      pick,
    };
  };

  const payloadFor = (
    pick: { nodeName: string; elementIndex: number; hitNodeName: string },
    button: number,
    ev: { clientX: number; clientY: number },
    link: string | null
  ): ElementPointerPayload => ({
    nodeName: pick.nodeName,
    elementIndex: pick.elementIndex,
    hitNodeName: pick.hitNodeName,
    button,
    x: ev.clientX,
    y: ev.clientY,
    link,
  });

  /** Canvas-local coordinates, matching the space picks are recorded in. */
  const toCanvas = (ev: { clientX: number; clientY: number }): { x: number; y: number } => {
    const r = canvas.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  };

  const openMenuFor = (
    resolved: ResolvedElementActions,
    pick: { nodeName: string; elementIndex: number },
    clientX: number,
    clientY: number
  ): void => {
    const items: ContextMenuItem[] = [];

    if (resolved.copyText !== null) {
      const text = resolved.copyText;
      items.push({
        label: `Copy "${elide(text)}"`,
        action: () => copyWithFeedback(ports, text, 'Text'),
      });
    }
    if (resolved.url !== null) {
      const url = resolved.url;
      items.push({
        // `_self` navigates this tab, so promising "in new tab" would lie.
        label: resolved.target === '_self' ? 'Open link' : 'Open link in new tab',
        separatorBefore: items.length > 0,
        action: () => (ports.openUrl ?? defaultOpenUrl)(url, resolved.target),
      });
      items.push({
        label: 'Copy link address',
        action: () => copyWithFeedback(ports, url, 'Link'),
      });
    }

    // Nothing to offer → no menu at all, rather than an empty box. A layer
    // with neither templates nor labels is the common case in most scenes.
    if (items.length === 0) return;

    (ports.openMenu ?? openContextMenu)({
      x: clientX,
      y: clientY,
      ariaLabel: `Actions for element ${pick.elementIndex} in ${pick.nodeName}`,
      items,
      // The canvas has no tabindex, so `document.activeElement` at open time
      // is <body> and focusing it back is a no-op. Say so explicitly rather
      // than leaning on the default doing nothing useful.
      restoreFocus: null,
    });
  };

  // --- pointer gestures -----------------------------------------------------

  events.on(canvas, 'pointerdown', (e) => {
    const ev = e as PointerEvent;
    down.set(ev.pointerId, { button: ev.button, x: ev.clientX, y: ev.clientY });
  });

  const forgetPointer = (e: Event): void => {
    down.delete((e as PointerEvent).pointerId);
  };
  events.on(canvas, 'pointercancel', forgetPointer);
  events.on(canvas, 'pointerleave', forgetPointer);

  events.on(canvas, 'pointerup', (e) => {
    const ev = e as PointerEvent;
    const start = down.get(ev.pointerId);
    down.delete(ev.pointerId);
    if (!start) return;
    // A second pointer was down at some point in this gesture — a pinch, not
    // a click.
    if (down.size > 0) return;
    if (start.button !== ev.button) return;

    const dx = ev.clientX - start.x;
    const dy = ev.clientY - start.y;
    if (dx * dx + dy * dy > CLICK_SLOP_PX * CLICK_SLOP_PX) return; // a drag

    // macOS secondary click is Ctrl + primary button, and fires with
    // `button === 0`. Treat it as the menu gesture, matching every native app.
    const isSecondary = ev.button === 2 || (ev.button === 0 && ev.ctrlKey);
    if (ev.button !== 0 && ev.button !== 2) return;

    const { x, y } = toCanvas(ev);
    const resolved = actionsAt(x, y);
    if (!resolved) return;

    const payload = payloadFor(resolved.pick, ev.button, ev, resolved.url);
    if (isSecondary) {
      ports.onElementContextMenu?.(payload);
      openMenuFor(resolved, resolved.pick, ev.clientX, ev.clientY);
    } else {
      ports.onElementClick?.(payload);
      if (resolved.url !== null) {
        (ports.openUrl ?? defaultOpenUrl)(resolved.url, resolved.target);
      }
    }
  });

  // Suppress the native menu. Both controls implementations already do this,
  // in orbit and fly mode alike; kept so this module does not silently depend
  // on that remaining true.
  events.on(canvas, 'contextmenu', (e) => e.preventDefault());

  // --- keyboard path --------------------------------------------------------

  events.on(window, OPEN_ELEMENT_MENU_EVENT, () => {
    // No cursor event to read, so anchor at the pick's own position and skip
    // the proximity check — `peek` still enforces that the pick is current.
    const pick = cache.peek(picking);
    if (!pick) return;
    const resolved = resolveElementActions(pick.mainNode, {
      label: pick.label,
      nodeName: pick.nodeName,
      elementIndex: pick.elementIndex,
    });
    const rect = canvas.getBoundingClientRect();
    openMenuFor(
      { ...resolved, url: ports.allowLinks ? resolved.url : null },
      pick,
      rect.left + pick.screenX,
      rect.top + pick.screenY
    );
  });

  // --- cursor affordance ----------------------------------------------------

  const refreshCursor = (): void => {
    if (!ports.allowLinks) return;
    const pick = cache.peek(picking);
    let linked = false;
    if (pick) {
      const { url } = resolveElementActions(pick.mainNode, {
        label: pick.label,
        nodeName: pick.nodeName,
        elementIndex: pick.elementIndex,
      });
      linked = url !== null;
    }
    // Only ever write our own two values. Nothing else in the viewer sets a
    // canvas cursor today, and clearing to '' restores whatever CSS says.
    canvas.style.cursor = linked ? 'pointer' : '';
  };

  // Leave no cursor behind when the session is torn down mid-hover.
  events.add(() => {
    canvas.style.cursor = '';
  });

  log.info(Modules.APP, 'Canvas element actions installed');
  return { refreshCursor };
}
