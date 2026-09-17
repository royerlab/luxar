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
 * ## Touch
 *
 * A finger neither hovers nor right-clicks, so the mouse model above has no
 * touch equivalent on its own. Touch-like pointers (`isTouchLikePointer`) get:
 * a wider slop ({@link TOUCH_CLICK_SLOP_PX}); **tap** = pick at the tap
 * (`PickGenerationPort.pickAt`, the tooltip shows through the normal result
 * path) and then the click action, after first discarding residual user-input
 * damping in orbit, ortho, or fly mode so accepted drift cannot immediately
 * clear that result; any navigation is deferred by {@link DOUBLE_TAP_MS} so a
 * second tap can pre-empt it; **long-press** =
 * the element menu, after which the release is inert; **double-tap** = the
 * second tap pre-empts the first tap's navigation and picks nothing — the
 * re-frame itself is `double-tap-to-fit.ts`, installed for every scene, since
 * this module only exists once picking is provisioned. Mouse and pen-as-mouse
 * paths are untouched.
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
import { isMacPlatform } from '../../../utils/platform';
import type { EventGroup } from '../../../utils/cross-layer/event-group';
import { openContextMenu, type ContextMenuItem } from '../../../ui/overlay-widgets/context-menu';
import { showToast } from '../../../ui/toast';
import {
  CLICK_SLOP_PX,
  TOUCH_CLICK_SLOP_PX,
  type CachedPick,
  type PickedElementCache,
  type PickGenerationPort,
} from './picked-element-cache';
import { DOUBLE_TAP_MS, DOUBLE_TAP_SLOP_PX } from './double-tap-to-fit';
import { isTouchLikePointer } from '../../../utils/input-capabilities';
import { LONG_PRESS_MS } from '../../../utils/long-press';
import { resolveElementActions, type ResolvedElementActions } from './element-actions';

/**
 * Window event that asks for the element menu at the current hover, dispatched
 * by the Shift+F10 / ContextMenu keybinding.
 *
 * A `CustomEvent` rather than a threaded command because the keybinding is
 * registered once for the app's lifetime while these listeners are rebuilt on
 * every dataset load — the same decoupling {@link OPEN_DATASET_BROWSER_EVENT}
 * already uses (`input/input-handler/key-bindings/navigation-bindings.ts`).
 */
export const OPEN_ELEMENT_MENU_EVENT = 'luxar-open-element-menu';

/**
 * Window event that toggles the dataset browser modal, dispatched by the `O`
 * keybinding and the rail's dataset button; listened for by
 * `core/app/dataset/browser-shortcut.ts`.
 *
 * `luxar-` prefixed like every other window event the viewer dispatches, so
 * an embedding page's own events can never collide with it. The layer
 * contract (`.dependency-cruiser.cjs`) forbids `input/` from importing `core/`,
 * so the dispatch in `input/input-handler.ts` repeats the literal; the
 * `dataset-browser-event-sync` unit test pins the two spellings together.
 */
export const OPEN_DATASET_BROWSER_EVENT = 'luxar-open-dataset-browser';

/** Payload for the `element-click` / `element-contextmenu` embedder events. */
export interface ElementPointerPayload {
  /** Reported layer path — outermost `kind=partition` wrapper if any. */
  nodeName: string;
  /** Element index within the hit leaf. */
  elementIndex: number;
  /** Scene node `elementIndex` is local to. */
  hitNodeName: string;
  /**
   * Which pointer button: 0 = primary, 2 = secondary.
   *
   * Reports the raw DOM value, so a macOS Ctrl+primary-click — the
   * platform's secondary gesture — arrives as `0` on an
   * `element-contextmenu` event. Switch on the event NAME, not on this,
   * to tell the two gestures apart.
   */
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
  /** Discard residual camera damping when a touch release is classified as a tap. */
  settleTouchNavigation?: () => void;
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

/**
 * Default opener. `noopener,noreferrer` on every call — see the target rules.
 *
 * The return value is deliberately discarded. It is tempting to treat a `null`
 * return as "popup blocked" and surface a toast, but `window.open` returns
 * `null` whenever `noopener` is set, blocked or not — so that check would fire
 * on every successful click instead. There is no way to detect blocking here;
 * the defence against it is upstream, in keeping the click path synchronous so
 * the user activation is never spent.
 */
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

  /**
   * Resolve what a pick offers, with the `allowLinks` gate applied.
   *
   * THE single place that gate is enforced. Three call paths can surface a URL
   * — left-click, the pointer menu, the keyboard menu — plus the cursor
   * affordance, and `allowLinks: false` is a security control an embedder
   * relies on. Applying it per call site would mean stating the rule four
   * times, which is precisely where a later edit drifts and quietly reopens
   * navigation on one path.
   */
  const resolveFor = (pick: CachedPick): ResolvedElementActions => {
    const resolved = resolveElementActions(pick.mainNode, {
      label: pick.label,
      key: pick.key,
      nodeName: pick.nodeName,
      elementIndex: pick.elementIndex,
    });
    return { ...resolved, url: ports.allowLinks ? resolved.url : null };
  };

  /** As {@link resolveFor}, for the valid pick at `(x, y)` — null if there is none. */
  const actionsAt = (
    x: number,
    y: number,
    slopPx: number
  ): (ResolvedElementActions & { pick: CachedPick }) | null => {
    const pick = cache.read(picking, x, y, slopPx);
    if (!pick) return null;
    return { ...resolveFor(pick), pick };
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

    // The canvas has no tabindex, so `document.activeElement` at open time
    // is <body>; the default focus restore is therefore a no-op.
    (ports.openMenu ?? openContextMenu)({
      x: clientX,
      y: clientY,
      ariaLabel: `Actions for element ${pick.elementIndex} in ${pick.nodeName}`,
      items,
    });
  };

  // --- pointer gestures -----------------------------------------------------

  interface DownPointer {
    button: number;
    x: number;
    y: number;
    /** Finger, or pen-as-finger — routes to the tap / long-press model. */
    touchLike: boolean;
    /** A long-press already acted on this pointer; its release is inert. */
    consumed: boolean;
    /** Gesture generation captured at pointerdown. */
    gesture: number;
  }
  /** Pointers currently down. */
  const down = new Map<number, DownPointer>();
  /**
   * Whether the gesture in progress has EVER had more than one pointer down.
   *
   * Checking `down.size` at release time is not enough, and the failure is
   * asymmetric enough to be easy to miss: releasing the first of two fingers
   * leaves `down.size === 1` and correctly bails, but releasing the *last* one
   * leaves `down.size === 0`, which looks exactly like a single click. Every
   * pinch would therefore end by opening a link. This latches on the second
   * concurrent pointerdown and only clears once the canvas has no pointers at
   * all, so both release orders are rejected.
   */
  let multiTouch = false;
  let longPress: { id: number; timer: ReturnType<typeof setTimeout> } | null = null;
  let deferredNavigation: ReturnType<typeof setTimeout> | null = null;
  let lastTap: { t: number; x: number; y: number } | null = null;
  let gestureGeneration = 0;

  const clearLongPress = (): void => {
    if (longPress) {
      clearTimeout(longPress.timer);
      longPress = null;
    }
  };
  const clearDeferredNavigation = (): void => {
    if (deferredNavigation !== null) {
      clearTimeout(deferredNavigation);
      deferredNavigation = null;
    }
  };

  /**
   * Pick at the pointer's position, then run `fn` once the result has landed.
   * Without a `pickAt` on the port, act on whatever the cache already holds.
   */
  const afterPick = (clientX: number, clientY: number, gesture: number, fn: () => void): void => {
    const run = (): void => {
      if (gesture === gestureGeneration) fn();
    };
    const pending = picking.pickAt?.(clientX, clientY);
    if (pending) {
      void pending.then(run, (err: unknown) => {
        log.warning(Modules.APP, `Touch pick failed: ${err}`);
      });
    } else run();
  };

  /** The element menu for a touch gesture at a viewport position. */
  const openTouchMenu = (clientX: number, clientY: number, gesture: number): void => {
    afterPick(clientX, clientY, gesture, () => {
      const { x, y } = toCanvas({ clientX, clientY });
      const resolved = actionsAt(x, y, TOUCH_CLICK_SLOP_PX);
      if (!resolved) return;
      ports.onElementContextMenu?.(
        payloadFor(resolved.pick, 0, { clientX, clientY }, resolved.url)
      );
      openMenuFor(resolved, resolved.pick, clientX, clientY);
    });
  };

  /** Arm the long-press → menu timer for a fresh touch-like primary press. */
  const armLongPress = (ev: PointerEvent): void => {
    clearLongPress();
    const id = ev.pointerId;
    longPress = {
      id,
      timer: setTimeout(() => {
        longPress = null;
        const entry = down.get(id);
        if (!entry || multiTouch) return;
        entry.consumed = true;
        openTouchMenu(entry.x, entry.y, entry.gesture);
      }, LONG_PRESS_MS),
    };
  };

  events.on(canvas, 'pointerdown', (e) => {
    const ev = e as PointerEvent;
    const touchLike = isTouchLikePointer(ev);
    const gesture = ++gestureGeneration;
    clearDeferredNavigation();
    down.set(ev.pointerId, {
      button: ev.button,
      x: ev.clientX,
      y: ev.clientY,
      touchLike,
      consumed: false,
      gesture,
    });
    if (down.size > 1) {
      multiTouch = true;
      clearLongPress();
      return;
    }
    if (touchLike && ev.button === 0) armLongPress(ev);
  });

  // A finger that travels past the tap slop is dragging (orbiting), not holding.
  events.on(canvas, 'pointermove', (e) => {
    const ev = e as PointerEvent;
    if (!longPress || longPress.id !== ev.pointerId) return;
    const start = down.get(ev.pointerId);
    if (!start) return;
    const dx = ev.clientX - start.x;
    const dy = ev.clientY - start.y;
    if (dx * dx + dy * dy > TOUCH_CLICK_SLOP_PX * TOUCH_CLICK_SLOP_PX) clearLongPress();
  });

  /** Drop a pointer, clearing the multi-touch latch once the canvas is idle. */
  const forgetPointer = (id: number): void => {
    if (longPress?.id === id) clearLongPress();
    down.delete(id);
    if (down.size === 0) multiTouch = false;
  };
  const onPointerGone = (e: Event): void => forgetPointer((e as PointerEvent).pointerId);
  events.on(canvas, 'pointercancel', onPointerGone);
  events.on(canvas, 'pointerleave', onPointerGone);

  /** Mouse (and pen-as-mouse) click: the original synchronous model. */
  const handleClick = (ev: PointerEvent): void => {
    // macOS secondary click is Ctrl + primary button, and fires with
    // `button === 0`. Treat it as the menu gesture, matching every native app.
    const isSecondary = ev.button === 2 || (ev.button === 0 && ev.ctrlKey && isMacPlatform());

    const { x, y } = toCanvas(ev);
    const resolved = actionsAt(x, y, CLICK_SLOP_PX);
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
  };

  /**
   * Touch tap: a second tap within {@link DOUBLE_TAP_MS} / {@link DOUBLE_TAP_SLOP_PX}
   * is left to `double-tap-to-fit.ts`; otherwise pick here (desktop hover + click parity), then
   * act, deferring any navigation so a double-tap can still pre-empt it. The
   * mouse path stays synchronous so its user activation is never spent.
   */
  const handleTap = (ev: PointerEvent, gesture: number): void => {
    const now = performance.now();
    const { clientX, clientY, button } = ev;
    if (lastTap && now - lastTap.t < DOUBLE_TAP_MS) {
      const ddx = clientX - lastTap.x;
      const ddy = clientY - lastTap.y;
      if (ddx * ddx + ddy * ddy <= DOUBLE_TAP_SLOP_PX * DOUBLE_TAP_SLOP_PX) {
        // The re-frame itself is `double-tap-to-fit.ts`, installed for every
        // scene; here the second tap only pre-empts the first tap's navigation.
        lastTap = null;
        clearDeferredNavigation();
        return;
      }
    }
    lastTap = { t: now, x: clientX, y: clientY };
    afterPick(clientX, clientY, gesture, () => {
      const { x, y } = toCanvas({ clientX, clientY });
      const resolved = actionsAt(x, y, TOUCH_CLICK_SLOP_PX);
      if (!resolved) return;
      ports.onElementClick?.(payloadFor(resolved.pick, button, { clientX, clientY }, resolved.url));
      const url = resolved.url;
      if (url === null) return;
      clearDeferredNavigation();
      deferredNavigation = setTimeout(() => {
        deferredNavigation = null;
        if (gesture !== gestureGeneration) return;
        (ports.openUrl ?? defaultOpenUrl)(url, resolved.target);
      }, DOUBLE_TAP_MS);
    });
  };

  const isClickRelease = (ev: PointerEvent, start: DownPointer): boolean => {
    if (start.button !== ev.button) return false;
    if (ev.button !== 0 && ev.button !== 2) return false;
    const slop = start.touchLike ? TOUCH_CLICK_SLOP_PX : CLICK_SLOP_PX;
    const dx = ev.clientX - start.x;
    const dy = ev.clientY - start.y;
    if (dx * dx + dy * dy <= slop * slop) return true;
    if (start.touchLike) lastTap = null;
    return false;
  };

  events.on(canvas, 'pointerup', (e) => {
    const ev = e as PointerEvent;
    const start = down.get(ev.pointerId);
    // Read the latch BEFORE releasing this pointer: releasing the last finger
    // of a pinch clears it, and we still need to know this gesture was one.
    const wasMultiTouch = multiTouch;
    forgetPointer(ev.pointerId);
    if (!start || start.consumed) return;
    // Any pointer still down, or any second pointer at any point during this
    // gesture — a pinch, not a click. The latch is what makes the second half
    // true: `down.size > 0` alone rejects releasing the FIRST of two fingers
    // but not the last, which leaves the map empty and reads as a single click.
    if (down.size > 0 || wasMultiTouch) {
      lastTap = null;
      return;
    }
    if (!isClickRelease(ev, start)) return;

    if (start.touchLike) {
      ports.settleTouchNavigation?.();
      handleTap(ev, start.gesture);
    } else handleClick(ev);
  });

  // Suppress the native menu. Both controls implementations already do this,
  // in orbit and fly mode alike; kept so this module does not silently depend
  // on that remaining true.
  events.on(canvas, 'contextmenu', (e) => e.preventDefault());

  events.add(() => {
    gestureGeneration++;
    clearLongPress();
    clearDeferredNavigation();
  });

  // --- keyboard path --------------------------------------------------------

  events.on(window, OPEN_ELEMENT_MENU_EVENT, () => {
    // No cursor event to read, so anchor at the pick's own position and skip
    // the proximity check — `peek` still enforces that the pick is current.
    const pick = cache.peek(picking);
    if (!pick) return;
    const rect = canvas.getBoundingClientRect();
    openMenuFor(resolveFor(pick), pick, rect.left + pick.screenX, rect.top + pick.screenY);
  });

  // --- cursor affordance ----------------------------------------------------

  const refreshCursor = (): void => {
    const pick = cache.peek(picking);
    // `resolveFor` already nulls the url when links are disabled, so the
    // affordance follows the gate without restating it.
    const linked = pick !== null && resolveFor(pick).url !== null;
    // Only ever write our own two values. Nothing else in the viewer sets a
    // canvas cursor today, and clearing to '' restores whatever CSS says.
    //
    // Write only on an actual change. This runs on every settled pick AND
    // every clear — and clears come from `onPickResult(null)`, which fires on
    // each mousemove and on each `markDirty`, i.e. once per frame for the
    // whole of a camera drag. The compute is irrelevant (measured 0.15 us
    // cleared / 0.60 us linked per call), but an unconditional assignment is
    // a style invalidation per frame for no reason.
    const next = linked ? 'pointer' : '';
    if (canvas.style.cursor !== next) canvas.style.cursor = next;
  };

  // Leave no cursor behind when the session is torn down mid-hover.
  events.add(() => {
    canvas.style.cursor = '';
  });

  log.info(Modules.APP, 'Canvas element actions installed');
  return { refreshCursor };
}
