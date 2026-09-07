// @vitest-environment jsdom
/**
 * Unit tests for canvas click / context-menu actions
 * (`core/app/interaction/canvas-actions.ts`, issue #1917).
 *
 * Two things dominate here:
 *
 * 1. **Click vs camera gesture.** Getting this wrong is the difference between
 *    a feature and an infuriating one — a link opening every time you finish
 *    an orbit. The rule is movement, never the button, because which button
 *    orbits is platform-dependent.
 * 2. **The kill switch.** `allowLinks: false` is a security control an
 *    embedder relies on, so it gets explicit coverage rather than being
 *    assumed from the resolution tests.
 *
 * `window.open`, the clipboard, the toast and the menu are all injected, so
 * nothing here opens a real popup or touches the real clipboard.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  installCanvasActions,
  OPEN_ELEMENT_MENU_EVENT,
  type CanvasActionsPorts,
} from '../../../../../core/app/interaction/canvas-actions';
import { DOUBLE_TAP_MS } from '../../../../../core/app/interaction/double-tap-to-fit';
import { LONG_PRESS_MS } from '../../../../../utils/long-press';
import {
  PickedElementCache,
  type PickGenerationPort,
} from '../../../../../core/app/interaction/picked-element-cache';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import type {
  ContextMenuItem,
  ContextMenuOptions,
} from '../../../../../ui/overlay-widgets/context-menu';
import { isMacPlatform } from '../../../../../utils/platform';
import {
  resetInputProfileForTests,
  setInputProfileOverride,
} from '../../../../../utils/input-capabilities';

vi.mock('../../../../../utils/platform', () => ({
  isMacPlatform: vi.fn(),
}));

class FakePicking implements PickGenerationPort {
  pickGeneration = 1;
  visibleSignature = 1;
  /** Tap-to-pick entry: resolved immediately, the cache is pre-stored by `setup`. */
  pickAt = vi.fn(() => Promise.resolve());
}

interface Harness {
  canvas: HTMLCanvasElement;
  cache: PickedElementCache;
  picking: FakePicking;
  events: EventGroup;
  openUrl: ReturnType<typeof vi.fn>;
  writeText: ReturnType<typeof vi.fn>;
  notify: ReturnType<typeof vi.fn>;
  openMenu: ReturnType<typeof vi.fn>;
  onElementClick: ReturnType<typeof vi.fn>;
  onElementContextMenu: ReturnType<typeof vi.fn>;
  refreshCursor: () => void;
  /** Items the last openMenu call was given. */
  menuItems(): ContextMenuItem[];
  /** Activate a menu item by (possibly elided) label prefix. */
  clickItem(prefix: string): void;
}

/**
 * Canvas is placed at viewport origin so canvas-local and client coordinates
 * coincide — jsdom reports a zero rect anyway, this makes that explicit.
 */
function setup(
  overrides: Partial<CanvasActionsPorts> = {},
  attrs: Record<string, unknown> = {}
): Harness {
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 800, height: 600 }) as DOMRect;

  const cache = new PickedElementCache();
  const picking = new FakePicking();
  const events = new EventGroup();

  const node = new THREE.Object3D();
  node.userData = { attrs };
  cache.store(
    {
      mainNode: node,
      nodeName: '/proteins',
      hitNodeName: '/proteins',
      elementIndex: 42,
      label: 'P04637',
      key: 'P04637',
      screenX: 100,
      screenY: 80,
    },
    picking
  );

  const openUrl = vi.fn();
  const writeText = vi.fn(() => Promise.resolve());
  const notify = vi.fn();
  const openMenu = vi.fn((_opts: ContextMenuOptions) => () => {});
  const onElementClick = vi.fn();
  const onElementContextMenu = vi.fn();

  const { refreshCursor } = installCanvasActions({
    canvas,
    events,
    cache,
    picking,
    allowLinks: true,
    openUrl,
    writeText,
    notify,
    openMenu: openMenu as unknown as (o: ContextMenuOptions) => () => void,
    onElementClick,
    onElementContextMenu,
    ...overrides,
  });

  const menuItems = (): ContextMenuItem[] =>
    (openMenu.mock.calls.at(-1)?.[0] as ContextMenuOptions | undefined)?.items ?? [];

  return {
    canvas,
    cache,
    picking,
    events,
    openUrl,
    writeText,
    notify,
    openMenu,
    onElementClick,
    onElementContextMenu,
    refreshCursor,
    menuItems,
    clickItem: (prefix) => {
      const item = menuItems().find((i) => i.label.startsWith(prefix));
      if (!item)
        throw new Error(
          `no menu item starting with ${prefix}; got ${menuItems()
            .map((i) => i.label)
            .join(' | ')}`
        );
      item.action?.();
    },
  };
}

/** Press and release at (x, y), optionally moving `by` pixels in between. */
function gesture(
  canvas: HTMLElement,
  opts: {
    x: number;
    y: number;
    button?: number;
    by?: number;
    ctrlKey?: boolean;
    pointerId?: number;
    pointerType?: string;
  }
): void {
  const { x, y, button = 0, by = 0, ctrlKey = false, pointerId = 1, pointerType = 'mouse' } = opts;
  canvas.dispatchEvent(
    new PointerEvent('pointerdown', {
      pointerId,
      button,
      clientX: x,
      clientY: y,
      pointerType,
      bubbles: true,
    })
  );
  canvas.dispatchEvent(
    new PointerEvent('pointerup', {
      pointerId,
      button,
      clientX: x + by,
      clientY: y,
      ctrlKey,
      pointerType,
      bubbles: true,
    })
  );
}

/** Let the `pickAt` promise chain settle. */
async function flush(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const LINKED = { link: 'https://www.uniprot.org/uniprotkb/{hover_label}/entry' };

beforeEach(() => {
  document.body.innerHTML = '';
  vi.mocked(isMacPlatform).mockReturnValue(false);
  resetInputProfileForTests();
});

afterEach(() => resetInputProfileForTests());

describe('left-click', () => {
  it('opens the resolved URL in a new tab', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.openUrl).toHaveBeenCalledExactlyOnceWith(
      'https://www.uniprot.org/uniprotkb/P04637/entry',
      '_blank'
    );
  });

  it('honours link_target="_self"', () => {
    const h = setup({}, { ...LINKED, link_target: '_self' });
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.openUrl).toHaveBeenCalledWith(expect.any(String), '_self');
  });

  it('does nothing when the element has no link', () => {
    const h = setup({}, { copy: '{hover_label}' });
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('does nothing when there is no valid pick under the cursor', () => {
    const h = setup({}, LINKED);
    h.picking.pickGeneration++; // camera moved
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.openUrl).not.toHaveBeenCalled();
    expect(h.onElementClick).not.toHaveBeenCalled();
  });
});

describe('click vs camera gesture', () => {
  it('a drag beyond the slop opens nothing', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, by: 25 });
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('a sub-slop wobble still counts as a click', () => {
    // Nobody presses a mouse button without moving it a pixel or two.
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, by: 2 });
    expect(h.openUrl).toHaveBeenCalledOnce();
  });

  /**
   * The discriminating case for the gesture threshold, found by mutation
   * testing: deleting the threshold left every existing test green.
   *
   * The cache carries its own 4 px slop, so a drag that ENDS far from the pick
   * is rejected by the cache regardless. The two guards only diverge when the
   * press starts far away and the release lands exactly ON the element — a
   * camera drag that happens to finish over a link. Only the gesture threshold
   * catches that one.
   */
  it('a drag that STARTS far away and ends on the element opens nothing', () => {
    const h = setup({}, LINKED);
    h.canvas.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 1,
        button: 0,
        clientX: 400,
        clientY: 300,
        bubbles: true,
      })
    );
    h.canvas.dispatchEvent(
      new PointerEvent('pointerup', {
        pointerId: 1,
        button: 0,
        clientX: 100, // exactly where the pick was taken
        clientY: 80,
        bubbles: true,
      })
    );
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('a right-drag opens no menu', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2, by: 25 });
    expect(h.openMenu).not.toHaveBeenCalled();
  });

  /**
   * A pinch must be rejected on BOTH release orders, and the second one is the
   * dangerous case. Releasing the first of two fingers leaves one pointer down
   * and bails on `down.size > 0`; releasing the LAST one leaves the map empty,
   * which is indistinguishable from a single click unless the gesture is
   * latched as multi-touch. Testing only the first order (as this file
   * originally did) passes against an implementation where every pinch ends by
   * opening a link.
   */
  describe('a pinch never becomes a click', () => {
    const send = (canvas: HTMLElement, type: string, pointerId: number, x: number, y: number) =>
      canvas.dispatchEvent(
        new PointerEvent(type, { pointerId, button: 0, clientX: x, clientY: y, bubbles: true })
      );

    it('when the first finger is released first', () => {
      const h = setup({}, LINKED);
      send(h.canvas, 'pointerdown', 1, 100, 80);
      send(h.canvas, 'pointerdown', 2, 300, 200);
      send(h.canvas, 'pointerup', 1, 100, 80);
      send(h.canvas, 'pointerup', 2, 300, 200);
      expect(h.openUrl).not.toHaveBeenCalled();
    });

    it('when the second finger is released first, leaving the first alone on release', () => {
      const h = setup({}, LINKED);
      send(h.canvas, 'pointerdown', 1, 100, 80);
      send(h.canvas, 'pointerdown', 2, 300, 200);
      send(h.canvas, 'pointerup', 2, 300, 200);
      send(h.canvas, 'pointerup', 1, 100, 80); // lands exactly on the element
      expect(h.openUrl).not.toHaveBeenCalled();
    });

    it('but a genuine click AFTER a pinch still works', () => {
      // The latch must clear once the canvas is idle, or the first pinch would
      // disable clicking for the rest of the session.
      const h = setup({}, LINKED);
      send(h.canvas, 'pointerdown', 1, 100, 80);
      send(h.canvas, 'pointerdown', 2, 300, 200);
      send(h.canvas, 'pointerup', 2, 300, 200);
      send(h.canvas, 'pointerup', 1, 100, 80);
      expect(h.openUrl).not.toHaveBeenCalled();

      gesture(h.canvas, { x: 100, y: 80 });
      expect(h.openUrl).toHaveBeenCalledOnce();
    });

    it('and a pointercancel mid-pinch still clears the latch', () => {
      const h = setup({}, LINKED);
      send(h.canvas, 'pointerdown', 1, 100, 80);
      send(h.canvas, 'pointerdown', 2, 300, 200);
      send(h.canvas, 'pointercancel', 1, 100, 80);
      send(h.canvas, 'pointercancel', 2, 300, 200);

      gesture(h.canvas, { x: 100, y: 80 });
      expect(h.openUrl).toHaveBeenCalledOnce();
    });
  });

  it('ignores the middle button', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 1 });
    expect(h.openUrl).not.toHaveBeenCalled();
    expect(h.openMenu).not.toHaveBeenCalled();
  });
});

describe('right-click menu', () => {
  it('offers copy, open and copy-link for a linked, labelled element', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    const labels = h.menuItems().map((i) => i.label);
    expect(labels).toEqual(['Copy "P04637"', 'Open link in new tab', 'Copy link address']);
  });

  it('says "Open link" (not "in new tab") when the target is _self', () => {
    const h = setup({}, { ...LINKED, link_target: '_self' });
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    expect(h.menuItems().map((i) => i.label)).toContain('Open link');
  });

  it('offers copy alone when there is no link', () => {
    const h = setup({}, {});
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    expect(h.menuItems().map((i) => i.label)).toEqual(['Copy "P04637"']);
  });

  it('opens no menu at all when there is nothing to offer', () => {
    // An unlabelled element with no templates — the common case in most
    // scenes. An empty menu box would be worse than none.
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 800, height: 600 }) as DOMRect;
    const cache = new PickedElementCache();
    const picking = new FakePicking();
    const node = new THREE.Object3D();
    node.userData = { attrs: {} };
    cache.store(
      {
        mainNode: node,
        nodeName: '/x',
        hitNodeName: '/x',
        elementIndex: 0,
        label: null,
        key: null,
        screenX: 10,
        screenY: 10,
      },
      picking
    );
    const openMenu = vi.fn((_opts: ContextMenuOptions) => () => {});
    installCanvasActions({
      canvas,
      events: new EventGroup(),
      cache,
      picking,
      allowLinks: true,
      openMenu: openMenu as unknown as (o: ContextMenuOptions) => () => void,
    });
    gesture(canvas, { x: 10, y: 10, button: 2 });
    expect(openMenu).not.toHaveBeenCalled();
  });

  it('macOS Ctrl+primary-click opens the menu, not the link', () => {
    vi.mocked(isMacPlatform).mockReturnValue(true);
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 0, ctrlKey: true });
    expect(h.openMenu).toHaveBeenCalledOnce();
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('non-macOS Ctrl+primary-click opens the link, not the menu', () => {
    vi.mocked(isMacPlatform).mockReturnValue(false);
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 0, ctrlKey: true });
    expect(h.openUrl).toHaveBeenCalledOnce();
    expect(h.openMenu).not.toHaveBeenCalled();
  });

  it('uses the default focus restore — the canvas has no tabindex to return to', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    const opts = h.openMenu.mock.calls.at(-1)?.[0] as ContextMenuOptions;
    expect(opts.restoreFocus).toBeUndefined();
    expect(opts.ariaLabel).toContain('42');
  });

  it('"Open link in new tab" opens the URL', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    h.clickItem('Open link');
    expect(h.openUrl).toHaveBeenCalledWith(
      'https://www.uniprot.org/uniprotkb/P04637/entry',
      '_blank'
    );
  });
});

describe('clipboard', () => {
  it('copies the resolved text and reports success', async () => {
    const h = setup({}, { copy: 'id={hover_label}' });
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    h.clickItem('Copy');
    expect(h.writeText).toHaveBeenCalledWith('id=P04637');
    await vi.waitFor(() => expect(h.notify).toHaveBeenCalledWith('Text copied'));
  });

  it('copies the link address', async () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    h.clickItem('Copy link address');
    expect(h.writeText).toHaveBeenCalledWith('https://www.uniprot.org/uniprotkb/P04637/entry');
    await vi.waitFor(() => expect(h.notify).toHaveBeenCalledWith('Link copied'));
  });

  it('reports a rejected write instead of an unhandled rejection', async () => {
    const writeText = vi.fn(() => Promise.reject(new Error('denied')));
    const h = setup({ writeText }, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    h.clickItem('Copy "');
    await vi.waitFor(() => expect(h.notify).toHaveBeenCalledWith('Could not copy text'));
  });

  it('reports an absent Clipboard API (insecure origin)', () => {
    const writeText = vi.fn(() => undefined);
    const h = setup({ writeText }, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    h.clickItem('Copy "');
    expect(h.notify).toHaveBeenCalledWith('Clipboard unavailable (needs a secure context)');
  });

  it('falls back to the bare label when no copy template is authored', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    h.clickItem('Copy "');
    expect(h.writeText).toHaveBeenCalledWith('P04637');
  });
});

describe('allowLinks: false — the embedder kill switch', () => {
  it('suppresses navigation on left-click', () => {
    const h = setup({ allowLinks: false }, LINKED);
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('drops both link menu items but keeps Copy', () => {
    // The clipboard is not navigation, so it stays.
    const h = setup({ allowLinks: false }, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    expect(h.menuItems().map((i) => i.label)).toEqual(['Copy "P04637"']);
  });

  it('still fires the embedder events, with link: null', () => {
    // A host that took exclusive control still needs to know about the click.
    const h = setup({ allowLinks: false }, LINKED);
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.onElementClick).toHaveBeenCalledWith(expect.objectContaining({ link: null }));
  });

  it('drops the link items from the KEYBOARD menu too', () => {
    // Previously its own copy of the gate, on a path with no pointer
    // feedback — the likeliest place for the rule to drift unnoticed.
    const h = setup({ allowLinks: false }, LINKED);
    window.dispatchEvent(new CustomEvent(OPEN_ELEMENT_MENU_EVENT));
    expect(h.menuItems().map((i) => i.label)).toEqual(['Copy "P04637"']);
  });

  it('leaves the cursor alone', () => {
    const h = setup({ allowLinks: false }, LINKED);
    h.refreshCursor();
    expect(h.canvas.style.cursor).toBe('');
  });
});

describe('embedder events', () => {
  it('element-click carries the element, the gesture and the resolved link', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.onElementClick).toHaveBeenCalledExactlyOnceWith({
      nodeName: '/proteins',
      hitNodeName: '/proteins',
      elementIndex: 42,
      button: 0,
      x: 100,
      y: 80,
      link: 'https://www.uniprot.org/uniprotkb/P04637/entry',
    });
    expect(h.onElementContextMenu).not.toHaveBeenCalled();
  });

  it('element-contextmenu fires on the secondary gesture', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, button: 2 });
    expect(h.onElementContextMenu).toHaveBeenCalledWith(expect.objectContaining({ button: 2 }));
    expect(h.onElementClick).not.toHaveBeenCalled();
  });
});

describe('cursor affordance', () => {
  it('becomes a pointer over a linked element and clears otherwise', () => {
    const h = setup({}, LINKED);
    h.refreshCursor();
    expect(h.canvas.style.cursor).toBe('pointer');

    h.picking.pickGeneration++; // hover moved off
    h.refreshCursor();
    expect(h.canvas.style.cursor).toBe('');
  });

  it('stays clear over an element with no link', () => {
    const h = setup({}, { copy: '{hover_label}' });
    h.refreshCursor();
    expect(h.canvas.style.cursor).toBe('');
  });

  it('is cleared when the session is disposed mid-hover', () => {
    const h = setup({}, LINKED);
    h.refreshCursor();
    expect(h.canvas.style.cursor).toBe('pointer');
    h.events.dispose();
    expect(h.canvas.style.cursor).toBe('');
  });
});

describe('keyboard path', () => {
  it('opens the menu at the pick position on the window event', () => {
    const h = setup({}, LINKED);
    window.dispatchEvent(new CustomEvent(OPEN_ELEMENT_MENU_EVENT));
    const opts = h.openMenu.mock.calls.at(-1)?.[0] as ContextMenuOptions;
    // Anchored at the PICK, not at a cursor — there is no cursor event.
    expect(opts.x).toBe(100);
    expect(opts.y).toBe(80);
  });

  it('does nothing when no pick is current', () => {
    const h = setup({}, LINKED);
    h.picking.pickGeneration++;
    window.dispatchEvent(new CustomEvent(OPEN_ELEMENT_MENU_EVENT));
    expect(h.openMenu).not.toHaveBeenCalled();
  });

  it('stops listening after the session is disposed', () => {
    const h = setup({}, LINKED);
    h.events.dispose();
    window.dispatchEvent(new CustomEvent(OPEN_ELEMENT_MENU_EVENT));
    expect(h.openMenu).not.toHaveBeenCalled();
  });
});

describe('native context menu', () => {
  it('is prevented on the canvas', () => {
    const h = setup({}, LINKED);
    const ev = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    h.canvas.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });
});

describe('touch: tap, long-press, double-tap', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('a tap picks at the tap and then opens the link (deferred past the double-tap window)', async () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, pointerType: 'touch' });
    expect(h.picking.pickAt).toHaveBeenCalledWith(100, 80);
    await flush();
    expect(h.onElementClick).toHaveBeenCalledTimes(1);
    expect(h.openUrl).not.toHaveBeenCalled(); // deferred
    vi.advanceTimersByTime(DOUBLE_TAP_MS);
    expect(h.openUrl).toHaveBeenCalledExactlyOnceWith(
      'https://www.uniprot.org/uniprotkb/P04637/entry',
      '_blank'
    );
  });

  it('a tap tolerates the finger drift a mouse click may not', async () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, by: 10 }); // mouse, 10 px > 4
    await flush();
    expect(h.onElementClick).not.toHaveBeenCalled();

    // Touch: the cache entry sits at (100, 80); the finger lands at 92 and
    // lifts at 102 — within the 12 px touch slop for both the drag test and
    // the pick-proximity test.
    gesture(h.canvas, { x: 92, y: 80, by: 10, pointerType: 'touch' });
    await flush();
    expect(h.onElementClick).toHaveBeenCalledTimes(1);
  });

  it('a pen treated as touch gets the touch drag slop', async () => {
    setInputProfileOverride('touch');
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 92, y: 80, by: 10, pointerType: 'pen' });
    await flush();
    expect(h.picking.pickAt).toHaveBeenCalledWith(102, 80);
    expect(h.onElementClick).toHaveBeenCalledTimes(1);
  });

  it('a mouse click is still synchronous and unchanged', () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80 });
    expect(h.picking.pickAt).not.toHaveBeenCalled();
    expect(h.openUrl).toHaveBeenCalledTimes(1); // no deferral
  });

  it('a long-press opens the element menu and the release is inert', async () => {
    const h = setup({}, LINKED);
    h.canvas.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 1,
        clientX: 100,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    vi.advanceTimersByTime(LONG_PRESS_MS);
    await flush();
    expect(h.openMenu).toHaveBeenCalledTimes(1);
    expect(h.onElementContextMenu).toHaveBeenCalledTimes(1);
    h.canvas.dispatchEvent(
      new PointerEvent('pointerup', {
        pointerId: 1,
        clientX: 100,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    await flush();
    vi.advanceTimersByTime(DOUBLE_TAP_MS * 2);
    expect(h.onElementClick).not.toHaveBeenCalled();
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('a finger that drags past the slop does not long-press', async () => {
    const h = setup({}, LINKED);
    h.canvas.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 1,
        clientX: 100,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    vi.advanceTimersByTime(200);
    h.canvas.dispatchEvent(
      new PointerEvent('pointermove', {
        pointerId: 1,
        clientX: 130,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    vi.advanceTimersByTime(LONG_PRESS_MS);
    await flush();
    expect(h.openMenu).not.toHaveBeenCalled();
  });

  it('a pinch (two fingers) neither taps nor long-presses', async () => {
    const h = setup({}, LINKED);
    h.canvas.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 1,
        clientX: 100,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    h.canvas.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 2,
        clientX: 300,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    vi.advanceTimersByTime(LONG_PRESS_MS);
    h.canvas.dispatchEvent(
      new PointerEvent('pointerup', {
        pointerId: 2,
        clientX: 300,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    h.canvas.dispatchEvent(
      new PointerEvent('pointerup', {
        pointerId: 1,
        clientX: 100,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    await flush();
    vi.advanceTimersByTime(DOUBLE_TAP_MS * 2);
    expect(h.openMenu).not.toHaveBeenCalled();
    expect(h.onElementClick).not.toHaveBeenCalled();
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('the second tap of a double-tap cancels the pending navigation and does not re-pick', async () => {
    // The re-frame itself lives in double-tap-to-fit.ts (installed for every
    // scene, picking or not); canvas-actions only has to stay out of its way.
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, pointerType: 'touch' });
    await flush();
    expect(h.picking.pickAt).toHaveBeenCalledTimes(1);
    expect(h.onElementClick).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(100);
    gesture(h.canvas, { x: 105, y: 84, pointerType: 'touch' });
    await flush();
    expect(h.picking.pickAt).toHaveBeenCalledTimes(1); // the second tap is not a tap
    expect(h.onElementClick).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(DOUBLE_TAP_MS * 2);
    expect(h.openUrl).not.toHaveBeenCalled(); // the first tap's navigation was cancelled
    // A third tap well after the window is a fresh single tap.
    gesture(h.canvas, { x: 100, y: 80, pointerType: 'touch' });
    await flush();
    vi.advanceTimersByTime(DOUBLE_TAP_MS);
    expect(h.openUrl).toHaveBeenCalledTimes(1);
  });

  it('a second tap invalidates the first tap even while its pick is unresolved', async () => {
    const h = setup({}, LINKED);
    const firstPick = deferred();
    h.picking.pickAt.mockReturnValueOnce(firstPick.promise);

    gesture(h.canvas, { x: 100, y: 80, pointerType: 'touch' });
    vi.advanceTimersByTime(100);
    gesture(h.canvas, { x: 104, y: 82, pointerType: 'touch' });
    firstPick.resolve();
    await flush();
    vi.advanceTimersByTime(DOUBLE_TAP_MS * 2);

    expect(h.onElementClick).not.toHaveBeenCalled();
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('a new press cancels navigation pending from the previous tap', async () => {
    const h = setup({}, LINKED);
    gesture(h.canvas, { x: 100, y: 80, pointerType: 'touch' });
    await flush();
    expect(h.onElementClick).toHaveBeenCalledTimes(1);

    h.canvas.dispatchEvent(
      new PointerEvent('pointerdown', {
        pointerId: 2,
        clientX: 100,
        clientY: 80,
        pointerType: 'touch',
        bubbles: true,
      })
    );
    vi.advanceTimersByTime(LONG_PRESS_MS);
    await flush();
    vi.advanceTimersByTime(DOUBLE_TAP_MS * 2);

    expect(h.openMenu).toHaveBeenCalledTimes(1);
    expect(h.openUrl).not.toHaveBeenCalled();
  });

  it('works against a port without pickAt (acts on the cache as-is)', async () => {
    const h = setup({}, LINKED);
    (h.picking as { pickAt?: unknown }).pickAt = undefined;
    gesture(h.canvas, { x: 100, y: 80, pointerType: 'touch' });
    await flush();
    expect(h.onElementClick).toHaveBeenCalledTimes(1);
  });
});
