/**
 * The settled hover pick, retained so a click can act on it (issue #1917).
 *
 * ## Why a cache rather than a fresh pick
 *
 * Picking is hover-driven: `PickingSystem` fires ~120 ms after the cursor
 * stops, reads back the GPU pick buffer asynchronously, and delivers the
 * result through a callback. There is no synchronous "pick at (x, y)".
 *
 * Adding one would not help, because of *user activation*: a browser only
 * honours `window.open` from within a short window after a real user gesture,
 * and an `await` on a GPU readback risks spending it — turning "open the link"
 * into "popup blocked". Acting on the pick the tooltip is ALREADY showing is
 * both synchronous and more honest: the click lands on the element the user
 * read the name of and decided to click.
 *
 * ## Why it is safe
 *
 * The risk is obvious — a cached pick describes a moment, and the moment can
 * pass. It is fully covered by two counters `PickingSystem` already maintains,
 * so nothing here needs its own invalidation scheme:
 *
 * | signal | advances on |
 * | --- | --- |
 * | `pickGeneration` | camera move (controls `change`), resize, persp↔ortho swap, FOV edit (`projection-changed`, #1916), layers-panel pick invalidator, any mousemove, mouseleave, dispose, and each new pick |
 * | `visibleSignature` | a layer being hidden or shown — which deliberately does NOT dirty the buffer, so the generation counter alone would miss it |
 *
 * A third check on the cursor position is belt-and-braces: real mouse movement
 * fires `mousemove`, which already advances the generation. It is kept so the
 * staleness rule is self-contained rather than resting on that invariant
 * holding elsewhere forever, and so a synthetic or coalesced event stream
 * cannot slip a click through at the wrong place.
 *
 * Note what is deliberately NOT an invalidator: `PickingSystem.suppress()`.
 * pointerdown → controls `start` → `suppress(true)` is the first half of an
 * ordinary click, so treating it as invalidating would make every click refuse
 * itself. `suppress` accordingly leaves `pickGeneration` alone, and a test
 * pins that.
 *
 * @module core/app/interaction/picked-element-cache
 */

import type * as THREE from 'three';

/**
 * How far the cursor may be from where the pick was taken, in CSS pixels, for
 * a click to still count as acting on it. Also the drag threshold that
 * separates a click from a camera gesture — the same number, because they
 * answer the same question ("did the pointer effectively stay put?").
 */
export const CLICK_SLOP_PX = 4;

/**
 * The same tolerance for a finger. A tap routinely drifts 8–15 px between
 * `pointerdown` and `pointerup`; at the mouse's 4 px every tap read as a camera
 * drag and no touch device could ever select an element.
 */
export const TOUCH_CLICK_SLOP_PX = 12;

/** The click slop for a pointer type: fingers get {@link TOUCH_CLICK_SLOP_PX}. */
export function clickSlopFor(pointerType: string): number {
  return pointerType === 'touch' ? TOUCH_CLICK_SLOP_PX : CLICK_SLOP_PX;
}

/**
 * The subset of `PickingSystem` this cache validates against. A structural
 * port, so tests can pass a plain object with two numbers.
 */
export interface PickGenerationPort {
  readonly pickGeneration: number;
  readonly visibleSignature: number;
  /**
   * Pick immediately at a viewport position (tap-to-pick). Resolves once the
   * result has been delivered to the pick consumers and their (asynchronous:
   * label fetch, then `store`) handler has finished, so a caller can read the
   * cache right after. Optional: a port without it (tests) means "act on
   * whatever the cache already holds".
   */
  pickAt?(clientX: number, clientY: number): Promise<void>;
}

/** A picked element, as much of it as a click needs. */
export interface CachedPick {
  /**
   * The hit leaf scene node. Interaction templates are resolved from its
   * `userData.attrs`, walking up to the nearest ancestor that carries one.
   */
  mainNode: THREE.Object3D;
  /** Reported layer path — outermost `kind=partition` wrapper if any. */
  nodeName: string;
  /** Scene node `elementIndex` is local to (the `part_<i>` leaf under a partition). */
  hitNodeName: string;
  /** Element index within the hit leaf. */
  elementIndex: number;
  /** The element's label, or null when it has none. */
  label: string | null;
  /** The element's machine-readable key, or null when it has none (#1917). */
  key: string | null;
  /** Canvas-local cursor position the pick was taken at, in CSS pixels. */
  screenX: number;
  screenY: number;
}

/** Retains the most recent settled pick and answers whether it is still true. */
export class PickedElementCache {
  /** The pick, plus the counters it was true under. */
  private entry: (CachedPick & { seq: number; visibleSig: number }) | null = null;

  /** Record a freshly settled pick, stamped with the current generation. */
  store(pick: CachedPick, ports: PickGenerationPort): void {
    this.entry = {
      ...pick,
      seq: ports.pickGeneration,
      visibleSig: ports.visibleSignature,
    };
  }

  /** Forget the current pick (hover cleared, session torn down). */
  clear(): void {
    this.entry = null;
  }

  /**
   * The current pick if it still describes reality, ignoring cursor position.
   *
   * For consumers that are not a click at a coordinate: the pointer-cursor
   * affordance, and the keyboard menu path (Shift+F10), which anchors at the
   * pick's own `screenX`/`screenY` because there is no cursor event to read.
   */
  peek(ports: PickGenerationPort): CachedPick | null {
    const e = this.entry;
    if (!e) return null;
    if (e.seq !== ports.pickGeneration) return null;
    if (e.visibleSig !== ports.visibleSignature) return null;
    return e;
  }

  /**
   * The current pick if it is still valid AND the given point is within
   * `slopPx` (default {@link CLICK_SLOP_PX}; a finger passes
   * {@link TOUCH_CLICK_SLOP_PX}) of where the pick was taken.
   *
   * Returns null rather than a best guess: refusing to act is always
   * recoverable (the user hovers again), whereas acting on a stale pick opens
   * a link for the wrong element, which is not.
   */
  read(
    ports: PickGenerationPort,
    screenX: number,
    screenY: number,
    slopPx: number = CLICK_SLOP_PX
  ): CachedPick | null {
    const e = this.peek(ports);
    if (!e) return null;
    const dx = screenX - e.screenX;
    const dy = screenY - e.screenY;
    if (dx * dx + dy * dy > slopPx * slopPx) return null;
    return e;
  }
}
