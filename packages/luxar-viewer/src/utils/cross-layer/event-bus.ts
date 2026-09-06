/**
 * Typed pub/sub bus for cross-layer events.
 *
 * Lower layers (data, scene, input) sometimes need to push state or
 * commands to specific UI panels — performance-monitor wants FPS
 * samples from the animation loop, data-loading-monitor wants
 * progress updates from the data manager, dimension-sliders wants a
 * "toggle me" trigger from the input handler. Direct imports would
 * violate the layer order documented in CONVENTIONS.md §10.
 *
 * This bus is the dependency-inversion target. The event map below
 * declares every cross-layer event; publishers `emit(...)` and
 * subscribers `on(...)` against it with full payload typing.
 *
 * Compared to the `notifier` (utils/cross-layer/notifier.ts) which has a
 * single backend and a fixed method dictionary:
 *   - the bus has open subscriber sets — many panels can react to
 *     the same event;
 *   - panels can be constructed late without bootstrap-order
 *     coupling — events with no listener are dropped silently;
 *   - per-event payload typing fails mistyped event names at the
 *     call site.
 *
 * @module utils/cross-layer/event-bus
 */

/**
 * Catalog of cross-layer events. Each entry maps an event name to its
 * payload type. Publishers and subscribers both type-check against
 * this map, so adding an event in one place forces the consumer to
 * adapt at the type level.
 */
export interface LuxarEventMap {
  // ── Publisher events (lower layers push state) ──────────────
  /**
   * Animation-loop frame-start hook. Fired right before per-frame
   * work begins. Subscribers (e.g., stats.js-backed
   * `PerformanceMonitor`) use this to drive their begin/end timing.
   */
  'frame-start': Record<string, never>;
  /**
   * Animation-loop frame-end hook. Fired right after per-frame
   * work — including post-processing render — completes. Pair with
   * `frame-start` for timing.
   */
  'frame-end': Record<string, never>;
  /**
   * Aggregate loading progress from the data layer. Used by the
   * loading-monitor UI to render the progress bar / spinner.
   */
  'loading-progress': {
    loaderId: string;
    loaded: number;
    total: number;
    activeQueries: number;
  };
  /**
   * A node's geometry was committed (or re-committed) to the GPU — the same
   * moment the pick buffer is invalidated. Consumers that derive something from
   * WHAT IS RESIDENT (the scene-derived environment capture in
   * `rendering/environment/`) mark themselves stale on it and rebuild once the
   * loader settles, rather than polling the scene graph.
   */
  'geometry-committed': Record<string, never>;

  // ── Command events (input → UI panel toggles) ───────────────
  /**
   * Toggle the named UI panel. Panels are internal-state-aware:
   * each subscriber decides whether to open or close based on its
   * own visibility flag, so the publisher doesn't need to know the
   * current state.
   */
  'panel-toggle': {
    panelId: 'dimension-sliders' | 'debug-console' | 'data-monitor';
  };
  /**
   * Cycle the named UI panel through its display states (e.g.,
   * hidden → mini → expanded → hidden). Distinct from `panel-toggle`
   * because the data-monitor has more than two visibility states.
   */
  'panel-cycle': { panelId: 'data-monitor' };
  /** Hide the named panel. Used by the panel-coordinator close-all flow. */
  'panel-hide': {
    panelId: 'data-monitor' | 'help-overlay';
  };
}

/**
 * Listener cleanup function returned by `on(...)`. Calling it removes
 * the subscription. Idempotent — safe to call after teardown.
 */
export type Unsubscribe = () => void;

/**
 * The typed pub/sub surface.
 */
export interface TypedEventBus<EventMap> {
  /**
   * Subscribe to `type`. Returns an unsubscribe thunk that callers
   * should store (e.g., in an EventGroup or a dispose() list).
   *
   * `replayLast: true` causes the last emitted payload for `type`,
   * if any, to fire immediately on subscription. Useful for
   * late-binding panels that need the current value (e.g., FPS
   * counter that wants to show *something* before the next frame).
   */
  on<K extends keyof EventMap>(
    type: K,
    listener: (payload: EventMap[K]) => void,
    options?: { replayLast?: boolean }
  ): Unsubscribe;

  /**
   * Emit `payload` to every subscriber of `type`. No-op if no one
   * is listening (events drop silently — that's the design).
   */
  emit<K extends keyof EventMap>(type: K, payload: EventMap[K]): void;

  /**
   * Whether at least one listener is currently subscribed to `type`.
   * Lets producers gate expensive work behind "anyone consuming?"
   * (e.g. GPU picking only runs when a `selection` listener exists).
   */
  hasListeners(type: keyof EventMap): boolean;

  /**
   * Drop all subscribers (or just those for `type`). Used by tests
   * that share the singleton between cases. Production code should
   * use the per-subscription unsubscribe instead.
   */
  clear(type?: keyof EventMap): void;
}

class EventBusImpl<EventMap> implements TypedEventBus<EventMap> {
  // Use Set so we can add/remove in O(1) without dealing with index
  // shifts during emit (a listener that unsubscribes during its own
  // call wouldn't otherwise misbehave, but Set is simpler).
  private listeners = new Map<keyof EventMap, Set<(payload: unknown) => void>>();
  private lastPayload = new Map<keyof EventMap, unknown>();

  on<K extends keyof EventMap>(
    type: K,
    listener: (payload: EventMap[K]) => void,
    options?: { replayLast?: boolean }
  ): Unsubscribe {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    const wrapped = listener as (payload: unknown) => void;
    set.add(wrapped);

    if (options?.replayLast && this.lastPayload.has(type)) {
      // Fire synchronously with the cached payload. Mirrors how the
      // listener would have seen the event if it had subscribed
      // before the emit.
      listener(this.lastPayload.get(type) as EventMap[K]);
    }

    return () => {
      this.listeners.get(type)?.delete(wrapped);
    };
  }

  emit<K extends keyof EventMap>(type: K, payload: EventMap[K]): void {
    this.lastPayload.set(type, payload);
    const set = this.listeners.get(type);
    if (!set) return;
    // Snapshot before iterating so a listener that subscribes /
    // unsubscribes during its own callback doesn't reorder the loop.
    for (const listener of [...set]) {
      listener(payload);
    }
  }

  hasListeners(type: keyof EventMap): boolean {
    return (this.listeners.get(type)?.size ?? 0) > 0;
  }

  clear(type?: keyof EventMap): void {
    if (type === undefined) {
      this.listeners.clear();
      this.lastPayload.clear();
    } else {
      this.listeners.delete(type);
      this.lastPayload.delete(type);
    }
  }
}

/**
 * Singleton bus. Most callers should use this — same shape and
 * lifetime as the singleton `notifier` from utils/cross-layer/notifier.ts.
 */
export const eventBus: TypedEventBus<LuxarEventMap> = new EventBusImpl<LuxarEventMap>();

/**
 * Construct a fresh bus instance. Useful for unit tests that want
 * isolation from the singleton, or for embedded clients that want
 * a private bus per app instance.
 */
export function createEventBus<EventMap = LuxarEventMap>(): TypedEventBus<EventMap> {
  return new EventBusImpl<EventMap>();
}
