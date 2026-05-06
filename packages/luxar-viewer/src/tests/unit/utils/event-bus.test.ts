/**
 * Unit tests for the typed event bus.
 *
 * Strategy: use `createEventBus` to get a fresh bus per test so the
 * singleton stays untouched. The event-map is the production
 * `LuxarEventMap` so we exercise the actual contract.
 */

import { describe, it, expect, vi } from 'vitest';
import { createEventBus, eventBus } from '../../../utils/event-bus';

describe('TypedEventBus', () => {
  it('delivers an emitted event to a subscribed listener', () => {
    const bus = createEventBus();
    const listener = vi.fn();
    bus.on('fps-sample', listener);
    bus.emit('fps-sample', { fps: 60, frameTimeMs: 16.7 });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith({ fps: 60, frameTimeMs: 16.7 });
  });

  it('drops events with no listeners (silent no-op)', () => {
    const bus = createEventBus();
    expect(() =>
      bus.emit('fps-sample', { fps: 0, frameTimeMs: 0 })
    ).not.toThrow();
  });

  it('delivers the same event to every subscriber', () => {
    const bus = createEventBus();
    const a = vi.fn();
    const b = vi.fn();
    bus.on('fps-sample', a);
    bus.on('fps-sample', b);
    bus.emit('fps-sample', { fps: 30, frameTimeMs: 33.3 });
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('isolates listeners by event type', () => {
    const bus = createEventBus();
    const fpsListener = vi.fn();
    const progressListener = vi.fn();
    bus.on('fps-sample', fpsListener);
    bus.on('loading-progress', progressListener);

    bus.emit('fps-sample', { fps: 60, frameTimeMs: 16.7 });
    expect(fpsListener).toHaveBeenCalledTimes(1);
    expect(progressListener).not.toHaveBeenCalled();
  });

  it('returns an unsubscribe thunk that removes the listener', () => {
    const bus = createEventBus();
    const listener = vi.fn();
    const unsubscribe = bus.on('fps-sample', listener);

    bus.emit('fps-sample', { fps: 60, frameTimeMs: 16.7 });
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    bus.emit('fps-sample', { fps: 30, frameTimeMs: 33.3 });
    expect(listener).toHaveBeenCalledTimes(1); // unchanged
  });

  it('treats double unsubscribe as a no-op', () => {
    const bus = createEventBus();
    const listener = vi.fn();
    const unsubscribe = bus.on('fps-sample', listener);
    unsubscribe();
    expect(() => unsubscribe()).not.toThrow();
  });

  it('replays the last emitted payload when replayLast is set', () => {
    const bus = createEventBus();
    bus.emit('fps-sample', { fps: 144, frameTimeMs: 6.94 });

    const listener = vi.fn();
    bus.on('fps-sample', listener, { replayLast: true });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith({ fps: 144, frameTimeMs: 6.94 });
  });

  it('does not replay when no payload has been emitted yet', () => {
    const bus = createEventBus();
    const listener = vi.fn();
    bus.on('fps-sample', listener, { replayLast: true });
    expect(listener).not.toHaveBeenCalled();
  });

  it('snapshots subscribers before iteration so reentrant unsubscribe is safe', () => {
    const bus = createEventBus();
    let unsubscribeB: (() => void) | null = null;
    const a = vi.fn(() => {
      // a unsubscribes b mid-emit; b should still receive THIS event
      // (snapshot semantics) but no subsequent ones.
      unsubscribeB?.();
    });
    const b = vi.fn();
    bus.on('fps-sample', a);
    unsubscribeB = bus.on('fps-sample', b);

    bus.emit('fps-sample', { fps: 60, frameTimeMs: 16.7 });
    expect(b).toHaveBeenCalledTimes(1);

    bus.emit('fps-sample', { fps: 30, frameTimeMs: 33.3 });
    expect(b).toHaveBeenCalledTimes(1); // b unsubscribed during first emit
    expect(a).toHaveBeenCalledTimes(2);
  });

  it('clear() with no args drops every subscriber', () => {
    const bus = createEventBus();
    const a = vi.fn();
    const b = vi.fn();
    bus.on('fps-sample', a);
    bus.on('loading-progress', b);

    bus.clear();
    bus.emit('fps-sample', { fps: 60, frameTimeMs: 16.7 });
    bus.emit('loading-progress', {
      loaderId: 'x',
      loaded: 0,
      total: 0,
      activeQueries: 0,
    });
    expect(a).not.toHaveBeenCalled();
    expect(b).not.toHaveBeenCalled();
  });

  it('clear(type) drops only that event type', () => {
    const bus = createEventBus();
    const a = vi.fn();
    const b = vi.fn();
    bus.on('fps-sample', a);
    bus.on('loading-progress', b);

    bus.clear('fps-sample');
    bus.emit('fps-sample', { fps: 60, frameTimeMs: 16.7 });
    bus.emit('loading-progress', {
      loaderId: 'x',
      loaded: 0,
      total: 0,
      activeQueries: 0,
    });
    expect(a).not.toHaveBeenCalled();
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('clear() also drops the cached last-payload', () => {
    const bus = createEventBus();
    bus.emit('fps-sample', { fps: 60, frameTimeMs: 16.7 });

    bus.clear('fps-sample');

    const listener = vi.fn();
    bus.on('fps-sample', listener, { replayLast: true });
    expect(listener).not.toHaveBeenCalled(); // no cached value to replay
  });

  it('singleton eventBus is shared across imports', async () => {
    // Spot-check: importing twice yields the same instance, and a
    // listener registered via one ref sees emits from the other.
    const { eventBus: again } = await import('../../../utils/event-bus');
    expect(again).toBe(eventBus);

    const listener = vi.fn();
    const unsubscribe = eventBus.on('panel-toggle', listener);
    again.emit('panel-toggle', { panelId: 'debug-console' });
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });
});
