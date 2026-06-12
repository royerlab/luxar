/**
 * Unit tests for the typed event bus.
 *
 * Strategy: use `createEventBus` to get a fresh bus per test so the
 * singleton stays untouched. The event-map is the production
 * `LuxarEventMap` so we exercise the actual contract.
 */

import { describe, it, expect, vi } from 'vitest';
import { createEventBus, eventBus } from '../../../../utils/cross-layer/event-bus';

// Custom test event-map — keeps tests decoupled from the production
// LuxarEventMap so renaming/restructuring real events doesn't churn
// these unit tests.
type TestMap = {
  ping: { value: number };
  pong: { value: number };
};

describe('TypedEventBus', () => {
  it('delivers an emitted event to a subscribed listener', () => {
    const bus = createEventBus<TestMap>();
    const listener = vi.fn();
    bus.on('ping', listener);
    bus.emit('ping', { value: 60 });
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith({ value: 60 });
  });

  it('drops events with no listeners (silent no-op)', () => {
    // Audit W25 fix: pin the observable contract — subscribing AFTER
    // the no-listener emit must not deliver the dropped payload (the
    // bus did not retain it). A mutant that silently buffers dropped
    // events would surface as the late listener being called.
    const bus = createEventBus<TestMap>();
    expect(() => bus.emit('ping', { value: 0 })).not.toThrow();
    const lateListener = vi.fn();
    bus.on('ping', lateListener);
    expect(lateListener).not.toHaveBeenCalled();
  });

  it('delivers the same event to every subscriber', () => {
    const bus = createEventBus<TestMap>();
    const a = vi.fn();
    const b = vi.fn();
    bus.on('ping', a);
    bus.on('ping', b);
    bus.emit('ping', { value: 30 });
    expect(a).toHaveBeenCalledTimes(1);
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('isolates listeners by event type', () => {
    const bus = createEventBus<TestMap>();
    const fpsListener = vi.fn();
    const progressListener = vi.fn();
    bus.on('ping', fpsListener);
    bus.on('pong', progressListener);

    bus.emit('ping', { value: 60 });
    expect(fpsListener).toHaveBeenCalledTimes(1);
    expect(progressListener).not.toHaveBeenCalled();
  });

  it('returns an unsubscribe thunk that removes the listener', () => {
    const bus = createEventBus<TestMap>();
    const listener = vi.fn();
    const unsubscribe = bus.on('ping', listener);

    bus.emit('ping', { value: 60 });
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    bus.emit('ping', { value: 30 });
    expect(listener).toHaveBeenCalledTimes(1); // unchanged
  });

  it('treats double unsubscribe as a no-op', () => {
    const bus = createEventBus<TestMap>();
    const listener = vi.fn();
    const unsubscribe = bus.on('ping', listener);
    unsubscribe();
    expect(() => unsubscribe()).not.toThrow();
  });

  it('replays the last emitted payload when replayLast is set', () => {
    const bus = createEventBus<TestMap>();
    bus.emit('ping', { value: 144 });

    const listener = vi.fn();
    bus.on('ping', listener, { replayLast: true });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith({ value: 144 });
  });

  it('does not replay when no payload has been emitted yet', () => {
    const bus = createEventBus<TestMap>();
    const listener = vi.fn();
    bus.on('ping', listener, { replayLast: true });
    expect(listener).not.toHaveBeenCalled();
  });

  it('snapshots subscribers before iteration so reentrant unsubscribe is safe', () => {
    const bus = createEventBus<TestMap>();
    let unsubscribeB: (() => void) | null = null;
    const a = vi.fn(() => {
      // a unsubscribes b mid-emit; b should still receive THIS event
      // (snapshot semantics) but no subsequent ones.
      unsubscribeB?.();
    });
    const b = vi.fn();
    bus.on('ping', a);
    unsubscribeB = bus.on('ping', b);

    bus.emit('ping', { value: 60 });
    expect(b).toHaveBeenCalledTimes(1);

    bus.emit('ping', { value: 30 });
    expect(b).toHaveBeenCalledTimes(1); // b unsubscribed during first emit
    expect(a).toHaveBeenCalledTimes(2);
  });

  it('clear() with no args drops every subscriber', () => {
    const bus = createEventBus<TestMap>();
    const a = vi.fn();
    const b = vi.fn();
    bus.on('ping', a);
    bus.on('pong', b);

    bus.clear();
    bus.emit('ping', { value: 60 });
    bus.emit('pong', { value: 0 });
    expect(a).not.toHaveBeenCalled();
    expect(b).not.toHaveBeenCalled();
  });

  it('clear(type) drops only that event type', () => {
    const bus = createEventBus<TestMap>();
    const a = vi.fn();
    const b = vi.fn();
    bus.on('ping', a);
    bus.on('pong', b);

    bus.clear('ping');
    bus.emit('ping', { value: 60 });
    bus.emit('pong', { value: 0 });
    expect(a).not.toHaveBeenCalled();
    expect(b).toHaveBeenCalledTimes(1);
  });

  it('clear() also drops the cached last-payload', () => {
    const bus = createEventBus<TestMap>();
    bus.emit('ping', { value: 60 });

    bus.clear('ping');

    const listener = vi.fn();
    bus.on('ping', listener, { replayLast: true });
    expect(listener).not.toHaveBeenCalled(); // no cached value to replay
  });

  it('hasListeners reflects live subscription state', () => {
    const bus = createEventBus<TestMap>();
    expect(bus.hasListeners('ping')).toBe(false);

    const off = bus.on('ping', vi.fn());
    expect(bus.hasListeners('ping')).toBe(true);
    expect(bus.hasListeners('pong')).toBe(false); // per-type, not global

    off();
    expect(bus.hasListeners('ping')).toBe(false);
  });

  it('singleton eventBus is shared across imports', async () => {
    // Spot-check: importing twice yields the same instance, and a
    // listener registered via one ref sees emits from the other.
    const { eventBus: again } = await import('../../../../utils/cross-layer/event-bus');
    expect(again).toBe(eventBus);

    const listener = vi.fn();
    const unsubscribe = eventBus.on('panel-toggle', listener);
    again.emit('panel-toggle', { panelId: 'debug-console' });
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });
});
