/**
 * Unit tests for the LoaderEventEmitter helper.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { LoaderEventEmitter } from '../../../../data/loaders';
import type { MonitorEvent } from '../../../../types/data-monitor-types';
import { log } from '../../../../utils/log';

function makeEvent(overrides: Partial<MonitorEvent> = {}): MonitorEvent {
  return {
    type: 'query',
    loader: 'point-spatial-index',
    timestamp: 0,
    data: { path: 'a/b' },
    ...overrides,
  };
}

describe('LoaderEventEmitter', () => {
  let emitter: LoaderEventEmitter;
  let logErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    emitter = new LoaderEventEmitter();
    logErrorSpy = vi.spyOn(log, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    logErrorSpy.mockRestore();
  });

  it('starts with zero listeners', () => {
    expect(emitter.size).toBe(0);
  });

  it('add() registers a listener and emit() invokes it', () => {
    const listener = vi.fn();
    emitter.add(listener);
    expect(emitter.size).toBe(1);

    const event = makeEvent();
    emitter.emit(event);
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith(event);
  });

  it('add() is idempotent — same listener registered twice runs once', () => {
    const listener = vi.fn();
    emitter.add(listener);
    emitter.add(listener);
    expect(emitter.size).toBe(1);

    emitter.emit(makeEvent());
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('remove() unsubscribes a listener', () => {
    const listener = vi.fn();
    emitter.add(listener);
    emitter.remove(listener);
    expect(emitter.size).toBe(0);

    emitter.emit(makeEvent());
    expect(listener).not.toHaveBeenCalled();
  });

  it('remove() is a no-op for a listener never registered', () => {
    expect(() => emitter.remove(() => {})).not.toThrow();
    expect(emitter.size).toBe(0);
  });

  it('emit() fans out to every listener', () => {
    const a = vi.fn();
    const b = vi.fn();
    const c = vi.fn();
    emitter.add(a);
    emitter.add(b);
    emitter.add(c);

    const event = makeEvent({ type: 'load' });
    emitter.emit(event);
    expect(a).toHaveBeenCalledWith(event);
    expect(b).toHaveBeenCalledWith(event);
    expect(c).toHaveBeenCalledWith(event);
  });

  it('a throwing listener is logged but does NOT prevent the next listeners from running', () => {
    const before = vi.fn();
    const thrower = vi.fn(() => {
      throw new Error('boom');
    });
    const after = vi.fn();
    emitter.add(before);
    emitter.add(thrower);
    emitter.add(after);

    emitter.emit(makeEvent());
    expect(before).toHaveBeenCalledTimes(1);
    expect(thrower).toHaveBeenCalledTimes(1);
    expect(after).toHaveBeenCalledTimes(1);
    expect(logErrorSpy).toHaveBeenCalledTimes(1);
  });

  it('clear() drops every listener', () => {
    const a = vi.fn();
    const b = vi.fn();
    emitter.add(a);
    emitter.add(b);

    emitter.clear();
    expect(emitter.size).toBe(0);

    emitter.emit(makeEvent());
    expect(a).not.toHaveBeenCalled();
    expect(b).not.toHaveBeenCalled();
  });
});
