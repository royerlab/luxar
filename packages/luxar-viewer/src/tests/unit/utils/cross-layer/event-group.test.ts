/**
 * EventGroup tests.
 *
 * Verifies that the helper:
 * - removes registered listeners on dispose()
 * - runs cleanup callbacks in LIFO order
 * - is idempotent (second dispose() is a no-op)
 * - never lets a throwing cleanup block the rest
 * - returns an early-removal function for fine-grained control
 */

import { describe, expect, it, vi } from 'vitest';
import { EventGroup } from '../../../../utils/cross-layer/event-group';

describe('EventGroup', () => {
  it('removes registered DOM listeners on dispose()', () => {
    const target = new EventTarget();
    const group = new EventGroup();
    const handler = vi.fn();

    group.on(target, 'click', handler);

    target.dispatchEvent(new Event('click'));
    expect(handler).toHaveBeenCalledTimes(1);

    group.dispose();

    target.dispatchEvent(new Event('click'));
    expect(handler).toHaveBeenCalledTimes(1); // No new calls after dispose.
  });

  it('removes multiple listeners on different targets', () => {
    const a = new EventTarget();
    const b = new EventTarget();
    const group = new EventGroup();
    const onA = vi.fn();
    const onB = vi.fn();

    group.on(a, 'foo', onA);
    group.on(b, 'bar', onB);

    a.dispatchEvent(new Event('foo'));
    b.dispatchEvent(new Event('bar'));
    expect(onA).toHaveBeenCalledTimes(1);
    expect(onB).toHaveBeenCalledTimes(1);

    group.dispose();

    a.dispatchEvent(new Event('foo'));
    b.dispatchEvent(new Event('bar'));
    expect(onA).toHaveBeenCalledTimes(1);
    expect(onB).toHaveBeenCalledTimes(1);
  });

  it('runs arbitrary cleanup callbacks added via add()', () => {
    const group = new EventGroup();
    const c1 = vi.fn();
    const c2 = vi.fn();

    group.add(c1);
    group.add(c2);

    expect(group.size).toBe(2);

    group.dispose();

    expect(c1).toHaveBeenCalledTimes(1);
    expect(c2).toHaveBeenCalledTimes(1);
  });

  it('runs cleanups in LIFO order', () => {
    const group = new EventGroup();
    const order: string[] = [];

    group.add(() => order.push('a'));
    group.add(() => order.push('b'));
    group.add(() => order.push('c'));

    group.dispose();

    // LIFO: c, b, a
    expect(order).toEqual(['c', 'b', 'a']);
  });

  it('is idempotent: second dispose() is a no-op', () => {
    const group = new EventGroup();
    const cleanup = vi.fn();

    group.add(cleanup);

    group.dispose();
    group.dispose();

    expect(cleanup).toHaveBeenCalledTimes(1);
    expect(group.size).toBe(0);
  });

  it('keeps running remaining cleanups when one throws', () => {
    const group = new EventGroup();
    const before = vi.fn();
    const after = vi.fn();

    // Order at registration: before, throws, after.
    // LIFO at dispose: after, throws, before.
    group.add(before);
    group.add(() => {
      throw new Error('boom');
    });
    group.add(after);

    group.dispose();

    expect(before).toHaveBeenCalledTimes(1);
    expect(after).toHaveBeenCalledTimes(1);
  });

  it('on() returns an early-remove function', () => {
    const target = new EventTarget();
    const group = new EventGroup();
    const handler = vi.fn();

    const remove = group.on(target, 'tick', handler);

    target.dispatchEvent(new Event('tick'));
    expect(handler).toHaveBeenCalledTimes(1);

    remove();

    target.dispatchEvent(new Event('tick'));
    expect(handler).toHaveBeenCalledTimes(1);

    // Subsequent dispose() should not double-remove.
    group.dispose();
    target.dispatchEvent(new Event('tick'));
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('early-remove splices the cleanup out so size shrinks', () => {
    // The returned cleanup function must splice its entry out of
    // `cleanups`, not just flip a `removed` flag — otherwise a
    // long-lived EventGroup that registers + early-removes many
    // listeners leaks no-op closures.
    const target = new EventTarget();
    const group = new EventGroup();
    const remove1 = group.on(target, 'tick', vi.fn());
    group.on(target, 'tick', vi.fn());
    group.on(target, 'tick', vi.fn());

    expect(group.size).toBe(3);

    remove1();

    expect(group.size).toBe(2);
  });
});
