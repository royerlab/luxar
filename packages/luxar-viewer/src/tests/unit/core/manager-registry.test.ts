/**
 * Tests for ManagerRegistry — central singleton-lifecycle coordinator.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  ManagerRegistry,
  __resetManagerRegistryForTests,
  getManagerRegistry,
} from '../../../core/manager-registry';

interface FakeManager {
  dispose: ReturnType<typeof vi.fn> & (() => void);
}

const fake = (): FakeManager => ({ dispose: vi.fn() }) as unknown as FakeManager;

describe('ManagerRegistry', () => {
  beforeEach(() => {
    __resetManagerRegistryForTests();
  });

  it('registers and looks up managers by name', () => {
    const reg = new ManagerRegistry();
    const m = fake();
    reg.register('Foo', m);
    expect(reg.get('Foo')).toBe(m);
    expect(reg.has('Foo')).toBe(true);
    expect(reg.has('Bar')).toBe(false);
  });

  it('logs a warning and keeps the original on duplicate registration', () => {
    const reg = new ManagerRegistry();
    const a = fake();
    const b = fake();
    reg.register('Foo', a);
    reg.register('Foo', b);
    expect(reg.get('Foo')).toBe(a);
  });

  it('disposeAll walks managers in REVERSE registration order', () => {
    const reg = new ManagerRegistry();
    const order: string[] = [];
    const m = (name: string): FakeManager =>
      ({
        dispose: vi.fn(() => {
          order.push(name);
        }),
      }) as unknown as FakeManager;
    reg.register('A', m('A'));
    reg.register('B', m('B'));
    reg.register('C', m('C'));
    reg.disposeAll();
    expect(order).toEqual(['C', 'B', 'A']);
  });

  it('disposeAll is idempotent — repeat calls do not re-dispose managers', () => {
    const reg = new ManagerRegistry();
    const m = fake();
    reg.register('Foo', m);
    reg.disposeAll();
    reg.disposeAll();
    reg.disposeAll();
    expect(m.dispose).toHaveBeenCalledTimes(1);
  });

  it('a thrown dispose() does not block the rest', () => {
    const reg = new ManagerRegistry();
    const a = {
      dispose: vi.fn(() => {
        throw new Error('A boom');
      }),
    } as unknown as FakeManager;
    const b = fake();
    reg.register('A', a);
    reg.register('B', b);
    reg.disposeAll();
    // Reverse order: B first, then A throws but doesn't stop the loop.
    expect(b.dispose).toHaveBeenCalled();
    expect(a.dispose).toHaveBeenCalled();
  });

  it('get() throws after the manager is disposed', () => {
    const reg = new ManagerRegistry();
    reg.register('Foo', fake());
    reg.disposeAll();
    expect(() => reg.get('Foo')).toThrow(/disposed/);
  });

  it('has() reflects disposal state', () => {
    const reg = new ManagerRegistry();
    reg.register('Foo', fake());
    expect(reg.has('Foo')).toBe(true);
    reg.disposeAll();
    expect(reg.has('Foo')).toBe(false);
  });

  it('getStatus returns a list reflecting registration order and disposal flags', () => {
    const reg = new ManagerRegistry();
    reg.register('A', fake());
    reg.register('B', fake());
    let status = reg.getStatus();
    expect(status).toEqual([
      { name: 'A', disposed: false },
      { name: 'B', disposed: false },
    ]);
    reg.disposeAll();
    status = reg.getStatus();
    expect(status).toEqual([
      { name: 'A', disposed: true },
      { name: 'B', disposed: true },
    ]);
  });

  it('reset() drops everything (tests-only)', () => {
    const reg = new ManagerRegistry();
    reg.register('Foo', fake());
    reg.reset();
    expect(reg.has('Foo')).toBe(false);
    expect(reg.getStatus()).toEqual([]);
  });

  it('getManagerRegistry() returns the same singleton across calls', () => {
    expect(getManagerRegistry()).toBe(getManagerRegistry());
  });

  it('__resetManagerRegistryForTests() yields a fresh singleton', () => {
    const a = getManagerRegistry();
    __resetManagerRegistryForTests();
    const b = getManagerRegistry();
    expect(a).not.toBe(b);
  });

  // re-init lifecycle. Pre-fix, register() rejected any
  // name in `managers` Map (no `disposed` check). After disposeAll(),
  // names stayed in `managers`/`order`, so a fresh app's register()
  // call was treated as duplicate-warn and the new manager was NEVER
  // lifecycle-managed. The fix recycles the slot when the slot's
  // previous occupant has been disposed.
  describe('re-registration after dispose', () => {
    it('register() replaces a previously-disposed manager with the same name', () => {
      const reg = new ManagerRegistry();
      const a = fake();
      reg.register('Foo', a);
      reg.disposeAll();

      const b = fake();
      reg.register('Foo', b);
      // The new manager replaces the disposed slot.
      expect(reg.get('Foo')).toBe(b);
      expect(reg.has('Foo')).toBe(true);
    });

    it('the replaced manager IS disposed by a subsequent disposeAll()', () => {
      const reg = new ManagerRegistry();
      const a = fake();
      reg.register('Foo', a);
      reg.disposeAll();
      expect(a.dispose).toHaveBeenCalledTimes(1);

      const b = fake();
      reg.register('Foo', b);
      reg.disposeAll();
      // b WAS lifecycle-managed (the bug pre-fix was that it wasn't).
      expect(b.dispose).toHaveBeenCalledTimes(1);
      // a was not re-disposed.
      expect(a.dispose).toHaveBeenCalledTimes(1);
    });

    it('replacement preserves the warn-on-live-duplicate behavior', () => {
      const reg = new ManagerRegistry();
      const a = fake();
      const b = fake();
      reg.register('Foo', a);
      // Live duplicate: NOT yet disposed → keep original, warn.
      reg.register('Foo', b);
      expect(reg.get('Foo')).toBe(a);
    });

    it('replacement removes the name from disposed and the dead order entry', () => {
      const reg = new ManagerRegistry();
      reg.register('Foo', fake());
      reg.disposeAll();
      const newFoo = fake();
      reg.register('Foo', newFoo);
      const status = reg.getStatus();
      // Single entry, fresh state.
      expect(status).toEqual([{ name: 'Foo', disposed: false }]);
    });
  });
});
