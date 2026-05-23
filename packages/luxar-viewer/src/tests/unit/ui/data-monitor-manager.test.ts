/**
 * Unit tests for the DataMonitorManager singleton lifecycle.
 *
 * Focuses on the cross-layer event-bus subscription contract: the
 * constructor wires `panel-cycle` / `panel-hide` handlers, and both
 * `destroyAll()` and `disposeInstance()` must release them so the
 * (now monitor-less) singleton is GC-eligible and stale emissions
 * don't fire stale handlers (HIGH-8).
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DataMonitorManager } from '../../../ui/data-monitor-manager';
import { eventBus } from '../../../utils/cross-layer/event-bus';

describe('DataMonitorManager — event-bus subscription lifecycle (HIGH-8)', () => {
  beforeEach(() => {
    // Each case starts from a fresh singleton + clean bus so handler
    // counts are predictable.
    DataMonitorManager.disposeInstance();
    eventBus.clear('panel-cycle');
    eventBus.clear('panel-hide');
  });

  afterEach(() => {
    DataMonitorManager.disposeInstance();
    eventBus.clear('panel-cycle');
    eventBus.clear('panel-hide');
    vi.restoreAllMocks();
  });

  it('event-bus subscriptions are released after destroyAll', () => {
    // Construct the singleton — this wires the two bus subscriptions
    // for 'panel-cycle' and 'panel-hide' (see constructor).
    const manager = DataMonitorManager.getInstance();

    // Spy on the methods the handlers route to. After destroyAll, an
    // emit MUST NOT route through these — proving the subscriptions
    // were released.
    const cycleSpy = vi.spyOn(manager, 'cycleMonitor');
    const hideSpy = vi.spyOn(manager, 'hideMonitor');

    // Sanity: pre-destroy, the subscriptions are live.
    eventBus.emit('panel-cycle', { panelId: 'data-monitor' });
    eventBus.emit('panel-hide', { panelId: 'data-monitor' });
    expect(cycleSpy).toHaveBeenCalledTimes(1);
    expect(hideSpy).toHaveBeenCalledTimes(1);

    // destroyAll() (without disposeInstance) must also release the
    // bus subscriptions — the bug being fixed is that it previously
    // didn't, leaving stale handlers on the bus.
    manager.destroyAll();
    cycleSpy.mockClear();
    hideSpy.mockClear();

    // Emit again — no handler should fire.
    eventBus.emit('panel-cycle', { panelId: 'data-monitor' });
    eventBus.emit('panel-hide', { panelId: 'data-monitor' });
    expect(cycleSpy).not.toHaveBeenCalled();
    expect(hideSpy).not.toHaveBeenCalled();
  });

  it('destroyAll is idempotent: calling it twice does not throw', () => {
    const manager = DataMonitorManager.getInstance();
    manager.destroyAll();
    expect(() => manager.destroyAll()).not.toThrow();
  });

  it('disposeInstance still releases event-bus subscriptions (delegation path)', () => {
    const manager = DataMonitorManager.getInstance();
    const cycleSpy = vi.spyOn(manager, 'cycleMonitor');

    DataMonitorManager.disposeInstance();

    // The post-dispose manager reference still exists locally but the
    // singleton is gone. Emitting on the bus must not call the spy —
    // disposeInstance now delegates the unsubscribe work to destroyAll.
    eventBus.emit('panel-cycle', { panelId: 'data-monitor' });
    expect(cycleSpy).not.toHaveBeenCalled();
  });
});
