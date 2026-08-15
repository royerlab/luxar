// @vitest-environment jsdom
/**
 * Tests for DataMonitorManager singleton lifecycle.
 *
 * [architecture.md/O2][P10] Moved from tests/unit/architecture/global-state.test.ts —
 * `DataMonitorManager` describe block is a behavioral singleton test for the
 * UI layer; belongs alongside other ui tests, not under "architecture".
 *
 * (The companion file `ui/data-monitor-manager.test.ts` covers the event-bus
 * subscription contract; this file pins the basic singleton/multi-monitor
 * lifecycle.)
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { DataMonitorManager } from '../../../ui/data-monitor-manager';

describe('DataMonitorManager — singleton lifecycle', () => {
  beforeEach(() => {
    DataMonitorManager.disposeInstance();
  });

  afterEach(() => {
    DataMonitorManager.disposeInstance();
  });

  it('should manage monitors without global variables', () => {
    const manager = DataMonitorManager.getInstance();

    // Should start with no monitors
    expect(manager.getMonitorCount()).toBe(0);
    expect(manager.getDefaultMonitor()).toBeNull();

    // Create a monitor (with a mock container)
    const mockContainer = document.createElement('div');
    const monitor1 = manager.createMonitor('monitor1', mockContainer);
    expect(manager.getMonitorCount()).toBe(1);
    expect(manager.getDefaultMonitor()).toBe(monitor1);

    // Create another monitor
    const monitor2 = manager.createMonitor('monitor2', mockContainer, undefined, false);
    expect(manager.getMonitorCount()).toBe(2);
    expect(manager.getDefaultMonitor()).toBe(monitor1); // Should still be monitor1

    // Destroy a monitor
    manager.destroyMonitor('monitor1');
    expect(manager.getMonitorCount()).toBe(1);
    expect(manager.getDefaultMonitor()).toBe(monitor2); // Should switch to monitor2
  });

  it('should use singleton pattern correctly', () => {
    const manager1 = DataMonitorManager.getInstance();
    const manager2 = DataMonitorManager.getInstance();

    // Should be the same instance
    expect(manager1).toBe(manager2);

    // Changes in one should be visible in the other
    const mockContainer = document.createElement('div');
    manager1.createMonitor('singleton-test', mockContainer);
    expect(manager2.hasMonitor('singleton-test')).toBe(true);
  });

  it('should handle getAllMonitors correctly', () => {
    const manager = DataMonitorManager.getInstance();
    const mockContainer = document.createElement('div');

    // Initially empty
    expect(manager.getAllMonitors().size).toBe(0);

    // Create multiple monitors
    manager.createMonitor('monitor1', mockContainer);
    manager.createMonitor('monitor2', mockContainer);
    manager.createMonitor('monitor3', mockContainer);

    const allMonitors = manager.getAllMonitors();
    expect(allMonitors.size).toBe(3);
    expect(allMonitors.has('monitor1')).toBe(true);
    expect(allMonitors.has('monitor2')).toBe(true);
    expect(allMonitors.has('monitor3')).toBe(true);
  });

  it('should handle destroying all monitors', () => {
    const manager = DataMonitorManager.getInstance();
    const mockContainer = document.createElement('div');

    // Create multiple monitors
    manager.createMonitor('monitor1', mockContainer);
    manager.createMonitor('monitor2', mockContainer);
    manager.createMonitor('monitor3', mockContainer);
    expect(manager.getMonitorCount()).toBe(3);

    // Destroy all
    manager.destroyAll();
    expect(manager.getMonitorCount()).toBe(0);
    expect(manager.getDefaultMonitor()).toBeNull();
  });

  it('should reset properly for testing', () => {
    const manager = DataMonitorManager.getInstance();
    const mockContainer = document.createElement('div');

    manager.createMonitor('test', mockContainer);
    expect(manager.getMonitorCount()).toBe(1);

    // Reset should clear everything
    DataMonitorManager.disposeInstance();

    // New instance should be empty
    const newManager = DataMonitorManager.getInstance();
    expect(newManager.getMonitorCount()).toBe(0);
  });
});
