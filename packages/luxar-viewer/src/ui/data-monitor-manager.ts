/**
 * DataMonitorManager - Manages DataLoadingMonitor instances without global state
 *
 * This manager provides a clean way to access DataLoadingMonitor instances without
 * polluting the global window object.
 */

import { DataLoadingMonitor } from './data-loading-monitor';
import type { MonitorConfig } from '../types/data-monitor-types';
import { eventBus, type Unsubscribe } from '../utils/cross-layer/event-bus';

/**
 * Manager for DataLoadingMonitor instances.
 * Provides centralized access to monitor instances without global variables.
 */
export class DataMonitorManager {
  private static instance: DataMonitorManager | null = null;
  private monitors = new Map<string, DataLoadingMonitor>();
  private defaultMonitorId: string | null = null;
  private busSubscriptions: Unsubscribe[] = [];

  /**
   * Private constructor to enforce singleton pattern. Subscribes to
   * panel-cycle / panel-hide events on the cross-layer event bus so
   * lower layers (input, scene) can drive the data-monitor without
   * importing this UI module directly, which keeps the layer order
   * clean.
   */
  private constructor() {
    this.busSubscriptions.push(
      eventBus.on('panel-cycle', ({ panelId }) => {
        if (panelId === 'data-monitor') this.cycleMonitor();
      }),
      eventBus.on('panel-hide', ({ panelId }) => {
        if (panelId === 'data-monitor') this.hideMonitor();
      })
    );
  }

  /**
   * Get the singleton instance of DataMonitorManager
   */
  static getInstance(): DataMonitorManager {
    if (!DataMonitorManager.instance) {
      DataMonitorManager.instance = new DataMonitorManager();
    }
    return DataMonitorManager.instance;
  }

  /**
   * Create a new DataLoadingMonitor instance
   *
   * @param id - Unique identifier for this monitor
   * @param container - DOM container element
   * @param config - Optional monitor configuration
   * @param setAsDefault - Whether to set this as the default monitor; the first
   *   monitor registered also becomes the default even when this is false
   * @returns The created DataLoadingMonitor instance
   */
  createMonitor(
    id: string = 'default',
    container: HTMLElement,
    config?: MonitorConfig,
    setAsDefault: boolean = true
  ): DataLoadingMonitor {
    // Dispose existing monitor with same ID if it exists
    if (this.monitors.has(id)) {
      this.destroyMonitor(id);
    }

    const monitor = new DataLoadingMonitor(container, config);
    this.monitors.set(id, monitor);

    if (setAsDefault || !this.defaultMonitorId) {
      this.defaultMonitorId = id;
    }

    return monitor;
  }

  /**
   * Get a DataLoadingMonitor by ID
   *
   * @param id - The monitor ID
   * @returns The DataLoadingMonitor instance or null if not found
   */
  getMonitor(id: string): DataLoadingMonitor | null {
    return this.monitors.get(id) || null;
  }

  /**
   * Get the default DataLoadingMonitor
   *
   * @returns The default DataLoadingMonitor instance or null
   */
  getDefaultMonitor(): DataLoadingMonitor | null {
    if (!this.defaultMonitorId) {
      return null;
    }
    return this.monitors.get(this.defaultMonitorId) || null;
  }

  /**
   * Get all active monitors
   *
   * @returns Map of all active monitors
   */
  getAllMonitors(): Map<string, DataLoadingMonitor> {
    return new Map(this.monitors);
  }

  /**
   * Destroy a specific monitor
   *
   * @param id - The monitor ID to destroy
   */
  destroyMonitor(id: string): void {
    const monitor = this.monitors.get(id);
    if (monitor) {
      monitor.dispose();
      this.monitors.delete(id);

      // Update default if needed
      if (this.defaultMonitorId === id) {
        this.defaultMonitorId =
          this.monitors.size > 0 ? (this.monitors.keys().next().value ?? null) : null;
      }
    }
  }

  /**
   * Destroy all monitors and reset the manager.
   *
   * HIGH-8: also releases the cross-layer event-bus subscriptions taken in
   * the constructor. Without this, `destroyAll()` (called outside the
   * `disposeInstance` path — e.g. between dataset switches in tests) left
   * stale handlers wired on the bus, keeping references to a monitor-less
   * singleton alive and crashing on later `panel-cycle` / `panel-hide`
   * emissions. The subscriptions list is cleared so this is idempotent.
   */
  destroyAll(): void {
    for (const monitor of this.monitors.values()) {
      monitor.dispose();
    }
    this.monitors.clear();
    this.defaultMonitorId = null;
    for (const unsubscribe of this.busSubscriptions) {
      unsubscribe();
    }
    this.busSubscriptions = [];
  }

  /**
   * Check if a monitor exists
   *
   * @param id - The monitor ID to check
   * @returns True if the monitor exists
   */
  hasMonitor(id: string): boolean {
    return this.monitors.has(id);
  }

  /**
   * Get the number of active monitors
   *
   * @returns The number of active monitors
   */
  getMonitorCount(): number {
    return this.monitors.size;
  }

  /**
   * Show the default monitor or a specific monitor
   *
   * @param id - Optional monitor ID, defaults to default monitor
   */
  showMonitor(id?: string): void {
    const monitor = id ? this.getMonitor(id) : this.getDefaultMonitor();
    if (monitor) {
      monitor.show();
    }
  }

  /**
   * Hide the default monitor or a specific monitor
   *
   * @param id - Optional monitor ID, defaults to default monitor
   */
  hideMonitor(id?: string): void {
    const monitor = id ? this.getMonitor(id) : this.getDefaultMonitor();
    if (monitor) {
      monitor.hide();
    }
  }

  /**
   * Toggle the default monitor or a specific monitor
   *
   * @param id - Optional monitor ID, defaults to default monitor
   */
  toggleMonitor(id?: string): void {
    const monitor = id ? this.getMonitor(id) : this.getDefaultMonitor();
    if (monitor) {
      monitor.toggle();
    }
  }

  /**
   * Cycle through monitor states: hidden → mini → expanded → hidden
   *
   * @param id - Optional monitor ID, defaults to default monitor
   */
  cycleMonitor(id?: string): void {
    const monitor = id ? this.getMonitor(id) : this.getDefaultMonitor();
    if (monitor) {
      monitor.cycleState();
    }
  }

  /**
   * Dispose the current instance and clear the singleton slot.
   *
   * Used at app shutdown and between tests. The next `getInstance()` call
   * lazily constructs a fresh manager.
   */
  static disposeInstance(): void {
    if (DataMonitorManager.instance) {
      // destroyAll() now also releases the event-bus subscriptions
      // (HIGH-8), so no extra unsubscribe loop is needed here.
      DataMonitorManager.instance.destroyAll();
      DataMonitorManager.instance = null;
    }
  }
}

// Export convenient accessor functions.
// Note: getDataMonitor / showDataMonitor / hideDataMonitor /
// toggleDataMonitor were removed — production code calls the
// event-bus ('panel-cycle' / 'panel-hide') instead. cycleDataMonitor
// is kept because data-monitor-integration tests import it
// directly to drive the cycle behavior.
/**
 * Advance the data-loading monitor panel through its display states via the
 * shared {@link DataMonitorManager} singleton.
 *
 * Production code drives the panel through the cross-layer event bus
 * ('panel-cycle'); this thin accessor is retained so integration tests can drive
 * the cycle behavior directly.
 *
 * @param id Optional monitor id to cycle; defaults to the manager's default monitor.
 */
export function cycleDataMonitor(id?: string): void {
  DataMonitorManager.getInstance().cycleMonitor(id);
}
