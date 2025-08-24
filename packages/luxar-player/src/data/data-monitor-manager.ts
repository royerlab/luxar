/**
 * DataMonitorManager - Manages DataLoadingMonitor instances without global state
 *
 * This manager provides a clean way to access DataLoadingMonitor instances without
 * polluting the global window object.
 */

import { DataLoadingMonitor } from '../ui/data-loading-monitor';
import type { MonitorConfig } from '../ui/data-monitor-types';

/**
 * Manager for DataLoadingMonitor instances.
 * Provides centralized access to monitor instances without global variables.
 */
export class DataMonitorManager {
  private static instance: DataMonitorManager | null = null;
  private monitors = new Map<string, DataLoadingMonitor>();
  private defaultMonitorId: string | null = null;

  /**
   * Private constructor to enforce singleton pattern
   */
  private constructor() {}

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
   * @param setAsDefault - Whether to set this as the default monitor
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
   * Destroy all monitors and reset the manager
   */
  destroyAll(): void {
    for (const monitor of this.monitors.values()) {
      monitor.dispose();
    }
    this.monitors.clear();
    this.defaultMonitorId = null;
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
   * Reset the singleton instance (mainly for testing)
   */
  static reset(): void {
    if (DataMonitorManager.instance) {
      DataMonitorManager.instance.destroyAll();
      DataMonitorManager.instance = null;
    }
  }
}

// Export convenient accessor functions
export function getDataMonitor(id?: string): DataLoadingMonitor | null {
  const manager = DataMonitorManager.getInstance();
  return id ? manager.getMonitor(id) : manager.getDefaultMonitor();
}

export function showDataMonitor(id?: string): void {
  DataMonitorManager.getInstance().showMonitor(id);
}

export function hideDataMonitor(id?: string): void {
  DataMonitorManager.getInstance().hideMonitor(id);
}

export function toggleDataMonitor(id?: string): void {
  DataMonitorManager.getInstance().toggleMonitor(id);
}
