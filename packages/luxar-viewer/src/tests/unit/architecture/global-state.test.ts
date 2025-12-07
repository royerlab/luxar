/**
 * Tests for verifying that global state has been properly removed
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SceneLoaderManager, DataMonitorManager, dispose } from '../../../data';

describe('Global State Management', () => {
  beforeEach(() => {
    // Reset managers before each test
    SceneLoaderManager.reset();
    DataMonitorManager.reset();
  });

  afterEach(() => {
    // Clean up after each test
    SceneLoaderManager.reset();
    DataMonitorManager.reset();
  });

  describe('SceneLoaderManager', () => {
    it('should manage instances without global variables', () => {
      const manager = SceneLoaderManager.getInstance();

      // Should start with no loaders
      expect(manager.getLoaderCount()).toBe(0);
      expect(manager.getDefaultLoader()).toBeNull();

      // Create a loader
      const loader1 = manager.createLoader('test1');
      expect(manager.getLoaderCount()).toBe(1);
      expect(manager.getDefaultLoader()).toBe(loader1);

      // Create another loader
      const loader2 = manager.createLoader('test2', undefined, false);
      expect(manager.getLoaderCount()).toBe(2);
      expect(manager.getDefaultLoader()).toBe(loader1); // Should still be loader1

      // Get loaders by ID
      expect(manager.getLoader('test1')).toBe(loader1);
      expect(manager.getLoader('test2')).toBe(loader2);
      expect(manager.getLoader('nonexistent')).toBeNull();

      // Destroy a loader
      manager.destroyLoader('test1');
      expect(manager.getLoaderCount()).toBe(1);
      expect(manager.getDefaultLoader()).toBe(loader2); // Should switch to loader2
    });

    it('should use singleton pattern correctly', () => {
      const manager1 = SceneLoaderManager.getInstance();
      const manager2 = SceneLoaderManager.getInstance();

      // Should be the same instance
      expect(manager1).toBe(manager2);

      // Changes in one should be visible in the other
      manager1.createLoader('singleton-test');
      expect(manager2.hasLoader('singleton-test')).toBe(true);
    });

    it('should reset properly for testing', () => {
      const manager = SceneLoaderManager.getInstance();
      manager.createLoader('test');
      expect(manager.getLoaderCount()).toBe(1);

      // Reset should clear everything
      SceneLoaderManager.reset();

      // New instance should be empty
      const newManager = SceneLoaderManager.getInstance();
      expect(newManager.getLoaderCount()).toBe(0);
    });

    it('should handle getAllLoaders correctly', () => {
      const manager = SceneLoaderManager.getInstance();

      // Initially empty
      expect(manager.getAllLoaders().size).toBe(0);

      // Create multiple loaders
      manager.createLoader('loader1');
      manager.createLoader('loader2');
      manager.createLoader('loader3');

      const allLoaders = manager.getAllLoaders();
      expect(allLoaders.size).toBe(3);
      expect(allLoaders.has('loader1')).toBe(true);
      expect(allLoaders.has('loader2')).toBe(true);
      expect(allLoaders.has('loader3')).toBe(true);
    });

    it('should check loader existence with hasLoader', () => {
      const manager = SceneLoaderManager.getInstance();

      expect(manager.hasLoader('nonexistent')).toBe(false);

      manager.createLoader('exists');
      expect(manager.hasLoader('exists')).toBe(true);

      manager.destroyLoader('exists');
      expect(manager.hasLoader('exists')).toBe(false);
    });

    it('should handle destroying all loaders', () => {
      const manager = SceneLoaderManager.getInstance();

      // Create multiple loaders
      manager.createLoader('loader1');
      manager.createLoader('loader2');
      manager.createLoader('loader3');
      expect(manager.getLoaderCount()).toBe(3);

      // Destroy all
      manager.destroyAll();
      expect(manager.getLoaderCount()).toBe(0);
      expect(manager.getDefaultLoader()).toBeNull();
    });

    it('should handle default loader switching when default is destroyed', () => {
      const manager = SceneLoaderManager.getInstance();

      const loader1 = manager.createLoader('loader1', undefined, true);
      const loader2 = manager.createLoader('loader2', undefined, false);
      const loader3 = manager.createLoader('loader3', undefined, false);

      expect(manager.getDefaultLoader()).toBe(loader1);

      // Destroy the default loader
      manager.destroyLoader('loader1');

      // Should auto-select next available loader as default
      const newDefault = manager.getDefaultLoader();
      expect(newDefault).toBeTruthy();
      expect(newDefault === loader2 || newDefault === loader3).toBe(true);
    });
  });

  describe('DataMonitorManager', () => {
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
      DataMonitorManager.reset();

      // New instance should be empty
      const newManager = DataMonitorManager.getInstance();
      expect(newManager.getMonitorCount()).toBe(0);
    });
  });

  describe('Integration with zarr-loader API', () => {
    it('should not create global window variables', () => {
      // Check that no global variables are created
      expect((window as any).__luxarSceneLoader).toBeUndefined();
      expect((window as any).__luxarLoader).toBeUndefined();
      expect((window as any).__luxarDataMonitor).toBeUndefined();
      expect((window as any).globalSceneLoader).toBeUndefined();
    });

    it('should support multiple independent loaders', () => {
      const manager = SceneLoaderManager.getInstance();

      // Create two independent loaders
      const loader1 = manager.createLoader('loader1');
      const loader2 = manager.createLoader('loader2');

      // They should be different instances
      expect(loader1).not.toBe(loader2);

      // Both should be accessible
      expect(manager.getLoader('loader1')).toBe(loader1);
      expect(manager.getLoader('loader2')).toBe(loader2);
    });

    it('should clean up properly with dispose', () => {
      const manager = SceneLoaderManager.getInstance();

      // Create some loaders
      manager.createLoader('test1');
      manager.createLoader('test2');
      expect(manager.getLoaderCount()).toBe(2);

      // Dispose a specific loader
      dispose('test1');
      expect(manager.getLoaderCount()).toBe(1);
      expect(manager.hasLoader('test1')).toBe(false);
      expect(manager.hasLoader('test2')).toBe(true);

      // Dispose all
      dispose();
      expect(manager.getLoaderCount()).toBe(0);
    });
  });

  describe('No global state pollution', () => {
    it('should not pollute window object during normal operation', async () => {
      const windowKeysBefore = Object.keys(window);

      // Perform some operations
      const manager = SceneLoaderManager.getInstance();
      manager.createLoader('test');

      const windowKeysAfter = Object.keys(window);

      // No new keys should be added to window
      const newKeys = windowKeysAfter.filter((key) => !windowKeysBefore.includes(key));
      const luxarKeys = newKeys.filter(
        (key) => key.toLowerCase().includes('luxar') || key.toLowerCase().includes('loader')
      );

      expect(luxarKeys).toHaveLength(0);
    });
  });
});
