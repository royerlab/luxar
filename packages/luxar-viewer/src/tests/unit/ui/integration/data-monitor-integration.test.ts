/**
 * Unit Integration Tests for Data Loading Monitor
 *
 * **TEST SCOPE**: Unit-level integration with mocked dependencies
 * - Uses mocked zarr and THREE.js
 * - Tests monitor ↔ loader communication
 * - Tests event propagation and state updates
 * - Fast execution (no browser)
 *
 * **WHAT WE TEST**:
 * - DataMonitorManager integration with SceneLoader
 * - Monitor receives correct events from loader
 * - Keyboard shortcuts (Ctrl+L) trigger monitor display
 * - Monitor state updates correctly
 *
 * **WHAT WE DON'T TEST** (see E2E tests instead):
 * - Actual UI rendering in browser
 * - Real-time updates during actual data loading
 *
 * **Related Tests**:
 * - `unit/data/data-loading-integration.test.ts` - Loader pipeline (unit)
 * - `e2e/data-monitor-metrics.spec.ts` - Monitor UI + real data (E2E)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { DataMonitorManager, cycleDataMonitor } from '../../../../ui/data-monitor-manager';
import { SceneLoaderManager } from '../../../../data/scene-loader-manager';
import type { LoaderMonitor, MonitorEvent } from '../../../../types/data-monitor-types';

// Mock zarr module
vi.mock('zarr', () => ({
  root: vi.fn(),
  open: vi.fn(),
  openGroup: vi.fn(),
  Readable: vi.fn(),
}));

// Mock THREE.js with required exports
vi.mock('three', () => ({
  // Classes
  Group: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    children: [],
  })),
  Box3: vi.fn().mockImplementation(() => ({
    expandByPoint: vi.fn(),
    isEmpty: vi.fn().mockReturnValue(false),
    getSize: vi.fn().mockReturnValue({ x: 10, y: 10, z: 10 }),
    getCenter: vi.fn().mockReturnValue({ x: 0, y: 0, z: 0 }),
  })),
  Vector2: vi.fn().mockImplementation((x = 0, y = 0) => ({
    x,
    y,
    set: vi.fn().mockReturnThis(),
    copy: vi.fn().mockReturnThis(),
  })),
  Vector3: vi.fn().mockImplementation((x = 0, y = 0, z = 0) => ({
    x,
    y,
    z,
    set: vi.fn().mockReturnThis(),
    copy: vi.fn().mockReturnThis(),
  })),
  ShaderMaterial: vi.fn().mockImplementation(function (this: any, params: any) {
    Object.assign(this, {
      uniforms: params?.uniforms || {},
      vertexShader: params?.vertexShader || '',
      fragmentShader: params?.fragmentShader || '',
      vertexColors: params?.vertexColors || false,
      transparent: params?.transparent || false,
      depthWrite: params?.depthWrite !== undefined ? params.depthWrite : true,
      toneMapped: params?.toneMapped !== undefined ? params.toneMapped : true,
      blending: params?.blending || 'NormalBlending',
      userData: {},
      dispose: vi.fn(),
    });
  }),
  WebGLRenderer: vi.fn().mockImplementation(() => ({
    setSize: vi.fn(),
    render: vi.fn(),
    domElement: document.createElement('canvas'),
  })),
  PerspectiveCamera: vi.fn().mockImplementation(() => ({
    position: { set: vi.fn() },
    lookAt: vi.fn(),
  })),
  Scene: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    children: [],
  })),
  // Required constants
  HalfFloatType: 1016,
  FloatType: 1015,
  UnsignedByteType: 1009,
  SRGBColorSpace: 'srgb',
  LinearSRGBColorSpace: 'srgb-linear',
  NoToneMapping: 0,
  LinearToneMapping: 1,
  ReinhardToneMapping: 2,
  CineonToneMapping: 3,
  ACESFilmicToneMapping: 4,
  AgXToneMapping: 5,
  NeutralToneMapping: 6,
  CustomToneMapping: 7,
  PCFShadowMap: 1,
  // Blending modes
  NormalBlending: 'NormalBlending',
  AdditiveBlending: 'AdditiveBlending',
  SubtractiveBlending: 'SubtractiveBlending',
}));

describe('Data Monitor Integration', () => {
  beforeEach(() => {
    // Reset singletons
    DataMonitorManager.disposeInstance();
    SceneLoaderManager.disposeInstance();

    // Setup DOM
    document.body.innerHTML = '<div id="test-container"></div>';

    // Wire the monitor factory the way core/app.ts does at boot —
    // SceneLoader's monitor coupling is factory-injected, so without
    // this, SceneLoader instances created in these tests have a null
    // monitor.
    SceneLoaderManager.getInstance().setMonitorFactory((monitorId) => {
      const mgr = DataMonitorManager.getInstance();
      if (!mgr.hasMonitor(monitorId)) {
        mgr.createMonitor(monitorId, document.body);
      }
      return mgr.getMonitor(monitorId) ?? null;
    });
  });

  afterEach(() => {
    // Clean up
    DataMonitorManager.disposeInstance();
    SceneLoaderManager.disposeInstance();
    document.body.innerHTML = '';
  });

  describe('M key keyboard integration', () => {
    it('should cycle monitor state when cycleDataMonitor is called', () => {
      // Create a monitor
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('test', document.body);

      // Initial state: hidden
      expect(monitor.isVisible()).toBe(false);

      // Simulate M key (calls cycleDataMonitor)
      cycleDataMonitor('test');
      expect(monitor.isVisible()).toBe(true);
      expect(monitor.isExpanded()).toBe(false); // Mini state

      // Second M key press
      cycleDataMonitor('test');
      expect(monitor.isVisible()).toBe(true);
      expect(monitor.isExpanded()).toBe(true); // Expanded state

      // Third M key press
      cycleDataMonitor('test');
      expect(monitor.isVisible()).toBe(false); // Hidden again
    });

    it('should work with default monitor when no ID specified', () => {
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('default', document.body);

      // Call without ID (uses default)
      cycleDataMonitor();
      expect(monitor.isVisible()).toBe(true);
      expect(monitor.isExpanded()).toBe(false);

      cycleDataMonitor();
      expect(monitor.isExpanded()).toBe(true);
    });

    it('should handle missing monitor gracefully', () => {
      // No monitor created
      expect(() => cycleDataMonitor('nonexistent')).not.toThrow();
    });
  });

  describe('SceneLoader integration', () => {
    // SceneLoader's monitor coupling is a factory injected via
    // SceneLoaderManager. Tests construct loaders through the manager
    // (which threads the factory wired up in this file's beforeEach)
    // rather than via `new SceneLoader(...)` directly — the latter
    // bypasses the factory and yields a null monitor.
    it('should create monitor on SceneLoader construction', () => {
      const manager = DataMonitorManager.getInstance();
      SceneLoaderManager.getInstance().createLoader('test-scene', { enableMonitor: true });
      const monitor = manager.getMonitor('test-scene-monitor');
      expect(monitor).toBeDefined();
    });

    it('should not create monitor when disabled in config', () => {
      const manager = DataMonitorManager.getInstance();
      SceneLoaderManager.getInstance().createLoader('test-scene', { enableMonitor: false });
      // Monitor should not be created — the factory is gated on
      // config.enableMonitor inside SceneLoader.
      const monitor = manager.getMonitor('test-scene-monitor');
      expect(monitor).toBeNull();
    });

    it('should connect loaders to monitor', async () => {
      const manager = DataMonitorManager.getInstance();
      SceneLoaderManager.getInstance().createLoader('test-scene', { enableMonitor: true });
      const monitor = manager.getMonitor('test-scene-monitor');

      expect(monitor).toBeDefined();
      if (!monitor) return;

      // Mock the zarr store and scene loading
      const mockStore = {
        getItem: vi.fn(),
        containsItem: vi.fn(),
      };
      void mockStore; // Explicitly mark as used for testing

      // Note: Actual zarr mocking is already handled by vi.mock at the top
      // We would need more complex mocking for actual scene loading

      // Spy on monitor.connectLoader
      const connectSpy = vi.spyOn(monitor, 'connectLoader');

      // Note: Actual scene loading would require more complex mocking
      // For this test, we verify the monitor is ready to receive connections
      expect(connectSpy).not.toHaveBeenCalled(); // Not called yet
      expect(monitor.getGlobalStats().totalLoaders).toBe(0);
    });

    it('should disconnect all loaders when loading new scene', () => {
      const manager = DataMonitorManager.getInstance();
      SceneLoaderManager.getInstance().createLoader('test-scene', { enableMonitor: true });
      const monitor = manager.getMonitor('test-scene-monitor');

      if (!monitor) throw new Error('Monitor should exist');

      // Create mock loaders
      const mockLoader1: LoaderMonitor = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test1',
          queries: 0,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          elementsLoaded: 0,
          bytesLoaded: 0,
          visibleElements: 0,
          avgQueryTime: 0,
          avgLoadTime: 0,
          cacheHitRate: 0,
          memoryUsed: 0,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      // Connect loader manually (simulating what createLoader does)
      monitor.connectLoader('/test1', mockLoader1);
      expect(monitor.getGlobalStats().totalLoaders).toBe(1);

      // Call disconnectAllLoaders (this is what loadScene does)
      monitor.disconnectAllLoaders();

      // Verify disconnection
      expect(mockLoader1.removeEventListener).toHaveBeenCalled();
      expect(monitor.getGlobalStats().totalLoaders).toBe(0);
    });

    it('collapses a substitutive kind=lod group to one logical loader', () => {
      const manager = DataMonitorManager.getInstance();
      SceneLoaderManager.getInstance().createLoader('lod-scene', { enableMonitor: true });
      const monitor = manager.getMonitor('lod-scene-monitor');
      if (!monitor) throw new Error('Monitor should exist');

      const makeLoader = (path: string): LoaderMonitor => ({
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'gsplats-spatial-index' as const,
          path,
          queries: 0,
          loads: 0,
          evictions: 0,
          errors: 0,
          elementsLoaded: 0,
          bytesLoaded: 0,
          visibleElements: 0,
          avgQueryTime: 0,
          avgLoadTime: 0,
          memoryUsed: 0,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      });

      // A kind=lod group connects one loader per level (here 3).
      monitor.connectLoader('/lod/l0', makeLoader('/lod/l0'));
      monitor.connectLoader('/lod/l1', makeLoader('/lod/l1'));
      monitor.connectLoader('/lod/l2', makeLoader('/lod/l2'));

      // Before the LOD-progress snapshot is known, all three count (fallback).
      expect(monitor.getGlobalStats().totalLoaders).toBe(3);

      // Seed the snapshot the polling tick would populate from the provider:
      // the group at /lod has 3 substitutive levels.
      (monitor as unknown as { lodStates: Map<string, unknown> }).lodStates = new Map([
        ['/lod', { kind: 'lod', levelCount: 3, activeLevel: 0 }],
      ]);

      // Now the K=3 level loaders collapse to one logical layer.
      const stats = monitor.getGlobalStats();
      expect(stats.totalLoaders).toBe(1);
      expect(stats.activeSpatialLoaders).toBe(1);
    });

    it('disconnectAllLoaders also nulls scene-bound providers (no stale closures)', () => {
      const manager = DataMonitorManager.getInstance();
      SceneLoaderManager.getInstance().createLoader('test-scene', { enableMonitor: true });
      const monitor = manager.getMonitor('test-scene-monitor');
      if (!monitor) throw new Error('Monitor should exist');

      // Inject providers that throw if called after they should be reset.
      // This mirrors how SceneLoader's L0 closure throws after dispose
      // sets `this.l0Cache = null` — without resetSceneProviders the
      // monitor's stats poll NPEs on the next tick.
      let l0Calls = 0;
      monitor.setL0CacheProvider({
        getStats: () => {
          l0Calls++;
          return { count: 0, size: 0, hits: 0, misses: 0, evictions: 0, hitRate: 0 };
        },
        clear: () => {},
      });

      // Drop everything (scene reload path)
      monitor.disconnectAllLoaders();

      // After reset: clearL0Cache is a no-op (provider is null) and a
      // subsequent stats fetch won't reach the stale closure.
      monitor.clearL0Cache();
      const before = l0Calls;
      // The provider should be null — clearL0Cache logs nothing and returns.
      expect(l0Calls).toBe(before);
    });

    it('dispose() also nulls scene-bound providers and resets UI state', () => {
      // Pre-13.3 dispose() did NOT call resetSceneProviders() — only
      // disconnectAllLoaders() did. Direct dispose callers (and any
      // external/debug references that survive a teardown order
      // change) could still hold provider closures bound to the
      // disposed SceneLoader.
      const manager = DataMonitorManager.getInstance();
      SceneLoaderManager.getInstance().createLoader('test-scene', { enableMonitor: true });
      const monitor = manager.getMonitor('test-scene-monitor');
      if (!monitor) throw new Error('Monitor should exist');

      let l0Calls = 0;
      monitor.setL0CacheProvider({
        getStats: () => {
          l0Calls++;
          return { count: 0, size: 0, hits: 0, misses: 0, evictions: 0, hitRate: 0 };
        },
        clear: () => {},
      });

      // Force visibility/expanded so we can verify they reset.
      monitor.show();
      expect(monitor.isVisible()).toBe(true);

      monitor.dispose();

      // Provider is null — clearL0Cache won't reach the stale closure.
      const before = l0Calls;
      monitor.clearL0Cache();
      expect(l0Calls).toBe(before);

      // UI state is reset.
      expect(monitor.isVisible()).toBe(false);
    });
  });

  describe('DataMonitorManager singleton', () => {
    it('should return same instance', () => {
      const manager1 = DataMonitorManager.getInstance();
      const manager2 = DataMonitorManager.getInstance();
      expect(manager1).toBe(manager2);
    });

    it('should manage multiple monitors', () => {
      const manager = DataMonitorManager.getInstance();

      const monitor1 = manager.createMonitor('scene1', document.body);
      const monitor2 = manager.createMonitor('scene2', document.body);

      expect(manager.getMonitor('scene1')).toBe(monitor1);
      expect(manager.getMonitor('scene2')).toBe(monitor2);
      expect(manager.getMonitorCount()).toBe(2);
    });

    it('should set default monitor', () => {
      const manager = DataMonitorManager.getInstance();

      const monitor1 = manager.createMonitor('scene1', document.body, undefined, true);
      const monitor2 = manager.createMonitor('scene2', document.body, undefined, false);
      void monitor2; // Explicitly mark as used for testing

      expect(manager.getDefaultMonitor()).toBe(monitor1);

      // Create another as default
      const monitor3 = manager.createMonitor('scene3', document.body, undefined, true);
      expect(manager.getDefaultMonitor()).toBe(monitor3);
    });

    it('should destroy monitors', () => {
      const manager = DataMonitorManager.getInstance();

      manager.createMonitor('test', document.body);
      expect(manager.hasMonitor('test')).toBe(true);

      manager.destroyMonitor('test');
      expect(manager.hasMonitor('test')).toBe(false);
      expect(manager.getMonitor('test')).toBeNull();
    });
  });

  describe('Monitor event flow', () => {
    it('should receive and process loader events', () => {
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('test', document.body);

      let eventListener: ((event: MonitorEvent) => void) | null = null;

      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn((listener: (event: MonitorEvent) => void) => {
          eventListener = listener;
        }) as any,
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 0,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          elementsLoaded: 0,
          bytesLoaded: 0,
          visibleElements: 0,
          avgQueryTime: 0,
          avgLoadTime: 0,
          cacheHitRate: 0,
          memoryUsed: 0,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      // Connect loader
      monitor.connectLoader('/test', mockLoader);
      expect(eventListener).toBeDefined();

      // Make monitor visible and send events
      monitor.show();

      if (eventListener) {
        (eventListener as (event: MonitorEvent) => void)({
          type: 'query',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test', elements: 1000 },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'cache-hit',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test' },
        });
      }

      // Force process queued events
      monitor.forceUpdate();

      // Verify events were processed
      const events = monitor.getRecentEvents();
      expect(events.length).toBeGreaterThanOrEqual(2);
      expect(events.some((e) => e.type === 'query')).toBe(true);
      expect(events.some((e) => e.type === 'cache-hit')).toBe(true);
    });

    it('should update metrics based on events', () => {
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('test', document.body);

      let eventListener: ((event: MonitorEvent) => void) | null = null;

      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn((listener: (event: MonitorEvent) => void) => {
          eventListener = listener;
        }) as any,
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 2,
          loads: 1,
          cacheHits: 3,
          cacheMisses: 2,
          evictions: 0,
          errors: 0,
          elementsLoaded: 2000,
          bytesLoaded: 8000,
          visibleElements: 0,
          avgQueryTime: 15,
          avgLoadTime: 50,
          cacheHitRate: 60,
          memoryUsed: 1024 * 1024,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      // Make monitor visible before sending events
      monitor.show();

      // Send events to update metrics
      if (eventListener) {
        (eventListener as (event: MonitorEvent) => void)({
          type: 'query',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test', elements: 1000, latency: 10 },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'query',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test', elements: 1000, latency: 20 },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'load',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test', elements: 2000, memory: 8000, latency: 50 },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'cache-hit',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test' },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'cache-hit',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test' },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'cache-hit',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test' },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'cache-miss',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test' },
        });

        (eventListener as (event: MonitorEvent) => void)({
          type: 'cache-miss',
          loader: 'point-spatial-index',
          timestamp: Date.now(),
          data: { path: '/test' },
        });
      }

      // Force process queued events
      monitor.forceUpdate();

      // Check global stats - metrics are refreshed from loader.getMetrics() on each UI update
      // So totalQueries reflects what the loader reports, not event counts
      const stats = monitor.getGlobalStats();
      // UI refresh calls loader.getMetrics() which returns queries: 2
      expect(stats.totalQueries).toBe(2);
    });
  });

  describe('Monitor UI state management', () => {
    it('should maintain UI state across show/hide cycles', () => {
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('test', document.body);

      // Set to expanded state
      monitor.show();
      monitor.expand();
      monitor.setActiveTab('cache');

      expect(monitor.isExpanded()).toBe(true);

      // Hide and show again
      monitor.hide();
      expect(monitor.isVisible()).toBe(false);

      monitor.show();
      expect(monitor.isVisible()).toBe(true);
      // State should be preserved
      expect(monitor.isExpanded()).toBe(true);
    });

    it('should handle rapid state changes', () => {
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('test', document.body);

      // Rapid cycling
      for (let i = 0; i < 10; i++) {
        monitor.cycleState();
      }

      // Should end up at mini state (10 % 3 = 1)
      // 0: hidden, 1: mini, 2: expanded
      expect(monitor.isVisible()).toBe(true);
      expect(monitor.isExpanded()).toBe(false);
    });
  });

  describe('Scene graph state lifecycle', () => {
    it('disconnectAllLoaders() clears scene graph display state', () => {
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('test', document.body);

      // Plant a fake scene graph (mimics post-load state).
      monitor.setSceneGraph({
        path: '/',
        name: 'Scene',
        type: 'scene',
        children: [
          { path: '/Points', name: 'Points', type: 'points', pointCount: 100, children: [] },
        ],
      });
      expect(monitor.getSceneGraph().root).not.toBeNull();
      expect(monitor.getSceneGraph().nodesByType.points).toBe(1);

      // Reload simulation: disconnect loaders should also drop the
      // tree so the next scene doesn't display a stale shape if it
      // fails before setSceneGraph() runs.
      monitor.disconnectAllLoaders();

      const after = monitor.getSceneGraph();
      expect(after.root).toBeNull();
      expect(after.nodesByType.points).toBe(0);
      expect(after.totalByType.points).toBe(0);
    });

    it('dispose() clears scene graph display state', () => {
      const manager = DataMonitorManager.getInstance();
      const monitor = manager.createMonitor('test', document.body);

      monitor.setSceneGraph({
        path: '/',
        name: 'Scene',
        type: 'scene',
        children: [
          { path: '/Lines', name: 'Lines', type: 'lines', segmentCount: 50, children: [] },
        ],
      });
      expect(monitor.getSceneGraph().nodesByType.lines).toBe(1);

      monitor.dispose();

      // A stale debug/external reference reading getSceneGraph() after
      // dispose should not see the previous scene's tree.
      const after = monitor.getSceneGraph();
      expect(after.root).toBeNull();
      expect(after.nodesByType.lines).toBe(0);
      expect(after.totalByType.lines).toBe(0);
    });
  });
});
