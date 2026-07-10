/**
 * Unit tests for the Data Loading Monitor
 *
 * AUDIT NOTE (data.md W7 — open audit acknowledgment):
 *   Many tests in this file lean on `expect(button).not.toBeNull()` style
 *   DOM-presence assertions rather than verifying that clicking the button
 *   exercises the documented state transition (minimize/maximize/close).
 *   Strengthening would mean dispatching `MouseEvent('click')` events and
 *   asserting the resulting class/visibility/style transitions, with at
 *   least one anti-test per pair so a no-op handler is caught.
 *   This refactor is OUT OF SCOPE for this audit pass.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { DataLoadingMonitor } from '../../../ui/data-loading-monitor';
import { nodeStatsContent } from '../../../ui/data-loading-monitor/templates';
import type {
  MonitorEvent,
  LoaderMonitor,
  LoaderMetrics,
  LODProgressState,
} from '../../../types/data-monitor-types';

// Mock DOM environment
beforeEach(() => {
  document.body.innerHTML = '<div id="test-container"></div>';
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('DataLoadingMonitor', () => {
  let monitor: DataLoadingMonitor;
  let container: HTMLElement;

  beforeEach(() => {
    container = document.getElementById('test-container')!;
    monitor = new DataLoadingMonitor(container);
  });

  describe('initialization', () => {
    it('should create monitor instance', () => {
      expect(monitor).toBeDefined();
    });

    it('should respect configuration options', () => {
      // [data.md/W7][P2] Previously toBeDefined() only — passed for any
      // constructor that returned anything truthy, even if the supplied
      // options were dropped. Pin each option onto the observable config
      // bag and the disposable lifecycle.
      const customMonitor = new DataLoadingMonitor(container, {
        position: 'top-left',
        theme: 'light',
        defaultView: 'detailed',
        maxEvents: 500,
      });
      // Config bag must store every option we asked for.
      const cfg = (
        customMonitor as unknown as {
          config: {
            position: string;
            theme: string;
            defaultView: string;
            maxEvents: number;
          };
        }
      ).config;
      expect(cfg.position).toBe('top-left');
      expect(cfg.theme).toBe('light');
      expect(cfg.defaultView).toBe('detailed');
      expect(cfg.maxEvents).toBe(500);
      // Sanity: dispose is wired even for the custom-constructed instance.
      expect(() => customMonitor.dispose()).not.toThrow();
    });
  });

  describe('loader connection', () => {
    it('should connect and disconnect loaders', () => {
      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn(),
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

      monitor.connectLoader('/test', mockLoader);
      expect(mockLoader.addEventListener).toHaveBeenCalledTimes(1);
      expect(mockLoader.getMetrics).toHaveBeenCalledTimes(1);

      monitor.disconnectLoader('/test');
      expect(mockLoader.removeEventListener).toHaveBeenCalledTimes(1);
    });
  });

  describe('event handling', () => {
    it('should handle loader events', () => {
      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn((listener) => {
          // Simulate an event
          const event: MonitorEvent = {
            type: 'query',
            loader: 'point-spatial-index',
            timestamp: Date.now(),
            data: {
              path: '/test',
              cells: 10,
              elements: 1000,
              latency: 50,
            },
          };
          listener(event);
        }),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 1,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          elementsLoaded: 1000,
          bytesLoaded: 0,
          visibleElements: 0,
          avgQueryTime: 50,
          avgLoadTime: 0,
          cacheHitRate: 0,
          memoryUsed: 0,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      // Verify event was processed
      const globalStats = monitor.getGlobalStats();
      expect(globalStats.totalQueries).toBeGreaterThan(0);
    });

    it('should limit event storage', () => {
      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn((listener) => {
          // Simulate many events
          for (let i = 0; i < 2000; i++) {
            const event: MonitorEvent = {
              type: 'cache-hit',
              loader: 'point-spatial-index',
              timestamp: Date.now(),
              data: { path: '/test' },
            };
            listener(event);
          }
        }),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 0,
          loads: 0,
          cacheHits: 2000,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          elementsLoaded: 0,
          bytesLoaded: 0,
          visibleElements: 0,
          avgQueryTime: 0,
          avgLoadTime: 0,
          cacheHitRate: 100,
          memoryUsed: 0,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      // Process queued events (with polling architecture, events are queued until tick)
      monitor.show();
      monitor.forceUpdate();

      // Events should be limited to maxEvents (default 1000)
      const events = monitor.getRecentEvents();
      expect(events.length).toBeGreaterThan(0); // Verify events were actually processed
      expect(events.length).toBeLessThanOrEqual(1000);
    });
  });

  describe('UI visibility', () => {
    it('should show and hide monitor', () => {
      expect(monitor.isVisible()).toBe(false);

      monitor.show();
      expect(monitor.isVisible()).toBe(true);

      monitor.hide();
      expect(monitor.isVisible()).toBe(false);
    });

    it('should toggle visibility', () => {
      expect(monitor.isVisible()).toBe(false);

      monitor.toggle();
      expect(monitor.isVisible()).toBe(true);

      monitor.toggle();
      expect(monitor.isVisible()).toBe(false);
    });

    it('should expand and collapse', () => {
      monitor.show();
      expect(monitor.isExpanded()).toBe(false);

      monitor.expand();
      expect(monitor.isExpanded()).toBe(true);

      monitor.collapse();
      expect(monitor.isExpanded()).toBe(false);
    });

    // Regression: show() must NOT hard-code inline `display: block`. Doing so
    // overrode `.luxar-data-monitor--expanded { display: flex }`, breaking the
    // flex-column scroll contract so a long scene-graph tree spilled past the
    // panel's max-height instead of scrolling inside __content. show() should
    // clear the inline display set by hide() and let the size class govern.
    it('show() clears inline display so the CSS size class governs layout', () => {
      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement;
      expect(panel).toBeTruthy();

      monitor.hide();
      expect(panel.style.display).toBe('none');

      monitor.show();
      monitor.expand();
      // Inline display is cleared (not 'block'), letting --expanded's
      // `display: flex` take effect.
      expect(panel.style.display).toBe('');
      expect(panel.classList.contains('luxar-data-monitor--expanded')).toBe(true);
    });
  });

  describe('event delegation', () => {
    it('should handle button clicks via data-action attributes', () => {
      monitor.show();
      monitor.expand();

      // Find and click the minimize button
      const minimizeBtn = container.querySelector('[data-action="minimize"]') as HTMLElement;
      expect(minimizeBtn).not.toBeNull();
      minimizeBtn.click();
      expect(monitor.isExpanded()).toBe(false);

      // Find and click the expand button
      monitor.minimize();
      const expandBtn = container.querySelector('[data-action="expand"]') as HTMLElement;
      expect(expandBtn).not.toBeNull();
      expandBtn.click();
      expect(monitor.isExpanded()).toBe(true);

      // Find and click the hide button
      const hideBtn = container.querySelector('[data-action="hide"]') as HTMLElement;
      expect(hideBtn).not.toBeNull();
      hideBtn.click();
      expect(monitor.isVisible()).toBe(false);
    });

    it('should not expose any global window variables', () => {
      // Create a new monitor to ensure it doesn't pollute global scope
      const testMonitor = new DataLoadingMonitor(container);

      const windowKeys = Object.keys(window);
      const luxarKeys = windowKeys.filter(
        (key) => key.includes('luxar') || key.includes('Monitor')
      );

      // Should not find any monitor-related global variables
      expect(luxarKeys).not.toContain('__luxarDataMonitor');
      expect(luxarKeys).not.toContain('__luxarMonitorManager');

      testMonitor.dispose();
    });
  });

  describe('metrics aggregation', () => {
    it('should aggregate metrics from multiple loaders', () => {
      const createMockLoader = (path: string, type: 'point-spatial-index'): LoaderMonitor => ({
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type,
          path,
          queries: 10,
          loads: 5,
          cacheHits: 0, // L0 cache removed
          cacheMisses: 0, // L0 cache removed
          evictions: 2,
          errors: 1,
          elementsLoaded: 10000,
          bytesLoaded: 40000,
          visibleElements: 0,
          avgQueryTime: 25,
          avgLoadTime: 100,
          cacheHitRate: 0, // L0 cache removed
          memoryUsed: 1024 * 1024,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      });

      monitor.connectLoader('/points1', createMockLoader('/points1', 'point-spatial-index'));
      monitor.connectLoader('/points2', createMockLoader('/points2', 'point-spatial-index'));

      const globalStats = monitor.getGlobalStats();
      expect(globalStats.totalQueries).toBe(20);
      expect(globalStats.totalLoads).toBe(10);
      expect(globalStats.totalPoints).toBe(20000);
      // Resident memory aggregates each loader's memoryUsed (2 × 1MB). This is
      // the figure the compact badge renders.
      expect(globalStats.totalMemory).toBe(2 * 1024 * 1024);
    });
  });

  describe('LOD loader-count collapse (Fix 4)', () => {
    const spatialLoader = (path: string): LoaderMonitor => ({
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      getMetrics: vi.fn(
        (): LoaderMetrics => ({
          type: 'gsplats-spatial-index',
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
          memoryLimit: 0,
        })
      ),
      getActiveQueries: vi.fn(() => []),
    });

    it('collapses a kind=lod group by its actual present loaders, even for multi-leaf levels', () => {
      // A substitutive LOD group at /lod with 3 levels, but the finest
      // level is itself a 2-leaf subtree → 4 loaders nested under /lod. Plus
      // one unrelated plain loader. The headline must count the group as ONE
      // logical layer (excess = present − 1 = 3), not (levelCount − 1 = 2)
      // which would leave the count high.
      monitor.connectLoader('/lod/l0', spatialLoader('/lod/l0'));
      monitor.connectLoader('/lod/l1', spatialLoader('/lod/l1'));
      monitor.connectLoader('/lod/l2/a', spatialLoader('/lod/l2/a'));
      monitor.connectLoader('/lod/l2/b', spatialLoader('/lod/l2/b'));
      monitor.connectLoader('/points', spatialLoader('/points'));

      // Provider reports the group with levelCount=3 (the OLD basis).
      (
        monitor as unknown as { lodStates: Map<string, { kind: string; levelCount: number }> }
      ).lodStates = new Map([['/lod', { kind: 'lod', levelCount: 3 }]]);

      const stats = monitor.getGlobalStats();
      // 5 loaders − (4 present under /lod − 1) = 2 logical layers.
      expect(stats.totalLoaders).toBe(2);
      expect(stats.activeSpatialLoaders).toBe(2);
    });

    it('does not over-subtract activeSpatial when a non-spatial loader nests under a LOD group', () => {
      // Defensive: today LoaderType is spatial-index only, but a future
      // non-spatial loader nested under a kind=lod path must be subtracted
      // from totalLoaders (all loaders) WITHOUT being subtracted from
      // activeSpatial (spatial-typed only). The two excesses are drawn from
      // matching populations so activeSpatial isn't driven below its true
      // spatial count. Forced via a type cast since the union can't yet
      // express a non-spatial loader.
      const nonSpatialLoader = (path: string): LoaderMonitor => ({
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(
          (): LoaderMetrics => ({
            type: 'gsplats-chunk-index' as unknown as LoaderMetrics['type'],
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
            memoryLimit: 0,
          })
        ),
        getActiveQueries: vi.fn(() => []),
      });

      // /lod has 2 spatial level loaders + 1 non-spatial nested loader = 3
      // present (2 spatial). Plus an unrelated plain spatial loader.
      monitor.connectLoader('/lod/l0', spatialLoader('/lod/l0'));
      monitor.connectLoader('/lod/l1', spatialLoader('/lod/l1'));
      monitor.connectLoader('/lod/aux', nonSpatialLoader('/lod/aux'));
      monitor.connectLoader('/points', spatialLoader('/points'));

      (
        monitor as unknown as { lodStates: Map<string, { kind: string; levelCount: number }> }
      ).lodStates = new Map([['/lod', { kind: 'lod', levelCount: 2 }]]);

      const stats = monitor.getGlobalStats();
      // totalLoaders: 4 − (3 present − 1) = 2.
      expect(stats.totalLoaders).toBe(2);
      // activeSpatial: 3 spatial loaders − (2 spatial present under /lod − 1)
      // = 2. The non-spatial loader is NOT subtracted from activeSpatial.
      expect(stats.activeSpatialLoaders).toBe(2);
    });
  });

  describe('performance tracking', () => {
    it('should track query performance', async () => {
      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn((listener) => {
          // Simulate query with latency
          const event: MonitorEvent = {
            type: 'query',
            loader: 'point-spatial-index',
            timestamp: Date.now(),
            data: {
              path: '/test',
              cells: 5,
              elements: 500,
              latency: 75,
            },
          };
          listener(event);
        }),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 1,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          elementsLoaded: 500,
          bytesLoaded: 2000,
          visibleElements: 0,
          avgQueryTime: 75,
          avgLoadTime: 0,
          cacheHitRate: 0,
          memoryUsed: 2000,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      const metrics = monitor.getLoaderMetrics('/test');
      expect(metrics?.avgQueryTime).toBe(75);
    });
  });

  describe('recommendations', () => {
    it('should generate recommendations based on metrics', () => {
      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 100,
          loads: 90,
          cacheHits: 0, // L0 cache removed
          cacheMisses: 0, // L0 cache removed
          evictions: 50,
          errors: 0,
          elementsLoaded: 90000,
          bytesLoaded: 360000,
          visibleElements: 0,
          avgQueryTime: 200,
          avgLoadTime: 150,
          cacheHitRate: 0, // L0 cache removed
          memoryUsed: 450 * 1024 * 1024,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      const recommendations = monitor.getRecommendations();

      // [data.md/W7][P2] Strengthen: toBeDefined on a `Array.find()` result
      // catches "no match" but lets through partially-malformed recs (e.g.
      // missing `message` or `suggestion` strings, undefined severity).
      // Pin the carrier-bag shape of each recommendation we expect.

      // Should recommend spatial indexing for slow fallback loader
      const spatialRec = recommendations.find((r) =>
        r.suggestion?.toLowerCase().includes('spatial')
      );
      expect(spatialRec).toBeDefined();
      expect(typeof spatialRec!.message).toBe('string');
      expect(spatialRec!.message.length).toBeGreaterThan(0);
      expect(typeof spatialRec!.suggestion).toBe('string');
      expect(spatialRec!.suggestion!.toLowerCase()).toContain('spatial');

      // L0 cache removed - no longer expect cache recommendations
      // const cacheRec = recommendations.find((r) => r.message.toLowerCase().includes('cache'));
      // expect(cacheRec).toBeDefined();

      // Should warn about high memory usage
      const memoryRec = recommendations.find((r) => r.message.toLowerCase().includes('memory'));
      expect(memoryRec).toBeDefined();
      expect(typeof memoryRec!.message).toBe('string');
      expect(memoryRec!.message.toLowerCase()).toContain('memory');
    });
  });

  describe('cycleState', () => {
    it('should cycle through states: hidden → mini → expanded → hidden', () => {
      // Initial state: hidden
      expect(monitor.isVisible()).toBe(false);
      expect(monitor.isExpanded()).toBe(false);

      // Hidden → Mini
      monitor.cycleState();
      expect(monitor.isVisible()).toBe(true);
      expect(monitor.isExpanded()).toBe(false);

      // Mini → Expanded
      monitor.cycleState();
      expect(monitor.isVisible()).toBe(true);
      expect(monitor.isExpanded()).toBe(true);

      // Expanded → Hidden
      monitor.cycleState();
      expect(monitor.isVisible()).toBe(false);
      // Note: isExpanded state is preserved when hidden
      expect(monitor.isExpanded()).toBe(true);

      // Verify it cycles back: Hidden → Mini
      monitor.cycleState();
      expect(monitor.isVisible()).toBe(true);
      expect(monitor.isExpanded()).toBe(false);
    });

    it('should update UI when cycling states', () => {
      // [data.md/W7][P2] Previously asserted only `calls.length > 0`. A
      // mutation that called updateUI only once for three cycleState calls
      // would survive. Pin EXACT count (one updateUI per cycleState) and
      // verify the DOM mutation that updateUI is responsible for: the
      // expanded/compact class on the panel.
      const updateUISpy = vi.spyOn(monitor as any, 'updateUI');
      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement;

      monitor.cycleState(); // Hidden -> Mini
      expect(panel.classList.contains('luxar-data-monitor--compact')).toBe(true);

      monitor.cycleState(); // Mini -> Expanded
      expect(panel.classList.contains('luxar-data-monitor--expanded')).toBe(true);

      monitor.cycleState(); // Expanded -> Hidden
      // Three cycleState calls -> at least three updateUI invocations
      // (the implementation may call updateUI more than once per cycle —
      // pin it to AT LEAST three so we catch a missed call).
      expect(updateUISpy.mock.calls.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('disconnectAllLoaders', () => {
    it('should disconnect all connected loaders', () => {
      const mockLoader1 = {
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

      const mockLoader2 = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test2',
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

      // Connect loaders
      monitor.connectLoader('/test1', mockLoader1);
      monitor.connectLoader('/test2', mockLoader2);

      // Verify they're connected
      expect(mockLoader1.addEventListener).toHaveBeenCalled();
      expect(mockLoader2.addEventListener).toHaveBeenCalled();

      // Disconnect all
      monitor.disconnectAllLoaders();

      // Verify they're disconnected
      expect(mockLoader1.removeEventListener).toHaveBeenCalled();
      expect(mockLoader2.removeEventListener).toHaveBeenCalled();

      // Verify state is cleared
      const globalStats = monitor.getGlobalStats();
      expect(globalStats.totalLoaders).toBe(0);
      expect(globalStats.totalQueries).toBe(0);
      expect(globalStats.totalMemory).toBe(0);
    });

    it('should clear metrics and queries but keep events', () => {
      const mockLoader = {
        addEventListener: vi.fn((listener) => {
          // Simulate an event
          const event: MonitorEvent = {
            type: 'query',
            loader: 'point-spatial-index',
            timestamp: Date.now(),
            data: { path: '/test', elements: 100 },
          };
          listener(event);
        }),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 1,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          elementsLoaded: 100,
          bytesLoaded: 400,
          visibleElements: 0,
          avgQueryTime: 10,
          avgLoadTime: 0,
          cacheHitRate: 0,
          memoryUsed: 1024,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      // Verify we have metrics
      // [data.md/W7][P2] Strengthen: toBeDefined passed even if getLoaderMetrics
      // returned an empty object. Pin the type + path that the connected
      // loader claimed it would emit.
      const metricsBefore = monitor.getLoaderMetrics('/test');
      expect(metricsBefore).toBeDefined();
      expect(metricsBefore!.type).toBe('point-spatial-index');
      expect(metricsBefore!.path).toBe('/test');
      expect(metricsBefore!.queries).toBe(1);
      expect(metricsBefore!.elementsLoaded).toBe(100);

      // Process queued events (with polling architecture, events are queued until tick)
      monitor.show(); // Make visible so forceUpdate works
      monitor.forceUpdate();

      // Events should exist after processing
      const eventsBefore = monitor.getRecentEvents();
      expect(eventsBefore.length).toBeGreaterThan(0);

      // Disconnect all
      monitor.disconnectAllLoaders();

      // Metrics should be cleared
      const metricsAfter = monitor.getLoaderMetrics('/test');
      expect(metricsAfter).toBeUndefined();

      // Events should still exist (for debugging)
      const eventsAfter = monitor.getRecentEvents();
      expect(eventsAfter.length).toBeGreaterThan(0);
    });

    it('should update UI if visible', () => {
      monitor.show();
      const updateUISpy = vi.spyOn(monitor as any, 'updateUI');

      monitor.disconnectAllLoaders();

      expect(updateUISpy).toHaveBeenCalled();
    });

    it('should clear advisor', () => {
      const advisorClearSpy = vi.spyOn((monitor as any).advisor, 'clear');

      monitor.disconnectAllLoaders();

      expect(advisorClearSpy).toHaveBeenCalled();
    });
  });

  describe('panel width stability', () => {
    it('should maintain expanded class when switching tabs in expanded mode', () => {
      // [data.md/W7][P2] Previously `expect(panel).toBeDefined()` — but the
      // cast-to-HTMLElement returns `null` if the selector misses, and
      // toBeDefined treats null as "defined". A mutation that stopped the
      // panel from being mounted would survive. Pin instanceof + non-null.
      monitor.show();
      monitor.expand();

      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement | null;
      expect(panel).not.toBeNull();
      expect(panel).toBeInstanceOf(HTMLElement);

      // Check that panel has expanded class (width defined in CSS)
      expect(panel!.classList.contains('luxar-data-monitor--expanded')).toBe(true);

      // Switch through all tabs and verify expanded class remains
      const tabs = ['overview', 'cache', 'performance', 'insights'];
      tabs.forEach((tab) => {
        monitor.setActiveTab(tab);
        expect(panel!.classList.contains('luxar-data-monitor--expanded')).toBe(true);
      });
    });

    it('should have compact class in compact mode', () => {
      // [data.md/W7][P2] See note above — toBeDefined on a `... as HTMLElement`
      // does not catch a missing selector; assert instanceof + non-null.
      monitor.show();
      monitor.minimize();

      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement | null;
      expect(panel).not.toBeNull();
      expect(panel).toBeInstanceOf(HTMLElement);

      // Check that panel has compact class (width defined in CSS)
      expect(panel!.classList.contains('luxar-data-monitor--compact')).toBe(true);
      expect(panel!.classList.contains('luxar-data-monitor--expanded')).toBe(false);
    });

    it('should update panel classes when transitioning between states', () => {
      // [data.md/W7][P2] Strengthen panel existence assertion.
      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement | null;
      expect(panel).not.toBeNull();
      expect(panel).toBeInstanceOf(HTMLElement);

      // Start hidden
      monitor.hide();

      // Hidden -> Mini (compact mode)
      monitor.cycleState();
      expect(panel!.classList.contains('luxar-data-monitor--compact')).toBe(true);

      // Mini -> Expanded
      monitor.cycleState();
      expect(panel!.classList.contains('luxar-data-monitor--expanded')).toBe(true);

      // Expanded -> Hidden: mini class flips off, but isExpanded preserved.
      // Pin: when hidden, the panel is no longer flagged compact (mini's role).
      monitor.cycleState();
      expect(panel!.classList.contains('luxar-data-monitor--compact')).toBe(false);
    });
  });

  describe('cleanup', () => {
    it('should dispose properly', () => {
      // [data.md/W7][P2] Previously asserted only .not.toThrow() on show/hide
      // after dispose. Strengthen by also pinning the observable disposal
      // contract: the loader's removeEventListener was called, and the panel
      // DOM was removed from the container.
      const fakeLoader = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({}) as LoaderMetrics),
        getActiveQueries: vi.fn(() => []),
      };
      monitor.connectLoader('/test1', fakeLoader);
      expect(container.querySelector('.luxar-data-monitor')).not.toBeNull();

      monitor.dispose();

      // Loader was actually disconnected.
      expect(fakeLoader.removeEventListener).toHaveBeenCalledTimes(1);
      // Panel removed from DOM (disposal contract).
      expect(container.querySelector('.luxar-data-monitor')).toBeNull();
      // Post-dispose calls are inert (no throw, no resurrected panel).
      expect(() => monitor.show()).not.toThrow();
      expect(() => monitor.hide()).not.toThrow();
      expect(container.querySelector('.luxar-data-monitor')).toBeNull();
    });

    it('should handle errors during disposal gracefully', () => {
      // Create loaders that will throw during cleanup
      const failingLoader: LoaderMonitor = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(() => {
          throw new Error('Simulated removeEventListener failure');
        }),
        getMetrics: vi.fn(() => ({}) as LoaderMetrics),
        getActiveQueries: vi.fn(() => []),
      };

      const normalLoader: LoaderMonitor = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({}) as LoaderMetrics),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/failing', failingLoader);
      monitor.connectLoader('/normal', normalLoader);

      // Mock console.warn to verify error logging
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Disposal should not throw despite individual failures
      expect(() => monitor.dispose()).not.toThrow();

      // Verify that error was logged
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('[DataLoadingMonitor] Disposal completed with errors:')
      );
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining("Failed to disconnect loader '/failing'")
      );

      // Verify that the normal loader was still cleaned up
      expect(normalLoader.removeEventListener).toHaveBeenCalled();

      // Restore console.warn
      consoleWarnSpy.mockRestore();
    });

    it('should handle panel removal errors gracefully', () => {
      monitor.show();

      // Mock panel.remove to throw an error
      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement;
      if (panel) {
        panel.remove = vi.fn(() => {
          throw new Error('Simulated DOM removal failure');
        });

        // Also break the fallback removeChild method
        if (panel.parentNode) {
          (panel.parentNode as any).removeChild = vi.fn(() => {
            throw new Error('Simulated removeChild failure');
          });
        }
      }

      // Mock console.warn
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Should not throw
      expect(() => monitor.dispose()).not.toThrow();

      // Verify error was logged
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        expect.stringContaining('[DataLoadingMonitor] Disposal completed with errors:')
      );

      consoleWarnSpy.mockRestore();
    });

    it('should throw only for critical errors', () => {
      // Mock the loaders.clear() method to simulate a critical failure
      const originalClear = Map.prototype.clear;
      Map.prototype.clear = vi.fn(function (this: Map<any, any>) {
        if (this === (monitor as any).loaders) {
          throw new Error('Critical: Failed to clear loaders');
        }
        return originalClear.call(this);
      });

      // Mock console.warn
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Should throw for critical error
      expect(() => monitor.dispose()).toThrow(
        'DataLoadingMonitor disposal failed with 1 critical error(s)'
      );

      // Restore Map.prototype.clear
      Map.prototype.clear = originalClear;
      consoleWarnSpy.mockRestore();
    });

    it('should clean up all resources even with multiple failures', () => {
      // Connect multiple loaders
      const loaders = Array.from({ length: 5 }, (_, i) => ({
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(() => {
          if (i % 2 === 0) {
            throw new Error(`Loader ${i} cleanup failed`);
          }
        }),
        getMetrics: vi.fn(() => ({}) as LoaderMetrics),
        getActiveQueries: vi.fn(() => []),
      }));

      loaders.forEach((loader, i) => {
        monitor.connectLoader(`/test${i}`, loader);
      });

      monitor.show();

      // Mock console.warn
      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      // Dispose should complete despite failures
      expect(() => monitor.dispose()).not.toThrow();

      // Verify all loaders attempted cleanup
      loaders.forEach((loader) => {
        expect(loader.removeEventListener).toHaveBeenCalled();
      });

      // Verify errors were logged
      expect(consoleWarnSpy).toHaveBeenCalled();

      consoleWarnSpy.mockRestore();
    });
  });

  describe('scene graph functionality', () => {
    it('should set and get scene graph', () => {
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/points1',
            name: 'points1',
            type: 'points' as const,
            pointCount: 1000,
            children: [],
          },
          {
            path: '/group1',
            name: 'group1',
            type: 'group' as const,
            children: [
              {
                path: '/group1/points2',
                name: 'points2',
                type: 'points' as const,
                pointCount: 500,
                children: [],
              },
            ],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);
      const state = monitor.getSceneGraph();

      expect(state.root).toBeDefined();
      expect(state.root?.path).toBe('/');
      expect(state.totalNodes).toBe(4); // Scene + points1 + group1 + points2
      expect(state.pointsNodes).toBe(2); // points1 + points2
      expect(state.totalPoints).toBe(1500); // 1000 + 500
    });

    it('does NOT sum substitutive kind=lod levels — counts the finest only', () => {
      // A kind=lod group's children are mutually-exclusive representations
      // of the same data at different resolutions. Summing them would
      // inflate the dataset total ~K×. The aggregator must take the finest
      // (last, coarsest→finest order) level instead.
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/lod_splats',
            name: 'lod_splats',
            type: 'group' as const,
            kind: 'lod' as const,
            lodGroupChildCount: 3,
            children: [
              {
                path: '/lod_splats/l0',
                name: 'l0',
                type: 'gsplats' as const,
                splatCount: 1_000,
                children: [],
              },
              {
                path: '/lod_splats/l1',
                name: 'l1',
                type: 'gsplats' as const,
                splatCount: 4_000,
                children: [],
              },
              {
                path: '/lod_splats/l2',
                name: 'l2',
                type: 'gsplats' as const,
                splatCount: 20_000,
                children: [],
              },
            ],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);
      const state = monitor.getSceneGraph();

      // Finest level (20_000), NOT 1_000 + 4_000 + 20_000 = 25_000.
      expect(state.totalSplats).toBe(20_000);
      // Structural counts still reflect the real tree (all 3 levels).
      expect(state.gsplatsNodes).toBe(3);
      expect(state.totalNodes).toBe(5); // Scene + lod group + 3 levels
    });

    it('DOES sum partition parts — disjoint BSP parts add up', () => {
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/parted',
            name: 'parted',
            type: 'group' as const,
            kind: 'partition' as const,
            partCount: 2,
            children: [
              {
                path: '/parted/p0',
                name: 'p0',
                type: 'points' as const,
                pointCount: 600,
                children: [],
              },
              {
                path: '/parted/p1',
                name: 'p1',
                type: 'points' as const,
                pointCount: 400,
                children: [],
              },
            ],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);
      const state = monitor.getSceneGraph();

      expect(state.totalPoints).toBe(1000); // 600 + 400 (disjoint parts)
    });

    it('should track expanded nodes', () => {
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [{ path: '/points', name: 'points', type: 'points' as const, children: [] }],
      };

      monitor.setSceneGraph(sceneGraph);

      // Root should be expanded by default
      expect(monitor.isNodeExpanded('/')).toBe(true);

      // Toggle expansion
      monitor.toggleNodeExpansion('/');
      expect(monitor.isNodeExpanded('/')).toBe(false);

      monitor.toggleNodeExpansion('/');
      expect(monitor.isNodeExpanded('/')).toBe(true);

      // New nodes start collapsed
      expect(monitor.isNodeExpanded('/points')).toBe(false);
      monitor.toggleNodeExpansion('/points');
      expect(monitor.isNodeExpanded('/points')).toBe(true);
    });

    it('should count lines nodes correctly', () => {
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/lines1',
            name: 'lines1',
            type: 'lines' as const,
            segmentCount: 100,
            children: [],
          },
          {
            path: '/lines2',
            name: 'lines2',
            type: 'lines' as const,
            segmentCount: 200,
            children: [],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);
      const state = monitor.getSceneGraph();

      expect(state.linesNodes).toBe(2);
      expect(state.totalSegments).toBe(300);
    });

    it('should track visible points separately from total points', () => {
      // Set up scene with points
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/points1',
            name: 'points1',
            type: 'points' as const,
            pointCount: 200000,
            children: [],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);

      // Initially visibleElements equals totalPoints
      let state = monitor.getSceneGraph();
      expect(state.totalPoints).toBe(200000);
      expect(state.visibleElements).toBe(200000);

      // Update visible points (simulating nD slicing / progressive LOD)
      monitor.updateVisiblePoints(50000);

      state = monitor.getSceneGraph();
      expect(state.totalPoints).toBe(200000); // Total unchanged
      expect(state.visibleElements).toBe(50000); // Only visible count updated

      // getGlobalStats should source points from the scene graph, so a
      // progressive points loader (no LoaderMonitor surface) still reports.
      const stats = monitor.getGlobalStats();
      expect(stats.datasetSize).toBe(200000);
      expect(stats.visibleElements).toBe(50000);
    });

    it('clears per-node visible counts for paths absent from the latest walk', () => {
      // The SceneLoader's visible-counts walk prunes non-visible subtrees,
      // so a hidden layer or switched-away substitutive level simply stops
      // appearing in the pushed map. Its previously merged count must be
      // cleared (back to unknown) — not left as a stale
      // "(N visible after slicing)" tooltip forever.
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/points1',
            name: 'points1',
            type: 'points' as const,
            pointCount: 1000,
            children: [],
          },
          {
            path: '/splats1',
            name: 'splats1',
            type: 'gsplats' as const,
            splatCount: 2000,
            children: [],
          },
        ],
      };
      monitor.setSceneGraph(sceneGraph);

      const sync = () =>
        (monitor as unknown as { syncVisibleCountsIntoTree(): void }).syncVisibleCountsIntoTree();

      // First walk: both layers rendered with partial visibility.
      monitor.updateVisibleCountsByPath(
        new Map([
          ['/points1', 250],
          ['/splats1', 700],
        ])
      );
      sync();
      const root = monitor.getSceneGraph().root!;
      const points = root.children[0];
      const splats = root.children[1];
      expect(points.visiblePointCount).toBe(250);
      expect(splats.visibleSplatCount).toBe(700);
      expect(nodeStatsContent(points)!.title).toContain('250 visible after slicing');
      expect(nodeStatsContent(splats)!.title).toContain('700 visible after slicing');

      // Second walk: the gsplats layer was hidden (pruned from the walk).
      // Its count must read as unknown (no suffix), not the stale 700.
      monitor.updateVisibleCountsByPath(new Map([['/points1', 100]]));
      sync();
      expect(points.visiblePointCount).toBe(100);
      expect(splats.visibleSplatCount).toBeUndefined();
      expect(nodeStatsContent(points)!.title).toContain('100 visible after slicing');
      expect(nodeStatsContent(splats)!.title).not.toContain('visible');
    });

    it('re-marks active/inactive substitutive level rows per tick (shared activeLevelRole derivation)', () => {
      // The per-tick patcher must derive each level row's role exactly like
      // the initial render (both call templates.ts's exported
      // activeLevelRole) and flip the marks in place when the LOD selector
      // switches levels between structural rebuilds.
      monitor.show();
      monitor.expand(); // Detailed view — the overview tab hosts the tree

      monitor.setSceneGraph({
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/lod',
            name: 'lod',
            type: 'group' as const,
            kind: 'lod' as const,
            lodGroupChildCount: 2,
            children: [
              {
                path: '/lod/l0',
                name: 'l0',
                type: 'gsplats' as const,
                splatCount: 100,
                children: [],
              },
              {
                path: '/lod/l1',
                name: 'l1',
                type: 'gsplats' as const,
                splatCount: 400,
                children: [],
              },
            ],
          },
        ],
      });
      monitor.toggleNodeExpansion('/lod'); // Render the level rows

      const states = new Map<string, LODProgressState>([
        ['/lod', { kind: 'lod', levelCount: 2, activeLevel: 0 }],
      ]);
      monitor.setLODProgressProvider({ getLODStates: () => states });
      const tick = () => (monitor as unknown as { onPollingTick(): void }).onPollingTick();

      const activeClass = 'luxar-scene-graph__node-row--active-level';
      const inactiveClass = 'luxar-scene-graph__node-row--inactive-level';
      const rows = () =>
        Array.from(container.querySelectorAll('[data-level-of="/lod"]')) as HTMLElement[];

      tick();
      let levelRows = rows();
      expect(levelRows.length).toBe(2);
      expect(levelRows[0].classList.contains(activeClass)).toBe(true);
      expect(levelRows[1].classList.contains(inactiveClass)).toBe(true);
      expect(levelRows[0].title).toContain('active substitutive level');

      // Selector switches to the fine level — no structural rebuild, the
      // patcher must flip both rows' classes and tooltips.
      states.set('/lod', { kind: 'lod', levelCount: 2, activeLevel: 1 });
      tick();
      levelRows = rows();
      expect(levelRows[0].classList.contains(activeClass)).toBe(false);
      expect(levelRows[0].classList.contains(inactiveClass)).toBe(true);
      expect(levelRows[1].classList.contains(activeClass)).toBe(true);
      expect(levelRows[1].classList.contains(inactiveClass)).toBe(false);
      expect(levelRows[0].title).toContain('inactive substitutive level');
      expect(levelRows[1].title).toContain('active substitutive level');
    });

    it('should report visible points in getGlobalStats across multiple nodes', () => {
      // Set up scene with multiple points nodes
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/points1',
            name: 'points1',
            type: 'points' as const,
            pointCount: 500,
            children: [],
          },
          {
            path: '/points2',
            name: 'points2',
            type: 'points' as const,
            pointCount: 500,
            children: [],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);

      // Update with combined visible count
      monitor.updateVisiblePoints(200);

      const stats = monitor.getGlobalStats();
      expect(stats.datasetSize).toBe(1000); // 500 + 500
      expect(stats.visibleElements).toBe(200); // Only what's visible
    });

    it('should track visible segments separately from total segments', () => {
      // Set up scene with lines
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/lines1',
            name: 'lines1',
            type: 'lines' as const,
            segmentCount: 1000,
            children: [],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);

      // Initially visibleSegments equals totalSegments
      let state = monitor.getSceneGraph();
      expect(state.totalSegments).toBe(1000);
      expect(state.visibleSegments).toBe(1000);

      // Update visible segments (simulating nD slicing that hides some segments)
      monitor.updateVisibleSegments(150);

      state = monitor.getSceneGraph();
      expect(state.totalSegments).toBe(1000); // Total unchanged
      expect(state.visibleSegments).toBe(150); // Only visible count updated

      // getGlobalStats should return the tracked visible count
      const stats = monitor.getGlobalStats();
      expect(stats.datasetSegments).toBe(1000);
      expect(stats.visibleSegments).toBe(150);
    });

    it('should report visible segments in getGlobalStats', () => {
      // Set up scene with multiple lines nodes
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/lines1',
            name: 'lines1',
            type: 'lines' as const,
            segmentCount: 500,
            children: [],
          },
          {
            path: '/lines2',
            name: 'lines2',
            type: 'lines' as const,
            segmentCount: 500,
            children: [],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);

      // Update with combined visible count
      monitor.updateVisibleSegments(200);

      const stats = monitor.getGlobalStats();
      expect(stats.datasetSegments).toBe(1000); // 500 + 500
      expect(stats.visibleSegments).toBe(200); // Only what's visible
    });

    it('should track visible splats separately from total splats', () => {
      // Set up scene with gsplats
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/gsplats1',
            name: 'gsplats1',
            type: 'gsplats' as const,
            splatCount: 5000,
            children: [],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);

      // Initially visibleSplats equals totalSplats
      let state = monitor.getSceneGraph();
      expect(state.totalSplats).toBe(5000);
      expect(state.visibleSplats).toBe(5000);

      // Update visible splats (simulating nD slicing that hides some splats)
      monitor.updateVisibleSplats(1200);

      state = monitor.getSceneGraph();
      expect(state.totalSplats).toBe(5000); // Total unchanged
      expect(state.visibleSplats).toBe(1200); // Only visible count updated

      // getGlobalStats should return the tracked visible count
      const stats = monitor.getGlobalStats();
      expect(stats.datasetSplats).toBe(5000);
      expect(stats.visibleSplats).toBe(1200);
    });

    it('should report visible splats in getGlobalStats', () => {
      // Set up scene with multiple gsplats nodes
      const sceneGraph = {
        path: '/',
        name: 'Scene',
        type: 'scene' as const,
        children: [
          {
            path: '/gsplats1',
            name: 'gsplats1',
            type: 'gsplats' as const,
            splatCount: 3000,
            children: [],
          },
          {
            path: '/gsplats2',
            name: 'gsplats2',
            type: 'gsplats' as const,
            splatCount: 2000,
            children: [],
          },
        ],
      };

      monitor.setSceneGraph(sceneGraph);

      // Update with combined visible count
      monitor.updateVisibleSplats(1000);

      const stats = monitor.getGlobalStats();
      expect(stats.datasetSplats).toBe(5000); // 3000 + 2000
      expect(stats.visibleSplats).toBe(1000); // Only what's visible
    });
  });

  describe('cache stats provider integration', () => {
    it('should connect cache stats provider', () => {
      const mockProvider = {
        getStats: vi.fn(() => ({
          l1: {
            metadataSize: 1024,
            chunksSize: 10240,
            metadataCount: 5,
            chunksCount: 50,
            hits: 100,
            misses: 20,
            evictions: 5,
          },
          l2: { size: 1024 * 1024, count: 100, reads: 80, writes: 50, misses: 20 },
          network: { bytesTransferred: 5000000, requestCount: 100, bandwidth: 100000 },
        })),
        clearL1: vi.fn(),
        clearL2: vi.fn(() => Promise.resolve()),
        clearAll: vi.fn(() => Promise.resolve()),
        isEnabled: vi.fn(() => true),
      };

      monitor.setCacheStatsProvider(mockProvider);

      // Verify provider is connected (will be used during UI rendering)
      expect(mockProvider.getStats).not.toHaveBeenCalled(); // Not called until needed
    });

    it('should call clearL1 on provider', () => {
      const mockProvider = {
        getStats: vi.fn(() => ({
          l1: {
            metadataSize: 0,
            chunksSize: 0,
            metadataCount: 0,
            chunksCount: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
          },
          l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
          network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
        })),
        clearL1: vi.fn(),
        clearL2: vi.fn(() => Promise.resolve()),
        clearAll: vi.fn(() => Promise.resolve()),
        isEnabled: vi.fn(() => true),
      };

      monitor.setCacheStatsProvider(mockProvider);
      monitor.clearL1Cache();

      expect(mockProvider.clearL1).toHaveBeenCalledTimes(1);
    });

    it('should call clearL2 on provider', async () => {
      const mockProvider = {
        getStats: vi.fn(() => ({
          l1: {
            metadataSize: 0,
            chunksSize: 0,
            metadataCount: 0,
            chunksCount: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
          },
          l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
          network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
        })),
        clearL1: vi.fn(),
        clearL2: vi.fn(() => Promise.resolve()),
        clearAll: vi.fn(() => Promise.resolve()),
        isEnabled: vi.fn(() => true),
      };

      monitor.setCacheStatsProvider(mockProvider);
      // skipConfirm avoids the new confirm-dialog (commit 7.3); tests
      // that exercise the dialog directly are below.
      await monitor.clearL2Cache({ skipConfirm: true });

      expect(mockProvider.clearL2).toHaveBeenCalledTimes(1);
    });

    it('should call clearAll on provider', async () => {
      const mockProvider = {
        getStats: vi.fn(() => ({
          l1: {
            metadataSize: 0,
            chunksSize: 0,
            metadataCount: 0,
            chunksCount: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
          },
          l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
          network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
        })),
        clearL1: vi.fn(),
        clearL2: vi.fn(() => Promise.resolve()),
        clearAll: vi.fn(() => Promise.resolve()),
        isEnabled: vi.fn(() => true),
      };

      monitor.setCacheStatsProvider(mockProvider);
      await monitor.clearAllCaches({ skipConfirm: true });

      expect(mockProvider.clearAll).toHaveBeenCalledTimes(1);
    });

    it('clearL2Cache without skipConfirm respects window.confirm cancel (commit 7.3)', async () => {
      const mockProvider = {
        getStats: vi.fn(() => ({
          l1: {
            metadataSize: 0,
            chunksSize: 0,
            metadataCount: 0,
            chunksCount: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
          },
          l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
          network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
        })),
        clearL1: vi.fn(),
        clearL2: vi.fn(() => Promise.resolve()),
        clearAll: vi.fn(() => Promise.resolve()),
        isEnabled: vi.fn(() => true),
      };
      monitor.setCacheStatsProvider(mockProvider);
      const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
      try {
        await monitor.clearL2Cache();
        expect(confirmSpy).toHaveBeenCalled();
        expect(mockProvider.clearL2).not.toHaveBeenCalled();
      } finally {
        confirmSpy.mockRestore();
      }
    });

    it('should handle missing provider gracefully', () => {
      // No provider set
      expect(() => monitor.clearL1Cache()).not.toThrow();
      expect(() => monitor.clearL2Cache()).not.toThrow();
      expect(() => monitor.clearAllCaches()).not.toThrow();
    });
  });
});
