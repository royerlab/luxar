/**
 * Unit tests for the Data Loading Monitor
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { DataLoadingMonitor } from '../ui/data-loading-monitor';
import type { MonitorEvent, LoaderMonitor, LoaderMetrics } from '../ui/data-monitor-types';

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
      const customMonitor = new DataLoadingMonitor(container, {
        position: 'top-left',
        theme: 'light',
        defaultView: 'detailed',
        maxEvents: 500,
      });
      expect(customMonitor).toBeDefined();
    });
  });

  describe('loader connection', () => {
    it('should connect and disconnect loaders', () => {
      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'spatial-index' as const,
          path: '/test',
          queries: 0,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          pointsLoaded: 0,
          bytesLoaded: 0,
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
            loader: 'spatial-index',
            timestamp: Date.now(),
            data: {
              path: '/test',
              cells: 10,
              points: 1000,
              latency: 50,
            },
          };
          listener(event);
        }),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'spatial-index' as const,
          path: '/test',
          queries: 1,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          pointsLoaded: 1000,
          bytesLoaded: 0,
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
              loader: 'spatial-index',
              timestamp: Date.now(),
              data: { path: '/test' },
            };
            listener(event);
          }
        }),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'spatial-index' as const,
          path: '/test',
          queries: 0,
          loads: 0,
          cacheHits: 2000,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          pointsLoaded: 0,
          bytesLoaded: 0,
          avgQueryTime: 0,
          avgLoadTime: 0,
          cacheHitRate: 100,
          memoryUsed: 0,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      // Events should be limited to maxEvents (default 1000)
      const events = monitor.getRecentEvents();
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
  });

  describe('event delegation', () => {
    it('should handle button clicks via data-action attributes', () => {
      monitor.show();
      monitor.expand();

      // Find and click the minimize button
      const minimizeBtn = container.querySelector('[data-action="minimize"]') as HTMLElement;
      if (minimizeBtn) {
        minimizeBtn.click();
        expect(monitor.isExpanded()).toBe(false);
      }

      // Find and click the expand button
      monitor.minimize();
      const expandBtn = container.querySelector('[data-action="expand"]') as HTMLElement;
      if (expandBtn) {
        expandBtn.click();
        expect(monitor.isExpanded()).toBe(true);
      }

      // Find and click the hide button
      const hideBtn = container.querySelector('[data-action="hide"]') as HTMLElement;
      if (hideBtn) {
        hideBtn.click();
        expect(monitor.isVisible()).toBe(false);
      }
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
      const createMockLoader = (path: string, type: 'spatial-index'): LoaderMonitor => ({
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type,
          path,
          queries: 10,
          loads: 5,
          cacheHits: 20,
          cacheMisses: 10,
          evictions: 2,
          errors: 1,
          pointsLoaded: 10000,
          bytesLoaded: 40000,
          avgQueryTime: 25,
          avgLoadTime: 100,
          cacheHitRate: 66.7,
          memoryUsed: 1024 * 1024,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      });

      monitor.connectLoader('/points1', createMockLoader('/points1', 'spatial-index'));
      monitor.connectLoader('/points2', createMockLoader('/points2', 'spatial-index'));

      const globalStats = monitor.getGlobalStats();
      expect(globalStats.totalQueries).toBe(20);
      expect(globalStats.totalLoads).toBe(10);
      expect(globalStats.totalCacheHits).toBe(40);
      expect(globalStats.totalPointsLoaded).toBe(20000);
      expect(globalStats.totalMemoryUsed).toBe(2 * 1024 * 1024);
    });
  });

  describe('performance tracking', () => {
    it('should track query performance', async () => {
      const mockLoader: LoaderMonitor = {
        addEventListener: vi.fn((listener) => {
          // Simulate query with latency
          const event: MonitorEvent = {
            type: 'query',
            loader: 'spatial-index',
            timestamp: Date.now(),
            data: {
              path: '/test',
              cells: 5,
              points: 500,
              latency: 75,
            },
          };
          listener(event);
        }),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({
          type: 'spatial-index' as const,
          path: '/test',
          queries: 1,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          pointsLoaded: 500,
          bytesLoaded: 2000,
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
          type: 'spatial-index' as const,
          path: '/test',
          queries: 100,
          loads: 90,
          cacheHits: 10,
          cacheMisses: 90,
          evictions: 50,
          errors: 0,
          pointsLoaded: 90000,
          bytesLoaded: 360000,
          avgQueryTime: 200,
          avgLoadTime: 150,
          cacheHitRate: 10,
          memoryUsed: 450 * 1024 * 1024,
          memoryLimit: 500 * 1024 * 1024,
        })),
        getActiveQueries: vi.fn(() => []),
      };

      monitor.connectLoader('/test', mockLoader);

      const recommendations = monitor.getRecommendations();

      // Should recommend spatial indexing for slow fallback loader
      const spatialRec = recommendations.find((r) =>
        r.suggestion?.toLowerCase().includes('spatial')
      );
      expect(spatialRec).toBeDefined();

      // Should warn about low cache hit rate
      const cacheRec = recommendations.find((r) => r.message.toLowerCase().includes('cache'));
      expect(cacheRec).toBeDefined();

      // Should warn about high memory usage
      const memoryRec = recommendations.find((r) => r.message.toLowerCase().includes('memory'));
      expect(memoryRec).toBeDefined();
    });
  });

  describe('cleanup', () => {
    it('should dispose properly', () => {
      monitor.connectLoader('/test1', {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        getMetrics: vi.fn(() => ({}) as LoaderMetrics),
        getActiveQueries: vi.fn(() => []),
      });

      monitor.dispose();

      // Should not throw when calling methods after dispose
      expect(() => monitor.show()).not.toThrow();
      expect(() => monitor.hide()).not.toThrow();
    });
  });
});
