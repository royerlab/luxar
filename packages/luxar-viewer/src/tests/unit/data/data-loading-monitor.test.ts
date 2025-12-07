/**
 * Unit tests for the Data Loading Monitor
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { DataLoadingMonitor } from '../../../ui/data-loading-monitor';
import type { MonitorEvent, LoaderMonitor, LoaderMetrics } from '../../../ui/data-monitor-types';

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
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 0,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          pointsLoaded: 0,
          bytesLoaded: 0,
          datasetSize: 0,
          visiblePoints: 0,
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
              points: 1000,
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
          pointsLoaded: 1000,
          bytesLoaded: 0,
          datasetSize: 0,
          visiblePoints: 0,
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
          pointsLoaded: 0,
          bytesLoaded: 0,
          datasetSize: 0,
          visiblePoints: 0,
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
          pointsLoaded: 10000,
          bytesLoaded: 40000,
          datasetSize: 0,
          visiblePoints: 0,
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
      // L0 cache removed - cache hits should now be 0
      expect(globalStats.totalCacheHits).toBe(0);
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
            loader: 'point-spatial-index',
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
          type: 'point-spatial-index' as const,
          path: '/test',
          queries: 1,
          loads: 0,
          cacheHits: 0,
          cacheMisses: 0,
          evictions: 0,
          errors: 0,
          pointsLoaded: 500,
          bytesLoaded: 2000,
          datasetSize: 0,
          visiblePoints: 0,
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
          pointsLoaded: 90000,
          bytesLoaded: 360000,
          datasetSize: 0,
          visiblePoints: 0,
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

      // Should recommend spatial indexing for slow fallback loader
      const spatialRec = recommendations.find((r) =>
        r.suggestion?.toLowerCase().includes('spatial')
      );
      expect(spatialRec).toBeDefined();

      // L0 cache removed - no longer expect cache recommendations
      // const cacheRec = recommendations.find((r) => r.message.toLowerCase().includes('cache'));
      // expect(cacheRec).toBeDefined();

      // Should warn about high memory usage
      const memoryRec = recommendations.find((r) => r.message.toLowerCase().includes('memory'));
      expect(memoryRec).toBeDefined();
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
      const updateUISpy = vi.spyOn(monitor as any, 'updateUI');

      // Cycle through all states
      monitor.cycleState(); // Hidden → Mini
      monitor.cycleState(); // Mini → Expanded
      monitor.cycleState(); // Expanded → Hidden

      // updateUI should be called multiple times
      expect(updateUISpy.mock.calls.length).toBeGreaterThan(0);
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
          pointsLoaded: 0,
          bytesLoaded: 0,
          datasetSize: 0,
          visiblePoints: 0,
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
          pointsLoaded: 0,
          bytesLoaded: 0,
          datasetSize: 0,
          visiblePoints: 0,
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
            data: { path: '/test', points: 100 },
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
          pointsLoaded: 100,
          bytesLoaded: 400,
          datasetSize: 0,
          visiblePoints: 0,
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
      const metricsBefore = monitor.getLoaderMetrics('/test');
      expect(metricsBefore).toBeDefined();

      // Events should exist
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

    it('should clear timeline and advisor', () => {
      const timelineClearSpy = vi.spyOn((monitor as any).timeline, 'clear');
      const advisorClearSpy = vi.spyOn((monitor as any).advisor, 'clear');

      monitor.disconnectAllLoaders();

      expect(timelineClearSpy).toHaveBeenCalled();
      expect(advisorClearSpy).toHaveBeenCalled();
    });
  });


  describe('panel width stability', () => {
    it('should maintain fixed width when switching tabs in expanded mode', () => {
      monitor.show();
      monitor.expand();

      // Get panel element
      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement;
      expect(panel).toBeDefined();

      // Check that panel has fixed width in expanded mode
      const style = panel.style.cssText;
      expect(style).toContain('width: 480px');
      expect(style).toContain('min-width: 480px');
      expect(style).toContain('max-width: 480px');

      // Switch through all tabs and verify width remains constant
      const tabs = ['overview', 'cache', 'performance', 'insights'];
      tabs.forEach((tab) => {
        monitor.setActiveTab(tab);
        const currentStyle = panel.style.cssText;
        expect(currentStyle).toContain('width: 480px');
        expect(currentStyle).toContain('min-width: 480px');
        expect(currentStyle).toContain('max-width: 480px');
      });
    });

    it('should not have fixed width in compact mode', () => {
      monitor.show();
      monitor.minimize();

      // Get panel element
      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement;
      expect(panel).toBeDefined();

      // Check that panel has auto width in compact mode
      const style = panel.style.cssText;
      expect(style).toContain('width: auto');
      expect(style).not.toContain('width: 480px');
    });

    it('should update panel width when transitioning between states', () => {
      const panel = container.querySelector('.luxar-data-monitor') as HTMLElement;
      expect(panel).toBeDefined();

      // Start hidden
      monitor.hide();

      // Hidden → Mini (compact mode)
      monitor.cycleState();
      expect(panel.style.cssText).toContain('width: auto');

      // Mini → Expanded
      monitor.cycleState();
      expect(panel.style.cssText).toContain('width: 480px');

      // Expanded → Hidden
      monitor.cycleState();
      // Panel style should still have the width set (even if hidden)
      // because isExpanded is still true internally
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
        expect.stringContaining('Failed to disconnect loader \'/failing\'')
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
});
