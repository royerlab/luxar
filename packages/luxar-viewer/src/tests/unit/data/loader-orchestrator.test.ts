/**
 * Unit tests for LoaderOrchestrator.
 *
 * Tests loader creation, registration, error tracking, and statistics aggregation.
 * Loader implementations are mocked to isolate orchestrator logic.
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { LoaderOrchestrator, type OrchestratorConfig } from '../../../data/loader-orchestrator';
import type { DataLoader, SceneNode, LoaderConfig } from '../../../data/data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import { ArrayRefRegistry } from '../../../data/array-decoder';

// Mock the loader implementations
vi.mock('../../../data/point-spatial-index-loader', () => ({
  PointSpatialIndexLoader: vi.fn().mockImplementation(() => ({
    dispose: vi.fn(),
    getAccumulatorStats: vi.fn().mockReturnValue({
      capacity: 1000,
      allocations: 5,
      growthEvents: 2,
      memoryMB: 0.5,
    }),
  })),
}));

vi.mock('../../../data/lines-spatial-index-loader', () => ({
  LinesSpatialIndexLoader: vi.fn().mockImplementation(() => ({
    dispose: vi.fn(),
    getAccumulatorStats: vi.fn().mockReturnValue({
      capacity: 500,
      allocations: 3,
      growthEvents: 1,
      memoryMB: 0.25,
    }),
  })),
}));

vi.mock('../../../data/gsplats-spatial-index-loader', () => ({
  GSplatsSpatialIndexLoader: vi.fn().mockImplementation(() => ({
    dispose: vi.fn(),
    getAccumulatorStats: vi.fn().mockReturnValue({
      capacity: 2000,
      allocations: 8,
      growthEvents: 4,
      memoryMB: 1.0,
    }),
  })),
}));

// Mock zarr module
vi.mock('zarrita', () => ({
  root: vi.fn().mockReturnValue({
    resolve: vi.fn().mockReturnValue({}),
  }),
}));

// Mock DataMonitorManager
vi.mock('../../../data/data-monitor-manager', () => ({
  DataMonitorManager: {
    getInstance: vi.fn().mockReturnValue({
      getMonitor: vi.fn().mockReturnValue({
        connectLoader: vi.fn(),
      }),
    }),
  },
}));

// Helper to create mock config
function createMockConfig(overrides: Partial<OrchestratorConfig> = {}): OrchestratorConfig {
  return {
    loaderConfig: {
      maxPoints: 1000000,
      chunkSize: 65536,
    } as LoaderConfig,
    arrayRefRegistry: new ArrayRefRegistry(),
    store: {
      get: vi.fn(),
    } as any,
    profiler: undefined,
    monitorId: null,
    ...overrides,
  };
}

// Helper to create mock SceneNode
function createMockNode(path: string, type: string = 'points'): SceneNode {
  return {
    path,
    type,
    attrs: {},
    hasSpatialIndex: false,
    children: [],
  };
}

// Helper to create mock location
function createMockLocation() {
  return {
    resolve: vi.fn().mockReturnValue({}),
  } as any;
}

// Helper to create mock DataLoader
function createMockDataLoader(): DataLoader {
  return {
    dispose: vi.fn(),
    getAccumulatorStats: vi.fn().mockReturnValue({
      capacity: 100,
      allocations: 1,
      growthEvents: 0,
      memoryMB: 0.1,
    }),
  } as any;
}

// Helper to create mock LinesDataLoader
function createMockLinesLoader(): LinesDataLoader {
  return {
    dispose: vi.fn(),
    getAccumulatorStats: vi.fn().mockReturnValue({
      capacity: 50,
      allocations: 1,
      growthEvents: 0,
      memoryMB: 0.05,
    }),
  } as any;
}

// Helper to create mock GSplatsDataLoader
function createMockGSplatsLoader(): GSplatsDataLoader {
  return {
    dispose: vi.fn(),
    getAccumulatorStats: vi.fn().mockReturnValue({
      capacity: 200,
      allocations: 2,
      growthEvents: 1,
      memoryMB: 0.2,
    }),
  } as any;
}

describe('LoaderOrchestrator', () => {
  let orchestrator: LoaderOrchestrator;
  let config: OrchestratorConfig;

  beforeEach(() => {
    vi.clearAllMocks();
    config = createMockConfig();
    orchestrator = new LoaderOrchestrator(config);
  });

  describe('constructor', () => {
    it('should create orchestrator with config', () => {
      expect(orchestrator).toBeInstanceOf(LoaderOrchestrator);
    });

    it('should initialize with empty loader maps', () => {
      expect(orchestrator.getPointsLoaders().size).toBe(0);
      expect(orchestrator.getLinesLoaders().size).toBe(0);
      expect(orchestrator.getGSplatsLoaders().size).toBe(0);
    });

    it('should initialize with no failures', () => {
      expect(orchestrator.hasFailures()).toBe(false);
      expect(orchestrator.getFailedLoaders().size).toBe(0);
    });
  });

  describe('loader creation', () => {
    it('should create points loader', () => {
      const node = createMockNode('/points1', 'points');
      const loc = createMockLocation();

      const loader = orchestrator.createPointsLoader(node, loc);

      expect(loader).toBeDefined();
      expect(loader.dispose).toBeDefined();
    });

    it('should create lines loader', () => {
      const node = createMockNode('/lines1', 'lines');
      const loc = createMockLocation();

      const loader = orchestrator.createLinesLoader(node, loc);

      expect(loader).toBeDefined();
      expect(loader.dispose).toBeDefined();
    });

    it('should create gsplats loader', () => {
      const node = createMockNode('/gsplats1', 'gsplats');
      const loc = createMockLocation();

      const loader = orchestrator.createGSplatsLoader(node, loc);

      expect(loader).toBeDefined();
      expect(loader.dispose).toBeDefined();
    });

    it('should handle root path correctly', () => {
      const node = createMockNode('/', 'points');
      const loc = createMockLocation();

      const loader = orchestrator.createPointsLoader(node, loc);

      expect(loader).toBeDefined();
    });

    it('should connect to monitor when monitorId provided', () => {
      const configWithMonitor = createMockConfig({ monitorId: 'test-monitor' });
      const orchestratorWithMonitor = new LoaderOrchestrator(configWithMonitor);
      const node = createMockNode('/points1', 'points');
      const loc = createMockLocation();

      // Should not throw when trying to connect to monitor
      expect(() => orchestratorWithMonitor.createPointsLoader(node, loc)).not.toThrow();
    });
  });

  describe('loader registration', () => {
    it('should register points loader', () => {
      const loader = createMockDataLoader();
      orchestrator.registerPointsLoader('/points1', loader);

      expect(orchestrator.getPointsLoaders().size).toBe(1);
      expect(orchestrator.getPointsLoader('/points1')).toBe(loader);
    });

    it('should register lines loader', () => {
      const loader = createMockLinesLoader();
      orchestrator.registerLinesLoader('/lines1', loader);

      expect(orchestrator.getLinesLoaders().size).toBe(1);
      expect(orchestrator.getLinesLoader('/lines1')).toBe(loader);
    });

    it('should register gsplats loader', () => {
      const loader = createMockGSplatsLoader();
      orchestrator.registerGSplatsLoader('/gsplats1', loader);

      expect(orchestrator.getGSplatsLoaders().size).toBe(1);
      expect(orchestrator.getGSplatsLoader('/gsplats1')).toBe(loader);
    });

    it('should allow registering multiple loaders', () => {
      orchestrator.registerPointsLoader('/points1', createMockDataLoader());
      orchestrator.registerPointsLoader('/points2', createMockDataLoader());
      orchestrator.registerLinesLoader('/lines1', createMockLinesLoader());

      expect(orchestrator.getPointsLoaders().size).toBe(2);
      expect(orchestrator.getLinesLoaders().size).toBe(1);
    });

    it('should overwrite loader with same path', () => {
      const loader1 = createMockDataLoader();
      const loader2 = createMockDataLoader();

      orchestrator.registerPointsLoader('/points1', loader1);
      orchestrator.registerPointsLoader('/points1', loader2);

      expect(orchestrator.getPointsLoaders().size).toBe(1);
      expect(orchestrator.getPointsLoader('/points1')).toBe(loader2);
    });
  });

  describe('loader access', () => {
    beforeEach(() => {
      orchestrator.registerPointsLoader('/points1', createMockDataLoader());
      orchestrator.registerLinesLoader('/lines1', createMockLinesLoader());
      orchestrator.registerGSplatsLoader('/gsplats1', createMockGSplatsLoader());
    });

    it('should get all points loaders', () => {
      const loaders = orchestrator.getPointsLoaders();
      expect(loaders.size).toBe(1);
      expect(loaders.has('/points1')).toBe(true);
    });

    it('should get all lines loaders', () => {
      const loaders = orchestrator.getLinesLoaders();
      expect(loaders.size).toBe(1);
      expect(loaders.has('/lines1')).toBe(true);
    });

    it('should get all gsplats loaders', () => {
      const loaders = orchestrator.getGSplatsLoaders();
      expect(loaders.size).toBe(1);
      expect(loaders.has('/gsplats1')).toBe(true);
    });

    it('should return undefined for non-existent loader', () => {
      expect(orchestrator.getPointsLoader('/nonexistent')).toBeUndefined();
      expect(orchestrator.getLinesLoader('/nonexistent')).toBeUndefined();
      expect(orchestrator.getGSplatsLoader('/nonexistent')).toBeUndefined();
    });
  });

  describe('getLoaderType', () => {
    beforeEach(() => {
      orchestrator.registerPointsLoader('/points1', createMockDataLoader());
      orchestrator.registerLinesLoader('/lines1', createMockLinesLoader());
      orchestrator.registerGSplatsLoader('/gsplats1', createMockGSplatsLoader());
    });

    it('should return "points" for points loader', () => {
      expect(orchestrator.getLoaderType('/points1')).toBe('points');
    });

    it('should return "lines" for lines loader', () => {
      expect(orchestrator.getLoaderType('/lines1')).toBe('lines');
    });

    it('should return "gsplats" for gsplats loader', () => {
      expect(orchestrator.getLoaderType('/gsplats1')).toBe('gsplats');
    });

    it('should return null for unknown path', () => {
      expect(orchestrator.getLoaderType('/nonexistent')).toBeNull();
    });
  });

  describe('error tracking', () => {
    it('should record failure', () => {
      const error = new Error('Test error');
      orchestrator.recordFailure('/points1', error);

      expect(orchestrator.hasFailures()).toBe(true);
      expect(orchestrator.getFailedLoaders().size).toBe(1);
    });

    it('should store failure info correctly', () => {
      const error = new Error('Test error');
      const beforeTime = Date.now();
      orchestrator.recordFailure('/points1', error);
      const afterTime = Date.now();

      const failedInfo = orchestrator.getFailedLoaders().get('/points1');
      expect(failedInfo).toBeDefined();
      expect(failedInfo?.error).toBe(error);
      expect(failedInfo?.timestamp).toBeGreaterThanOrEqual(beforeTime);
      expect(failedInfo?.timestamp).toBeLessThanOrEqual(afterTime);
      expect(failedInfo?.retryCount).toBe(0);
    });

    it('should increment retry count on subsequent failures', () => {
      const error1 = new Error('First error');
      const error2 = new Error('Second error');

      orchestrator.recordFailure('/points1', error1);
      orchestrator.recordFailure('/points1', error2);

      const failedInfo = orchestrator.getFailedLoaders().get('/points1');
      expect(failedInfo?.retryCount).toBe(1);
      expect(failedInfo?.error).toBe(error2);
    });

    it('should clear specific failure', () => {
      orchestrator.recordFailure('/points1', new Error('Error 1'));
      orchestrator.recordFailure('/points2', new Error('Error 2'));

      orchestrator.clearFailure('/points1');

      expect(orchestrator.getFailedLoaders().size).toBe(1);
      expect(orchestrator.getFailedLoaders().has('/points1')).toBe(false);
      expect(orchestrator.getFailedLoaders().has('/points2')).toBe(true);
    });

    it('should clear all failures', () => {
      orchestrator.recordFailure('/points1', new Error('Error 1'));
      orchestrator.recordFailure('/points2', new Error('Error 2'));

      orchestrator.clearAllFailures();

      expect(orchestrator.hasFailures()).toBe(false);
      expect(orchestrator.getFailedLoaders().size).toBe(0);
    });

    it('should handle clearing non-existent failure gracefully', () => {
      expect(() => orchestrator.clearFailure('/nonexistent')).not.toThrow();
    });
  });

  describe('statistics aggregation', () => {
    it('should aggregate points accumulator stats', () => {
      // Register multiple loaders with mocked stats
      // Cast to any since getAccumulatorStats is implementation detail
      const loader1 = createMockDataLoader() as any;
      const loader2 = createMockDataLoader() as any;
      (loader1.getAccumulatorStats as Mock).mockReturnValue({
        capacity: 100,
        allocations: 2,
        growthEvents: 1,
        memoryMB: 0.1,
      });
      (loader2.getAccumulatorStats as Mock).mockReturnValue({
        capacity: 200,
        allocations: 3,
        growthEvents: 2,
        memoryMB: 0.2,
      });

      orchestrator.registerPointsLoader('/points1', loader1);
      orchestrator.registerPointsLoader('/points2', loader2);

      const stats = orchestrator.getAggregatedPointsAccumulatorStats();

      expect(stats.capacity).toBe(300);
      expect(stats.allocations).toBe(5);
      expect(stats.growthEvents).toBe(3);
      expect(stats.memoryMB).toBeCloseTo(0.3);
    });

    it('should aggregate lines accumulator stats', () => {
      // Cast to any since getAccumulatorStats is implementation detail
      const loader1 = createMockLinesLoader() as any;
      const loader2 = createMockLinesLoader() as any;
      (loader1.getAccumulatorStats as Mock).mockReturnValue({
        capacity: 50,
        allocations: 1,
        growthEvents: 0,
        memoryMB: 0.05,
      });
      (loader2.getAccumulatorStats as Mock).mockReturnValue({
        capacity: 100,
        allocations: 2,
        growthEvents: 1,
        memoryMB: 0.1,
      });

      orchestrator.registerLinesLoader('/lines1', loader1);
      orchestrator.registerLinesLoader('/lines2', loader2);

      const stats = orchestrator.getAggregatedLinesAccumulatorStats();

      expect(stats.capacity).toBe(150);
      expect(stats.allocations).toBe(3);
      expect(stats.growthEvents).toBe(1);
      expect(stats.memoryMB).toBeCloseTo(0.15);
    });

    it('should aggregate gsplats accumulator stats', () => {
      // Cast to any since getAccumulatorStats is implementation detail
      const loader1 = createMockGSplatsLoader() as any;
      const loader2 = createMockGSplatsLoader() as any;
      (loader1.getAccumulatorStats as Mock).mockReturnValue({
        capacity: 200,
        allocations: 4,
        growthEvents: 2,
        memoryMB: 0.4,
      });
      (loader2.getAccumulatorStats as Mock).mockReturnValue({
        capacity: 300,
        allocations: 6,
        growthEvents: 3,
        memoryMB: 0.6,
      });

      orchestrator.registerGSplatsLoader('/gsplats1', loader1);
      orchestrator.registerGSplatsLoader('/gsplats2', loader2);

      const stats = orchestrator.getAggregatedGSplatsAccumulatorStats();

      expect(stats.capacity).toBe(500);
      expect(stats.allocations).toBe(10);
      expect(stats.growthEvents).toBe(5);
      expect(stats.memoryMB).toBe(1.0);
    });

    it('should return zero stats when no loaders registered', () => {
      const pointsStats = orchestrator.getAggregatedPointsAccumulatorStats();
      const linesStats = orchestrator.getAggregatedLinesAccumulatorStats();
      const gsplatsStats = orchestrator.getAggregatedGSplatsAccumulatorStats();

      expect(pointsStats.capacity).toBe(0);
      expect(linesStats.capacity).toBe(0);
      expect(gsplatsStats.capacity).toBe(0);
    });

    it('should handle loaders without getAccumulatorStats', () => {
      const loaderWithoutStats = {
        dispose: vi.fn(),
        // No getAccumulatorStats method
      } as any;

      orchestrator.registerPointsLoader('/points1', loaderWithoutStats);

      const stats = orchestrator.getAggregatedPointsAccumulatorStats();

      // Should not throw and return zeros
      expect(stats.capacity).toBe(0);
    });
  });

  describe('dispose', () => {
    it('should dispose all loaders', () => {
      const pointsLoader = createMockDataLoader();
      const linesLoader = createMockLinesLoader();
      const gsplatsLoader = createMockGSplatsLoader();

      orchestrator.registerPointsLoader('/points1', pointsLoader);
      orchestrator.registerLinesLoader('/lines1', linesLoader);
      orchestrator.registerGSplatsLoader('/gsplats1', gsplatsLoader);

      orchestrator.dispose();

      expect(pointsLoader.dispose).toHaveBeenCalled();
      expect(linesLoader.dispose).toHaveBeenCalled();
      expect(gsplatsLoader.dispose).toHaveBeenCalled();
    });

    it('should clear all loader maps', () => {
      orchestrator.registerPointsLoader('/points1', createMockDataLoader());
      orchestrator.registerLinesLoader('/lines1', createMockLinesLoader());
      orchestrator.registerGSplatsLoader('/gsplats1', createMockGSplatsLoader());

      orchestrator.dispose();

      expect(orchestrator.getPointsLoaders().size).toBe(0);
      expect(orchestrator.getLinesLoaders().size).toBe(0);
      expect(orchestrator.getGSplatsLoaders().size).toBe(0);
    });

    it('should clear failure tracking', () => {
      orchestrator.recordFailure('/points1', new Error('Test'));

      orchestrator.dispose();

      expect(orchestrator.hasFailures()).toBe(false);
    });
  });
});
