/**
 * Lazy Data Manager for Luxar
 *
 * Manages chunk-based lazy loading of large nD datasets with intelligent
 * caching and memory management. Designed to handle datasets that exceed
 * available GPU memory by loading only the necessary chunks.
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';
import { detectMemory, MemoryMonitor } from '../utils/memory-detector';

/**
 * Represents a loaded chunk with metadata for cache management
 */
interface ChunkEntry {
  /** Unique identifier for the chunk */
  id: string;
  /** The actual data (Float32Array or Uint8Array) */
  data: ArrayBuffer | Float32Array | Uint8Array;
  /** Timestamp when chunk was last accessed */
  lastAccessed: number;
  /** Size in bytes */
  sizeBytes: number;
  /** Array path this chunk belongs to */
  arrayPath: string;
  /** Chunk indices in the array */
  chunkIndices: number[];
}

/**
 * Configuration for lazy loading behavior
 */
export interface LazyLoadConfig {
  /** Enable lazy loading (default: true for large datasets) */
  enabled: boolean;
  /** Maximum memory usage in MB (default: 500) */
  maxMemoryMB: number;
  /** Number of slices to preload around current position (default: 1) */
  preloadRadius: number;
  /** Cache eviction strategy (default: 'lru') */
  evictionStrategy: 'lru' | 'lfu';
  /** Enable debug logging (default: false) */
  debug: boolean;
}

/**
 * Default configuration for lazy loading with automatic memory detection
 */
const DEFAULT_CONFIG: LazyLoadConfig = {
  enabled: true,
  maxMemoryMB: detectMemory().recommendedCacheMB, // Automatically detect optimal size
  preloadRadius: 1,
  evictionStrategy: 'lru',
  debug: false,
};

/**
 * Manages lazy loading and caching of zarr array chunks for massive datasets.
 *
 * This system enables loading of datasets much larger than available RAM by:
 * - Loading only the currently visible chunks on demand
 * - Maintaining an LRU cache of recently accessed chunks
 * - Automatically evicting old chunks when memory limits are reached
 * - Preloading adjacent chunks for smooth navigation
 * - Preventing duplicate loads of the same chunk
 *
 * The cache size is automatically determined based on available system memory,
 * using 80% of the heap for safety. No manual configuration is required.
 */
export class LazyDataManager {
  public cache: Map<string, ChunkEntry> = new Map(); // Made public for monitoring
  private totalCacheSize: number = 0;
  private config: LazyLoadConfig;
  private loadingPromises: Map<string, Promise<ChunkEntry>> = new Map();
  private eventCallback?: (type: string, message: string, details?: any) => void;
  private memoryMonitor?: MemoryMonitor;
  private datasetMetadata: Map<string, any> = new Map(); // Store metadata about datasets

  constructor(config: Partial<LazyLoadConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Log memory detection result
    const memInfo = detectMemory();
    console.log(
      `💾 [Luxar] Auto-detected cache size: ${this.config.maxMemoryMB}MB (${memInfo.confidence} confidence, source: ${memInfo.source})`
    );

    // Start memory pressure monitoring
    this.memoryMonitor = new MemoryMonitor(this.config.maxMemoryMB, (newSizeMB) => {
      this.config.maxMemoryMB = newSizeMB;
      this.emitEvent('memory', `Cache limit adjusted to ${newSizeMB}MB`, { newSizeMB });
    });
    this.memoryMonitor.start();

    if (this.config.debug) {
      console.log('[🔄] [Luxar] LazyDataManager initialized with config:', this.config);
    }
  }

  /**
   * Set event callback for monitoring
   */
  public setEventCallback(callback: (type: string, message: string, details?: any) => void): void {
    this.eventCallback = callback;
  }

  /**
   * Emit an event for monitoring
   */
  private emitEvent(type: string, message: string, details?: any): void {
    if (this.eventCallback) {
      this.eventCallback(type, message, details);
    }
  }

  /**
   * Get the chunk ID for a given array and chunk indices
   */
  private getChunkId(arrayPath: string, chunkIndices: number[]): string {
    return `${arrayPath}:${chunkIndices.join(',')}`;
  }

  /**
   * Calculate which chunks are needed for a given slice position.
   *
   * This method determines the optimal set of chunks to load based on:
   * - The current position in the nD array
   * - Which dimensions should be fully loaded vs partially loaded
   * - The preload radius for anticipating navigation
   *
   * @param arrayShape - The shape of the full array (e.g., [1000000, 4] for 1M points with 4 coords)
   * @param chunkShape - The chunk dimensions (e.g., [200000, 4])
   * @param slicePosition - Current position in array indices
   * @param sliceDimensions - Dimensions that should be fully loaded (e.g., [1] to load all coordinates)
   * @param preloadRadius - How many adjacent chunks to preload (default: 1)
   * @returns Array of chunk indices to load
   */
  calculateRequiredChunks(
    arrayShape: number[],
    chunkShape: number[],
    slicePosition: number[],
    sliceDimensions: number[],
    preloadRadius: number = this.config.preloadRadius
  ): number[][] {
    const requiredChunks: number[][] = [];
    const nDims = arrayShape.length;

    // Calculate chunk grid dimensions
    const chunkGrid = arrayShape.map((size, i) => Math.ceil(size / chunkShape[i]));

    // For each dimension in the zarr array, determine chunk range
    const chunkRanges: [number, number][] = [];
    for (let d = 0; d < nDims; d++) {
      // sliceDimensions contains indices of array dimensions that should be fully loaded
      // Other dimensions will be partially loaded around the current position

      if (sliceDimensions.includes(d)) {
        // This dimension should be fully loaded
        chunkRanges.push([0, chunkGrid[d] - 1]);
      } else {
        // This dimension should be partially loaded around current position
        const pos = slicePosition[d];
        const chunkIdx = Math.floor(pos / chunkShape[d]);
        const minChunk = Math.max(0, chunkIdx - preloadRadius);
        const maxChunk = Math.min(chunkGrid[d] - 1, chunkIdx + preloadRadius);
        chunkRanges.push([minChunk, maxChunk]);
      }
    }

    // Generate all chunk combinations within ranges
    const generateChunkIndices = (dimIdx: number, current: number[]): void => {
      if (dimIdx === nDims) {
        requiredChunks.push([...current]);
        return;
      }

      const [min, max] = chunkRanges[dimIdx];
      for (let i = min; i <= max; i++) {
        current.push(i);
        generateChunkIndices(dimIdx + 1, current);
        current.pop();
      }
    };

    generateChunkIndices(0, []);

    if (this.config.debug) {
      console.log(`[🔄] [Luxar] Calculated ${requiredChunks.length} required chunks`);
    }

    return requiredChunks;
  }

  /**
   * Load a specific chunk from a zarr array
   */
  async loadChunk(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    arrayPath: string,
    chunkIndices: number[]
  ): Promise<ChunkEntry> {
    const chunkId = this.getChunkId(arrayPath, chunkIndices);

    // Check if already cached
    const cached = this.cache.get(chunkId);
    if (cached) {
      cached.lastAccessed = Date.now();
      if (this.config.debug) {
        console.log(`[✓] [Luxar] Cache hit for chunk ${chunkId}`);
      }
      this.emitEvent('hit', `Cache hit: ${chunkId}`, { chunkId });
      return cached;
    }

    // Check if already loading
    const loading = this.loadingPromises.get(chunkId);
    if (loading) {
      if (this.config.debug) {
        console.log(`[⏳] [Luxar] Waiting for in-progress load of chunk ${chunkId}`);
      }
      this.emitEvent('miss', `Waiting for load: ${chunkId}`, { chunkId });
      return loading;
    }

    // Start loading
    const loadPromise = this.doLoadChunk(array, arrayPath, chunkIndices);
    this.loadingPromises.set(chunkId, loadPromise);

    try {
      const entry = await loadPromise;
      this.loadingPromises.delete(chunkId);
      return entry;
    } catch (error) {
      this.loadingPromises.delete(chunkId);
      throw error;
    }
  }

  /**
   * Actually load the chunk data from zarr
   */
  private async doLoadChunk(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    arrayPath: string,
    chunkIndices: number[]
  ): Promise<ChunkEntry> {
    const chunkId = this.getChunkId(arrayPath, chunkIndices);
    const startTime = Date.now();

    if (this.config.debug) {
      console.log(`[📥] [Luxar] Loading chunk ${chunkId}`);
    }

    this.emitEvent('load', `Loading: ${chunkId}`, { chunkId, arrayPath });

    // Calculate slice for this chunk
    const chunkShape = array.chunks;
    const slice: zarr.Slice[] = chunkIndices.map((idx, dim) => {
      const start = idx * chunkShape[dim];
      const end = Math.min(start + chunkShape[dim], array.shape[dim]);
      return zarr.slice(start, end);
    });

    // Load the chunk data
    const chunkData = await get(array, slice);

    // Create chunk entry
    const entry: ChunkEntry = {
      id: chunkId,
      data: chunkData.data as Float32Array | Uint8Array,
      lastAccessed: Date.now(),
      sizeBytes: (chunkData.data as any).byteLength || (chunkData.data as any).length * 4,
      arrayPath,
      chunkIndices,
    };

    // Add to cache with eviction if needed
    this.addToCache(entry);

    const loadTime = Date.now() - startTime;
    this.emitEvent('load', `Loaded: ${chunkId}`, {
      chunkId,
      sizeBytes: entry.sizeBytes,
      loadTime,
    });

    return entry;
  }

  /**
   * Add a chunk to the cache, evicting old chunks if necessary
   */
  private addToCache(entry: ChunkEntry): void {
    // Check if we need to evict based on memory limit
    const maxSizeBytes = this.config.maxMemoryMB * 1024 * 1024;

    // Only evict if we're actually running out of memory
    // The chunk count limit is a safety valve, not a primary constraint
    while (this.totalCacheSize + entry.sizeBytes > maxSizeBytes && this.cache.size > 0) {
      this.evictOldest();
    }

    // Add to cache
    this.cache.set(entry.id, entry);
    this.totalCacheSize += entry.sizeBytes;

    if (this.config.debug) {
      const sizeMB = this.totalCacheSize / (1024 * 1024);
      console.log(`[💾] [Luxar] Cached chunk ${entry.id}, total cache: ${sizeMB.toFixed(1)}MB`);
    }
  }

  /**
   * Evict the least recently used chunk
   */
  private evictOldest(): void {
    if (this.cache.size === 0) return;

    let oldestEntry: ChunkEntry | null = null;
    let oldestTime = Infinity;

    for (const entry of this.cache.values()) {
      if (entry.lastAccessed < oldestTime) {
        oldestTime = entry.lastAccessed;
        oldestEntry = entry;
      }
    }

    if (oldestEntry) {
      this.cache.delete(oldestEntry.id);
      this.totalCacheSize -= oldestEntry.sizeBytes;

      if (this.config.debug) {
        console.log(`[🗑️] [Luxar] Evicted chunk ${oldestEntry.id}`);
      }

      this.emitEvent('evict', `Evicted: ${oldestEntry.id}`, {
        chunkId: oldestEntry.id,
        sizeBytes: oldestEntry.sizeBytes,
      });
    }
  }

  /**
   * Load multiple chunks for a given array slice
   */
  async loadSlice(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    arrayPath: string,
    sliceSpec: (zarr.Slice | null)[]
  ): Promise<Float32Array | Uint8Array> {
    // Debug log to understand what's being loaded
    const sliceInfo = sliceSpec
      .map((s) => (s ? `${s.start || 0}-${s.stop || 'end'}` : 'all'))
      .join(', ');
    console.log(`[🔍] [Luxar] Loading slice: ${arrayPath} [${sliceInfo}]`);
    const arrayShape = array.shape;
    const chunkShape = array.chunks;

    // Determine which chunks overlap with the requested slice
    const requiredChunks: number[][] = [];

    for (let d = 0; d < arrayShape.length; d++) {
      const slice = sliceSpec[d];
      let start: number, stop: number;

      if (typeof slice === 'number') {
        start = slice;
        stop = slice + 1;
      } else if (slice === null) {
        start = 0;
        stop = arrayShape[d];
      } else {
        start = slice.start ?? 0;
        stop = slice.stop ?? arrayShape[d];
      }

      const startChunk = Math.floor(start / chunkShape[d]);
      const endChunk = Math.floor((stop - 1) / chunkShape[d]);

      const dimChunks: number[] = [];
      for (let c = startChunk; c <= endChunk; c++) {
        dimChunks.push(c);
      }
      requiredChunks.push(dimChunks);
    }

    // Load all required chunks
    const chunkPromises: Promise<ChunkEntry>[] = [];
    const chunkCoords: number[][] = [];

    const generateCombinations = (dimIdx: number, current: number[]): void => {
      if (dimIdx === requiredChunks.length) {
        chunkCoords.push([...current]);
        chunkPromises.push(this.loadChunk(array, arrayPath, [...current]));
        return;
      }

      for (const chunkIdx of requiredChunks[dimIdx]) {
        current.push(chunkIdx);
        generateCombinations(dimIdx + 1, current);
        current.pop();
      }
    };

    generateCombinations(0, []);

    // Wait for all chunks to load
    const chunks = await Promise.all(chunkPromises);

    // Assemble the slice from chunks
    return this.assembleSliceFromChunks(
      chunks,
      chunkCoords,
      arrayShape,
      chunkShape,
      sliceSpec,
      array.dtype
    );
  }

  /**
   * Assemble a contiguous array from loaded chunks
   */
  private assembleSliceFromChunks(
    chunks: ChunkEntry[],
    chunkCoords: number[][],
    arrayShape: number[],
    chunkShape: number[],
    sliceSpec: (zarr.Slice | null)[],
    dtype: zarr.DataType
  ): Float32Array | Uint8Array {
    // Calculate slice bounds for each dimension
    const sliceBounds: Array<{ start: number; stop: number; step: number }> = [];
    const outputShape: number[] = [];

    for (let d = 0; d < arrayShape.length; d++) {
      const slice = sliceSpec[d];
      if (typeof slice === 'number') {
        sliceBounds.push({ start: slice, stop: slice + 1, step: 1 });
        // Single index - dimension is collapsed
      } else if (slice === null) {
        sliceBounds.push({ start: 0, stop: arrayShape[d], step: 1 });
        outputShape.push(arrayShape[d]);
      } else {
        const start = slice.start ?? 0;
        const stop = slice.stop ?? arrayShape[d];
        const step = slice.step ?? 1;
        sliceBounds.push({ start, stop, step });
        outputShape.push(Math.ceil((stop - start) / step));
      }
    }

    // For point cloud data, we expect shape [n_points, n_dims]
    // After slicing non-displayed dims, we get [points_in_slice, n_dims]
    const totalSize = outputShape.reduce((a, b) => a * b, 1);

    // Create output array
    const ArrayConstructor = dtype === 'uint8' ? Uint8Array : Float32Array;
    const output = new ArrayConstructor(totalSize);

    // Map chunks to output array
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const chunkCoord = chunkCoords[i];
      const data = chunk.data as Float32Array | Uint8Array;

      // Calculate the position of this chunk in the global array
      const chunkStartIndices = chunkCoord.map((c, d) => c * chunkShape[d]);

      // Iterate through elements in the chunk
      const chunkDims = chunkShape.map((size, d) => {
        const globalStart = chunkStartIndices[d];
        const globalEnd = Math.min(globalStart + size, arrayShape[d]);
        const sliceStart = sliceBounds[d].start;
        const sliceStop = sliceBounds[d].stop;

        // Find overlap between chunk and slice
        const overlapStart = Math.max(globalStart, sliceStart);
        const overlapEnd = Math.min(globalEnd, sliceStop);

        return {
          localStart: overlapStart - globalStart,
          localEnd: overlapEnd - globalStart,
          outputStart: overlapStart - sliceStart,
          size: overlapEnd - overlapStart,
        };
      });

      // Copy relevant data from chunk to output
      // For 2D arrays (points × dims), iterate through points
      if (arrayShape.length === 2) {
        const pointDim = chunkDims[0];
        const dimDim = chunkDims[1];

        for (let p = pointDim.localStart; p < pointDim.localEnd; p++) {
          for (let d = dimDim.localStart; d < dimDim.localEnd; d++) {
            const chunkIdx = p * chunkShape[1] + d;
            const outputP = pointDim.outputStart + (p - pointDim.localStart);
            const outputD = dimDim.outputStart + (d - dimDim.localStart);
            const outputIdx = outputP * outputShape[1] + outputD;

            if (chunkIdx < data.length && outputIdx < output.length) {
              output[outputIdx] = data[chunkIdx];
            }
          }
        }
      } else {
        // For 1D arrays, simple copy
        const dim = chunkDims[0];
        for (let i = dim.localStart; i < dim.localEnd; i++) {
          const outputIdx = dim.outputStart + (i - dim.localStart);
          if (i < data.length && outputIdx < output.length) {
            output[outputIdx] = data[i];
          }
        }
      }
    }

    return output;
  }

  /**
   * Preload chunks that are likely to be needed soon
   */
  async preloadChunks(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    arrayPath: string,
    currentPosition: number[],
    sliceDimensions: number[]
  ): Promise<void> {
    const requiredChunks = this.calculateRequiredChunks(
      array.shape,
      array.chunks,
      currentPosition,
      sliceDimensions,
      this.config.preloadRadius + 1 // Preload a bit further
    );

    // Load chunks in background (don't await)
    for (const chunkIndices of requiredChunks) {
      const chunkId = this.getChunkId(arrayPath, chunkIndices);
      this.emitEvent('preload', `Preloading: ${chunkId}`, { chunkId });

      this.loadChunk(array, arrayPath, chunkIndices).catch((error) => {
        console.warn('[⚠️] [Luxar] Failed to preload chunk:', error);
      });
    }
  }

  /**
   * Clear the entire cache
   */
  clearCache(): void {
    const oldSize = this.cache.size;
    this.cache.clear();
    this.totalCacheSize = 0;
    this.loadingPromises.clear();

    if (this.config.debug) {
      console.log('[🗑️] [Luxar] Cache cleared');
    }

    this.emitEvent('clear', `Cache cleared (${oldSize} chunks)`, {
      clearedChunks: oldSize,
    });
  }

  /**
   * Store metadata about a dataset (e.g., total slices)
   */
  public setDatasetMetadata(key: string, value: any): void {
    this.datasetMetadata.set(key, value);
  }

  /**
   * Get metadata about a dataset
   */
  public getDatasetMetadata(key: string): any {
    return this.datasetMetadata.get(key);
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    numChunks: number;
    totalSizeMB: number;
    maxSizeMB: number;
    utilizationPercent: number;
  } {
    const totalSizeMB = this.totalCacheSize / (1024 * 1024);
    const maxSizeMB = this.config.maxMemoryMB;

    return {
      numChunks: this.cache.size,
      totalSizeMB,
      maxSizeMB,
      utilizationPercent: (totalSizeMB / maxSizeMB) * 100,
    };
  }

  /**
   * Check if lazy loading should be enabled for a dataset
   */
  static shouldUseLazyLoading(
    numPoints: number,
    ndim: number,
    config: Partial<LazyLoadConfig> = {}
  ): boolean {
    const cfg = { ...DEFAULT_CONFIG, ...config };

    // Use lazy loading for large datasets or high-dimensional data
    const isLarge = numPoints > 10_000_000; // 10M points
    const isHighDim = ndim > 3;

    return cfg.enabled && (isLarge || isHighDim);
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.clearCache();
    if (this.memoryMonitor) {
      this.memoryMonitor.stop();
      this.memoryMonitor = undefined;
    }
  }
}
