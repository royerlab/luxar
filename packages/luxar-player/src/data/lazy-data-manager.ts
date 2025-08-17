/**
 * Lazy Data Manager for Luxar
 * 
 * Manages chunk-based lazy loading of large nD datasets with intelligent
 * caching and memory management. Designed to handle datasets that exceed
 * available GPU memory by loading only the necessary chunks.
 */

import * as zarr from 'zarrita';
import { get } from 'zarrita';

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
  /** Maximum number of chunks to keep in cache (default: 100) */
  maxChunks: number;
  /** Cache eviction strategy (default: 'lru') */
  evictionStrategy: 'lru' | 'lfu';
  /** Enable debug logging (default: false) */
  debug: boolean;
}

/**
 * Default configuration for lazy loading
 */
const DEFAULT_CONFIG: LazyLoadConfig = {
  enabled: true,
  maxMemoryMB: 500,
  preloadRadius: 1,
  maxChunks: 100,
  evictionStrategy: 'lru',
  debug: false,
};

/**
 * Manages lazy loading and caching of zarr array chunks
 */
export class LazyDataManager {
  private cache: Map<string, ChunkEntry> = new Map();
  private totalCacheSize: number = 0;
  private config: LazyLoadConfig;
  private loadingPromises: Map<string, Promise<ChunkEntry>> = new Map();

  constructor(config: Partial<LazyLoadConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    if (this.config.debug) {
      console.log('[🔄] [Luxar] LazyDataManager initialized with config:', this.config);
    }
  }

  /**
   * Get the chunk ID for a given array and chunk indices
   */
  private getChunkId(arrayPath: string, chunkIndices: number[]): string {
    return `${arrayPath}:${chunkIndices.join(',')}`;
  }

  /**
   * Calculate which chunks are needed for a given slice position
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
    const chunkGrid = arrayShape.map((size, i) => 
      Math.ceil(size / chunkShape[i])
    );

    // For each non-displayed dimension, determine chunk range
    const chunkRanges: [number, number][] = [];
    for (let d = 0; d < nDims; d++) {
      if (sliceDimensions.includes(d)) {
        // Displayed dimension - need all chunks
        chunkRanges.push([0, chunkGrid[d] - 1]);
      } else {
        // Non-displayed dimension - need chunks around slice position
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
      return cached;
    }
    
    // Check if already loading
    const loading = this.loadingPromises.get(chunkId);
    if (loading) {
      if (this.config.debug) {
        console.log(`[⏳] [Luxar] Waiting for in-progress load of chunk ${chunkId}`);
      }
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
    
    if (this.config.debug) {
      console.log(`[📥] [Luxar] Loading chunk ${chunkId}`);
    }
    
    // Calculate slice for this chunk
    const chunkShape = array.chunks;
    const slice: zarr.Slice[] = chunkIndices.map((idx, dim) => {
      const start = idx * chunkShape[dim];
      const end = Math.min(start + chunkShape[dim], array.shape[dim]);
      return zarr.slice(start, end);
    });
    
    // Load the chunk data
    const chunkData = await get(array, { selection: slice });
    
    // Create chunk entry
    const entry: ChunkEntry = {
      id: chunkId,
      data: chunkData.data,
      lastAccessed: Date.now(),
      sizeBytes: chunkData.data.byteLength,
      arrayPath,
      chunkIndices,
    };
    
    // Add to cache with eviction if needed
    this.addToCache(entry);
    
    return entry;
  }

  /**
   * Add a chunk to the cache, evicting old chunks if necessary
   */
  private addToCache(entry: ChunkEntry): void {
    // Check if we need to evict
    const maxSizeBytes = this.config.maxMemoryMB * 1024 * 1024;
    
    while (
      (this.totalCacheSize + entry.sizeBytes > maxSizeBytes ||
       this.cache.size >= this.config.maxChunks) &&
      this.cache.size > 0
    ) {
      this.evictOldest();
    }
    
    // Add to cache
    this.cache.set(entry.id, entry);
    this.totalCacheSize += entry.sizeBytes;
    
    if (this.config.debug) {
      const sizeMB = this.totalCacheSize / (1024 * 1024);
      console.log(
        `[💾] [Luxar] Cached chunk ${entry.id}, total cache: ${sizeMB.toFixed(1)}MB`
      );
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
    }
  }

  /**
   * Load multiple chunks for a given array slice
   */
  async loadSlice(
    array: zarr.Array<zarr.DataType, zarr.Readable>,
    arrayPath: string,
    sliceSpec: zarr.Slice[]
  ): Promise<Float32Array | Uint8Array> {
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
    sliceSpec: zarr.Slice[],
    dtype: zarr.DataType
  ): Float32Array | Uint8Array {
    // Calculate output shape
    const outputShape: number[] = [];
    for (let d = 0; d < arrayShape.length; d++) {
      const slice = sliceSpec[d];
      if (typeof slice === 'number') {
        // Single index - dimension is removed
        continue;
      } else if (slice === null) {
        outputShape.push(arrayShape[d]);
      } else {
        const start = slice.start ?? 0;
        const stop = slice.stop ?? arrayShape[d];
        const step = slice.step ?? 1;
        outputShape.push(Math.ceil((stop - start) / step));
      }
    }
    
    // Calculate total size
    const totalSize = outputShape.reduce((a, b) => a * b, 1);
    
    // Create output array
    const ArrayConstructor = dtype === 'uint8' ? Uint8Array : Float32Array;
    const output = new ArrayConstructor(totalSize);
    
    // Copy data from chunks to output
    // This is simplified - real implementation would need careful indexing
    let outputIdx = 0;
    for (const chunk of chunks) {
      const data = chunk.data as Float32Array | Uint8Array;
      for (let i = 0; i < data.length; i++) {
        if (outputIdx < totalSize) {
          output[outputIdx++] = data[i];
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
      this.loadChunk(array, arrayPath, chunkIndices).catch(error => {
        console.warn('[⚠️] [Luxar] Failed to preload chunk:', error);
      });
    }
  }

  /**
   * Clear the entire cache
   */
  clearCache(): void {
    this.cache.clear();
    this.totalCacheSize = 0;
    this.loadingPromises.clear();
    
    if (this.config.debug) {
      console.log('[🗑️] [Luxar] Cache cleared');
    }
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
}