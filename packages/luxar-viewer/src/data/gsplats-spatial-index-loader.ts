/**
 * GSplats spatial index-based data loader for efficient nD gsplats loading.
 *
 * This loader implements spatial-index based loading:
 * 1. Query chunk bounds to find chunks intersecting the view
 * 2. Load splat data for those chunks
 *
 * Unlike lines, gsplats don't need two-phase loading - all data is per-splat.
 *
 * @module data/gsplats-spatial-index-loader
 */

import * as zarr from 'zarrita';
import { get, slice } from 'zarrita';
import { log, Modules, LogEmoji } from '../utils/log';
import {
  loadGSplatsChunkSpatialIndex,
  queryGSplatsChunksForView,
  chunkIndicesToSplatRanges,
  mergeRanges,
  computeToleranceFromViewState,
} from './gsplats-chunk-spatial-index';
import type {
  GSplatsMetadata,
  GSplatsChunkSpatialIndex,
  LoadedGSplatsData,
  GSplatsDataLoader,
  GSplatsViewState,
  SplatRange,
} from '../types/gsplats';
import type { SceneNode } from './data-loader-types';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from './array-decoder';
import { choleskyPackedSize } from '../types/gsplats';
import { GSplatsDataAccumulator } from './data-accumulator';
import { config as appConfig } from '../config';
import { getWorkerPool } from '../workers/worker-pool';

/**
 * GSplats data loader using spatial indices for efficient nD queries.
 *
 * Key features:
 * - Chunk-based loading using spatial index
 * - Handles all array encoding types (broadcasted, quantized, LUT, etc.)
 * - Support for optional arrays (colors, sharpness)
 */
export class GSplatsSpatialIndexLoader implements GSplatsDataLoader {
  private chunkIndex: GSplatsChunkSpatialIndex | null = null;
  private zarrLocation: zarr.Location<zarr.Readable>;
  private node: SceneNode;
  private initPromise: Promise<void> | null = null;
  private initLock = false;
  private decoder: ArrayDecoder;
  private zarrStore: zarr.Readable | null = null;

  // Data accumulator for object pooling (Phase 1 optimization)
  private _accumulator: GSplatsDataAccumulator | null = null;

  private arrays: {
    centers?: zarr.Array<zarr.DataType, zarr.Readable>;
    amplitudes?: zarr.Array<zarr.DataType, zarr.Readable>;
    cholesky_factors?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
    sharpness?: zarr.Array<zarr.DataType, zarr.Readable>;
  } = {};

  constructor(
    zarrLocation: zarr.Location<zarr.Readable>,
    node: SceneNode,
    refRegistry?: ArrayRefRegistry,
    zarrStore?: zarr.Readable
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.decoder = new ArrayDecoder(refRegistry || new ArrayRefRegistry());
    this.zarrStore = zarrStore || null;
  }

  /**
   * Initialize the loader by loading spatial index and opening arrays
   */
  async initialize(): Promise<void> {
    const attrs = this.node.attrs as unknown as GSplatsMetadata;

    // Load spatial index
    try {
      this.chunkIndex = await loadGSplatsChunkSpatialIndex(this.zarrLocation, attrs);

      if (!this.chunkIndex) {
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `No spatial index for GSplats ${this.node.path} - will load all data`
        );
      } else {
        log.query(
          Modules.SPATIAL_INDEX_LOADER,
          `GSplats index loaded: ${this.chunkIndex.chunkCount} chunks`
        );
      }
    } catch (error) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
        `Failed to load GSplats spatial index for ${this.node.path}:`,
        error
      );
      throw error;
    }

    // Open required arrays
    try {
      this.arrays.centers = await zarr.open(this.zarrLocation.resolve('centers'), {
        kind: 'array',
      });
      this.arrays.amplitudes = await zarr.open(this.zarrLocation.resolve('amplitudes'), {
        kind: 'array',
      });
      this.arrays.cholesky_factors = await zarr.open(
        this.zarrLocation.resolve('cholesky_factors'),
        {
          kind: 'array',
        }
      );
    } catch (e) {
      log.error(Modules.SPATIAL_INDEX_LOADER, 'Failed to open required GSplats arrays:', e);
      throw e;
    }

    // Try to open optional arrays
    try {
      this.arrays.colors = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No colors array found (using default white)');
    }

    try {
      this.arrays.sharpness = await zarr.open(this.zarrLocation.resolve('sharpness'), {
        kind: 'array',
      });
    } catch {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No sharpness array found (using default 2.0)');
    }

    // Initialize data accumulator for object pooling (Phase 1 optimization)
    // NOTE: Infrastructure-only for Phase 1. Full hot path integration deferred to Phase 2.
    // See src/data/DATA_ACCUMULATOR_STATUS.md for details.
    if (appConfig.dataLoading.performance.useAccumulators) {
      const metadata = attrs as GSplatsMetadata;
      const totalSplats = this.chunkIndex?.metadata.n_splats || metadata.n_splats || 0;
      const ndim = this.chunkIndex?.metadata.ndim || this.arrays.centers?.shape[1] || 3;

      // Estimate initial capacity (at least 1024, or ~10% of total)
      const initialCapacity = Math.min(
        appConfig.dataLoading.performance.initialAccumulatorCapacity,
        Math.max(1024, Math.ceil(totalSplats / 10))
      );

      this._accumulator = new GSplatsDataAccumulator(initialCapacity, ndim);

      if (appConfig.dataLoading.performance.enablePerformanceMonitoring) {
        const stats = this._accumulator.getStats();
        log.info(
          Modules.DATA_ACCUMULATOR,
          `Initialized GSplatsDataAccumulator for ${this.node.path}: ` +
            `capacity=${stats.capacity}, ndim=${ndim}, totalSplats=${totalSplats} ` +
            '(infrastructure-only, hot path integration in Phase 2)'
        );
      }
    }
  }

  /**
   * Load gsplats data for the given view state
   */
  async loadGSplats(viewState: GSplatsViewState): Promise<LoadedGSplatsData> {
    // Prevent race conditions during initialization
    if (!this.initPromise && !this.initLock) {
      this.initLock = true;
      this.initPromise = this.initialize().finally(() => {
        this.initLock = false;
      });
    }

    if (this.initPromise) {
      await this.initPromise;
    }

    if (!this.arrays.centers || !this.arrays.amplitudes || !this.arrays.cholesky_factors) {
      throw new Error('GSplats loader not properly initialized');
    }

    const attrs = this.node.attrs as unknown as GSplatsMetadata;

    // Query visible splat ranges (Phase 2: async for worker support)
    const splatRanges = await this.queryVisibleSplatRanges(viewState);

    if (splatRanges.length === 0) {
      log.info(Modules.SPATIAL_INDEX_LOADER, 'No visible gsplats - returning empty data');
      return this.createEmptyGSplatsData(attrs);
    }

    // Count total splats to load
    const totalSplats = splatRanges.reduce((sum, r) => sum + (r.end - r.start), 0);

    log.info(
      LogEmoji.LOAD,
      Modules.SPATIAL_INDEX_LOADER,
      `Loading ${totalSplats} gsplats from ${splatRanges.length} ranges`
    );

    // Phase 1 DEEP Integration: Load DIRECTLY to accumulator buffers (ZERO allocations!)
    if (this._accumulator && appConfig.dataLoading.performance.useAccumulators) {
      // Ensure capacity FIRST
      this._accumulator.ensureCapacity(totalSplats);

      // Get direct buffer references for zero-allocation loading
      const centerBuffer = this._accumulator['centerBuffer'] as Float32Array;
      const amplitudeBuffer = this._accumulator['amplitudeBuffer'] as Float32Array;
      const choleskyBuffer = this._accumulator['choleskyBuffer'] as Float32Array;

      // Load directly into accumulator buffers (ZERO intermediate allocations!)
      await this.loadArrayRanges('centers', splatRanges, attrs.ndim, centerBuffer);
      await this.loadArrayRanges('amplitudes', splatRanges, 1, amplitudeBuffer);
      await this.loadArrayRanges(
        'cholesky_factors',
        splatRanges,
        choleskyPackedSize(attrs.ndim),
        choleskyBuffer
      );

      // Load optional arrays directly to accumulator
      if (this.arrays.colors) {
        const colorBuffer = this._accumulator['colorBuffer'] as Float32Array;
        await this.loadArrayRanges('colors', splatRanges, 3, colorBuffer);
      }
      if (this.arrays.sharpness) {
        const sharpnessBuffer = this._accumulator['sharpnessBuffer'] as Float32Array;
        await this.loadArrayRanges('sharpness', splatRanges, 1, sharpnessBuffer);
      }

      // Return from accumulator (subarrays, zero copy!)
      // NO fill() needed - data already in buffers!
      return this._accumulator.getData(totalSplats);
    }

    // Fallback: Load to separate arrays (allocations when accumulator disabled)
    const centers = await this.loadArrayRanges('centers', splatRanges, attrs.ndim);
    const amplitudes = await this.loadArrayRanges('amplitudes', splatRanges, 1);
    const choleskyFactors = await this.loadArrayRanges(
      'cholesky_factors',
      splatRanges,
      choleskyPackedSize(attrs.ndim)
    );

    const colors = this.arrays.colors ? await this.loadArrayRanges('colors', splatRanges, 3) : null;
    const sharpness = this.arrays.sharpness
      ? await this.loadArrayRanges('sharpness', splatRanges, 1)
      : null;

    return {
      centers,
      amplitudes,
      choleskyFactors,
      colors,
      sharpness,
      splatCount: totalSplats,
      ndim: attrs.ndim,
    };
  }

  /**
   * Update view for new position.
   */
  async updateView(viewState: GSplatsViewState): Promise<LoadedGSplatsData> {
    return this.loadGSplats(viewState);
  }

  /**
   * Query visible splat ranges based on view state
   */
  private async queryVisibleSplatRanges(viewState: GSplatsViewState): Promise<SplatRange[]> {
    const attrs = this.node.attrs as unknown as GSplatsMetadata;

    // Check if this node has extend_to_all dimensions
    const extendDims: string[] = attrs.extend_to_all || [];

    if (extendDims.length > 0 && viewState.dimensions) {
      // Check if we're navigating through an extended dimension
      const currentNonDisplayedDims: string[] =
        viewState.dimensions
          ?.filter((_meta: { name?: string }, idx: number) => !viewState.displayDims.includes(idx))
          ?.map((meta: { name?: string }) => meta.name)
          ?.filter((name: string | undefined): name is string => !!name) || [];

      const isExtending = extendDims.some((edim: string) => currentNonDisplayedDims.includes(edim));

      if (isExtending) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.SPATIAL_INDEX_LOADER,
          `Extending ${this.node.path} visibility across: ${extendDims.join(', ')}`
        );
        // Return all splats for extended dimensions
        return [{ start: 0, end: attrs.n_splats }];
      }
    }

    if (!this.chunkIndex) {
      // No spatial index - load all splats
      return [{ start: 0, end: attrs.n_splats }];
    }

    // Compute tolerance for queries
    const tolerance = computeToleranceFromViewState(viewState);

    // Ensure slicePosition has correct length
    const slicePosition = new Array(attrs.ndim).fill(0);
    for (let i = 0; i < Math.min(viewState.slicePosition.length, attrs.ndim); i++) {
      slicePosition[i] = viewState.slicePosition[i] ?? 0;
    }

    // Query chunks (worker or main thread based on config)
    let chunkIndices: number[];

    if (appConfig.dataLoading.performance.useWebWorkers) {
      // Phase 2: Use worker for spatial queries
      try {
        const worker = await getWorkerPool().getWorker();
        const result = await worker.querySpatialIndex({
          chunkBounds: this.chunkIndex.chunkBounds,
          slicePosition: new Float32Array(slicePosition),
          tolerance: new Float32Array(tolerance),
          numChunks: this.chunkIndex.chunkCount,
          ndim: attrs.ndim,
        });
        chunkIndices = Array.from(result);
      } catch (error) {
        log.error(Modules.SPATIAL_INDEX_LOADER, 'Worker query failed, using main thread:', error);
        // Fallback to main thread
        chunkIndices = queryGSplatsChunksForView(this.chunkIndex, slicePosition, tolerance);
      }
    } else {
      // Main thread query
      chunkIndices = queryGSplatsChunksForView(this.chunkIndex, slicePosition, tolerance);
    }

    if (chunkIndices.length === 0) {
      return [];
    }

    // Convert to ranges and merge
    const ranges = chunkIndicesToSplatRanges(chunkIndices, attrs.chunk_size, attrs.n_splats);

    return mergeRanges(ranges);
  }

  /**
   * Load array data for given splat ranges with optimized encoding handling.
   *
   * Handles different encoding types efficiently:
   * - Broadcasted: Load single value, replicate to all splats
   * - Quantized: Load only needed ranges, then dequantize
   * - LUT: Load only needed ranges of indices, decode with LUT
   * - Array ref: Resolve target, apply optimized loading
   * - Direct: Load ranges directly from zarr
   */
  /**
   * Load array ranges with optional target buffer for zero-allocation operation
   *
   * @param arrayName - Name of array to load
   * @param ranges - Ranges to load
   * @param elementsPerSplat - Elements per splat
   * @param targetBuffer - Optional target buffer (for accumulator integration)
   * @returns Loaded data (new array or subarray of target)
   */
  private async loadArrayRanges(
    arrayName: string,
    ranges: SplatRange[],
    elementsPerSplat: number,
    targetBuffer?: Float32Array
  ): Promise<Float32Array> {
    const array = this.arrays[arrayName as keyof typeof this.arrays];
    if (!array) {
      throw new Error(`Array ${arrayName} not initialized`);
    }

    const totalSplats = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const totalElements = totalSplats * elementsPerSplat;

    // Check for different encoding types
    const attrs = array.attrs as unknown as ArrayMetadata;
    const isBroadcasted = ArrayDecoder.isBroadcasted(attrs);
    const isQuantized = ArrayDecoder.isQuantizedEncoding(attrs);
    const isLUTEncoded = ArrayDecoder.isLUTEncoded(attrs);
    const isArrayRef = ArrayDecoder.isArrayRef(attrs);

    // Get array shape for slice specification
    const shape = array.shape;

    // Use target buffer or allocate (ZERO allocation when targetBuffer provided!)
    const output = targetBuffer ? targetBuffer : new Float32Array(totalElements);
    let destOffset = 0;

    if (isBroadcasted) {
      // Broadcasted encoding: Load single value and replicate to all splats
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `GSplats: Broadcasted array ${arrayName}: replicating single value to ${totalSplats} splats`
      );

      const fullData = await get(array);
      const broadcastValue = fullData.data as Float32Array | Uint8Array;
      const broadcastLen = broadcastValue.length;

      // Use elementsPerSplat (caller's expectation) for output indexing
      // Use modulo to handle cases where broadcast value has fewer elements than expected
      for (let i = 0; i < totalSplats; i++) {
        for (let j = 0; j < elementsPerSplat; j++) {
          output[i * elementsPerSplat + j] = broadcastValue[j % broadcastLen] ?? broadcastValue[0];
        }
      }
    } else if (isQuantized) {
      // Quantized encoding: Load only needed ranges of quantized data, then dequantize
      const quantMetadata = ArrayDecoder.getQuantizationMetadata(attrs);
      if (!quantMetadata) {
        throw new Error(
          `GSplats: Quantization metadata missing for ${arrayName}. ` +
            'This should not happen if isQuantizedEncoding returned true.'
        );
      }

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `GSplats: Quantized range loading: ${arrayName}, ${ranges.length} ranges (${ArrayDecoder.getEncodingMode(attrs)}, ${totalSplats} values)`
      );

      // PARALLEL FETCH: Load all chunks simultaneously (I/O parallelism)
      const chunkPromises = ranges.map((range) => {
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)]
            : [slice(range.start, range.end)];
        return get(array, sliceSpec);
      });

      const chunks = await Promise.all(chunkPromises);

      // SEQUENTIAL DECODE: Process chunks in order
      for (const chunkData of chunks) {
        const quantizedData = chunkData.data as Float32Array | Uint8Array | Uint16Array;
        const dequantized = this.decoder.dequantizeRange(quantizedData, quantMetadata);

        output.set(dequantized, destOffset);
        destOffset += dequantized.length;
      }
    } else if (isLUTEncoded) {
      // LUT encoding: Load only the range of indices, then decode with LUT from metadata
      const lutMetadata = ArrayDecoder.getLUTMetadata(attrs);
      if (!lutMetadata) {
        throw new Error(
          `GSplats: LUT metadata missing for ${arrayName}. ` +
            'This should not happen if isLUTEncoded returned true.'
        );
      }

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `GSplats: LUT range loading: ${arrayName}, ${ranges.length} ranges (${totalSplats} indices → ${totalElements} elements)`
      );

      // PARALLEL FETCH: Load all index chunks simultaneously
      const chunkPromises = ranges.map((range) => {
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)]
            : [slice(range.start, range.end)];
        return get(array, sliceSpec);
      });

      const chunks = await Promise.all(chunkPromises);

      // SEQUENTIAL DECODE: Process chunks in order
      for (const chunkData of chunks) {
        const indices = chunkData.data as Float32Array | Uint8Array | Uint16Array;
        const decoded = this.decoder.decodeLUTIndices(indices, lutMetadata);

        output.set(decoded, destOffset);
        destOffset += decoded.length;
      }
    } else if (isArrayRef) {
      // Array reference: Resolve target and apply optimized range loading based on target encoding
      const targetPath = attrs.encoding!.target!;

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `GSplats: Array ref: ${arrayName} → ${targetPath} (resolving with optimized range loading)`
      );

      const storeToUse = this.zarrStore || this.zarrLocation.store;
      const zarrRootLoc = zarr.root(storeToUse);

      const targetLoc = zarrRootLoc.resolve(targetPath);
      const targetArray = await zarr.open(targetLoc, { kind: 'array' });
      const targetAttrs = targetArray.attrs as unknown as ArrayMetadata;

      if (ArrayDecoder.isQuantizedEncoding(targetAttrs)) {
        // Target is quantized → use quantized range loading
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `GSplats: Array ref target is quantized (${ArrayDecoder.getEncodingMode(targetAttrs)})`
        );

        const quantMeta = ArrayDecoder.getQuantizationMetadata(targetAttrs);
        if (!quantMeta) {
          throw new Error(
            `GSplats: Quantization metadata missing for array_ref target: ${targetPath}`
          );
        }

        // PARALLEL FETCH: Load all quantized chunks from target
        const chunkPromises = ranges.map((range) => {
          const sliceSpec: zarr.Slice[] =
            targetArray.shape.length === 2
              ? [slice(range.start, range.end), slice(null)]
              : [slice(range.start, range.end)];
          return get(targetArray, sliceSpec);
        });

        const chunks = await Promise.all(chunkPromises);

        // SEQUENTIAL DECODE
        for (const quantizedData of chunks) {
          const dequantized = this.decoder.dequantizeRange(
            quantizedData.data as Uint8Array | Uint16Array,
            quantMeta
          );

          output.set(dequantized, destOffset);
          destOffset += dequantized.length;
        }
      } else if (ArrayDecoder.isLUTEncoded(targetAttrs)) {
        // Target is LUT → use LUT range loading
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `GSplats: Array ref target is LUT (${ArrayDecoder.getEncodingMode(targetAttrs)})`
        );

        const lutMetadata = ArrayDecoder.getLUTMetadata(targetAttrs);
        if (!lutMetadata) {
          throw new Error(`GSplats: LUT metadata missing for array_ref target: ${targetPath}`);
        }

        // PARALLEL FETCH: Load all LUT index chunks from target
        const chunkPromises = ranges.map((range) => {
          const sliceSpec: zarr.Slice[] =
            targetArray.shape.length === 2
              ? [slice(range.start, range.end), slice(null)]
              : [slice(range.start, range.end)];
          return get(targetArray, sliceSpec);
        });

        const chunks = await Promise.all(chunkPromises);

        // SEQUENTIAL DECODE
        for (const chunkData of chunks) {
          const decoded = this.decoder.decodeLUTIndices(
            chunkData.data as Float32Array | Uint8Array | Uint16Array,
            lutMetadata
          );

          output.set(decoded, destOffset);
          destOffset += decoded.length;
        }
      } else if (ArrayDecoder.isBroadcasted(targetAttrs)) {
        // Target is broadcasted → load once, replicate
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          'GSplats: Array ref target is broadcasted (loading single value)'
        );

        const fullData = await get(targetArray);
        const broadcastValue = fullData.data as Float32Array | Uint8Array | Uint16Array;
        const broadcastLen = broadcastValue.length;

        // Use elementsPerSplat (caller's expectation) for output indexing
        for (let i = 0; i < totalSplats; i++) {
          for (let j = 0; j < elementsPerSplat; j++) {
            output[i * elementsPerSplat + j] =
              broadcastValue[j % broadcastLen] ?? broadcastValue[0];
          }
        }
      } else {
        // Target is direct or unknown encoding → decode full target, extract ranges
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `GSplats: Array ref target is direct/unknown (${ArrayDecoder.getEncodingMode(targetAttrs)}) - using full decode`
        );

        const decoded = await this.decoder.decode(
          targetArray,
          targetAttrs,
          totalElements,
          zarrRootLoc
        );

        // Use elementsPerSplat (caller's expectation) for indexing
        for (const range of ranges) {
          const rangeSize = (range.end - range.start) * elementsPerSplat;
          const srcOffset = range.start * elementsPerSplat;

          output.set(decoded.subarray(srcOffset, srcOffset + rangeSize), destOffset);
          destOffset += rangeSize;
        }
      }
    } else {
      // Direct arrays: load ranges directly from zarr
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `GSplats: Direct array loading: ${arrayName}, ${ranges.length} ranges (${totalSplats} splats)`
      );

      // PARALLEL FETCH: Load all chunks simultaneously
      const chunkPromises = ranges.map((range) => {
        const sliceSpec: zarr.Slice[] =
          shape.length === 2
            ? [slice(range.start, range.end), slice(null)]
            : [slice(range.start, range.end)];
        return get(array, sliceSpec);
      });

      const chunks = await Promise.all(chunkPromises);

      // SEQUENTIAL WRITE
      for (const data of chunks) {
        const floatData =
          data.data instanceof Float32Array
            ? data.data
            : new Float32Array(data.data as unknown as ArrayBufferLike);

        output.set(floatData, destOffset);
        destOffset += floatData.length;
      }
    }

    return output;
  }

  /**
   * Create empty gsplats data
   */
  private createEmptyGSplatsData(attrs: GSplatsMetadata): LoadedGSplatsData {
    return {
      centers: new Float32Array(0),
      amplitudes: new Float32Array(0),
      choleskyFactors: new Float32Array(0),
      colors: null,
      sharpness: null,
      splatCount: 0,
      ndim: attrs.ndim,
    };
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.chunkIndex = null;
    this.arrays = {};
    this.initPromise = null;

    // Dispose accumulator
    if (this._accumulator) {
      this._accumulator.dispose();
      this._accumulator = null;
    }
  }
}
