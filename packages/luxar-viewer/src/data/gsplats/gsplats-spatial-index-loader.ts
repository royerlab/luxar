/**
 * GSplats spatial index-based data loader for efficient nD gsplats loading.
 *
 * This loader implements spatial-index based loading:
 * 1. Query chunk bounds to find chunks intersecting the view
 * 2. Load splat data for those chunks
 *
 * Unlike lines, gsplats load directly because all data is per-splat.
 *
 * @module data/gsplats-spatial-index-loader
 */

import * as zarr from '../zarr';
import { log, Modules } from '../../utils/log';
import type {
  GSplatsMetadata,
  LoadedGSplatsData,
  GSplatsDataLoader,
  GSplatsViewState,
  SplatRange,
} from '../../types/gsplats';
import type { SceneNode, PointRange } from '../data-loader-types';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from '../array-decoder/decoder';
import {
  RangeLoader,
  SpatialQueryBuilder,
  type ChunkSpatialIndex,
  type LoadRange,
  getExpectedColorType,
  loadColorRanges,
  prefetchRangesIntoCache,
  isAbortError,
  computeLoadLatency,
  recordLoadEvent,
  LoaderEventEmitter,
  OnceInit,
  warnExtendToAllNoDimensions,
  announceExtendToAllOnce,
} from '../loaders';
import { loadGSplatsChunkIndex, registerGSplatsArrayBounds } from './chunk-index-loader';
import { createEmptyGSplatsData } from './projection';
import type {
  LoaderMetrics,
  MonitorEvent,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import {
  choleskyPackedSize,
  choleskyDiagIndices,
  choleskyOffdiagIndices,
} from '../../types/gsplats';
import { GSplatsDataAccumulator, type AccumulatorStats } from '../accumulators/gsplats';
import { config as appConfig } from '../../config';
import type { UpdateProfiler, UpdateSession } from '../../profiling/update-profiler';
import { DecompressedChunkCache } from '../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../cache/decompressed-chunk-cache/cached-zarr-array';
import { ResidencyAccumulator } from '../../cache/residency-probe';
import { ChunkPrefetcher } from '../../cache/chunk-prefetcher';

/**
 * GSplats data loader using spatial indices for efficient nD queries.
 *
 * Key features:
 * - Chunk-based loading using spatial index
 * - Handles all array encoding types (broadcasted, quantized, LUT, etc.)
 * - Support for optional arrays (colors)
 */
export class GSplatsSpatialIndexLoader implements GSplatsDataLoader {
  private chunkIndex: ChunkSpatialIndex | null = null;
  private zarrLocation: zarr.Location<zarr.Readable>;
  private node: SceneNode;
  private _onceInit = new OnceInit();
  private rangeLoader: RangeLoader;
  private zarrStore: zarr.Readable | null = null;

  // Data accumulator for object pooling.
  private _accumulator: GSplatsDataAccumulator | null = null;

  // L0 decompressed chunk cache (optional, avoids Blosc decompression on repeat access)
  private l0Cache: DecompressedChunkCache | null = null;

  // Active cache-residency probe for the in-flight demand load (see
  // updateViewWithResidency); null at all other times.
  private _activeProbe: ResidencyAccumulator | null = null;
  // Per-update abort signal for the in-flight `updateView`; set at its top and
  // cleared in `finally`. Read by the `wrapWithCache` L0 proxy so a superseded
  // update bails before fetch/decode. Mirrors `_activeProbe`'s lifetime.
  private _activeSignal: AbortSignal | null = null;

  // Chunk prefetcher (optional, for registering array bounds to suppress 404s)
  private prefetcher: ChunkPrefetcher | null = null;

  // Suppress detail logs after first successful view update
  private _initialLoadDone = false;

  // LoaderMonitor surface — same shape as the points and lines facades.
  private readonly events = new LoaderEventEmitter();
  private readonly metrics: LoaderMetrics;
  private readonly activeQueries = new Map<string, QueryInfo>();
  private nextQueryId = 0;

  private arrays: {
    centers?: zarr.Array<zarr.DataType, zarr.Readable>;
    amplitudes?: zarr.Array<zarr.DataType, zarr.Readable>;
    // v3.1 split Cholesky layout (diagonal + off-diagonal)…
    cholesky_factors_diag?: zarr.Array<zarr.DataType, zarr.Readable>;
    cholesky_factors_offdiag?: zarr.Array<zarr.DataType, zarr.Readable>;
    // …or the legacy v3.0 single packed array.
    cholesky_factors?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
  } = {};

  constructor(
    zarrLocation: zarr.Location<zarr.Readable>,
    node: SceneNode,
    refRegistry?: ArrayRefRegistry,
    zarrStore?: zarr.Readable,
    profiler?: UpdateProfiler,
    l0Cache?: DecompressedChunkCache,
    prefetcher?: ChunkPrefetcher
  ) {
    this.zarrLocation = zarrLocation;
    this.node = node;
    this.rangeLoader = new RangeLoader(refRegistry || new ArrayRefRegistry());
    // Forward the per-update abort signal into worker decodes (LUT/quantized/
    // broadcasted) so a superseded update's decode bails before dispatch.
    this.rangeLoader.setSignalSource(() => this._activeSignal);
    this.zarrStore = zarrStore || null;
    this.l0Cache = l0Cache || null;
    this.prefetcher = prefetcher || null;
    // profiler parameter kept for API compatibility; session is passed directly to methods
    void profiler;

    this.metrics = {
      type: 'gsplats-spatial-index',
      path: node.path,
      queries: 0,
      loads: 0,
      evictions: 0,
      errors: 0,
      pointsLoaded: 0, // Shared loader metric; counts splats for gsplats.
      bytesLoaded: 0,
      visiblePoints: 0, // counts visible splats for gsplats
      avgQueryTime: 0,
      avgLoadTime: 0,
      memoryUsed: 0,
      memoryLimit: 0,
    };
  }

  /**
   * Thin wrapper around the shared `registerGSplatsArrayBounds` helper
   * so the call sites read more naturally than passing the prefetcher
   * and node path on every call.
   */
  private registerBounds(arrayName: string, array: zarr.Array<zarr.DataType, zarr.Readable>): void {
    registerGSplatsArrayBounds(this.prefetcher, this.node.path, arrayName, array);
  }

  /**
   * Initialize the loader by loading spatial index and opening arrays
   */
  async initialize(): Promise<void> {
    const attrs = this.node.attrs as unknown as GSplatsMetadata;

    // Load spatial index
    try {
      this.chunkIndex = await this.loadChunkBounds(attrs);

      if (!this.chunkIndex) {
        log.info(
          Modules.GSPLATS_SPATIAL_INDEX_LOADER,
          `No spatial index for GSplats ${this.node.path} - will load all data`
        );
      } else {
        log.query(
          Modules.GSPLATS_SPATIAL_INDEX_LOADER,
          `GSplats index loaded: ${this.chunkIndex.chunkCount} chunks`
        );
      }
    } catch (error) {
      log.error(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `Failed to load GSplats spatial index for ${this.node.path}:`,
        error
      );
      throw error;
    }

    // Open required arrays
    try {
      let centersArray = await zarr.open(this.zarrLocation.resolve('centers'), {
        kind: 'array',
      });
      let amplitudesArray = await zarr.open(this.zarrLocation.resolve('amplitudes'), {
        kind: 'array',
      });
      // Wrap with L0 cache if enabled (caches decoded chunks to avoid Blosc decompression)
      if (this.l0Cache) {
        centersArray = wrapWithCache(
          centersArray,
          this.l0Cache,
          `${this.node.path}/centers`,
          () => this._activeProbe,
          () => this._activeSignal
        );
        amplitudesArray = wrapWithCache(
          amplitudesArray,
          this.l0Cache,
          `${this.node.path}/amplitudes`,
          () => this._activeProbe,
          () => this._activeSignal
        );
      }
      this.arrays.centers = centersArray;
      this.arrays.amplitudes = amplitudesArray;

      // Register array bounds with prefetcher for upper-bounds checking
      this.registerBounds('centers', centersArray);
      this.registerBounds('amplitudes', amplitudesArray);

      // Cholesky factors: v3.1 stores a diagonal + off-diagonal split; v3.0
      // stores a single packed `cholesky_factors`. Presence of the diagonal
      // array selects the layout — the loader recombines the split into the
      // packed buffer the geometry expects (see loadCholeskyRanges).
      await this.openCholeskyArrays();
    } catch (e) {
      log.error(Modules.GSPLATS_SPATIAL_INDEX_LOADER, 'Failed to open required GSplats arrays:', e);
      throw e;
    }

    // Try to open optional arrays
    try {
      let colorsArray = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
      this.registerBounds('colors', colorsArray);
      if (this.l0Cache) {
        colorsArray = wrapWithCache(
          colorsArray,
          this.l0Cache,
          `${this.node.path}/colors`,
          () => this._activeProbe,
          () => this._activeSignal
        );
      }
      this.arrays.colors = colorsArray;
    } catch {
      log.info(Modules.GSPLATS_SPATIAL_INDEX_LOADER, 'No colors array found (using default white)');
    }

    // Initialize data accumulator for object pooling. The hot path
    // (loadGSplats below) reuses this accumulator's buffers across
    // updates when `useAccumulators` is true.
    if (appConfig.dataLoading.performance.useAccumulators) {
      const totalSplats = attrs.n_splats || 0;
      const ndim = attrs.ndim || this.arrays.centers?.shape[1] || 3;

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
            `capacity=${stats.capacity}, ndim=${ndim}, totalSplats=${totalSplats}`
        );
      }
    }
  }

  /**
   * Load gsplats data for the given view state
   * @param viewState - Current view state
   * @param session - Optional profiler session for nested timing
   */
  async loadGSplats(
    viewState: GSplatsViewState,
    session?: UpdateSession
  ): Promise<LoadedGSplatsData> {
    const startTime = Date.now();
    const queryId = `${this.node.path}-${startTime}-${this.nextQueryId++}`;

    try {
      const result = await this.loadGSplatsInternal(viewState, session, queryId, startTime);
      this.finishQueryTracking(queryId, startTime, 'complete');
      return result;
    } catch (err) {
      // A superseded scrub aborts the in-flight read on purpose; runLoaderUpdates
      // classifies it as 'superseded' (not a failure), so don't inflate the
      // error counter or flood the monitor's error stream with non-errors.
      // Still finish query tracking (removes the active query, records timing).
      this.finishQueryTracking(queryId, startTime, 'error');
      if (!isAbortError(err)) {
        this.metrics.errors += 1;
        // Emit a monitor 'error' event for parity with Points (see
        // lines-spatial-index-loader.ts comment).
        this.emitEvent({
          type: 'error',
          loader: 'gsplats-spatial-index',
          timestamp: Date.now(),
          data: {
            path: this.node.path,
            error: String(err),
          },
        });
      }
      throw err;
    }
  }

  private async loadGSplatsInternal(
    viewState: GSplatsViewState,
    session: UpdateSession | undefined,
    queryId: string,
    startTime: number
  ): Promise<LoadedGSplatsData> {
    await this._onceInit.ensure(() => this.initialize());

    const hasCholesky = !!(this.arrays.cholesky_factors || this.arrays.cholesky_factors_diag);
    if (!this.arrays.centers || !this.arrays.amplitudes || !hasCholesky) {
      throw new Error('[GSplatsLoader] Loader not properly initialized');
    }

    const attrs = this.node.attrs as unknown as GSplatsMetadata;

    // Query visible splat ranges (async to allow worker offload).
    let splatRanges: SplatRange[];
    if (session) {
      const querySession = session.begin('Spatial Query');
      try {
        splatRanges = await this.queryVisibleSplatRanges(viewState);
      } finally {
        querySession.end();
      }
    } else {
      splatRanges = await this.queryVisibleSplatRanges(viewState);
    }

    // Count total splats to load and begin query tracking.
    const totalSplats = splatRanges.reduce((sum, r) => sum + (r.end - r.start), 0);
    this.metrics.queries += 1;
    this.metrics.visiblePoints = totalSplats;
    this.activeQueries.set(queryId, {
      id: queryId,
      loader: 'gsplats-spatial-index',
      path: this.node.path,
      startTime,
      status: 'loading',
      cells: splatRanges.length,
      points: totalSplats,
      ranges: splatRanges as unknown as PointRange[],
    });
    this.emitEvent({
      type: 'query',
      loader: 'gsplats-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        ranges: splatRanges as unknown as PointRange[],
        cells: splatRanges.length,
        points: totalSplats,
        queryPosition: viewState.slicePosition,
        queryTolerance: viewState.tolerance,
      },
    });

    if (splatRanges.length === 0) {
      log.info(Modules.GSPLATS_SPATIAL_INDEX_LOADER, 'No visible gsplats - returning empty data');
      return createEmptyGSplatsData(attrs);
    }

    if (!this._initialLoadDone) {
      log.load(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `Loading ${totalSplats} gsplats from ${splatRanges.length} ranges`
      );
    }

    // Load directly into the accumulator buffers (zero allocations).
    if (this._accumulator && appConfig.dataLoading.performance.useAccumulators) {
      // Ensure capacity FIRST
      this._accumulator.ensureCapacity(totalSplats);

      // Initialize accumulator types based on array metadata (must be done BEFORE loading!)
      // This ensures colorBuffer has the correct type (Uint8/Uint16/Float32)
      if (this.arrays.colors) {
        const colorDtype = String(this.arrays.colors.dtype);
        const colorType = getExpectedColorType(colorDtype);
        // Create a small typed array to initialize accumulator types
        const sampleColors =
          colorType === 'Uint8Array'
            ? new Uint8Array(3)
            : colorType === 'Uint16Array'
              ? new Uint16Array(3)
              : new Float32Array(3);
        this._accumulator.fill(0, {
          positions: new Float32Array(attrs.ndim),
          amplitudes: new Float32Array(1),
          choleskyFactors: new Float32Array(choleskyPackedSize(attrs.ndim)),
          colors: sampleColors,
        });
      }

      // Get direct buffer references for zero-allocation loading (now colorBuffer has correct type!)
      const centerBuffer = this._accumulator.getCenterBuffer();
      const amplitudeBuffer = this._accumulator.getAmplitudeBuffer();
      const choleskyBuffer = this._accumulator.getCholeskyBuffer();

      // Load directly into accumulator buffers (ZERO intermediate allocations!)
      // All four attribute arrays load CONCURRENTLY — distinct zarr arrays
      // writing into distinct accumulator buffers; the global fetch gate
      // (utils/fetch-concurrency.ts) bounds total network concurrency.
      const loadSession = session?.begin('Load Arrays');
      try {
        const colorLoad = this.arrays.colors
          ? (async () => {
              // Use loadColorRanges for proper multi-type handling
              // NOTE: For LUT encoding, loadColorRanges may return a different buffer type
              // (Float32Array) than the accumulator's colorBuffer (Uint8Array based on stored dtype).
              // We MUST use the returned buffer since it contains the decoded colors.
              const colorBuffer = this._accumulator!.getColorBuffer();
              const loadedColors = await this.loadColorRanges(splatRanges, colorBuffer);

              // If loadColorRanges returned a different buffer (e.g., LUT decoded to Float32Array),
              // we need to update the accumulator with the new buffer
              if (loadedColors !== colorBuffer) {
                // Replace accumulator's color buffer with the decoded colors
                // This handles LUT encoding where decoded output is Float32Array
                this._accumulator!.setColorBuffer(loadedColors);
              }
            })()
          : Promise.resolve();

        await Promise.all([
          this.loadArrayRanges('centers', splatRanges, attrs.ndim, centerBuffer),
          this.loadArrayRanges('amplitudes', splatRanges, 1, amplitudeBuffer),
          this.loadCholeskyRanges(splatRanges, attrs.ndim, choleskyBuffer),
          colorLoad,
        ]);
      } finally {
        loadSession?.end();
      }

      // Return from accumulator (subarrays, zero copy!)
      // NO fill() needed - data already in buffers!
      return this._accumulator.getData(totalSplats);
    }

    // Fallback: Load to separate arrays (allocations when accumulator disabled)
    let centers: Float32Array;
    let amplitudes: Float32Array;
    let choleskyFactors: Float32Array;
    let colors: Float32Array | Uint8Array | Uint16Array | null = null;

    // All four attribute arrays load CONCURRENTLY (distinct zarr arrays,
    // distinct freshly-allocated output buffers).
    const loadSession = session?.begin('Load Arrays');
    try {
      [centers, amplitudes, choleskyFactors, colors] = await Promise.all([
        this.loadArrayRanges('centers', splatRanges, attrs.ndim),
        this.loadArrayRanges('amplitudes', splatRanges, 1),
        this.loadCholeskyRanges(splatRanges, attrs.ndim),
        this.arrays.colors ? this.loadColorRanges(splatRanges) : Promise.resolve(null),
      ]);
    } finally {
      loadSession?.end();
    }

    return {
      positions: centers,
      amplitudes,
      choleskyFactors,
      colors,
      splatCount: totalSplats,
      ndim: attrs.ndim,
    };
  }

  /**
   * Update view for new position.
   * @param viewState - Current view state
   * @param session - Optional profiler session for nested timing
   */
  async updateView(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedGSplatsData> {
    // Publish the per-update signal for the L0 proxy chokepoint, then clear it
    // in `finally` so a later cache hit/prefetch isn't seen as abortable.
    this._activeSignal = signal ?? null;
    try {
      const result = await this.loadGSplats(viewState, session);
      if (!this._initialLoadDone) {
        this._initialLoadDone = true;
        this.rangeLoader.setVerbose(false);
      }
      return result;
    } finally {
      this._activeSignal = null;
    }
  }

  /**
   * Like {@link updateView} but also reports whether the load was served
   * entirely from cache (see PointsSpatialIndexLoader.updateViewWithResidency).
   */
  async updateViewWithResidency(
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<{ data: LoadedGSplatsData; allResident: boolean }> {
    const probe = new ResidencyAccumulator();
    this._activeProbe = probe;
    try {
      const data = await this.updateView(viewState, session, signal);
      return { data, allResident: probe.allResident };
    } finally {
      this._activeProbe = null;
    }
  }

  /**
   * Probe the gsplats `chunk_bounds` array.
   *
   * Implementation lives in `gsplats/chunk-index-loader.ts`. The thin
   * wrapper here exists for symmetry with the points + lines facades,
   * which follow the same pattern.
   */
  private async loadChunkBounds(attrs: GSplatsMetadata): Promise<ChunkSpatialIndex | null> {
    return loadGSplatsChunkIndex(this.zarrLocation, attrs);
  }

  /**
   * Query visible splat ranges based on view state.
   *
   * Delegates the chunk-bounds AABB scan and range coalescing to the canonical
   * `SpatialQueryBuilder`, which also handles the `extend_to_all` short-circuit.
   * Returns a load-all range when no spatial index is available.
   */
  private async queryVisibleSplatRanges(viewState: GSplatsViewState): Promise<SplatRange[]> {
    const attrs = this.node.attrs as unknown as GSplatsMetadata;
    const extendDims: string[] = attrs.extend_to_all || [];

    warnExtendToAllNoDimensions({
      extendDims,
      hasResolvedDimensions: !!viewState.dimensions && viewState.dimensions.length > 0,
      nodePath: this.node.path,
      logModule: Modules.GSPLATS_SPATIAL_INDEX_LOADER,
    });

    if (!this.chunkIndex) {
      return [{ start: 0, end: attrs.n_splats }];
    }

    if (!this._initialLoadDone) {
      announceExtendToAllOnce({
        extendDims,
        nodePath: this.node.path,
        logModule: Modules.GSPLATS_SPATIAL_INDEX_LOADER,
      });
    }

    const ranges = await new SpatialQueryBuilder(this.chunkIndex, viewState, {
      geometryType: 'gsplats',
      totalElements: attrs.n_splats,
      chunkSize: attrs.chunk_size,
      extendDims,
      logModule: Modules.GSPLATS_SPATIAL_INDEX_LOADER,
    }).execute();

    return ranges;
  }

  /**
   * Load array ranges with optional target buffer for zero-allocation operation.
   *
   * Uses RangeLoader for unified encoding dispatch (broadcasted, quantized, lut, direct).
   * Array references are handled specially (need zarrStore access to resolve target).
   *
   * @param arrayName - Name of array to load
   * @param ranges - Ranges to load
   * @param elementsPerSplat - Elements per splat
   * @param targetBuffer - Optional target buffer (for accumulator integration)
   * @returns Loaded data (new array or subarray of target)
   */
  /**
   * Open the Cholesky factor array(s), supporting both the v3.1 split layout
   * (`cholesky_factors_diag` + optional `cholesky_factors_offdiag`) and the
   * legacy v3.0 single packed `cholesky_factors`. The off-diagonal array is
   * absent for 1D gsplats. Recombination into the packed buffer happens in
   * {@link loadCholeskyRanges}.
   */
  private async openCholeskyArrays(): Promise<void> {
    const wrap = (
      arr: zarr.Array<zarr.DataType, zarr.Readable>,
      name: string
    ): zarr.Array<zarr.DataType, zarr.Readable> =>
      this.l0Cache
        ? wrapWithCache(
            arr,
            this.l0Cache,
            `${this.node.path}/${name}`,
            () => this._activeProbe,
            () => this._activeSignal
          )
        : arr;

    let diagArray: zarr.Array<zarr.DataType, zarr.Readable> | undefined;
    try {
      diagArray = await zarr.open(this.zarrLocation.resolve('cholesky_factors_diag'), {
        kind: 'array',
      });
    } catch (e) {
      // Only a genuine "not found" means this is a legacy v3.0 single-array
      // file. A transient/network/permission error must surface, not be
      // silently misread as "no split" (which would then fail confusingly on
      // the legacy open below).
      if (!zarr.isNotFoundError(e)) throw e;
      diagArray = undefined;
    }

    if (diagArray) {
      // v3.1 split layout.
      diagArray = wrap(diagArray, 'cholesky_factors_diag');
      this.arrays.cholesky_factors_diag = diagArray;
      this.registerBounds('cholesky_factors_diag', diagArray);

      let offdiagArray: zarr.Array<zarr.DataType, zarr.Readable> | undefined;
      try {
        offdiagArray = await zarr.open(this.zarrLocation.resolve('cholesky_factors_offdiag'), {
          kind: 'array',
        });
      } catch (e) {
        // A missing off-diagonal array is legitimate only for 1D gsplats; a
        // transient error must surface. (loadCholeskyRanges still rejects a
        // d > 1 store whose off-diagonal is genuinely absent.)
        if (!zarr.isNotFoundError(e)) throw e;
        offdiagArray = undefined; // 1D gsplats: no off-diagonal terms
      }
      if (offdiagArray) {
        offdiagArray = wrap(offdiagArray, 'cholesky_factors_offdiag');
        this.arrays.cholesky_factors_offdiag = offdiagArray;
        this.registerBounds('cholesky_factors_offdiag', offdiagArray);
      }
      return;
    }

    // Legacy v3.0 single packed array (required).
    let choleskyArray = await zarr.open(this.zarrLocation.resolve('cholesky_factors'), {
      kind: 'array',
    });
    choleskyArray = wrap(choleskyArray, 'cholesky_factors');
    this.arrays.cholesky_factors = choleskyArray;
    this.registerBounds('cholesky_factors', choleskyArray);
  }

  /**
   * Load Cholesky factors into the packed (N, k) form, recombining the v3.1
   * split arrays when present. For the legacy single-array layout this is a
   * direct passthrough to {@link loadArrayRanges}. On success every packed
   * position is written (diagonal ∪ off-diagonal = all k columns), so a reused
   * target buffer never leaks stale values. The corrupt-file precondition (a
   * d > 1 store missing the off-diagonal array) is checked BEFORE any write, so
   * a throw never leaves the (possibly reused) target buffer half-populated.
   *
   * @param ranges - Visible splat ranges
   * @param ndim - Dimensionality (k = ndim*(ndim+1)/2 packed elements/splat)
   * @param targetBuffer - Optional packed output buffer (zero-alloc path)
   */
  private async loadCholeskyRanges(
    ranges: SplatRange[],
    ndim: number,
    targetBuffer?: Float32Array
  ): Promise<Float32Array> {
    const k = choleskyPackedSize(ndim);

    // Legacy v3.0: single packed array — load directly, no interleave.
    if (this.arrays.cholesky_factors) {
      return this.loadArrayRanges('cholesky_factors', ranges, k, targetBuffer);
    }

    const d = ndim;
    const offLen = k - d;

    // Precondition FIRST (before touching the target buffer): the off-diagonal
    // array is legitimately absent ONLY for 1D gsplats (offLen === 0). For
    // d > 1 a missing off-diagonal array means a corrupt / partially-written
    // file — fail loud rather than silently zero/stale-filling the off-diagonals
    // (which would scramble every splat's covariance). Checking up front means a
    // throw never half-populates a reused accumulator buffer. Mirrors the Python
    // reader, where merge_tril() raises on a size mismatch.
    if (offLen > 0 && !this.arrays.cholesky_factors_offdiag) {
      throw new Error(
        `[GSplatsLoader] ${this.node.path}: missing 'cholesky_factors_offdiag' ` +
          `for ${ndim}D splats (expected ${offLen} off-diagonal elements per splat). ` +
          'The .gsplats.zarr is corrupt or was only partially written.'
      );
    }

    const totalSplats = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const packed = targetBuffer ?? new Float32Array(totalSplats * k);

    // Load both halves concurrently (two independent zarr reads), then scatter
    // each into its packed columns. Diagonal is always present in the split
    // layout; off-diagonal is read only when it exists (d > 1).
    const [diag, offdiag] = await Promise.all([
      this.loadArrayRanges('cholesky_factors_diag', ranges, d),
      offLen > 0
        ? this.loadArrayRanges('cholesky_factors_offdiag', ranges, offLen)
        : Promise.resolve(null),
    ]);

    // Per-column dequantizers from each array's encoding metadata. For the v3.1
    // differential encodings (diag log-uint8/16, off signed-log-uint8/16) the
    // arrays loaded as raw integer levels (routed to `direct`); we invert
    // per-column here, mirroring the Python decoder. For float32/legacy arrays
    // the dequantizer is the identity, so values pass through unchanged.
    const diagAttrs = this.arrays.cholesky_factors_diag?.attrs as unknown as
      | { encoding?: Parameters<typeof ArrayDecoder.makePerChannelDequant>[0] }
      | undefined;
    const diagDequant = ArrayDecoder.makePerChannelDequant(diagAttrs?.encoding, d);

    const diagIdx = choleskyDiagIndices(ndim);
    for (let s = 0; s < totalSplats; s++) {
      const base = s * k;
      const dbase = s * d;
      for (let c = 0; c < d; c++) packed[base + diagIdx[c]] = diagDequant(diag[dbase + c], c);
    }

    if (offdiag) {
      const offAttrs = this.arrays.cholesky_factors_offdiag?.attrs as unknown as
        | { encoding?: Parameters<typeof ArrayDecoder.makePerChannelDequant>[0] }
        | undefined;
      const offDequant = ArrayDecoder.makePerChannelDequant(offAttrs?.encoding, offLen);
      const offIdx = choleskyOffdiagIndices(ndim);
      for (let s = 0; s < totalSplats; s++) {
        const base = s * k;
        const obase = s * offLen;
        for (let c = 0; c < offLen; c++) {
          packed[base + offIdx[c]] = offDequant(offdiag[obase + c], c);
        }
      }
    }

    return packed;
  }

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

    // Use target buffer or allocate (ZERO allocation when targetBuffer provided!)
    const output = targetBuffer ? targetBuffer : new Float32Array(totalElements);
    const attrs = array.attrs as unknown as ArrayMetadata;

    // The shared helper resolves array_ref against zarrStore when needed and
    // delegates to RangeLoader.loadRanges for everything else. The hint
    // elementsPerItem is only consulted when no ref is in play; ref targets
    // recompute it from their own shape.
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    await this.rangeLoader.loadRangesResolvingRef(
      array,
      attrs,
      ranges as LoadRange[],
      output,
      totalSplats,
      elementsPerSplat,
      storeToUse,
      'GSplats'
    );

    this.recordLoadMetrics(arrayName, totalSplats, output);
    return output;
  }

  /**
   * Load color ranges with multi-type support (preserves original_dtype)
   *
   * This method handles the full encoding/decoding pipeline for colors,
   * including original_dtype restoration for encoded arrays.
   */
  private async loadColorRanges(
    ranges: SplatRange[],
    targetBuffer?: Float32Array | Uint8Array | Uint16Array
  ): Promise<Float32Array | Uint8Array | Uint16Array> {
    const array = this.arrays.colors;
    if (!array) {
      throw new Error('[GSplatsLoader] Colors array not initialized');
    }
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    const totalSplats = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    const output = await loadColorRanges(
      array,
      ranges,
      this.rangeLoader,
      storeToUse,
      'GSplats',
      targetBuffer
    );
    this.recordLoadMetrics('colors', totalSplats, output);
    return output;
  }

  /**
   * Prefetch chunks for the given view state into the cache without decoding.
   *
   * Performs the same spatial index query as updateView() and issues zarr
   * get() calls for each visible range on every array.  The fetched data
   * populates the HTTP cache and L0 decompressed-chunk cache but is NOT
   * accumulated into output buffers — the typed-array results are immediately
   * discarded.  This makes the subsequent updateView() call a fast cache hit
   * without the memory cost of allocating full-size output arrays that would
   * only be thrown away.
   */
  async prefetchChunks(viewState: GSplatsViewState): Promise<void> {
    await this._onceInit.ensure(() => this.initialize());

    if (!this.arrays.centers) return;

    // Query which splat ranges are visible
    let splatRanges: SplatRange[];
    try {
      splatRanges = await this.queryVisibleSplatRanges(viewState);
    } catch {
      // Spatial query failed (e.g. malformed viewState during an animation
      // edge case). Skip the prefetch silently — production loadGSplats will
      // surface the error on the next demand frame. Matches the points/lines
      // prefetch siblings so this best-effort path is self-protecting.
      return;
    }
    if (splatRanges.length === 0) return;

    // Warm every array over the visible ranges. The reads populate L0/L1/L2 as
    // a side-effect and are discarded — no output buffers allocated.
    const arrays = [
      this.arrays.centers,
      this.arrays.amplitudes,
      // v3.1 split Cholesky arrays, or the legacy v3.0 single packed array.
      this.arrays.cholesky_factors_diag,
      this.arrays.cholesky_factors_offdiag,
      this.arrays.cholesky_factors,
      this.arrays.colors,
    ].filter((a): a is zarr.Array<zarr.DataType, zarr.Readable> => a != null);

    await prefetchRangesIntoCache(arrays, splatRanges);
  }

  /**
   * Get accumulator stats for memory monitoring
   */
  getAccumulatorStats(): AccumulatorStats | null {
    return this._accumulator?.getStats() ?? null;
  }

  // ────────────────────────────────────────────────────────────────────
  // LoaderMonitor surface — same shape as the points and lines facades.
  // ────────────────────────────────────────────────────────────────────

  addEventListener(listener: MonitorEventListener): void {
    this.events.add(listener);
  }

  removeEventListener(listener: MonitorEventListener): void {
    this.events.remove(listener);
  }

  getMetrics(): LoaderMetrics {
    return { ...this.metrics };
  }

  getActiveQueries(): QueryInfo[] {
    return Array.from(this.activeQueries.values());
  }

  /**
   * Update metrics + emit a 'load' event. Mirrors the points / lines facade's
   * recordLoadMetrics shape.
   */
  private recordLoadMetrics(arrayName: string, items: number, output: ArrayBufferView): void {
    const queryStart = this.activeQueries.values().next().value?.startTime;
    const loadTime = computeLoadLatency(queryStart);
    const bytes = output.byteLength;

    recordLoadEvent(this.metrics, items, bytes, loadTime);

    // Resident memory = current accumulator allocation (MB → bytes). Assignment
    // (not +=): memoryUsed is a live footprint that grows/shrinks with the pool,
    // unlike the cumulative bytesLoaded counter updated above.
    this.metrics.memoryUsed = Math.round((this.getAccumulatorStats()?.memoryMB ?? 0) * 1024 * 1024);

    this.emitEvent({
      type: 'load',
      loader: 'gsplats-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        arrayName,
        points: items,
        memory: bytes,
        latency: loadTime,
      },
    });
  }

  private emitEvent(event: MonitorEvent): void {
    this.events.emit(event);
  }

  private finishQueryTracking(
    queryId: string,
    startTime: number,
    status: 'complete' | 'error'
  ): void {
    const query = this.activeQueries.get(queryId);
    if (query) {
      query.status = status;
      query.endTime = Date.now();
      this.activeQueries.delete(queryId);
    }
    const queryTime = Date.now() - startTime;
    if (this.metrics.queries > 0) {
      this.metrics.avgQueryTime =
        (this.metrics.avgQueryTime * (this.metrics.queries - 1) + queryTime) / this.metrics.queries;
    }
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.chunkIndex = null;
    this.arrays = {};
    this._onceInit.reset();
    this.events.clear();
    this.activeQueries.clear();

    // Dispose accumulator
    if (this._accumulator) {
      this._accumulator.dispose();
      this._accumulator = null;
    }
  }
}
