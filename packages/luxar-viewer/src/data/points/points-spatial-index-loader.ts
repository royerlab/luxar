/**
 * Point spatial index-based data loader for efficient nD points loading.
 *
 * This loader uses the point spatial index built by the Python compiler to load
 * only the points that are visible in the current nD slice. It ensures
 * all attributes are loaded with the same point ranges for proper alignment.
 */

import * as zarr from '../zarr';
import { isArrayListed } from '../loaders/optional-array-listing';
import { log, Modules } from '../../utils/log';
import { clamp } from '../../utils/clamp';
import {
  DataLoader,
  ViewState,
  LoadedPointsData,
  PointRange,
  SceneNode,
  type ColorArray,
  type ScalarArray,
  type PointScalarArray,
} from '../data-loader-types';
import {
  calculateSpatialQueryTolerance,
  fallbackQueryTolerance,
  type EffectiveRadiusConfig,
} from './effective-radius-calculator';
import {
  loadPointsChunkIndex,
  registerPointsArrayBounds,
  type PointsChunkIndex,
  type PointsNodeAttrsForIndex,
} from './chunk-index-loader';
import {
  createEmptyPointsData as createEmptyPointsDataHelper,
  projectPointsTo3D,
  type ProjectionContext,
  type ProjectionTargetBuffers,
} from './projection';
import { getPointsBackend } from '../../workers/data-worker/projection/in-process';
import type { WasmModule } from '../../wasm/types';
import type {
  MonitorEvent,
  MonitorEventListener,
  LoaderMonitor,
  LoaderMetrics,
  QueryInfo,
} from '../../types/data-monitor-types';
import { ArrayDecoder, ArrayRefRegistry, type ArrayMetadata } from '../array-decoder/decoder';
import {
  RangeLoader,
  SpatialQueryBuilder,
  type BaseViewState,
  type LoadRange,
  loadColorRanges,
  colorComponentsOf,
  prefetchRangesIntoCache,
  planChunkBoundaryViewStates,
  OnceInit,
  makeInitialLoaderMetrics,
  buildSpatialIndexMetrics,
  loadSliceWithCache,
  recordLoadMetrics,
  runWithActiveSignal,
  runWithResidencyProbe,
  type SpatialFacadeCtx,
  LoaderEventEmitter,
  warnExtendToAllNoDimensions,
  announceExtendToAllOnce,
  formatTolerance,
} from '../loaders';
import type { ZarrSceneAttrs } from '../../types/zarr';
import type { PointsMetadata } from '../../types/points';
import { LoadedPointsDataAccumulator, type AccumulatorStats } from '../accumulators/points';
import { config as appConfig } from '../../config';
import type { UpdateSession } from '../../profiling/update-profiler';
import { DecompressedChunkCache } from '../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../cache/decompressed-chunk-cache/cached-zarr-array';
import { ResidencyAccumulator } from '../../cache/residency-probe';
import { ChunkPrefetcher } from '../../cache/chunk-prefetcher';
import type { SliceCache } from '../../cache/slice-cache';

/**
 * `PointsNodeAttrs` and `PointsChunkIndex` are now defined alongside the
 * chunk-index probe in `data/point-loader/chunk-index-loader.ts`; the
 * loader keeps the original `PointsNodeAttrs` alias for in-file readability.
 */
type PointsNodeAttrs = PointsNodeAttrsForIndex;

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isNotFoundError(error: unknown): boolean {
  const message = getErrorMessage(error);
  return message.includes('404') || message.includes('Not Found');
}

/**
 * Loader implementation that uses spatial indices for efficient nD queries.
 *
 * Key features:
 * - Queries spatial index to find visible point ranges
 * - Loads all attributes with identical ranges (fixes alignment bug)
 * - Projects nD points to 3D display space
 * - Real-time monitoring and performance tracking
 */
export class PointsSpatialIndexLoader implements DataLoader, LoaderMonitor {
  private chunkIndex: PointsChunkIndex | null = null;
  // Total points count for datasets without chunk-based index (simple fallback)
  private totalPointsNoIndex: number = 0;
  private _effectiveRadiusConfig: EffectiveRadiusConfig | null = null;
  private zarrLocation: zarr.Location<zarr.Readable>;
  private node: SceneNode;
  private _onceInit = new OnceInit();
  private arrays: {
    positions?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
    radii?: zarr.Array<zarr.DataType, zarr.Readable>;
    sharpness?: zarr.Array<zarr.DataType, zarr.Readable>;
    /** optional scalars array for colormap lookup. */
    scalars?: zarr.Array<zarr.DataType, zarr.Readable>;
  } = {};

  // Range loader for unified encoding dispatch (replaces decoder for range-based loading)
  private rangeLoader: RangeLoader;

  // Data accumulator for object pooling.
  private _accumulator: LoadedPointsDataAccumulator | null = null;
  private _accumulatorConfig: { capacity: number; ndim: number; totalPoints: number } | null = null;

  /**
   * Components per color item, learned from the zarr `colors` shape at
   * open time: 3 = RGB, 4 = RGBA (alpha = per-point opacity). Strides
   * the accumulator buffers and the projection compaction. Mirrors
   * `GSplatsSpatialIndexLoader.colorComponents`.
   */
  private colorComponents: 3 | 4 = 3;

  // Monitoring
  private readonly events = new LoaderEventEmitter();
  private readonly metrics: LoaderMetrics;
  private readonly activeQueries = new Map<string, QueryInfo>();
  private nextQueryId = 0;
  // Shared facade-helper context (data/loaders/spatial-facade.ts): stable
  // references + this-bound accessors, built once in the constructor.
  private readonly facadeCtx: SpatialFacadeCtx;
  // Cumulative queried cells across the session (drives avgCellsPerQuery).
  private totalQueryCells = 0;
  private zarrStore: zarr.Readable | null = null;

  // L0 decompressed chunk cache (optional, avoids Blosc decompression on repeat access)
  private l0Cache: DecompressedChunkCache | null = null;

  // Active cache-residency probe for the in-flight demand load. Set by
  // updateViewWithResidency() and read by the wrapped arrays' getChunk;
  // null at all other times (so prefetch traffic isn't recorded).
  private _activeProbe: ResidencyAccumulator | null = null;
  // Per-update abort signal for the in-flight `updateView`. Set at the top of
  // `updateView` and cleared in its `finally`; read by the `wrapWithCache` L0
  // proxy (via the `() => this._activeSignal` thunk below) so a superseded
  // update bails before fetch/decode. Mirrors `_activeProbe`'s lifetime.
  private _activeSignal: AbortSignal | null = null;

  // Chunk prefetcher (optional, for registering array bounds to suppress 404s)
  private prefetcher: ChunkPrefetcher | null = null;

  // Per-slice decoded-result cache (S-cache). Wired ONLY for plain-leaf nodes
  // by the plain factory helper — progressive sub-LOD instances stay
  // cache-less (their wrapper owns the whole ladder; see loader-factory.ts).
  private sliceCache: SliceCache | null = null;

  // Suppress detail logs after first successful view update
  private _initialLoadDone = false;

  /**
   * Get node attributes with proper PointsMetadata typing.
   * This provides type-safe access to point node attributes.
   */
  private get attrs(): PointsMetadata {
    return this.node.attrs as unknown as PointsMetadata;
  }

  constructor(
    zarrLocation: zarr.Location<zarr.Readable>,
    node: SceneNode,
    refRegistry?: ArrayRefRegistry,
    zarrStore?: zarr.Readable,
    l0Cache?: DecompressedChunkCache,
    prefetcher?: ChunkPrefetcher,
    sliceCache?: SliceCache
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
    this.sliceCache = sliceCache || null;
    this.metrics = makeInitialLoaderMetrics('point-spatial-index', node.path);
    this.facadeCtx = {
      metrics: this.metrics,
      activeQueries: this.activeQueries,
      loader: 'point-spatial-index',
      path: node.path,
      sliceCache: this.sliceCache,
      nextQueryId: () => this.nextQueryId++,
      accumulatorMemoryMB: () => this.getAccumulatorStats()?.memoryMB ?? 0,
      emit: (event) => this.emitEvent(event),
    };
  }

  /**
   * Register array shape with the prefetcher for upper-bounds checking.
   * Thin wrapper around the shared `registerPointsArrayBounds` helper
   * so the three call sites in `initialize()` keep their compact form.
   */
  private registerBounds(arrayName: string, array: zarr.Array<zarr.DataType, zarr.Readable>): void {
    registerPointsArrayBounds(this.prefetcher, this.node.path, arrayName, array);
  }

  /**
   * Initialize the loader by loading spatial index and opening arrays
   */
  async initialize(): Promise<void> {
    // Load chunk-based spatial index
    try {
      this.chunkIndex = await loadPointsChunkIndex(
        this.zarrLocation,
        this.node.attrs as PointsNodeAttrs
      );

      if (!this.chunkIndex) {
        // For 3D datasets without Morton ordering, fall back to loading all points
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `No chunk-based index for ${this.node.path} - will load all points`
        );

        // For 3D datasets, we need to get the actual number of points from the positions array
        // Open the positions array to get its shape
        try {
          const positionsArray = await zarr.open(this.zarrLocation.resolve('positions'), {
            kind: 'array',
          });
          // The shape is [n_points, 3] for 3D data
          this.totalPointsNoIndex = positionsArray.shape[0];

          // For encoded arrays (like array_ref), shape may be [0, k] but original_shape has the real count
          if (this.totalPointsNoIndex === 0) {
            const attrs = positionsArray.attrs as unknown as ArrayMetadata;
            if (attrs.encoding?.original_shape && attrs.encoding.original_shape.length > 0) {
              this.totalPointsNoIndex = attrs.encoding.original_shape[0];
              log.query(
                Modules.SPATIAL_INDEX_LOADER,
                `Detected ${this.totalPointsNoIndex} points from encoding.original_shape for ${this.node.path}`
              );
            }
          }

          if (this.totalPointsNoIndex > 0) {
            log.query(
              Modules.SPATIAL_INDEX_LOADER,
              `Detected ${this.totalPointsNoIndex} points in 3D dataset`
            );
          }

          // Check if this is an nD dataset (ndim > 3) without spatial indexing
          // This is inefficient because ALL points must be loaded for each view
          const ndim = positionsArray.shape[1] || 3;
          if (ndim > 3) {
            log.warning(
              Modules.SPATIAL_INDEX_LOADER,
              `⚠️ nD dataset (${ndim}D) without spatial index for ${this.node.path}. ` +
                `This is inefficient: ALL ${this.totalPointsNoIndex} points will be loaded for every view. ` +
                'Enable spatial ordering in Python with enable_spatial_ordering=True for efficient nD slicing.'
            );
          }
        } catch {
          log.warning(
            Modules.SPATIAL_INDEX_LOADER,
            `Could not determine point count for ${this.node.path}, using 0`
          );
          this.totalPointsNoIndex = 0;
        }
      } else {
        // Chunk index loaded successfully
        log.query(
          Modules.SPATIAL_INDEX_LOADER,
          `Chunk index loaded: ${this.chunkIndex.metadata.total_chunks} chunks, ${this.chunkIndex.metadata.total_points} points`
        );
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `  Ordering: ${this.chunkIndex.metadata.ordering}, ${this.chunkIndex.metadata.ndim}D space`
        );
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `  Ordering dims: [${this.chunkIndex.metadata.ordering_dims.join(', ')}], Slice dims: [${this.chunkIndex.metadata.slice_dims.join(', ')}]`
        );
      }

      // Load spatial extension configuration from scene_dimensions (derived from root attributes)
      // This determines which dimensions points physically extend through vs categorical dimensions
      const spatialExtendDims = await this.loadSpatialExtendDimsFromSceneDimensions();
      if (spatialExtendDims) {
        this._effectiveRadiusConfig = {
          spatialExtendDims: spatialExtendDims,
          maxRadius: this.attrs.max_radius || appConfig.dataLoading.spatial.defaultMaxRadius,
        };

        // Log which dimensions are spatial
        const spatialDims = spatialExtendDims
          .map((isSpatial: boolean, idx: number) => (isSpatial ? idx : null))
          .filter((idx: number | null) => idx !== null);

        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Spatial extension enabled for dimensions: [${spatialDims.join(', ')}]`
        );
      }
    } catch (error) {
      log.error(
        Modules.SPATIAL_INDEX_LOADER,
        `Failed to load spatial index for ${this.node.path}:`,
        error
      );
      throw error;
    }

    // Open arrays for later access
    try {
      let positionsArray = await zarr.open(this.zarrLocation.resolve('positions'), {
        kind: 'array',
      });
      this.registerBounds('positions', positionsArray);
      // Wrap with L0 cache if enabled (caches decoded chunks to avoid Blosc decompression)
      if (this.l0Cache) {
        positionsArray = wrapWithCache(
          positionsArray,
          this.l0Cache,
          `${this.node.path}/positions`,
          () => this._activeProbe,
          () => this._activeSignal
        );
      }
      this.arrays.positions = positionsArray;
    } catch (e) {
      log.error(Modules.SPATIAL_INDEX_LOADER, 'Failed to open positions array:', e);
      throw e;
    }

    // Try to open optional arrays - these may not exist and that's OK
    if (!isArrayListed(this.node, 'colors')) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        'No colors array in the store listing (using default colors)'
      );
    } else {
      try {
        let colorsArray = await zarr.open(this.zarrLocation.resolve('colors'), { kind: 'array' });
        this.registerBounds('colors', colorsArray);
        // Wrap with L0 cache if enabled
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
        // Learn the color layout (3 = RGB, 4 = RGBA) from the zarr shape at
        // open time — it strides every downstream copy (accumulator,
        // projection compaction, texel packing). Mirrors the gsplat loader.
        this.colorComponents = colorComponentsOf(colorsArray);
      } catch (e: unknown) {
        // Colors are optional - only log if it's not a 404
        if (!isNotFoundError(e)) {
          log.info(Modules.SPATIAL_INDEX_LOADER, 'No colors array found (using default colors)');
        }
      }
    }

    if (!isArrayListed(this.node, 'radii')) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        'No radii array in the store listing (using default radii)'
      );
    } else {
      try {
        let radiiArray = await zarr.open(this.zarrLocation.resolve('radii'), { kind: 'array' });
        this.registerBounds('radii', radiiArray);
        // Wrap with L0 cache if enabled
        if (this.l0Cache) {
          radiiArray = wrapWithCache(
            radiiArray,
            this.l0Cache,
            `${this.node.path}/radii`,
            () => this._activeProbe,
            () => this._activeSignal
          );
        }
        this.arrays.radii = radiiArray;
      } catch (e: unknown) {
        // Radii are optional - only log if it's not a 404
        if (!isNotFoundError(e)) {
          log.info(Modules.SPATIAL_INDEX_LOADER, 'No radii array found (using default radii)');
        }
      }
    }

    if (!isArrayListed(this.node, 'sharpnesses')) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        'No sharpnesses array in the store listing (using default sharpness)'
      );
    } else {
      try {
        let sharpnessArray = await zarr.open(this.zarrLocation.resolve('sharpnesses'), {
          kind: 'array',
        });
        this.registerBounds('sharpnesses', sharpnessArray);
        // Wrap with L0 cache if enabled
        if (this.l0Cache) {
          sharpnessArray = wrapWithCache(
            sharpnessArray,
            this.l0Cache,
            `${this.node.path}/sharpnesses`,
            () => this._activeProbe,
            () => this._activeSignal
          );
        }
        this.arrays.sharpness = sharpnessArray;
      } catch (e: unknown) {
        // Sharpness is optional - only log if it's not a 404
        if (!isNotFoundError(e)) {
          log.info(
            Modules.SPATIAL_INDEX_LOADER,
            'No sharpness array found (using default sharpness)'
          );
        }
      }
    }

    // open optional `scalars` array (per-point values for colormap lookup).
    // Gated on `attrs.has_scalars` so we don't 404-spam when it's not authored.
    if (this.attrs?.has_scalars) {
      try {
        let scalarsArray = await zarr.open(this.zarrLocation.resolve('scalars'), {
          kind: 'array',
        });
        this.registerBounds('scalars', scalarsArray);
        if (this.l0Cache) {
          scalarsArray = wrapWithCache(
            scalarsArray,
            this.l0Cache,
            `${this.node.path}/scalars`,
            () => this._activeProbe,
            () => this._activeSignal
          );
        }
        this.arrays.scalars = scalarsArray;
      } catch (e: unknown) {
        if (!isNotFoundError(e)) {
          log.info(
            Modules.SPATIAL_INDEX_LOADER,
            'has_scalars=true but no scalars array found; colormap mode disabled.'
          );
        }
      }
    }

    // Initialize data accumulator for object pooling. The hot path
    // (loadPoints below) reuses this accumulator's buffers across
    // updates when `useAccumulators` is true — see the
    // `if (this._accumulator && appConfig.dataLoading.performance.useAccumulators)`
    // branch later in this file.
    if (appConfig.dataLoading.performance.useAccumulators) {
      const totalPoints = this.chunkIndex?.metadata.total_points || this.totalPointsNoIndex || 0;
      const ndim = this.chunkIndex?.metadata.ndim || this.arrays.positions?.shape?.[1] || 3;
      const initialCapacity = Math.min(
        appConfig.dataLoading.performance.initialAccumulatorCapacity,
        Math.max(1024, Math.ceil(totalPoints / 10)) // At least 1024, or ~10% of total
      );

      this._accumulatorConfig = { capacity: initialCapacity, ndim, totalPoints };
      this._accumulator = new LoadedPointsDataAccumulator(initialCapacity, ndim, totalPoints);

      if (appConfig.dataLoading.performance.enablePerformanceMonitoring) {
        const stats = this._accumulator.getStats();
        log.info(
          Modules.DATA_ACCUMULATOR,
          `Initialized LoadedPointsDataAccumulator for ${this.node.path}: ` +
            `capacity=${stats.capacity}, ndim=${ndim}, totalPoints=${totalPoints}`
        );
      }
    }
  }

  /**
   * Load points data for the given view state
   * @param viewState - Current view state
   * @param session - Optional profiler session for nested timing
   */
  async loadPoints(viewState: ViewState, session?: UpdateSession): Promise<LoadedPointsData> {
    // Plain-leaf S-cache + query close-out via the shared facade template
    // (see `loadSliceWithCache`). Points-specific wrinkle: the cached payload
    // is POST-projection data (projection is folded into loadPointsInternal
    // and consumes only key fields + node-static context), so a hit skips
    // the WASM projection too.
    return loadSliceWithCache(this.facadeCtx, viewState, (queryId, startTime) =>
      this.loadPointsInternal(viewState, session, queryId, startTime)
    );
  }

  private async loadPointsInternal(
    viewState: ViewState,
    session: UpdateSession | undefined,
    queryId: string,
    startTime: number
  ): Promise<LoadedPointsData> {
    await this._onceInit.ensure(() => this.initialize());

    // Check if loader is properly initialized (chunk index OR fallback with total points count)
    if (!this.chunkIndex && this.totalPointsNoIndex === 0) {
      // Zero points is still valid - it just means we have no points to load
      log.warning(
        Modules.SPATIAL_INDEX_LOADER,
        `Loader initialized with no chunk index and 0 points for ${this.node.path}`
      );
    }

    if (!this.arrays.positions) {
      throw new Error('[PointsLoader] Loader not properly initialized: positions array not loaded');
    }

    // Query spatial index for visible ranges (async to allow worker
    // offload).
    let ranges: PointRange[];
    if (session) {
      const querySession = session.begin('Spatial Query');
      try {
        ranges = await this.queryVisiblePointRanges(viewState);
      } finally {
        querySession.end();
      }
    } else {
      ranges = await this.queryVisiblePointRanges(viewState);
    }

    // Emit query event
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);
    this.totalQueryCells += ranges.length;
    this.metrics.queries++;
    this.metrics.visibleElements = totalPoints; // Track current visible points (non-cumulative)

    this.emitEvent({
      type: 'query',
      loader: 'point-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        ranges,
        cells: ranges.length,
        elements: totalPoints,
        queryPosition: viewState.slicePosition,
        queryTolerance: viewState.tolerance,
      },
    });

    // Track active query
    this.activeQueries.set(queryId, {
      id: queryId,
      loader: 'point-spatial-index',
      path: this.node.path,
      startTime,
      status: 'loading',
      cells: ranges.length,
      elements: totalPoints,
      ranges,
    });

    if (ranges.length === 0) {
      // No visible points — return empty dataset; the wrapper caches it
      // (an empty slice is a valid, ~0-byte result that revisits should
      // skip).
      return this.createEmptyPointsData(viewState);
    }

    // Load all arrays with the SAME ranges (critical for alignment!)
    // All five attribute arrays load CONCURRENTLY: distinct zarr arrays,
    // distinct freshly-allocated output buffers. The unified request queue
    // this used to wait for exists now — every chunk fetch funnels through
    // the global data fetch lane (utils/fetch-concurrency.ts, 24 slots, HTTP/2) —
    // so parallel attributes overlap network + decode latency instead of
    // saturating the connection pool.
    type ArrayType = Float32Array | Uint8Array | Uint16Array | Float16Array;
    let positions: ArrayType;
    let colors: ArrayType | null;
    let radii: ArrayType | null;
    let sharpness: ArrayType | null;
    // optional per-point scalars for colormap lookup.
    let scalars: ArrayType | null;

    const loadSession = session?.begin('Load Arrays');
    try {
      if (!this._initialLoadDone) {
        log.load(
          Modules.SPATIAL_INDEX_LOADER,
          `Loading attributes concurrently for ${ranges.length} ranges`
        );
      }

      const nullResult = Promise.resolve(null);
      let positionsResult: ArrayType | null;
      [positionsResult, colors, radii, sharpness, scalars] = await Promise.all([
        // positions (required)
        this.loadRanges('positions', ranges),
        // optional attributes; scalars zarr opened in initialize() when has_scalars=true.
        this.arrays.colors ? this.loadColorRanges(ranges) : nullResult,
        this.arrays.radii ? this.loadRanges('radii', ranges) : nullResult,
        this.arrays.sharpness ? this.loadRanges('sharpness', ranges) : nullResult,
        this.arrays.scalars ? this.loadRanges('scalars', ranges) : nullResult,
      ]);

      if (!positionsResult) {
        throw new Error('[PointsLoader] Failed to load positions array');
      }
      positions = positionsResult;
    } finally {
      loadSession?.end();
    }

    // Prepare accumulator buffers if enabled.
    let targetBuffers: ProjectionTargetBuffers | null = null;

    if (
      !this._accumulator &&
      appConfig.dataLoading.performance.useAccumulators &&
      this._accumulatorConfig
    ) {
      const { capacity, ndim, totalPoints } = this._accumulatorConfig;
      this._accumulator = new LoadedPointsDataAccumulator(capacity, ndim, totalPoints);
    }

    if (this._accumulator && appConfig.dataLoading.performance.useAccumulators) {
      const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

      // Declare the color layout BEFORE any color fill so the buffers
      // are strided for RGB(A) — a no-op when unchanged.
      this._accumulator.configureColorComponents(this.colorComponents);

      // Ensure capacity and initialize types
      this._accumulator.ensureCapacity(totalPoints);

      // Initialize types if not already done
      if (!this._accumulator.hasTypes()) {
        this._accumulator.fill(0, {
          positions: new Float32Array(3),
          colors: colors
            ? (colors.subarray(0, Math.min(this.colorComponents, colors.length)) as ColorArray)
            : undefined,
          radii: radii ? (radii.subarray(0, Math.min(1, radii.length)) as ScalarArray) : undefined,
          sharpness: sharpness
            ? (sharpness.subarray(0, Math.min(1, sharpness.length)) as PointScalarArray)
            : undefined,
          // type-detect scalar buffer on first fill (Float32 / Float16 / Uint8 / Uint16).
          scalars: scalars
            ? (scalars.subarray(0, Math.min(1, scalars.length)) as PointScalarArray)
            : undefined,
        });
      }

      // Get references to accumulator buffers (ZERO allocations!)
      targetBuffers = {
        positions3D: this._accumulator.getPositionBuffer(),
        colors: this._accumulator.getColorBuffer() as ColorArray,
        radii: this._accumulator.getRadiiBuffer() as ScalarArray,
        // sharpness/scalars keep their native dtype (uint8/uint16/float32) —
        // ProjectionTargetBuffers widens to include Uint16Array, so no cast.
        sharpness: this._accumulator.getSharpnessBuffer(),
        // scalar target — only populated when the source has scalars
        // (the projection helper checks both `scalars` and
        // `targetBuffers.scalars` before compacting).
        scalars: scalars ? this._accumulator.getScalarBuffer() : undefined,
      };

      // Copy source colors/sharpness to accumulator buffers (needed for filtering later)
      if (colors) {
        if (colors instanceof Uint8Array && targetBuffers.colors instanceof Uint8Array) {
          (targetBuffers.colors as Uint8Array).set(colors);
        } else if (colors instanceof Uint16Array && targetBuffers.colors instanceof Uint16Array) {
          (targetBuffers.colors as Uint16Array).set(colors);
        } else if (colors instanceof Float32Array && targetBuffers.colors instanceof Float32Array) {
          (targetBuffers.colors as Float32Array).set(colors as Float32Array);
        }
      }

      if (sharpness) {
        // Uint8/Uint16 are kept dtype-preserving (the accumulator normalizes
        // at the GPU upload site — ÷255 / ÷65535). A Float16/Float32 source
        // pins the buffer to Float32, and `.set()` widens value-preserving.
        // The strict Float32→Float32-only branch used to silently zero a
        // Float16/Uint16 source (issue #751); mirrors the colors copy above.
        if (sharpness instanceof Uint8Array && targetBuffers.sharpness instanceof Uint8Array) {
          (targetBuffers.sharpness as Uint8Array).set(sharpness);
        } else if (
          sharpness instanceof Uint16Array &&
          targetBuffers.sharpness instanceof Uint16Array
        ) {
          (targetBuffers.sharpness as Uint16Array).set(sharpness);
        } else if (targetBuffers.sharpness instanceof Float32Array) {
          (targetBuffers.sharpness as Float32Array).set(sharpness);
        }
      }

      // copy source scalars into accumulator buffer so projection's
      // filter pass has them available for in-place compaction. Uint8/Uint16
      // stay dtype-preserving (normalized on upload); Float16/Float32 widen
      // value-preserving via the Float32 buffer's `.set()`.
      if (scalars && targetBuffers.scalars) {
        if (scalars instanceof Uint8Array && targetBuffers.scalars instanceof Uint8Array) {
          (targetBuffers.scalars as Uint8Array).set(scalars);
        } else if (scalars instanceof Uint16Array && targetBuffers.scalars instanceof Uint16Array) {
          (targetBuffers.scalars as Uint16Array).set(scalars);
        } else if (targetBuffers.scalars instanceof Float32Array) {
          (targetBuffers.scalars as Float32Array).set(scalars);
        }
      }
    }

    // Project to 3D display space on the main thread, WASM-accelerated.
    // Points projection is memory-bandwidth-bound and pairs with the
    // zero-allocation accumulator (targetBuffers), so it stays on the
    // main thread rather than a worker — offloading would pay transfer
    // cost both ways for negligible compute savings. The WASM kernels
    // (extract_3d_positions / calculate_effective_radii) run via the
    // backend resolved here; `getPointsBackend` routes ndim > 16 to the
    // uncapped TS reference.
    const ndimForBackend =
      totalPoints > 0
        ? Math.round(positions.length / totalPoints)
        : this.chunkIndex?.metadata.ndim || 3;
    const wasm = await getPointsBackend(ndimForBackend);

    let result: LoadedPointsData;
    if (session) {
      const projectSession = session.begin('Project to 3D');
      try {
        result = this.projectTo3D(
          wasm,
          positions,
          colors,
          radii,
          sharpness,
          viewState,
          ranges,
          targetBuffers,
          scalars
        );
      } finally {
        projectSession.end();
      }
    } else {
      result = this.projectTo3D(
        wasm,
        positions,
        colors,
        radii,
        sharpness,
        viewState,
        ranges,
        targetBuffers,
        scalars
      );
    }

    return result;
  }

  /**
   * Update view for a new slice position.
   *
   * The spatial index makes each reload proportional to the visible ranges,
   * so re-querying those ranges is the canonical update path for now. This
   * keeps buffer ownership and range-cache behavior deterministic.
   *
   * @param viewState - Current view state
   * @param session - Optional profiler session for nested timing
   */
  async updateView(
    viewState: ViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedPointsData> {
    return runWithActiveSignal(
      (s) => (this._activeSignal = s),
      signal,
      async () => {
        const result = await this.loadPoints(viewState, session);
        if (!this._initialLoadDone) {
          this._initialLoadDone = true;
          this.rangeLoader.setVerbose(false);
        }
        return result;
      }
    );
  }

  /**
   * Like {@link updateView} but also reports whether the load was served
   * entirely from cache (see `runWithResidencyProbe`).
   */
  async updateViewWithResidency(
    viewState: ViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<{ data: LoadedPointsData; allResident: boolean }> {
    return runWithResidencyProbe(
      (p) => (this._activeProbe = p),
      () => this.updateView(viewState, session, signal)
    );
  }

  /**
   * Warm the cache for the given viewState without producing geometry.
   *
   * Mirrors `GSplatsSpatialIndexLoader.prefetchChunks` so the dimension
   * animation controller can fire `prefetchChunks(nextSlice)` while the
   * current frame renders, hiding network latency. The returned typed
   * arrays are discarded — only the L0 + L1 + L2 caches and the prefetch
   * queue's `seen`/parsed-cache get populated as a side-effect.
   */
  async prefetchChunks(viewState: ViewState): Promise<void> {
    await this._onceInit.ensure(() => this.initialize());

    if (!this.arrays.positions) return;

    let ranges: PointRange[];
    try {
      ranges = await this.queryVisiblePointRanges(viewState);
    } catch {
      // Spatial query failed (e.g. malformed viewState during animation
      // edge case). Skip the prefetch silently — production loadPoints
      // will surface the error on the next demand frame.
      return;
    }
    if (ranges.length === 0) return;

    const arrays = [
      this.arrays.positions,
      this.arrays.colors,
      this.arrays.radii,
      this.arrays.sharpness,
    ].filter((a): a is zarr.Array<zarr.DataType, zarr.Readable> => a != null);

    await prefetchRangesIntoCache(arrays, ranges);
  }

  async prefetchChunkBoundary(current: ViewState, predicted: ViewState): Promise<void> {
    await this._onceInit.ensure(() => this.initialize());
    if (!this.chunkIndex || !this.arrays.positions) {
      await this.prefetchChunks(predicted);
      return;
    }
    const ranges = await this.queryVisiblePointRanges(current);
    const arrays = [
      this.arrays.positions,
      this.arrays.colors,
      this.arrays.radii,
      this.arrays.sharpness,
    ].filter((array): array is zarr.Array<zarr.DataType, zarr.Readable> => array != null);
    const views = planChunkBoundaryViewStates(current, predicted, ranges, this.chunkIndex, arrays);
    await Promise.all(views.map((view) => this.prefetchChunks(view)));
  }

  /**
   * Query spatial index for visible point ranges.
   *
   * Tolerance is computed by the points-specific
   * `calculateSpatialQueryTolerance` (`EffectiveRadiusConfig`-aware) and passed
   * to the canonical `SpatialQueryBuilder` via its `tolerance` option. Falls
   * back to a uniform tolerance derived from `viewState.tolerance`/maxRadius
   * for nodes without an `EffectiveRadiusConfig`.
   *
   * The builder also handles the `extend_to_all` short-circuit; we only emit
   * the defensive metadata-missing warning ourselves.
   */
  private async queryVisiblePointRanges(viewState: ViewState): Promise<PointRange[]> {
    // This node's nd_transform gives the current world slice no local preimage
    // on a discrete dimension, so nothing here belongs to it. Checked ahead of
    // both the no-index load-all fallback and the extend_to_all short-circuit,
    // either of which would otherwise hand every element to the half-step
    // membership gate and let a neighbouring category through. See
    // `ViewState.noPreimage` / `invertNdTransformForQuery`.
    if (viewState.noPreimage) {
      log.query(Modules.SPATIAL_INDEX_LOADER, 'No preimage for this slice under nd_transform');
      return [];
    }

    const extendDims = this.node.attrs.extend_to_all || [];

    warnExtendToAllNoDimensions({
      extendDims,
      hasResolvedDimensions: !!viewState.dimensions && viewState.dimensions.length > 0,
      nodePath: this.node.path,
      logModule: Modules.SPATIAL_INDEX_LOADER,
    });

    if (!this.chunkIndex) {
      // No chunk index - load all points (fallback for 3D datasets without Morton/Hilbert ordering)
      const totalPoints: number = (this.node.attrs.n_points ||
        this.totalPointsNoIndex ||
        0) as number;
      log.query(Modules.SPATIAL_INDEX_LOADER, `No index: loading all ${totalPoints} points`);
      return [{ start: 0, end: totalPoints }];
    }

    if (!this._initialLoadDone) {
      announceExtendToAllOnce({
        extendDims,
        nodePath: this.node.path,
        logModule: Modules.SPATIAL_INDEX_LOADER,
      });
    }

    const { slicePosition } = viewState;
    const maxRadius = this.node.attrs.max_radius ?? appConfig.dataLoading.spatial.defaultMaxRadius;
    const fullDim = this.chunkIndex.metadata.ndim;

    // Build query tolerance. The EffectiveRadiusConfig-aware path knows
    // about discrete dims, the `>= 1e9` extend-to-all sentinel, and the
    // rendering-side `calculateEffectiveRadii` behavior. Without that
    // config we use a uniform fallback: infinite for displayed dims and
    // explicit/maxRadius tolerance otherwise.
    let queryTolerance: number[];
    if (this._effectiveRadiusConfig) {
      queryTolerance = calculateSpatialQueryTolerance(
        viewState,
        this._effectiveRadiusConfig,
        fullDim
      );
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Query tolerance (with discrete awareness): [${queryTolerance.map(formatTolerance).join(', ')}]`
      );
    } else {
      // No effective-radius config: uniform fallback. Discrete non-spatial dims
      // use the shared quarter-cell reach (NOT the ride-along tolerance the
      // SliceCache key drops) so a cache hit can't serve a different-reach
      // decode. See fallbackQueryTolerance.
      queryTolerance = fallbackQueryTolerance(viewState, fullDim, maxRadius);
    }

    log.query(Modules.SPATIAL_INDEX_LOADER, 'Querying spatial index:');
    log.info(Modules.SPATIAL_INDEX_LOADER, `  Full dimensions: ${fullDim}`);
    log.info(
      Modules.SPATIAL_INDEX_LOADER,
      `  Query position: [${slicePosition
        .slice(0, fullDim)
        .map((p) => (p ?? 0).toFixed(2))
        .join(', ')}]`
    );

    // ViewState and BaseViewState share the same `dimensions:
    // DimensionMetadata[]` shape, so the assignment is a structural
    // narrowing — only displayDims, slicePosition, tolerance, and
    // dimensions reach the builder.
    const baseViewState: BaseViewState = {
      displayDims: viewState.displayDims,
      slicePosition: viewState.slicePosition,
      tolerance: viewState.tolerance,
      dimensions: viewState.dimensions,
    };

    return new SpatialQueryBuilder(
      {
        chunkBounds: this.chunkIndex.chunkBounds,
        chunkCount: this.chunkIndex.chunkCount,
        metadata: {
          ndim: fullDim,
          chunk_size: this.chunkIndex.metadata.chunk_size,
        },
      },
      baseViewState,
      {
        tolerance: queryTolerance,
        totalElements: this.chunkIndex.metadata.total_points,
        chunkSize: this.chunkIndex.metadata.chunk_size,
        extendDims,
        logModule: Modules.SPATIAL_INDEX_LOADER,
      }
    ).execute();
  }

  /**
   * Load data for specific point ranges
   */
  /**
   * Allocate output buffer for decoded data.
   * For encoded arrays, always use Float32Array (decoder output).
   * For direct arrays, match the source dtype.
   */
  private allocateOutputBuffer(
    totalElements: number,
    isEncoded: boolean,
    dtype: string
  ): Float32Array | Uint8Array | Uint16Array | Float16Array {
    // CRITICAL: For encoded arrays, ALWAYS use Float32Array because the decoder
    // always returns Float32Array (it dequantizes uint8/uint16 to float32).
    if (isEncoded) {
      return new Float32Array(totalElements);
    }

    // For direct arrays, preserve native type
    // Handle all numpy dtype string variants (byte-order prefixes: < little, > big, | native)
    if (dtype === 'uint8' || dtype === '|u1' || dtype === '<u1' || dtype === '>u1') {
      return new Uint8Array(totalElements);
    }
    if (dtype === 'uint16' || dtype === '|u2' || dtype === '<u2' || dtype === '>u2') {
      return new Uint16Array(totalElements);
    }
    if (dtype === 'float16' || dtype === '|f2' || dtype === '<f2' || dtype === '>f2') {
      // Float16 with fallback
      if (typeof Float16Array !== 'undefined') {
        return new Float16Array(totalElements);
      }
      log.warning(Modules.SPATIAL_INDEX_LOADER, 'Float16Array not supported, using Float32Array');
    }
    return new Float32Array(totalElements);
  }

  /**
   * Load an array_ref attribute. Delegates to the shared
   * RangeLoader.loadRangesResolvingRef helper, which opens the target,
   * recomputes per-item element count from the target's shape, and runs
   * RangeLoader.loadRanges against it.
   *
   * `actualElementsPerPoint` is passed as the non-ref hint; when an
   * array_ref is in play it's ignored in favour of the target's shape.
   */
  private async loadArrayRefRanges(
    attrs: ArrayMetadata,
    ranges: LoadRange[],
    output: Float32Array,
    totalPoints: number,
    actualElementsPerPoint: number
  ): Promise<number> {
    // The synthetic array_ref attrs has no usable .arrays handle of its own —
    // pass the same Float32Array-shaped placeholder the original code did
    // and let the helper switch into ref-resolution.
    const placeholder = {} as zarr.Array<zarr.DataType, zarr.Readable>;
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    return this.rangeLoader.loadRangesResolvingRef(
      placeholder,
      attrs,
      ranges,
      output,
      totalPoints,
      actualElementsPerPoint,
      storeToUse,
      'Points'
    );
  }

  /**
   * Load color ranges with multi-type support (preserves original_dtype).
   *
   * Delegates to the shared color-attribute helper used by the lines and
   * gsplats facades. Handles direct (uint8/uint16/float32 native), encoded
   * (quantized / LUT / broadcasted decoded to Float32 then cast back to
   * `original_dtype`), and array_ref dispatch in one call.
   *
   * Kept separate from {@link loadRanges} (used for positions / radii /
   * sharpness) because:
   *   - Colors hard-code RGB (3 channels per item) — the shared helper
   *     does too.
   *   - Positions / scalars need Float16 preservation that the shared
   *     helper's Float32-only direct path would silently widen.
   */
  private async loadColorRanges(
    ranges: PointRange[]
  ): Promise<Float32Array | Uint8Array | Uint16Array | null> {
    const array = this.arrays.colors;
    if (!array) return null;
    const storeToUse = this.zarrStore || this.zarrLocation.store;
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    if (!this._initialLoadDone) {
      log.load(Modules.SPATIAL_INDEX_LOADER, `Loading colors for ${ranges.length} ranges`);
    }

    const output = await loadColorRanges(array, ranges, this.rangeLoader, storeToUse, 'Points');
    recordLoadMetrics(this.facadeCtx, 'colors', totalPoints, output);
    return output;
  }

  /**
   * Load array data for specified point ranges.
   *
   * This method handles multiple encoding strategies:
   * - Broadcasted: Load once, replicate
   * - Quantized: Load ranges, dequantize
   * - LUT: Load indices, decode with lookup table
   * - Array Reference: Resolve target, dispatch based on target encoding
   * - Generic Encoded: Decode full array, extract ranges
   * - Direct: Load ranges directly
   *
   * @param arrayName - Name of the array to load (positions, radii, sharpness).
   *   For colors, use {@link loadColorRanges} which delegates to the shared
   *   color-attribute helper.
   * @param ranges - Point ranges to load
   * @returns Typed array with loaded data, or null if array doesn't exist
   */
  private async loadRanges(
    arrayName: string,
    ranges: PointRange[]
  ): Promise<Float32Array | Uint8Array | Uint16Array | Float16Array | null> {
    const array = this.arrays[arrayName as keyof typeof this.arrays];
    if (!array) return null;

    if (!this._initialLoadDone) {
      log.load(Modules.SPATIAL_INDEX_LOADER, `Loading ${arrayName} for ${ranges.length} ranges`);
    }

    // Analyze array metadata
    const attrs = array.attrs as unknown as ArrayMetadata;
    const isEncoded = ArrayDecoder.isEncoded(attrs);
    const shape = array.shape;
    const elementsPerPoint = shape.length === 2 ? shape[1] : 1;
    const totalPoints = ranges.reduce((sum, r) => sum + (r.end - r.start), 0);

    // Determine actual elements per point (may differ for encoded arrays)
    const actualElementsPerPoint =
      isEncoded && attrs.encoding?.original_shape?.[1]
        ? attrs.encoding.original_shape[1]
        : elementsPerPoint;

    const totalElements = totalPoints * actualElementsPerPoint;

    // Sanity check
    if (totalElements > 1000000000 || totalElements < 0) {
      throw new Error(
        `Invalid array size: ${totalElements} (points=${totalPoints}, elements/point=${actualElementsPerPoint})`
      );
    }

    // Handle array_ref specially (needs zarrStore access to resolve target)
    const isArrayRef = ArrayDecoder.isArrayRef(attrs);
    if (isArrayRef) {
      const decodedFloat32 = new Float32Array(totalElements);
      await this.loadArrayRefRanges(
        attrs,
        ranges as LoadRange[],
        decodedFloat32,
        totalPoints,
        actualElementsPerPoint
      );

      // array_ref also has original_dtype - restore it!
      // Python str(np.dtype('uint8')) returns 'uint8', but handle all variants for safety
      const originalDtype = attrs.encoding?.original_dtype;
      let output: Float32Array | Uint8Array | Uint16Array | Float16Array = decodedFloat32;

      if (
        originalDtype === 'uint8' ||
        originalDtype === '|u1' ||
        originalDtype === '<u1' ||
        originalDtype === '>u1'
      ) {
        const uint8Output = new Uint8Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
          uint8Output[i] = Math.round(clamp(decodedFloat32[i], 0, 255));
        }
        output = uint8Output;
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Restored original_dtype=uint8 for array_ref ${arrayName} (${totalElements} elements)`
        );
      } else if (
        originalDtype === 'uint16' ||
        originalDtype === '|u2' ||
        originalDtype === '<u2' ||
        originalDtype === '>u2'
      ) {
        const uint16Output = new Uint16Array(totalElements);
        for (let i = 0; i < totalElements; i++) {
          uint16Output[i] = Math.round(clamp(decodedFloat32[i], 0, 65535));
        }
        output = uint16Output;
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          `Restored original_dtype=uint16 for array_ref ${arrayName} (${totalElements} elements)`
        );
      }

      recordLoadMetrics(this.facadeCtx, arrayName, totalPoints, output);
      return output;
    }

    // Detect encoding type to decide loading strategy
    const encoding = RangeLoader.detectEncoding(attrs);

    // CRITICAL: For direct (unencoded) arrays, preserve native type!
    // Rendering pipeline depends on actual array type:
    // - Uint8Array colors: THREE.js normalizes (0-255 → 0-1) with normalized=true
    // - Float32Array colors: Expected to be 0-1, no normalization
    // Converting Uint8Array(255) to Float32Array(255.0) would make colors ~255x too bright!
    if (encoding === 'direct') {
      const output = this.allocateOutputBuffer(totalElements, false, array.dtype);
      await this.rangeLoader.loadDirectTyped(array, ranges as LoadRange[], output);
      recordLoadMetrics(this.facadeCtx, arrayName, totalPoints, output);
      return output;
    }

    // Use RangeLoader for encoded arrays (broadcasted, quantized, lut)
    // Decode to Float32Array first (decoding math produces floats)
    const decodedFloat32 = new Float32Array(totalElements);
    await this.rangeLoader.loadRanges(
      array,
      attrs,
      ranges as LoadRange[],
      decodedFloat32,
      totalPoints,
      actualElementsPerPoint
    );

    // CRITICAL: Restore original dtype for correct rendering!
    // Python encoder stores original_dtype, decoder must restore it:
    // - uint8 colors: THREE.js normalizes (0-255 → 0-1) with normalized=true
    // - float32 colors: Expected to be 0-1, no normalization
    // Python str(np.dtype('uint8')) returns 'uint8', but handle all variants for safety
    const originalDtype = attrs.encoding?.original_dtype;
    let output: Float32Array | Uint8Array | Uint16Array | Float16Array = decodedFloat32;

    if (
      originalDtype === 'uint8' ||
      originalDtype === '|u1' ||
      originalDtype === '<u1' ||
      originalDtype === '>u1'
    ) {
      // Convert float32 → uint8 (values should already be in 0-255 range)
      const uint8Output = new Uint8Array(totalElements);
      for (let i = 0; i < totalElements; i++) {
        uint8Output[i] = Math.round(clamp(decodedFloat32[i], 0, 255));
      }
      output = uint8Output;
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Restored original_dtype=uint8 for ${arrayName} (${totalElements} elements)`
      );
    } else if (
      originalDtype === 'uint16' ||
      originalDtype === '|u2' ||
      originalDtype === '<u2' ||
      originalDtype === '>u2'
    ) {
      // Convert float32 → uint16
      const uint16Output = new Uint16Array(totalElements);
      for (let i = 0; i < totalElements; i++) {
        uint16Output[i] = Math.round(clamp(decodedFloat32[i], 0, 65535));
      }
      output = uint16Output;
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Restored original_dtype=uint16 for ${arrayName} (${totalElements} elements)`
      );
    }
    // For float32/float64 or unspecified, keep as Float32Array

    // Update metrics and emit load event
    recordLoadMetrics(this.facadeCtx, arrayName, totalPoints, output);
    return output;
  }

  /**
   * Build a `ProjectionContext` from instance fields. Done per-call so
   * the helpers in `point-loader/projection.ts` can stay pure with
   * respect to the loader.
   */
  private buildProjectionContext(): ProjectionContext {
    return {
      chunkIndex: this.chunkIndex,
      effectiveRadiusConfig: this._effectiveRadiusConfig,
      accumulator: this._accumulator,
      nodeAttrs: this.attrs,
    };
  }

  /**
   * Main-thread, WASM-accelerated nD → 3D projection. Thin delegate to
   * `projectPointsTo3D` in `point-loader/projection.ts`. The `wasm`
   * backend (compiled or TS-reference fallback) is resolved by the caller
   * via `getPointsBackend(ndim)`.
   */
  private projectTo3D(
    wasm: WasmModule,
    positions: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    colors: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    radii: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    sharpness: Float32Array | Uint8Array | Uint16Array | Float16Array | null,
    viewState: ViewState,
    ranges: PointRange[],
    targetBuffers?: ProjectionTargetBuffers | null,
    scalars?: Float32Array | Uint8Array | Uint16Array | Float16Array | null
  ): LoadedPointsData {
    return projectPointsTo3D(
      wasm,
      positions,
      colors,
      radii,
      sharpness,
      viewState,
      ranges,
      this.buildProjectionContext(),
      targetBuffers,
      scalars ?? null,
      this.colorComponents
    );
  }

  /**
   * Empty-points payload for the "no visible points" branches. Thin
   * delegate to `createEmptyPointsData` in `point-loader/projection.ts`.
   */
  private createEmptyPointsData(viewState: ViewState): LoadedPointsData {
    return createEmptyPointsDataHelper(this.buildProjectionContext(), viewState);
  }

  // LoaderMonitor implementation

  /**
   * Add event listener
   */
  addEventListener(listener: MonitorEventListener): void {
    this.events.add(listener);
  }

  /**
   * Remove event listener
   */
  removeEventListener(listener: MonitorEventListener): void {
    this.events.remove(listener);
  }

  /**
   * Get current metrics
   */
  getMetrics(): LoaderMetrics {
    // Chunk-index telemetry for the monitor advisor (shared across the
    // three facades; see buildSpatialIndexMetrics).
    if (this.chunkIndex) {
      this.metrics.spatialIndex = buildSpatialIndexMetrics(
        this.chunkIndex.metadata.total_chunks,
        this.metrics.queries,
        this.totalQueryCells,
        this.metrics.elementsLoaded
      );
    }

    return { ...this.metrics };
  }

  /**
   * Get active queries
   */
  getActiveQueries(): QueryInfo[] {
    return Array.from(this.activeQueries.values());
  }

  /**
   * Emit event to listeners. Thin wrapper around `LoaderEventEmitter.emit`
   * so the three call sites in this file keep their `this.emitEvent(...)`
   * shape; per-listener error isolation lives in the emitter.
   */
  private emitEvent(event: MonitorEvent): void {
    this.events.emit(event);
  }

  /**
   * Load spatial_extend_dims from scene_dimensions stored in root zarr attributes.
   *
   * This derives the spatial extension flags from the scene_dimensions which is
   * the single source of truth for dimension configuration. Each dimension has
   * a `spatial` flag indicating whether points physically extend through it.
   *
   * For displayed dimensions, we always treat them as spatial (they're in the view plane).
   * For non-displayed dimensions:
   *   - spatial=true: Points extend as hyperspheres (use effective radius)
   *   - spatial=false: Categorical/discrete dimension (exact match required)
   *
   * @returns Array of booleans indicating spatial extension per dimension, or null if unavailable
   */
  private async loadSpatialExtendDimsFromSceneDimensions(): Promise<boolean[] | null> {
    if (!this.zarrStore) {
      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        'No zarr store available - cannot derive spatial_extend_dims'
      );
      return null;
    }

    try {
      // Open the root group to get scene_dimensions from root attributes
      const rootGroup = await zarr.open(this.zarrStore, { kind: 'group' });
      const rootAttrs = rootGroup.attrs as unknown as ZarrSceneAttrs;

      if (!rootAttrs?.scene_dimensions?.dimensions) {
        log.info(
          Modules.SPATIAL_INDEX_LOADER,
          'No scene_dimensions in root attributes - spatial filtering disabled'
        );
        return null;
      }

      const dimensions = rootAttrs.scene_dimensions.dimensions;

      // Derive spatial_extend_dims: for each dimension, check its spatial flag
      // Default behavior: displayed dimensions are always spatial (they're the viewing plane)
      // Non-displayed dimensions use their explicit spatial flag (defaulting to false if unset)
      const spatialExtendDims = dimensions.map((dim) => {
        if (dim.display) {
          // Displayed dimensions are always treated as spatial (points are in the plane)
          return true;
        }
        // Non-displayed: use explicit spatial flag, default false (categorical)
        return dim.spatial ?? false;
      });

      log.info(
        Modules.SPATIAL_INDEX_LOADER,
        `Derived spatial_extend_dims from scene_dimensions: [${spatialExtendDims.join(', ')}]`
      );

      return spatialExtendDims;
    } catch (error) {
      log.warning(
        Modules.SPATIAL_INDEX_LOADER,
        'Failed to load scene_dimensions from root:',
        error
      );
      return null;
    }
  }

  /**
   * Get accumulator stats for memory monitoring
   */
  getAccumulatorStats(): AccumulatorStats | null {
    return this._accumulator?.getStats() ?? null;
  }

  /**
   * Release pooled decoded buffers after a progressive parent has copied them.
   * Earlier payloads remain valid because `dispose()` replaces the accumulator
   * buffers rather than clearing their old arrays. This deliberately gives up
   * pooling across view changes to release the progressive ladder's second copy.
   */
  releaseAccumulator(): void {
    if (!this._accumulatorConfig) return;
    this._accumulator?.dispose();
    this._accumulator = null;
    this.metrics.memoryUsed = 0;
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
    this._accumulatorConfig = null;
  }
}
