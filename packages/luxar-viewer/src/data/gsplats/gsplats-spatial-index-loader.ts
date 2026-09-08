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
import type { SceneNode } from '../data-loader-types';
import { compactGSplatLabelIds, type GSplatLabelChannel } from './label-channel';
import { ArrayRefRegistry, type ArrayMetadata } from '../array-decoder/decoder';
import {
  RangeLoader,
  SpatialQueryBuilder,
  type ChunkSpatialIndex,
  type LoadRange,
  getExpectedColorType,
  loadColorRanges,
  colorComponentsOf,
  prefetchRangesIntoCache,
  makeInitialLoaderMetrics,
  buildSpatialIndexMetrics,
  loadSliceWithCache,
  recordLoadMetrics,
  runWithActiveSignal,
  runWithResidencyProbe,
  type SpatialFacadeCtx,
  type ToleranceOptions,
  LoaderEventEmitter,
  OnceInit,
  warnExtendToAllNoDimensions,
  announceExtendToAllOnce,
} from '../loaders';
import { loadGSplatsChunkIndex, registerGSplatsArrayBounds } from './chunk-index-loader';
import { createEmptyGSplatsData } from './projection';
// `rendering` sits BELOW `data` in the package layering (see
// `.dependency-cruiser.cjs`), so this is a downward import, not a cross-layer
// escape. It buys the SINGLE truncation-radius sanitizer the material path uses —
// see `resolveTruncationRadius` below for why that identity matters. The module
// pulls in only `utils/log`, `config/constants` and a dependency-free `erf`, so
// nothing heavy rides along.
import { clampTruncationRadius } from '../../rendering/materials/gsplat/math';
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
import type { UpdateSession } from '../../profiling/update-profiler';
import {
  DecompressedChunkCache,
  type DecompressedChunkCacheStats,
} from '../../cache/decompressed-chunk-cache';
import { wrapWithCache } from '../../cache/decompressed-chunk-cache/cached-zarr-array';
import { ResidencyAccumulator } from '../../cache/residency-probe';
import { ChunkPrefetcher } from '../../cache/chunk-prefetcher';
import type { SliceCache } from '../../cache/slice-cache';

/**
 * One-shot latch for the ignored-`slice_dims` warning.
 *
 * The warning is a property of the STORE, not of the loader instance, and one store
 * mints many loaders: a progressive ladder builds one per `additive_<i>` sub-LOD and
 * a `kind=partition` one per part, all carrying the same (identical, identically
 * unusable) attr. Per-loader logging turned a single corrupt store into hundreds of
 * identical lines. Latched per PROCESS instead, mirroring `truncationClampWarned` in
 * `rendering/materials/gsplat/math.ts`; the first offender is named in full, and the
 * fallback itself is unconditional, so nothing but the logging is suppressed.
 */
let sliceDimsWarned = false;
const PREFETCH_CHUNK_METADATA_BYTES = 64;

/** Test-only: re-arm the one-shot above so cases stay order-independent. */
export function resetSliceDimsWarningForTests(): void {
  sliceDimsWarned = false;
}

/**
 * Read the writer's published BARRIER set off the node attrs, for
 * `ToleranceOptions.barrierDims`.
 *
 * `slice_dims` is exactly the set where `compute_chunk_bounds_gsplats` withheld
 * the `truncation_radius · σ` expansion and used the tight `_BARRIER_BOUND_EPS`
 * plus any encoder coordinate round-trip slack instead
 * (`luxar/io/_compiler/gsplat_assembly.py` stamps it whenever `ordering != 'none'`),
 * in center-column indices — the same index space as `attrs.ndim`,
 * `viewState.slicePosition` and the `chunk_bounds` columns. Handing it to the
 * tolerance computer is what stops the reader re-deriving barrier-ness from the
 * scene's `discrete` flags and disagreeing with the writer (issue #1655 item 2).
 *
 * An EMPTY array is meaningful and is returned as such: the writer ordered purely
 * spatially, so nothing is a barrier. `undefined` means "no published set" and lets
 * the tolerance computer fall back to `DimensionInfo.discrete`.
 *
 * VALIDATION IS ALL-OR-NOTHING, on purpose. These attrs come off disk and are
 * untrusted. Filtering a bad array element-wise would silently DROP a genuine
 * barrier dim, and the reader would then apply the ~1e-3 continuous epsilon to an
 * axis whose bounds are barrier-tight — i.e. narrow the fetch window below what the
 * data needs, which is the exact failure this plumbing exists to prevent. Rejecting
 * the whole attr instead falls back to the documented legacy rule — which IS the
 * pre-#1655 behaviour, byte for byte, so no store gets a narrower window than it had
 * before the attr was read (it is not a claim that the fallback is as wide as
 * honouring a well-formed set: for a demoted dim the published set now buys a half
 * cell where `discrete` alone gives a quarter). Anything not a plain array, or any
 * entry that is not an integer in `[0, ndim)`, rejects the lot.
 *
 * Both attrs are re-widened to `unknown` before inspection even though
 * `GSplatsMetadata` declares their shapes: that interface describes what a WELL-FORMED
 * store carries, and validating against a declaration the data may not honour would
 * be circular.
 */
function readBarrierDims(attrs: GSplatsMetadata, nodePath: string): readonly number[] | undefined {
  const raw: unknown = (attrs as unknown as Record<string, unknown>).slice_dims;
  if (raw === undefined || raw === null) return undefined;

  const ndim = attrs.ndim;
  const reject = (why: string): undefined => {
    if (!sliceDimsWarned) {
      sliceDimsWarned = true;
      log.warning(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `${nodePath}: ignoring the slice_dims attr (${why}) — falling back to ` +
          "the scene dimensions' discrete flags for barrier classification. " +
          'Further ignored slice_dims attrs are silent.'
      );
    }
    return undefined;
  };

  if (!Array.isArray(raw)) return reject('not an array');
  // A well-formed attr can still be UNCHECKABLE: `[0, ndim)` is the only bound the
  // entry test has, so without a usable column count there is nothing to validate
  // against. Worded as its own reason rather than folded into "malformed", because
  // the codebase does contemplate a node without `ndim` (see `initialize`, which
  // falls back to the centers array's column count for accumulator sizing).
  if (!Number.isInteger(ndim) || ndim <= 0) {
    return reject(`cannot validate its entries: node ndim is ${String(ndim)}`);
  }
  for (const entry of raw) {
    if (typeof entry !== 'number' || !Number.isInteger(entry) || entry < 0 || entry >= ndim) {
      return reject(`entry ${String(entry)} is not an integer in [0, ${ndim})`);
    }
  }
  return raw as number[];
}

/**
 * The node's Gaussian truncation radius `T`, for `ToleranceOptions.truncationRadius`
 * (issue #1655 item 3).
 *
 * Sanitized through the SAME `clampTruncationRadius` the material path uses
 * (`rendering/node-factory/create-gsplats-node.ts`), and that identity is the point,
 * not just code reuse: the tolerance's degenerate-band term has to cover the band the
 * renderer actually draws, and the renderer draws with the CLAMPED radius. A second,
 * looser rule here would let the two disagree — and an unsanitized attr (`0`, a
 * negative, `NaN`, `1e308`, a JSON string) would either collapse the band or turn the
 * epsilon into "fetch the entire node".
 *
 * `clampTruncationRadius` is typed for a `number` and a zarr attr is `unknown`, so the
 * type guard is here; a non-number attr is treated as absent and `undefined` lets the
 * tolerance computer apply its own default, so there is one spelling of the fallback.
 * The material path reaches the SAME answer for such an attr, but not through this
 * guard: `create-gsplats-node.ts` casts the attr and hands it to
 * `clampTruncationRadius`, whose own non-number branch substitutes
 * `GSPLAT_DEFAULT_TRUNCATION_RADIUS`. That branch is why the identity above holds for
 * a JSON `"6"` too — every numeric test in the clamp coerces a numeric string
 * (`"6" * "6" === 36`), so without it the string reached the `uTruncate` uniform (a
 * 6σ material band) while this side used 2.75, which is precisely the fetch-vs-render
 * gap item 3 exists to close.
 */
function resolveTruncationRadius(attrs: GSplatsMetadata): number | undefined {
  const raw: unknown = (attrs as unknown as Record<string, unknown>).truncation_radius;
  if (typeof raw !== 'number') return undefined;
  return clampTruncationRadius(raw);
}

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
  private _accumulatorConfig: { capacity: number; ndim: number } | null = null;
  private _labelIndicesScratch = new Uint32Array(0);
  /** Color layout of this dataset: 3 (RGB) or 4 (RGBA, alpha = per-splat opacity). */
  private colorComponents: 3 | 4 = 3;

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

  // Per-slice decoded-result cache (S-cache). Wired ONLY for plain-leaf nodes
  // by the plain factory helper — progressive sub-LOD instances stay
  // cache-less (their wrapper owns the whole ladder; see loader-factory.ts).
  private sliceCache: SliceCache | null = null;

  // Suppress detail logs after first successful view update
  private _initialLoadDone = false;

  // Per-node tolerance inputs (the writer's published barrier set + this node's
  // truncation radius). Derived from attrs, which never change for the life of a
  // loader, and memoized because the query path runs on EVERY view update: without
  // it each slice move would re-validate the attr array and mint a fresh options
  // object. (The malformed-attr warning is separately latched per process — see
  // `sliceDimsWarned` — so log volume does not depend on this memo.)
  private _toleranceOptions: ToleranceOptions | null = null;

  // LoaderMonitor surface — same shape as the points and lines facades.
  private readonly events = new LoaderEventEmitter();
  private readonly metrics: LoaderMetrics;
  private readonly activeQueries = new Map<string, QueryInfo>();
  private nextQueryId = 0;
  // Cumulative queried cells across the session (drives avgCellsPerQuery).
  private totalQueryCells = 0;
  // Shared facade-helper context (data/loaders/spatial-facade.ts): stable
  // references + this-bound accessors, built once in the constructor.
  private readonly facadeCtx: SpatialFacadeCtx;

  /**
   * Conservative whole-rung L0 footprint used to bound speculative ladder
   * lookahead before the exact visible chunk set is known. Zarr scalar types
   * top out at 8 bytes; counting every optional channel and whole stored rung
   * deliberately overestimates sliced/broadcast arrays so lookahead cannot
   * evict more useful cache entries on a large scene.
   */
  get prefetchByteUpperBound(): number {
    const attrs = this.node.attrs as unknown as GSplatsMetadata;
    const nSplats = Math.max(0, attrs.n_splats || 0);
    const ndim = Math.max(1, attrs.ndim || 3);
    const scalarCount =
      ndim +
      1 +
      choleskyPackedSize(ndim) +
      (attrs.has_colors ? 4 : 0) +
      (attrs.has_label_ids ? 1 : 0);
    const arrayCount = 4 + (attrs.has_colors ? 1 : 0) + (attrs.has_label_ids ? 1 : 0);
    const chunkSize = Math.max(1, attrs.chunk_size || 1);
    const chunkCount = Math.ceil(nSplats / chunkSize);
    return nSplats * scalarCount * 8 + chunkCount * arrayCount * PREFETCH_CHUNK_METADATA_BYTES;
  }

  /** Live L0 accounting shared by every rung in this progressive ladder. */
  getPrefetchCacheStats(): DecompressedChunkCacheStats | null {
    return this.l0Cache?.getStats() ?? null;
  }

  private arrays: {
    centers?: zarr.Array<zarr.DataType, zarr.Readable>;
    amplitudes?: zarr.Array<zarr.DataType, zarr.Readable>;
    // v3.1 split Cholesky layout (diagonal + off-diagonal)…
    cholesky_factors_diag?: zarr.Array<zarr.DataType, zarr.Readable>;
    cholesky_factors_offdiag?: zarr.Array<zarr.DataType, zarr.Readable>;
    // …or the legacy v3.0 single packed array.
    cholesky_factors?: zarr.Array<zarr.DataType, zarr.Readable>;
    colors?: zarr.Array<zarr.DataType, zarr.Readable>;
    label_ids?: zarr.Array<zarr.DataType, zarr.Readable>;
  } = {};

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
    // Geometry-neutral counters: elementsLoaded / visibleElements count
    // splats for gsplats.
    this.metrics = makeInitialLoaderMetrics('gsplats-spatial-index', node.path);
    this.facadeCtx = {
      metrics: this.metrics,
      activeQueries: this.activeQueries,
      loader: 'gsplats-spatial-index',
      path: node.path,
      sliceCache: this.sliceCache,
      nextQueryId: () => this.nextQueryId++,
      accumulatorMemoryMB: () => this.getAccumulatorStats()?.memoryMB ?? 0,
      emit: (event) => this.emitEvent(event),
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

    // Try to open the optional colors array. Skip the probe entirely when the
    // node declares `has_colors: false` (e.g. single-channel / colormapped
    // gsplats) — otherwise each leaf fires ~3 doomed 404s (colors/.zattrs,
    // .zarray, zarr.json) that spam the console and waste round-trips per
    // refinement. `undefined` (legacy datasets) still probes for compatibility.
    if (attrs.has_colors !== false) {
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
        // Layout (3 = RGB, 4 = RGBA) is a property of the dataset, read once
        // from the array shape. Downstream (accumulator sizing, texel
        // packing, uHasElementAlpha) keys off this.
        this.colorComponents = colorComponentsOf(colorsArray);
      } catch {
        log.info(
          Modules.GSPLATS_SPATIAL_INDEX_LOADER,
          'No colors array found (using default white)'
        );
      }
    }

    if (attrs.has_label_ids) {
      let labelIdsArray = await zarr.open(this.zarrLocation.resolve('label_ids'), {
        kind: 'array',
      });
      this.registerBounds('label_ids', labelIdsArray);
      if (this.l0Cache) {
        labelIdsArray = wrapWithCache(
          labelIdsArray,
          this.l0Cache,
          `${this.node.path}/label_ids`,
          () => this._activeProbe,
          () => this._activeSignal
        );
      }
      this.arrays.label_ids = labelIdsArray;
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

      this._accumulatorConfig = { capacity: initialCapacity, ndim };
      this._accumulator = new GSplatsDataAccumulator(initialCapacity, ndim);
      this._accumulator.configureColorComponents(this.colorComponents);

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

  /** Initialize metadata once without loading any attribute payload chunks. */
  async ensureInitialized(): Promise<void> {
    await this._onceInit.ensure(() => this.initialize());
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
    // Plain-leaf S-cache + query close-out via the shared facade template
    // (see `loadSliceWithCache`). GSplats cache PRE-projection decoded data —
    // projection re-runs on every hit downstream.
    return loadSliceWithCache(this.facadeCtx, viewState, (queryId, startTime) =>
      this.loadGSplatsInternal(viewState, session, queryId, startTime)
    );
  }

  private async loadGSplatsInternal(
    viewState: GSplatsViewState,
    session: UpdateSession | undefined,
    queryId: string,
    startTime: number
  ): Promise<LoadedGSplatsData> {
    await this.ensureInitialized();

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
    this.totalQueryCells += splatRanges.length;
    this.metrics.queries += 1;
    this.metrics.visibleElements = totalSplats;
    this.activeQueries.set(queryId, {
      id: queryId,
      loader: 'gsplats-spatial-index',
      path: this.node.path,
      startTime,
      status: 'loading',
      cells: splatRanges.length,
      elements: totalSplats,
      ranges: splatRanges,
    });
    this.emitEvent({
      type: 'query',
      loader: 'gsplats-spatial-index',
      timestamp: Date.now(),
      data: {
        path: this.node.path,
        ranges: splatRanges,
        cells: splatRanges.length,
        elements: totalSplats,
        queryPosition: viewState.slicePosition,
        queryTolerance: viewState.tolerance,
      },
    });

    if (splatRanges.length === 0) {
      // No visible splats — return empty dataset; the wrapper caches it
      // (an empty slice is a valid, ~0-byte result that revisits should skip).
      log.info(Modules.GSPLATS_SPATIAL_INDEX_LOADER, 'No visible gsplats - returning empty data');
      return createEmptyGSplatsData(attrs);
    }

    if (!this._initialLoadDone) {
      log.load(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        `Loading ${totalSplats} gsplats from ${splatRanges.length} ranges`
      );
    }

    // Publish the visible ranges only for a node that declares a per-element
    // label CSR — they are the loader's half of the slot → on-disk element-ID
    // map picking resolves labels through (issue #1423), and nothing else
    // reads them. Ungated, every gsplats node would drag the array through
    // each SliceCache snapshot for no reader. `LabelLoader.hasLabels()` keys
    // on the same three attrs; the Points twin gates identically.
    const wantsElementIds =
      attrs.has_labels === true || attrs.has_image_labels === true || attrs.has_keys === true;

    // Load directly into the accumulator buffers (zero allocations).
    if (
      !this._accumulator &&
      appConfig.dataLoading.performance.useAccumulators &&
      this._accumulatorConfig
    ) {
      const { capacity, ndim } = this._accumulatorConfig;
      this._accumulator = new GSplatsDataAccumulator(capacity, ndim);
      this._accumulator.configureColorComponents(this.colorComponents);
    }

    if (this._accumulator && appConfig.dataLoading.performance.useAccumulators) {
      // Capture locally: dispose() (dataset switch) can null + dispose
      // `this._accumulator` while the chunk loads below are in flight —
      // dereferencing the field again after the awaits raced a TypeError
      // ("Cannot read properties of null"). The identity re-check after
      // the loads turns that race into a quiet cancellation.
      const accumulator = this._accumulator;
      // Ensure capacity FIRST
      accumulator.ensureCapacity(totalSplats);

      // Initialize accumulator types based on array metadata (must be done BEFORE loading!)
      // This ensures colorBuffer has the correct type (Uint8/Uint16/Float32)
      // IMPORTANT: For encoded arrays, use encoding.original_dtype NOT the zarr array
      // dtype (mirrors the lines loader). The zarr dtype is the quantized container
      // (uint16 for geolog_perchannel HDR colors) while loadColorRanges returns the
      // ORIGINAL dtype (float32). Typing the accumulator by the container made
      // loadColorRanges swap in an EXACT-size Float32Array on the first HDR load;
      // a later larger load then reused that undersized buffer and silently
      // truncated colors (splats past the first load's count rendered colorless).
      if (this.arrays.colors) {
        const colorAttrs = this.arrays.colors.attrs as unknown as ArrayMetadata;
        const colorDtype = colorAttrs?.encoding?.original_dtype || String(this.arrays.colors.dtype);
        const colorType = getExpectedColorType(colorDtype);
        // Create a small typed array to initialize accumulator types
        const sampleColors =
          colorType === 'Uint8Array'
            ? new Uint8Array(3)
            : colorType === 'Uint16Array'
              ? new Uint16Array(3)
              : new Float32Array(3);
        accumulator.fill(0, {
          positions: new Float32Array(attrs.ndim),
          amplitudes: new Float32Array(1),
          choleskyFactors: new Float32Array(choleskyPackedSize(attrs.ndim)),
          colors: sampleColors,
        });
      }

      // Get direct buffer references for zero-allocation loading (now colorBuffer has correct type!)
      const centerBuffer = accumulator.getCenterBuffer();
      const amplitudeBuffer = accumulator.getAmplitudeBuffer();
      const choleskyBuffer = accumulator.getCholeskyBuffer();

      // Load directly into accumulator buffers (ZERO intermediate allocations!)
      // All four attribute arrays load CONCURRENTLY — distinct zarr arrays
      // writing into distinct accumulator buffers; the global fetch gate
      // (utils/fetch-concurrency.ts) bounds total network concurrency.
      const loadSession = session?.begin('Load Arrays');
      const labelLoad = this.loadLabelChannel(splatRanges, attrs);
      try {
        const colorLoad = this.arrays.colors
          ? (async () => {
              // Use loadColorRanges for proper multi-type handling
              // NOTE: For LUT encoding, loadColorRanges may return a different buffer type
              // (Float32Array) than the accumulator's colorBuffer (Uint8Array based on stored dtype).
              // We MUST use the returned buffer since it contains the decoded colors.
              const colorBuffer = accumulator.getColorBuffer();
              const loadedColors = await this.loadColorRanges(splatRanges, colorBuffer);

              // If loadColorRanges returned a different buffer (e.g., LUT decoded to Float32Array),
              // we need to update the accumulator with the new buffer
              if (loadedColors !== colorBuffer) {
                // Replace accumulator's color buffer with the decoded colors
                // This handles LUT encoding where decoded output is Float32Array
                accumulator.setColorBuffer(loadedColors);
              }
            })()
          : Promise.resolve();

        await Promise.all([
          this.loadArrayRanges('centers', splatRanges, attrs.ndim, centerBuffer),
          this.loadArrayRanges('amplitudes', splatRanges, 1, amplitudeBuffer),
          this.loadCholeskyRanges(splatRanges, attrs.ndim, choleskyBuffer),
          colorLoad,
          labelLoad,
        ]);
      } finally {
        loadSession?.end();
      }

      // Disposed mid-load (dataset switch tore this loader down while the
      // chunk reads were in flight): the update was abandoned on purpose.
      // Bail as a cancellation — run-loader-updates' isAbortError branch
      // stages null quietly (no failure record, no retry, no error log) —
      // instead of reading subarrays out of a disposed accumulator.
      if (this._accumulator !== accumulator) {
        throw new DOMException(
          `GSplats loader disposed during load: ${this.node.path}`,
          'AbortError'
        );
      }

      // Return from accumulator (subarrays, zero copy!)
      // NO fill() needed - data already in buffers!
      // `getData` mints a FRESH object literal every call, so stamping the
      // visible ranges onto it cannot mutate a previously returned payload.
      const data = accumulator.getData(totalSplats);
      const labels = await labelLoad;
      if (labels) {
        data.labelIndices = labels.indices;
        data.labelVocabulary = labels.vocabulary;
      }
      if (wantsElementIds) data.ranges = splatRanges;
      return data;
    }

    // Fallback: Load to separate arrays (allocations when accumulator disabled)
    let centers: Float32Array;
    let amplitudes: Float32Array;
    let choleskyFactors: Float32Array;
    let colors: Float32Array | Uint8Array | Uint16Array | null;
    let labels: GSplatLabelChannel | null;

    // All four attribute arrays load CONCURRENTLY (distinct zarr arrays,
    // distinct freshly-allocated output buffers).
    const loadSession = session?.begin('Load Arrays');
    try {
      [centers, amplitudes, choleskyFactors, colors, labels] = await Promise.all([
        this.loadArrayRanges('centers', splatRanges, attrs.ndim),
        this.loadArrayRanges('amplitudes', splatRanges, 1),
        this.loadCholeskyRanges(splatRanges, attrs.ndim),
        this.arrays.colors ? this.loadColorRanges(splatRanges) : Promise.resolve(null),
        this.loadLabelChannel(splatRanges, attrs),
      ]);
    } finally {
      loadSession?.end();
    }

    return {
      positions: centers,
      amplitudes,
      choleskyFactors,
      colors,
      colorComponents: this.colorComponents,
      ...(labels ? { labelIndices: labels.indices, labelVocabulary: labels.vocabulary } : {}),
      splatCount: totalSplats,
      ndim: attrs.ndim,
      ...(wantsElementIds ? { ranges: splatRanges } : {}),
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
    return runWithActiveSignal(
      (s) => (this._activeSignal = s),
      signal,
      async () => {
        const result = await this.loadGSplats(viewState, session);
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
    viewState: GSplatsViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<{ data: LoadedGSplatsData; allResident: boolean }> {
    return runWithResidencyProbe(
      (p) => (this._activeProbe = p),
      () => this.updateView(viewState, session, signal)
    );
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
   * The per-node tolerance inputs for `queryVisibleSplatRanges`, computed once
   * (see `_toleranceOptions`).
   */
  private toleranceOptions(attrs: GSplatsMetadata): ToleranceOptions {
    this._toleranceOptions ??= {
      barrierDims: readBarrierDims(attrs, this.node.path),
      truncationRadius: resolveTruncationRadius(attrs),
    };
    return this._toleranceOptions;
  }

  /**
   * Query visible splat ranges based on view state.
   *
   * Delegates the chunk-bounds AABB scan and range coalescing to the canonical
   * `SpatialQueryBuilder`, which also handles the `extend_to_all` short-circuit.
   * Returns a load-all range when no spatial index is available.
   */
  private async queryVisibleSplatRanges(viewState: GSplatsViewState): Promise<SplatRange[]> {
    // No local preimage for this world slice under the node's nd_transform —
    // see the identical guard in the points/lines loaders and
    // `ViewState.noPreimage`.
    if (viewState.noPreimage) {
      log.query(
        Modules.GSPLATS_SPATIAL_INDEX_LOADER,
        'No preimage for this slice under nd_transform'
      );
      return [];
    }

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
      // Both of these are things only THIS node knows and the tolerance computer
      // cannot see from a `DimensionInfo`: which dims the writer barrier-padded
      // (rather than σ-expanded) in `chunk_bounds`, and the truncation radius the
      // renderer will draw this node with. See `readBarrierDims` /
      // `resolveTruncationRadius`, and issue #1655 items 2 and 3.
      toleranceOptions: this.toleranceOptions(attrs),
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

    // `diag`/`offdiag` arrive already decoded to float32: the split arrays use
    // the generic per-channel encodings (diag log-uint8/16, off signed-log-uint8/16),
    // which the shared RangeLoader fully decodes via its `'perchannel'` path
    // (float32/legacy arrays pass through unchanged). This is pure geometry
    // assembly now — scatter each decoded half into its packed columns, mirroring
    // the Python reader's `recombine_cholesky` → `merge_tril` on decoded arrays.
    const diagIdx = choleskyDiagIndices(ndim);
    for (let s = 0; s < totalSplats; s++) {
      const base = s * k;
      const dbase = s * d;
      for (let c = 0; c < d; c++) packed[base + diagIdx[c]] = diag[dbase + c];
    }

    if (offdiag) {
      const offIdx = choleskyOffdiagIndices(ndim);
      for (let s = 0; s < totalSplats; s++) {
        const base = s * k;
        const obase = s * offLen;
        for (let c = 0; c < offLen; c++) {
          packed[base + offIdx[c]] = offdiag[obase + c];
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

    recordLoadMetrics(this.facadeCtx, arrayName, totalSplats, output);
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
    recordLoadMetrics(this.facadeCtx, 'colors', totalSplats, output);
    return output;
  }

  private async loadLabelChannel(
    ranges: SplatRange[],
    attrs: GSplatsMetadata
  ): Promise<GSplatLabelChannel | null> {
    const array = this.arrays.label_ids;
    if (!array) return null;
    const vocabulary = attrs.label_vocabulary;
    if (!vocabulary) {
      throw new Error('[GSplatsLoader] has_label_ids requires label_vocabulary');
    }

    const expectedLength = ranges.reduce((sum, range) => sum + range.end - range.start, 0);
    let output: Uint32Array;
    if (this._accumulator && appConfig.dataLoading.performance.useAccumulators) {
      if (this._labelIndicesScratch.length < expectedLength) {
        this._labelIndicesScratch = new Uint32Array(expectedLength);
      }
      output = this._labelIndicesScratch.subarray(0, expectedLength);
    } else {
      output = new Uint32Array(expectedLength);
    }
    const isBroadcast = array.shape[0] === 1;
    const reads = isBroadcast ? [{ start: 0, end: 1 }] : ranges;
    const chunks = await Promise.all(
      reads.map((range) =>
        zarr.readArray(
          array,
          [zarr.slice(range.start, range.end)],
          zarr.abortOptions(this._activeSignal)
        )
      )
    );
    let labelVocabulary: GSplatLabelChannel['vocabulary'];
    if (isBroadcast) {
      const values = chunks[0].data as Uint8Array | Uint16Array | Uint32Array | BigUint64Array;
      if (values.length !== 1) {
        throw new Error(`[GSplatsLoader] broadcast label_ids length ${values.length}, expected 1`);
      }
      const compacted = compactGSplatLabelIds(values, vocabulary);
      output.fill(compacted.indices[0]);
      labelVocabulary = compacted.vocabulary;
    } else {
      let offset = 0;
      for (const chunk of chunks) {
        const values = chunk.data as Uint8Array | Uint16Array | Uint32Array | BigUint64Array;
        if (offset + values.length > expectedLength) {
          throw new Error(`[GSplatsLoader] label_ids length exceeds expected ${expectedLength}`);
        }
        const compacted = compactGSplatLabelIds(
          values,
          vocabulary,
          output.subarray(offset, offset + values.length)
        );
        labelVocabulary = compacted.vocabulary;
        offset += values.length;
      }
      if (offset !== expectedLength) {
        throw new Error(`[GSplatsLoader] label_ids length ${offset}, expected ${expectedLength}`);
      }
    }
    recordLoadMetrics(this.facadeCtx, 'label_ids', output.length, output);
    return {
      indices: output,
      vocabulary: labelVocabulary!,
    };
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
  async prefetchChunks(viewState: GSplatsViewState, signal?: AbortSignal): Promise<void> {
    await this.ensureInitialized();

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
      this.arrays.label_ids,
    ].filter((a): a is zarr.Array<zarr.DataType, zarr.Readable> => a != null);

    await prefetchRangesIntoCache(arrays, splatRanges, signal);
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
    // Chunk-index telemetry for the monitor advisor; shared across the
    // three facades (see buildSpatialIndexMetrics).
    if (this.chunkIndex) {
      this.metrics.spatialIndex = buildSpatialIndexMetrics(
        this.chunkIndex.chunkCount,
        this.metrics.queries,
        this.totalQueryCells,
        this.metrics.elementsLoaded
      );
    }

    return { ...this.metrics };
  }

  getActiveQueries(): QueryInfo[] {
    return Array.from(this.activeQueries.values());
  }

  private emitEvent(event: MonitorEvent): void {
    this.events.emit(event);
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

    this._toleranceOptions = null;

    // Dispose accumulator
    if (this._accumulator) {
      this._accumulator.dispose();
      this._accumulator = null;
    }
    this._accumulatorConfig = null;
  }
}
