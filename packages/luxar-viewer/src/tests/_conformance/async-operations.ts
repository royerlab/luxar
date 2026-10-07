/**
 * Every cancellable or disposable async operation in the viewer, and the test
 * that proves its abort / dispose / failure / retry behaviour.
 *
 * `unit/conformance/async-operations.test.ts` keeps this table honest: it
 * scans `src/` for exported functions and class methods that take an
 * `AbortSignal`, and for exported classes that define `dispose()` beside an
 * `async` method, and fails for any such symbol that no row names and that
 * carries no `// lifecycle-exempt: <reason>` comment. It also fails for a row
 * naming a symbol that no longer exists or a test file that does not exist.
 *
 * A row with a `contract` runs the shared lifecycle contract
 * (`unit/_shared/lifecycle-contract.ts`, `defineLifecycleContract(<contract>,
 * …)`) in one of its test files; a row without one is covered by the
 * operation's own tests, which must name one of its symbols. New operations
 * should prefer the contract: it states abort, dispose, failure, retry,
 * supersede and double-dispose once, and an operation that cannot express a
 * case says why.
 */

export interface AsyncOperation {
  /** What the operation is. */
  readonly id: string;
  /** The symbols it covers: `<path under src/>#<exported name>`. */
  readonly symbols: readonly string[];
  /** Test files (under `src/tests/unit/`) that exercise its lifecycle. */
  readonly tests: readonly string[];
  /** The `defineLifecycleContract` name one of `tests` registers, when it runs the contract. */
  readonly contract?: string;
}

export const ASYNC_OPERATIONS: readonly AsyncOperation[] = [
  // ── Under the shared lifecycle contract ──────────────────────────────────
  {
    id: 'dataset load (loadDataset)',
    symbols: ['core/app/dataset/load-dataset.ts#loadDataset'],
    tests: ['core/app/dataset/load-dataset.test.ts', 'core/app/dataset/dataset-session.test.ts'],
    contract: 'loadDataset',
  },
  {
    id: 'HDRI environment load',
    symbols: ['rendering/environment/scene-environment.ts#SceneEnvironment'],
    tests: ['rendering/environment/scene-environment.test.ts'],
    contract: 'SceneEnvironment HDRI load',
  },
  {
    id: 'audio clip decode',
    symbols: ['audio/audio-engine.ts#AudioEngine'],
    tests: ['audio/audio-engine.test.ts'],
    contract: 'AudioEngine clip decode',
  },
  {
    id: 'L0 in-flight decode',
    symbols: [
      'cache/decompressed-chunk-cache.ts#DecompressedChunkCache',
      'cache/decompressed-chunk-cache/cached-zarr-array.ts#wrapWithCache',
    ],
    tests: ['cache/l0-coalescing.test.ts', 'cache/cached-zarr-array.test.ts'],
    contract: 'L0 in-flight decode',
  },
  {
    id: 'worker-pool dispatch, re-dispatch and init',
    symbols: ['workers/worker-pool.ts#WorkerPool'],
    tests: [
      'workers/worker-pool/lifecycle/contract.test.ts',
      'workers/worker-pool/abort/abort-signal.test.ts',
      'workers/worker-pool/selection/slot-accounting.test.ts',
      'workers/worker-pool/lifecycle/dispose.test.ts',
    ],
    contract: 'WorkerPool dispatch',
  },

  // ── Covered by their own tests ───────────────────────────────────────────
  {
    id: 'depth-sort coordinator (late commits after release)',
    symbols: ['rendering/depth-sort-coordinator.ts#DepthSortCoordinator'],
    tests: [
      'rendering/depth-sort-coordinator/multi-instance.test.ts',
      'rendering/depth-sort-coordinator.test.ts',
    ],
  },
  {
    id: 'scene loader: update passes, waitForUpdate, failed-loader retry, dispose',
    symbols: [
      'data/scene-loader.ts#SceneLoader',
      'data/scene-loader-manager.ts#SceneLoaderManager',
      'data/scene-identity-watchdog.ts#SceneIdentityWatchdog',
      'data/scene-loader/view-state/predicted-view-state.ts#dispatchPredictivePrefetch',
    ],
    tests: [
      'data/scene-loader.test.ts',
      'data/scene-loader/lifecycle/retry.test.ts',
      'data/scene-loader-dataset-signal.test.ts',
      'data/scene-identity-watchdog.test.ts',
      'data/scene-loader/view-state/predicted-view-state.test.ts',
    ],
  },
  {
    id: 'slice prefetcher shadow loads',
    symbols: [
      'data/scene-loader/prefetch/slice-prefetcher.ts#SlicePrefetcher',
      'data/loaders/progressive/slice-cache-helper.ts#awaitShadowStore',
    ],
    tests: [
      'data/scene-loader/prefetch/slice-prefetcher.test.ts',
      'data/scene-loader/prefetch/slice-prefetcher-pins.test.ts',
    ],
  },
  {
    id: 'per-type loaders (initialize, updateView, prefetch)',
    symbols: [
      'data/points/points-progressive-loader.ts#PointsProgressiveLoader',
      'data/points/points-spatial-index-loader.ts#PointsSpatialIndexLoader',
      'data/lines/lines-progressive-loader.ts#LinesProgressiveLoader',
      'data/lines/lines-spatial-index-loader.ts#LinesSpatialIndexLoader',
      'data/gsplats/gsplats-progressive-loader.ts#GSplatsProgressiveLoader',
      'data/gsplats/gsplats-spatial-index-loader.ts#GSplatsSpatialIndexLoader',
      'data/mesh/mesh-progressive-loader.ts#MeshProgressiveLoader',
      'data/mesh/mesh-whole-node-loader.ts#MeshWholeNodeLoader',
      'data/loaders/progressive/additive-ladder-core.ts#AdditiveLadderCore',
      'data/loaders/loader-lifetime.ts#LoaderLifetime',
      'data/loaders/active-load-context.ts#ActiveLoadContext',
      'data/loaders/abortable-wait.ts#abortableWait',
    ],
    tests: [
      'data/additive-ladder-core-contract.test.ts',
      'data/loaders/loader-lifetime.test.ts',
      'data/loaders/active-load-context.test.ts',
      'data/mesh/whole-node-loader.test.ts',
      'data/mesh-progressive-loader.test.ts',
    ],
  },
  {
    id: 'chunk reads: indices, bounds, ranges, decode',
    symbols: [
      'data/points/chunk-index-loader.ts#loadPointsChunkIndex',
      'data/lines/chunk-index-loader.ts#loadLinesDualChunkIndex',
      'data/gsplats/chunk-index-loader.ts#loadGSplatsChunkIndex',
      'data/loaders/chunk-bounds-loader.ts#fetchChunkBoundsArray',
      'data/loaders/spatial-query/range-loader.ts#RangeLoader',
      'data/loaders/spatial-query/range-loader/packed-ranges.ts#readRanges',
      'data/loaders/spatial-query/prefetch-ranges.ts#prefetchRangesIntoCache',
      'data/array-decoder/decoder.ts#ArrayDecoder',
      'data/array-decoder/broadcast-row.ts#readBroadcastRow',
      'data/mesh/texture-decode.ts#decodeMeshTexture',
    ],
    tests: [
      'data/points/chunk-index-loader.test.ts',
      'data/lines/chunk-index-loader.test.ts',
      'data/gsplats/chunk-index-loader.test.ts',
      'data/loaders/chunk-bounds-loader.test.ts',
      'data/loaders/spatial-query/range-loader-array-ref-l0.test.ts',
      'data/loaders/spatial-query/prefetch-ranges.test.ts',
      'data/mesh/whole-node-loader.test.ts',
      'data/mesh/texture-decode.test.ts',
    ],
  },
  {
    id: 'projection dispatch',
    symbols: [
      'data/scene-loader/process/data-processor-gsplats.ts#projectGSplatsTo3DUsingWorker',
      'data/scene-loader/process/data-processor-gsplats.ts#processGSplatsData',
      'data/scene-loader/process/data-processor-lines.ts#projectLinesTo3DUsingWorker',
      'data/scene-loader/process/data-processor-lines.ts#processLinesData',
    ],
    tests: [
      'data/scene-loader/data-processor-gsplats.test.ts',
      'data/scene-loader/data-processor-lines.test.ts',
      'data/scene-loader-dataset-signal.test.ts',
    ],
  },
  {
    id: 'partition part activation',
    symbols: [
      'scene/lod-group-registry.ts#LODGroupRegistry',
      'scene/partition-gate.ts#PartitionGate',
    ],
    tests: ['scene/lod-group-registry-partition-activation.test.ts'],
  },
  {
    id: 'fetch with retry, the fetch gate and the async gate',
    symbols: [
      'cache/multi-level-caching-store/fetch-retry.ts#fetchWithRetry',
      'utils/fetch-concurrency.ts#withFetchGate',
      'utils/async-gate.ts#AsyncGate',
    ],
    tests: [
      'cache/fetch-retry.test.ts',
      'utils/fetch-concurrency.test.ts',
      'utils/fetch-concurrency-priority.test.ts',
      'utils/async-gate.test.ts',
    ],
  },
  {
    id: 'multi-level caching store, validation and the root document',
    symbols: [
      'cache/multi-level-caching-store.ts#MultiLevelCachingStore',
      'cache/multi-level-caching-store/validation-queue.ts#ValidationQueue',
      'cache/multi-level-caching-store/validation-queue.ts#getRemoteContentHash',
      'cache/root-document-prefetch.ts#SharedRootDocumentSource',
    ],
    tests: [
      'cache/multi-level-caching-store.test.ts',
      'cache/validation-queue.test.ts',
      'cache/root-document-prefetch.test.ts',
    ],
  },
  {
    id: 'OPFS store, read gate and write queue',
    symbols: [
      'cache/multi-level-caching-store/opfs-store.ts#OPFSStore',
      'cache/multi-level-caching-store/opfs-read-gate.ts#withOpfsReadGate',
      'cache/multi-level-caching-store/opfs-write-queue.ts#OpfsWriteQueue',
    ],
    tests: [
      'cache/opfs-store.test.ts',
      'cache/opfs-read-gate.test.ts',
      'cache/opfs-write-queue.test.ts',
      'cache/opfs-persistence.test.ts',
    ],
  },
  {
    id: 'chunk sources and zip reads',
    symbols: [
      'cache/chunk-source/http-chunk-source.ts#HttpChunkSource',
      'cache/chunk-source/packed-chunk-source.ts#PackedChunkSource',
      'cache/chunk-source/zip-chunk-source.ts#ZipChunkSource',
      'data/zip/range-reader.ts#LuxarHttpRangeReader',
      'data/zip/store.ts#LuxarZipStore',
    ],
    tests: [
      'cache/chunk-source/http-chunk-source.test.ts',
      'cache/chunk-source/packed-chunk-source.test.ts',
      'cache/chunk-source/zip-chunk-source.test.ts',
      'data/zip/range-reader-signals.test.ts',
      'data/zip/member-read-abort.test.ts',
    ],
  },
  {
    id: 'abort-signal derivation',
    symbols: [
      'utils/abort-signals.ts#combineAbortSignals',
      'data/loaders/progressive/child-signal.ts#createChildController',
      'data/loaders/progressive/lookahead-signal.ts#createLookaheadController',
    ],
    tests: [
      'utils/abort-signals.test.ts',
      'data/loaders/progressive/child-signal.test.ts',
      'data/loaders/progressive/lookahead-signal.test.ts',
    ],
  },
  {
    id: 'camera flights',
    symbols: ['core/app/camera/camera-flight.ts#CameraFlight'],
    tests: ['core/app/camera/camera-flight.test.ts'],
  },
  {
    id: 'app and host lifetimes (init, dispose, device loss)',
    symbols: [
      'core/app.ts#LuxarApp',
      'core/layer/luxar-layer.ts#LuxarLayer',
      'scene/scene-manager.ts#SceneManager',
      'core/app/control/control-client.ts#ControlClient',
    ],
    tests: [
      'core/app.test.ts',
      'core/layer.test.ts',
      'core/multi-host.test.ts',
      'scene/scene-manager-events.test.ts',
      'core/app/init/pipeline.test.ts',
      'core/app/control/control-client.test.ts',
    ],
  },
  {
    id: 'picking and label loads',
    symbols: [
      'rendering/picking/picking-system.ts#PickingSystem',
      'data/loaders/picking/label-loader.ts#LabelLoader',
      'data/loaders/picking/image-label-loader.ts#ImageLabelLoader',
    ],
    tests: [
      'core/app/picking/init-picking.test.ts',
      'data/loaders/picking/label-loader.test.ts',
      'data/loaders/picking/image-label-loader.test.ts',
    ],
  },
  {
    id: 'post-processing pipeline',
    symbols: ['rendering/post-processing/post-processing-manager.ts#PostProcessingManager'],
    tests: ['rendering/post-processing-manager-lifecycle.test.ts'],
  },
  {
    id: 'UI panels with async work',
    symbols: [
      'ui/data-loading-monitor.ts#DataLoadingMonitor',
      'ui/overlay-manager.ts#OverlayManager',
      'ui/recording-panel.ts#RecordingPanel',
      'ui/recording-panel/offline-capture-strategy.ts#OfflineCaptureStrategy',
      'ui/recording-panel/screenshot-strategy.ts#ScreenshotStrategy',
      'ui/recording-panel/video-recording-strategy.ts#VideoRecordingStrategy',
    ],
    tests: [
      'ui/data-loading-monitor.test.ts',
      'ui/overlay-manager.test.ts',
      'core/app/overlays/dispose-overlays.test.ts',
      'ui/recording-panel.test.ts',
      'ui/recording-panel/offline-capture-strategy.test.ts',
      'ui/recording-panel/screenshot-strategy.test.ts',
      'ui/recording-panel/video-recording-strategy.test.ts',
    ],
  },
];
