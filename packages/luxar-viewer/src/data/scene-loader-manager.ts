/**
 * SceneLoaderManager - Manages SceneLoader instances without global state
 *
 * This manager provides a clean way to access SceneLoader instances without
 * polluting the global window object. It uses a singleton pattern with
 * explicit instance management.
 *
 * Also owns the UpdateProfiler singleton for timing scene updates.
 */

import { SceneLoader, type SceneLoaderLODGroupRegistryFactory } from './scene-loader';
import { LoaderConfig } from './data-loader-types';
import { UpdateProfiler } from '../profiling/update-profiler';
import type { SceneLoaderMonitorFactory } from './scene-loader-monitor-port';
import type { KTX2TextureDecoder } from '../types/mesh';
import type {
  DensityGateCaps,
  ProjectedDensityProvider,
} from './scene-loader/progressive/density-gate';

/**
 * Manager for SceneLoader instances.
 * Provides centralized access to loader instances without global variables.
 */
export class SceneLoaderManager {
  private static instance: SceneLoaderManager | null = null;
  private loaders = new Map<string, SceneLoader>();
  private defaultLoaderId: string | null = null;

  /**
   * Update profiler for timing scene updates
   * Singleton owned by the manager, shared with all loaders
   */
  private readonly profiler: UpdateProfiler;

  /**
   * Optional monitor factory injected by `core/app.ts` so each
   * `SceneLoader` we create can resolve a UI monitor without the
   * `data/` layer importing `ui/`. Null means "no UI monitor wired
   * up" (tests / embedders) and SceneLoader treats every monitor
   * call as a no-op.
   */
  private monitorFactory: SceneLoaderMonitorFactory | null = null;

  /**
   * Optional LOD-group registry factory. Same dependency-inversion
   * pattern as ``monitorFactory`` — the app pipeline owns the live
   * SceneManager (camera, viewport) and closes over it to build a
   * registry per loader without the data/ layer importing scene/.
   */
  private lodGroupRegistryFactory: SceneLoaderLODGroupRegistryFactory | null = null;

  /**
   * Optional render-loop wake-up forwarded to every created loader
   * (→ `SceneLoader.setRequestRender`). Same dependency-inversion
   * pattern as ``monitorFactory``: the app pipeline wires it to
   * `AnimationController.startAnimation` so late geometry commits
   * (refinement, retries, lazy LOD loads) repaint after the rAF loop
   * has idle-paused. Null (tests / embedders) means commits never
   * wake a loop — SceneLoader treats it as a no-op.
   */
  private requestRender: (() => void) | null = null;
  /**
   * Projected-density provider + caps for the refinement rung gate, forwarded
   * to every created loader (→ `SceneLoader.setRefinementDensityProvider`).
   * Null provider = bytes-only admission.
   */
  private refinementDensity: {
    provider: ProjectedDensityProvider | null;
    caps: DensityGateCaps;
  } | null = null;
  private decodeKTX2: KTX2TextureDecoder | null = null;

  /**
   * Private constructor to enforce singleton pattern
   */
  private constructor() {
    this.profiler = new UpdateProfiler();
  }

  /**
   * Provide the monitor factory. Called once at app boot from
   * `core/app.ts` (which holds the `DataMonitorManager` reference).
   * Subsequent `createLoader` calls forward the factory to each
   * `SceneLoader` instance.
   */
  setMonitorFactory(factory: SceneLoaderMonitorFactory | null): void {
    this.monitorFactory = factory;
  }

  /**
   * Provide the LOD-group registry factory. Called once at app boot
   * from the init pipeline (which holds SceneManager + the live
   * camera). Subsequent ``createLoader`` calls forward the factory
   * to each ``SceneLoader`` instance.
   */
  setLODGroupRegistryFactory(factory: SceneLoaderLODGroupRegistryFactory | null): void {
    this.lodGroupRegistryFactory = factory;
  }

  /**
   * Provide the render-loop wake-up callback. Called once at app boot
   * from the init pipeline; forwarded to each subsequently created
   * ``SceneLoader`` (see the ``requestRender`` field).
   */
  setRequestRender(callback: (() => void) | null): void {
    this.requestRender = callback;
  }

  /**
   * Provide the projected-density source for the refinement rung gate.
   * Forwarded to every existing and subsequently created ``SceneLoader``.
   */
  setRefinementDensityProvider(
    provider: ProjectedDensityProvider | null,
    caps: DensityGateCaps
  ): void {
    this.refinementDensity = { provider, caps };
    for (const loader of this.loaders.values()) {
      loader.setRefinementDensityProvider(provider, caps);
    }
  }

  setKTX2TextureDecoder(decoder: KTX2TextureDecoder | null): void {
    if (this.decodeKTX2 !== decoder) this.decodeKTX2?.dispose();
    this.decodeKTX2 = decoder;
  }

  /**
   * Get the update profiler instance
   * Used by DataLoadingMonitor to display timing data
   */
  getProfiler(): UpdateProfiler {
    return this.profiler;
  }

  /**
   * Get the singleton instance of SceneLoaderManager
   */
  static getInstance(): SceneLoaderManager {
    if (!SceneLoaderManager.instance) {
      SceneLoaderManager.instance = new SceneLoaderManager();
    }
    return SceneLoaderManager.instance;
  }

  /**
   * Create a new SceneLoader instance
   *
   * @param id - Unique identifier for this loader
   * @param config - Optional loader configuration
   * @param setAsDefault - Whether to set this as the default loader
   * @returns The created SceneLoader instance
   */
  createLoader(
    id: string = 'default',
    config?: LoaderConfig,
    setAsDefault: boolean = true
  ): SceneLoader {
    // Dispose existing loader with same ID if it exists
    if (this.loaders.has(id)) {
      this.destroyLoader(id);
    }

    // Pass the profiler + monitor + LOD-group-registry factories to
    // the loader. Both factories are dependency-inversion handles
    // that let `SceneLoader` reach the UI monitor / live SceneManager
    // without importing `ui/` or `scene/` directly.
    const loader = new SceneLoader(
      config,
      id,
      this.profiler,
      this.monitorFactory,
      this.lodGroupRegistryFactory,
      this.decodeKTX2
    );
    loader.setRequestRender(this.requestRender);
    if (this.refinementDensity) {
      loader.setRefinementDensityProvider(
        this.refinementDensity.provider,
        this.refinementDensity.caps
      );
    }
    this.loaders.set(id, loader);

    if (setAsDefault || !this.defaultLoaderId) {
      this.defaultLoaderId = id;
    }

    return loader;
  }

  /**
   * Create a new SceneLoader, awaiting disposal of any existing loader with the
   * same ID first.
   *
   * Unlike {@link createLoader} (which fires the previous loader's async dispose
   * without awaiting), this awaits {@link destroyLoaderAsync} so caching-store
   * teardown and OPFS metadata flush fully drain before the replacement loader
   * is constructed. Dataset switches must use this path so the new loader never
   * races the old one's late teardown for cache ownership / OPFS metadata.
   *
   * @param id - Unique identifier for this loader
   * @param config - Optional loader configuration
   * @param setAsDefault - Whether to set this as the default loader
   * @returns The created SceneLoader instance
   */
  async createLoaderAsync(
    id: string = 'default',
    config?: LoaderConfig,
    setAsDefault: boolean = true
  ): Promise<SceneLoader> {
    // Await disposal of any existing loader with the same ID before building
    // the replacement (deterministic teardown; see destroyLoaderAsync).
    if (this.loaders.has(id)) {
      await this.destroyLoaderAsync(id);
    }

    const loader = new SceneLoader(
      config,
      id,
      this.profiler,
      this.monitorFactory,
      this.lodGroupRegistryFactory,
      this.decodeKTX2
    );
    loader.setRequestRender(this.requestRender);
    if (this.refinementDensity) {
      loader.setRefinementDensityProvider(
        this.refinementDensity.provider,
        this.refinementDensity.caps
      );
    }
    this.loaders.set(id, loader);

    if (setAsDefault || !this.defaultLoaderId) {
      this.defaultLoaderId = id;
    }

    return loader;
  }

  /**
   * Get a SceneLoader by ID
   *
   * @param id - The loader ID
   * @returns The SceneLoader instance or null if not found
   */
  getLoader(id: string): SceneLoader | null {
    return this.loaders.get(id) || null;
  }

  /**
   * Get the default SceneLoader
   *
   * @returns The default SceneLoader instance or null
   */
  getDefaultLoader(): SceneLoader | null {
    if (!this.defaultLoaderId) {
      return null;
    }
    return this.loaders.get(this.defaultLoaderId) || null;
  }

  /**
   * Get all active loaders
   *
   * @returns Map of all active loaders
   */
  getAllLoaders(): Map<string, SceneLoader> {
    return new Map(this.loaders);
  }

  /**
   * Whether ANY registered loader has a LOAD PASS outstanding — an `updateView`
   * sweep (fetch/decode/upload) up to its geometry commit, or a sweep that is
   * queued and has not started yet. See
   * {@link SceneLoader.isLoadPassInProgress} for the exact scope, in particular
   * why the progressive-LOD refinement drain is excluded and why the queued
   * slot counts.
   *
   * Consumed by the debug snapshot (`__luxarDebug.getState().isLoading`,
   * built in `core/app/debug/debug-interface.ts`), which the E2E "wait for
   * data" helpers poll to decide when a load has settled.
   *
   * The answer is "any loader", not "the default loader", because this
   * manager's contract admits several: `createLoader`/`createLoaderAsync` take
   * an id, `getAllLoaders()` returns a map, and the default is merely one
   * elected entry. So the aggregate answers for all of them rather than
   * trusting the default slot to be the only busy one. (Production registers
   * exactly one, under `'default'` — a dataset switch disposes the outgoing
   * loader before constructing its replacement, so the two never overlap.)
   *
   * @returns True if at least one loader is mid-load-pass; false when idle or
   *   when no loader is registered.
   */
  isAnyLoadPassInProgress(): boolean {
    for (const loader of this.loaders.values()) {
      if (loader.isLoadPassInProgress()) return true;
    }
    return false;
  }

  /**
   * Destroy a specific loader (best-effort, non-awaiting).
   *
   * Synchronously removes the loader from the manager (so subsequent
   * `getLoaderCount()` / `hasLoader()` calls reflect the change) and
   * fires the async dispose without awaiting. Suits the `beforeunload`
   * path and other call sites that cannot meaningfully await teardown.
   * Callers that need deterministic teardown (e.g. dataset switches)
   * must use {@link destroyLoaderAsync}.
   *
   * @param id - The loader ID to destroy
   */
  destroyLoader(id: string): void {
    const loader = this.detachLoader(id);
    if (loader) {
      void loader.dispose().catch((error) => {
        // Already-logged inside SceneLoader.dispose; this catch keeps
        // an unhandled rejection from leaking out of the fire-and-forget
        // path.
        void error;
      });
    }
  }

  /**
   * Destroy a specific loader and await its disposal.
   *
   * Awaits {@link SceneLoader.dispose} so caching-store teardown,
   * prefetcher teardown, and OPFS metadata flush all complete before
   * this method resolves. Use during dataset switches when the next
   * loader's init must see fully-drained state.
   */
  async destroyLoaderAsync(id: string): Promise<void> {
    const loader = this.detachLoader(id);
    if (!loader) return;
    try {
      await loader.dispose();
    } catch (error) {
      // Already-logged inside SceneLoader.dispose. Swallow here so the
      // caller can still proceed with the next dataset switch.
      void error;
    }
  }

  /**
   * Destroy all loaders and reset the manager (best-effort, non-awaiting).
   *
   * Mirror of {@link destroyLoader}: removes loaders synchronously and
   * fires async disposes without awaiting. Use {@link destroyAllAsync}
   * where deterministic teardown matters.
   */
  destroyAll(): void {
    const loaders = Array.from(this.loaders.values());
    this.loaders.clear();
    this.defaultLoaderId = null;
    for (const loader of loaders) {
      void loader.dispose().catch((error) => {
        void error;
      });
    }
  }

  /**
   * Destroy all loaders concurrently and await every disposal.
   *
   * `Promise.all`s every loader's `dispose()` so the caller can wait for
   * every prefetcher, caching store, and L0 cache to drain before
   * proceeding. Always clears the loaders map and default-loader id,
   * even if individual disposals reject.
   */
  async destroyAllAsync(): Promise<void> {
    const loaders = Array.from(this.loaders.values());
    this.loaders.clear();
    this.defaultLoaderId = null;
    await Promise.all(
      loaders.map((loader) =>
        loader.dispose().catch(() => {
          // Already-logged inside SceneLoader.dispose; swallow here so a
          // single bad loader cannot prevent the others from completing.
        })
      )
    );
  }

  /**
   * Synchronously remove a loader from the manager and update the
   * default-loader id. Returns the loader instance for the caller to
   * dispose (sync or async). Centralizes the bookkeeping so the
   * fire-and-forget and awaitable variants stay in sync.
   *
   * Default-election contract: when the detached loader was the current
   * default, the next default is the FIRST remaining loader by insertion
   * order — i.e. the oldest loader still registered. This ordering is
   * a stable property of `Map` and therefore deterministic across reloads
   * for any given sequence of registrations. Callers that need a specific
   * default should call `setDefault()` explicitly rather than relying on
   * the implicit election.
   */
  private detachLoader(id: string): SceneLoader | null {
    const loader = this.loaders.get(id);
    if (!loader) return null;
    this.loaders.delete(id);
    if (this.defaultLoaderId === id) {
      // Per the contract above: next default is the oldest remaining
      // loader by registration order (Map insertion order).
      this.defaultLoaderId =
        this.loaders.size > 0 ? (this.loaders.keys().next().value ?? null) : null;
    }
    return loader;
  }

  /**
   * Check if a loader exists
   *
   * @param id - The loader ID to check
   * @returns True if the loader exists
   */
  hasLoader(id: string): boolean {
    return this.loaders.has(id);
  }

  /**
   * Get the number of active loaders
   *
   * @returns The number of active loaders
   */
  getLoaderCount(): number {
    return this.loaders.size;
  }

  /**
   * Dispose the current instance and clear the singleton slot.
   *
   * Used at app shutdown and between tests. The next `getInstance()` call
   * lazily constructs a fresh manager.
   */
  static disposeInstance(): void {
    if (SceneLoaderManager.instance) {
      SceneLoaderManager.instance.destroyAll();
      SceneLoaderManager.instance.setKTX2TextureDecoder(null);
      SceneLoaderManager.instance = null;
    }
  }
}

/**
 * Convenience accessor for a managed {@link SceneLoader}.
 *
 * @param id - Loader id to look up; omit to return the manager's default loader.
 * @returns The matching loader, or null when no loader is registered under
 *   that id (or, for the default lookup, when no default has been set).
 */
export function getSceneLoader(id?: string): SceneLoader | null {
  const manager = SceneLoaderManager.getInstance();
  return id ? manager.getLoader(id) : manager.getDefaultLoader();
}
