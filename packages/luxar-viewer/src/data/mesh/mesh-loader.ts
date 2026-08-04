/**
 * Whole-node mesh loader.
 *
 * The simplest of the four geometry loaders, and deliberately so. It fetches
 * `vertices`, `faces` and the optional per-vertex arrays **in full**, decodes
 * them once, and holds them: no spatial index, no progressive refinement, no
 * chunk-bounds query, no range arithmetic.
 *
 * ## Why whole-node is the right shape, not a shortcut
 *
 * `lines-spatial-index-loader.ts` runs to ~1400 LOC because line datasets reach
 * tens of millions of vertices with a *meaningful per-slice working set* — the
 * index earns its complexity by letting a slice change fetch a fraction of the
 * data. A mesh has no such fraction: faces share vertices across any cut, so the
 * working set after a `displayDims` change is the whole mesh regardless. An index
 * would add machinery and skip nothing.
 *
 * Meshes in this domain (isosurfaces, segmentation boundaries) are typically at
 * most a few million triangles and fit comfortably. `MESH_DECODE_BUDGET_BYTES`
 * is what keeps "comfortably" from being an assumption.
 *
 * This class still implements the full {@link MeshDataLoader} interface, so a
 * spatial-index implementation can be swapped in behind it later with no caller
 * change.
 *
 * ## The consequence for `updateView`
 *
 * Because the mesh is resident in full, `updateView` returns the *same* data
 * every time — it does not re-fetch. What varies with the view is only which
 * faces get indexed, and that is `projection.ts`'s job, downstream. This is the
 * one place a reader might expect a re-fetch and find none, so it is stated
 * plainly rather than left to be inferred from the absence of code.
 *
 * ## Two-stage admission
 *
 * Every fetch is gated. `mesh-preflight.ts` (Stage 1) runs on metadata alone,
 * before a single chunk is requested; `mesh-validate.ts` (Stage 2) runs on the
 * materialized values. See those modules for why the split is load-bearing.
 *
 * @module data/mesh/mesh-loader
 */

import * as zarr from '../zarr';
import { log, Modules } from '../../utils/log';
import { ArrayDecoder, type ArrayMetadata, type ArrayRefRegistry } from '../array-decoder/decoder';
import { colorComponentsOf, loadColorRanges } from '../loaders';
import { RangeLoader } from '../loaders/spatial-query/range-loader';
import { LoaderError, classifyLoaderError } from '../scene-loader/nodes/load-leaf-error-dispatch';
import {
  MESH_ARRAY_NAMES,
  preflightMesh,
  type MeshArrayHandles,
  type MeshPreflightResult,
} from './mesh-preflight';
import { validateFaceIndices, validateMaterializedLength } from './mesh-validate';
import type { FaceIndexSource } from './mesh-validate';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoadedMeshData,
  MeshColorArray,
  MeshDataLoader,
  MeshMetadata,
  MeshViewState,
} from '../../types/mesh';

/** What the loader needs from the scene-loader to do its job. */
export interface MeshLoaderDeps {
  /** Store handle, for resolving `array_ref` targets during decode. */
  zarrStore: zarr.Readable;
  /** Shared registry so an `array_ref` shared with a sibling node decodes once. */
  arrayRefRegistry: ArrayRefRegistry;
}

/** The optional arrays, mapped to the attr flag that declares each present. */
const OPTIONAL_ARRAYS = [
  ['normals', 'has_normals'],
  ['colors', 'has_colors'],
  ['scalars', 'has_scalars'],
  ['labelOffsets', 'has_labels'],
  ['labelBytes', 'has_labels'],
  ['imageLabelOffsets', 'has_image_labels'],
  ['imageLabelBytes', 'has_image_labels'],
] as const satisfies readonly (readonly [keyof MeshArrayHandles, keyof MeshMetadata])[];

export class MeshLoader implements MeshDataLoader {
  private readonly decoder: ArrayDecoder;
  private readonly rangeLoader: RangeLoader;
  private readonly zarrStore: zarr.Readable;

  /** Metadata-only handles, opened once by {@link initialize}. */
  private handles: MeshArrayHandles | null = null;

  /** What Stage 1 established. */
  private preflight: MeshPreflightResult | null = null;

  /**
   * The decoded mesh, cached for the loader's lifetime.
   *
   * This is the whole point of the whole-node design: one fetch per node, and
   * every later `updateView` is served from here.
   */
  private data: LoadedMeshData | null = null;

  /** Serializes concurrent first loads onto one fetch (see {@link load}). */
  private inFlight: Promise<LoadedMeshData> | null = null;

  /**
   * Bumped by {@link dispose}, so a fetch that settles afterwards cannot write its
   * result back into a loader that has been torn down.
   *
   * Without it, `dispose()` during an in-flight load leaves the completion free to
   * repopulate `this.data` — resurrecting the cache on a dead loader, and for a
   * near-budget mesh pinning up to half a gigabyte that nothing will ever read.
   * Cheap enough that the whole-mesh payload makes it worth having.
   */
  private generation = 0;

  constructor(
    private readonly path: string,
    private readonly attrs: MeshMetadata,
    private readonly location: zarr.Location<zarr.Readable>,
    deps: MeshLoaderDeps
  ) {
    this.decoder = new ArrayDecoder(deps.arrayRefRegistry);
    this.rangeLoader = new RangeLoader(deps.arrayRefRegistry);
    this.zarrStore = deps.zarrStore;
  }

  /**
   * Open metadata handles for every array the attrs declare, then run Stage 1.
   *
   * Opening is metadata-only — `zarr.open(..., { kind: 'array' })` reads
   * `.zarray` and `.zattrs` and nothing else — which is what makes it safe to do
   * for *all* declared arrays, including the label CSR pair v1 never fetches.
   * The budget has to see them to be a budget.
   */
  private async initialize(): Promise<void> {
    if (this.handles) return;

    const open = async (name: string) => zarr.open(this.location.resolve(name), { kind: 'array' });

    let vertices: zarr.Array<zarr.DataType, zarr.Readable>;
    let faces: zarr.Array<zarr.DataType, zarr.Readable>;
    try {
      [vertices, faces] = await Promise.all([open('vertices'), open('faces')]);
    } catch (error) {
      // A mesh without both of these is not a mesh. But the CAUSE decides the kind:
      // `classifyLoaderError` tells a genuinely malformed store from a transient
      // network failure, and mis-labelling the latter 'Validation' would have the
      // failure record treat it as deterministic and never retry it. Matches how the
      // sibling node loaders route raw errors.
      throw new LoaderError(classifyLoaderError(error), this.path, error);
    }

    const handles: MeshArrayHandles = { vertices, faces };
    await Promise.all(
      OPTIONAL_ARRAYS.map(async ([slot, flag]) => {
        if (!this.attrs[flag]) return;
        try {
          handles[slot] = await open(MESH_ARRAY_NAMES[slot]);
        } catch {
          // Leave the slot empty. The preflight turns a flag-with-no-array into
          // an explicit rejection, which reports better than a bare open error.
        }
      })
    );

    // `storeRoot` is passed so the preflight can follow an `array_ref` to the array
    // whose bytes are actually fetched — the referring array is a `(0, k)` stub, so
    // budgeting the handle alone charges ~48 bytes for a read that can pull gigabytes.
    this.preflight = await preflightMesh(this.path, this.attrs, handles, zarr.root(this.zarrStore));
    this.handles = handles;
  }

  /**
   * Fetch + decode everything, then run Stage 2.
   *
   * Ordering is deliberate: `faces` is validated and widened *before* it is
   * stored on {@link LoadedMeshData}, so no caller can ever observe an
   * unvalidated index array.
   */
  private async fetch(signal?: AbortSignal): Promise<LoadedMeshData> {
    await this.initialize();
    const handles = this.handles;
    const pre = this.preflight;
    if (!handles || !pre) {
      throw new LoaderError('Unexpected', this.path, new Error('mesh loader not initialized'));
    }

    const { nVertices, nFaces, ndim } = pre;
    const opts = zarr.abortOptions(signal);
    const storeRoot = zarr.root(this.zarrStore);

    // `vertices` and `normals` are both SemanticType.COORDINATE, which the
    // encoder may quantize (per-channel u16) or LUT-encode, so both must go
    // through the decoder rather than a raw read — the on-disk dtype is NOT
    // float32 in the general case. The decoder always yields Float32Array.
    const verticesAttrs = handles.vertices.attrs as unknown as ArrayMetadata;
    const vertices = await this.decoder.decode(
      handles.vertices,
      verticesAttrs,
      nVertices * ndim,
      storeRoot
    );
    validateMaterializedLength(this.path, 'vertices', vertices.length, nVertices * ndim);

    // `faces` is read RAW, in the store's own dtype, and widened only after the
    // two-sided range check. Routing it through the decoder would hand back a
    // Float32Array, whose 24-bit mantissa cannot represent every index a
    // 2^27-vertex mesh may carry — it would silently round indices above
    // 16,777,216. See mesh-validate.ts.
    const rawFaces = await zarr.readArray(handles.faces, undefined, opts);
    const faces = validateFaceIndices(
      this.path,
      rawFaces.data as FaceIndexSource,
      nVertices,
      nFaces * 3
    );

    let normals: Float32Array | null = null;
    if (handles.normals) {
      // Validate, then publish. Assigning first and checking after would leave a
      // window where the field holds an array Stage 2 has not vetted.
      const decoded = await this.decoder.decode(
        handles.normals,
        handles.normals.attrs as unknown as ArrayMetadata,
        nVertices * 3,
        storeRoot
      );
      validateMaterializedLength(this.path, 'normals', decoded.length, nVertices * 3);
      normals = decoded;
    }

    let colors: MeshColorArray | null = null;
    let colorComponents: 3 | 4 | undefined;
    if (handles.colors) {
      colorComponents = colorComponentsOf(handles.colors);
      // The shared colour path, reused verbatim: it preserves the native dtype
      // on the direct/rgb_uint8 branches (the GPU normalizes uint8/uint16 to
      // [0,1] for free, at a third of float32's bytes) and handles the
      // quantized / LUT / broadcast / array_ref cases. A whole-node read is
      // just the single range [0, V).
      colors = (await loadColorRanges(
        handles.colors,
        [{ start: 0, end: nVertices }],
        this.rangeLoader,
        this.zarrStore,
        `[MeshLoader] ${this.path}`
      )) as MeshColorArray;
      validateMaterializedLength(this.path, 'colors', colors.length, nVertices * colorComponents);
    }

    let scalars: Float32Array | undefined;
    if (handles.scalars) {
      const decoded = await this.decoder.decode(
        handles.scalars,
        handles.scalars.attrs as unknown as ArrayMetadata,
        nVertices,
        storeRoot
      );
      validateMaterializedLength(this.path, 'scalars', decoded.length, nVertices);
      scalars = decoded;
    }

    const data: LoadedMeshData = {
      vertices,
      faces,
      normals,
      colors,
      colorComponents,
      scalars,
      vertexCount: nVertices,
      faceCount: nFaces,
      ndim,
    };

    log.success(
      Modules.SCENE_LOADER,
      `Loaded mesh ${this.path}: ${nVertices.toLocaleString()} vertices, ` +
        `${nFaces.toLocaleString()} faces (${ndim}D)`
    );
    return data;
  }

  /**
   * Serve the mesh, fetching it at most once per loader instance.
   *
   * The `inFlight` latch is not defensive padding. Both entry points below
   * funnel here, and the scene-loader legitimately calls `updateView` while an
   * initial `loadMesh` is still in flight (a slice scrub during load). Without
   * the latch each call would start its own full-mesh fetch — the exact
   * duplicate-work the whole-node design exists to avoid.
   *
   * One consequence of sharing that fetch: only the FIRST caller's `signal`
   * reaches the reads. A later caller joining an in-flight fetch cannot abort it
   * and will receive the mesh even if its own update was superseded. That is
   * benign here in a way it would not be for the range-query siblings — a
   * superseded caller gets data it no longer needs, never data for the wrong
   * query, because there is only one thing to fetch and it does not depend on the
   * view. The wasted work is bounded by one mesh, once per loader.
   */
  private load(signal?: AbortSignal): Promise<LoadedMeshData> {
    if (this.data) return Promise.resolve(this.data);
    if (this.inFlight) return this.inFlight;

    const generation = this.generation;
    const mine: Promise<LoadedMeshData> = this.fetch(signal)
      .then((data) => {
        // Only publish if this loader has not been disposed since the fetch began.
        // The caller still receives the data — it is view-independent, so it is not
        // wrong, just unwanted — but the loader does not retain it.
        if (generation === this.generation) this.data = data;
        return data;
      })
      .finally(() => {
        // Clear the latch only if it is still OURS. `dispose()` nulls `inFlight`
        // while this promise may still be pending, so a stale completion landing
        // after a replacement load has begun would otherwise wipe the NEW load's
        // latch — and from then until that fetch publishes, every `updateView` (a
        // slice scrub, precisely the case the latch exists for) starts another
        // whole-mesh fetch. Dispose-then-reload is a designed, tested path here, so
        // this is a live race rather than a theoretical one.
        //
        // Cleared on failure too, so a retry can start a fresh fetch rather than
        // re-awaiting the rejected promise forever.
        if (this.inFlight === mine) this.inFlight = null;
      });
    this.inFlight = mine;
    return mine;
  }

  /** Load the mesh. `viewState` is accepted for interface symmetry and unused. */
  loadMesh(_viewState: MeshViewState): Promise<LoadedMeshData> {
    return this.load();
  }

  /**
   * Re-serve the mesh for a new view state.
   *
   * Returns the cached whole mesh — a view change never re-fetches, because
   * there is no subset to fetch. The visibility cull that *does* depend on the
   * view runs downstream in `projection.ts`.
   */
  updateView(
    _viewState: MeshViewState,
    _session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedMeshData> {
    return this.load(signal);
  }

  /**
   * Clear all cached state.
   *
   * State-clearing rather than terminal, matching the sibling loaders (the points
   * loader's `dispose` likewise drops its arrays and calls `_onceInit.reset()`):
   * a subsequent `loadMesh` re-initializes and re-fetches rather than throwing.
   */
  dispose(): void {
    this.generation++;
    this.handles = null;
    this.preflight = null;
    this.data = null;
    this.inFlight = null;
  }
}
