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
 * Every fetch is gated. `preflight.ts` (Stage 1) runs on metadata alone,
 * before a single chunk is requested; `validate.ts` (Stage 2) runs on the
 * materialized values. See those modules for why the split is load-bearing.
 *
 * @module data/mesh/mesh-whole-node-loader
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
} from './preflight';
import { validateFaceIndices, validateMaterializedLength } from './validate';
import { decodeMeshTexture } from './texture-decode';
import type { FaceIndexSource } from './validate';
import { combineSignals } from '../../workers/worker-pool/timeout/combine-signals';
import { isAbortError } from '../loaders/abort-error';
import { LoaderEventEmitter } from '../loaders/monitor-events';
import { makeInitialLoaderMetrics, recordLoadEvent } from '../loaders/loader-metrics';
import type { UpdateSession } from '../../profiling/update-profiler';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import type {
  LoadedMeshData,
  KTX2TextureDecoder,
  MeshColorArray,
  MeshDataLoader,
  MeshMetadata,
  MeshTextureData,
  MeshViewState,
} from '../../types/mesh';

/** What the loader needs from the scene-loader to do its job. */
export interface MeshLoaderDeps {
  /** Store handle, for resolving `array_ref` targets during decode. */
  zarrStore: zarr.Readable;
  /** Shared registry so an `array_ref` shared with a sibling node decodes once. */
  arrayRefRegistry: ArrayRefRegistry;
  decodeKTX2?: KTX2TextureDecoder;
}

/** The optional arrays, mapped to the attr flag that declares each present. */
const OPTIONAL_ARRAYS = [
  ['normals', 'has_normals'],
  ['colors', 'has_colors'],
  ['scalars', 'has_scalars'],
  ['uvs', 'has_uvs'],
  ['texture', 'has_texture'],
  ['labelOffsets', 'has_labels'],
  ['labelBytes', 'has_labels'],
  ['imageLabelOffsets', 'has_image_labels'],
  ['imageLabelBytes', 'has_image_labels'],
  ['keyOffsets', 'has_keys'],
  ['keyBytes', 'has_keys'],
] as const satisfies readonly (readonly [keyof MeshArrayHandles, keyof MeshMetadata])[];

/**
 * Decoded bytes of one mesh payload — what the monitor reports as `bytesLoaded`
 * (and the resident half of `memoryUsed`).
 *
 * Measured from the materialized arrays, matching the sibling loaders, whose
 * `recordLoadMetrics` charges `output.byteLength` per array: this is DECODED
 * size, not wire size. The two differ (chunks arrive compressed, and a
 * quantized u16 array widens to float32 on decode), and decoded is the honest
 * answer for "how much data does this layer amount to in the viewer". Wire
 * bytes are the Cache tab's job, which measures them at the store boundary
 * where every tier can be attributed.
 *
 * The texture arms are branched exhaustively rather than as
 * `raw`-or-everything-else, so a fourth arm is a compile error here instead of
 * a silently mischarged one:
 *   - `raw` — the materialized surface itself.
 *   - `bitmap` — `w * h * 4`, because an `ImageBitmap` is always 4-channel
 *     8-bit once decoded regardless of the source codec.
 *   - `compressed` — `ceil(w * h * 4 / 3)`: a KTX2 payload stays
 *     GPU-compressed at ~1 byte per texel and the `4/3` covers its mip chain.
 *     Charging it `w * h * 4` like a bitmap would overstate a transcoded
 *     basemap by 3x, which is the whole point of using KTX2. Conservative on
 *     hardware Basis transcodes to an RGB-only target at half a byte per
 *     texel, which is deliberate: the charge stays device-independent and
 *     agrees with the admission budget, which cannot know the GPU either.
 *
 * The two encoded arms use the same figures as the decoded-surface term
 * `preflight.ts` adds for them, so the admission gate and the telemetry cannot
 * disagree about what a texture expands to. This is NOT an equality of totals:
 * preflight is a PEAK ADMISSION number (stored bytes plus what they decode to
 * plus the largest single chunk, all coexisting) and charges every array
 * including the texture handle in its own loop, while this is a RESIDENT one.
 * Only the surface term is shared, and it is the term worth keeping in step —
 * it is the one that varies by three-fold between codecs.
 *
 * Textures dominate this figure when present: an 8192² basemap is 268 MB as a
 * bitmap (89 MB compressed) against a few MB of geometry.
 *
 * Charged PER NODE, so an array two nodes share through an `array_ref` (the two
 * halves of a partitioned globe sharing one basemap) is counted once for each —
 * the same convention the byte budget in `preflight.ts` uses, which likewise
 * follows the ref and charges the target per node. Reporting shared payloads
 * once would need a store-wide ledger that neither the gate nor this telemetry
 * has; the alternative (charging the `(0, k)` stub) understates a gigabyte read
 * as ~48 bytes.
 *
 * Exported for unit test.
 */
export function meshPayloadBytes(data: LoadedMeshData): number {
  let bytes = data.vertices.byteLength + data.faces.byteLength;
  bytes += data.normals?.byteLength ?? 0;
  bytes += data.colors?.byteLength ?? 0;
  bytes += data.scalars?.byteLength ?? 0;
  bytes += data.uvs?.byteLength ?? 0;
  const texture = data.texture;
  if (texture) {
    bytes += textureResidentBytes(texture);
  }
  return bytes;
}

/**
 * Resident bytes of one decoded texture, per arm — see {@link meshPayloadBytes}.
 *
 * MIRROR: the texture term of the byte budget in `preflight.ts`, which for the
 * `ktx2` arm is in turn kept aligned with `luxar/validation/base.py`.
 */
function textureResidentBytes(texture: NonNullable<LoadedMeshData['texture']>): number {
  switch (texture.kind) {
    case 'raw':
      return texture.pixels.byteLength;
    case 'bitmap':
      return texture.width * texture.height * 4;
    case 'compressed':
      return Math.ceil((texture.width * texture.height * 4) / 3);
  }
}

/**
 * Bytes of the per-node projection scratch (`position` / `mask` /
 * `faceScratch`), allocated once per node and held for its whole life.
 *
 * Counted into `memoryUsed` but NOT into `bytesLoaded`: nothing fetched it, but
 * the loader's payload really does keep it resident, and it is not small — the
 * display-space `position` buffer alone is 12 bytes per vertex. This is the
 * mesh counterpart of what the siblings report as their accumulator allocation.
 *
 * Exported for unit test.
 */
export function meshProjectionBytes(data: LoadedMeshData): number {
  const p = data.projection;
  if (!p) return 0;
  return p.position.byteLength + p.mask.byteLength + p.faceScratch.byteLength;
}

export class MeshWholeNodeLoader implements MeshDataLoader {
  private readonly decoder: ArrayDecoder;
  private readonly rangeLoader: RangeLoader;
  private readonly zarrStore: zarr.Readable;
  private readonly decodeKTX2?: KTX2TextureDecoder;

  /** Metadata-only handles, opened once by {@link initialize}. */
  private handles: MeshArrayHandles | null = null;

  /** What Stage 1 established. */
  private preflight: MeshPreflightResult | null = null;

  /**
   * Serializes concurrent calls into {@link initialize} onto one metadata open.
   *
   * `initialize()` used to guard only with `if (this.handles) return` before its
   * awaits — the exact "async initialization race" pattern the repo's own
   * CLAUDE.md calls out. `fetch()`/`load()` were the only caller before
   * {@link runPreflight} made `initialize` reachable from a second, independent
   * entry point, so a concurrent `runPreflight()` + `loadMesh()` opens every
   * `.zarray`/`.zattrs` twice and runs `preflightMesh` twice. Mirrors the
   * `inFlight` latch on {@link load}: cleared once the attempt settles, and only
   * if it is still OURS — a later call may already have installed a fresh
   * attempt after this one settled — so a transient failure is retried rather
   * than cached forever.
   */
  private initInFlight: Promise<void> | null = null;

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
   * Aborts the in-flight reads when the loader is torn down.
   *
   * The generation token below stops a post-dispose completion from
   * repopulating the cache, but on its own it lets the transfer RUN to
   * completion first — for a near-budget mesh that is up to half a gigabyte
   * fetched and decoded for nothing, concurrently with whatever replaced the
   * node. Aborting the reads stops the spend, not just the publish.
   * Replaced (not just aborted) on {@link dispose} so the reuse-after-dispose
   * path starts with a live signal.
   */
  private aborter = new AbortController();

  /**
   * The signal governing the current fetch, sourced by the shared
   * {@link RangeLoader} through its thunk — that loader takes its abort signal
   * from a callback rather than a parameter, so the colour path picks this up
   * without any signature change. Single-flight (`inFlight`) makes one field
   * sufficient.
   *
   * NAMED DIFFERENTLY from its three siblings on purpose. Points, Lines and GSplats
   * each hold `_activeSignal` and wire the identical
   * `setSignalSource(() => this._activeSignal)` one line into their constructors, so
   * this looks like a symmetry break — it is a lifetime difference. Their signal is
   * per-UPDATE: set at the top of every `updateView` and cleared in its `finally`,
   * live whenever the loader is doing anything. Mesh is whole-node resident, so it
   * fetches ONCE and then serves every later `updateView` from `this.data` without
   * any I/O; this field is live only for that single fetch and is `null` during the
   * scrubs that make up almost all of a session. Calling it "active" would claim the
   * opposite of what holds.
   */
  private fetchSignal: AbortSignal | null = null;

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

  /**
   * Monitor telemetry for this node — see the LoaderMonitor section at the
   * bottom of the class for what a whole-node loader can honestly report.
   */
  private readonly metrics: LoaderMetrics;

  /** Monitor listeners, shared implementation with the three sibling facades. */
  private readonly events = new LoaderEventEmitter();

  constructor(
    private readonly path: string,
    private readonly attrs: MeshMetadata,
    private readonly location: zarr.Location<zarr.Readable>,
    deps: MeshLoaderDeps
  ) {
    this.decoder = new ArrayDecoder(deps.arrayRefRegistry);
    this.rangeLoader = new RangeLoader(deps.arrayRefRegistry);
    this.rangeLoader.setSignalSource(() => this.fetchSignal);
    this.zarrStore = deps.zarrStore;
    this.decodeKTX2 = deps.decodeKTX2;
    this.metrics = makeInitialLoaderMetrics('mesh-whole-node', path);
  }

  /**
   * Open metadata handles for every array the attrs declare, then run Stage 1.
   *
   * Single-flight: a concurrent second call joins the first attempt rather than
   * opening the same metadata again (see {@link initInFlight}).
   */
  private async initialize(): Promise<void> {
    if (this.handles) return;
    if (this.initInFlight) return this.initInFlight;

    const mine = this.doInitialize();
    this.initInFlight = mine;
    try {
      await mine;
    } finally {
      if (this.initInFlight === mine) this.initInFlight = null;
    }
  }

  /**
   * Open metadata handles for every array the attrs declare, then run Stage 1.
   *
   * Opening is metadata-only — `zarr.open(..., { kind: 'array' })` reads
   * `.zarray` and `.zattrs` and nothing else — which is what makes it safe to do
   * for *all* declared arrays, including the string-channel CSR pairs v1 never fetches.
   * The budget has to see them to be a budget.
   *
   * Captures `generation` up front and guards BOTH publishes below with it,
   * mirroring {@link load}'s own generation guard. Without this, a `dispose()`
   * racing an in-flight `doInitialize()` — reachable with no `load()` in
   * flight at all, since {@link runPreflight} calls this too, including during
   * a dataset switch — would still let the metadata open complete and publish
   * `this.preflight` / `this.handles` onto a torn-down loader, making the
   * `dispose()` docstring's "a subsequent `loadMesh` re-initializes" false in
   * that window: the stale, non-null handles would look already-initialized.
   */
  private async doInitialize(): Promise<void> {
    const generation = this.generation;
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
        } catch (error) {
          // A genuine not-found means the presence flag disagrees with the store:
          // leave the slot empty and let the preflight turn the flag-with-no-array
          // into an explicit rejection, which reports better than a bare open error.
          // But a TRANSIENT open failure (network blip, abort) must NOT be recorded
          // as deterministic `Validation` — the retry policy skips that kind, so the
          // node would never recover on reconnect. Route it through
          // `classifyLoaderError`, exactly as the required vertices/faces open does.
          if (!zarr.isNotFoundError(error)) {
            throw new LoaderError(classifyLoaderError(error), this.path, error);
          }
        }
      })
    );

    // `storeRoot` is passed so the preflight can follow an `array_ref` to the array
    // whose bytes are actually fetched — the referring array is a `(0, k)` stub, so
    // budgeting the handle alone charges ~48 bytes for a read that can pull gigabytes.
    const preflight = await preflightMesh(
      this.path,
      this.attrs,
      handles,
      zarr.root(this.zarrStore)
    );

    // Only publish if this loader has not been disposed since this attempt began —
    // otherwise a dispose()-then-settle would resurrect `handles`/`preflight` on a
    // torn-down loader (see this method's docstring).
    if (generation === this.generation) {
      this.preflight = preflight;
      this.handles = handles;
    }
  }

  /**
   * Run Stage 1 (the metadata-only preflight) eagerly, and return what it
   * established.
   *
   * Exists for one caller: `MeshProgressiveLoader.assertWithinByteBudget`
   * (`./mesh-progressive-loader.ts`), which needs to charge every level of a
   * reveal ladder against ONE byte budget before any level's chunk data is
   * fetched — each level's own preflight only ever sees its own
   * `MESH_DECODE_BUDGET_BYTES` ceiling, so nothing else adds them up. This is
   * just {@link initialize} (idempotent — a `load()` that runs later reuses the
   * SAME cached `handles`/`preflight`, so calling this first duplicates no
   * request) with its result surfaced instead of stashed on a private field.
   *
   * Fetches no chunk data, same as the preflight it wraps.
   */
  async runPreflight(): Promise<MeshPreflightResult> {
    await this.initialize();
    const pre = this.preflight;
    if (!pre) {
      throw new LoaderError('Unexpected', this.path, new Error('mesh loader not initialized'));
    }
    return pre;
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
      storeRoot,
      signal
    );
    validateMaterializedLength(this.path, 'vertices', vertices.length, nVertices * ndim);

    // `faces` is read RAW, in the store's own dtype, and widened only after the
    // two-sided range check. Routing it through the decoder would hand back a
    // Float32Array, whose 24-bit mantissa cannot represent every index a
    // 2^27-vertex mesh may carry — it would silently round indices above
    // 16,777,216. See validate.ts.
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
        storeRoot,
        signal
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
        `[MeshWholeNodeLoader] ${this.path}`
      )) as MeshColorArray;
      validateMaterializedLength(this.path, 'colors', colors.length, nVertices * colorComponents);
    }

    let uvs: Float32Array | undefined;
    if (handles.uvs) {
      // COORDINATE-encoded on the write side, exactly like `normals`, so it must
      // go through the decoder rather than a raw read — a quantized store holds
      // per-channel u16 codes, not floats.
      const decoded = await this.decoder.decode(
        handles.uvs,
        handles.uvs.attrs as unknown as ArrayMetadata,
        nVertices * 2,
        storeRoot,
        signal
      );
      validateMaterializedLength(this.path, 'uvs', decoded.length, nVertices * 2);
      // Non-finite UVs are refused on both sides. A NaN coordinate does not
      // merely mis-sample its own vertex: it interpolates across every triangle
      // that shares it, so one bad vertex smears an undefined sample over a
      // patch of surface with nothing to attribute it to. The write side rejects
      // it too, but a third-party store never went through the write side.
      for (let i = 0; i < decoded.length; i++) {
        if (!Number.isFinite(decoded[i])) {
          throw new LoaderError(
            'Validation',
            this.path,
            new Error(
              `uvs contains a non-finite value at index ${i} (vertex ` +
                `${Math.floor(i / 2)}). A NaN or infinite texture coordinate ` +
                'interpolates across every triangle sharing that vertex.'
            )
          );
        }
      }
      uvs = decoded;
    }

    let texture: MeshTextureData | undefined;
    if (handles.texture && pre.texture) {
      // Fed the PREFLIGHT's validated declaration, never `this.attrs`: these are
      // the numbers the byte budget admitted the node on, and allocating from a
      // second unvalidated copy is how a ceiling stops meaning anything.
      texture = await decodeMeshTexture(
        this.path,
        handles.texture,
        pre.texture,
        this.decoder,
        storeRoot,
        this.decodeKTX2,
        signal
      );
    }

    let scalars: Float32Array | undefined;
    if (handles.scalars) {
      const decoded = await this.decoder.decode(
        handles.scalars,
        handles.scalars.attrs as unknown as ArrayMetadata,
        nVertices,
        storeRoot,
        signal
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
      uvs,
      texture,
      vertexCount: nVertices,
      faceCount: nFaces,
      ndim,
      // Allocated once here, reused by every projection epoch. This object is what
      // `updateView` hands back for the node's whole life and drops on dispose, so
      // the buffer gets exactly the node's lifetime without a separate cache — the
      // Mesh counterpart of the Points accumulator's reusable target buffers (#1245).
      projection: {
        position: new Float32Array(nVertices * 3),
        displayDimsKey: null,
        mask: new Uint8Array(nVertices),
        faceScratch: new Uint32Array(nFaces * 3),
        fastPathBounds: null,
        fastPathBoundsKey: null,
      },
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
   * One consequence of sharing that fetch: the reads run under the FIRST
   * caller's `signal` combined with the loader-lifetime one ({@link dispose}
   * aborts the latter). A later caller joining an in-flight fetch cannot abort
   * it and will receive the mesh even if its own update was superseded. That is
   * benign here in a way it would not be for the range-query siblings — a
   * superseded caller gets data it no longer needs, never data for the wrong
   * query, because there is only one thing to fetch and it does not depend on the
   * view. The wasted work is bounded by one mesh, once per loader.
   *
   * The combined signal reaches EVERY read: the raw `faces` read directly, the
   * decoder-routed arrays through {@link ArrayDecoder.decode}'s signal
   * parameter, and the shared colour path through the {@link RangeLoader}
   * signal-source thunk wired in the constructor.
   */
  private load(signal?: AbortSignal): Promise<LoadedMeshData> {
    if (this.data) return Promise.resolve(this.data);
    if (this.inFlight) return this.inFlight;

    const generation = this.generation;
    const scope = combineSignals(signal, this.aborter.signal);
    const fetchSignal = scope?.signal ?? this.aborter.signal;
    this.fetchSignal = fetchSignal;
    // Wall-clock start of the ONE fetch this loader makes, for `avgLoadTime`.
    // Taken here rather than inside `fetch()` so it spans the metadata open and
    // preflight too — the panel's load latency should be what the user waited
    // for, not just the chunk reads.
    const startedAt = Date.now();
    const mine: Promise<LoadedMeshData> = this.fetch(fetchSignal)
      .then((data) => {
        // Only publish if this loader has not been disposed since the fetch began.
        // The caller still receives the data — it is view-independent, so it is not
        // wrong, just unwanted — but the loader does not retain it.
        const retained = generation === this.generation;
        if (retained) this.data = data;
        // Counted either way (the bytes really were fetched and decoded), but a
        // payload the loader did not retain must not be reported as resident
        // memory — see `recordLoad`.
        this.recordLoad(data, startedAt, retained);
        return data;
      })
      .catch((error: unknown) => {
        this.recordError(error);
        throw error;
      })
      .finally(() => {
        scope?.dispose();
        // Only clear what is still OURS — a dispose-then-reload may have installed
        // a replacement fetch's signal before this stale completion lands.
        if (this.fetchSignal === fetchSignal) this.fetchSignal = null;
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
   * {@link updateView}, plus whether the mesh was already in hand.
   *
   * Exists for one caller: `MeshProgressiveLoader` (`./mesh-progressive-loader`),
   * whose streaming loop
   * asks each level "was that cheap?" to decide whether to keep going this pass
   * or leave the rest to a later one (`streaming-policy.ts`). The three sibling
   * progressive loaders call the identically named method on their spatial-index
   * sub-loaders, so the ladder loop is the same shape for all four types.
   *
   * `allResident` reads the loader's OWN decode cache rather than the chunk
   * cache the siblings report, because that is where the cost actually is here:
   * a whole-node level either has been fetched and decoded (free to re-serve) or
   * has not (a full network read). Sampled BEFORE the await, so a level that this
   * very call fetches reports `false` — reporting the post-fetch state would say
   * "resident" for every level and defeat the refine pass's stop rule.
   */
  async updateViewWithResidency(
    viewState: MeshViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<{ data: LoadedMeshData; allResident: boolean }> {
    const allResident = this.data !== null;
    const data = await this.updateView(viewState, session, signal);
    return { data, allResident };
  }

  // ────────────────────────────────────────────────────────────────────
  // LoaderMonitor surface
  //
  // The same four methods the three spatial-index facades expose, so a mesh
  // node passes the duck-typed shape check in `connect-loader-to-monitor.ts`
  // and reaches the data-loading monitor. Without them that check skipped mesh
  // SILENTLY — the panel had no mesh loader row, no mesh bytes in its
  // loader-memory total, and no mesh load in its loads/bandwidth windows or in
  // the advisor's slow-load detection, while still showing mesh nodes in its
  // scene-graph tree. Nothing reported a problem; mesh was simply absent.
  //
  // What a whole-node loader can honestly report differs from its siblings, and
  // the difference is the point rather than a gap:
  //   - `queries` / `avgQueryTime` / `spatialIndex` stay zero/absent. There is
  //     no spatial index and no per-slice range query here; a view change
  //     re-serves the resident mesh (see `updateView`). Counting `updateView`
  //     calls as queries would feed ~0 ms samples for work that never touched
  //     the store into the panel's QUERY SPEED average, flattering it — and
  //     would make the compact badge claim spatial-index streaming for a scene
  //     that streams nothing (`metrics/global-stats.ts`, `isSpatialType`).
  //   - `loads` / `bytesLoaded` / `avgLoadTime` cover the ONE fetch this loader
  //     makes (one per level on a reveal ladder, rolled up by
  //     `ProgressiveMonitorAdapter`).
  //   - `elementsLoaded` counts TRIANGLES, the drawn-primitive convention the
  //     whole monitor uses for mesh.
  //   - `memoryUsed` is the resident payload plus the per-node projection
  //     scratch — the mesh counterpart of the siblings' accumulator allocation.
  // ────────────────────────────────────────────────────────────────────

  addEventListener(listener: MonitorEventListener): void {
    this.events.add(listener);
  }

  removeEventListener(listener: MonitorEventListener): void {
    this.events.remove(listener);
  }

  getMetrics(): LoaderMetrics {
    // Copied, like the siblings: the monitor holds snapshots per tick and must
    // not observe later mutation of a record it already recorded.
    return { ...this.metrics };
  }

  /**
   * Always empty: a whole-node loader runs no spatial queries, so none can be
   * in flight. The in-flight FETCH is reported through `loads` instead.
   */
  getActiveQueries(): QueryInfo[] {
    return [];
  }

  /**
   * Record how many of this node's triangles the current slice indexes.
   *
   * Called by `commit-mesh-geometry.ts` from the same place it stamps
   * `userData.visibleTriangleCount`, because that count is produced DOWNSTREAM
   * of the loader: projection decides which faces the index buffer receives,
   * and the loader (which holds the whole mesh either way) cannot know it. The
   * three sibling loaders set `visibleElements` themselves for the opposite
   * reason — for them the query result IS the visible set.
   */
  recordVisibleElements(triangles: number): void {
    this.metrics.visibleElements = triangles;
  }

  /**
   * Fold one completed fetch into the metrics and emit the monitor `load` event
   * that feeds the panel's load-rate and bandwidth windows.
   *
   * `retained` is false when the loader was disposed while the fetch was in
   * flight. The cumulative counters still take it — those bytes were really
   * spent — but `memoryUsed` is a LIVE footprint, and the payload was dropped
   * rather than published, so claiming it would leave the panel reporting
   * resident memory for a torn-down node.
   */
  private recordLoad(data: LoadedMeshData, startedAt: number, retained: boolean): void {
    const bytes = meshPayloadBytes(data);
    const loadTime = Date.now() - startedAt;
    // Triangles, not vertices — `elementsLoaded` is the geometry-neutral
    // throughput counter and mesh counts its drawn primitive everywhere.
    recordLoadEvent(this.metrics, data.faceCount, bytes, loadTime);
    this.metrics.memoryUsed = retained ? bytes + meshProjectionBytes(data) : 0;
    this.events.emit({
      type: 'load',
      loader: 'mesh-whole-node',
      timestamp: Date.now(),
      data: {
        path: this.path,
        elements: data.faceCount,
        memory: bytes,
        latency: loadTime,
      },
    });
  }

  /**
   * Record a failed fetch — EXCEPT a deliberate abort.
   *
   * A dataset switch or a dispose during load aborts the in-flight reads
   * ({@link dispose}), which is a control path, not a failure: counting it
   * would raise the advisor's error-rate recommendation every time the user
   * switches scenes. Same exclusion the sibling loaders apply through
   * {@link isAbortError}.
   */
  private recordError(error: unknown): void {
    if (isAbortError(error)) return;
    this.metrics.errors += 1;
    this.events.emit({
      type: 'error',
      loader: 'mesh-whole-node',
      timestamp: Date.now(),
      data: { path: this.path, error: String(error) },
    });
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
    // Abort the in-flight reads, not just the publish: the generation bump
    // alone lets a near-budget transfer run to completion for a loader nothing
    // will ever read. A fresh controller replaces the tripped one so the
    // re-initialize path below starts with a live signal.
    this.aborter.abort();
    this.aborter = new AbortController();
    // An `ImageBitmap` holds a decoded surface OUTSIDE the JS heap, so dropping
    // the last reference does not free it — the spec requires an explicit
    // `close()`. A 2048x1024 basemap is 8 MB of native memory per node, and a
    // dataset switch disposes every node at once, so leaking it is how a few
    // switches turn into hundreds of megabytes the GC cannot reclaim and no heap
    // profiler attributes to us.
    if (this.data?.texture?.kind === 'bitmap') {
      this.data.texture.bitmap.close();
    }
    this.handles = null;
    this.preflight = null;
    this.data = null;
    this.inFlight = null;
    this.initInFlight = null;
    // The LIVE monitor figures only. `memoryUsed` and `visibleElements`
    // describe what the loader holds and shows RIGHT NOW, and it now holds and
    // shows nothing; the cumulative counters (`loads` / `bytesLoaded` /
    // `errors`) are session history and stay.
    this.metrics.memoryUsed = 0;
    this.metrics.visibleElements = 0;
    // Listeners are deliberately NOT dropped here, unlike the sibling facades:
    // this dispose is documented as state-clearing rather than terminal (a
    // later `loadMesh` re-initializes), and clearing them would leave that
    // re-load invisible to the monitor it is still connected to. Teardown
    // disconnects the loader from the monitor explicitly instead.
  }
}
