/**
 * Progressive Mesh loader for reveal-ladder (multi-additive-LOD) nodes.
 *
 * Wraps N {@link MeshWholeNodeLoader} instances — one per `additive_<i>`
 * subgroup — behind the single {@link MeshDataLoader} interface, so the scene
 * loader, the commit path and the Layers panel see one mesh node that gains
 * triangles over time. The fourth member of the progressive-loader family
 * (`points-` / `lines-` / `gsplats-progressive-loader.ts`).
 *
 * ## What a mesh ladder IS, and why the prefix is presentable
 *
 * A prefix of an arbitrary index buffer is a surface full of holes, not a
 * coarser one — which is why mesh has no *level-of-detail* additive ladder and
 * never will (`docs/specs/MESH_NODE_SPEC.md` §9). What it has is a REVEAL: the
 * writer orders faces by best-first growth through face adjacency keyed on
 * radius, so **every prefix is ONE connected patch**. The renderer does nothing
 * special; each level is just more triangles. That is the whole design —
 * `MESH_ADDITIVE_METHODS` on the Python side admits `radial` and nothing else
 * for exactly this reason.
 *
 * Connectivity is the guarantee; "grows outward from the centre" is what it
 * LOOKS like only where the radius actually varies over the surface — an
 * isosurface with depth structure, the target case. On a shell at near-constant
 * radius (a sphere, a membrane) the radius carries almost no information, the
 * frontier admits a whole radius class at once, and the half-revealed surface
 * reads as a sieve filling in rather than a cap spreading. Observed on an
 * icosphere, not inferred. Still one connected patch, still far better than an
 * arbitrary order — but worth knowing before promising a user a growing cap.
 *
 * ## Why this loader is half the size of its three siblings
 *
 * They are per-slice STREAMING ladders: their sub-loaders answer a spatial range
 * query, so a slice move genuinely changes what each level holds. That forces
 * the whole apparatus this file does not have — a view-state equality check, a
 * reset generation, a departure-store/restore against the `SliceCache`, and a
 * per-view concat.
 *
 * A mesh is whole-node resident (spec §7): every level fetches its faces ONCE
 * and serves every later `updateView` from that decode, so **the ladder is
 * view-independent**. What is loaded depends only on how many levels have
 * arrived; what is *drawn* depends on the slice, and that is decided downstream
 * in `projection.ts` exactly as it is for an unladdered mesh.
 *
 * Two consequences worth stating, because both are load-bearing:
 *
 * 1. **No `SliceCache`.** Each sub-loader's own cached decode is the cache, and
 *    it lasts the node's lifetime. A whole-ladder S-cache entry keyed by slice
 *    would store the same bytes under every key.
 * 2. **The concat memo is keyed on level count alone**, so a slice move returns
 *    the SAME object — and with it the same loader-owned projection scratch
 *    (#1245). Had the ladder reset per view like its siblings, every scrub frame
 *    would have reallocated and re-uploaded the whole vertex buffer, which is
 *    precisely the regression that scratch exists to prevent.
 *
 * @module data/mesh/mesh-progressive-loader
 */

import type {
  LoadedMeshData,
  MeshColorArray,
  MeshDataLoader,
  MeshViewState,
} from '../../types/mesh';
import {
  meshPayloadBytes,
  meshProjectionBytes,
  type MeshWholeNodeLoader,
} from './mesh-whole-node-loader';
import type { MeshPreflightResult } from './preflight';
import type { UpdateSession } from '../../profiling/update-profiler';
import { concatRequiredField } from '../loaders/progressive/concat-helpers';
import { planLadderRollback } from '../loaders/progressive/pass-rollback';
import { measureLodBytes } from '../loaders/progressive/slice-cache-helper';
import {
  ladderResidentBytes,
  type LadderResidency,
} from '../scene-loader/progressive/residency-budget';
import {
  classifyStreamingPass,
  shouldStopBeforeLevel,
  shouldStopAfterLevel,
} from '../loaders/progressive/streaming-policy';
import { assertColorLayout } from '../loaders';
import { MESH_DECODE_BUDGET_BYTES } from '../../config/constants';
import { LoaderError } from '../scene-loader/nodes/load-leaf-error-dispatch';
import { log, Modules, LogEmoji } from '../../utils/log';
import { timeLodStageWithResult } from '../scene-loader/lod-load-stats';
import { ProgressiveMonitorAdapter } from '../loaders/progressive-monitor-adapter';
import type {
  LoaderMetrics,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';

/** The empty payload a ladder with no committed level hands back. */
function emptyMeshData(): LoadedMeshData {
  return {
    vertices: new Float32Array(0),
    faces: new Uint32Array(0),
    normals: null,
    colors: null,
    vertexCount: 0,
    faceCount: 0,
    ndim: 3,
  };
}

/**
 * Reject a ladder whose levels disagree about whether an optional array exists.
 *
 * The three sibling loaders are LENIENT here — a missing level's colours are
 * filled with white — and they are right to be, because their levels answer a
 * spatial range query and a level legitimately contributes zero rows to a given
 * slice. A mesh ladder has no such case: every level is whole-node resident and
 * the levels are one source mesh partitioned by face, so `has_normals` /
 * `has_colors` / `has_scalars` are properties of that ONE mesh. A disagreement
 * is a malformed store.
 *
 * It also cannot be papered over the way colours can. `normal` and `aScalar` are
 * bound to the geometry once, at creation, from the node's metadata, and
 * `updateMeshGeometry` may never add or remove an attribute afterwards (the
 * WebGPU backend bakes the vertex layout into its pipeline). So a level that
 * omitted normals would leave the PREVIOUS level's normals bound at the wrong
 * length — stale values silently indexed past their end — and the only safe
 * fill, a zero vector, denormalizes the lighting calculation into black. Failing loudly is
 * the honest option.
 */
function assertUniformPresence(
  parts: LoadedMeshData[],
  label: string,
  has: (p: LoadedMeshData) => boolean
): void {
  const first = has(parts[0]);
  for (const [levelIdx, part] of parts.entries()) {
    if (has(part) !== first) {
      throw new Error(
        `concatenateMeshData: LOD level ${levelIdx} ${has(part) ? 'carries' : 'omits'} ` +
          `'${label}' but level 0 ${first ? 'carries' : 'omits'} it — every level of a ` +
          'mesh reveal ladder is a face-partition of one source mesh, so the optional ' +
          'arrays are all-or-nothing across the ladder.'
      );
    }
  }
}

/**
 * Concatenate per-level `LoadedMeshData` into the revealed prefix.
 *
 * Face indices are LOCAL to each level's own vertex array (the writer's
 * `split_mesh_by_faces` renumbers every part), so they are offset-adjusted by
 * the cumulative vertex count — the same value-adding loop
 * `concatenateLinesData` uses for segment indices, and the reason faces cannot
 * go through {@link concatRequiredField}, which `.set()`s verbatim. Using the
 * helper would produce a plausible-looking garbage surface rather than a crash.
 */
export function concatenateMeshData(parts: LoadedMeshData[]): LoadedMeshData {
  if (parts.length === 0) return emptyMeshData();
  // One level: hand back its payload unchanged, including its own projection
  // scratch — allocating a copy would double the memory of the first-paint
  // state for nothing.
  if (parts.length === 1) return parts[0];

  const ndim = parts[0].ndim;
  // `ndim` strides the vertex concat, so levels disagreeing about it would
  // mis-stride every vertex after the first level — silent corruption, not a
  // crash. Same fail-fast contract as `concatenateLinesData`'s ndim guard
  // (Points is immune to this because it pre-projects to stride 3; mesh, like
  // lines, carries full nD vertices).
  for (const part of parts) {
    if (part.ndim !== ndim) {
      throw new Error(
        'concatenateMeshData: mixed dimensionality across LOD levels ' +
          `(ndim ${part.ndim} vs ${ndim}) — ladder levels must share the dataset ` +
          'dimensionality.'
      );
    }
  }
  assertUniformPresence(parts, 'normals', (p) => p.normals !== null);
  assertUniformPresence(parts, 'colors', (p) => p.colors !== null);
  assertUniformPresence(parts, 'scalars', (p) => p.scalars !== undefined);

  const colorK = parts[0].colorComponents ?? 3;
  for (const [levelIdx, part] of parts.entries()) {
    if (part.colors) {
      // A level can carry an RGBA buffer while omitting `colorComponents` (which
      // defaults to 3): that satisfies the `count·3` minimum yet mis-strides
      // every vertex after it. Assert each level against its OWN declaration
      // before allocating, then compare declarations across levels.
      assertColorLayout(
        part.colors,
        part.vertexCount,
        part.colorComponents ?? 3,
        `concatenateMeshData (LOD level ${levelIdx})`
      );
      if ((part.colorComponents ?? 3) !== colorK) {
        throw new Error(
          'concatenateMeshData: mixed color layouts across LOD levels ' +
            `(level ${levelIdx}: ${part.colorComponents ?? 3} components vs ${colorK}) — ` +
            'ladder levels must share the color layout.'
        );
      }
    }
  }

  const totalVertices = parts.reduce((s, p) => s + p.vertexCount, 0);
  const totalFaces = parts.reduce((s, p) => s + p.faceCount, 0);
  const vertexCountOf = (p: LoadedMeshData) => p.vertexCount;

  const vertices = concatRequiredField(parts, (p) => p.vertices, vertexCountOf, ndim, 'vertices');
  const normals =
    parts[0].normals !== null
      ? concatRequiredField(parts, (p) => p.normals as Float32Array, vertexCountOf, 3, 'normals')
      : null;
  const colors =
    parts[0].colors !== null
      ? (concatRequiredField(
          parts,
          (p) => p.colors as MeshColorArray,
          vertexCountOf,
          colorK,
          'colors'
        ) as MeshColorArray)
      : null;
  const scalars =
    parts[0].scalars !== undefined
      ? concatRequiredField(parts, (p) => p.scalars as Float32Array, vertexCountOf, 1, 'scalars')
      : undefined;

  // Faces: value-adding, not a verbatim copy. Level k's index `j` addresses
  // vertex `j` of ITS OWN array, which is vertex `vertexOffset + j` here.
  const faces = new Uint32Array(totalFaces * 3);
  let vertexOffset = 0;
  let faceOffset = 0;
  for (const part of parts) {
    for (let i = 0; i < part.faces.length; i++) {
      faces[faceOffset * 3 + i] = part.faces[i] + vertexOffset;
    }
    vertexOffset += part.vertexCount;
    faceOffset += part.faceCount;
  }

  const result: LoadedMeshData = {
    vertices,
    faces,
    normals,
    colors,
    ...(colors ? { colorComponents: colorK } : {}),
    vertexCount: totalVertices,
    faceCount: totalFaces,
    ndim,
    // A fresh scratch sized to the revealed prefix. Reallocated once per LEVEL
    // (the prefix genuinely grew), never per slice move — the memo below returns
    // this same object for every later view, which is what keeps the #1245
    // no-reallocation-per-scrub property that motivated the buffer.
    projection: {
      position: new Float32Array(totalVertices * 3),
      displayDimsKey: null,
      mask: new Uint8Array(totalVertices),
      faceScratch: new Uint32Array(totalFaces * 3),
      fastPathBounds: null,
      fastPathBoundsKey: null,
    },
  };
  if (scalars !== undefined) result.scalars = scalars;
  return result;
}

/**
 * Progressive Mesh loader — the reveal ladder's composite.
 */
export class MeshProgressiveLoader implements MeshDataLoader {
  private lodLoaders: MeshWholeNodeLoader[];
  private loadedLODs: LoadedMeshData[] = [];
  private readonly nLods: number;
  private readonly path: string;
  private _initialLoadDone = false;
  private _disposed = false;
  /**
   * Latched once {@link assertWithinByteBudget} has fully accounted the ladder
   * and admitted it. Set ONLY on that success path — never on a rejection — so
   * a rejection is never cached as a pass: a live instance genuinely
   * re-preflights on the next `updateView` (a failed `initialize()` caches
   * nothing), and a disposed one is terminal and simply returns from the gate's
   * first line. The one verdict that CANNOT come out differently is latched
   * separately, on {@link _budgetRefusal}, so it is not re-derived either.
   */
  private _budgetChecked = false;
  /**
   * The sticky AGGREGATE-over-budget refusal, once thrown — deliberately the
   * mirror image of {@link _budgetChecked} above, which is never set on a
   * rejection.
   *
   * The AGGREGATE comparison this field guards is DETERMINISTIC. By the time
   * `assertWithinByteBudget` reaches the sum, every level's
   * `MeshPreflightResult` came from an `initialize()` that already succeeded
   * and is cached (`MeshWholeNodeLoader.doInitialize` never re-runs once
   * `this.handles` is set) — there is no I/O left to vary, so re-running the
   * gate on a later `updateView` can only ever repeat the same verdict.
   *
   * Recomputing it anyway is actively harmful: without this latch,
   * `hasMoreLODs` stays `true` forever (the loaded-level count never grows on
   * a ladder that is never allowed to fetch a single level), so
   * `queue-next.ts` keeps scheduling `runMeshRefinement` on every slice
   * scrub, which burns `MAX_CONSECUTIVE_REFINEMENT_FAILURES` refinement
   * passes per scrub and toasts "Refinement failed … — showing a partial
   * surface" — false, since zero triangles were ever committed.
   *
   * So the refusal is computed once, cached here, and every later call
   * rethrows the SAME `LoaderError` object with no further `runPreflight()`
   * calls. `hasMoreLODs` also reads this field directly (see above) and
   * reports `false` once it is set, which is what actually removes the dead
   * node from `queueNext`'s refinement loop rather than merely making its
   * gate cheap to re-fail.
   *
   * A level's OWN `runPreflight()` rejection stays unlatched — and NOT because
   * it is always transient. `preflightMesh` raises a deterministic
   * `LoaderError('Validation')` for a genuinely malformed store, so some of
   * those rejections do repeat forever. The reason is that the error KIND
   * cannot be trusted to separate the two here, and it fails in both
   * directions. Over-inclusive: a 404 on `additive_2/normals` while a store is
   * still being written arrives as `Validation` BY DESIGN —
   * `mesh-whole-node-loader.ts`'s optional-array open catch reads a zarr
   * not-found as "the presence flag disagrees with the store", leaves the slot
   * empty, and `preflightMesh`'s flag-with-no-array check then `rejectMesh`es
   * it. A genuinely transient outage lands in that same branch, because
   * `MultiLevelCachingStore.get` returns `undefined` for a retry-exhausted
   * `NetworkError` to keep zarrita's "key missing" contract — so an offline
   * blip on an optional array is indistinguishable here from a store that
   * really lacks it. Under-inclusive: an
   * absent REQUIRED `vertices`/`faces` array throws zarrita's `NotFoundError`,
   * which `classifyLoaderError` matches nowhere and files as `Unexpected`. So
   * latching by kind would strand nodes that the failed-loads banner's manual
   * Retry and the LOD registry's not-ready self-heal — both of which
   * deliberately IGNORE the kind — recover the moment the store or the
   * connection is fixed, and would treat two 404s a few array names apart
   * oppositely.
   *
   * The residual cost is real and accepted: while no level has committed, the
   * refinement loop keeps re-firing on a deterministically broken level, and
   * its "showing a partial surface" toast is inaccurate there too. Separating
   * the two honestly would mean marking determinism where each rejection is
   * CONSTRUCTED (in `preflight.ts` and the open catches) rather than inferring
   * it from the kind after the fact — deliberately out of scope here.
   */
  private _budgetRefusal: LoaderError | null = null;
  /**
   * Memoized concatenation, keyed on the loaded LEVEL COUNT alone.
   *
   * Its siblings additionally key on a reset generation because their ladders
   * are rebuilt per view; a mesh ladder is view-independent (see the module
   * docstring), so an unchanged level count returns the same object across any
   * number of slice moves. That identity is what the commit pipeline reads to
   * skip no-op re-commits, and what keeps the projection scratch alive.
   */
  private _concatCache: { lodCount: number; result: LoadedMeshData } | null = null;
  private _loadedLODCount = 0;
  // Logical level and retained-payload watermarks at pass start. A successful
  // concat folds every loaded level into one cumulative payload, so these
  // counts intentionally differ.
  private _levelsAtPassStart = 0;
  private _payloadsAtPassStart = 0;
  // A completed pass can still fail after loading, during projection or commit.
  // Keep it schedulable for one retry even though the ladder cursor is full.
  private _retryFoldedPass = false;
  /** Per-pass playback budget from the CURRENT `updateView`; null outside playback. */
  private _frameBudgetMs: number | null = null;
  private _lastAllResident = true;
  /** Monitor telemetry, rolled up over the levels — see the surface below. */
  private readonly monitor: ProgressiveMonitorAdapter;
  /**
   * Triangles the current slice indexes, as reported by the commit
   * (`recordVisibleElements`). Node-level, so it cannot come from the per-level
   * roll-up — see {@link getMetrics}.
   */
  private _visibleTriangles = 0;

  constructor(lodLoaders: MeshWholeNodeLoader[], nLods: number, path: string) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
    this.path = path;
    this.monitor = new ProgressiveMonitorAdapter(() => this.lodLoaders, path, 'mesh-whole-node');
  }

  get hasMoreLODs(): boolean {
    // A disposed loader has its work state cleared; report no further work so a
    // refinement loop holding a stale reference stops instead of indexing into
    // the now-empty `lodLoaders`. Mirrors the sibling loaders.
    if (this._disposed) return false;
    // A latched aggregate-over-budget refusal means this ladder will never load
    // another level (see `_budgetRefusal`'s docstring) — report no further work
    // so the refinement loop leaves the node instead of re-failing it forever.
    if (this._budgetRefusal) return false;
    if (this._retryFoldedPass) return true;
    // While a playback frame budget is active the budgeted prefix IS the target:
    // no background refinement between animation ticks.
    if (this._frameBudgetMs !== null) return false;
    return this._loadedLODCount < this.nLods;
  }

  get loadedLODCount(): number {
    return this._loadedLODCount;
  }

  /**
   * Measured footprint of the loaded ladder, for the shared sweep residency
   * budget (`scene-loader/progressive/residency-budget`). Sums real
   * `byteLength`s rather than modelling a per-element cost, so it stays correct
   * as payload columns come and go. Rung count is reported alongside so the
   * budget can estimate the next rung without needing per-rung sizes.
   */
  ladderResidency(): LadderResidency {
    return {
      residentBytes: measureLodBytes(this.loadedLODs),
      loadedRungs: this.loadedLODCount,
      // Mesh is NOT element-texture backed — it has no per-element row and no
      // capacity clamp, which is why `element-texture-layout` defines no mesh
      // layout. Zero here is the honest value, not an omission: its decoded
      // payload IS its footprint. Stated explicitly so mesh is visibly exempt
      // rather than looking like a geometry someone forgot to wire up.
      elementCount: 0,
      bytesPerElement: 0,
    };
  }

  get totalLODCount(): number {
    return this.nLods;
  }

  /**
   * Discard the levels the current pass appended, restoring the ladder to the
   * prefix the pass started from. Called by the main-update and refinement catches; see
   * `../loaders/progressive/pass-rollback` for why a failed commit must not
   * leave the cursor advanced.
   *
   * @returns Levels discarded (0 when the pass appended none).
   */
  rollbackToPassStart(): number {
    const plan = planLadderRollback({
      loadedLevelCount: this._loadedLODCount,
      levelsAtPassStart: this._levelsAtPassStart,
      concatCacheLodCount: this._concatCache?.lodCount ?? null,
      retainedPayloadCount: this.loadedLODs.length,
      payloadsAtPassStart: this._payloadsAtPassStart,
      restoredFullLadderAtPassStart: false,
      totalLevelCount: this.nLods,
    });
    if (plan.action === 'retry-folded-pass') this._retryFoldedPass = true;
    if (plan.action === 'truncate') {
      this.loadedLODs.length = this._payloadsAtPassStart;
      this._loadedLODCount = plan.keep;
      this._retryFoldedPass = false;
      if (plan.invalidateConcatCache) this._concatCache = null;
    }
    return plan.dropped;
  }

  get lastAllResident(): boolean {
    return this._lastAllResident;
  }

  /**
   * Always `null` — a mesh reveal ladder carries no energy stamps, by
   * construction at three independent layers.
   *
   * The stamps exist so the display gate can release an upgrade early, and so
   * `energyCompensation` can brighten an incomplete EMISSIVE ladder by `1/e(k)`:
   * a coarse prefix of a splat cloud is a dim version of the whole, and dividing
   * by the committed energy fraction restores its brightness. A reveal prefix is
   * nothing of the kind — it is a PARTIAL OBJECT AT FULL BRIGHTNESS — so the same
   * multiplier would blow out the first shell by ~1/e and then fade it as the
   * surface completes: the exact inverse of growing in. `MESH_NODE_SPEC.md` §9.1
   * states the rule; the Python `add_mesh` refuses `lod_stats` energy keys, the
   * factory reads no energy table, and this getter closes the loop.
   *
   * It must EXIST and return `null` rather than be absent, and the difference is
   * not cosmetic: `stampLadderComplete` probes
   * `'committedEnergyFraction' in loader` and takes the NON-progressive branch
   * when the getter is missing — stamping a half-revealed mesh
   * `committedEnergyFraction: 1`, i.e. "all of it is on screen". Returning `null`
   * makes the stamp absent instead, which is what the display gate reads as
   * "unstamped, fall back to committed-count crossover".
   */
  get committedEnergyFraction(): number | null {
    return null;
  }

  loadMesh(viewState: MeshViewState, session?: UpdateSession): Promise<LoadedMeshData> {
    return this.updateView(viewState, session);
  }

  /**
   * Charge the ladder's aggregate byte budget ONCE, at the ladder's first
   * `updateView` — before any level's chunks are fetched.
   *
   * ## Why here, and not at ladder-construction time
   *
   * The obvious home for this aggregate is where the ladder is BUILT —
   * `createProgressiveMeshLoader` (`../scene-loader/loaders/loader-factory.ts`),
   * during `loadMeshNodeCheap` — and that spot is unusable, which is worth
   * recording because it is the first place a reader will look for it. The
   * cheap half runs OUTSIDE `loadMeshNode`'s try/placeholder-attach and before
   * `registerMeshLoader`, so a rejection thrown from there has nowhere to go:
   * no placeholder to mark failed, no `recordFailure`, no retry, no monitor
   * banner — a transient blip on one level's metadata open would permanently
   * lose the whole node, silently (`reportLoadOutcome` still logs success).
   * Swallowing that rejection instead only trades it for the other failure:
   * `retryFailedLoader` reuses the existing loader rather than re-entering the
   * factory, so a swallowed level's bytes would go uncounted forever, the
   * level would then load fine on retry, and the ladder that motivates this
   * whole check would sail through over budget with nothing left to refuse it.
   *
   * Both failure modes trace to the same cause: a check with no containment
   * around it. A leaf's own byte budget is enforced inside
   * `MeshWholeNodeLoader.fetch()`, i.e. inside `loadMeshNodeExpensive`'s try —
   * so charging the LADDER at the equivalent point, its own first load, gives
   * it the identical containment (failure recorded, banner shown, siblings
   * unaffected) as well as the identical ceiling: ladder ≡ leaf in both
   * respects. Retryability carries over for a level's OWN rejection; the
   * aggregate over-budget verdict is deliberately cached and rethrown instead,
   * since no retry can make the sum fit (see {@link _budgetRefusal}). It is
   * still strictly before any chunk is fetched — `runPreflight()` only opens
   * metadata — so the two-stage gate's "refuse before allocation" property
   * survives the move unchanged.
   *
   * ## The first-paint tradeoff this creates, stated honestly
   *
   * Gating on EVERY level's metadata before level 0 is fetched has a real cost,
   * and it is worth stating rather than leaving implicit:
   *
   * 1. Any single level's `runPreflight()` rejection now costs the WHOLE first
   *    paint. Before this change, levels `0..k-1` painted and the ladder simply
   *    stalled at the bad level `k` with a "partial surface" toast; now nothing
   *    paints until every level's metadata has opened successfully.
   * 2. On a store WITHOUT consolidated metadata, the added latency is real, not
   *    theoretical. A 6-level ladder with normals and colors is ~30 arrays,
   *    i.e. ~60 metadata objects (`.zarray` + `.zattrs` each) to open before
   *    level 0 can even start fetching chunks. Behind a browser's ~6-connection
   *    limit per origin, that is several serialized round trips — measurably
   *    slower first paint on exactly the slow link this reveal ladder exists
   *    to serve well.
   *
   * For a Luxar-written store this is close to free: the compiler writes a
   * `.zmetadata` consolidated-metadata document, and `src/data/zarr.ts` wraps
   * every store with `zarrita.withMaybeConsolidatedMetadata`, so every
   * `zarr.open(..., { kind: 'array' })` above is served from an in-memory
   * document instead of a network round trip. The cost above is real only for
   * an arbitrary `?src=` store that omits it.
   *
   * An alternative was considered and NOT taken: charge levels `1..N-1` only
   * AFTER level 0 has been committed, so a slow/failing deeper level would
   * never block first paint (level 0 already has its own leaf-sized
   * `runPreflight()`, so the pre-refusal peak would still be bounded by one
   * leaf's ceiling). Rejected here because it weakens the property this gate
   * is FOR — "refuse the whole ladder before any chunk is fetched" — down to
   * "refuse before the second chunk," and because the common (consolidated)
   * case already pays nothing for gating everything up front. Not
   * implemented.
   *
   * ## Why the sum is the right quantity
   *
   * `MESH_DECODE_BUDGET_BYTES` is a per-NODE ceiling, and a reveal ladder is
   * one node (`MESH_NODE_SPEC.md` §9.1): this class concatenates every
   * level's vertices/faces/normals/colors/scalars into ONE committed buffer
   * set (`concatenateMeshData` above), and all of it stays resident for the
   * node's whole life. Each level's own `runPreflight()` only ever sees its
   * own ceiling, with no knowledge of its siblings — so without this, a
   * ladder's real ceiling was `nAdditive x budget`: a plain leaf with the
   * ladder's total geometry is refused up front, but the same geometry split
   * into levels sails through, N times over budget, and the tab dies on the
   * concatenated allocation. Summing every level's `accountedBytes` and
   * charging that once against the SAME ceiling is what turns N budgets back
   * into one.
   *
   * The sum overcharges relative to what a single level's own preflight would
   * need to: it includes each level's own largest-chunk term even though the
   * ladder loads levels sequentially (never two chunk buffers alive at once
   * across levels), and an `array_ref` target shared between levels is
   * charged once per referring level rather than once total. In that sense it
   * is a deliberate over-estimate.
   *
   * That does NOT make the charged figure a bound on true peak residency,
   * which is a separate quantity this sum does not track. Per level `i`, the
   * charged term is `stored_i + 4·decoded_i` (plus that level's own
   * `maxChunk_i`, which is transient and never resident). After each successful
   * concat, the level loaders release their decoded payloads and the composite
   * retains only the cumulative payload plus its projection scratch. Tightening
   * the ceiling to a true peak-residency model remains a separate concern.
   *
   * ## Why a level's own rejection propagates, unswallowed
   *
   * `initialize()` routes a transient open failure (a network blip) through
   * `classifyLoaderError` precisely so it stays retryable, and that property
   * has to survive reaching here. Letting the rejection propagate instead of
   * catching it is correct HERE, and only because of where "here" is:
   * `loadMeshNodeExpensive`'s catch records the failure and keeps the blip
   * retryable, and a later retry re-enters `updateView`, which re-runs this
   * gate — `_budgetChecked` is latched only on a fully successful accounting,
   * and the failed level's own `initialize()` cached nothing, so the retry
   * genuinely re-preflights rather than replaying a stale rejection.
   *
   * Propagating is not the same as forgetting, though: the AGGREGATE
   * over-budget refusal below IS latched on the way out and rethrown as the
   * same error object rather than re-derived. A level's own rejection is not —
   * see {@link _budgetRefusal} for why the error kind cannot be trusted to tell
   * a deterministic one from a transient one here.
   */
  private async assertWithinByteBudget(): Promise<void> {
    if (this._budgetChecked || this._disposed) return;
    // A latched refusal is deterministic (see `_budgetRefusal`'s docstring) —
    // rethrow the cached error rather than re-running every level's
    // `runPreflight()` for a verdict that cannot change.
    if (this._budgetRefusal) throw this._budgetRefusal;

    let results: MeshPreflightResult[];
    try {
      results = await Promise.all(this.lodLoaders.map((loader) => loader.runPreflight()));
    } catch (error) {
      // A dispose() racing the check tore the ladder down mid-await, and with the
      // real sub-loader that arrives as a REJECTION rather than a late resolve:
      // `dispose()` disposes every level, and `MeshWholeNodeLoader.dispose()`
      // bumps its generation and nulls `preflight`, so the in-flight
      // `runPreflight()` throws `LoaderError('Unexpected', …, 'mesh loader not
      // initialized')`. Return quietly — the same answer the streaming loop's two
      // `_disposed` re-checks give, so a teardown is not mis-counted as a load
      // failure — a teardown is not a verdict about the store. (This composite's
      // `dispose()` is TERMINAL, unlike `MeshWholeNodeLoader.dispose()`'s
      // documented state-clearing one: `_disposed` is never cleared and this
      // gate returns on its first line forever after, so there is nothing left
      // to latch or to re-check.) Checked FIRST for that reason.
      if (this._disposed) return;
      throw error;
    }

    // The other shape of the same race: a dispose() landing after the preflights
    // RESOLVED. There is nothing left to charge a budget against, and a
    // torn-down ladder must not throw a refusal at a dataset that is already
    // gone — so return quietly, exactly as the catch above does.
    if (this._disposed) return;

    const totalBytes = results.reduce((sum, r) => sum + r.accountedBytes, 0);
    if (totalBytes > MESH_DECODE_BUDGET_BYTES) {
      const mib = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
      const error = new LoaderError(
        'Validation',
        this.path,
        new Error(
          `Mesh reveal ladder's ${this.lodLoaders.length} levels account for ${mib(totalBytes)} ` +
            `combined, over the ${mib(MESH_DECODE_BUDGET_BYTES)} per-node budget. The levels ` +
            "are concatenated into one node's buffers and all stay resident, so they are " +
            'charged together rather than against separate budgets. Decimate the mesh, ' +
            'write fewer levels, or split it across nodes.'
        )
      );
      this._budgetRefusal = error;
      throw error;
    }
    this._budgetChecked = true;
  }

  async updateView(
    viewState: MeshViewState,
    session?: UpdateSession,
    signal?: AbortSignal,
    residencyAllowanceBytes?: number
  ): Promise<LoadedMeshData> {
    // Record the per-pass playback budget FIRST: a pause re-trigger arrives with
    // the same view state and must still clear the budget.
    this._frameBudgetMs = viewState.frameBudgetMs ?? null;
    this._retryFoldedPass = false;
    const budgetDeadline =
      this._frameBudgetMs !== null ? performance.now() + this._frameBudgetMs : null;

    // Charge the ladder's aggregate byte budget before a single level is
    // fetched (see {@link assertWithinByteBudget}). A no-op once admitted.
    await this.assertWithinByteBudget();

    // Background prefetch (shadow) passes only warm caches — the caller discards
    // the return value — so hand back the empty payload instead of the O(N)
    // main-thread concat.
    //
    // UNREACHABLE TODAY, and kept rather than dropped: `SlicePrefetcher.prefetch`
    // enumerates points, lines and gsplats only, so no mesh node is ever shadowed
    // (a mesh is whole-node resident, so there is no next-timepoint slice to warm
    // — the shadow would re-fetch the same bytes into a second loader). It stays
    // because `GeometryDescriptor` hands this factory to the prefetcher for every
    // kind, so the day mesh joins that list the ladder must not answer a shadow
    // pass with a full concat.
    const isPrefetch = viewState.prefetch === true;

    const pass = classifyStreamingPass(budgetDeadline !== null, isPrefetch);
    const startLevel = this._loadedLODCount;
    const residentBytesAtPassStart = ladderResidentBytes(this.ladderResidency());
    this._levelsAtPassStart = startLevel;
    this._payloadsAtPassStart = this.loadedLODs.length;

    for (let level = startLevel; level < this.nLods; level++) {
      // A dispose() racing the awaited level below clears `lodLoaders`, so the
      // next iteration would TypeError — a teardown mis-counted as a refinement
      // failure. Stop streaming instead.
      if (this._disposed) break;
      // Pass-budget guard: playback only guarantees a level for an empty
      // ladder; prefetch retains one-level progress after a restore (#2379).
      if (shouldStopBeforeLevel(pass, level, startLevel, performance.now(), budgetDeadline)) {
        break;
      }
      const t0 = performance.now();
      const { data: lodData, allResident } = await timeLodStageWithResult(
        ({ allResident }) => `additive:mesh:level:${level}:${allResident ? 'resident' : 'miss'}`,
        `additive:mesh:level:${level}:aborted`,
        () => this.lodLoaders[level].updateViewWithResidency(viewState, session, signal)
      );
      const elapsed = performance.now() - t0;
      this._lastAllResident = allResident;

      // Re-checked after the await: a dispose() during the fetch cleared the
      // ladder, and pushing here would resurrect it on a dead loader (and pin
      // the whole prefix's arrays that nothing will read).
      if (this._disposed) break;
      this.loadedLODs.push(lodData);
      this._loadedLODCount++;

      if (!this._initialLoadDone) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.SCENE_LOADER,
          `Mesh reveal ${this.path} LOD ${level}/${this.nLods - 1}: ` +
            `${lodData.faceCount} faces (${elapsed.toFixed(1)}ms${allResident ? '' : ', miss'})`
        );
      }

      const additionalResidentBytes = Math.max(
        0,
        ladderResidentBytes(this.ladderResidency()) - residentBytesAtPassStart
      );
      if (
        shouldStopAfterLevel(
          pass,
          level,
          startLevel,
          allResident,
          elapsed,
          additionalResidentBytes,
          residencyAllowanceBytes
        )
      ) {
        break;
      }
    }

    if (!this._initialLoadDone && startLevel === 0) this._initialLoadDone = true;

    const totalFaces = this.loadedLODs.reduce((s, d) => s + d.faceCount, 0);
    if (this._loadedLODCount < this.nLods) {
      log.info(
        Modules.SCENE_LOADER,
        `Progressive Mesh: ${this._loadedLODCount}/${this.nLods} LODs ` +
          `(${totalFaces} faces) — revealing`
      );
    } else if (startLevel < this.nLods) {
      log.info(
        Modules.SCENE_LOADER,
        `Progressive Mesh: ${this.nLods}/${this.nLods} LODs (${totalFaces} faces) — complete`
      );
    }

    return isPrefetch ? emptyMeshData() : this.concatenateMemoized(session);
  }

  /**
   * Concatenate the loaded levels, memoized on the level count.
   *
   * Returning the SAME object reference when nothing has changed is safe because
   * the result is never mutated downstream (the worker projection's inputs are
   * structured-cloned, not transferred) and is what lets the commit pipeline
   * skip no-op re-commits by identity.
   *
   * No `setPrefixParent` stamp, unlike the three siblings. That lineage exists
   * so the commit layer can recognise a prefix EXTENSION and take the depth-sort
   * append fast path, which is defined over the instanced element buffers mesh
   * does not have — its faces are re-emitted by the projection every epoch.
   */
  private concatenateMemoized(session?: UpdateSession): LoadedMeshData {
    const concatSession = session?.begin('Concatenate LODs');
    try {
      if (this._concatCache && this._concatCache.lodCount === this._loadedLODCount) {
        return this._concatCache.result;
      }
      const result = concatenateMeshData(this.loadedLODs);
      const preservesSinglePayload = this._loadedLODCount === 1 && result === this.loadedLODs[0];
      for (let level = 0; level < this._loadedLODCount; level++) {
        this.lodLoaders[level]?.releaseData(level === 0 && preservesSinglePayload);
      }
      this.loadedLODs = [result];
      this._concatCache = { lodCount: this._loadedLODCount, result };
      return result;
    } finally {
      concatSession?.end();
    }
  }

  // ---- LoaderMonitor surface (delegated to ProgressiveMonitorAdapter) ----
  // Lets `connectLoaderToMonitor` wire the reveal ladder to the data monitor as
  // ONE loader keyed by this node's path — the adapter re-paths each level's
  // events, so the `additive_<i>` sub-loaders stay an implementation detail
  // instead of appearing as N separate rows. Same wiring as the three sibling
  // progressive loaders.

  addEventListener(listener: MonitorEventListener): void {
    this.monitor.addEventListener(listener);
  }

  removeEventListener(listener: MonitorEventListener): void {
    this.monitor.removeEventListener(listener);
  }

  /** Always empty — no level runs spatial queries (see `MeshWholeNodeLoader`). */
  getActiveQueries(): QueryInfo[] {
    return this.monitor.getActiveQueries();
  }

  getMetrics(): LoaderMetrics {
    // `visibleElements` is OVERRIDDEN rather than summed, unlike every other
    // counter here. The commit reports the visible-triangle count to the loader
    // it finds in `userData` — this wrapper, not the levels — so the per-level
    // values are all 0 and their sum would be too. It is a node-level fact
    // anyway: the committed surface is the revealed prefix's concatenation, not
    // a quantity each level owns a share of.
    const metrics = this.monitor.getMetrics();
    const concatMemory = this._concatCache
      ? meshPayloadBytes(this._concatCache.result) + meshProjectionBytes(this._concatCache.result)
      : 0;
    return {
      ...metrics,
      visibleElements: this._visibleTriangles,
      memoryUsed: metrics.memoryUsed + concatMemory,
    };
  }

  /** See `MeshWholeNodeLoader.recordVisibleElements`. */
  recordVisibleElements(triangles: number): void {
    this._visibleTriangles = triangles;
  }

  dispose(): void {
    this._disposed = true;
    if (this._concatCache?.result.texture?.kind === 'bitmap') {
      this._concatCache.result.texture.bitmap.close();
    }
    for (const loader of this.lodLoaders) loader.dispose();
    this.lodLoaders = [];
    this.loadedLODs = [];
    this._loadedLODCount = 0;
    this._retryFoldedPass = false;
    this._concatCache = null;
    // Nothing is on screen for this node any more; the cumulative counters live
    // on the (now-disposed) levels and go with them.
    this._visibleTriangles = 0;
  }
}
