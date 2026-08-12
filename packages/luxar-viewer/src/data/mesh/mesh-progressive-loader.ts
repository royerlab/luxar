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
import type { MeshWholeNodeLoader } from './mesh-whole-node-loader';
import type { UpdateSession } from '../../profiling/update-profiler';
import { concatRequiredField } from '../loaders/progressive/concat-helpers';
import {
  classifyStreamingPass,
  shouldLoadLevel,
  shouldStopAfterLevel,
} from '../loaders/progressive/streaming-policy';
import { assertColorLayout } from '../loaders';
import { log, Modules, LogEmoji } from '../../utils/log';

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
 * fill, a zero vector, denormalizes the headlight into black. Failing loudly is
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
   * Memoized concatenation, keyed on the loaded LEVEL COUNT alone.
   *
   * Its siblings additionally key on a reset generation because their ladders
   * are rebuilt per view; a mesh ladder is view-independent (see the module
   * docstring), so an unchanged level count returns the same object across any
   * number of slice moves. That identity is what the commit pipeline reads to
   * skip no-op re-commits, and what keeps the projection scratch alive.
   */
  private _concatCache: { lodCount: number; result: LoadedMeshData } | null = null;
  /** Per-pass playback budget from the CURRENT `updateView`; null outside playback. */
  private _frameBudgetMs: number | null = null;

  constructor(lodLoaders: MeshWholeNodeLoader[], nLods: number, path: string) {
    this.lodLoaders = lodLoaders;
    this.nLods = nLods;
    this.path = path;
  }

  get hasMoreLODs(): boolean {
    // A disposed loader has its work state cleared; report no further work so a
    // refinement loop holding a stale reference stops instead of indexing into
    // the now-empty `lodLoaders`. Mirrors the sibling loaders.
    if (this._disposed) return false;
    // While a playback frame budget is active the budgeted prefix IS the target:
    // no background refinement between animation ticks.
    if (this._frameBudgetMs !== null) return false;
    return this.loadedLODs.length < this.nLods;
  }

  get loadedLODCount(): number {
    return this.loadedLODs.length;
  }

  get totalLODCount(): number {
    return this.nLods;
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

  async updateView(
    viewState: MeshViewState,
    session?: UpdateSession,
    signal?: AbortSignal
  ): Promise<LoadedMeshData> {
    // Record the per-pass playback budget FIRST: a pause re-trigger arrives with
    // the same view state and must still clear the budget.
    this._frameBudgetMs = viewState.frameBudgetMs ?? null;
    const budgetDeadline =
      this._frameBudgetMs !== null ? performance.now() + this._frameBudgetMs : null;

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
    const startLevel = this.loadedLODs.length;

    for (let level = startLevel; level < this.nLods; level++) {
      // A dispose() racing the awaited level below clears `lodLoaders`, so the
      // next iteration would TypeError — a teardown mis-counted as a refinement
      // failure. Stop streaming instead.
      if (this._disposed) break;
      if (!shouldLoadLevel(pass, level, startLevel)) break;
      // Playback frame budget: stop once the tick's time is spent (≥1 level
      // always loads — the `level > startLevel` guard).
      if (budgetDeadline !== null && level > startLevel && performance.now() > budgetDeadline) {
        break;
      }
      const t0 = performance.now();
      const { data: lodData, allResident } = await this.lodLoaders[level].updateViewWithResidency(
        viewState,
        session,
        signal
      );
      const elapsed = performance.now() - t0;

      // Re-checked after the await: a dispose() during the fetch cleared the
      // ladder, and pushing here would resurrect it on a dead loader (and pin
      // the whole prefix's arrays that nothing will read).
      if (this._disposed) break;
      this.loadedLODs.push(lodData);

      if (!this._initialLoadDone) {
        log.custom(
          LogEmoji.BROADCAST,
          Modules.SCENE_LOADER,
          `Mesh reveal ${this.path} LOD ${level}/${this.nLods - 1}: ` +
            `${lodData.faceCount} faces (${elapsed.toFixed(1)}ms${allResident ? '' : ', miss'})`
        );
      }

      if (shouldStopAfterLevel(pass, level, startLevel, allResident, elapsed)) break;
    }

    if (!this._initialLoadDone && startLevel === 0) this._initialLoadDone = true;

    const totalFaces = this.loadedLODs.reduce((s, d) => s + d.faceCount, 0);
    if (this.loadedLODs.length < this.nLods) {
      log.info(
        Modules.SCENE_LOADER,
        `Progressive Mesh: ${this.loadedLODs.length}/${this.nLods} LODs ` +
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
      if (this._concatCache && this._concatCache.lodCount === this.loadedLODs.length) {
        return this._concatCache.result;
      }
      const result = concatenateMeshData(this.loadedLODs);
      this._concatCache = { lodCount: this.loadedLODs.length, result };
      return result;
    } finally {
      concatSession?.end();
    }
  }

  dispose(): void {
    this._disposed = true;
    for (const loader of this.lodLoaders) loader.dispose();
    this.lodLoaders = [];
    this.loadedLODs = [];
    this._concatCache = null;
  }
}
