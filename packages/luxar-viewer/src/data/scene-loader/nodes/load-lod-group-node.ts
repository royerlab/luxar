/**
 * Initial-load path for a single ``lod_group`` scene-graph node.
 *
 * Mirrors the generic-group branch in ``load-scene-nodes.ts`` — creates
 * a ``THREE.Group``, applies the transform, recurses into children —
 * with two additions:
 *
 *   1. Each child carries a ``coverage_fraction`` attribute (a per-child
 *      threshold whose UNITS the group's ``selector`` attr names: literal
 *      screen-area fractions under ``'screen-area'`` — derived whole-object
 *      ladders span [0, 1/2], a partition tile anchors at 1.0 — or the legacy
 *      diagonal metric under ``'coverage'``, up to 4.0 ==
 *      ``SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR``)
 *      plus its own ``position_bounds`` (the raw nD AABB) and optional
 *      ``lod_bounds`` (a robust selector-only AABB). All are read from the
 *      child's zarr attrs.
 *      Legacy (pre-v3.2) datasets that still carry ``min_pixel_size`` /
 *      selector ``'pixel_size'`` are auto-adapted with a warning
 *      (see ``resolveCoverageFractions``).
 *   2. Once children are loaded, an entry is registered with the
 *      :class:`LODGroupRegistry` so the per-frame selector can pick
 *      which child renders.
 *
 * The lod_group walks the **scene-graph** child order (which the
 * Python writer guarantees is coarsest→finest); the registry stores
 * children in that same order for consistent threshold comparisons.
 *
 * **Lazy loading**: only the default level's geometry is loaded eagerly.
 * Every other gsplats, points, lines *or mesh* level is *cheap-attached*
 * (placeholder + loader, no array fetch) with an ``ensureLoaded`` thunk; the
 * registry fires the thunk on demand the first time the per-frame selector wants
 * to show that level. This is what keeps a scene of many lod_groups from loading
 * every level of every group up front — distant groups stay coarse and
 * their fine levels are never fetched. Deferring the points/lines/mesh level
 * matters for their substitutive ladders, whose finest child is the full
 * cloud / line set / full-resolution surface (eager-loading it would defeat
 * progressive loading). The selector math needs
 * only the per-child ``coverage_fraction`` / ``position_bounds`` / optional
 * ``lod_bounds`` attrs (read here), not loaded geometry, so deferral is fully
 * correct.
 *
 * Sibling of `data/scene-loader/nodes/load-scene-nodes.ts` (dispatch),
 * `data/scene-loader/nodes/load-gsplats-node.ts` +
 * `data/scene-loader/nodes/load-points-node.ts` (cheap/expensive splits),
 * and `scene/lod-group-registry.ts` (per-frame eval + lazy trigger).
 *
 * @module data/scene-loader/nodes/load-lod-group-node
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { archiveFaultFrom } from '../../../cache/chunk-source';
import { log, Modules } from '../../../utils/log';
import { loadGSplatsNodeCheap, loadGSplatsNodeExpensive } from './load-gsplats-node';
import { loadPointsNodeCheap, loadPointsNodeExpensive } from './load-points-node';
import { loadLinesNodeCheap, loadLinesNodeExpensive } from './load-lines-node';
import { loadMeshNodeCheap, loadMeshNodeExpensive } from './load-mesh-node';
import { timeLodStageSync } from '../lod-load-stats';
import type { SceneNode } from '../../data-loader-types';
import type { LODGroupChild, LODGroupEntry } from '../../../scene/lod-group-registry';
import type { LODGroupMetadata, LODGroupSelectorMode } from '../../../types/lod-group';
import { supportsLod } from '../../../types/geometry-capabilities';
import type { NodeBuildCtx } from './build-ctx';
import { acquireEagerWorkingSet, type LoadSceneChildren } from './load-children-concurrently';

/**
 * Default raw bounds for a malformed / missing ``position_bounds``
 * attribute. The empty array makes the registry skip projection for
 * that child (its bbox won't contribute to the union), which is the
 * least-surprising fallback.
 */
const EMPTY_BOUNDS: { min: readonly number[]; max: readonly number[] } = {
  min: [] as readonly number[],
  max: [] as readonly number[],
};

/**
 * Build a deferred (lazy) ``LODGroupChild`` from an already cheap-attached
 * placeholder. Geometry-agnostic: the caller supplies ``runExpensive`` (fetch +
 * commit) and an optional ``releaseLoaded`` (return GPU buffers to the evictable
 * pool, and/or drop depth-sort state). Shared between the gsplats, points, lines
 * and mesh defer paths so the ready/failed/loading state machine and the
 * abort-discard error handling live in exactly one place. Container-wide
 * archive faults additionally latch retry-addressable leaf children as
 * permanently failed so the per-frame registry cannot retry a dataset already
 * known to be unreadable.
 *
 * **Lazy LEAF levels never join the per-slice update sweep.** ``runExpensive``
 * commits independently and the registry — not the sweep — drives their reload
 * on a slice change once the scrub settles (``LODGroupRegistry.maybeKickReload``).
 * Keeping a fine level out of the sweep is what lets the cheap coarse (eager)
 * level commit a new timepoint immediately instead of being gated behind the
 * slow fine reload. The deferred-GROUP caller below is the one exception, and
 * only from activation onwards: its ``runExpensive`` is the ``loadChildren``
 * recursion, whose nested leaf loaders register themselves exactly as they would
 * anywhere else. The placeholder this function holds is never registered either
 * way.
 */
function attachLazyChild(
  placeholder: THREE.Object3D,
  child: SceneNode,
  coverageFraction: number,
  ctx: NodeBuildCtx,
  runExpensive: () => Promise<void>,
  releaseLoaded?: () => void,
  hasMoreLODs?: () => boolean
): LODGroupChild {
  placeholder.visible = false;
  const positionBounds = readPositionBounds(child.attrs);
  const entryChild: LODGroupChild = {
    object: placeholder,
    coverageFraction,
    positionBounds,
    lodBounds: readLodBounds(child.attrs, child.path, positionBounds),
    ready: false,
    // Progressive (additive-laddered) levels report remaining LODs so the
    // registry can settle-gate further ``ensureLoaded`` passes to completion;
    // single-LOD levels omit it (no extra refinement).
    hasMoreLODs,
  };
  entryChild.ensureLoaded = () => {
    // Fire-and-forget; fully self-contained error handling so a rejected
    // promise never escapes as an unhandled rejection. The thunk owns
    // ready/failed/loading; it must not touch visibility (the registry swaps
    // once ``ready`` flips true).
    void (async () => {
      try {
        await runExpensive();
        // Liveness re-check: the dataset may have been switched/disposed while
        // this deferred load was in flight. The expensive halves skip their
        // commit when not live but return normally, so without this guard we
        // would mark a geometry-less level ready. Drop silently — the
        // abort-discard policy.
        if (!ctx.isDatasetLive()) return;
        // NOTE: the level is deliberately NOT registered into the per-slice
        // update sweep (see the function doc). It commits independently here;
        // the registry reloads it on a settled slice change.
        entryChild.ready = true;
      } catch (error) {
        entryChild.failed = true;
        // Anonymous group placeholders cannot be reached by retryLazyChildByLeafPath.
        const archiveFault = entryChild.object.name ? archiveFaultFrom(error) : undefined;
        if (archiveFault) {
          entryChild.permanentlyFailed = true;
          entryChild.failedTick = undefined;
          // A container fault makes the whole archive unreadable, not just this lazy level.
          if (ctx.isDatasetLive()) ctx.reportArchiveFault(archiveFault);
        }
        log.warning(
          Modules.SCENE_LOADER,
          `lod_group lazy level ${child.path} failed to load: ${String(error)}`
        );
      } finally {
        entryChild.loading = false;
      }
    })();
  };
  if (releaseLoaded) {
    entryChild.release = () => {
      // Return the GPU buffer to the evictable pool, then reset readiness so a
      // later selection reloads via the same ``ensureLoaded`` path. Raw chunks
      // remain cached, so reload is cheap. Timing is debug-only.
      timeLodStageSync('lazy:release', releaseLoaded);
      entryChild.ready = false;
      entryChild.loading = false;
      if (!entryChild.permanentlyFailed) {
        entryChild.failed = false;
        entryChild.failedTick = undefined;
      }
    };
  }
  return entryChild;
}

/**
 * Resolve the per-child selector thresholds for a lod_group, auto-adapting
 * legacy datasets.
 *
 * Current stores carry a per-child ``coverage_fraction`` derived by
 * SCREEN-OCCUPANCY HALVING in the units the group's ``selector`` names
 * (``'screen-area'`` for every derived ladder): coarsest 0.0, strictly
 * ascending, one halving of occupied screen area per level. A whole-object
 * ladder anchors its finest at 0.5 (half the screen); a ladder bound to a
 * spatial partition is re-anchored at fills-screen (area 1.0). Explicit
 * ``coverage_fractions=[...]`` lists and older stores keep the legacy
 * ``'coverage'`` diagonal metric (in [0,
 * ``SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR``] == [0, 4]).
 *
 * Datasets written before the v3.2 rename instead carry a
 * per-child ``min_pixel_size`` (absolute pixel thresholds; group ``selector``
 * = ``'pixel_size'``). Silently defaulting those to 0 would make the selector
 * permanently pick the FINEST child — eager-downloading full-res geometry and
 * defeating progressive LOD — so legacy ladders are **derived** instead:
 * normalizing the (strictly ascending, positive) legacy pixel thresholds by
 * the finest value maps them onto the coverage scale. For the legacy
 * count-anchored ladder (``base·sqrt(N_i/N_0)``) this yields the pre-halving
 * derived shape ``sqrt(N_i/N_finest)``; extent-anchored ladders keep their
 * relative switch points with finest == 1.0. One warning per group names
 * ``luxar gsplat migrate-format`` so the producer knows to upgrade.
 *
 * A child with neither attr is a genuinely malformed producer output: an
 * actionable error is logged (previously this silently defaulted to 0 and then
 * blamed the producer with a misleading "not strictly ascending" warning) and
 * the child falls back to threshold 0.
 */
function resolveCoverageFractions(node: SceneNode, children: SceneNode[]): number[] {
  const coverageRaw = children.map((c) => (c.attrs as Record<string, unknown>).coverage_fraction);
  const hasCoverage = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
  if (coverageRaw.every(hasCoverage)) {
    return coverageRaw as number[];
  }

  // Legacy (pre-v3.2) dataset: every threshold-less child carries the old
  // ``min_pixel_size`` attr instead (the legacy writer stamped it on EVERY
  // child, leaf or nested group; the group's ``selector`` was 'pixel_size').
  const legacyRaw = children.map((c) => (c.attrs as Record<string, unknown>).min_pixel_size);
  const isLegacy =
    children.length > 0 &&
    children.every(
      (_, i) =>
        hasCoverage(coverageRaw[i]) ||
        (typeof legacyRaw[i] === 'number' &&
          Number.isFinite(legacyRaw[i]) &&
          (legacyRaw[i] as number) > 0)
    ) &&
    legacyRaw.some((v, i) => !hasCoverage(coverageRaw[i]) && typeof v === 'number');
  if (isLegacy) {
    // Normalize by the finest (largest) legacy threshold → coverage scale.
    const merged = children.map((_, i) =>
      hasCoverage(coverageRaw[i]) ? (coverageRaw[i] as number) : (legacyRaw[i] as number)
    );
    const finest = Math.max(...merged.filter((_, i) => !hasCoverage(coverageRaw[i])));
    const derived = children.map((_, i) =>
      hasCoverage(coverageRaw[i]) ? (coverageRaw[i] as number) : (legacyRaw[i] as number) / finest
    );
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group ${node.path}: legacy 'min_pixel_size' selector attrs (pre-v3.2 ` +
        "'pixel_size' selector) auto-adapted to coverage fractions " +
        `[${derived.map((v) => v.toFixed(3)).join(', ')}]. Progressive LOD works, but ` +
        'please re-generate this dataset or upgrade it with ' +
        '`luxar gsplat migrate-format <in> <out>`.'
    );
    return derived;
  }

  // Malformed: some children carry NO selector threshold at all.
  const missing = children.filter((_, i) => !hasCoverage(coverageRaw[i])).map((c) => c.path);
  log.error(
    Modules.SCENE_LOADER,
    `lod_group ${node.path}: ${missing.length} of ${children.length} children carry ` +
      "no 'coverage_fraction' (or legacy 'min_pixel_size') selector threshold " +
      `(${missing.join(', ')}). Defaulting them to 0 — LOD selection for this group ` +
      'will be wrong (the finest level may load eagerly). Re-generate the dataset ' +
      'with the current writer, or upgrade a legacy file with ' +
      '`luxar gsplat migrate-format <in> <out>`.'
  );
  return coverageRaw.map((v) => (hasCoverage(v) ? v : 0));
}

/** Read raw nD position bounds from a child node's attrs. */
function readPositionBounds(childAttrs: SceneNode['attrs']): {
  min: readonly number[];
  max: readonly number[];
} {
  const attrs = childAttrs as Record<string, unknown>;
  // The Python compiler writes ``position_bounds`` on every gsplats /
  // points / lines node; ``center_bounds`` is the gsplats-internal
  // equivalent (same shape, same semantics) — accept either so we
  // work for any leaf type.
  const raw = (attrs.position_bounds ?? attrs.center_bounds) as
    { min?: unknown; max?: unknown } | undefined;
  if (!raw) return EMPTY_BOUNDS;
  const min = raw.min;
  const max = raw.max;
  if (!Array.isArray(min) || !Array.isArray(max)) return EMPTY_BOUNDS;
  return {
    min: (min as unknown[]).map(Number),
    max: (max as unknown[]).map(Number),
  };
}

/** Read optional robust nD bounds used only by the LOD metric. */
function readLodBounds(
  childAttrs: SceneNode['attrs'],
  childPath: string,
  positionBounds: { min: readonly number[]; max: readonly number[] }
): { min: readonly number[]; max: readonly number[] } | undefined {
  const attrs = childAttrs as Record<string, unknown>;
  if (!Object.prototype.hasOwnProperty.call(attrs, 'lod_bounds')) return undefined;
  const expectedDimensions = positionBounds.min.length;
  if (expectedDimensions === 0 || positionBounds.min.length !== positionBounds.max.length) {
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group child ${childPath}: rejected lod_bounds; ` +
        'child has no usable position_bounds to validate against'
    );
    return undefined;
  }
  const reject = (reason: string, appendDimensions: boolean = true): undefined => {
    const dimensions = appendDimensions ? ` (${expectedDimensions} values per bound)` : '';
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group child ${childPath}: rejected lod_bounds; expected ${reason}${dimensions}, ` +
        'falling back to position_bounds'
    );
    return undefined;
  };
  const raw = attrs.lod_bounds as { min?: unknown; max?: unknown } | undefined;
  if (!raw || !Array.isArray(raw.min) || !Array.isArray(raw.max)) {
    return reject('an object with min/max arrays');
  }
  if (raw.min.length === 0) return reject('non-empty bounds');
  if (raw.min.length !== raw.max.length) return reject('equal min/max lengths');
  if (raw.min.length !== expectedDimensions) {
    return reject(`${expectedDimensions} values per bound`, false);
  }
  const min = raw.min;
  const max = raw.max;
  for (let i = 0; i < min.length; i++) {
    if (typeof min[i] !== 'number' || typeof max[i] !== 'number') {
      return reject('numeric entries');
    }
    if (!Number.isFinite(min[i]) || !Number.isFinite(max[i])) return reject('finite numbers');
    if (min[i] > max[i]) return reject('ordered bounds');
    if (min[i] < positionBounds.min[i] || max[i] > positionBounds.max[i]) {
      return reject('bounds contained in position_bounds');
    }
  }
  return { min: min as number[], max: max as number[] };
}

/**
 * Load an ``lod_group`` node on initial scene construction.
 *
 * Pattern:
 *   1. Create a ``THREE.Group`` for the lod_group; apply transform.
 *   2. Recurse each child through ``loadSceneNodes`` so its
 *      geometry-specific loader runs and a placeholder mesh attaches.
 *   3. After each child loads, locate its THREE node by name and
 *      record its ``coverage_fraction`` / ``position_bounds``.
 *   4. Register a ``LODGroupEntry`` with the registry — the entry
 *      controls per-child visibility on subsequent frames.
 */
export async function loadLodGroupNode(
  node: SceneNode,
  parentThree: THREE.Object3D,
  parentLoc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx,
  loadChildren: LoadSceneChildren
): Promise<THREE.Group> {
  const attrs = node.attrs as unknown as LODGroupMetadata;
  log.custom('🎚️', Modules.SCENE_LOADER, `Loading lod_group: ${node.path}`);

  const lodThreeGroup = new THREE.Group();
  lodThreeGroup.name = node.path;
  // Mark the THREE node with its specialized-group kind so picking +
  // any future wrapper-aware machinery can identify it without
  // re-reading the on-disk attrs.
  lodThreeGroup.userData.kind = 'lod';
  if (attrs.transform) {
    ctx.nodeFactory.applyTransform(lodThreeGroup, attrs.transform);
  }
  parentThree.add(lodThreeGroup);

  const sceneChildren = node.children ?? [];
  if (sceneChildren.length === 0) {
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group ${node.path} has no children — registering empty entry`
    );
  }

  // Lazy loading: with a registry to drive the per-frame selector, we
  // load ONLY the default level's geometry eagerly (so the group shows
  // something immediately) and defer every other level — its arrays are
  // fetched on demand the first time the selector wants to display it.
  // For a substitutive ladder the default level is the coarsest
  // (~handful of splats), so a scene of N groups loads ~N coarse levels
  // up front instead of every level of every group. Without a registry
  // there is nothing to drive the deferred loads, so we fall back to
  // eagerly loading all children.
  const hasRegistry = !!ctx.lodGroupRegistry;
  const eagerIdx = clampDefaultLevel(attrs.default_level, sceneChildren.length);

  // Per-child selector thresholds, resolved up front so a legacy (pre-v3.2)
  // dataset carrying ``min_pixel_size`` instead of ``coverage_fraction`` is
  // auto-adapted (see resolveCoverageFractions) rather than silently
  // defaulting every level to 0 (which would pin the selector to the finest
  // child and defeat progressive LOD).
  const coverageFractions = resolveCoverageFractions(node, sceneChildren);

  const registryChildren: LODGroupChild[] = [];
  const childPaths = new Map<LODGroupChild, string>();
  // Registry index of the eagerly-loaded default child. `eagerIdx` indexes
  // `sceneChildren`, but a child that fails to attach is dropped from
  // `registryChildren`, shifting indices. Recomputing the default level from
  // `attrs.default_level` over the (possibly shorter) `registryChildren`
  // would then point at the wrong — possibly lazy, not-ready — level. So we
  // record where the eager child actually landed and use that as the active
  // index.
  let eagerRegistryIdx = -1;
  for (let i = 0; i < sceneChildren.length; i++) {
    const child = sceneChildren[i];
    const childLoc = parentLoc.resolve(child.path.slice(1));

    const coverageFraction = coverageFractions[i];

    // Defer only when there's a selector to trigger the load AND the child is a
    // LOD-capable leaf type, i.e. one with a cheap/expensive split. The
    // eager/default child and any other type (e.g. nested groups) load fully
    // now. Deferring the points/lines child matters for the points-/lines-
    // substitutive LOD ladder, whose finest child is the full cloud / line set —
    // without this it would be fetched eagerly on load, defeating progressive
    // loading. A geometry type with no LOD support (`types/geometry-capabilities`)
    // can never legitimately be a child here.
    const canDefer = hasRegistry && i !== eagerIdx && supportsLod(child.type);

    if (canDefer) {
      // Cheap-attach: placeholder mesh + loader, no array fetch. The activation
      // thunk runs the expensive tail ONLY: registration stays on the eager
      // `loadXNode` path, so a lazy level never joins the per-slice sweep.
      const lazyChild = child;
      let entryChild: LODGroupChild;
      if (child.type === 'gsplats') {
        const { placeholder, loader } = await loadGSplatsNodeCheap(
          child,
          lodThreeGroup,
          childLoc,
          ctx
        );
        entryChild = attachLazyChild(
          placeholder,
          lazyChild,
          coverageFraction,
          ctx,
          () => loadGSplatsNodeExpensive(lazyChild, ctx, loader),
          () => ctx.releaseLazyGSplats(lazyChild.path),
          () => (loader as { hasMoreLODs?: boolean }).hasMoreLODs === true
        );
      } else if (child.type === 'points') {
        const { placeholder, loader } = await loadPointsNodeCheap(
          child,
          lodThreeGroup,
          childLoc,
          ctx
        );
        entryChild = attachLazyChild(
          placeholder,
          lazyChild,
          coverageFraction,
          ctx,
          () => loadPointsNodeExpensive(lazyChild, ctx, loader),
          () => ctx.releaseLazyPoints(lazyChild.path),
          () => (loader as { hasMoreLODs?: boolean }).hasMoreLODs === true
        );
      } else if (child.type === 'mesh') {
        const { placeholder, loader } = await loadMeshNodeCheap(
          child,
          lodThreeGroup,
          childLoc,
          ctx
        );
        entryChild = attachLazyChild(
          placeholder,
          lazyChild,
          coverageFraction,
          ctx,
          () => loadMeshNodeExpensive(lazyChild, ctx, loader),
          // `releaseLazyMesh` does LESS than its three peers, not nothing. They
          // release a pooled GPU buffer back to the evictable pool on demotion; a
          // mesh is `pooled: false` (an indexed BufferGeometry, not the
          // instanced-quad stack), so there is nothing to hand back and no pool
          // adapter to hand it to — a demoted level keeps its geometry until the
          // node is disposed, the same lifetime a non-LOD mesh already has. What it
          // DOES share is the depth-sort release, because mesh is `depthSortable`
          // (#1347): a demoted `normal`-mode level would otherwise pin its
          // coordinator state and worker-side centroids while not being drawn.
          () => ctx.releaseLazyMesh(lazyChild.path),
          // The same probe the other three pass, and it became load-bearing when
          // mesh gained a reveal ladder (#1476). A lazy level is deliberately kept
          // out of the per-slice sweep, so the registry is the ONLY thing that can
          // advance an additive ladder inside it: it re-fires `ensureLoaded` while
          // this reports true. Reporting `undefined` — correct while a mesh level
          // was whole-node resident in one fetch and therefore complete the moment
          // it was ready — would now freeze a laddered mesh level at its first
          // patch forever, with nothing in the logs to say why.
          () => (loader as { hasMoreLODs?: boolean }).hasMoreLODs === true
        );
      } else {
        // `canDefer` admits every LOD-capable type, so this is the lines branch.
        // Assert it explicitly so a future deferrable type added to the capability
        // table but not here fails loudly instead of being mis-loaded as lines.
        if (child.type !== 'lines') {
          throw new Error(
            `lod_group defer dispatch: unhandled deferrable child type "${child.type}" ` +
              `for ${child.path} — add a branch above`
          );
        }
        const { placeholder, loader } = await loadLinesNodeCheap(
          child,
          lodThreeGroup,
          childLoc,
          ctx
        );
        entryChild = attachLazyChild(
          placeholder,
          lazyChild,
          coverageFraction,
          ctx,
          () => loadLinesNodeExpensive(lazyChild, ctx, loader),
          () => ctx.releaseLazyLines(lazyChild.path),
          () => (loader as { hasMoreLODs?: boolean }).hasMoreLODs === true
        );
      }
      registryChildren.push(entryChild);
      childPaths.set(entryChild, child.path);
      continue;
    }

    // Deferred GROUP path: a non-leaf child (a nested kind=partition or
    // kind=lod) that is not the eager default. The leaf cheap/expensive split
    // doesn't apply, but the selector only needs the child's coverage_fraction +
    // position_bounds (both on attrs, read by attachLazyChild) — not loaded
    // geometry — so we cheap-attach an empty placeholder group and load the
    // whole subtree lazily on first activation. This is what keeps multiscale's
    // fine kind=partition branch from loading eagerly at scene-init while the
    // coarse cap is the visible level (it loads only once you zoom in close
    // enough to select it).
    //
    // Geometry-agnostic by construction: it branches on the wrapper's *kind*
    // (lod/partition), never on the inner leaf type, and the load runs through
    // the same ``loadChildren`` recursion as any other node — so a
    // partition/lod nesting of points, lines or mesh defers identically to
    // gsplats (all four stay symmetric here; see the parametrized test).
    //
    // A per-child transform would make the transform-less placeholder
    // mis-project its bounds, so those (rare) fall through to the eager path
    // below. Grouped subtrees have no leaf-style evictable buffer pool, so
    // there is no release(): once loaded they stay resident and scene teardown
    // disposes them — matching the prior eager behaviour, just deferred to
    // first view.
    const childAttrs = child.attrs as Record<string, unknown>;
    const canDeferGroup =
      hasRegistry &&
      i !== eagerIdx &&
      child.type === 'group' &&
      (childAttrs.kind === 'lod' || childAttrs.kind === 'partition') &&
      !childAttrs.transform;

    if (canDeferGroup) {
      // Transparent lazy wrapper: an anonymous, empty group. The registry holds
      // it by reference for visibility toggling, so it needs no name/kind — and
      // must NOT take the child's name/kind, or it would duplicate the identity
      // of the real node that ``loadChildren`` attaches *under* it on activation
      // (which owns the path + kind for picking / getObjectByName).
      const placeholder = new THREE.Group();
      lodThreeGroup.add(placeholder);
      const lazyChild = child;
      // No releaseLoaded: grouped subtrees have no leaf-style evictable buffer
      // pool, so once loaded they stay resident until scene teardown (matching
      // the prior behaviour, just deferred to first view). Nested leaf / lod
      // loaders self-register during loadChildren, which runs only on activation.
      const entryChild = attachLazyChild(
        placeholder,
        lazyChild,
        coverageFraction,
        ctx,
        async () => {
          await loadChildren(lazyChild, placeholder, childLoc, ctx);
          // The subtree's part leaves registered into the sweep maps just
          // now, mid-session — but refinement is only scheduled at
          // update-view tails, so without this kick their additive ladders
          // would sit at chunk-1 until the next slice change (and the
          // never-downgrade display gate would hold the previous level
          // indefinitely). Safe on a dead dataset: the kick no-ops once the
          // owning loader is disposed.
          ctx.kickRefinementIfIdle();
        }
      );
      registryChildren.push(entryChild);
      childPaths.set(entryChild, child.path);
      continue;
    }

    // Eager path: load fully via the generic recursion (handles any
    // geometry type), then look up the attached THREE node by name.
    const releaseWorkingSet = await acquireEagerWorkingSet(child, ctx);
    try {
      await loadChildren(child, lodThreeGroup, childLoc, ctx);
    } finally {
      releaseWorkingSet();
    }

    const childObject = lodThreeGroup.getObjectByName(child.path);
    if (!childObject) {
      log.warning(
        Modules.SCENE_LOADER,
        `lod_group child ${child.path} did not attach a THREE node — skipping`
      );
      continue;
    }

    // Hide the child immediately. Each leaf loader attaches its
    // placeholder with the THREE default ``visible = true`` and the
    // loop above/below may `await` the next child, so without this line
    // every already-loaded sibling renders simultaneously during the
    // load — a brief "stacked LOD levels" flash on initial load (and on
    // the no-registry fallback path too). ``register()`` re-enables the
    // chosen active child synchronously at the end of this function, so
    // the swap is atomic from the user's POV.
    childObject.visible = false;

    if (i === eagerIdx) eagerRegistryIdx = registryChildren.length;
    const positionBounds = readPositionBounds(child.attrs);
    const entryChild: LODGroupChild = {
      object: childObject,
      coverageFraction,
      positionBounds,
      lodBounds: readLodBounds(child.attrs, child.path, positionBounds),
    };
    registryChildren.push(entryChild);
    childPaths.set(entryChild, child.path);
  }

  // Defense-in-depth: the per-frame selector (``pickChildWithHysteresis``)
  // assumes children are in ascending ``coverage_fraction`` order (coarsest→finest)
  // — it scans upward and stops at the first threshold above the metric, so a
  // later out-of-order (smaller) threshold would never be reached and the wrong
  // level renders. The Python writer guarantees ascending order
  // (``coverage_fractions`` + its monotonicity guard), but a hand-authored
  // or otherwise malformed scene could violate it. Rather than refuse the scene
  // (the geometry is fine — only the order is wrong; cf. ``validateTransformFormat``
  // which DOES refuse, because a row-major transform renders catastrophically
  // wrong), recover gracefully: stable-sort to ascending and warn so the
  // producer bug is surfaced. Almost always a no-op (already ascending).
  //
  // The check is for STRICTLY ascending (``<``, not ``<=``), matching the
  // Python writer's invariant (``_assert_strict_ascending`` rejects equal
  // thresholds). Two equal thresholds give the selector a zero-width hysteresis
  // band between those levels — a degenerate, producer-side bug — so we surface
  // it with the same warning. The stable sort leaves equal entries in place, so
  // the only effect for the equal case is the diagnostic.
  const isStrictlyAscending = registryChildren.every(
    (c, k) => k === 0 || registryChildren[k - 1].coverageFraction < c.coverageFraction
  );
  if (!isStrictlyAscending) {
    const before = registryChildren.map((c) => c.coverageFraction);
    // Object identity survives the sort, so remap the eager index by reference.
    const eagerChild = eagerRegistryIdx >= 0 ? registryChildren[eagerRegistryIdx] : null;
    // ES2019+ Array.sort is stable, so equal thresholds keep their relative order.
    registryChildren.sort((a, b) => a.coverageFraction - b.coverageFraction);
    if (eagerChild) eagerRegistryIdx = registryChildren.indexOf(eagerChild);
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group ${node.path}: child coverage_fraction thresholds are not strictly ` +
        `ascending (${before.join(', ')}). The coverage selector needs distinct ` +
        'coarsest-to-finest thresholds; re-sorted to ascending. Common causes: a ' +
        'non-geometry group (e.g. a metadata sidecar) was adopted as a child and ' +
        'defaulted to coverage_fraction=0, or the producer emitted a malformed ladder ' +
        '(coverage_fractions guarantees strictly ascending thresholds).'
    );
  }

  const lodBoundsCount = registryChildren.filter((child) => child.lodBounds != null).length;
  if (lodBoundsCount > 0 && lodBoundsCount < registryChildren.length) {
    const missingPaths = registryChildren
      .filter((child) => child.lodBounds == null)
      .map((child) => childPaths.get(child) ?? '<unknown>');
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group ${node.path}: lod_bounds are only usable on part of the ladder; ` +
        `missing ${missingPaths.join(', ')}. The metric falls back to position_bounds ` +
        'for those children, so one raw AABB can dominate the group union.'
    );
  }

  // Prefer the eager child's actual registry index. If it failed to attach
  // (eagerRegistryIdx still -1), fall back to the first ready/eager level so
  // the group shows something, else the clamped metadata default.
  const defaultLevel =
    eagerRegistryIdx >= 0
      ? eagerRegistryIdx
      : (() => {
          const firstReady = registryChildren.findIndex((c) => c.ready !== false);
          return firstReady >= 0
            ? firstReady
            : clampDefaultLevel(attrs.default_level, registryChildren.length);
        })();

  const entry: LODGroupEntry = {
    path: node.path,
    groupObject: lodThreeGroup,
    children: registryChildren,
    // Threshold units. Whitelisted: anything other than the literal
    // 'screen-area' (including the legacy 'pixel_size' spelling and a missing
    // attr) falls back to the legacy diagonal metric, whose units every older
    // store's thresholds were authored/derived in.
    selector: attrs.selector === 'screen-area' ? 'screen-area' : 'coverage',
    selectorMode: 'auto' satisfies LODGroupSelectorMode,
    defaultLevel,
    activeChildIndex: defaultLevel,
  };

  if (ctx.lodGroupRegistry) {
    ctx.lodGroupRegistry.register(entry);
    log.info(
      Modules.SCENE_LOADER,
      `  Registered lod_group with ${registryChildren.length} children, ` +
        `default_level=${defaultLevel}`
    );
  } else {
    // Defensive fallback: without a registry, hide everything but the
    // default level so we don't draw all alternatives on top of each
    // other. Matches the registry's initial-visibility logic.
    for (let i = 0; i < registryChildren.length; i++) {
      registryChildren[i].object.visible = i === defaultLevel;
    }
    log.warning(
      Modules.SCENE_LOADER,
      `lod_group ${node.path}: no registry on ctx, ` +
        `falling back to default_level=${defaultLevel}`
    );
  }

  return lodThreeGroup;
}

function clampDefaultLevel(value: number | undefined, nChildren: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0;
  const idx = Math.trunc(value);
  if (idx < 0) return 0;
  if (nChildren === 0) return 0;
  if (idx >= nChildren) return nChildren - 1;
  return idx;
}
