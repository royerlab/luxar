/**
 * Initial-load path for a single ``lod_group`` scene-graph node.
 *
 * Mirrors the generic-group branch in ``load-scene-nodes.ts`` — creates
 * a ``THREE.Group``, applies the transform, recurses into children —
 * with two additions:
 *
 *   1. Each child carries a ``min_pixel_size`` attribute (a per-child
 *      pixel threshold) plus its own ``position_bounds`` (the raw nD
 *      AABB). Both are read from the child's zarr attrs.
 *   2. Once children are loaded, an entry is registered with the
 *      :class:`LODGroupRegistry` so the per-frame selector can pick
 *      which child renders.
 *
 * The lod_group walks the **scene-graph** child order (which the
 * Python writer guarantees is coarsest→finest); the registry stores
 * children in that same order for consistent threshold comparisons.
 *
 * **Lazy loading**: only the default level's geometry is loaded eagerly.
 * Every other gsplats level is *cheap-attached* (placeholder + loader,
 * no array fetch) with an ``ensureLoaded`` thunk; the registry fires the
 * thunk on demand the first time the per-frame selector wants to show
 * that level. This is what keeps a scene of many lod_groups from loading
 * every level of every group up front — distant groups stay coarse and
 * their fine levels are never fetched. The selector math needs only the
 * per-child ``min_pixel_size`` / ``position_bounds`` attrs (read here),
 * not loaded geometry, so deferral is fully correct.
 *
 * Sibling of `data/scene-loader/nodes/load-scene-nodes.ts` (dispatch),
 * `data/scene-loader/nodes/load-gsplats-node.ts` (cheap/expensive split),
 * and `scene/lod-group-registry.ts` (per-frame eval + lazy trigger).
 *
 * @module data/scene-loader/nodes/load-lod-group-node
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import { loadGSplatsNodeCheap, loadGSplatsNodeExpensive } from './load-gsplats-node';
import { timeLodStageSync } from '../../../scene/lod-load-stats';
import type { SceneNode } from '../../data-loader-types';
import type { LODGroupChild, LODGroupEntry } from '../../../scene/lod-group-registry';
import type { LODGroupMetadata, LODGroupSelectorMode } from '../../../types/lod-group';
import type { NodeBuildCtx } from './build-ctx';

/**
 * Signature of the recursive scene-graph walker. Injected at the
 * call site to break the otherwise-cyclic import with
 * `load-scene-nodes.ts` — the recursion is genuine (lod_group
 * children may themselves be groups or further lod_groups) but a
 * static back-reference would fail the dep-cruiser cycle check.
 */
export type LoadSceneChildren = (
  node: SceneNode,
  parentThree: THREE.Object3D,
  parentLoc: zarr.Location<zarr.Readable>,
  ctx: NodeBuildCtx
) => Promise<void>;

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
    | { min?: unknown; max?: unknown }
    | undefined;
  if (!raw) return EMPTY_BOUNDS;
  const min = raw.min;
  const max = raw.max;
  if (!Array.isArray(min) || !Array.isArray(max)) return EMPTY_BOUNDS;
  return {
    min: (min as unknown[]).map(Number),
    max: (max as unknown[]).map(Number),
  };
}

/**
 * Load an ``lod_group`` node on initial scene construction.
 *
 * Pattern:
 *   1. Create a ``THREE.Group`` for the lod_group; apply transform.
 *   2. Recurse each child through ``loadSceneNodes`` so its
 *      geometry-specific loader runs and a placeholder mesh attaches.
 *   3. After each child loads, locate its THREE node by name and
 *      record its ``min_pixel_size`` / ``position_bounds``.
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

  const registryChildren: LODGroupChild[] = [];
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

    const minPixelSizeRaw = (child.attrs as Record<string, unknown>).min_pixel_size;
    const minPixelSize = typeof minPixelSizeRaw === 'number' ? minPixelSizeRaw : 0;

    // Defer only when there's a selector to trigger the load AND the
    // child is a gsplats node (the cheap/expensive split lives in the
    // gsplats loader). The eager/default child and any non-gsplats child
    // load fully now.
    const canDefer = hasRegistry && i !== eagerIdx && child.type === 'gsplats';

    if (canDefer) {
      // Cheap-attach: placeholder mesh + registered loader, no array
      // fetch. The thunk runs the expensive tail on first activation.
      const { placeholder, loader } = await loadGSplatsNodeCheap(
        child,
        lodThreeGroup,
        childLoc,
        ctx
      );
      placeholder.visible = false;

      const entryChild: LODGroupChild = {
        object: placeholder,
        minPixelSize,
        positionBounds: readPositionBounds(child.attrs),
        ready: false,
      };
      const lazyChild = child;
      entryChild.ensureLoaded = () => {
        // Fire-and-forget; fully self-contained error handling so a
        // rejected promise never escapes as an unhandled rejection. The
        // thunk owns ready/failed/loading; it must not touch visibility
        // (the registry performs the swap once ``ready`` flips true).
        void (async () => {
          try {
            await loadGSplatsNodeExpensive(lazyChild, ctx, loader);
            // Liveness re-check: the dataset may have been switched/disposed
            // while this deferred load was in flight. `loadGSplatsNodeExpensive`
            // skips its commit when not live but returns normally, so without
            // this guard we would re-register the loader into the shared
            // registry (possibly after dispose) and mark a geometry-less level
            // ready. Drop silently — the abort-discard policy.
            if (!ctx.isDatasetLive()) return;
            // Register only now that the level is loaded+committed, so it
            // joins subsequent updateView sweeps. Registering earlier
            // would pull this level into the scene-wide loader update and
            // load it regardless of selection — defeating laziness.
            ctx.registry.registerGSplatsLoader(lazyChild.path, loader);
            entryChild.ready = true;
          } catch (error) {
            entryChild.failed = true;
            log.warning(
              Modules.SCENE_LOADER,
              `lod_group lazy level ${lazyChild.path} failed to load: ${String(error)}`
            );
          } finally {
            entryChild.loading = false;
          }
        })();
      };
      entryChild.release = () => {
        // Return the GPU buffer to the evictable pool + drop the loader,
        // then reset readiness so a later selection reloads via the same
        // ``ensureLoaded`` path. Raw chunks remain cached, so reload is a
        // cheap re-projection. Timing is debug-only (no-op unless ?debug).
        timeLodStageSync('lazy:release', () => ctx.releaseLazyGSplats(lazyChild.path));
        entryChild.ready = false;
        entryChild.loading = false;
        entryChild.failed = false;
        entryChild.failedTick = undefined;
      };
      registryChildren.push(entryChild);
      continue;
    }

    // Eager path: load fully via the generic recursion (handles any
    // geometry type), then look up the attached THREE node by name.
    await loadChildren(child, lodThreeGroup, childLoc, ctx);

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
    registryChildren.push({
      object: childObject,
      minPixelSize,
      positionBounds: readPositionBounds(child.attrs),
    });
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
