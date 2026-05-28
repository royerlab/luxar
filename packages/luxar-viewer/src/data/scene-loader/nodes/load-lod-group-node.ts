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
 * Sibling of `data/scene-loader/nodes/load-scene-nodes.ts` (dispatch)
 * and `scene/lod-group-registry.ts` (per-frame eval).
 *
 * @module data/scene-loader/nodes/load-lod-group-node
 */

import * as THREE from 'three';
import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';
import type { SceneNode } from '../../data-loader-types';
import type { LODGroupChild, LODGroupEntry } from '../../../scene/lod-group-registry';
import type { LODGroupMetadata, LODGroupSelectorMode } from '../../../types/lod-group';
import type { NodeBuildCtx } from './build-ctx';
// Circular-import safe: load-scene-nodes pulls us in too, but only
// through dynamic dispatch at runtime — the recursive call below
// reads the symbol after the module graph settles.
import { loadSceneNodes } from './load-scene-nodes';

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
function readPositionBounds(
  childAttrs: SceneNode['attrs']
): { min: readonly number[]; max: readonly number[] } {
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
  ctx: NodeBuildCtx
): Promise<THREE.Group> {
  const attrs = node.attrs as unknown as LODGroupMetadata;
  log.custom('🎚️', Modules.SCENE_LOADER, `Loading lod_group: ${node.path}`);

  const lodThreeGroup = new THREE.Group();
  lodThreeGroup.name = node.path;
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

  // Recurse into each child. Each child's loader (gsplats / points /
  // lines / nested group) attaches its own THREE node to lodThreeGroup
  // — we look it up by name afterward to wire the registry entry.
  const registryChildren: LODGroupChild[] = [];
  for (const child of sceneChildren) {
    const childLoc = parentLoc.resolve(child.path.slice(1));
    await loadSceneNodes(child, lodThreeGroup, childLoc, ctx);

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
    // loop below `await`s the next child's commit, so without this
    // line every already-loaded sibling renders simultaneously during
    // the load — a brief "stacked LOD levels" flash on initial load
    // (and on the no-registry fallback path too). ``register()``
    // re-enables the chosen active child synchronously at the end of
    // this function, so the swap is atomic from the user's POV.
    childObject.visible = false;

    const minPixelSizeRaw = (child.attrs as Record<string, unknown>).min_pixel_size;
    const minPixelSize =
      typeof minPixelSizeRaw === 'number' ? minPixelSizeRaw : 0;

    registryChildren.push({
      object: childObject,
      minPixelSize,
      positionBounds: readPositionBounds(child.attrs),
    });
  }

  const defaultLevel = clampDefaultLevel(attrs.default_level, registryChildren.length);

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
