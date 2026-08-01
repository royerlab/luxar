/**
 * Material lifecycle: dispose-event subscription, registry teardown,
 * and the soft-dispose flag that lets callers fire `'dispose'` purely
 * to evict Three's cached `RenderObject` without actually tearing
 * down the material.
 *
 * MaterialManager keeps ownership of the registries / caches; this
 * module operates on them via a `LifecycleCtx` so the helpers don't
 * pull in a back-reference to the orchestrator.
 *
 * @module rendering/material-manager/lifecycle
 */

import * as THREE from 'three';
import { PointMaterial } from '../materials/point/material-glsl';
import { LineMaterial } from '../materials/line/material-glsl';
import { GSplatMaterial } from '../materials/gsplat/material-glsl';
import { PointTSLMaterial } from '../materials/point/material-tsl';
import { LineTSLMaterial } from '../materials/line/material-tsl';
import { GSplatTSLMaterial } from '../materials/gsplat/material-tsl';
import type { CameraAwareMaterial } from '../materials/_shared/camera-aware-material';
import { SOFT_DISPOSE_FLAG } from './soft-dispose-flag';

/** Type alias matching the Manager's three cached unions. */
type AnyPointMaterial = PointMaterial | PointTSLMaterial;
type AnyLineMaterial = LineMaterial | LineTSLMaterial;
type AnyGSplatMaterial = GSplatMaterial | GSplatTSLMaterial;

// The soft-dispose sentinel lives in its own leaf module
// (`./soft-dispose-flag`) so the dispatcher can import it without
// dragging in this file's material-factory imports; re-exported here so
// existing importers of `lifecycle` are unaffected.
export { SOFT_DISPOSE_FLAG };

/**
 * Registry + cache references the lifecycle helpers operate on.
 * Supplied by the `MaterialManager` orchestrator as it owns the
 * actual `Set` / `Map` instances; the helpers never mutate state
 * that isn't reachable through this context.
 */
export interface LifecycleCtx {
  readonly registeredMaterials: Set<THREE.Material & CameraAwareMaterial>;
  readonly ownedMaterials: Set<THREE.Material & CameraAwareMaterial>;
  readonly subscribedMaterials: WeakSet<THREE.Material & CameraAwareMaterial>;
  readonly pointMaterialCache: Map<string, AnyPointMaterial>;
  readonly lineMaterialCache: Map<string, AnyLineMaterial>;
  readonly gsplatMaterialCache: Map<string, AnyGSplatMaterial>;
}

/**
 * Subscribe to a material's `dispose` event so the manager can clean
 * up its registry / cache entries automatically. THREE.Material's
 * EventDispatcher fires `dispose` synchronously inside `dispose()`,
 * so by the time `super.dispose()` returns, the manager has already
 * forgotten about this material.
 *
 * Wiring cleanup this way (manager → material) instead of having
 * materials call back into the manager avoids an import cycle
 * between material-manager.ts and the per-geometry material modules.
 */
export function subscribeToDispose(
  material: THREE.Material & CameraAwareMaterial,
  ctx: LifecycleCtx
): void {
  if (ctx.subscribedMaterials.has(material)) return;
  const onDispose = (): void => {
    // Soft-dispose: caller dispatched `'dispose'` purely to evict
    // Three's cached `RenderObject` (see `SOFT_DISPOSE_FLAG`). Skip
    // registry/cache cleanup so the material keeps receiving global
    // camera updates and stays in its allocation cache.
    const tagged = material as unknown as Record<symbol, boolean | undefined>;
    if (tagged[SOFT_DISPOSE_FLAG]) return;
    ctx.subscribedMaterials.delete(material);
    removeFromRegistries(material, ctx);
    material.removeEventListener('dispose', onDispose);
  };
  material.addEventListener('dispose', onDispose);
  ctx.subscribedMaterials.add(material);
}

/**
 * Remove `material` from every registry and from whichever cache
 * matches its class. Called by the dispose listener above and (for
 * backwards compatibility) by the public `unregister` method on the
 * manager. Idempotent.
 */
export function removeFromRegistries(
  material: THREE.Material & CameraAwareMaterial,
  ctx: LifecycleCtx
): void {
  ctx.registeredMaterials.delete(material);
  ctx.ownedMaterials.delete(material);

  if (material instanceof PointMaterial || material instanceof PointTSLMaterial) {
    deleteFromCache(ctx.pointMaterialCache, material);
  } else if (material instanceof LineMaterial || material instanceof LineTSLMaterial) {
    deleteFromCache(ctx.lineMaterialCache, material);
  } else if (material instanceof GSplatMaterial || material instanceof GSplatTSLMaterial) {
    deleteFromCache(ctx.gsplatMaterialCache, material);
  }
}

function deleteFromCache<T>(cache: Map<string, T>, target: T): void {
  for (const [key, value] of cache.entries()) {
    if (value === target) {
      cache.delete(key);
      return;
    }
  }
}
