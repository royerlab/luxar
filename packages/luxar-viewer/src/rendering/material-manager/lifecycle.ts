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
import type { CameraAwareMaterial } from '../materials/_shared/camera-aware-material';
import { SOFT_DISPOSE_FLAG } from './soft-dispose-flag';

// The soft-dispose sentinel lives in its own leaf module
// (`./soft-dispose-flag`) so the dispatcher can import it without
// dragging in this file's material-factory imports; re-exported here so
// existing importers of `lifecycle` are unaffected.
export { SOFT_DISPOSE_FLAG };

/**
 * Registry references the lifecycle helpers operate on. Supplied by the
 * `MaterialManager` orchestrator as it owns the actual `Set` instances; the
 * helpers never mutate state that isn't reachable through this context.
 */
export interface LifecycleCtx {
  readonly registeredMaterials: Set<THREE.Material & CameraAwareMaterial>;
  readonly ownedMaterials: Set<THREE.Material & CameraAwareMaterial>;
  /**
   * Materials tracked for disposal that take NO camera broadcast.
   *
   * Mesh is the first and so far only member: it draws real geometry, so it has no
   * screen-space size to recompute from fov/resolution and therefore no
   * `updateCameraParams`. Giving it an empty one purely to fit
   * `registeredMaterials` would be a lie that also costs a per-frame call per node,
   * so it gets its own registry instead — tracked for dispose and counted in the
   * stats snapshot, but never iterated by the camera broadcast.
   */
  readonly staticMaterials: Set<THREE.Material>;
  readonly subscribedMaterials: WeakSet<THREE.Material>;
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
export function subscribeToDispose(material: THREE.Material, ctx: LifecycleCtx): void {
  if (ctx.subscribedMaterials.has(material)) return;
  const onDispose = (): void => {
    // Soft-dispose: caller dispatched `'dispose'` purely to evict
    // Three's cached `RenderObject` (see `SOFT_DISPOSE_FLAG`). Skip
    // registry cleanup so the material keeps receiving global camera
    // updates.
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
 * Remove `material` from every registry. Called by the dispose listener above
 * and by the public `unregister` method on the manager. Idempotent.
 */
export function removeFromRegistries(material: THREE.Material, ctx: LifecycleCtx): void {
  // `Set<A & B>.delete(a: A)` is accepted (TS method params are bivariant) and is
  // exactly what we want: a plain `THREE.Material` can only ever be absent from the
  // camera-aware sets, so the delete is a no-op there rather than a type hole.
  ctx.registeredMaterials.delete(material as THREE.Material & CameraAwareMaterial);
  ctx.ownedMaterials.delete(material as THREE.Material & CameraAwareMaterial);
  ctx.staticMaterials.delete(material);
}
