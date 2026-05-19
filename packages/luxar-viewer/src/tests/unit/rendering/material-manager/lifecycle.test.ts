/**
 * Direct unit tests for the material lifecycle helpers extracted in
 * P6/step 4.3. Covers the dispose-event subscription, the
 * `SOFT_DISPOSE_FLAG` soft-dispose path, and the per-class cache
 * teardown branches of `removeFromRegistries`.
 *
 * Uses real PointMaterial / LineMaterial / GSplatMaterial instances
 * because the soft-dispose path keys on `instanceof` — a fake material
 * wouldn't exercise the branch.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  subscribeToDispose,
  removeFromRegistries,
  SOFT_DISPOSE_FLAG,
  type LifecycleCtx,
} from '../../../../rendering/material-manager/lifecycle';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import type { CameraAwareMaterial } from '../../../../rendering/materials/_shared/camera-aware-material';

/** Build an empty LifecycleCtx with fresh Set/Map/WeakSet for each test. */
function makeCtx(): LifecycleCtx {
  return {
    registeredMaterials: new Set<THREE.Material & CameraAwareMaterial>(),
    ownedMaterials: new Set<THREE.Material & CameraAwareMaterial>(),
    subscribedMaterials: new WeakSet<THREE.Material & CameraAwareMaterial>(),
    pointMaterialCache: new Map(),
    lineMaterialCache: new Map(),
    gsplatMaterialCache: new Map(),
  };
}

describe('subscribeToDispose', () => {
  it('attaches a dispose listener and tracks the material in subscribedMaterials', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    expect(ctx.subscribedMaterials.has(material)).toBe(false);

    subscribeToDispose(material, ctx);

    expect(ctx.subscribedMaterials.has(material)).toBe(true);
  });

  it('is idempotent — a second subscribe call is a no-op', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.registeredMaterials.add(material);

    subscribeToDispose(material, ctx);
    subscribeToDispose(material, ctx);
    subscribeToDispose(material, ctx);

    // Dispatching once must still produce exactly one cleanup, not three
    // (which would happen if listeners stacked).
    material.dispose();
    expect(ctx.registeredMaterials.has(material)).toBe(false);
  });

  it('cleans up the material from every registry when dispose fires', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.registeredMaterials.add(material);
    ctx.ownedMaterials.add(material);
    ctx.pointMaterialCache.set('test-key', material);

    subscribeToDispose(material, ctx);
    material.dispose();

    expect(ctx.registeredMaterials.has(material)).toBe(false);
    expect(ctx.ownedMaterials.has(material)).toBe(false);
    expect(ctx.pointMaterialCache.has('test-key')).toBe(false);
    // subscribedMaterials is also cleared so a re-subscribe (e.g. after
    // restore) attaches a fresh listener instead of being a no-op.
    expect(ctx.subscribedMaterials.has(material)).toBe(false);
  });

  it('SOFT_DISPOSE_FLAG=true skips registry/cache cleanup when dispose fires', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.registeredMaterials.add(material);
    ctx.pointMaterialCache.set('soft-key', material);

    subscribeToDispose(material, ctx);
    // Caller flags this as a soft-dispose (RenderObject cache eviction).
    (material as unknown as Record<symbol, boolean>)[SOFT_DISPOSE_FLAG] = true;
    material.dispatchEvent({ type: 'dispose' });

    // Cleanup MUST be skipped — material still in every registry.
    expect(ctx.registeredMaterials.has(material)).toBe(true);
    expect(ctx.pointMaterialCache.has('soft-key')).toBe(true);
    // Listener is still attached for the next (real) dispose.
    expect(ctx.subscribedMaterials.has(material)).toBe(true);
  });

  it('honours the soft-dispose flag once, then runs cleanup on the next real dispose', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.registeredMaterials.add(material);
    ctx.pointMaterialCache.set('two-step', material);
    subscribeToDispose(material, ctx);

    // 1st dispose: soft-flag set → cleanup skipped.
    const tagged = material as unknown as Record<symbol, boolean>;
    tagged[SOFT_DISPOSE_FLAG] = true;
    material.dispatchEvent({ type: 'dispose' });
    expect(ctx.registeredMaterials.has(material)).toBe(true);

    // 2nd dispose: clear flag → cleanup runs.
    tagged[SOFT_DISPOSE_FLAG] = false;
    material.dispatchEvent({ type: 'dispose' });
    expect(ctx.registeredMaterials.has(material)).toBe(false);
    expect(ctx.pointMaterialCache.has('two-step')).toBe(false);
  });
});

describe('removeFromRegistries', () => {
  it('removes from registeredMaterials and ownedMaterials regardless of class', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.registeredMaterials.add(material);
    ctx.ownedMaterials.add(material);

    removeFromRegistries(material, ctx);

    expect(ctx.registeredMaterials.has(material)).toBe(false);
    expect(ctx.ownedMaterials.has(material)).toBe(false);
  });

  it('removes a PointMaterial from pointMaterialCache (instanceof Point branch)', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.pointMaterialCache.set('pt', material);
    ctx.lineMaterialCache.set('ln', new LineMaterial());
    ctx.gsplatMaterialCache.set('gs', new GSplatMaterial());

    removeFromRegistries(material, ctx);

    expect(ctx.pointMaterialCache.has('pt')).toBe(false);
    // Other caches are untouched
    expect(ctx.lineMaterialCache.has('ln')).toBe(true);
    expect(ctx.gsplatMaterialCache.has('gs')).toBe(true);
  });

  it('removes a LineMaterial from lineMaterialCache (instanceof Line branch)', () => {
    const ctx = makeCtx();
    const material = new LineMaterial();
    ctx.lineMaterialCache.set('ln', material);

    removeFromRegistries(material, ctx);

    expect(ctx.lineMaterialCache.has('ln')).toBe(false);
  });

  it('removes a GSplatMaterial from gsplatMaterialCache (instanceof GSplat branch)', () => {
    const ctx = makeCtx();
    const material = new GSplatMaterial();
    ctx.gsplatMaterialCache.set('gs', material);

    removeFromRegistries(material, ctx);

    expect(ctx.gsplatMaterialCache.has('gs')).toBe(false);
  });

  it('is idempotent — calling on an already-absent material is a no-op', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();

    expect(() => removeFromRegistries(material, ctx)).not.toThrow();
    expect(ctx.registeredMaterials.size).toBe(0);
  });

  it('only deletes the matching cache entry; siblings in the same cache are preserved', () => {
    const ctx = makeCtx();
    const target = new PointMaterial();
    const sibling = new PointMaterial();
    ctx.pointMaterialCache.set('target', target);
    ctx.pointMaterialCache.set('sibling', sibling);

    removeFromRegistries(target, ctx);

    expect(ctx.pointMaterialCache.has('target')).toBe(false);
    expect(ctx.pointMaterialCache.has('sibling')).toBe(true);
    expect(ctx.pointMaterialCache.get('sibling')).toBe(sibling);
  });
});

describe('SOFT_DISPOSE_FLAG', () => {
  it('is a global registry symbol — same identity across imports', () => {
    // Symbol.for('...') returns the same Symbol instance for the same key,
    // so a dispatcher and a listener in different modules see the same flag.
    expect(SOFT_DISPOSE_FLAG).toBe(Symbol.for('luxar.invalidateRenderObject.softDispose'));
  });
});
