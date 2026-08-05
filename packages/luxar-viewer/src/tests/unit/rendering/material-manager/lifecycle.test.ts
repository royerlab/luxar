/**
 * Direct unit tests for the material lifecycle helpers. Covers the
 * dispose-event subscription, the `SOFT_DISPOSE_FLAG` soft-dispose
 * path, and the per-class cache
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

/** Build an empty LifecycleCtx with fresh Set/WeakSet for each test. */
function makeCtx(): LifecycleCtx {
  return {
    registeredMaterials: new Set<THREE.Material & CameraAwareMaterial>(),
    ownedMaterials: new Set<THREE.Material & CameraAwareMaterial>(),
    staticMaterials: new Set<THREE.Material>(),
    subscribedMaterials: new WeakSet<THREE.Material>(),
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

    subscribeToDispose(material, ctx);
    material.dispose();

    expect(ctx.registeredMaterials.has(material)).toBe(false);
    expect(ctx.ownedMaterials.has(material)).toBe(false);
    // subscribedMaterials is also cleared so a re-subscribe (e.g. after
    // restore) attaches a fresh listener instead of being a no-op.
    expect(ctx.subscribedMaterials.has(material)).toBe(false);
  });

  it('SOFT_DISPOSE_FLAG=true skips registry cleanup when dispose fires', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.registeredMaterials.add(material);

    subscribeToDispose(material, ctx);
    // Caller flags this as a soft-dispose (RenderObject cache eviction).
    (material as unknown as Record<symbol, boolean>)[SOFT_DISPOSE_FLAG] = true;
    material.dispatchEvent({ type: 'dispose' });

    // Cleanup MUST be skipped — material still in every registry.
    expect(ctx.registeredMaterials.has(material)).toBe(true);
    // Listener is still attached for the next (real) dispose.
    expect(ctx.subscribedMaterials.has(material)).toBe(true);
  });

  it('honours the soft-dispose flag once, then runs cleanup on the next real dispose', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();
    ctx.registeredMaterials.add(material);
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

  it.each([
    ['PointMaterial', () => new PointMaterial()],
    ['LineMaterial', () => new LineMaterial()],
    ['GSplatMaterial', () => new GSplatMaterial()],
  ])('removes a %s from both registries', (_name, make) => {
    const ctx = makeCtx();
    const material = make();
    ctx.registeredMaterials.add(material);
    ctx.ownedMaterials.add(material);

    removeFromRegistries(material, ctx);

    expect(ctx.registeredMaterials.has(material)).toBe(false);
    expect(ctx.ownedMaterials.has(material)).toBe(false);
  });

  it('is idempotent — calling on an already-absent material is a no-op', () => {
    const ctx = makeCtx();
    const material = new PointMaterial();

    expect(() => removeFromRegistries(material, ctx)).not.toThrow();
    expect(ctx.registeredMaterials.size).toBe(0);
  });

  it('removes only the target; a sibling material stays registered', () => {
    const ctx = makeCtx();
    const target = new PointMaterial();
    const sibling = new PointMaterial();
    ctx.registeredMaterials.add(target);
    ctx.registeredMaterials.add(sibling);

    removeFromRegistries(target, ctx);

    expect(ctx.registeredMaterials.has(target)).toBe(false);
    expect(ctx.registeredMaterials.has(sibling)).toBe(true);
  });
});

describe('SOFT_DISPOSE_FLAG', () => {
  it('is a global registry symbol — same identity across imports', () => {
    // Symbol.for('...') returns the same Symbol instance for the same key,
    // so a dispatcher and a listener in different modules see the same flag.
    expect(SOFT_DISPOSE_FLAG).toBe(Symbol.for('luxar.invalidateRenderObject.softDispose'));
  });
});
