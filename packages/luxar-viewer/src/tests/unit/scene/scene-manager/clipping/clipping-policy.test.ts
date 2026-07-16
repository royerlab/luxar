/**
 * Unit tests for the clipping-policy helpers used by SceneManager.
 *
 * The three helpers — applyClippingPlanes, autoAdjustFromBounds,
 * updateDynamicFromCache — are pure with respect to SceneManager,
 * consuming a narrow ClippingCtx. These tests pin the validation /
 * fallback / stability gates that the inline class methods used to
 * own.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import {
  type ClippingCtx,
  applyClippingPlanes,
  autoAdjustFromBounds,
  updateDynamicFromCache,
} from '../../../../../scene/scene-manager/clipping/clipping-policy';
import { SceneBoundsCache } from '../../../../../scene/scene-manager/clipping/scene-bounds-cache';
import type { ControlsManager } from '../../../../../controls/controls-manager';
import {
  type BoundingBox,
  boundingBoxToSphere,
  calculateClippingPlanesFromSphere,
  getBoundingBoxDiagonal,
} from '../../../../../scene/scene-manager/clipping/bounds-math';

function makeCamera(
  position = new THREE.Vector3(0, 0, 100),
  near = 0.1,
  far = 1000
): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(60, 16 / 9, near, far);
  cam.position.copy(position);
  cam.updateProjectionMatrix();
  return cam;
}

function makeControlsMock(): {
  controls: ControlsManager;
  setSceneScale: ReturnType<typeof vi.fn>;
} {
  const setSceneScale = vi.fn();
  return { controls: { setSceneScale } as unknown as ControlsManager, setSceneScale };
}

function makeCtx(opts: {
  camera?: THREE.PerspectiveCamera;
  scene?: THREE.Scene;
  metadataBounds?: BoundingBox | null;
}): {
  ctx: ClippingCtx;
  setSceneScale: ReturnType<typeof vi.fn>;
} {
  const camera = opts.camera ?? makeCamera();
  const scene = opts.scene ?? new THREE.Scene();
  const { controls, setSceneScale } = makeControlsMock();
  const boundsCache = new SceneBoundsCache();
  const getSceneBoundsFromMetadata = vi.fn(() => opts.metadataBounds ?? null);
  return {
    ctx: { camera, controls, scene, boundsCache, getSceneBoundsFromMetadata },
    setSceneScale,
  };
}

describe('applyClippingPlanes', () => {
  it('updates camera.near / camera.far and re-runs updateProjectionMatrix', () => {
    const camera = makeCamera();
    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');

    applyClippingPlanes(camera, 0.5, 500);

    expect(camera.near).toBe(0.5);
    expect(camera.far).toBe(500);
    expect(updateSpy).toHaveBeenCalledTimes(1);
  });

  it('rejects near >= far without mutating the camera', () => {
    const camera = makeCamera(new THREE.Vector3(), 1, 1000);
    applyClippingPlanes(camera, 1000, 1000);
    expect(camera.near).toBe(1);
    expect(camera.far).toBe(1000);
  });

  it('still applies values when far/near > 10000 (warning logged but not rejected)', () => {
    const camera = makeCamera(new THREE.Vector3(), 1, 1000);
    applyClippingPlanes(camera, 0.001, 100);
    expect(camera.near).toBe(0.001);
    expect(camera.far).toBe(100);
  });
});

describe('autoAdjustFromBounds — metadata path', () => {
  it('applies bounds-derived near/far when metadata bounds are present', () => {
    const camera = makeCamera(new THREE.Vector3(0, 0, 100));
    const metadataBounds: BoundingBox = {
      min: { x: -10, y: -10, z: -10 },
      max: { x: 10, y: 10, z: 10 },
    };
    const { ctx, setSceneScale } = makeCtx({ camera, metadataBounds });

    const result = autoAdjustFromBounds(ctx);

    expect(result.near).toBeGreaterThan(0);
    expect(result.far).toBeGreaterThan(result.near);
    expect(camera.near).toBe(result.near);
    expect(camera.far).toBe(result.far);
    expect(setSceneScale).toHaveBeenCalledTimes(1);

    // M6: pin that near/far are actually DERIVED from the metadata bounds via
    // the sphere formula — not arbitrary positive values. A mutant that set
    // far to a fixed large constant would pass the "> near" check but fail
    // this exact comparison.
    const expectedSphere = boundingBoxToSphere(metadataBounds);
    const expected = calculateClippingPlanesFromSphere(expectedSphere, { x: 0, y: 0, z: 100 });
    expect(result.near).toBeCloseTo(expected.near, 6);
    expect(result.far).toBeCloseTo(expected.far, 6);

    // M6: scene scale fed to controls is the bounds diagonal, not a placeholder.
    expect(setSceneScale).toHaveBeenCalledWith(getBoundingBoxDiagonal(metadataBounds));
  });

  it('skips setSceneScale when metadata bounds have zero extent', () => {
    const metadataBounds: BoundingBox = {
      min: { x: 5, y: 5, z: 5 },
      max: { x: 5, y: 5, z: 5 },
    };
    const { ctx, setSceneScale } = makeCtx({ metadataBounds });
    autoAdjustFromBounds(ctx);
    expect(setSceneScale).not.toHaveBeenCalled();
  });
});

describe('autoAdjustFromBounds — geometry fallback', () => {
  it('returns configured defaults when scene is empty (no metadata, no geometry)', () => {
    const camera = makeCamera(new THREE.Vector3(), 1, 1000);
    const { ctx } = makeCtx({ camera, metadataBounds: null });
    const { near, far } = autoAdjustFromBounds(ctx);
    expect(near).toBeGreaterThan(0);
    expect(far).toBeGreaterThan(near);
    // Camera not mutated when scene is empty.
    expect(camera.near).toBe(1);
    expect(camera.far).toBe(1000);
  });

  it('derives bounds from loaded geometry when metadata is missing', () => {
    const scene = new THREE.Scene();
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(20, 20, 20), new THREE.MeshBasicMaterial());
    scene.add(mesh);
    const camera = makeCamera(new THREE.Vector3(0, 0, 50));
    const { ctx, setSceneScale } = makeCtx({ camera, scene, metadataBounds: null });

    const { near, far } = autoAdjustFromBounds(ctx);

    expect(near).toBeGreaterThan(0);
    expect(far).toBeGreaterThan(near);
    expect(camera.near).toBe(near);
    expect(camera.far).toBe(far);
    expect(setSceneScale).toHaveBeenCalledTimes(1);
  });
});

describe('updateDynamicFromCache', () => {
  beforeEach(() => {
    // ensure clean state
  });

  it('is a no-op when the bounds-cache is empty', () => {
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 1, 1000);
    const { ctx } = makeCtx({ camera, metadataBounds: null });

    updateDynamicFromCache(ctx);

    // Cache was empty → no projection matrix update, near/far unchanged.
    expect(camera.near).toBe(1);
    expect(camera.far).toBe(1000);
  });

  it('updates near/far when changes exceed 0.1% threshold', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, 10] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.001, 10000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });

    // Prime the cache.
    ctx.boundsCache.ensure(scene);

    updateDynamicFromCache(ctx);

    // Camera position is at distance ~100 from origin with sphere radius ~17;
    // near should be tight, far should be ~117.
    expect(camera.near).toBeGreaterThan(0);
    expect(camera.far).toBeLessThan(10000);
    expect(camera.far).toBeGreaterThan(camera.near);

    // W7: pin the exact near/far against the same sphere formula the source
    // uses, so a sign error or missing safety-expansion factor is caught.
    const sphere = ctx.boundsCache.getSphere();
    expect(sphere).not.toBeNull();
    const expected = calculateClippingPlanesFromSphere(sphere!, { x: 0, y: 0, z: 100 });
    expect(camera.near).toBeCloseTo(expected.near, 4);
    expect(camera.far).toBeCloseTo(expected.far, 4);
  });

  // H4: the 0.1% stability gate. A sub-threshold camera move must NOT touch the
  // projection matrix; a supra-threshold move must. The existing "< 0.1%" test
  // only covers a zero-move no-op — this exercises the boundary on both sides.
  it('respects the 0.1% stability threshold on small vs large camera moves', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, 10] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.001, 10000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    // First call sets near/far from dist = 100 (near ≈ 81.8, far ≈ 118.2).
    updateDynamicFromCache(ctx);
    const nearAfterFirst = camera.near;
    const farAfterFirst = camera.far;

    // Sub-threshold move: Δdist = 0.05 ⇒ Δnear/near ≈ 0.06% < 0.1% → no-op.
    camera.position.set(0, 0, 100.05);
    const spySmall = vi.spyOn(camera, 'updateProjectionMatrix');
    updateDynamicFromCache(ctx);
    expect(spySmall).not.toHaveBeenCalled();
    expect(camera.near).toBe(nearAfterFirst);
    expect(camera.far).toBe(farAfterFirst);
    spySmall.mockRestore();

    // Supra-threshold move: Δdist = 1.0 ⇒ Δnear/near ≈ 1.2% > 0.1% → update.
    camera.position.set(0, 0, 101);
    const spyBig = vi.spyOn(camera, 'updateProjectionMatrix');
    updateDynamicFromCache(ctx);
    expect(spyBig).toHaveBeenCalledTimes(1);
    expect(camera.near).not.toBe(nearAfterFirst);
  });

  // G9: interaction invariant — after autoAdjust seeds the planes, repeated
  // per-frame updates while the camera moves must always leave the camera with
  // a valid frustum (0 < near < far).
  it('keeps a valid frustum (0 < near < far) across autoAdjust + per-frame moves', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, 10] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.001, 10000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });

    autoAdjustFromBounds(ctx);
    for (const z of [80, 50, 20, 10, 5, 1]) {
      camera.position.set(0, 0, z);
      updateDynamicFromCache(ctx);
      expect(camera.near).toBeGreaterThan(0);
      expect(camera.far).toBeGreaterThan(camera.near);
    }
  });

  it('skips projection-matrix update when changes are < 0.1%', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, 10] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.001, 10000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    // First call: significant change → update.
    updateDynamicFromCache(ctx);
    const nearAfterFirst = camera.near;
    const farAfterFirst = camera.far;

    // Second call without moving camera: must be a no-op.
    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');
    updateDynamicFromCache(ctx);
    expect(updateSpy).not.toHaveBeenCalled();
    expect(camera.near).toBe(nearAfterFirst);
    expect(camera.far).toBe(farAfterFirst);
  });

  it('keeps near positive even when camera is inside the sphere', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-100, -100, -100], max: [100, 100, 100] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 0), 0.001, 1000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    updateDynamicFromCache(ctx);

    expect(camera.near).toBeGreaterThan(0);
    expect(camera.far).toBeGreaterThan(camera.near);
  });

  // Inside-sphere PARITY with the pure helper (the W7 test above covers
  // the outside branch): the per-frame inlined math must produce the
  // exact same scale-aware near floor as calculateClippingPlanesFromSphere
  // when the camera is inside the sphere. Kills a mutant that reverts
  // the dynamic path's floor to a constant while the helper stays
  // scale-aware (the paths are intentionally duplicated for the
  // zero-allocation contract, so parity is the seam that stops drift).
  it('matches calculateClippingPlanesFromSphere inside the sphere (scale-aware floor parity)', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-100, -100, -100], max: [100, 100, 100] } };
    const camera = makeCamera(new THREE.Vector3(5, -3, 10), 0.001, 1000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    updateDynamicFromCache(ctx);

    const sphere = ctx.boundsCache.getSphere();
    expect(sphere).not.toBeNull();
    const expected = calculateClippingPlanesFromSphere(sphere!, { x: 5, y: -3, z: 10 });
    // Inside the sphere the helper returns the scale-aware floor —
    // pin the dynamic path to the identical value (exact, not close).
    expect(camera.near).toBe(expected.near);
    expect(camera.far).toBeCloseTo(expected.far, 10);
  });

  // Degenerate guard: a zero-extent (single-point) scene produces a
  // radius-0 sphere, for which near == far (camera away from the point)
  // or near > far (camera at the point). Writing either to the camera
  // puts (far - near) = 0 into the projection matrix and NaNs the
  // frustum — the function must refuse, like applyClippingPlanes does
  // on the explicit path.
  it('refuses to write a degenerate frustum from a radius-0 sphere', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [5, 5, 5], max: [5, 5, 5] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.001, 1000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    // Prove the guard (not the empty-cache early return) is what fires:
    // the cache DOES hold a sphere — a degenerate radius-0 one.
    expect(ctx.boundsCache.getSphere()).not.toBeNull();
    expect(ctx.boundsCache.getSphere()!.radius).toBe(0);

    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');
    updateDynamicFromCache(ctx);

    expect(updateSpy).not.toHaveBeenCalled();
    expect(camera.near).toBe(0.001);
    expect(camera.far).toBe(1000);
  });
});
