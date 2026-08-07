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
  MAX_NEAR_FAR_RATIO,
  minNearForRadius,
  SPHERE_SAFETY_EXPANSION,
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

  // Every comparison against NaN is false, so `near >= far` does NOT reject a
  // NaN — it would sail into the projection matrix and blank the view with no
  // diagnostic. Reachable from snapshot-restore (which writes a captured pair
  // straight back) and from any caller bypassing validateRenderingSettings.
  // Infinity is refused for the same reason: it makes the matrix degenerate
  // rather than meaning "see everything".
  it.each([
    ['NaN near', NaN, 1000],
    ['NaN far', 0.1, NaN],
    ['both NaN', NaN, NaN],
    ['Infinite far', 0.1, Infinity],
    ['Infinite near', Infinity, 1000],
    ['-Infinity near', -Infinity, 1000],
    ['zero near', 0, 1000],
    ['negative near', -2, -1],
    ['negative near, positive far', -2, 1000],
  ])('refuses invalid planes (%s) without mutating the camera', (_label, near, far) => {
    const camera = makeCamera(new THREE.Vector3(), 1, 1000);
    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');
    applyClippingPlanes(camera, near, far);
    expect(camera.near).toBe(1);
    expect(camera.far).toBe(1000);
    expect(updateSpy).not.toHaveBeenCalled();
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

  // The auto-adjust path forwards the same ortho opt-out as the per-frame one
  // (it runs once on scene load, and a scene can be loaded while already in
  // ortho mode). Covered separately because the two paths are independent
  // code — the dynamic-path ortho test would not catch this one regressing.
  it('drops the ratio bound for an orthographic camera on the load path', () => {
    const metadataBounds: BoundingBox = {
      min: { x: -28.9, y: -28.9, z: -28.9 },
      max: { x: 28.9, y: 28.9, z: 28.9 },
    };
    const sphere = boundingBoxToSphere(metadataBounds);
    const R = sphere.radius * SPHERE_SAFETY_EXPANSION;

    // Camera INSIDE the sphere, where the floor is what sets near.
    const ortho = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 1000);
    ortho.position.set(0, 0, 8.5);
    ortho.updateProjectionMatrix();
    expect(8.5).toBeLessThan(R); // precondition: the floor really binds

    const { ctx } = makeCtx({
      camera: ortho as unknown as THREE.PerspectiveCamera,
      metadataBounds,
    });
    const { near } = autoAdjustFromBounds(ctx);

    expect(near).toBe(minNearForRadius(R));
    expect(near).toBe(
      calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: 8.5 }, false).near
    );
    // And strictly below what perspective would have used at the same pose.
    expect(near).toBeLessThan(
      calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: 8.5 }, true).near
    );
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

  // The reported z-fighting regression, at the pose that produced it: a
  // diagonal-100 scene (the viewer's own normalized scale) viewed from 8.5
  // units off centre — well inside the bounding sphere (R = 52.5), which is
  // simply "zoomed in". This used to write near = 1.05e-4 / far = 61, a
  // 580,000:1 ratio that quantizes 24-bit depth to ~4e-2 WORLD UNITS at the
  // orbit target. Pins both the ratio bound and the resulting depth
  // resolution, since the ratio alone does not say whether the scene is
  // actually renderable.
  it('keeps depth quantization usable when zoomed inside the bounding sphere', () => {
    const scene = new THREE.Scene();
    // diagonal 100 → radius 50 → expanded R = 52.5.
    const h = 100 / (2 * Math.sqrt(3));
    scene.userData = { positionBounds: { min: [-h, -h, -h], max: [h, h, h] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 8.5), 0.1, 1000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    updateDynamicFromCache(ctx);

    // Precondition: the camera really is inside the sphere (otherwise this
    // test would silently exercise the outside branch and prove nothing).
    const sphere = ctx.boundsCache.getSphere()!;
    expect(8.5).toBeLessThan(sphere.radius * SPHERE_SAFETY_EXPANSION);

    expect(camera.far).toBeCloseTo(61, 0);
    expect(camera.far / camera.near).toBeCloseTo(MAX_NEAR_FAR_RATIO, 6);

    // Δz(d) = d² · (far − near) / (near · far) · 2⁻²⁴ on the 24-bit depth
    // renderbuffer three allocates for the HDR target. At the orbit target
    // this must stay far below the scale of visible scene features.
    const d = 8.5;
    const deltaZ =
      ((d * d * (camera.far - camera.near)) / (camera.near * camera.far)) * Math.pow(2, -24);
    expect(deltaZ).toBeLessThan(1e-3);
    // And specifically ~1000x better than the pre-fix 2e-6·R floor gave.
    const deltaZBefore =
      ((d * d * (camera.far - 1.05e-4)) / (1.05e-4 * camera.far)) * Math.pow(2, -24);
    expect(deltaZBefore / deltaZ).toBeGreaterThan(100);
  });

  // Recovery from an already-poisoned camera. The change gate compares
  // `Math.abs(camera.near - near) / camera.near`, which is NaN — hence false —
  // once `camera.near` is NaN, so a gate WITHOUT the non-finite escape hatch
  // locks the camera at NaN permanently. Both planes are poisoned here on
  // purpose: with only `near` poisoned, `farChanged` alone would reopen the
  // gate and the test would pass either way (measured — that spelling did not
  // detect removal of the escape hatch).
  it('recovers a camera whose near AND far are already NaN', () => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, 10] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.1, 1000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    camera.near = NaN;
    camera.far = NaN;
    updateDynamicFromCache(ctx);

    // Bounds are healthy, so a finite pair must be written back.
    expect(Number.isFinite(camera.near)).toBe(true);
    expect(Number.isFinite(camera.far)).toBe(true);
    expect(camera.near).toBeGreaterThan(0);
    expect(camera.far).toBeGreaterThan(camera.near);
  });

  // Each escape hatch, individually. With BOTH planes poisoned either hatch
  // alone reopens the gate, so that test cannot tell which one works
  // (measured: removing either singly still passed). Here the OTHER plane is
  // pre-set to exactly the value the formula will produce, so its own
  // changed-check is false and only the hatch under test can unstick the
  // camera.
  it.each([['near'], ['far']] as const)(
    'the %s non-finite escape hatch is individually required',
    (poisoned) => {
      const scene = new THREE.Scene();
      scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, 10] } };
      const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.1, 1000);
      const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
      ctx.boundsCache.ensure(scene);
      const expected = calculateClippingPlanesFromSphere(ctx.boundsCache.getSphere()!, {
        x: 0,
        y: 0,
        z: 100,
      });

      // Pin the non-poisoned plane so ONLY the poisoned one can reopen the gate.
      camera.near = poisoned === 'near' ? NaN : expected.near;
      camera.far = poisoned === 'far' ? NaN : expected.far;

      updateDynamicFromCache(ctx);

      expect(Number.isFinite(camera.near)).toBe(true);
      expect(Number.isFinite(camera.far)).toBe(true);
      expect(camera.near).toBeCloseTo(expected.near, 10);
      expect(camera.far).toBeCloseTo(expected.far, 10);
    }
  );

  // A FINITE but non-positive current plane is the sticky case a plain
  // `Number.isFinite` hatch misses: `Math.abs(current - near) / current` is
  // NEGATIVE for a negative `current`, so the > 0.001 gate is false for every
  // possible new value and the camera stays poisoned forever. `restoreCamera`
  // (`core/app/snapshot/viewer-snapshot.ts`) writes a snapshot's planes onto
  // the camera verbatim, so such a pair really can arrive. As above, the
  // other plane is pinned to the value the formula will produce so only the
  // hatch under test can unstick the camera.
  it.each([['near'], ['far']] as const)('recovers a camera whose %s is negative', (poisoned) => {
    const bad = -1;
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, 10] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.1, 1000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);
    const expected = calculateClippingPlanesFromSphere(ctx.boundsCache.getSphere()!, {
      x: 0,
      y: 0,
      z: 100,
    });

    camera.near = poisoned === 'near' ? bad : expected.near;
    camera.far = poisoned === 'far' ? bad : expected.far;

    updateDynamicFromCache(ctx);

    expect(camera.near).toBeCloseTo(expected.near, 10);
    expect(camera.far).toBeCloseTo(expected.far, 10);
  });

  // INFINITE bounds are the case where the non-finite guard on the COMPUTED
  // pair is load-bearing: near and far are both Infinity, and
  // `Math.abs(0.1 - Infinity) / 0.1` really is > 0.001, so the change gate
  // fires and an unguarded path would write Infinity into the projection
  // matrix. (NaN bounds are also refused, but the gate blocks those anyway —
  // measured — so they cannot prove the guard.)
  it.each([
    ['Infinity', Infinity],
    ['NaN', NaN],
  ])('refuses to write a non-finite pair from %s bounds', (_label, poison) => {
    const scene = new THREE.Scene();
    scene.userData = { positionBounds: { min: [-10, -10, -10], max: [10, 10, poison] } };
    const camera = makeCamera(new THREE.Vector3(0, 0, 100), 0.1, 1000);
    const { ctx } = makeCtx({ camera, scene, metadataBounds: null });
    ctx.boundsCache.ensure(scene);

    const updateSpy = vi.spyOn(camera, 'updateProjectionMatrix');
    updateDynamicFromCache(ctx);

    expect(updateSpy).not.toHaveBeenCalled();
    expect(camera.near).toBe(0.1);
    expect(camera.far).toBe(1000);
  });

  // Ortho must NOT get the perspective ratio bound (see MAX_NEAR_FAR_RATIO):
  // linear depth means no precision gain, and no `perspectiveNearFade` means
  // the raised floor really would clip drawn geometry. Both clipping paths
  // read the LIVE camera, so this also covers the V-key projection swap —
  // the stale-camera bug class from #573.
  it('drops the ratio bound when the live camera is orthographic', () => {
    const scene = new THREE.Scene();
    const h = 100 / (2 * Math.sqrt(3));
    scene.userData = { positionBounds: { min: [-h, -h, -h], max: [h, h, h] } };

    const persp = makeCamera(new THREE.Vector3(0, 0, 8.5), 0.1, 1000);
    const { ctx: perspCtx } = makeCtx({ camera: persp, scene, metadataBounds: null });
    perspCtx.boundsCache.ensure(scene);
    updateDynamicFromCache(perspCtx);
    const perspNear = persp.near;

    const ortho = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 1000);
    ortho.position.set(0, 0, 8.5);
    ortho.updateProjectionMatrix();
    const { ctx: orthoCtx } = makeCtx({
      camera: ortho as unknown as THREE.PerspectiveCamera,
      scene,
      metadataBounds: null,
    });
    orthoCtx.boundsCache.ensure(scene);
    updateDynamicFromCache(orthoCtx);

    // Same pose, same bounds, same far — only the projection differs.
    expect(ortho.far).toBeCloseTo(persp.far, 10);
    // Perspective is ratio-bounded; ortho keeps the much smaller scale floor.
    expect(persp.far / perspNear).toBeCloseTo(MAX_NEAR_FAR_RATIO, 6);

    // PARITY for the ortho arm too. The per-frame path re-implements the pure
    // helper's math for the zero-allocation contract, so every parameter of
    // that math needs a parity anchor or the duplication drifts — the
    // perspective-only parity test below would not have caught an ortho arm
    // that forgot to forward the flag.
    const sphere = orthoCtx.boundsCache.getSphere()!;
    const expectedOrtho = calculateClippingPlanesFromSphere(sphere, { x: 0, y: 0, z: 8.5 }, false);
    expect(ortho.near).toBe(expectedOrtho.near);
    expect(ortho.far).toBeCloseTo(expectedOrtho.far, 10);
    expect(ortho.near).toBeLessThan(perspNear);
    expect(ortho.near).toBe(
      minNearForRadius(orthoCtx.boundsCache.getSphere()!.radius * SPHERE_SAFETY_EXPANSION)
    );
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
