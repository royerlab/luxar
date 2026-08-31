/**
 * Unit tests for GPU picking materials.
 *
 * Verifies that picking materials instantiate correctly, implement
 * CameraAwareMaterial, and have the expected uniforms.
 */

// NOTE (#1352 flip): these pins exercise the QUAD (screen-space)
// primitive's shader internals, so constructions pass it explicitly —
// the session default is now 'capsule'. The quad pins go away with the
// primitive itself in the deletion PR.
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointPickingMaterial } from '../../../rendering/picking/point/material';
import { PointPickingTSLMaterial } from '../../../rendering/picking/point/material-tsl';
import { LinePickingMaterial } from '../../../rendering/picking/line/material';
import { LinePickingTSLMaterial } from '../../../rendering/picking/line/material-tsl';
import { GSplatPickingMaterial } from '../../../rendering/picking/gsplat/material';
import { GSplatPickingTSLMaterial } from '../../../rendering/picking/gsplat/material-tsl';

describe('PointPickingMaterial', () => {
  it('instantiates with correct nodeId uniform', () => {
    const material = new PointPickingMaterial({ nodeId: 42 });
    expect(material.uniforms.uNodeId.value).toBe(42);
    expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    material.dispose();
  });

  it('uses correct material settings for picking', () => {
    const material = new PointPickingMaterial({ nodeId: 1 });
    expect(material.transparent).toBe(false);
    expect(material.depthTest).toBe(true);
    expect(material.depthWrite).toBe(true);
    expect(material.blending).toBe(THREE.NoBlending);
    material.dispose();
  });

  it('applies radiusScale', () => {
    const material = new PointPickingMaterial({
      nodeId: 1,
      radiusScale: 0.5,
    });
    expect(material.uniforms.radiusScale.value).toBe(0.5);
    material.dispose();
  });

  it('implements updateCameraParams', () => {
    const material = new PointPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);
    const fov = (60 * Math.PI) / 180;

    material.updateCameraParams(fov, resolution, false);

    expect(material.uniforms.uIsOrtho.value).toBe(0);
    expect(material.uniforms.maxPointSize.value).toBe(540); // 1080 * 0.5
    expect(material.uniforms.pointSizeFactor.value).toBeGreaterThan(0);
    material.dispose();
  });

  it('handles orthographic camera params', () => {
    const material = new PointPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(800, 600);
    const frustumHeight = 10; // world units

    material.updateCameraParams(frustumHeight, resolution, true);

    expect(material.uniforms.uIsOrtho.value).toBe(1);
    expect(material.uniforms.pointSizeFactor.value).toBe(240); // 2 * 600 / (10 * 0.5)
    material.dispose();
  });

  // Mirrors GSplatPickingMaterial's explicit clone (inherited
  // Material.clone() calls the constructor with no config and throws;
  // three-geometry symmetry rule).
  it('clone preserves config and tuned uniforms (independent of source)', () => {
    const material = new PointPickingMaterial({ nodeId: 5, radiusScale: 0.5 });
    material.updateCameraParams(1.0, new THREE.Vector2(800, 600), true, 0.25);

    const cloned = material.clone();
    expect(cloned.uniforms.uNodeId.value).toBe(5);
    expect(cloned.uniforms.radiusScale.value).toBe(0.5);
    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
    expect(cloned.uniforms.uNearCull.value).toBe(0.25);
    expect(cloned.uniforms.pointSizeFactor.value).toBe(material.uniforms.pointSizeFactor.value);
    expect(cloned.uniforms.maxPointSize.value).toBe(material.uniforms.maxPointSize.value);
    expect(cloned.uniforms.uResolution.value.x).toBe(800);
    expect(cloned.uniforms.uResolution.value.y).toBe(600);

    // Clone is independent — mutating the source must not leak.
    material.updateRadiusScale(2.0);
    expect(cloned.uniforms.radiusScale.value).toBe(0.5);

    material.dispose();
    cloned.dispose();
  });
});

describe('PointPickingTSLMaterial', () => {
  it('clone preserves config and tuned uniforms (independent of source)', () => {
    // One-for-one mirror of the GLSL wrapper's clone test above
    // (GLSL ↔ TSL symmetry, same rule as the gsplat pick wrappers).
    const material = new PointPickingTSLMaterial({ nodeId: 5, radiusScale: 0.5 });
    material.updateCameraParams(1.0, new THREE.Vector2(800, 600), true, 0.25);

    const cloned = material.clone();
    expect(cloned.uniforms.uNodeId.value).toBe(5);
    expect(cloned.uniforms.radiusScale.value).toBe(0.5);
    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
    expect(cloned.uniforms.uNearCull.value).toBe(0.25);
    expect(cloned.uniforms.pointSizeFactor.value).toBe(material.uniforms.pointSizeFactor.value);
    expect(cloned.uniforms.maxPointSize.value).toBe(material.uniforms.maxPointSize.value);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).x).toBe(800);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).y).toBe(600);

    // Clone is independent — mutating the source must not leak.
    material.updateRadiusScale(2.0);
    expect(cloned.uniforms.radiusScale.value).toBe(0.5);

    material.dispose();
    cloned.dispose();
  });
});

describe('LinePickingMaterial', () => {
  it('instantiates with correct nodeId uniform', () => {
    const material = new LinePickingMaterial({ primitive: 'screen-space', nodeId: 7 });
    expect(material.uniforms.uNodeId.value).toBe(7);
    expect(material).toBeInstanceOf(THREE.ShaderMaterial);
    material.dispose();
  });

  it('uses correct material settings for picking', () => {
    const material = new LinePickingMaterial({ primitive: 'screen-space', nodeId: 1 });
    expect(material.transparent).toBe(false);
    expect(material.depthTest).toBe(true);
    expect(material.blending).toBe(THREE.NoBlending);
    expect(material.side).toBe(THREE.DoubleSide);
    material.dispose();
  });

  it('implements updateCameraParams', () => {
    const material = new LinePickingMaterial({ primitive: 'screen-space', nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false);

    expect(material.uniforms.uResolution.value.x).toBe(1920);
    expect(material.uniforms.uResolution.value.y).toBe(1080);
    material.dispose();
  });

  // The sharpness profile is now a shifted-truncated super-Gaussian
  // (beta = 2^(6s - 2)) with no per-buffer fast path, so the picking
  // shader carries no LUXAR_SHARPNESS_TWO define.
  it('GLSL line picking shader uses the super-Gaussian falloff, no LUXAR_SHARPNESS_TWO', () => {
    const material = new LinePickingMaterial({ primitive: 'screen-space', nodeId: 1 });
    expect(material.fragmentShader).toContain('exp2(6.0 * vSharpness - 2.0)');
    expect(material.fragmentShader).toContain('exp(-K * pow(p, beta))');
    expect(material.fragmentShader).not.toContain('LUXAR_SHARPNESS_TWO');
    expect('LUXAR_SHARPNESS_TWO' in (material.defines ?? {})).toBe(false);
    material.dispose();
  });

  // Cap factor must stay in lockstep with the visual shader: one ramp per
  // endpoint, each lifted by its own suppression, combined with min()
  // (issue #796 — mirrors the material-glsl.test.ts assertion).
  it('GLSL line picking shader uses per-endpoint cap ramps combined with min()', () => {
    const material = new LinePickingMaterial({ primitive: 'screen-space', nodeId: 1 });
    expect(material.fragmentShader).toContain('mix(0.5 + 0.5 * startRamp, 1.0, vCapSuppressStart)');
    expect(material.fragmentShader).toContain('mix(0.5 + 0.5 * endRamp, 1.0, vCapSuppressEnd)');
    expect(material.fragmentShader).toContain('min(startCap, endCap)');
    material.dispose();
  });

  // Pathological discard must be segment-constant, in lockstep with the
  // visual shader: rawPixelWidth varies across the shared quad's t=0/t=1
  // corners, so gating on it sentinels only half the quad and leaves a
  // visible wedge (issue #849). The fix gates on segMaxPixelWidth.
  it('GLSL line picking shader gates the pathological discard on the per-segment max width (issue #849)', () => {
    const material = new LinePickingMaterial({ primitive: 'screen-space', nodeId: 1 });
    expect(material.vertexShader).toContain('segMaxPixelWidth');
    expect(material.vertexShader).toContain('segMaxPixelWidth > maxPW * 2.0');
    expect(material.vertexShader).not.toContain('rawPixelWidth > maxPW * 2.0');
    material.dispose();
  });

  // Mirrors GSplatPickingMaterial's explicit clone (inherited
  // Material.clone() calls the constructor with no config and throws;
  // three-geometry symmetry rule).
  it('clone preserves config and tuned uniforms (independent of source)', () => {
    const material = new LinePickingMaterial({ primitive: 'screen-space', nodeId: 7 });
    material.updateCameraParams(10, new THREE.Vector2(800, 600), true, 0.25);

    const cloned = material.clone();
    expect(cloned.uniforms.uNodeId.value).toBe(7);
    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
    expect(cloned.uniforms.uNearCull.value).toBe(0.25);
    expect(cloned.uniforms.uMaxLinePixelWidth.value).toBe(300); // 600 * 0.5
    expect(cloned.uniforms.uOrthoLineScale.value).toBe(material.uniforms.uOrthoLineScale.value);
    expect(cloned.uniforms.uPerspectiveLineScale.value).toBe(
      material.uniforms.uPerspectiveLineScale.value
    );
    expect(cloned.uniforms.uResolution.value.x).toBe(800);
    expect(cloned.uniforms.uResolution.value.y).toBe(600);

    // Clone is independent — mutating the source must not leak.
    material.updateCameraParams(10, new THREE.Vector2(1920, 1080), true, 0.9);
    expect(cloned.uniforms.uNearCull.value).toBe(0.25);
    expect(cloned.uniforms.uResolution.value.x).toBe(800);

    material.dispose();
    cloned.dispose();
  });
});

describe('GSplatPickingMaterial', () => {
  it('instantiates with correct nodeId and tighter truncation', () => {
    const material = new GSplatPickingMaterial({ nodeId: 99 });
    expect(material.uniforms.uNodeId.value).toBe(99);
    // Tighter truncation: 1.5σ instead of 3.0σ
    expect(material.uniforms.uTruncate.value).toBe(1.5);
    expect(material.uniforms.uTruncateSq.value).toBe(2.25);
    material.dispose();
  });

  it('uses max projection mode for picking (no uProjectionMode uniform; shader hard-codes max)', () => {
    // The picking shader hard-codes max projection — it has no
    // sum-projection ray-integral path — so neither the GLSL nor the
    // TSL picking materials bind a `uProjectionMode` uniform.
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    expect(material.uniforms.uProjectionMode).toBeUndefined();
    material.dispose();
  });

  it('implements updateCameraParams with nearCull', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false, 0.5);

    expect(material.uniforms.uNearCull.value).toBe(0.5);
    expect(material.uniforms.uFx.value).toBeGreaterThan(0);
    material.dispose();
  });

  // rendering.md G5 fix: GSplatPickingMaterial previously only had
  // an instantiation test. Orthographic branch (uIsOrtho=1) is an
  // independent code path in `computeFocalLength` (frustumHeight
  // semantics, not tan(fov/2)), so it must be exercised separately
  // to kill mutations to the `isOrtho ? 1 : 0` flag and the fy=fx
  // assignment.
  it('handles orthographic camera params (uIsOrtho=1)', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(800, 600);
    const frustumHeight = 10; // world units, ortho semantics

    material.updateCameraParams(frustumHeight, resolution, true);

    expect(material.uniforms.uIsOrtho.value).toBe(1);
    // For ortho: focal = resolution.y / frustumHeight = 600 / 10 = 60.
    // Both fx and fy must be set identically (square pixels assumption).
    expect(material.uniforms.uFx.value).toBeCloseTo(60, 5);
    expect(material.uniforms.uFy.value).toBeCloseTo(60, 5);
    expect(material.uniforms.uFx.value).toBe(material.uniforms.uFy.value);
    material.dispose();
  });

  it('updateCameraParams without nearCull leaves uNearCull at its default', () => {
    // Pins the contract that nearCull is opt-in (mirrors LinePicking
    // and the visual GSplat material).
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    const initialNearCull = material.uniforms.uNearCull.value;
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false);

    expect(material.uniforms.uNearCull.value).toBe(initialNearCull);
    material.dispose();
  });

  // Front-most-wins pick depth for normal-mode gsplats (S2): under
  // depth-sorted alpha-over the user sees an occluding surface, so the
  // pick shader must write real projected depth instead of
  // brightness-as-depth. Default OFF — commutative modes keep
  // brightest-wins.
  it('defaults to brightness-as-depth (uSurfaceDepth = 0)', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    expect(material.uniforms.uSurfaceDepth.value).toBe(0);
    material.dispose();
  });

  it('setSurfacePickDepth toggles uSurfaceDepth 0 ↔ 1', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    material.setSurfacePickDepth(true);
    expect(material.uniforms.uSurfaceDepth.value).toBe(1);
    material.setSurfacePickDepth(false);
    expect(material.uniforms.uSurfaceDepth.value).toBe(0);
    material.dispose();
  });

  it('filters picks with the same compact class index as the visual material', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    material.updateLabelFilter(4.9);

    expect(material.uniforms.uLabelFilterIndex.value).toBe(4);
    expect(material.vertexShader).toContain('int(aLabelIndex + 0.5) != uLabelFilterIndex');
    expect(material.clone().uniforms.uLabelFilterIndex.value).toBe(4);
  });

  it('fragment shader selects real depth vs brightness-as-depth on uSurfaceDepth', () => {
    const material = new GSplatPickingMaterial({ nodeId: 1 });
    expect(material.fragmentShader).toContain('uniform int uSurfaceDepth;');
    expect(material.fragmentShader).toContain(
      'gl_FragDepth = (uSurfaceDepth == 1) ? gl_FragCoord.z : 1.0 - brightness;'
    );
    material.dispose();
  });

  it('clone preserves uSurfaceDepth (and the other tuned uniforms)', () => {
    const material = new GSplatPickingMaterial({ nodeId: 5 });
    material.setSurfacePickDepth(true);
    material.uniforms.uCov2DDilation.value = 0.7;

    const cloned = material.clone();
    expect(cloned.uniforms.uNodeId.value).toBe(5);
    expect(cloned.uniforms.uSurfaceDepth.value).toBe(1);
    expect(cloned.uniforms.uCov2DDilation.value).toBe(0.7);

    // Clone is independent — flipping the source must not leak.
    material.setSurfacePickDepth(false);
    expect(cloned.uniforms.uSurfaceDepth.value).toBe(1);

    material.dispose();
    cloned.dispose();
  });
});

// rendering.md G3, G4 fix: GSplatPickingTSLMaterial had ZERO direct
// tests despite being referenced by material-manager/factories.ts.
// Mirrors the LinePickingTSLMaterial block above one-for-one — same
// nodeId / camera-params / ortho-branch coverage (project memory
// "three-geometry symmetry rule": Points/Lines/GSplats parallel tests).
describe('GSplatPickingTSLMaterial', () => {
  it('instantiates with correct nodeId uniform', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 99 });
    expect(material.uniforms.uNodeId.value).toBe(99);
    // Tighter truncation: 1.5σ — must match GLSL picking path.
    expect(material.uniforms.uTruncate.value).toBe(1.5);
    expect(material.uniforms.uTruncateSq.value).toBe(2.25);
    material.dispose();
  });

  it('uses max projection mode (no uProjectionMode uniform)', () => {
    // Symmetric with GSplatPickingMaterial (GLSL): picking shader
    // hard-codes max projection; uProjectionMode is intentionally
    // NOT exposed (see material-tsl.ts module preamble).
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    expect(material.uniforms.uProjectionMode).toBeUndefined();
    material.dispose();
  });

  it('updateCameraParams (perspective) sets fx=fy and uIsOrtho=0', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(1920, 1080);

    material.updateCameraParams(1.0, resolution, false, 0.5);

    expect(material.uniforms.uIsOrtho.value).toBe(0);
    expect(material.uniforms.uNearCull.value).toBe(0.5);
    expect(material.uniforms.uFx.value).toBeGreaterThan(0);
    expect(material.uniforms.uFx.value).toBe(material.uniforms.uFy.value);
    material.dispose();
  });

  it('updateCameraParams (orthographic) sets uIsOrtho=1 and matching ortho focal', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    const resolution = new THREE.Vector2(800, 600);
    const frustumHeight = 10;

    material.updateCameraParams(frustumHeight, resolution, true);

    expect(material.uniforms.uIsOrtho.value).toBe(1);
    // Mirror the GLSL ortho test: focal = res.y / frustumHeight = 60.
    expect(material.uniforms.uFx.value).toBeCloseTo(60, 5);
    expect(material.uniforms.uFy.value).toBeCloseTo(60, 5);
    material.dispose();
  });

  it('updateCameraParams without nearCull preserves uNearCull default', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    const initial = material.uniforms.uNearCull.value;
    material.updateCameraParams(1.0, new THREE.Vector2(800, 600), false);
    expect(material.uniforms.uNearCull.value).toBe(initial);
    material.dispose();
  });

  it('uniform proxy writes land on the underlying TSLNode (no .onUpdate bridge)', () => {
    // Documents the IUniform-proxy contract called out in the TSL
    // material's module preamble: mutating uniforms.uX.value must
    // mutate node.value directly. Symmetric with how LinePickingTSL
    // uniforms behave — and a cheap mutation-killer for the proxy.
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    material.uniforms.uNodeId.value = 42;
    expect(material.uniforms.uNodeId.value).toBe(42);
    material.dispose();
  });

  // Surface-pick depth — one-for-one mirror of the GLSL wrapper block
  // (three-geometry/material symmetry rule: same names, same surface).
  it('defaults to brightness-as-depth (uSurfaceDepth = 0)', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    expect(material.uniforms.uSurfaceDepth.value).toBe(0);
    material.dispose();
  });

  it('setSurfacePickDepth toggles uSurfaceDepth 0 ↔ 1', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 1 });
    material.setSurfacePickDepth(true);
    expect(material.uniforms.uSurfaceDepth.value).toBe(1);
    material.setSurfacePickDepth(false);
    expect(material.uniforms.uSurfaceDepth.value).toBe(0);
    material.dispose();
  });

  it('clone preserves uSurfaceDepth (and the other tuned uniforms)', () => {
    const material = new GSplatPickingTSLMaterial({ nodeId: 5 });
    material.setSurfacePickDepth(true);
    material.uniforms.uCov2DDilation.value = 0.7;

    const cloned = material.clone();
    expect(cloned.uniforms.uNodeId.value).toBe(5);
    expect(cloned.uniforms.uSurfaceDepth.value).toBe(1);
    expect(cloned.uniforms.uCov2DDilation.value).toBe(0.7);

    // Clone is independent — flipping the source must not leak.
    material.setSurfacePickDepth(false);
    expect(cloned.uniforms.uSurfaceDepth.value).toBe(1);

    material.dispose();
    cloned.dispose();
  });

  it('both wrappers expose the same setSurfacePickDepth surface (GLSL ↔ TSL symmetry)', () => {
    const glsl = new GSplatPickingMaterial({ nodeId: 1 });
    const tsl = new GSplatPickingTSLMaterial({ nodeId: 1 });
    expect(typeof glsl.setSurfacePickDepth).toBe('function');
    expect(typeof tsl.setSurfacePickDepth).toBe('function');
    expect(glsl.uniforms.uSurfaceDepth.value).toBe(tsl.uniforms.uSurfaceDepth.value);
    glsl.dispose();
    tsl.dispose();
  });
});

describe('LinePickingTSLMaterial', () => {
  it('clone preserves config and tuned uniforms (independent of source)', () => {
    // One-for-one mirror of the GLSL wrapper's clone test above
    // (GLSL ↔ TSL symmetry, same rule as the gsplat pick wrappers).
    // Ortho camera params also exercise the clone's graph rebuild on
    // the copied projection mode (the pick graph is JS-specialized).
    const material = new LinePickingTSLMaterial({ nodeId: 7 });
    material.updateCameraParams(10, new THREE.Vector2(800, 600), true, 0.25);

    const cloned = material.clone();
    expect(cloned.uniforms.uNodeId.value).toBe(7);
    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
    expect(cloned.uniforms.uNearCull.value).toBe(0.25);
    expect(cloned.uniforms.uMaxLinePixelWidth.value).toBe(300); // 600 * 0.5
    expect(cloned.uniforms.uOrthoLineScale.value).toBe(material.uniforms.uOrthoLineScale.value);
    expect(cloned.uniforms.uPerspectiveLineScale.value).toBe(
      material.uniforms.uPerspectiveLineScale.value
    );
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).x).toBe(800);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).y).toBe(600);

    // Clone is independent — mutating the source must not leak.
    material.updateCameraParams(10, new THREE.Vector2(1920, 1080), true, 0.9);
    expect(cloned.uniforms.uNearCull.value).toBe(0.25);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).x).toBe(800);

    material.dispose();
    cloned.dispose();
  });
});

describe('LinePickingTSLMaterial sharpness', () => {
  // The sharpness fast path (LUXAR_SHARPNESS_TWO / setSharpnessAllTwo)
  // was removed when the perpendicular profile became a shifted-truncated
  // super-Gaussian (beta = 2^(6s - 2)). The material no longer exposes a
  // setter and carries no such define.

  it('exposes no setSharpnessAllTwo and no LUXAR_SHARPNESS_TWO define', () => {
    const material = new LinePickingTSLMaterial({ nodeId: 1 });
    expect(
      (material as unknown as { setSharpnessAllTwo?: unknown }).setSharpnessAllTwo
    ).toBeUndefined();
    expect('LUXAR_SHARPNESS_TWO' in (material.defines ?? {})).toBe(false);
    material.dispose();
  });
});

// [rendering.md/O3][P10] computePickBufferSize tests live in their canonical
// location at tests/unit/rendering/picking/picking-system.test.ts. The
// previous duplicate block here covered the same algorithm with overlapping
// inputs; all unique cases (asymmetric cap, floor-fractional, negative-clamp)
// have been merged into the canonical file.

/**
 * The pick pass emits `vElementId` from the SAME draw-slot → storage-slot
 * index as the visual pass, so it must read whichever ordering buffer is
 * currently active. The depth-sort coordinator pushes that through
 * `applySortedIndexSlotToMaterial`, which reaches a material ONLY via
 * `material.uniforms.uSortedIndexSlot`.
 *
 * REGRESSION GUARD. The three TSL pick materials originally shipped
 * without this uniform: their `tslNodes` literal omitted it, an
 * `as …PickTSLNodes` cast at the factory call silenced the type error,
 * and `sortedIndexNode(undefined)` folds to `int(0)` — a CONSTANT. So
 * TSL picking silently resolved every hover against `aSortedIndex`
 * whatever the active slot was, i.e. against a stale permutation after
 * any flip. The casts are gone (so an omitted NODE is now a compile
 * error), but `uniforms` is a `Record<string, IUniform>`, so an omitted
 * PROXY would still be silent — hence this test.
 */
describe('picking materials — depth-sort ordering slot', () => {
  const cases: Array<
    [string, () => { uniforms: Record<string, THREE.IUniform>; dispose(): void }]
  > = [
    ['PointPickingMaterial', () => new PointPickingMaterial({ nodeId: 1 })],
    ['PointPickingTSLMaterial', () => new PointPickingTSLMaterial({ nodeId: 1 })],
    [
      'LinePickingMaterial',
      () => new LinePickingMaterial({ primitive: 'screen-space', nodeId: 1 }),
    ],
    ['LinePickingTSLMaterial', () => new LinePickingTSLMaterial({ nodeId: 1 })],
    ['GSplatPickingMaterial', () => new GSplatPickingMaterial({ nodeId: 1 })],
    ['GSplatPickingTSLMaterial', () => new GSplatPickingTSLMaterial({ nodeId: 1 })],
  ];

  for (const [name, make] of cases) {
    it(`${name} exposes a writable uSortedIndexSlot uniform defaulting to 0`, () => {
      const material = make();
      const uniform = material.uniforms.uSortedIndexSlot;
      expect(uniform, `${name}.uniforms.uSortedIndexSlot is missing`).toBeDefined();
      // Default 0 matches a freshly attached geometry's slot, so a node
      // that never sorts is consistent without anyone pushing anything.
      expect(uniform.value).toBe(0);
      // Writable THROUGH the record: the TSL wrappers expose a
      // getter/setter proxy onto the UniformNode, not a plain object, so
      // a read-back is the only proof the write actually lands.
      uniform.value = 1;
      expect(material.uniforms.uSortedIndexSlot.value).toBe(1);
      material.dispose();
    });

    it(`${name} clone() preserves a non-default uSortedIndexSlot`, () => {
      // A clone taken while the geometry draws from slot 1 must not fall
      // back to the constructor default 0 — it would read the stale
      // ordering buffer until the coordinator's next per-frame re-assert
      // (and the settle-scheduled pick pass can render before that).
      const material = make() as ReturnType<typeof make> & {
        clone(): { uniforms: Record<string, THREE.IUniform>; dispose(): void };
      };
      material.uniforms.uSortedIndexSlot.value = 1;
      const cloned = material.clone();
      expect(cloned.uniforms.uSortedIndexSlot.value).toBe(1);
      material.dispose();
      cloned.dispose();
    });
  }
});
