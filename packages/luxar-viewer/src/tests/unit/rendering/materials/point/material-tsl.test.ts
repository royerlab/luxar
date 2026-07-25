/**
 * Unit tests for PointTSLMaterial wrapper semantics.
 *
 * P8 three-geometry symmetry: mirrors `materials/line/material-tsl.test.ts`
 * and `materials/gsplat/material-tsl.test.ts`. PointTSLMaterial owns one
 * persistent `UniformNode` per shader input (the `tslNodes` table) and
 * exposes each through `material.uniforms` as an `IUniform`-shaped
 * getter/setter proxy (`proxyIUniform`). These tests pin that contract:
 *
 *   - Proxy writes to `material.uniforms.X.value` land directly on the
 *     wrapper-owned TSL node (the property that replaced the old
 *     `.onUpdate('render')` per-frame callback bridge).
 *   - `updateCameraParams` routes the shared camera-uniform math into
 *     the right uniforms (perspective and ortho).
 *   - `applyBlendingMode` transitions toggle the full blend state +
 *     the LUXAR_MAX_RGB_CONTRIBUTION define, matching the GLSL twin
 *     (`materials/point/blending-mode.test.ts`).
 *   - `clone()` preserves numerics / blending mode / colormap state and
 *     produces an independent uniform identity (a clone that shared
 *     TSLNodes with its source would leak uniform writes back).
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { PointTSLMaterial } from '../../../../../rendering/materials/point/material-tsl';
import {
  computePointSizeFactor,
  computeMaxPointSize,
} from '../../../../../rendering/materials/_shared/camera-uniforms';

/** Reach the private wrapper-owned node table (unit-test-only access). */
function nodesOf(mat: PointTSLMaterial): Record<string, { value: unknown }> {
  return (mat as unknown as { tslNodes: Record<string, { value: unknown }> }).tslNodes;
}

/** Minimal LUT that survives point colormap binding (matches the gsplat test fixture). */
function buildLut(): THREE.DataTexture {
  const lut = new THREE.DataTexture(
    new Uint8Array(256 * 2 * 4),
    256,
    2,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  lut.needsUpdate = true;
  return lut;
}

describe('PointTSLMaterial uniform proxies', () => {
  it('writes to material.uniforms.X.value land on the wrapper-owned TSL nodes', () => {
    const mat = new PointTSLMaterial();
    const nodes = nodesOf(mat);

    mat.uniforms.uOpacity.value = 0.42;
    mat.uniforms.radiusScale.value = 3.5;
    mat.uniforms.uIntensity.value = 2.0;
    mat.uniforms.uOffset.value = 0.125;

    expect(nodes.uOpacity.value).toBe(0.42);
    expect(nodes.radiusScale.value).toBe(3.5);
    expect(nodes.uIntensity.value).toBe(2.0);
    expect(nodes.uOffset.value).toBe(0.125);
  });

  it('update methods route through the proxies onto the nodes', () => {
    const mat = new PointTSLMaterial();
    const nodes = nodesOf(mat);

    mat.updateOpacity(0.7);
    mat.updateRadiusScale(12.0);
    mat.updateIntensity(1.5);
    mat.updateOffset(0.05);

    expect(nodes.uOpacity.value).toBeCloseTo(0.7, 6);
    expect(nodes.radiusScale.value).toBeCloseTo(12.0, 6);
    expect(nodes.uIntensity.value).toBeCloseTo(1.5, 6);
    expect(nodes.uOffset.value).toBeCloseTo(0.05, 6);
    expect(mat.getOpacity()).toBeCloseTo(0.7, 6);
  });

  it('uResolution proxy shares the Vector2 identity with the node (in-place .copy propagates)', () => {
    const mat = new PointTSLMaterial();
    const nodes = nodesOf(mat);
    expect(mat.uniforms.uResolution.value).toBe(nodes.uResolution.value);
  });
});

describe('PointTSLMaterial updateCameraParams', () => {
  it('perspective: derives pointSizeFactor / maxPointSize via the shared camera math', () => {
    const mat = new PointTSLMaterial();
    const fov = Math.PI / 3;
    const res = new THREE.Vector2(1600, 900);

    mat.updateCameraParams(fov, res, false, 0.25);

    expect(mat.uniforms.uIsOrtho.value).toBe(0);
    expect(mat.uniforms.uNearCull.value).toBe(0.25);
    expect(mat.uniforms.pointSizeFactor.value).toBeCloseTo(
      computePointSizeFactor(fov, res.y, false),
      5
    );
    expect(mat.uniforms.maxPointSize.value).toBeCloseTo(computeMaxPointSize(res.y), 5);
    const bound = mat.uniforms.uResolution.value as THREE.Vector2;
    expect(bound.x).toBe(1600);
    expect(bound.y).toBe(900);
  });

  it('ortho: sets uIsOrtho = 1 with the ortho size factor (no graph-specialized rebuild needed)', () => {
    const mat = new PointTSLMaterial();
    const graphBefore = mat.vertexNode;

    mat.updateCameraParams(2.0, new THREE.Vector2(64, 64), true);

    expect(mat.uniforms.uIsOrtho.value).toBe(1);
    expect(mat.uniforms.pointSizeFactor.value).toBeCloseTo(
      computePointSizeFactor(2.0, 64, true),
      5
    );
    // uIsOrtho is a RUNTIME uniform in the point graph (unlike the
    // line factory's compile-time config.isOrtho), so flipping the
    // camera mode must NOT rebuild the TSL graph.
    expect(mat.vertexNode).toBe(graphBefore);
  });

  it('accepts nearCull = 0 (no stale-value gate)', () => {
    const mat = new PointTSLMaterial();
    mat.updateCameraParams(Math.PI / 3, new THREE.Vector2(64, 64), false, 0);
    expect(mat.uniforms.uNearCull.value).toBe(0);
  });
});

describe('PointTSLMaterial.applyBlendingMode', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor and the contribution define', () => {
    const mat = new PointTSLMaterial();
    mat.applyBlendingMode('max');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
    expect(mat.userData.blendingMode).toBe('max');
  });

  it('max → additive resets the blend state and clears the contribution define', () => {
    const mat = new PointTSLMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('constructor max mode applies the contribution define from the very first graph build', () => {
    const mat = new PointTSLMaterial({ blendingMode: 'max' });
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.userData.blendingMode).toBe('max');
  });
});

describe('PointTSLMaterial clone', () => {
  it('preserves gamma, opacity, intensity, offset, radiusScale numerics through clone', () => {
    const original = new PointTSLMaterial({
      gamma: 2.2,
      opacity: 0.7,
      intensity: 1.5,
      offset: 0.25,
      radiusScale: 4.0,
    });

    const cloned = original.clone();

    expect(cloned.uniforms.uOpacity.value).toBeCloseTo(0.7, 5);
    expect(cloned.uniforms.uIntensity.value).toBeCloseTo(1.5, 5);
    expect(cloned.uniforms.uOffset.value).toBeCloseTo(0.25, 5);
    expect(cloned.uniforms.uInvGamma.value).toBeCloseTo(1 / 2.2, 5);
    expect(cloned.uniforms.radiusScale.value).toBeCloseTo(4.0, 5);
    expect(cloned.userData.gamma).toBeCloseTo(2.2, 5);
  });

  it('carries the camera-STATE uniforms (uIsOrtho, uNearCull, uResolution) onto the clone', () => {
    // A clone taken in ortho mode used to keep the constructor defaults
    // (perspective, nearCull 0.1, 1920×1080) until the next global
    // updateCameraParams broadcast — rendering the wrong projection
    // branch in the meantime. Lines clones are the reference.
    const original = new PointTSLMaterial();
    original.updateCameraParams(2.0, new THREE.Vector2(640, 480), /*isOrtho=*/ true, 0.42);

    const cloned = original.clone();

    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
    expect(cloned.uniforms.uNearCull.value).toBeCloseTo(0.42, 5);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).x).toBe(640);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).y).toBe(480);
  });

  it('resyncs camera-derived uniforms (pointSizeFactor, maxPointSize) from source onto clone', () => {
    const original = new PointTSLMaterial();
    original.updateCameraParams(Math.PI / 3, new THREE.Vector2(1600, 900), false);

    const cloned = original.clone();

    expect(cloned.uniforms.pointSizeFactor.value).toBeCloseTo(
      original.uniforms.pointSizeFactor.value as number,
      5
    );
    expect(cloned.uniforms.maxPointSize.value).toBeCloseTo(
      original.uniforms.maxPointSize.value as number,
      5
    );
  });

  it('preserves the max blending mode through clone', () => {
    const original = new PointTSLMaterial({ blendingMode: 'max' });
    const cloned = original.clone();

    expect(cloned.userData.blendingMode).toBe('max');
    expect(cloned.blending).toBe(THREE.CustomBlending);
    expect(cloned.blendEquation).toBe(THREE.MaxEquation);
    expect(cloned.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
  });

  it('preserves the USE_COLORMAP define + scalar range when source has a colormap texture', () => {
    const lut = buildLut();
    const original = new PointTSLMaterial({
      colormapTexture: lut,
      scalarRange: [0.1, 0.9],
    });
    expect('USE_COLORMAP' in (original.defines ?? {})).toBe(true);

    const cloned = original.clone();

    expect('USE_COLORMAP' in (cloned.defines ?? {})).toBe(true);
    expect(cloned.uniforms.uColormapTex?.value).toBe(lut);
    expect(cloned.uniforms.uScalarMin?.value).toBeCloseTo(0.1, 5);
    // scalarScale = 1/(max-min) = 1/0.8 = 1.25.
    expect(cloned.uniforms.uScalarScale?.value).toBeCloseTo(1.25, 5);

    lut.dispose();
  });

  it('clone of a default material has no colormap define and no scalar uniforms', () => {
    const original = new PointTSLMaterial();
    const cloned = original.clone();

    expect('USE_COLORMAP' in (cloned.defines ?? {})).toBe(false);
    expect(cloned.uniforms.uScalarMin).toBeUndefined();
    expect(cloned.uniforms.uScalarScale).toBeUndefined();
  });

  it('cloned material has independent uniform identity (mutation does not leak back)', () => {
    // The cheapest mutation-killer for "clone forgot to construct
    // fresh TSLNodes" — mirrors the gsplat sibling test.
    const original = new PointTSLMaterial();
    const cloned = original.clone();

    cloned.uniforms.uOpacity.value = 0.123;
    expect(original.uniforms.uOpacity.value).not.toBe(0.123);

    cloned.uniforms.radiusScale.value = 99;
    expect(original.uniforms.radiusScale.value).not.toBe(99);
  });
});

describe('PointTSLMaterial explicit depthTest/transparent overrides', () => {
  const makePointTex = (): THREE.DataTexture =>
    new THREE.DataTexture(new Float32Array(12), 3, 1, THREE.RGBAFormat, THREE.FloatType);

  it('survive the texture-swap graph rebuild (placeholder → real at first commit)', () => {
    // additive derives depthTest=false / transparent=true; the explicit
    // config says the opposite — a real divergence the rebuild's
    // factory tail (which re-applies mode-derived state) must not
    // silently revert. (An explicit depthTest:false on additive would
    // be vacuous — it EQUALS the mode-derived value.)
    const mat = new PointTSLMaterial({
      blendingMode: 'additive',
      depthTest: true,
      transparent: false,
    });
    expect(mat.depthTest).toBe(true);
    expect(mat.transparent).toBe(false);

    // Guaranteed rebuild: every node's first commit swaps the
    // placeholder point texture for the pool entry's real one.
    mat.updatePointTexture(makePointTex());
    expect(mat.depthTest).toBe(true);
    expect(mat.transparent).toBe(false);
    expect(mat.userData.depthTest).toBe(true);
  });

  it('an explicit applyBlendingMode call takes full ownership (overrides cleared)', () => {
    // Matches the GLSL twin: applyBlendingStateToMaterial re-derives
    // depthTest/transparent from the mode on every call, overwriting
    // any constructor override.
    const mat = new PointTSLMaterial({
      blendingMode: 'additive',
      depthTest: true,
      transparent: false,
    });
    mat.applyBlendingMode('additive'); // user-driven mode application

    expect(mat.depthTest).toBe(false); // mode-derived again
    expect(mat.transparent).toBe(true);

    // …and stays mode-derived across later rebuilds (overrides gone).
    mat.updatePointTexture(makePointTex());
    expect(mat.depthTest).toBe(false);
    expect(mat.transparent).toBe(true);
  });
});

describe('PointTSLMaterial updatePointTexture', () => {
  const makePointTex = (): THREE.DataTexture =>
    new THREE.DataTexture(new Float32Array(12), 3, 1, THREE.RGBAFormat, THREE.FloatType);

  it('rebinds to a new texture (graph rebuild) and no-ops on identical identity', () => {
    // Mirrors GSplatTSLMaterial.updateSplatTexture: TSL texture() nodes
    // are factory-time bound, so an identity CHANGE builds a fresh node
    // + reruns the factory (vertexNode identity changes), while the
    // identity-unchanged rebind — the per-commit hot path — must be a
    // complete no-op (no node churn, no graph rebuild).
    const mat = new PointTSLMaterial({});
    const tex = makePointTex();

    mat.updatePointTexture(tex);
    expect(mat.uniforms.uPointTex.value).toBe(tex);
    expect(mat.getPointTexture()).toBe(tex);

    // Identity-unchanged rebind: proxy AND graph stay untouched.
    const proxyAfterBind = mat.uniforms.uPointTex;
    const vertexNodeAfterBind = mat.vertexNode;
    mat.updatePointTexture(tex);
    expect(mat.uniforms.uPointTex).toBe(proxyAfterBind);
    expect(mat.vertexNode).toBe(vertexNodeAfterBind);

    // Identity change: rebinds (and the factory re-ran → new vertexNode).
    const tex2 = makePointTex();
    mat.updatePointTexture(tex2);
    expect(mat.uniforms.uPointTex.value).toBe(tex2);
    expect(mat.vertexNode).not.toBe(vertexNodeAfterBind);
  });

  it('clone carries the bound point texture', () => {
    const mat = new PointTSLMaterial({});
    const tex = makePointTex();
    mat.updatePointTexture(tex);
    const cloned = mat.clone();
    expect(cloned.uniforms.uPointTex.value).toBe(tex);
  });
});
