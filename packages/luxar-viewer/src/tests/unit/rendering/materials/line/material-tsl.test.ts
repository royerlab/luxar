/**
 * Unit tests for LineTSLMaterial clone semantics and its camera push. The
 * projection (pixel-width scale and ortho branch) is read per draw from
 * `cameraProjectionMatrix`, so neither is a uniform or a graph variant —
 * see ortho-from-projection.test.ts for the no-rebuild-on-flip contract.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';

describe('LineTSLMaterial clone', () => {
  it('updateCameraParams writes no projection uniform', () => {
    // The pixel-width scale and the ortho test are both read in the graph from
    // cameraProjectionMatrix, so the former uIsOrtho flag and perspective /
    // ortho line-scale uniforms are gone.
    const material = new LineTSLMaterial();
    material.updateCameraParams(new THREE.Vector2(800, 600), 0.25, 2);
    expect(material.uniforms.uIsOrtho).toBeUndefined();
    expect(material.uniforms.uNearCull.value).toBe(0.25);
    expect(material.uniforms.uPixelRatio.value).toBe(2);
    expect(material.uniforms.uPerspectiveLineScale).toBeUndefined();
    expect(material.uniforms.uOrthoLineScale).toBeUndefined();
  });

  it('does not introduce a LUXAR_SHARPNESS_TWO define (fast path removed)', () => {
    const original = new LineTSLMaterial();
    const cloned = original.clone();
    expect('LUXAR_SHARPNESS_TWO' in (cloned.defines ?? {})).toBe(false);
  });
});

describe('LineTSLMaterial explicit depthTest/transparent overrides', () => {
  const makeLineTex = (): THREE.DataTexture =>
    new THREE.DataTexture(new Float32Array(12), 3, 1, THREE.RGBAFormat, THREE.FloatType);

  it('survive the texture-swap graph rebuild (placeholder → real at first commit)', () => {
    // additive derives depthTest=false / transparent=true; the explicit
    // config says the opposite — a real divergence the rebuild's
    // factory tail (which re-applies mode-derived state) must not
    // silently revert. (An explicit depthTest:false on additive would
    // be vacuous — it EQUALS the mode-derived value.)
    const mat = new LineTSLMaterial({
      blendingMode: 'additive',
      depthTest: true,
      transparent: false,
    });
    expect(mat.depthTest).toBe(true);
    expect(mat.transparent).toBe(false);

    // Guaranteed rebuild: every node's first commit swaps the
    // placeholder line texture for the pool entry's real one.
    mat.updateLineTexture(makeLineTex());
    expect(mat.depthTest).toBe(true);
    expect(mat.transparent).toBe(false);
    expect(mat.userData.depthTest).toBe(true);
  });

  it('an explicit applyBlendingMode call takes full ownership (overrides cleared)', () => {
    // Matches the GLSL twin: applyBlendingStateToMaterial re-derives
    // depthTest/transparent from the mode on every call, overwriting
    // any constructor override.
    const mat = new LineTSLMaterial({
      blendingMode: 'additive',
      depthTest: true,
      transparent: false,
    });
    mat.applyBlendingMode('additive'); // user-driven mode application

    expect(mat.depthTest).toBe(false); // mode-derived again
    expect(mat.transparent).toBe(true);

    // …and stays mode-derived across later rebuilds (overrides gone).
    mat.updateLineTexture(makeLineTex());
    expect(mat.depthTest).toBe(false);
    expect(mat.transparent).toBe(true);
  });
});

describe('LineTSLMaterial updateLineTexture', () => {
  const makeLineTex = (): THREE.DataTexture =>
    new THREE.DataTexture(new Float32Array(12), 3, 1, THREE.RGBAFormat, THREE.FloatType);

  it('rebinds to a new texture (graph rebuild) and no-ops on identical identity', () => {
    // Mirrors PointTSLMaterial.updatePointTexture: TSL texture() nodes
    // are factory-time bound, so an identity CHANGE builds a fresh node
    // + reruns the factory (vertexNode identity changes), while the
    // identity-unchanged rebind — the per-commit hot path — must be a
    // complete no-op (no node churn, no graph rebuild).
    const mat = new LineTSLMaterial({});
    const tex = makeLineTex();

    mat.updateLineTexture(tex);
    expect(mat.uniforms.uLineTex.value).toBe(tex);
    expect(mat.getLineTexture()).toBe(tex);

    // Identity-unchanged rebind: proxy AND graph stay untouched.
    const proxyAfterBind = mat.uniforms.uLineTex;
    const vertexNodeAfterBind = mat.vertexNode;
    mat.updateLineTexture(tex);
    expect(mat.uniforms.uLineTex).toBe(proxyAfterBind);
    expect(mat.vertexNode).toBe(vertexNodeAfterBind);

    // Identity change: rebinds. A same-width, same-format texture is a
    // VALUE of the configuration's shared graph (shared-graph-tsl.ts), so
    // the graph itself is unchanged — the draw forwards the new texture.
    const tex2 = makeLineTex();
    mat.updateLineTexture(tex2);
    expect(mat.uniforms.uLineTex.value).toBe(tex2);
    expect(mat.vertexNode).toBe(vertexNodeAfterBind);
  });

  it('clone carries the bound line texture', () => {
    const mat = new LineTSLMaterial({});
    const tex = makeLineTex();
    mat.updateLineTexture(tex);
    const cloned = mat.clone();
    expect(cloned.uniforms.uLineTex.value).toBe(tex);
  });
});
