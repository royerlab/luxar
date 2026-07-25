/**
 * Unit tests for LineTSLMaterial clone semantics.
 *
 * Pins the clone contract for the `isOrtho` graph-specialized config.
 * The constructor builds the TSL graph using its default value
 * (perspective); the node-factory / camera updates flip `uIsOrtho`
 * post-construction. A naïve clone copies uniforms but drops the
 * rebuild, so the clone would silently render with the wrong
 * projection branch. These tests pin that the clone re-applies it.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';

describe('LineTSLMaterial clone', () => {
  it('preserves the orthographic uIsOrtho uniform value', () => {
    const original = new LineTSLMaterial();
    original.uniforms.uIsOrtho.value = 1;

    const cloned = original.clone();

    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
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

    // Identity change: rebinds (and the factory re-ran → new vertexNode).
    const tex2 = makeLineTex();
    mat.updateLineTexture(tex2);
    expect(mat.uniforms.uLineTex.value).toBe(tex2);
    expect(mat.vertexNode).not.toBe(vertexNodeAfterBind);
  });

  it('clone carries the bound line texture', () => {
    const mat = new LineTSLMaterial({});
    const tex = makeLineTex();
    mat.updateLineTexture(tex);
    const cloned = mat.clone();
    expect(cloned.uniforms.uLineTex.value).toBe(tex);
  });
});
