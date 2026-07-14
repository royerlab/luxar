/**
 * Unit tests for `applyLensDistortion()` in `picking-system/lens-distortion.ts`.
 *
 * The TypeScript implementation must stay numerically identical to the
 * GLSL `applyDistortion()` in `mega/shader.glsl.ts:117-128`, otherwise
 * mouse coordinates won't index into the correct pick-buffer pixel when
 * post-processing lens distortion is active.
 *
 * The "GLSL parity" test ports the shader formula directly into the test
 * to lock in the contract — if either side drifts, this fails.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  applyLensDistortion,
  type LensDistortionParams,
  type UVScratch,
} from '../../../../../rendering/picking/picking-system/lens-distortion';

/** Identity intrinsics: focal length (1, 1), centred principal point, no skew, no distortion. */
function identityParams(): LensDistortionParams {
  return {
    distortion: new THREE.Vector2(0, 0),
    focalLength: new THREE.Vector2(1, 1),
    principalPoint: new THREE.Vector2(0, 0),
    skew: 0,
  };
}

function makeScratch(): UVScratch {
  return { x: 0, y: 0 };
}

describe('applyLensDistortion', () => {
  it('returns the input UV unchanged for identity parameters', () => {
    const params = identityParams();
    const out = makeScratch();
    // Sample a few representative UVs across the image
    for (const [u, v] of [
      [0.25, 0.25],
      [0.5, 0.5],
      [0.75, 0.25],
      [0.1, 0.9],
    ]) {
      applyLensDistortion(u, v, params, out);
      expect(out.x).toBeCloseTo(u, 6);
      expect(out.y).toBeCloseTo(v, 6);
    }
  });

  it('fixes the image centre (0.5, 0.5) regardless of distortion coefficients', () => {
    // At UV = (0.5, 0.5), normalised xn = yn = 0, so r² = 0 and the
    // radial distortion term vanishes. Output must be (0.5, 0.5).
    const params: LensDistortionParams = {
      distortion: new THREE.Vector2(0.5, -0.3), // arbitrary non-zero
      focalLength: new THREE.Vector2(1, 1),
      principalPoint: new THREE.Vector2(0, 0),
      skew: 0,
    };
    const out = makeScratch();
    applyLensDistortion(0.5, 0.5, params, out);
    expect(out.x).toBeCloseTo(0.5, 6);
    expect(out.y).toBeCloseTo(0.5, 6);
  });

  it('is the flip-CONJUGATE of the GLSL reference (parity contract with mega/shader.glsl.ts)', () => {
    // GLSL formula from mega/shader.glsl.ts (applyDistortion), ported
    // verbatim INCLUDING the negated principalPoint.y / skew — the GLSL
    // runs under WebGLRenderer's BOTTOM-UP fullscreen uv and applies
    // the exact y-flip conjugation of this file's canonical TOP-DOWN
    // map by negating the two flip-odd intrinsics:
    //   vec2 xn = 2.0 * (uv - 0.5);
    //   float r2 = dot(xn, xn);
    //   vec3 xDistorted = vec3((1.0 + distortionCoeff * r2) * xn, 1.0);
    //   mat3 kk = mat3(
    //     vec3(uFocalLength.x, 0.0, 0.0),
    //     vec3(-uSkew * uFocalLength.x, uFocalLength.y, 0.0),
    //     vec3(uPrincipalPoint.x, -uPrincipalPoint.y, 1.0)
    //   );
    //   return (kk * xDistorted).xy * 0.5 + 0.5;
    function glslReference(
      u: number,
      v: number,
      params: LensDistortionParams
    ): { x: number; y: number } {
      const xnX = 2.0 * (u - 0.5);
      const xnY = 2.0 * (v - 0.5);
      const r2 = xnX * xnX + xnY * xnY;
      const xdX = (1.0 + params.distortion.x * r2) * xnX;
      const xdY = (1.0 + params.distortion.y * r2) * xnY;
      // mat3 stored column-major in GLSL; compute (kk · xDistorted).xy
      const kkX =
        params.focalLength.x * xdX -
        params.skew * params.focalLength.x * xdY +
        params.principalPoint.x;
      const kkY = params.focalLength.y * xdY - params.principalPoint.y;
      return { x: kkX * 0.5 + 0.5, y: kkY * 0.5 + 0.5 };
    }

    // Deliberately LARGE flip-odd intrinsics — the whole point of the
    // conjugation contract is the ppy/skew terms that a same-space
    // textual-identity test is blind to.
    const params: LensDistortionParams = {
      distortion: new THREE.Vector2(0.2, 0.15),
      focalLength: new THREE.Vector2(0.95, 0.95),
      principalPoint: new THREE.Vector2(0.02, 0.3),
      skew: 0.05,
    };

    const samples = [
      [0.1, 0.1],
      [0.3, 0.7],
      [0.5, 0.5],
      [0.7, 0.3],
      [0.9, 0.9],
    ];
    const out = makeScratch();
    for (const [u, v] of samples) {
      // Same physical screen point: top-down (u, v) here, bottom-up
      // (u, 1-v) in the GLSL. The sampled source points must coincide:
      //   flipY(glsl(u, 1 - v)) === canonical(u, v)
      applyLensDistortion(u, v, params, out);
      const ref = glslReference(u, 1 - v, params);
      expect(ref.x).toBeCloseTo(out.x, 6);
      expect(1 - ref.y).toBeCloseTo(out.y, 6);
    }
  });

  it('GLSL reference degenerates to the canonical formula when ppy = skew = 0 (baseline invariance)', () => {
    // With both flip-odd intrinsics at zero the negations are no-ops —
    // the GLSL matrix is bit-identical to the historical one, which is
    // why the conjugation fix cannot move any shipped visual baseline
    // (every shipped lens preset has principalPointY = 0, skew = 0).
    const params: LensDistortionParams = {
      distortion: new THREE.Vector2(0.35, 0.35),
      focalLength: new THREE.Vector2(0.9, 0.9),
      principalPoint: new THREE.Vector2(0.04, 0.0),
      skew: 0.0,
    };
    function glslReference(u: number, v: number): { x: number; y: number } {
      const xnX = 2.0 * (u - 0.5);
      const xnY = 2.0 * (v - 0.5);
      const r2 = xnX * xnX + xnY * xnY;
      const xdX = (1.0 + params.distortion.x * r2) * xnX;
      const xdY = (1.0 + params.distortion.y * r2) * xnY;
      const kkX =
        params.focalLength.x * xdX -
        params.skew * params.focalLength.x * xdY +
        params.principalPoint.x;
      const kkY = params.focalLength.y * xdY - params.principalPoint.y;
      return { x: kkX * 0.5 + 0.5, y: kkY * 0.5 + 0.5 };
    }
    const out = makeScratch();
    for (const [u, v] of [
      [0.2, 0.2],
      [0.5, 0.5],
      [0.8, 0.4],
    ]) {
      // Radial + focal + ppx are flip-EVEN: same-space evaluation
      // already agrees, no conjugation needed.
      applyLensDistortion(u, v, params, out);
      const ref = glslReference(u, v);
      expect(ref.x).toBeCloseTo(out.x, 6);
      expect(ref.y).toBeCloseTo(out.y, 6);
    }
  });

  it('reuses the scratch buffer across calls without leaking state', () => {
    const params = identityParams();
    const out = makeScratch();
    applyLensDistortion(0.25, 0.25, params, out);
    const first = { x: out.x, y: out.y };
    applyLensDistortion(0.75, 0.75, params, out);
    // Second call writes new values; the scratch identity is preserved
    // but the contents reflect the second input.
    expect(out.x).toBeCloseTo(0.75, 6);
    expect(out.y).toBeCloseTo(0.75, 6);
    // First-call output stored separately stays as captured (sanity check).
    expect(first.x).toBeCloseTo(0.25, 6);
    expect(first.y).toBeCloseTo(0.25, 6);
  });

  it('returns the same UVScratch instance passed in (enables chaining)', () => {
    const params = identityParams();
    const out = makeScratch();
    const result = applyLensDistortion(0.4, 0.6, params, out);
    expect(result).toBe(out);
  });
});
