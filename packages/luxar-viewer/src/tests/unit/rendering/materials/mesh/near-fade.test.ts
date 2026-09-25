/**
 * The mesh near-plane fade (#1431), on BOTH backends and in BOTH shader pairs.
 *
 * Mesh was the one geometry type that clipped hard against the near plane while the
 * other three faded out through the shared `perspectiveNearFade`. Closing that has
 * two halves, and this file pins both:
 *
 *   1. **The wrappers.** All four (visual GLSL/TSL, pick GLSL/TSL) became
 *      `CameraAwareMaterial`s so the manager's broadcast reaches them, and `clone()`
 *      has to carry the live values — a clone that reverted to the perspective/0.1
 *      constructor defaults would fade against the wrong near plane, and under ortho
 *      (where the fade is the identity) would fade at all. Table-driven over the two
 *      backends, because the two classes reimplement the same contract
 *      independently. The pick pair's own construction/clone coverage lives in
 *      `picking/mesh/material.test.ts`; what is asserted here is the guard, so the
 *      routing into `registeredMaterials` cannot regress for either pass.
 *   2. **The shader sources.** String-grep regression locks, the established pattern
 *      for shader content in this repo (`materials/point/material-glsl.test.ts`,
 *      `materials/gsplat/material-glsl.test.ts`, `shader-hot-path.test.ts`). Pixel
 *      behaviour is owned by the parity harness's `mesh-near-fade` /
 *      `mesh-pick-near-fade` entries — which render under a PERSPECTIVE camera,
 *      since the harness default is orthographic and would make the fade inert.
 *
 * One behaviour is deliberately NOT covered anywhere, and saying so is more useful
 * than implying otherwise: the occlusion case the unconditional `< 0.01` reject
 * exists for. Proving it needs two overlapping nodes — a depth-writing mesh faded to
 * nothing in front of a second node that must still be visible through it — which is
 * a scene-level E2E fixture, not a shader-string or single-material assertion. What
 * is pinned here is that the reject is present and unconditional; that a depth-write
 * without it would occlude is an argument, not a measurement.
 *
 * The TSL wrappers build a real node graph in their constructors, which works
 * headless (`picking/mesh/material.test.ts` already relies on it), so THREE is NOT
 * mocked here.
 *
 * @module tests/unit/rendering/materials/mesh/near-fade
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { MeshMaterial } from '../../../../../rendering/materials/mesh/material-glsl';
import { MeshTSLMaterial } from '../../../../../rendering/materials/mesh/material-tsl';
import { MeshPickingMaterial } from '../../../../../rendering/picking/mesh/material';
import { MeshPickingTSLMaterial } from '../../../../../rendering/picking/mesh/material-tsl';
import { isCameraAwareMaterial } from '../../../../../rendering/materials/_shared/camera-aware-material';
import {
  MESH_FRAGMENT_SHADER,
  MESH_VERTEX_SHADER,
} from '../../../../../rendering/materials/mesh/shader-glsl';
import {
  MESH_PICK_FRAGMENT_SHADER,
  MESH_PICK_VERTEX_SHADER,
} from '../../../../../rendering/picking/mesh/shaders';

/**
 * How many `#if` / `#ifdef` / `#ifndef` blocks are still open at `index`.
 *
 * `#else` / `#elif` leave the depth alone — they continue a block rather than opening
 * one — so a zero here means "unconditional", not merely "before some directive".
 */
function preprocessorDepthAt(source: string, index: number): number {
  let depth = 0;
  for (const line of source.slice(0, index).split('\n')) {
    const directive = line.trim();
    if (/^#if(def|ndef)?\b/.test(directive)) depth++;
    else if (/^#endif\b/.test(directive)) depth--;
  }
  return depth;
}

/** Every mesh material, visual and pick, behind one no-arg constructor. */
const ALL_WRAPPERS: ReadonlyArray<[string, () => THREE.Material]> = [
  ['visual glsl', () => new MeshMaterial()],
  ['visual tsl', () => new MeshTSLMaterial()],
  ['pick glsl', () => new MeshPickingMaterial({ nodeId: 3 })],
  ['pick tsl', () => new MeshPickingTSLMaterial({ nodeId: 3 })],
];

/** Just the visual pair — the one with a `clone()` carrying appearance config. */
const VISUAL_BACKENDS: ReadonlyArray<[string, () => MeshMaterial | MeshTSLMaterial]> = [
  ['glsl', () => new MeshMaterial()],
  ['tsl', () => new MeshTSLMaterial()],
];

describe.each(ALL_WRAPPERS)('mesh material [%s] — the camera contract', (_label, make) => {
  it('passes the isCameraAwareMaterial guard', () => {
    // The guard is what `MaterialManager.register` dispatches on, so this is the
    // assertion that keeps mesh inside the camera broadcast. Before #1431 all four
    // of these deliberately failed it.
    expect(isCameraAwareMaterial(make())).toBe(true);
  });

  it('starts perspective, with the 0.1 near-cull default the siblings use', () => {
    const m = make() as unknown as { uniforms: Record<string, THREE.IUniform> };
    expect(m.uniforms.uIsOrtho.value).toBe(0);
    expect(m.uniforms.uNearCull.value).toBe(0.1);
  });

  it('writes BOTH near-fade uniforms, and ignores fov / resolution', () => {
    const m = make() as unknown as {
      uniforms: Record<string, THREE.IUniform>;
      updateCameraParams: (
        fov: number,
        res: THREE.Vector2,
        isOrtho?: boolean,
        nearCull?: number
      ) => void;
    };
    m.updateCameraParams(1.25, new THREE.Vector2(1234, 777), true, 0.42);
    expect(m.uniforms.uIsOrtho.value).toBe(1);
    expect(m.uniforms.uNearCull.value).toBe(0.42);
    // A mesh has no screen-space size to recompute, so the first two arguments are
    // accepted and dropped — nothing may appear for them.
    expect(m.uniforms.uResolution).toBeUndefined();
    expect(m.uniforms.uFx).toBeUndefined();
  });

  it('leaves the last nearCull standing when the argument is omitted', () => {
    // Matches every sibling material: `nearCull` is optional, and a caller that
    // omits it (an ortho toggle mid-session) must not silently reset the scene's
    // value back to the 0.1 default and fade against the wrong plane.
    const m = make() as unknown as {
      uniforms: Record<string, THREE.IUniform>;
      updateCameraParams: (
        fov: number,
        res: THREE.Vector2,
        isOrtho?: boolean,
        nearCull?: number
      ) => void;
    };
    m.updateCameraParams(1.0, new THREE.Vector2(800, 600), false, 7.5);
    m.updateCameraParams(1.0, new THREE.Vector2(800, 600), true);
    expect(m.uniforms.uNearCull.value).toBe(7.5);
  });
});

describe.each(VISUAL_BACKENDS)('MeshMaterial [%s] — clone carries the camera state', (_l, make) => {
  it('copies uIsOrtho and uNearCull across', () => {
    // The layers panel clones on first interaction. A clone built from the config
    // alone would come back at perspective/0.1: on an ortho scene it would start
    // fading geometry the fade is supposed to leave alone, and on a perspective one
    // it would fade against a near plane the scene never had.
    const m = make();
    m.updateCameraParams(1.0, new THREE.Vector2(800, 600), true, 0.42);
    const c = m.clone();
    expect(c).not.toBe(m);
    expect(c.uniforms.uIsOrtho.value).toBe(1);
    expect(c.uniforms.uNearCull.value).toBeCloseTo(0.42);
  });
});

describe('the mesh GLSL sources carry the fade', () => {
  it('evaluates it PER FRAGMENT, off vViewPos, with the shared 1e-20 floor', () => {
    // Per fragment and not per vertex: a triangle spans depth, so a per-vertex value
    // would interpolate the RAMP across the face. The varying it reads is the one
    // the shade term already carries, so no new vertex output was needed.
    // The ortho test is three's per-draw `isOrthographic` (the camera being
    // drawn with), not the CPU-pushed uIsOrtho.
    expect(MESH_FRAGMENT_SHADER).toContain(
      'perspectiveNearFade(isOrthographic ? 1 : 0, vViewPos.z, max(uNearCull, 1e-20))'
    );
    expect(MESH_FRAGMENT_SHADER).toContain('float perspectiveNearFade(');
    expect(MESH_FRAGMENT_SHADER).not.toMatch(/perspectiveNearFade\(uIsOrtho/);
    expect(MESH_FRAGMENT_SHADER).toContain('uniform float uNearCull;');
    // The vertex stage is untouched — no fade varying was added there.
    expect(MESH_VERTEX_SHADER).not.toContain('perspectiveNearFade');
  });

  it('rejects below 0.01 in EVERY mode, unconditionally', () => {
    // Outside every `#if*`: a mesh may WRITE depth — `opaque` always, `normal` at
    // opacity >= 0.99 (`normalModeDepthWrite`, which mesh feeds its real opacity) —
    // so a faded-out fragment left rasterizing would occlude everything behind it. A
    // reject guarded by the translucent modes would leave exactly the depth-writing
    // ones broken.
    //
    // Asserted as a real PREPROCESSOR NESTING DEPTH rather than "before the first
    // mode ifdef". The fragment body already opens `#ifdef LUXAR_MESH_FLAT_NORMAL`,
    // `#ifdef LUXAR_NO_GOG` and `#if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)`
    // upstream of this line, so an index comparison against one chosen directive
    // would happily pass with the reject buried inside any of those three.
    const REJECT = 'if (nearFade < 0.01) discard;';
    expect(MESH_FRAGMENT_SHADER).toContain(REJECT);
    const rejectIndex = MESH_FRAGMENT_SHADER.indexOf(REJECT);
    expect(preprocessorDepthAt(MESH_FRAGMENT_SHADER, rejectIndex)).toBe(0);
  });

  it('ramps the opaque mode RGB, and folds into coverage everywhere else', () => {
    // `opaque` emits alpha 1.0, so there is no alpha to fade — the shaded RGB takes
    // the ramp instead. Every other mode multiplies the coverage, which is what
    // makes `max`'s premultiply pick it up for free.
    expect(MESH_FRAGMENT_SHADER).toContain('fragColor = vec4(shadedColor * nearFade, 1.0);');
    expect(MESH_FRAGMENT_SHADER).toContain('a *= nearFade;');
    // ...and the cutout still compares the UNFADED coverage, or its holes would
    // dissolve open as the camera approached.
    expect(MESH_FRAGMENT_SHADER).toContain('if (a < uAlphaCutoff) discard;');
  });
});

describe('the mesh-pick GLSL sources carry the same fade', () => {
  it('adds a view-depth varying rather than the whole view position', () => {
    // The visual pair's `vViewPos` exists to be differentiated for the flat-normal
    // fallback; the pick pass has no shading, so it carries just the z.
    expect(MESH_PICK_VERTEX_SHADER).toContain('out highp float vViewZ;');
    expect(MESH_PICK_VERTEX_SHADER).toContain('vViewZ = mvPosition.z;');
    expect(MESH_PICK_VERTEX_SHADER).not.toContain('vViewPos');
  });

  it('rejects at the same 0.01 and folds the fade into brightness', () => {
    // Pick coverage must track visible coverage as the camera closes in, or a
    // surface the user can barely see stays fully pickable AND keeps depth-occluding
    // whatever is behind it.
    expect(MESH_PICK_FRAGMENT_SHADER).toContain(
      'perspectiveNearFade(isOrthographic ? 1 : 0, vViewZ, max(uNearCull, 1e-20))'
    );
    expect(MESH_PICK_FRAGMENT_SHADER).toContain('if (nearFade < 0.01) discard;');
    expect(MESH_PICK_FRAGMENT_SHADER).toContain(
      'mediump float brightness = clamp(a * nearFade, 0.0, 1.0);'
    );
  });
});
