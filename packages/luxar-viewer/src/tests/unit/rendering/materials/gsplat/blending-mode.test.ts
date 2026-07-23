/**
 * GSplatMaterial / GSplatTSLMaterial `applyBlendingMode` parity tests.
 *
 * Sibling of `materials/point/blending-mode.test.ts` (three-geometry
 * test symmetry). Ensures runtime UI transitions produce the same
 * complete blend state as material creation, on BOTH backends.
 *
 * GSplat-specific contracts pinned here (PR #561 semantics):
 *
 *   - `normal` uses the gsplat premultiplied alpha-over state from
 *     `getGSplatNormalBlendingState()` — CustomBlending + AddEquation
 *     + One / OneMinusSrcAlpha, depthWrite false UNCONDITIONALLY (no
 *     generic opacity >= 0.99 depth-write flip) — not the generic
 *     `normal` entry.
 *   - `uProjectionMode` selects PEAK (1, 2D-projected surface density)
 *     for the surface modes `max`/`normal`/`opaque`
 *     (`usesPeakProjection`), SUM ray-integral (0) for the emissive
 *     modes `additive`/`luminous`.
 *   - The GLSL wrapper owns the `LUXAR_NORMAL_PREMULT` define
 *     lifecycle (set on `normal`, deleted on every other mode); the
 *     TSL wrapper JS-branches the graph on the mode instead and never
 *     touches that define.
 */
import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { GSplatMaterial } from '../../../../../rendering/materials/gsplat/material-glsl';
import { GSplatTSLMaterial } from '../../../../../rendering/materials/gsplat/material-tsl';

describe('GSplatMaterial.applyBlendingMode (GLSL)', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor and peak projection', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('max');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.uniforms.uProjectionMode.value).toBe(1); // peak
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('max');
  });

  it('normal mode uses the gsplat premultiplied state (One/OneMinusSrcAlpha) + peak projection', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('normal');
    // getGSplatNormalBlendingState — NOT the generic normal entry.
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    // Symmetric alpha channel — separate alpha-equation state trips
    // gl.getError() under the WebGPU→WebGL2 bridge.
    expect(mat.blendEquationAlpha).toBeNull();
    expect(mat.blendSrcAlpha).toBeNull();
    expect(mat.blendDstAlpha).toBeNull();
    // Premultiplied coverage-alpha fragment output is define-gated.
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBe('');
    // Peak projection for alpha-over surfaces (matches classical 3DGS).
    expect(mat.uniforms.uProjectionMode.value).toBe(1);
    expect(mat.userData.blendingMode).toBe('normal');
  });

  it('normal depthWrite stays OFF at opacity 1.0 (opacity-inert, unlike points/lines)', () => {
    const mat = new GSplatMaterial({ opacity: 1.0 });
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);

    mat.uniforms.uOpacity.value = 0.5;
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);
  });

  it('additive mode resets CustomBlending state and returns to sum projection', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // Shared getCompleteBlendingState: SrcAlpha/One (resets stranded OneFactor)
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.uniforms.uProjectionMode.value).toBe(0); // sum ray-integral
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('leaving normal clears LUXAR_NORMAL_PREMULT and restores sum projection', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('normal');
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBe('');
    mat.applyBlendingMode('additive');
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBeUndefined();
    expect(mat.uniforms.uProjectionMode.value).toBe(0);
  });

  it('normal → max stays peak-projected but swaps to the max blend state', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('normal');
    mat.applyBlendingMode('max');
    expect(mat.uniforms.uProjectionMode.value).toBe(1); // still peak
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBeUndefined();
  });

  it('opaque mode disables transparency, writes depth, PEAK projection (surface mode)', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
    // Opaque alpha-overs a surface — 2D-projected peak, like max/normal.
    expect(mat.uniforms.uProjectionMode.value).toBe(1);
    expect(mat.userData.blendingMode).toBe('opaque');
  });

  it('luminous mode is additive but depth-tested, sum projection', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.uniforms.uProjectionMode.value).toBe(0);
    expect(mat.userData.blendingMode).toBe('luminous');
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    // THREE.Material.needsUpdate is a setter that increments `version`.
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('normal');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('normal');
    expect(mat.version).toBe(versionAfterFirst);
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBe('');
  });

  it('mode changes that toggle defines bump material.version (normal→additive)', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('normal');
    const versionAfterNormal = mat.version;
    mat.applyBlendingMode('additive');
    // Define toggled (LUXAR_NORMAL_PREMULT removed) → recompile required
    expect(mat.version).toBeGreaterThan(versionAfterNormal);
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBeUndefined();
  });

  it('gsplat fragment shader contains LUXAR_NORMAL_PREMULT guard', () => {
    const mat = new GSplatMaterial();
    expect(mat.fragmentShader).toContain('#ifdef LUXAR_NORMAL_PREMULT');
  });

  it('volumetric mode: One/OneMinusSrcAlpha state, SUM projection, LUXAR_VOLUMETRIC define', () => {
    const mat = new GSplatMaterial();
    mat.applyBlendingMode('volumetric');
    // Same framebuffer state as gsplat-normal — the semantics live in
    // the fragment shader (emission–absorption vs coverage alpha).
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    // …but SUM ray-integral projection (emissive taxonomy), NOT peak.
    expect(mat.uniforms.uProjectionMode.value).toBe(0);
    expect(mat.defines.LUXAR_VOLUMETRIC).toBe('');
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('volumetric');
  });

  it('volumetric define lifecycle: every transition owns BOTH mode defines', () => {
    const mat = new GSplatMaterial();
    // volumetric → normal must not strand LUXAR_VOLUMETRIC…
    mat.applyBlendingMode('volumetric');
    mat.applyBlendingMode('normal');
    expect(mat.defines.LUXAR_VOLUMETRIC).toBeUndefined();
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBe('');
    // …and normal → volumetric must not strand LUXAR_NORMAL_PREMULT.
    mat.applyBlendingMode('volumetric');
    expect(mat.defines.LUXAR_NORMAL_PREMULT).toBeUndefined();
    expect(mat.defines.LUXAR_VOLUMETRIC).toBe('');
    // volumetric → additive clears it and keeps sum projection.
    mat.applyBlendingMode('additive');
    expect(mat.defines.LUXAR_VOLUMETRIC).toBeUndefined();
    expect(mat.uniforms.uProjectionMode.value).toBe(0);
    // additive → volumetric bumps version (define toggle ⇒ recompile).
    const versionAfterAdditive = mat.version;
    mat.applyBlendingMode('volumetric');
    expect(mat.version).toBeGreaterThan(versionAfterAdditive);
  });

  it('gsplat fragment shader contains the LUXAR_VOLUMETRIC emission–absorption branch', () => {
    const mat = new GSplatMaterial();
    expect(mat.fragmentShader).toContain('defined(LUXAR_VOLUMETRIC)');
    expect(mat.fragmentShader).toContain('uAbsorption');
    expect(mat.fragmentShader).toContain('exp(-tau)');
  });

  it('clone() round-trips absorption (the layers panel clones on any first interaction)', () => {
    const mat = new GSplatMaterial({ absorption: 3.5, blendingMode: 'volumetric' });
    expect(mat.getAbsorption()).toBe(3.5);
    const cloned = mat.clone();
    expect(cloned.getAbsorption()).toBe(3.5);
    expect(cloned.userData.blendingMode).toBe('volumetric');
    expect(cloned.defines.LUXAR_VOLUMETRIC).toBe('');
  });

  it('clone() carries the camera-STATE uniforms (uIsOrtho, uNearCull, uResolution)', () => {
    // Same class as the points/lines clone fix: a clone taken in ortho
    // mode used to keep the constructor defaults (perspective branch,
    // stale near-cull) until the next global camera broadcast.
    const mat = new GSplatMaterial();
    mat.updateCameraParams(2.0, new THREE.Vector2(640, 480), /*isOrtho=*/ true, 0.42);
    const cloned = mat.clone();
    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
    expect(cloned.uniforms.uNearCull.value).toBeCloseTo(0.42, 5);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).x).toBe(640);
  });

  it('updateAbsorption writes the uniform without a recompile', () => {
    const mat = new GSplatMaterial({ blendingMode: 'volumetric' });
    const version = mat.version;
    mat.updateAbsorption(0.25);
    expect(mat.uniforms.uAbsorption.value).toBe(0.25);
    expect(mat.version).toBe(version);
  });
});

describe('GSplatTSLMaterial.applyBlendingMode (TSL)', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor and peak projection', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('max');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.uniforms.uProjectionMode.value).toBe(1); // peak
    expect(mat.userData.blendingMode).toBe('max');
  });

  it('normal mode uses the gsplat premultiplied state (One/OneMinusSrcAlpha) + peak projection', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('normal');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    // TSL JS-branches the graph on the mode — no define, and NEVER the
    // premultipliedAlpha flag (NodeMaterial.setup() would auto-inject a
    // second RGB×alpha transform).
    expect('LUXAR_NORMAL_PREMULT' in (mat.defines ?? {})).toBe(false);
    expect(mat.premultipliedAlpha).toBe(false);
    expect(mat.uniforms.uProjectionMode.value).toBe(1); // peak
    expect(mat.userData.blendingMode).toBe('normal');
  });

  it('normal depthWrite stays OFF at opacity 1.0 (opacity-inert, unlike points/lines)', () => {
    const mat = new GSplatTSLMaterial({ opacity: 1.0 });
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);

    mat.uniforms.uOpacity.value = 0.5;
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);
  });

  it('additive mode resets CustomBlending state and returns to sum projection', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.uniforms.uProjectionMode.value).toBe(0); // sum ray-integral
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('normal → max stays peak-projected but swaps to the max blend state', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('normal');
    mat.applyBlendingMode('max');
    expect(mat.uniforms.uProjectionMode.value).toBe(1); // still peak
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendDst).toBe(THREE.OneFactor);
  });

  it('opaque mode disables transparency, writes depth, PEAK projection (surface mode)', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
    // Opaque alpha-overs a surface — 2D-projected peak, like max/normal.
    expect(mat.uniforms.uProjectionMode.value).toBe(1);
    expect(mat.userData.blendingMode).toBe('opaque');
  });

  it('luminous mode is additive but depth-tested, sum projection', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.uniforms.uProjectionMode.value).toBe(0);
    expect(mat.userData.blendingMode).toBe('luminous');
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('max');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('max');
    expect(mat.version).toBe(versionAfterFirst);
  });

  it('projection-crossing mode changes bump material.version (max→additive graph rebuild)', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('max');
    const versionAfterMax = mat.version;
    mat.applyBlendingMode('additive');
    // sum↔peak crossing → rebuildGraph → needsUpdate → version bump
    expect(mat.version).toBeGreaterThan(versionAfterMax);
    expect(mat.uniforms.uProjectionMode.value).toBe(0);
  });

  it('additive→opaque crosses the sum↔peak boundary and rebuilds the graph', () => {
    // LOAD-BEARING: the rebuild boundary compares usesPeakProjection —
    // an isMaxMode-only comparison would keep the SUM graph alive on an
    // additive→opaque switch (stale ray-integral projection). Mirrors
    // the max→additive crossing test above.
    const mat = new GSplatTSLMaterial({ blendingMode: 'additive' });
    const versionAfterAdditive = mat.version;
    mat.applyBlendingMode('opaque');
    expect(mat.version).toBeGreaterThan(versionAfterAdditive);
    expect(mat.uniforms.uProjectionMode.value).toBe(1); // peak
    expect(mat.userData.blendingMode).toBe('opaque');
  });

  it('additive→volumetric crosses the OUTPUT-BRANCH boundary and rebuilds the graph', () => {
    // LOAD-BEARING (fail-first verified against the pre-fix predicate):
    // additive and volumetric are BOTH sum-projected and BOTH non-normal,
    // so neither usesPeakProjection nor isNormalMode crosses — the old
    // `projectionChanged || premultChanged` predicate would keep the
    // stale alpha=1 additive graph while the framebuffer switched to
    // One/OneMinusSrcAlpha. The isVolumetricMode term in
    // outputBranchChanged is what forces the rebuild.
    //
    // NOTE: must spy on rebuildGraph itself — `material.version` is NOT
    // a valid proxy (the `needsUpdate` fallback for state-only changes
    // bumps version too, masking a missing rebuild).
    const mat = new GSplatTSLMaterial({ blendingMode: 'additive' });
    const rebuildSpy = vi.spyOn(mat as unknown as { rebuildGraph(): void }, 'rebuildGraph');

    mat.applyBlendingMode('volumetric');
    expect(rebuildSpy).toHaveBeenCalledTimes(1);
    expect(mat.uniforms.uProjectionMode.value).toBe(0); // still sum
    expect(mat.userData.blendingMode).toBe('volumetric');

    // And back: volumetric→additive rebuilds too.
    mat.applyBlendingMode('additive');
    expect(rebuildSpy).toHaveBeenCalledTimes(2);
    rebuildSpy.mockRestore();
  });

  it('normal→volumetric rebuilds the TSL graph (both output flags + projection cross)', () => {
    // The remaining compound transition class: BOTH isNormalMode and
    // isVolumetricMode flip AND usesPeakProjection crosses — any OR term
    // suffices, but pin the class directly.
    const mat = new GSplatTSLMaterial({ blendingMode: 'normal' });
    const rebuildSpy = vi.spyOn(mat as unknown as { rebuildGraph(): void }, 'rebuildGraph');
    mat.applyBlendingMode('volumetric');
    expect(rebuildSpy).toHaveBeenCalledTimes(1);
    expect(mat.uniforms.uProjectionMode.value).toBe(0); // peak -> sum
    rebuildSpy.mockRestore();
  });

  it('volumetric mode: One/OneMinusSrcAlpha state, SUM projection (TSL)', () => {
    const mat = new GSplatTSLMaterial();
    mat.applyBlendingMode('volumetric');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.uniforms.uProjectionMode.value).toBe(0); // sum
    expect(mat.userData.blendingMode).toBe('volumetric');
  });

  it('clone() round-trips absorption (TSL)', () => {
    const mat = new GSplatTSLMaterial({ absorption: 3.5, blendingMode: 'volumetric' });
    expect(mat.getAbsorption()).toBe(3.5);
    const cloned = mat.clone();
    expect(cloned.getAbsorption()).toBe(3.5);
    expect(cloned.userData.blendingMode).toBe('volumetric');
  });

  it('clone() carries the camera-STATE uniforms (uIsOrtho, uNearCull, uResolution) (TSL)', () => {
    // Same class as the points/lines clone fix; gsplat uIsOrtho is a
    // runtime uniform in the TSL graph, so a value copy suffices.
    const mat = new GSplatTSLMaterial();
    mat.updateCameraParams(2.0, new THREE.Vector2(640, 480), /*isOrtho=*/ true, 0.42);
    const cloned = mat.clone();
    expect(cloned.uniforms.uIsOrtho.value).toBe(1);
    expect(cloned.uniforms.uNearCull.value).toBeCloseTo(0.42, 5);
    expect((cloned.uniforms.uResolution.value as THREE.Vector2).x).toBe(640);
  });

  it('updateAbsorption writes the uniform without a graph rebuild (TSL)', () => {
    const mat = new GSplatTSLMaterial({ blendingMode: 'volumetric' });
    const version = mat.version;
    mat.updateAbsorption(0.25);
    expect(mat.uniforms.uAbsorption.value).toBe(0.25);
    expect(mat.version).toBe(version);
  });
});

// H — TSL constructor honors explicit transparent/depthTest overrides
// AFTER the factory tail's mode-derived state, exactly like the GLSL
// twin (additive state otherwise forces depthTest=false).
describe('GSplatTSLMaterial constructor explicit overrides', () => {
  it('depthTest: true survives additive construction', () => {
    const mat = new GSplatTSLMaterial({ blendingMode: 'additive', depthTest: true });
    expect(mat.depthTest).toBe(true);
    expect(mat.userData.depthTest).toBe(true);
  });

  it('transparent: false survives additive construction', () => {
    const mat = new GSplatTSLMaterial({ blendingMode: 'additive', transparent: false });
    expect(mat.transparent).toBe(false);
  });
});
