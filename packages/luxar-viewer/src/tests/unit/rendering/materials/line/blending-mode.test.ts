/**
 * LineMaterial / LineTSLMaterial `applyBlendingMode` parity tests.
 *
 * Sibling of `materials/point/blending-mode.test.ts` (three-geometry
 * test symmetry). Ensures runtime UI transitions produce the same
 * complete blend state as material creation, on BOTH backends. Both
 * wrappers draw from the SHARED `getCompleteBlendingState` +
 * `applyBlendingStateToMaterial` helpers, so for every mode the GLSL
 * and TSL materials carry identical blending / blend-factor / depth /
 * transparency state (the convergence suite below pins that).
 *
 * Both toggle the LUXAR_MAX_RGB_CONTRIBUTION shader define when
 * entering max and clear it when leaving.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { LineMaterial } from '../../../../../rendering/materials/line/material-glsl';
import { LineTSLMaterial } from '../../../../../rendering/materials/line/material-tsl';
import { LINE_CHORD_SCALE } from '../../../../../rendering/materials/line/math';
import { POINT_CHORD_SCALE } from '../../../../../rendering/materials/point/math';
import { getCompleteBlendingState } from '../../../../../rendering/blending-state';
import type { BlendingMode } from '../../../../../rendering/material-manager';

describe('LineMaterial.applyBlendingMode (GLSL)', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    expect(mat.blending).toBe(THREE.CustomBlending);
    expect(mat.blendEquation).toBe(THREE.MaxEquation);
    expect(mat.blendSrc).toBe(THREE.OneFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
    expect(mat.userData.blendingMode).toBe('max');
  });

  it('additive mode resets CustomBlending state via the shared helper (SrcAlpha/One)', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // Shared getCompleteBlendingState: SrcAlpha/One (resets stranded
    // OneFactor from the previous max state; inert under the
    // AdditiveBlending preset, and now byte-identical to the TSL twin).
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('normal at opacity=1.0 writes depth; at 0.5 does not', () => {
    const mat = new LineMaterial({ opacity: 1.0 });
    mat.applyBlendingMode('normal');
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(true);

    mat.uniforms.uOpacity.value = 0.5;
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);
  });

  it('opaque mode disables transparency and writes depth', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.userData.blendingMode).toBe('opaque');
  });

  it('luminous mode is additive but depth-tested', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('luminous');
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    // THREE.Material.needsUpdate is a setter that increments `version`.
    // Read via `version` to detect whether a recompile would be triggered.
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('max');
    expect(mat.version).toBe(versionAfterFirst);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
  });

  it('mode changes that toggle defines bump material.version (max→additive)', () => {
    const mat = new LineMaterial();
    mat.applyBlendingMode('max');
    const versionAfterMax = mat.version;
    mat.applyBlendingMode('additive');
    // Define toggled (LUXAR_MAX_RGB_CONTRIBUTION removed) → recompile required
    expect(mat.version).toBeGreaterThan(versionAfterMax);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
  });

  it('line fragment shader contains LUXAR_MAX_RGB_CONTRIBUTION guard', () => {
    // Phase 4 made the max branch an `#elif` of the volumetric `#if` —
    // pin the `defined(...)` form (matches the point twin's pin).
    const mat = new LineMaterial();
    expect(mat.fragmentShader).toContain('defined(LUXAR_MAX_RGB_CONTRIBUTION)');
    expect(mat.fragmentShader).toContain('gammaColor * a');
  });

  it('line fragment shader contains the volumetric emission–absorption branch', () => {
    // Shader-text pins for the LUXAR_VOLUMETRIC output branch — a
    // pure-TS state test can't guard the emitted GLSL (the blending
    // campaign's mutation lesson). τ = κ·density·chord (the transverse
    // ribbon through-thickness), physical absorption alpha, and the
    // color-discard bypass (a black line still absorbs) are each
    // distinct generated code. Mirrors the point twin.
    const mat = new LineMaterial();
    expect(mat.fragmentShader).toContain('#if defined(LUXAR_VOLUMETRIC)');
    expect(mat.fragmentShader).toContain('float tau = uAbsorption * alpha * vWidthAtT *');
    expect(mat.fragmentShader).toContain('float volAlpha = 1.0 - exp(-tau);');
    expect(mat.fragmentShader).toContain('tau < 1e-4) discard');
    expect(mat.fragmentShader).toContain('fragColor = vec4(gammaColor * alpha * screen, volAlpha)');
    // Per-endpoint alpha → optical depth map, gated by uHasElementAlpha.
    // The gate must be pinned at its USAGE inside the mix() — a bare
    // `toContain('uHasElementAlpha')` also matches the uniform
    // DECLARATION and survives a mutation that hardwires the gate to
    // 1.0 (the point twin's mutation-found w ≈ 6.24 identity-alpha
    // blowup for RGB data would ship silently).
    expect(mat.fragmentShader).toContain('-log(1.0 - min(vAlpha,');
    expect(mat.fragmentShader).toContain('), uHasElementAlpha);');
  });

  it('vertex shader sanitizes the per-endpoint alpha read (NaN/Inf → 1.0, finite clamped to [0, 1])', () => {
    // Alpha is load-bearing in every mode and feeds optical depth under
    // volumetric — an unsanitized NaN from hand-crafted zarr poisons τ
    // past the discard into NaN pixels. Pinned at the USAGE (the
    // interpolated assignment), matching the point twin's pin.
    const mat = new LineMaterial();
    expect(mat.vertexShader).toContain(
      'vAlpha = mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), t);'
    );
  });

  it('LINE_CHORD_SCALE equals POINT_CHORD_SCALE (the κ-scale alignment contract, executable)', () => {
    // Both math.ts files derive √(π/ln 100) independently (isotropic
    // ball chord ∝ radius vs transverse ribbon chord ∝ width) and their
    // comments declare the values identical — which is what makes the
    // shared κ slider mean the same optical depth per unit size across
    // geometry types. Retuning one constant without the other would
    // silently de-calibrate κ; this pin makes the contract executable.
    expect(LINE_CHORD_SCALE).toBe(POINT_CHORD_SCALE);
    expect(LINE_CHORD_SCALE).toBeCloseTo(Math.sqrt(Math.PI / Math.log(100.0)), 15);
  });

  it('non-volumetric fragment folds vAlpha into the contribution (alpha active in EVERY mode)', () => {
    // Phase-2 doctrine: the per-element alpha is a plain linear
    // contribution scale outside volumetric. Mutation-found gap: dropping
    // `intensity *= vAlpha;` survived the entire unit suite (only the
    // playwright-tier codegen snapshot would catch it) — an RGBA
    // dataset's translucent segments would render fully opaque in
    // additive/normal/max. Pinned at the USAGE.
    const mat = new LineMaterial();
    expect(mat.fragmentShader).toContain('intensity *= vAlpha;');
  });
});

describe('LineTSLMaterial.applyBlendingMode (TSL)', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor', () => {
    const mat = new LineTSLMaterial();
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

  it('additive mode resets CustomBlending state via the shared helper (SrcAlpha/One)', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // TSL wrapper applies getCompleteBlendingState: SrcAlpha/One (the
    // GLSL wrapper's hand-rolled reset differs in blendDst; both are
    // inert under the AdditiveBlending preset).
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('normal at opacity=1.0 writes depth; at 0.5 does not', () => {
    const mat = new LineTSLMaterial({ opacity: 1.0 });
    mat.applyBlendingMode('normal');
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(true);

    mat.uniforms.uOpacity.value = 0.5;
    mat.applyBlendingMode('normal');
    expect(mat.depthWrite).toBe(false);
  });

  it('opaque mode disables transparency and writes depth', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
    expect(mat.userData.blendingMode).toBe('opaque');
  });

  it('luminous mode is additive but depth-tested', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
    expect(mat.transparent).toBe(true);
    expect(mat.userData.blendingMode).toBe('luminous');
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('max');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('max');
    expect(mat.version).toBe(versionAfterFirst);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
  });

  it('mode changes that toggle defines bump material.version (max→additive)', () => {
    const mat = new LineTSLMaterial();
    mat.applyBlendingMode('max');
    const versionAfterMax = mat.version;
    mat.applyBlendingMode('additive');
    // Define toggled → rebuildGraph → needsUpdate → version bump
    expect(mat.version).toBeGreaterThan(versionAfterMax);
    expect(mat.defines?.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
  });
});

// Both wrappers delegate to getCompleteBlendingState +
// applyBlendingStateToMaterial, so for EVERY mode the applied THREE
// state must equal the helper's canonical values and the two backends
// must agree field-for-field. This is the convergence contract that
// replaced the GLSL wrapper's hand-rolled per-mode dispatch.
describe('LineMaterial ↔ LineTSLMaterial blending-state convergence', () => {
  // 'volumetric' joins the loop since phase 4: line materials apply the
  // shared getCompleteBlendingState('volumetric') like points/gsplats.
  const ALL_MODES: BlendingMode[] = [
    'additive',
    'volumetric',
    'normal',
    'max',
    'opaque',
    'luminous',
  ];
  const STATE_FIELDS = [
    'blending',
    'blendEquation',
    'blendSrc',
    'blendDst',
    'depthTest',
    'depthWrite',
    'transparent',
  ] as const;

  for (const mode of ALL_MODES) {
    it(`'${mode}': GLSL state equals getCompleteBlendingState and matches TSL`, () => {
      const glsl = new LineMaterial();
      const tsl = new LineTSLMaterial();
      glsl.applyBlendingMode(mode);
      tsl.applyBlendingMode(mode);
      const expected = getCompleteBlendingState(mode, 1.0);
      for (const field of STATE_FIELDS) {
        expect(glsl[field], `GLSL ${field} for '${mode}'`).toBe(expected[field]);
        expect(tsl[field], `TSL ${field} for '${mode}'`).toBe(expected[field]);
      }
      expect(glsl.userData.blendingMode).toBe(mode);
      expect(tsl.userData.blendingMode).toBe(mode);
    });
  }

  it("'volumetric' applies the REAL emission–absorption state (phase 4), userData keeps 'volumetric'", () => {
    // Lines implement the volumetric fragment math since phase 4
    // (VOLUMETRIC_BLENDING_SPEC.md): premultiplied self-screened
    // emission over One/OneMinusSrcAlpha — the same framebuffer state
    // points (phase 3) and gsplats (phase 1) carry — never
    // depth-writes, depth-tested.
    for (const mat of [new LineMaterial(), new LineTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      expect(mat.blending).toBe(THREE.CustomBlending);
      expect(mat.blendEquation).toBe(THREE.AddEquation);
      expect(mat.blendSrc).toBe(THREE.OneFactor);
      expect(mat.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
      expect(mat.depthTest).toBe(true);
      expect(mat.depthWrite).toBe(false);
      expect(mat.transparent).toBe(true);
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBe('');
      expect(mat.userData.blendingMode).toBe('volumetric');
    }
  });

  it('every non-volumetric transition clears LUXAR_VOLUMETRIC (no stranded define)', () => {
    // Risk #6 of the spec: a volumetric→normal switch must not strand
    // the define — the normal branch would then never be reached.
    for (const mat of [new LineMaterial(), new LineTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBe('');
      mat.applyBlendingMode('normal');
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBeUndefined();
      mat.applyBlendingMode('volumetric');
      mat.applyBlendingMode('additive');
      expect(mat.defines?.LUXAR_VOLUMETRIC).toBeUndefined();
    }
  });

  it('volumetric state survives TSL CONSTRUCTION and graph REBUILDS (factory-tail interception)', () => {
    // Mirror of the point twin: the TSL factory tail is the ONLY state
    // writer at construction (the ctor never calls applyBlendingMode,
    // unlike GLSL) and re-runs on every rebuildGraph — so it must
    // derive the volumetric state (and output branch) itself from
    // config.blendingMode.
    const expected = getCompleteBlendingState('volumetric', 1.0);
    const constructed = new LineTSLMaterial({ blendingMode: 'volumetric' });
    for (const field of STATE_FIELDS) {
      expect(constructed[field], `constructed ${field}`).toBe(expected[field]);
    }
    expect(constructed.defines?.LUXAR_VOLUMETRIC).toBe('');
    expect(constructed.userData.blendingMode).toBe('volumetric');

    // max→volumetric toggles BOTH defines → definesChanged →
    // rebuildGraph — the tail must re-derive the volumetric state, not
    // clobber it with a stale-mode default.
    const switched = new LineTSLMaterial({ blendingMode: 'max' });
    switched.applyBlendingMode('volumetric');
    for (const field of STATE_FIELDS) {
      expect(switched[field], `post-switch ${field}`).toBe(expected[field]);
    }
    expect(switched.defines?.LUXAR_VOLUMETRIC).toBe('');

    // Any later rebuild while volumetric (e.g. gamma crossing 1.0)
    // must not clobber the state either.
    switched.updateGamma(2.2); // gamma crossing 1.0 → rebuildGraph
    for (const field of STATE_FIELDS) {
      expect(switched[field], `post-rebuild ${field}`).toBe(expected[field]);
    }
  });

  it('clone() carries uAbsorption and uHasElementAlpha (both backends)', () => {
    // The layers panel clones on first interaction; a clone that reset
    // κ to 1.0 or dropped the RGBA-alpha flag would silently change the
    // volumetric render (the gsplat phase-1 review caught the same bug
    // class in its clones).
    for (const mat of [new LineMaterial(), new LineTSLMaterial()]) {
      mat.applyBlendingMode('volumetric');
      mat.updateAbsorption(2.5);
      mat.updateHasElementAlpha(true);
      const cloned = mat.clone();
      expect(cloned.uniforms.uAbsorption.value).toBe(2.5);
      expect(cloned.uniforms.uHasElementAlpha.value).toBe(1);
      expect(cloned.userData.blendingMode).toBe('volumetric');
      expect(cloned.blending).toBe(THREE.CustomBlending);
    }
  });

  it('updateAbsorption writes the uniform without a recompile, getAbsorption reads it back (both backends)', () => {
    // Mirror of the gsplat twin's pin: κ is a live slider — a
    // recompile (GLSL version bump) or graph rebuild (TSL) per tick
    // would hitch the volumetric render on every drag step.
    for (const mat of [
      new LineMaterial({ blendingMode: 'volumetric' }),
      new LineTSLMaterial({ blendingMode: 'volumetric' }),
    ]) {
      const version = mat.version;
      mat.updateAbsorption(2.5);
      expect(mat.uniforms.uAbsorption.value).toBe(2.5);
      expect(mat.getAbsorption()).toBe(2.5);
      expect(mat.version).toBe(version);
    }
  });

  it('hasElementAlpha config seeds uHasElementAlpha at construction (both backends)', () => {
    // The commit sync pushes the geometry stamp on every commit, but a
    // material constructed FROM config (clone, cache warm-up) must seed
    // the gate itself — a dropped seed would strand a clone at the 0
    // default until the next commit.
    for (const Ctor of [LineMaterial, LineTSLMaterial]) {
      expect(new Ctor({ hasElementAlpha: true }).uniforms.uHasElementAlpha.value).toBe(1);
      expect(new Ctor({}).uniforms.uHasElementAlpha.value).toBe(0);
    }
  });
});

// H — TSL constructors honor explicit transparent/depthTest overrides
// AFTER the factory tail's mode-derived state, exactly like the GLSL
// twins (additive state otherwise forces depthTest=false).
describe('LineTSLMaterial constructor explicit overrides', () => {
  it('depthTest: true survives additive construction', () => {
    const mat = new LineTSLMaterial({ blendingMode: 'additive', depthTest: true });
    expect(mat.depthTest).toBe(true);
    expect(mat.userData.depthTest).toBe(true);
  });

  it('transparent: false survives additive construction', () => {
    const mat = new LineTSLMaterial({ blendingMode: 'additive', transparent: false });
    expect(mat.transparent).toBe(false);
  });
});

describe('LineMaterial single-pass billboards (both backends)', () => {
  it('forceSinglePass stays true with DoubleSide — deleting it silently DOUBLES fragment work', () => {
    // line quads are screen-space billboards. transparent + DoubleSide
    // without forceSinglePass trips THREE's two-pass transparent render:
    // measured live, the mesh rasterizes ~2x the triangles (and sorted
    // modes split each mesh's draw independent of the depth sort).
    for (const mat of [new LineMaterial(), new LineTSLMaterial()]) {
      expect(mat.forceSinglePass).toBe(true);
      expect(mat.side).toBe(THREE.DoubleSide);
    }
  });
});
