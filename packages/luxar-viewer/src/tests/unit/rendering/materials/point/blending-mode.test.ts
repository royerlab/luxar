/**
 * PointMaterial.applyBlendingMode parity tests.
 *
 * Ensures runtime UI transitions produce the same complete blend state
 * as material creation. Also verifies the
 * LUXAR_MAX_RGB_CONTRIBUTION shader define is set when entering max
 * and cleared when leaving.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { PointMaterial } from '../../../../../rendering/materials/point/material-glsl';
import { PointTSLMaterial } from '../../../../../rendering/materials/point/material-tsl';
import { getPointBlendingState } from '../../../../../rendering/blending-state';
import type { BlendingMode } from '../../../../../rendering/material-manager';

describe('PointMaterial.applyBlendingMode', () => {
  it('max mode sets CustomBlending + MaxEquation + OneFactor/OneFactor', () => {
    const mat = new PointMaterial();
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

  it('additive mode resets CustomBlending state (no stranded MaxEquation/OneFactor)', () => {
    const mat = new PointMaterial();
    mat.applyBlendingMode('max');
    mat.applyBlendingMode('additive');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.blendEquation).toBe(THREE.AddEquation);
    // Additive uses SrcAlpha/One (resets stranded OneFactor)
    expect(mat.blendSrc).toBe(THREE.SrcAlphaFactor);
    expect(mat.blendDst).toBe(THREE.OneFactor);
    expect(mat.depthTest).toBe(false);
    expect(mat.depthWrite).toBe(false);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
    expect(mat.userData.blendingMode).toBe('additive');
  });

  it('normal mode never writes depth, at any opacity (both backends)', () => {
    for (const Ctor of [PointMaterial, PointTSLMaterial]) {
      const mat = new Ctor({ opacity: 1.0 });
      mat.applyBlendingMode('normal');
      expect(mat.blending).toBe(THREE.NormalBlending);
      expect(mat.depthTest).toBe(true);
      // A point sprite stamps a flat depth plane across the whole disc,
      // fringe included — sorted transparency never depth-writes (#1002).
      expect(mat.depthWrite).toBe(false);
      mat.uniforms.uOpacity.value = 0.5;
      mat.applyBlendingMode('normal');
      expect(mat.depthWrite).toBe(false);
    }
  });

  it('opaque mode disables transparency and writes depth', () => {
    const mat = new PointMaterial();
    mat.applyBlendingMode('opaque');
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
    expect(mat.depthTest).toBe(true);
    expect(mat.blending).toBe(THREE.NormalBlending);
  });

  it('luminous mode is additive but depth-tested', () => {
    const mat = new PointMaterial();
    mat.applyBlendingMode('luminous');
    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(false);
  });

  it('reapplying same mode does not bump material.version (idempotent)', () => {
    // THREE.Material.needsUpdate is a setter that increments `version`.
    // Read via `version` to detect whether a recompile would be triggered.
    const mat = new PointMaterial();
    mat.applyBlendingMode('max');
    const versionAfterFirst = mat.version;
    mat.applyBlendingMode('max');
    expect(mat.version).toBe(versionAfterFirst);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBe('');
  });

  it('mode changes that toggle defines bump material.version (max→additive)', () => {
    const mat = new PointMaterial();
    mat.applyBlendingMode('max');
    const versionAfterMax = mat.version;
    mat.applyBlendingMode('additive');
    // Define toggled (LUXAR_MAX_RGB_CONTRIBUTION removed) → recompile required
    expect(mat.version).toBeGreaterThan(versionAfterMax);
    expect(mat.defines.LUXAR_MAX_RGB_CONTRIBUTION).toBeUndefined();
  });

  it('point fragment shader contains LUXAR_MAX_RGB_CONTRIBUTION guard', () => {
    const mat = new PointMaterial();
    expect(mat.fragmentShader).toContain('defined(LUXAR_MAX_RGB_CONTRIBUTION)');
    expect(mat.fragmentShader).toContain('finalColor * alpha');
  });

  it('point fragment shader contains the volumetric emission–absorption branch', () => {
    // Shader-text pins for the LUXAR_VOLUMETRIC output branch — a
    // pure-TS state test can't guard the emitted GLSL (the blending
    // campaign's mutation lesson). τ = κ·density·chord, S(τ) series
    // branch, physical absorption alpha, and the color-discard bypass
    // (a black point still absorbs) are each distinct generated code.
    const mat = new PointMaterial();
    expect(mat.fragmentShader).toContain('#if defined(LUXAR_VOLUMETRIC)');
    // τ is κ × the SAME ray mass every other mode emits. A point's opacity is
    // a peak screen alpha (already integrated), so the old
    // the former `* vRadius * <chord scale>` read it as a volume density in this one
    // mode — which is what made a Points node and its lifted-gsplat twin
    // disagree by one path length. Any size factor here re-breaks that.
    expect(mat.fragmentShader).toContain('float tau = uAbsorption * alpha;');
    expect(mat.fragmentShader).not.toMatch(/tau\s*=[^;]*vRadius/);
    expect(mat.fragmentShader).toContain('float volAlpha = 1.0 - exp(-tau);');
    expect(mat.fragmentShader).toContain('tau < 1e-4) discard');
    expect(mat.fragmentShader).toContain('fragColor = vec4(finalColor * alpha * screen, volAlpha)');
    // Per-point alpha → optical depth map, gated by uHasElementAlpha.
    // The gate must be pinned at its USAGE inside the mix() — a bare
    // `toContain('uHasElementAlpha')` also matches the uniform
    // DECLARATION and survives a mutation that hardwires the gate to
    // 1.0 (mutation-found: the w ≈ 6.24 identity-alpha blowup for RGB
    // data would ship silently).
    expect(mat.fragmentShader).toContain('-log(1.0 - min(vAlpha,');
    expect(mat.fragmentShader).toContain('), uHasElementAlpha);');
  });

  it('vertex shader sanitizes the per-point alpha read (NaN/Inf → 1.0, finite clamped to [0, 1])', () => {
    // Alpha is load-bearing in every mode and feeds optical depth under
    // volumetric — an unsanitized NaN from hand-crafted zarr poisons τ
    // past the discard into NaN pixels. Pinned at the USAGE (the
    // assignment), matching the gsplat twin's pin.
    const mat = new PointMaterial();
    expect(mat.vertexShader).toContain('vAlpha = sanitizeAlpha(pointT2.y);');
  });

  it('non-volumetric fragment folds vAlpha into the contribution (alpha active in EVERY mode)', () => {
    // Phase-2 doctrine: the per-element alpha is a plain linear
    // contribution scale outside volumetric. Mutation-found gap (on the
    // line twin): dropping the fold survived the entire unit suite —
    // only the playwright-tier codegen snapshot would catch an RGBA
    // dataset's translucent points rendering fully opaque in
    // additive/normal/max. Pinned at the USAGE.
    const mat = new PointMaterial();
    expect(mat.fragmentShader).toContain('alpha *= vAlpha;');
  });

  it("'volumetric' applies the REAL emission–absorption state (phase 3), userData keeps 'volumetric'", () => {
    // Points implement the volumetric fragment math since phase 3
    // (VOLUMETRIC_BLENDING_SPEC.md): premultiplied self-screened
    // emission over One/OneMinusSrcAlpha — the same framebuffer state
    // as gsplat volumetric — never depth-writes, depth-tested.
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
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
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
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
    // The TSL factory tail is the ONLY state writer at construction
    // (the ctor never calls applyBlendingMode, unlike GLSL) and re-runs
    // on every rebuildGraph — so it must derive the volumetric state
    // (and output branch) itself from config.blendingMode.
    const constructed = new PointTSLMaterial({ blendingMode: 'volumetric' });
    expect(constructed.blending).toBe(THREE.CustomBlending);
    expect(constructed.blendSrc).toBe(THREE.OneFactor);
    expect(constructed.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(constructed.defines?.LUXAR_VOLUMETRIC).toBe('');
    expect(constructed.userData.blendingMode).toBe('volumetric');

    // max→volumetric toggles BOTH defines → definesChanged →
    // rebuildGraph — the tail must re-derive the volumetric state, not
    // clobber it with a stale-mode default.
    const switched = new PointTSLMaterial({ blendingMode: 'max' });
    switched.applyBlendingMode('volumetric');
    expect(switched.blending).toBe(THREE.CustomBlending);
    expect(switched.blendSrc).toBe(THREE.OneFactor);
    expect(switched.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
    expect(switched.defines?.LUXAR_VOLUMETRIC).toBe('');

    // Any later rebuild while volumetric (e.g. gamma crossing 1.0)
    // must not clobber the state either.
    switched.updateGamma(2.2);
    expect(switched.blending).toBe(THREE.CustomBlending);
    expect(switched.blendDst).toBe(THREE.OneMinusSrcAlphaFactor);
  });

  it('normal depthWrite=false survives TSL construction and graph rebuilds (factory-tail)', () => {
    // applyBlendingMode('normal') toggles no defines → no rebuildGraph, so
    // the factory tail (shader-tsl.ts) is the only path that re-derives state
    // on a later rebuild (texture/gamma/colormap change). It must keep points
    // out of depthWrite in normal (#1002), not fall back to the generic gate.
    const constructed = new PointTSLMaterial({ blendingMode: 'normal', opacity: 1.0 });
    expect(constructed.userData.blendingMode).toBe('normal');
    expect(constructed.depthWrite).toBe(false);
    // A rebuild while in normal (gamma crossing 1.0 rebuilds the graph) must
    // not resurrect the opacity-gated depthWrite.
    constructed.updateGamma(2.2);
    expect(constructed.depthWrite).toBe(false);
  });

  it('clone() carries uAbsorption and uHasElementAlpha (both backends)', () => {
    // The layers panel clones on first interaction; a clone that reset
    // κ to 1.0 or dropped the RGBA-alpha flag would silently change the
    // volumetric render (the gsplat phase-1 review caught the same bug
    // class in its clones).
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
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
      new PointMaterial({ blendingMode: 'volumetric' }),
      new PointTSLMaterial({ blendingMode: 'volumetric' }),
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
    for (const Ctor of [PointMaterial, PointTSLMaterial]) {
      expect(new Ctor({ hasElementAlpha: true }).uniforms.uHasElementAlpha.value).toBe(1);
      expect(new Ctor({}).uniforms.uHasElementAlpha.value).toBe(0);
    }
  });
});

// Both wrappers delegate to getPointBlendingState +
// applyBlendingStateToMaterial (points force `normal` depthWrite off), so
// for EVERY mode the applied THREE state must equal the point helper's
// canonical values and the two backends must agree field-for-field.
// Mirrors the line twin's convergence contract (three-geometry test
// symmetry).
describe('PointMaterial ↔ PointTSLMaterial blending-state convergence', () => {
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
    it(`'${mode}': GLSL state equals getPointBlendingState and matches TSL`, () => {
      const glsl = new PointMaterial();
      const tsl = new PointTSLMaterial();
      glsl.applyBlendingMode(mode);
      tsl.applyBlendingMode(mode);
      const expected = getPointBlendingState(mode, 1.0);
      for (const field of STATE_FIELDS) {
        expect(glsl[field], `GLSL ${field} for '${mode}'`).toBe(expected[field]);
        expect(tsl[field], `TSL ${field} for '${mode}'`).toBe(expected[field]);
      }
      expect(glsl.userData.blendingMode).toBe(mode);
      expect(tsl.userData.blendingMode).toBe(mode);
    });
  }
});

// H — TSL constructor honors explicit transparent/depthTest overrides
// AFTER the factory tail's mode-derived state, exactly like the GLSL
// twin (additive state otherwise forces depthTest=false).
describe('PointTSLMaterial constructor explicit overrides', () => {
  it('depthTest: true survives additive construction', () => {
    const mat = new PointTSLMaterial({ blendingMode: 'additive', depthTest: true });
    expect(mat.depthTest).toBe(true);
    expect(mat.userData.depthTest).toBe(true);
  });

  it('transparent: false survives additive construction', () => {
    const mat = new PointTSLMaterial({ blendingMode: 'additive', transparent: false });
    expect(mat.transparent).toBe(false);
  });
});

describe('PointMaterial single-pass billboards (both backends)', () => {
  it('stays OFF the transparent two-pass path (FrontSide — no DoubleSide, no second pass)', () => {
    // THREE's two-pass transparent render trips only on
    // transparent + DoubleSide + !forceSinglePass. Line/gsplat billboards
    // need DoubleSide and pin forceSinglePass=true; points avoid the
    // guard entirely by staying FrontSide (their quad winding is fixed
    // by the screen-space expansion). Flipping side to DoubleSide here
    // without forceSinglePass would silently DOUBLE fragment work.
    for (const mat of [new PointMaterial(), new PointTSLMaterial()]) {
      expect(mat.side).not.toBe(THREE.DoubleSide);
    }
  });
});
