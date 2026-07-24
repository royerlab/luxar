/**
 * Shader hot-path regression locks.
 *
 * These tests are intentionally string-grep assertions over the
 * generated GLSL. The recheck report (`performance-memory.md` §W5, §W6)
 * flagged that several deliberate optimizations and visual-correctness
 * patterns have no test coverage and can silently regress when shader
 * code changes:
 *
 * - Point vertex sizes from VIEW-SPACE DEPTH (`-mvPosition.z`, matching
 *   lines/gsplats) — reverting to Euclidean distance shrinks
 *   edge-of-screen points by cos(theta).
 * - GSplat vertex applies the unified near fade
 *   (`perspectiveNearFade(uIsOrtho, centerCam.z, uNearCull)`) and a screen-
 *   coverage fade (`smoothstep(maxExtent * 0.5, maxExtent, projectedExtent)`)
 *   combined via `min(depthFade, coverageFade)`. Removing either side
 *   re-introduces large-splat flicker near the camera.
 * - GSplat fragment alpha contract (PR #561): the
 *   `LUXAR_NORMAL_PREMULT` branch ('normal' mode) emits the clamped
 *   coverage alpha `clamp(intensity * uOpacity, 0, 1)`; every OTHER
 *   mode keeps `fragColor = vec4(finalColor, 1.0)`.
 * - Line vertex computes `vWidthFade` to keep pixel-clamped lines
 *   from overcontributing to additive blending.
 *
 * Tests live here (not in the per-material spec files) so that a single
 * shader-pattern rename is forced to update one centralized place
 * rather than slipping through.
 *
 * Audit acknowledgment (rendering.md [W2], [EXCLUDED-CATEGORY:
 * shader-tsl-parity]): every assertion in this file is intentionally
 * an `expect(SHADER_SOURCE).toMatch(/regex/)` — see project memory
 * "Keep GLSL shaders as reference" + the TSL/GLSL parity harness.
 * These checks are documented "regression locks" and are explicitly
 * excluded from the mutation-killing rubric. Do NOT replace with
 * behavioural assertions: the GLSL3 sources are retained as the
 * reference implementation and these tests are the only thing pinning
 * the optimization patterns at the string level.
 */

import { describe, it, expect } from 'vitest';
import {
  POINT_VERTEX_SHADER,
  POINT_FRAGMENT_SHADER,
} from '../../../rendering/materials/point/shader-glsl';
import {
  GSPLAT_VERTEX_SHADER,
  GSPLAT_FRAGMENT_SHADER,
} from '../../../rendering/materials/gsplat/shader-glsl';
import { VOLUMETRIC_SERIES_TAU_THRESHOLD } from '../../../rendering/materials/_shared/volumetric';
import { LINE_VERTEX_SHADER } from '../../../rendering/materials/line/shader-glsl';

describe('Shader hot-path string regressions', () => {
  describe('Point vertex', () => {
    it('sizes from view-space depth, not Euclidean distance', () => {
      // 1e-20 = pure INF guard (scale-free); an absolute 1e-4 shrank
      // sprites ~100x on tiny-unit scenes.
      expect(POINT_VERTEX_SHADER).toMatch(/max\s*\(\s*-mvPosition\.z\s*,\s*1e-20\s*\)/);
      // Reverting to Euclidean distance would shrink edge-of-screen
      // points by cos(theta) relative to lines/gsplats.
      expect(POINT_VERTEX_SHADER).not.toMatch(/inversesqrt\s*\(\s*dot\s*\(/);
    });
  });

  describe('Point fragment', () => {
    it('does not compile in fragColor with stranded transparency state', () => {
      // Two branches: premultiplied alpha for additive/luminous and
      // straight color for opaque. Both must end with a fragColor
      // assignment that does not multiply by an undefined alpha.
      expect(POINT_FRAGMENT_SHADER).toMatch(/fragColor\s*=\s*vec4\(/);
    });
  });

  describe('GSplat vertex', () => {
    it('applies the unified perspectiveNearFade around uNearCull', () => {
      // The floor is 1e-20 — a degenerate-smoothstep guard only, so the
      // scene-relative uNearCull is never overridden (tiny-unit scenes).
      expect(GSPLAT_VERTEX_SHADER).toMatch(
        /perspectiveNearFade\s*\(\s*uIsOrtho\s*,\s*centerCam\.z\s*,\s*max\(uNearCull, 1e-20\)\s*\)/
      );
      // The shared helper carries the smoothstep.
      expect(GSPLAT_VERTEX_SHADER).toMatch(
        /smoothstep\s*\(\s*nearCull\s*,\s*nearCull\s*\*\s*2\.0\s*,\s*-viewZ\s*\)/
      );
    });

    it('applies the screen-coverage fade smoothstep around maxExtent', () => {
      expect(GSPLAT_VERTEX_SHADER).toMatch(
        /smoothstep\s*\(\s*maxExtent\s*\*\s*0\.5\s*,\s*maxExtent\s*,\s*projectedExtent\s*\)/
      );
    });

    it('combines the two fades via min(depthFade, coverageFade)', () => {
      expect(GSPLAT_VERTEX_SHADER).toMatch(
        /nearFade\s*=\s*min\s*\(\s*depthFade\s*,\s*coverageFade\s*\)/
      );
    });

    it('uses inversesqrt for the Mahalanobis ray-quad reciprocal', () => {
      expect(GSPLAT_VERTEX_SHADER).toMatch(/inversesqrt\s*\(\s*quad\s*\)/);
    });

    it('reads the per-splat alpha from texel3.y and passes it through vAlpha (sanitized)', () => {
      // Per-element opacity rides in texel3.y (element-texture-layout.ts). A
      // mutation that read the wrong channel or dropped the varying would
      // silently ignore imported 3DGS opacity — pin both the read and the
      // varying assignment. A pure-math test can't see either. The
      // assignment routes through sanitizeNonNegative(·, 1.0): alpha is
      // load-bearing in every mode, and a NaN from hand-crafted zarr
      // would otherwise poison volumetric τ into NaN pixels.
      expect(GSPLAT_VERTEX_SHADER).toMatch(/float\s+aAlpha\s*=\s*splatT3\.y/);
      expect(GSPLAT_VERTEX_SHADER).toMatch(/vAlpha\s*=\s*sanitizeNonNegative\(aAlpha, 1\.0\)/);
    });

    it('computes the Mahalanobis quadratic form rᵀΣ⁻¹r before the inversesqrt', () => {
      // The shader assembles the per-axis pre-multiplied vector
      // (prx/pry/prz = Σ⁻¹·r) and then folds it with `rayDir` to form
      // the scalar quadratic that feeds `inversesqrt(quad)`. Locking
      // the rayDir × pr* assembly guards against accidentally dropping
      // the cross terms (which would produce wrong
      // splat sizes that look "almost right" — the worst kind of
      // regression).
      expect(GSPLAT_VERTEX_SHADER).toMatch(/rayDir\.x\s*\*\s*prx/);
      expect(GSPLAT_VERTEX_SHADER).toMatch(/rayDir\.y\s*\*\s*pry/);
      expect(GSPLAT_VERTEX_SHADER).toMatch(/rayDir\.z\s*\*\s*prz/);
    });
  });

  describe('GSplat fragment alpha contract (NORMAL_PREMULT / VOLUMETRIC / alpha-1 split)', () => {
    // Split the fragment source on the preprocessor markers so each
    // regex asserts within its OWN branch text — matching against the
    // whole source could hit another branch and pass vacuously. The
    // output chain is #ifdef LUXAR_NORMAL_PREMULT / #elif
    // defined(LUXAR_VOLUMETRIC) / #else / #endif.
    const ifdefStart = GSPLAT_FRAGMENT_SHADER.indexOf('#ifdef LUXAR_NORMAL_PREMULT');
    const elifIdx = GSPLAT_FRAGMENT_SHADER.indexOf('#elif defined(LUXAR_VOLUMETRIC)', ifdefStart);
    const elseIdx = GSPLAT_FRAGMENT_SHADER.indexOf('#else', elifIdx);
    const endifIdx = GSPLAT_FRAGMENT_SHADER.indexOf('#endif', elseIdx);
    const premultBranch = GSPLAT_FRAGMENT_SHADER.slice(ifdefStart, elifIdx);
    const volumetricBranch = GSPLAT_FRAGMENT_SHADER.slice(elifIdx, elseIdx);
    const alphaOneBranch = GSPLAT_FRAGMENT_SHADER.slice(elseIdx, endifIdx);

    it('has exactly the #ifdef / #elif / #else / #endif structure the branch split relies on', () => {
      expect(ifdefStart).toBeGreaterThanOrEqual(0);
      expect(elifIdx).toBeGreaterThan(ifdefStart);
      expect(elseIdx).toBeGreaterThan(elifIdx);
      expect(endifIdx).toBeGreaterThan(elseIdx);
    });

    it("'normal' (#ifdef branch) emits the clamped coverage alpha, premultiplied", () => {
      // PR #561: normal mode is real alpha-over — RGB carries the full
      // premultiplied contribution, alpha a CLAMPED coverage term for
      // the One / OneMinusSrcAlpha framebuffer state.
      expect(premultBranch).toMatch(
        /coverage\s*=\s*clamp\s*\(\s*intensity\s*\*\s*uOpacity\s*,\s*0\.0\s*,\s*1\.0\s*\)/
      );
      expect(premultBranch).toMatch(/fragColor\s*=\s*vec4\s*\(\s*finalColor\s*,\s*coverage\s*\)/);
      expect(premultBranch).not.toMatch(/vec4\s*\(\s*finalColor\s*,\s*1\.0\s*\)/);
    });

    it("'volumetric' (#elif branch) emits self-screened emission + absorption alpha", () => {
      // Emission–absorption (VOLUMETRIC_BLENDING_SPEC.md §3.1): alpha is
      // the physical 1 − e^(−τ), RGB is screened by S(τ) with the τ→0
      // series — NOT the alpha=1 contract and NOT the coverage clamp.
      expect(volumetricBranch).toMatch(/alpha\s*=\s*1\.0\s*-\s*exp\s*\(\s*-tau\s*\)/);
      // Series guard — the threshold is interpolated from ./math (the
      // ALPHA_CLAMP single-source pattern; 1e-3 serializes as "0.001"),
      // so build the expected literal from the same constant.
      const threshold = String(VOLUMETRIC_SERIES_TAU_THRESHOLD).replace('.', '\\.');
      expect(volumetricBranch).toMatch(new RegExp(`tau\\s*<\\s*${threshold}`));
      expect(volumetricBranch).toMatch(
        /fragColor\s*=\s*vec4\s*\(\s*finalColor\s*\*\s*screen\s*,\s*alpha\s*\)/
      );
      expect(volumetricBranch).not.toMatch(/coverage/);
      expect(volumetricBranch).not.toMatch(/vec4\s*\(\s*finalColor\s*,\s*1\.0\s*\)/);
    });

    it('every remaining mode (#else branch) keeps the alpha=1.0 contract', () => {
      // additive/luminous rely on SrcAlpha being the identity factor
      // (what makes the shared AdditiveBlending state the linear
      // One + One sum); max compares premultiplied RGB directly.
      expect(alphaOneBranch).toMatch(/fragColor\s*=\s*vec4\s*\(\s*finalColor\s*,\s*1\.0\s*\)/);
      expect(alphaOneBranch).not.toMatch(/coverage/);
    });

    it('folds per-splat alpha into intensity — linear (non-volumetric) vs w-mapped (volumetric)', () => {
      // Per-element opacity multiplies into `intensity` right after the
      // shifted-Gaussian eval, in BOTH branch families. A dropped multiply
      // would silently ignore imported 3DGS opacity — pin both forms. The
      // volumetric form maps a → optical density w = −log(1−a), gated by
      // uHasElementAlpha; the else form is the plain linear factor.
      // (Mutation lesson: a pure-TS math test cannot see either.)
      const intensityDecl = GSPLAT_FRAGMENT_SHADER.indexOf('float intensity =');
      const foldStart = GSPLAT_FRAGMENT_SHADER.indexOf('#ifdef LUXAR_VOLUMETRIC', intensityDecl);
      const foldElse = GSPLAT_FRAGMENT_SHADER.indexOf('#else', foldStart);
      const foldEndif = GSPLAT_FRAGMENT_SHADER.indexOf('#endif', foldElse);
      const volFold = GSPLAT_FRAGMENT_SHADER.slice(foldStart, foldElse);
      const linFold = GSPLAT_FRAGMENT_SHADER.slice(foldElse, foldEndif);
      expect(volFold).toMatch(
        /intensity\s*\*=\s*mix\s*\(\s*1\.0\s*,\s*-log\s*\(\s*1\.0\s*-\s*min\s*\(\s*vAlpha/
      );
      expect(volFold).toMatch(/uHasElementAlpha/);
      expect(linFold).toMatch(/intensity\s*\*=\s*vAlpha/);
      expect(linFold).not.toMatch(/uHasElementAlpha/);
    });

    it('volumetric color-discard is τ-aware: a black splat still absorbs', () => {
      // The zero-color early-discard must only fire when the optical
      // depth is ALSO negligible — a black splat (e.g. gain→0 pure-ink
      // occluder) keeps its absorption. Pin the conjunction inside the
      // LUXAR_VOLUMETRIC discard guard (mutation `&& tau < 1e-4` →
      // removed survived the suite before this test existed) and the
      // plain discard in the non-volumetric branch.
      // Anchor at the DISCARD guard specifically — the shader now has an
      // earlier `#ifdef LUXAR_VOLUMETRIC` block (the per-splat alpha →
      // optical-depth intensity mapping), so search from `vec3 adjusted`
      // (declared just above the discard guard) rather than the first ifdef.
      const adjustedDecl = GSPLAT_FRAGMENT_SHADER.indexOf('vec3 adjusted');
      const discardIfdef = GSPLAT_FRAGMENT_SHADER.indexOf('#ifdef LUXAR_VOLUMETRIC', adjustedDecl);
      const discardElse = GSPLAT_FRAGMENT_SHADER.indexOf('#else', discardIfdef);
      const discardEndif = GSPLAT_FRAGMENT_SHADER.indexOf('#endif', discardElse);
      expect(discardIfdef).toBeGreaterThanOrEqual(0);
      const volumetricDiscard = GSPLAT_FRAGMENT_SHADER.slice(discardIfdef, discardElse);
      const plainDiscard = GSPLAT_FRAGMENT_SHADER.slice(discardElse, discardEndif);
      expect(volumetricDiscard).toMatch(
        /if\s*\(\s*max\s*\(adjusted\.r,\s*max\(adjusted\.g,\s*adjusted\.b\)\)\s*<\s*1e-4\s*&&\s*tau\s*<\s*1e-4\s*\)\s*discard;/
      );
      expect(plainDiscard).toMatch(
        /if\s*\(\s*max\s*\(adjusted\.r,\s*max\(adjusted\.g,\s*adjusted\.b\)\)\s*<\s*1e-4\s*\)\s*discard;/
      );
      expect(plainDiscard).not.toMatch(/tau/);
    });
  });

  describe('Line vertex', () => {
    it('exposes vWidthFade so the fragment can damp pixel-clamped lines', () => {
      expect(LINE_VERTEX_SHADER).toMatch(/vWidthFade\s*=/);
    });
  });
});
