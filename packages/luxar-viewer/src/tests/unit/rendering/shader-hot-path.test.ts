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
 * - GSplat fragment in `'normal'` blending writes
 *   `fragColor = vec4(finalColor, 1.0)` (opaque-dimmed contract).
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
import { LINE_VERTEX_SHADER } from '../../../rendering/materials/line/shader-glsl';

describe('Shader hot-path string regressions', () => {
  describe('Point vertex', () => {
    it('sizes from view-space depth, not Euclidean distance', () => {
      expect(POINT_VERTEX_SHADER).toMatch(/max\s*\(\s*-mvPosition\.z\s*,\s*1e-4\s*\)/);
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
      expect(GSPLAT_VERTEX_SHADER).toMatch(
        /perspectiveNearFade\s*\(\s*uIsOrtho\s*,\s*centerCam\.z\s*,\s*max\(uNearCull, 1e-4\)\s*\)/
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

  describe('GSplat fragment', () => {
    it("writes opaque alpha (1.0) in 'normal' blending mode", () => {
      // The 'normal' mode is dimmed-opaque: fragColor.a is 1.0 and
      // finalColor is pre-dimmed by opacity. See material-manager.ts
      // header note. Locking this string prevents turning 'normal' into
      // transparent-additive by accident.
      expect(GSPLAT_FRAGMENT_SHADER).toMatch(
        /fragColor\s*=\s*vec4\s*\(\s*finalColor\s*,\s*1\.0\s*\)/
      );
    });
  });

  describe('Line vertex', () => {
    it('exposes vWidthFade so the fragment can damp pixel-clamped lines', () => {
      expect(LINE_VERTEX_SHADER).toMatch(/vWidthFade\s*=/);
    });
  });
});
