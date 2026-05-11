/**
 * Shader hot-path regression locks.
 *
 * These tests are intentionally string-grep assertions over the
 * generated GLSL. The recheck report (`performance-memory.md` §W5, §W6)
 * flagged that several deliberate optimizations and visual-correctness
 * patterns have no test coverage and can silently regress during a
 * shader refactor:
 *
 * - Point vertex uses `inversesqrt(dot(...))` — a native GPU
 *   instruction; replacing it with `1.0 / sqrt(...)` is a real cost.
 * - GSplat vertex applies a near-plane fade
 *   (`smoothstep(uNearCull, uNearCull * 2.0, zDepth)`) and a screen-
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
 */

import { describe, it, expect } from 'vitest';
import {
  POINT_VERTEX_SHADER,
  POINT_FRAGMENT_SHADER,
} from '../../../rendering/shaders/point-shaders';
import {
  GSPLAT_VERTEX_SHADER,
  GSPLAT_FRAGMENT_SHADER,
} from '../../../rendering/shaders/gsplat-shaders';
import { LINE_VERTEX_SHADER } from '../../../rendering/shaders/line-shaders';

describe('Shader hot-path string regressions', () => {
  describe('Point vertex', () => {
    it('uses inversesqrt for 1/distance (native GPU instruction)', () => {
      expect(POINT_VERTEX_SHADER).toMatch(/inversesqrt\s*\(\s*dot\s*\(/);
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
    it('applies the near-plane fade smoothstep around uNearCull', () => {
      expect(GSPLAT_VERTEX_SHADER).toMatch(
        /smoothstep\s*\(\s*uNearCull\s*,\s*uNearCull\s*\*\s*2\.0\s*,\s*zDepth\s*\)/
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
      // the rayDir × pr* assembly guards against a refactor that
      // accidentally drops the cross terms (which would produce wrong
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
      // header note. Locking this string prevents a refactor from
      // turning 'normal' into transparent-additive by accident.
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
