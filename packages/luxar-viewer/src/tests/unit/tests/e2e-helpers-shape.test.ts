/**
 * shape tests for the new e2e helpers.
 *
 * The Playwright helpers are exercised in the browser by the new
 * shader-material-compile / line-rendering-visual / gsplat-rendering-
 * visual specs, but the JS-side function shapes are testable here:
 *   - assertNoShaderErrors / samplePixelAt are exported correctly.
 *   - The shader-error regex catches representative GLSL error
 *     messages (compile, link, attribute, uniform).
 *
 * IMPORTANT (tests-meta.md C1/C2): SHADER_ERROR_RX below is a LOCAL COPY
 * that diverges intentionally from the real pattern in
 * src/tests/e2e/helpers.ts:1228 (used by assertNoShaderErrors). The real
 * regex is stricter — it requires e.g. `GLSL\s*(error|failure|failed)`
 * rather than the bare `GLSL` match used here. This file is testing the
 * local copy for documentation purposes; do not rely on it to predict
 * the runtime behavior of assertNoShaderErrors. Keep the two patterns
 * in sync MANUALLY when adding new browser-specific error variants.
 */
import { describe, it, expect } from 'vitest';
import { assertNoShaderErrors, samplePixelAt } from '../../e2e/helpers';

const SHADER_ERROR_RX =
  /shader|GLSL|attribute.*not\s*found|uniform.*not\s*found|fragment\s*shader|vertex\s*shader|program\s*link|invalid_operation/i;

describe('e2e helper exports', () => {
  it('assertNoShaderErrors is a function', () => {
    expect(typeof assertNoShaderErrors).toBe('function');
  });

  it('samplePixelAt is a function', () => {
    expect(typeof samplePixelAt).toBe('function');
  });
});

describe('shader-error regex coverage', () => {
  it('matches a Chromium-style fragment shader compile error', () => {
    const msg = 'WebGL: ERROR: 0:42: compilation error in fragment shader';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);
  });

  it('matches an attribute-not-found warning', () => {
    const msg = 'WebGL: WARNING: attribute aStartScalar not found in shader program';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);
  });

  it('matches a uniform-not-found warning', () => {
    const msg = 'WebGL: uniform uColormapTex not found in linked program';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);
  });

  it('matches a program link failure', () => {
    const msg = 'WebGL: failed to program link: varying vT not consumed by fragment shader';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);
  });

  it('[tests-meta.md C2] "GLSL syntax error" matches LOCAL regex — but documents that REAL regex would NOT', () => {
    // tests-meta.md C2[P2]: the previous test name "matches a generic GLSL
    // syntax error" implied the real `assertNoShaderErrors` regex catches
    // this string. It does NOT — the real regex at e2e/helpers.ts:1228 is
    // `GLSL\s*(error|failure|failed)` (requires error/failure/failed), so
    // `'GLSL syntax error at line 42'` (has `GLSL` then `syntax`, NOT one
    // of the keyword group) does not match. Pin BOTH facts:
    //   1. LOCAL regex matches (bare `GLSL` alternative).
    //   2. The REAL regex shape from helpers.ts would NOT match this string.
    //
    // A reader auditing "does our production helper catch this Chrome
    // message?" must NOT conclude "yes" from the local-copy test alone.
    const msg = 'GLSL syntax error at line 42';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);

    // Mirror the real production regex (kept in sync manually per the
    // file header). If this constant drifts, search e2e/helpers.ts for
    // the real value — drift here ≡ drift there.
    const REAL_SHADER_ERROR_RX = /GLSL\s*(error|failure|failed)/i;
    expect(
      REAL_SHADER_ERROR_RX.test(msg),
      'real regex requires GLSL+error/failure/failed; "GLSL syntax error" does NOT match'
    ).toBe(false);
  });

  it('does NOT match unrelated browser warnings', () => {
    const msg = 'Failed to load resource: the server responded with status 404';
    expect(SHADER_ERROR_RX.test(msg)).toBe(false);
  });

  it('the local SHADER_ERROR_RX over-matches info logs that mention "shader" (acceptable for this docstring-test)', () => {
    // tests-meta.md W4 fix: the previous comment claimed
    // "assertNoShaderErrors filters by message content, not level". That's
    // INCORRECT — the real helper at e2e/helpers.ts:1226 filters to
    // `errors + warnings` only, so info-level logs never reach the regex
    // in production. This test exercises only the local-copy regex; it
    // accepts the false positive as a documentation artifact.
    const msg = '[ℹ️] [Renderer] reusing shader program from cache';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);
  });
});
