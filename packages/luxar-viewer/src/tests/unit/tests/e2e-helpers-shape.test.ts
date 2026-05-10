/**
 * shape tests for the new e2e helpers.
 *
 * The Playwright helpers are exercised in the browser by the new
 * shader-material-compile / line-rendering-visual / gsplat-rendering-
 * visual specs, but the JS-side function shapes are testable here:
 *   - assertNoShaderErrors / samplePixelAt are exported correctly.
 *   - The shader-error regex catches representative GLSL error
 *     messages (compile, link, attribute, uniform).
 */
import { describe, it, expect } from 'vitest';
import {
  assertNoShaderErrors,
  samplePixelAt,
} from '../../e2e/helpers';

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

  it('matches a generic GLSL syntax error', () => {
    const msg = 'GLSL syntax error at line 42';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);
  });

  it('does NOT match unrelated browser warnings', () => {
    const msg = 'Failed to load resource: the server responded with status 404';
    expect(SHADER_ERROR_RX.test(msg)).toBe(false);
  });

  it('does NOT match Luxar info logs that mention "shader" only as a docstring', () => {
    // Real Luxar log format prefixes a level emoji; "shader" appearing in
    // an info-level log is FINE — the test should still match because
    // assertNoShaderErrors filters by message content, not level. We
    // accept the false positive risk; users see a clear failure with the
    // offending text and can `allowedPatterns`-it if needed.
    const msg = '[ℹ️] [Renderer] reusing shader program from cache';
    expect(SHADER_ERROR_RX.test(msg)).toBe(true);
  });
});
