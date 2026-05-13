/**
 * TSL ↔ GLSL parity spec.
 *
 * For each shader with both a GLSL3 source and a TSL factory, renders
 * both backends through a deterministic fullscreen-pass harness
 * (`/tsl-harness.html`) and compares pixels with a small per-channel
 * tolerance.
 *
 * The spec starts with a diagnostic `const-rgb` shader that simply
 * outputs `vec4(0.5, 0.25, 0.75, 1.0)`. If THAT fails parity, the
 * divergence is at the renderer level (output color space, gamma,
 * tone mapping) — not in a per-shader port. The remaining cases
 * become meaningful only once the diagnostic passes.
 *
 * Why this lives in E2E (Playwright) rather than vitest unit tests:
 * the harness needs a real WebGL2 context, which Three.js's
 * `WebGPURenderer({ forceWebGL: true })` only initialises inside a
 * browser.
 *
 * Captured GLSL-string snapshot diffs are deferred — the path through
 * Three.js's internal `_objects` ChainMap that holds compiled shader
 * source isn't reconstructible from the harness yet. A follow-up will
 * wire it in once the pixel-parity tests stabilise.
 *
 * @module tests/e2e/tsl-shader-parity.spec
 */

import { test, expect, type Page } from '@playwright/test';

const HARNESS_URL = '/tsl-harness.html';

type TSLResult = {
  pixels: number[];
  vertexShader: string;
  fragmentShader: string;
};

async function bootHarness(page: Page): Promise<string[]> {
  await page.goto(HARNESS_URL);
  await page.waitForFunction(() => Boolean(window.__tslHarness));
  await page.evaluate(() => window.__tslHarness!.ready);
  return page.evaluate(() => window.__tslHarness!.listShaders());
}

async function runGLSL(page: Page, shaderName: string): Promise<number[]> {
  return page.evaluate(
    (name) => Array.from(window.__tslHarness!.renderGLSL(name)),
    shaderName
  );
}

async function runTSL(page: Page, shaderName: string): Promise<TSLResult> {
  return page.evaluate(async (name) => {
    const result = await window.__tslHarness!.renderTSL(name);
    return {
      pixels: Array.from(result.pixels),
      vertexShader: result.vertexShader,
      fragmentShader: result.fragmentShader,
    };
  }, shaderName);
}

/** Mean absolute per-channel difference on a 0-255 scale. */
function meanAbsDiff(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum / a.length;
}

/** First N RGBA quadruplets of a buffer, formatted for human reading. */
function previewPixels(pixels: number[], count = 4): string {
  const lines: string[] = [];
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    lines.push(
      `  [${i}] r=${pixels[o]}  g=${pixels[o + 1]}  b=${pixels[o + 2]}  a=${pixels[o + 3]}`
    );
  }
  return lines.join('\n');
}


test.describe('TSL ↔ GLSL shader parity', () => {
  test('const-rgb diagnostic: solid-colour fragment matches between backends', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'const-rgb');
    const tslResult = await runTSL(page, 'const-rgb');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);

    // If this assertion fails, the divergence is renderer-level
    // (output color space, gamma, etc.) and the rest of the
    // shader-parity tests below cannot give meaningful signal.
    expect(
      diff,
      'Diagnostic failure: TSL & GLSL produce different pixels for a constant-colour shader.\n' +
        `Mean abs diff: ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  test('fxaa renders identically through both backends', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'fxaa');
    const tslResult = await runTSL(page, 'fxaa');

    expect(tslResult.pixels.length).toBe(glslPixels.length);
    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `FXAA parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  test('bloom-threshold renders identically through both backends', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'bloom-threshold');
    const tslResult = await runTSL(page, 'bloom-threshold');

    expect(tslResult.pixels.length).toBe(glslPixels.length);
    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Bloom-threshold parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  test('mega (default config, Linear tone mapping) renders identically', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'mega');
    const tslResult = await runTSL(page, 'mega');

    expect(tslResult.pixels.length).toBe(glslPixels.length);
    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Mega parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  test('mega with USE_BLOOM matches across backends', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'mega-bloom');
    const tslResult = await runTSL(page, 'mega-bloom');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Mega+bloom parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  // The points-hello test was *removed* after diagnostic dumps showed
  // it passed by accident — the TSL path's output is all zeros because
  // `gl_PointSize = 1.0` is hardcoded in r184's `GLSLNodeBuilder`
  // (`renderers/webgl-fallback/nodes/GLSLNodeBuilder.js:1416`). The
  // GLSL3 path produced a small disk; the TSL path produced nothing;
  // the mean-abs-diff fell below the 2.0 tolerance through accidental
  // averaging. See MIGRATION_PROGRESS.md for the corrected approach:
  // M11-M16 must switch the geometry container from `THREE.Points` to
  // `THREE.Sprite` / `THREE.InstancedMesh` so the TSL setupVertexSprite
  // path (which respects `sizeNode`) actually fires.

  test('mega with USE_DETECTOR_NOISE matches across backends', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'mega-detector-noise');
    const tslResult = await runTSL(page, 'mega-detector-noise');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      'Mega+detector-noise parity: mean abs diff ' +
        diff.toFixed(2) +
        ' on 0-255 scale.\n' +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  test('mega with USE_VIGNETTE matches across backends', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'mega-vignette');
    const tslResult = await runTSL(page, 'mega-vignette');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Mega+vignette parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });
});
