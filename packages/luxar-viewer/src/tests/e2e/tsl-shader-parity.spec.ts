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
  return page.evaluate((name) => Array.from(window.__tslHarness!.renderGLSL(name)), shaderName);
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
  test('const-rgb diagnostic: solid-colour fragment matches between backends', async ({ page }) => {
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

  test('mega (ACES tone mapping — the production default) renders identically', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'mega-aces');
    const tslResult = await runTSL(page, 'mega-aces');

    expect(tslResult.pixels.length).toBe(glslPixels.length);
    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Mega+ACES parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
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

  test('point: PointMaterial sprite expansion + GOG + Gaussian falloff', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point');
    const tslResult = await runTSL(page, 'point');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    // Sample sprite-area pixels. 64x64 viewport, sprite centred at (32, 32).
    const samplePx = (px: number[], x: number, y: number) => {
      const o = (y * 64 + x) * 4;
      return `${px[o]},${px[o + 1]},${px[o + 2]},${px[o + 3]}`;
    };
    const offsets = [32, 30, 28, 26, 24, 20];
    const samples = offsets
      .map(
        (xo) =>
          `  (${xo},32) GLSL=${samplePx(glslPixels, xo, 32)} TSL=${samplePx(tslResult.pixels, xo, 32)}`
      )
      .join('\n');
    expect(
      diff,
      `Point parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSprite samples:\n${samples}`
    ).toBeLessThan(2.0);
  });

  test('point-colormap: USE_COLORMAP LUT lookup with gamma applied to the value', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-colormap');
    const tslResult = await runTSL(page, 'point-colormap');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [32, 30, 28]
      .map(
        (xo) =>
          `  (${xo},32) GLSL=${samplePx(glslPixels, xo, 32)} TSL=${samplePx(tslResult.pixels, xo, 32)}`
      )
      .join('\n');
    expect(
      diff,
      `Point-colormap parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
    ).toBeLessThan(2.0);
  });

  test('point-gamma-one: LUXAR_GAMMA_ONE fast path matches the slow pow() path', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-gamma-one');
    const tslResult = await runTSL(page, 'point-gamma-one');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [32, 30, 28]
      .map(
        (xo) =>
          `  (${xo},32) GLSL=${samplePx(glslPixels, xo, 32)} TSL=${samplePx(tslResult.pixels, xo, 32)}`
      )
      .join('\n');
    expect(
      diff,
      `Point-gamma-one parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
    ).toBeLessThan(2.0);
  });

  test('line: instanced quad line with width / sharpness / GOG', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line');
    const tslResult = await runTSL(page, 'line');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    // Line should run horizontally across y=32 in the 64×64 viewport.
    // Sample along the line (varying x at y=32) and across the line
    // (varying y at x=32, the centre).
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [
      `  (16,32) GLSL=${samplePx(glslPixels, 16, 32)} TSL=${samplePx(tslResult.pixels, 16, 32)}`,
      `  (32,32) GLSL=${samplePx(glslPixels, 32, 32)} TSL=${samplePx(tslResult.pixels, 32, 32)}`,
      `  (48,32) GLSL=${samplePx(glslPixels, 48, 32)} TSL=${samplePx(tslResult.pixels, 48, 32)}`,
      `  (32,28) GLSL=${samplePx(glslPixels, 32, 28)} TSL=${samplePx(tslResult.pixels, 32, 28)}`,
      `  (32,30) GLSL=${samplePx(glslPixels, 32, 30)} TSL=${samplePx(tslResult.pixels, 32, 30)}`,
      `  (32,34) GLSL=${samplePx(glslPixels, 32, 34)} TSL=${samplePx(tslResult.pixels, 32, 34)}`,
    ].join('\n');
    expect(
      diff,
      `Line parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nLine samples:\n${samples}`
    ).toBeLessThan(2.0);
  });

  test('line-colormap: USE_COLORMAP LUT with per-endpoint scalars + value gamma', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-colormap');
    const tslResult = await runTSL(page, 'line-colormap');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [16, 32, 48]
      .map(
        (x) =>
          `  (${x},32) GLSL=${samplePx(glslPixels, x, 32)} TSL=${samplePx(tslResult.pixels, x, 32)}`
      )
      .join('\n');
    expect(
      diff,
      `Line-colormap parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
    ).toBeLessThan(2.0);
  });

  test('gsplat: isotropic splat covariance projection + Mahalanobis fragment', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat');
    const tslResult = await runTSL(page, 'gsplat');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [
      `  (32,32) GLSL=${samplePx(glslPixels, 32, 32)} TSL=${samplePx(tslResult.pixels, 32, 32)}`,
      `  (28,32) GLSL=${samplePx(glslPixels, 28, 32)} TSL=${samplePx(tslResult.pixels, 28, 32)}`,
      `  (24,32) GLSL=${samplePx(glslPixels, 24, 32)} TSL=${samplePx(tslResult.pixels, 24, 32)}`,
      `  (32,28) GLSL=${samplePx(glslPixels, 32, 28)} TSL=${samplePx(tslResult.pixels, 32, 28)}`,
    ].join('\n');
    // The gsplat shader's covariance projection has more accumulated
    // float-precision drift than the simpler pipelines (matrix
    // products + reciprocals + eigendecomp). Allow a slightly looser
    // tolerance — the structural correctness is what matters.
    expect(
      diff,
      `GSplat parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
    ).toBeLessThan(3.0);
  });

  test('gsplat-colormap: USE_COLORMAP LUT keyed on amplitude + value gamma', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-colormap');
    const tslResult = await runTSL(page, 'gsplat-colormap');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [
      `  (32,32) GLSL=${samplePx(glslPixels, 32, 32)} TSL=${samplePx(tslResult.pixels, 32, 32)}`,
      `  (28,32) GLSL=${samplePx(glslPixels, 28, 32)} TSL=${samplePx(tslResult.pixels, 28, 32)}`,
    ].join('\n');
    // Same looser tolerance as `gsplat` — covariance projection drift.
    expect(
      diff,
      `GSplat-colormap parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
    ).toBeLessThan(3.0);
  });

  test('gsplat-gamma-one: LUXAR_GAMMA_ONE fast path matches the slow pow() path', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-gamma-one');
    const tslResult = await runTSL(page, 'gsplat-gamma-one');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [
      `  (32,32) GLSL=${samplePx(glslPixels, 32, 32)} TSL=${samplePx(tslResult.pixels, 32, 32)}`,
      `  (28,32) GLSL=${samplePx(glslPixels, 28, 32)} TSL=${samplePx(tslResult.pixels, 28, 32)}`,
    ].join('\n');
    // Same looser tolerance as `gsplat` — covariance projection drift.
    expect(
      diff,
      `GSplat-gamma-one parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
    ).toBeLessThan(3.0);
  });

  test('gsplat-pick: covariance projection with nodeId / elementId / brightness output', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-pick');
    const tslResult = await runTSL(page, 'gsplat-pick');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `GSplat-pick parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(3.0);
  });

  test('line-pick: instanced quad line with nodeId / elementId / brightness output', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-pick');
    const tslResult = await runTSL(page, 'line-pick');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Line-pick parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  test('point-pick: tight sprite with nodeId / elementId / brightness output', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-pick');
    const tslResult = await runTSL(page, 'point-pick');

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Point-pick parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
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
