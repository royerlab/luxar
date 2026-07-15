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

/**
 * Count RGBA quadruplets that differ from the first pixel. A fully culled
 * frame is uniform (all clear-colour) → 0 differing pixels; a rendered sprite
 * makes some pixels differ. Used to assert the behind-camera guard produced no
 * fragments.
 */
function nonUniformPixelCount(pixels: number[]): number {
  const r0 = pixels[0];
  const g0 = pixels[1];
  const b0 = pixels[2];
  const a0 = pixels[3];
  let n = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i] !== r0 || pixels[i + 1] !== g0 || pixels[i + 2] !== b0 || pixels[i + 3] !== a0) {
      n++;
    }
  }
  return n;
}

/**
 * Non-vacuousness guard: a parity comparison is meaningless if either
 * side rendered nothing. This is exactly how the gsplat TSL
 * uninitialized-eigenvalue bug stayed invisible — every gsplat
 * variant's TSL side was empty and the mean-abs-diff of a small GLSL
 * sprite against a black buffer squeaked under the tolerance. Applies
 * to sprite/instanced variants (which have background); fullscreen
 * variants (const-rgb, mega, fxaa) are uniform by design and are
 * covered by their own solid-colour assertions.
 */
function assertBothRendered(glsl: number[], tsl: number[], name: string): void {
  const g = nonUniformPixelCount(glsl);
  const t = nonUniformPixelCount(tsl);
  expect(g, `${name}: GLSL side rendered ZERO pixels — vacuous parity`).toBeGreaterThan(0);
  expect(t, `${name}: TSL side rendered ZERO pixels — vacuous parity`).toBeGreaterThan(0);
  // The per-covered metric treats each buffer's own first pixel as its
  // background, so a cross-backend BACKGROUND divergence (clear color /
  // output transform drift) would be invisible to it. Pin equality here.
  expect(glsl.slice(0, 4), `${name}: backgrounds differ between backends`).toEqual(tsl.slice(0, 4));
}

/**
 * Mean absolute per-channel difference normalized by COVERED pixels —
 * quadruplets where either buffer differs from its own background
 * (first pixel). The whole-buffer `meanAbsDiff` dilutes errors by
 * footprint: a 200-px sprite that is ENTIRELY wrong contributes only
 * ~5% of a 64×64 buffer and can pass a small global tolerance. This
 * metric is footprint-invariant.
 */
function meanAbsDiffPerCoveredPixel(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
  }
  const isBg = (px: number[], i: number) =>
    px[i] === px[0] && px[i + 1] === px[1] && px[i + 2] === px[2] && px[i + 3] === px[3];
  let sum = 0;
  let covered = 0;
  for (let i = 0; i < a.length; i += 4) {
    if (isBg(a, i) && isBg(b, i)) continue;
    covered++;
    for (let c = 0; c < 4; c++) {
      sum += Math.abs(a[i + c] - b[i + c]);
    }
  }
  return covered === 0 ? 0 : sum / (covered * 4);
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

  test('point: PointMaterial sprite expansion + GOG + super-Gaussian falloff', async ({ page }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point');
    const tslResult = await runTSL(page, 'point');

    assertBothRendered(glslPixels, tslResult.pixels, 'point');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

  // The super-Gaussian exponent beta = 2^(6s - 2) drives a per-fragment
  // pow(rho, beta) + exp(...) chain. Parity must hold across the beta range,
  // not just the default — soft cusp (s=0.1) and hard edge (s=0.9).
  for (const variant of ['point-soft', 'point-hard'] as const) {
    test(`${variant}: super-Gaussian falloff parity across the beta range`, async ({ page }) => {
      await bootHarness(page);
      const glslPixels = await runGLSL(page, variant);
      const tslResult = await runTSL(page, variant);

      assertBothRendered(glslPixels, tslResult.pixels, variant);
      expect(
        meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
        `${variant}: per-covered-pixel parity (footprint-invariant)`
      ).toBeLessThan(2.0);
      const diff = meanAbsDiff(glslPixels, tslResult.pixels);
      expect(
        diff,
        `${variant} parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.`
      ).toBeLessThan(2.0);
    });
  }

  test('point-colormap: USE_COLORMAP LUT lookup with gamma applied to the value', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-colormap');
    const tslResult = await runTSL(page, 'point-colormap');

    assertBothRendered(glslPixels, tslResult.pixels, 'point-colormap');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-colormap: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'point-gamma-one');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-gamma-one: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'line');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'line-colormap');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-colormap: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-colormap');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-colormap: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-gamma-one');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-gamma-one: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

  test('gsplat-offcenter: off-center splat parity (fragcoord y-convention guard)', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-offcenter');
    const tslResult = await runTSL(page, 'gsplat-offcenter');

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-offcenter');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-offcenter: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);
    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    // Centered fixtures are mirror-symmetric about y = H/2, so they are
    // blind to top-left/bottom-left fragcoord convention bugs; this
    // off-center case is the guard for that whole class.
    expect(
      diff,
      `GSplat-offcenter parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.`
    ).toBeLessThan(3.0);
  });

  test('gsplat-tiny-sigma-fade: sub-0.1-sigma splat in the fade band renders at HALF amplitude', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-tiny-sigma-fade');
    const tslResult = await runTSL(page, 'gsplat-tiny-sigma-fade');

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-tiny-sigma-fade');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-tiny-sigma-fade: per-covered-pixel parity'
    ).toBeLessThan(2.0);

    // ABSOLUTE brightness assertion — parity alone is blind to this
    // bug because BOTH backends shared the maxLateralVar > 0.01 gate.
    // projectedExtent = uFx·sigma·truncate = 3200·0.005·3 = 48 px sits
    // mid-band (32, 64) → coverageFade = 0.5 → red-channel peak ≈ 127.
    // Pre-fix the gate skipped the fade for this sigma and the peak
    // saturated at ~255.
    let peak = 0;
    for (let i = 0; i < glslPixels.length; i += 4) {
      if (glslPixels[i] > peak) peak = glslPixels[i];
    }
    expect(
      peak,
      'tiny-sigma splat in the fade band must be coverage-faded (~50%); ' +
        `red peak ${peak} implies the fade was skipped`
    ).toBeLessThan(200);
    expect(peak).toBeGreaterThan(60); // sanity: still visibly rendered
  });

  test('gsplat-tiny-sigma-reject: sub-0.1-sigma splat past the fade band is coverage-culled on both backends', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-tiny-sigma-reject');
    const tslResult = await runTSL(page, 'gsplat-tiny-sigma-reject');

    // projectedExtent = 12800·0.005·3 = 192 px > maxExtent = 64 → the
    // coverage cull rejects the vertex. Pre-fix, the gate skipped the
    // fade and the unconditional extent clamp rendered a full-intensity
    // hard-edged rectangle (the deep-zoom artifact this guards).
    expect(
      nonUniformPixelCount(glslPixels),
      'GLSL: tiny-sigma splat past the fade band must be culled, not clamped to a hard rectangle'
    ).toBe(0);
    expect(
      nonUniformPixelCount(tslResult.pixels),
      'TSL: tiny-sigma splat past the fade band must be culled, not clamped to a hard rectangle'
    ).toBe(0);
  });

  test('gsplat-normal-premult: LUXAR_NORMAL_PREMULT coverage alpha matches TSL normal branch', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-normal-premult');
    const tslResult = await runTSL(page, 'gsplat-normal-premult');

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-normal-premult');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-normal-premult: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    const samplePx = (px: number[], x: number, y: number) =>
      `${px[(y * 64 + x) * 4]},${px[(y * 64 + x) * 4 + 1]},${px[(y * 64 + x) * 4 + 2]},${px[(y * 64 + x) * 4 + 3]}`;
    const samples = [
      `  (32,32) GLSL=${samplePx(glslPixels, 32, 32)} TSL=${samplePx(tslResult.pixels, 32, 32)}`,
      `  (28,32) GLSL=${samplePx(glslPixels, 28, 32)} TSL=${samplePx(tslResult.pixels, 28, 32)}`,
    ].join('\n');
    // The alpha channel must carry clamp(intensity·uOpacity) identically on
    // both backends (guards the NodeMaterial double-premultiplication trap).
    const centerAlpha = glslPixels[(32 * 64 + 32) * 4 + 3];
    expect(
      centerAlpha,
      `GSplat-normal-premult: expected sub-saturated coverage alpha at splat center, got ${centerAlpha}`
    ).toBeGreaterThan(0);
    expect(centerAlpha).toBeLessThan(255);
    // Same looser tolerance as `gsplat` — covariance projection drift (this
    // variant additionally exercises the sum-projection ray-integral path).
    expect(
      diff,
      `GSplat-normal-premult parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
    ).toBeLessThan(3.0);
  });

  test('gsplat-pick: covariance projection with nodeId / elementId / brightness output', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-pick');
    const tslResult = await runTSL(page, 'gsplat-pick');

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-pick');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-pick: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'line-pick');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-pick: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

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

    assertBothRendered(glslPixels, tslResult.pixels, 'point-pick');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-pick: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `Point-pick parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
        `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
        `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
    ).toBeLessThan(2.0);
  });

  // Behind-camera parity. A perspective camera (uIsOrtho:0) with the point
  // behind it — the only cases exercising the perspective path and the
  // behind-camera reject (`uIsOrtho == 0 && mvPosition.z >= 0`); every other
  // point case is ortho. Each asserts GLSL ↔ TSL produce identical frames and
  // that the frame is uniform (no fragments).
  //
  // NOTE on what this does and does NOT prove: the GPU already clips primitives
  // with clip-space w <= 0, so a behind-camera point yields an empty frame in
  // both backends *whether or not* the shader guard is present (verified: the
  // test still passes with the guard removed). So this is a cross-backend
  // *parity* lock for the behind-camera branch, not proof the guard alone culls.
  // The guard's actual presence is pinned with teeth by the generated-shader
  // codegen snapshot (tsl-codegen-snapshot.spec.ts: point / point-pick vertex
  // snapshots contain the `mvPosition.z >= 0 → vec4(0,0,-2,1)` reject). The
  // guard itself is a defensive early-out (skips wasted vertex math; explicit
  // intent; safe at the w≈0 singularity).
  for (const variant of [
    'point-behind',
    'point-pick-behind',
    'line-behind',
    'line-pick-behind',
    'gsplat-behind',
    'gsplat-pick-behind',
  ] as const) {
    test(`${variant}: behind-camera element renders identically (empty) in both backends`, async ({
      page,
    }) => {
      await bootHarness(page);

      const glslPixels = await runGLSL(page, variant);
      const tslResult = await runTSL(page, variant);

      const diff = meanAbsDiff(glslPixels, tslResult.pixels);
      expect(
        diff,
        `${variant} parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\n` +
          `GLSL first 4 pixels:\n${previewPixels(glslPixels)}\n` +
          `TSL first 4 pixels:\n${previewPixels(tslResult.pixels)}`
      ).toBeLessThan(2.0);

      // No fragments emitted, so each frame is a uniform clear colour in both
      // backends (the behind-camera point is culled).
      expect(
        nonUniformPixelCount(glslPixels),
        'GLSL behind-camera frame should be empty (uniform), got a rendered sprite'
      ).toBe(0);
      expect(
        nonUniformPixelCount(tslResult.pixels),
        'TSL behind-camera frame should be empty (uniform), got a rendered sprite'
      ).toBe(0);
    });
  }

  test('point-persp: off-axis point footprint equals the centered one (view-z sizing, B9a)', async ({
    page,
  }) => {
    await bootHarness(page);

    const center = await runGLSL(page, 'point-persp-center');
    const offaxis = await runGLSL(page, 'point-persp-offaxis');
    const centerTSL = await runTSL(page, 'point-persp-center');
    const offaxisTSL = await runTSL(page, 'point-persp-offaxis');

    assertBothRendered(center, centerTSL.pixels, 'point-persp-center');
    assertBothRendered(offaxis, offaxisTSL.pixels, 'point-persp-offaxis');
    expect(
      meanAbsDiffPerCoveredPixel(center, centerTSL.pixels),
      'point-persp-center parity'
    ).toBeLessThan(2.0);
    expect(
      meanAbsDiffPerCoveredPixel(offaxis, offaxisTSL.pixels),
      'point-persp-offaxis parity'
    ).toBeLessThan(2.0);

    // Both points sit at view depth 1; the off-axis one is 26.6° from
    // the view axis. With view-z sizing their footprints are EQUAL;
    // the old Euclidean sizing shrank the off-axis sprite by
    // cos(26.6°) ≈ 0.894 linear (~20% fewer covered pixels).
    const centerCount = nonUniformPixelCount(center);
    const offaxisCount = nonUniformPixelCount(offaxis);
    expect(centerCount).toBeGreaterThan(20);
    const ratio = offaxisCount / centerCount;
    expect(
      ratio,
      `off-axis footprint ${offaxisCount}px vs centered ${centerCount}px — ` +
        'view-z sizing requires equal footprints (Euclidean sizing shrinks off-axis)'
    ).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.1);
  });

  test('point-subpixel: sub-pixel point is energy-compensated, not full-brightness (B9b)', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-subpixel');
    const tslResult = await runTSL(page, 'point-subpixel');

    // The 1.5px sprite floor guarantees rasterization on both backends.
    assertBothRendered(glslPixels, tslResult.pixels, 'point-subpixel');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-subpixel parity'
    ).toBeLessThan(2.0);

    // Raw projected size ≈ 0.96px → sizeScale² = (0.96/1.5)² ≈ 0.41.
    // The sprite center sits exactly on pixel (32,32)'s center where
    // falloff = 1, so the written alpha is deterministically
    // ≈ 0.41·255 ≈ 105. Pre-fix (no compensation) it wrote 255 —
    // indistinguishable from the opaque clear alpha. Scan the 3×3
    // around the center for the MIN alpha (the sprite is the only
    // thing lowering alpha; row order is irrelevant by symmetry).
    let alphaMin = 255;
    for (let y = 31; y <= 33; y++) {
      for (let x = 31; x <= 33; x++) {
        const a = glslPixels[(y * 64 + x) * 4 + 3];
        if (a < alphaMin) alphaMin = a;
      }
    }
    expect(
      alphaMin,
      `sub-pixel point center alpha ${alphaMin} — must be ≈105 (0.41× compensated); ` +
        '255 means the sizeScale² compensation is missing'
    ).toBeLessThan(160);
    expect(alphaMin).toBeGreaterThan(60);
  });

  test('point-near-fade: mid-band near fade renders identically across backends (B9c)', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-near-fade');
    const tslResult = await runTSL(page, 'point-near-fade');

    // Fade ≈ 0.39 at view depth 1 with uNearCull 0.7 — visible but
    // dimmed, and byte-identical across backends (shared helper).
    assertBothRendered(glslPixels, tslResult.pixels, 'point-near-fade');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-near-fade parity'
    ).toBeLessThan(2.0);

    // Clear alpha is opaque (255); the faded sprite LOWERS alpha at
    // the center. smoothstep((1-0.7)/(1.4-0.7)) ≈ 0.39 → center ≈ 100;
    // pre-fix points had NO near fade → 255.
    let centerAlpha = 255;
    for (const [x, y] of [
      [31, 31],
      [32, 31],
      [31, 32],
      [32, 32],
    ]) {
      const a = glslPixels[(y * 64 + x) * 4 + 3];
      if (a < centerAlpha) centerAlpha = a;
    }
    expect(centerAlpha).toBeLessThan(160);
    expect(centerAlpha).toBeGreaterThan(40);
  });

  test('line-ortho-near: in-frustum ortho line inside the nearCull slab RENDERS (B9c bug A)', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-ortho-near');
    const tslResult = await runTSL(page, 'line-ortho-near');

    // View depth 0.15 < uNearCull 0.5 but inside the ortho frustum
    // (near 0.1): pre-fix the ungated bothBehind cull hid this line
    // while a point/gsplat at the same spot drew — under ortho, NDC
    // clipping is the sole cull authority.
    assertBothRendered(glslPixels, tslResult.pixels, 'line-ortho-near');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-ortho-near parity'
    ).toBeLessThan(2.0);
    expect(
      nonUniformPixelCount(glslPixels),
      'ortho near-slab line must render (GLSL)'
    ).toBeGreaterThan(20);
    expect(
      nonUniformPixelCount(tslResult.pixels),
      'ortho near-slab line must render (TSL)'
    ).toBeGreaterThan(20);
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
