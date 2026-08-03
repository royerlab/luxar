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

/**
 * CONTENT check shared by the camera-plane-crossing line fixtures
 * (`line-crossing`, `line-crossing-reversed`, `line-pick-crossing`,
 * `line-pick-crossing-reversed`).
 *
 * The crossing segment is horizontal in world space (both endpoints at
 * y = 0), so after the vertex-stage near-plane clip its in-front half
 * must project to a band that is SYMMETRIC about the screen centerline:
 * every lit column's vertical centroid sits at y ≈ 31.5 (64-px buffer).
 * Before the clip, the behind-camera endpoint (clip w < 0) wrapped the
 * quad into an external primitive whose visible half drooped toward the
 * bottom of the frame with a razor edge through the profile — the
 * close-zoom "one-sided profile" bug. A lit pixel is any pixel whose
 * summed RGB clears a small threshold; this works for both the visual
 * output and the pick output (nodeId=42 saturates the R channel).
 *
 * Two teeth:
 *  1. the worst per-column centroid offset from the centerline must stay
 *     small — catches the drooping wedge.
 *  2. a MINIMUM number of qualifying columns (≥3 lit pixels each) — the
 *     centroid metric returns 0 when NO column qualifies, so a symmetric
 *     regression that renders only a 1-2px sliver (or nothing) would pass
 *     vacuously. The in-front half projects a streak from its far
 *     endpoint (screen x ≈ 40) rightward to the frame edge — ~20 columns,
 *     each ~6-24 px wide — so ≥8 qualifying columns is a robust floor
 *     that still fails on a sliver/empty frame.
 */
function assertCrossingBandCentered(pixels: number[], label: string): void {
  let worst = 0;
  let qualifyingColumns = 0;
  for (let x = 0; x < 64; x++) {
    let n = 0;
    let sy = 0;
    for (let y = 0; y < 64; y++) {
      const o = (y * 64 + x) * 4;
      if (pixels[o] + pixels[o + 1] + pixels[o + 2] > 10) {
        n++;
        sy += y;
      }
    }
    if (n >= 3) {
      qualifyingColumns++;
      worst = Math.max(worst, Math.abs(sy / n - 31.5));
    }
  }
  expect(
    qualifyingColumns,
    `${label}: too few lit columns — the crossing band collapsed to a sliver/empty frame (vacuous centroid pass)`
  ).toBeGreaterThanOrEqual(8);
  expect(
    worst,
    `${label}: band must stay centered on the projected centerline (drooping wedge = wrapped-quad regression)`
  ).toBeLessThan(2.0);
}

/** Peak max-mode red contribution near the left line endpoint. */
function capEndpointContribution(pixels: number[]): number {
  let peak = 0;
  for (let y = 30; y <= 33; y++) {
    for (let x = 15; x <= 17; x++) {
      peak = Math.max(peak, pixels[(y * 64 + x) * 4]);
    }
  }
  return peak;
}

/**
 * Content stats for the clipped (right-hand) side of the remap fixture.
 * Correct tEff starts at 2/3 there, so it is blue-dominant and thick; using
 * raw t starts from the red/thin source endpoint instead.
 */
function clippedSideStats(pixels: number[]): { maxRows: number; red: number; blue: number } {
  let maxRows = 0;
  let red = 0;
  let blue = 0;
  for (let x = 44; x < 64; x++) {
    let rows = 0;
    for (let y = 0; y < 64; y++) {
      const o = (y * 64 + x) * 4;
      if (pixels[o] + pixels[o + 1] + pixels[o + 2] > 10) {
        rows++;
        red += pixels[o];
        blue += pixels[o + 2];
      }
    }
    maxRows = Math.max(maxRows, rows);
  }
  return { maxRows, red, blue };
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

  // Point volumetric (phase 3): the GLSL side compiles with
  // LUXAR_VOLUMETRIC, the TSL side builds the volumetric output branch
  // from `blendingMode: 'volumetric'`. τ = κ·alpha (κ times the same
  // ray mass the additive branch emits), the S(τ) screening, the
  // physical absorption alpha, AND the per-point RGBA alpha → w(a)
  // optical-depth map (the harness texture carries alpha 0.6 with
  // uHasElementAlpha = 1) must match pixel-for-pixel across backends.
  test('point-volumetric: LUXAR_VOLUMETRIC emission–absorption matches TSL volumetric branch', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-volumetric');
    const tslResult = await runTSL(page, 'point-volumetric');

    assertBothRendered(glslPixels, tslResult.pixels, 'point-volumetric');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-volumetric: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // The alpha channel must carry the PHYSICAL absorption 1 − e^(−τ):
    // sub-saturated and non-zero at the sprite center on both backends
    // (guards a stale alpha-weighted graph that never entered the
    // volumetric branch — and, via non-zero, that the w(a) map with the
    // texture's alpha 0.6 didn't zero the density).
    const centerAlphaGLSL = glslPixels[(32 * 64 + 32) * 4 + 3];
    const centerAlphaTSL = tslResult.pixels[(32 * 64 + 32) * 4 + 3];
    for (const [backend, a] of [
      ['GLSL', centerAlphaGLSL],
      ['TSL', centerAlphaTSL],
    ] as const) {
      expect(
        a,
        `point-volumetric ${backend}: expected sub-saturated absorption alpha at center, got ${a}`
      ).toBeGreaterThan(0);
      expect(a).toBeLessThan(255);
    }
  });

  // COMBINED colormap + volumetric (points): USE_COLORMAP sources the
  // colour from the LUT while LUXAR_VOLUMETRIC maps the per-element
  // alpha through w(a) into τ — both branches read the SAME single
  // texel fetch, so this is the case that breaks if either read
  // displaces the other. Both defines co-compile on GLSL; the TSL side
  // builds with {useColormap: true, blendingMode: 'volumetric'}.
  test('point-volumetric-colormap: combined USE_COLORMAP + LUXAR_VOLUMETRIC matches across backends', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-volumetric-colormap');
    const tslResult = await runTSL(page, 'point-volumetric-colormap');

    assertBothRendered(glslPixels, tslResult.pixels, 'point-volumetric-colormap');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-volumetric-colormap: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // Physical absorption alpha still sub-saturated + non-zero at the
    // centre on both backends (the volumetric branch survived the
    // colormap combination; the LUT path did not zero the density).
    const centerAlphaGLSL = glslPixels[(32 * 64 + 32) * 4 + 3];
    const centerAlphaTSL = tslResult.pixels[(32 * 64 + 32) * 4 + 3];
    for (const [backend, a] of [
      ['GLSL', centerAlphaGLSL],
      ['TSL', centerAlphaTSL],
    ] as const) {
      expect(
        a,
        `point-volumetric-colormap ${backend}: expected sub-saturated absorption alpha at center, got ${a}`
      ).toBeGreaterThan(0);
      expect(a).toBeLessThan(255);
    }
  });

  // Line volumetric: the GLSL side compiles with LUXAR_VOLUMETRIC, the
  // TSL side builds the volumetric output branch from
  // `blendingMode: 'volumetric'`. τ = κ·alpha (κ times the same ray
  // mass the additive branch emits), the S(τ) screening,
  // the physical absorption alpha, AND the per-endpoint texel5.zw
  // alphas (0.6 → 0.9, mixed along t → 0.75 at the centre pixel) →
  // w(a) optical-depth map (uHasElementAlpha = 1) must match
  // pixel-for-pixel across backends. τ is kept sub-saturated on the
  // line body so S(τ) and volAlpha are non-trivial and any divergence
  // is detectable.
  test('line-volumetric: LUXAR_VOLUMETRIC emission–absorption matches TSL volumetric branch', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-volumetric');
    const tslResult = await runTSL(page, 'line-volumetric');

    assertBothRendered(glslPixels, tslResult.pixels, 'line-volumetric');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-volumetric: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // The alpha channel must carry the PHYSICAL absorption 1 − e^(−τ):
    // sub-saturated and non-zero at the line-body centre on both
    // backends (guards a stale alpha-weighted graph that never entered
    // the volumetric branch — and, via non-zero, that the w(a) map
    // with the interpolated texel5.zw alpha 0.75 didn't zero the
    // density).
    const centerAlphaGLSL = glslPixels[(32 * 64 + 32) * 4 + 3];
    const centerAlphaTSL = tslResult.pixels[(32 * 64 + 32) * 4 + 3];
    for (const [backend, a] of [
      ['GLSL', centerAlphaGLSL],
      ['TSL', centerAlphaTSL],
    ] as const) {
      expect(
        a,
        `line-volumetric ${backend}: expected sub-saturated absorption alpha at center, got ${a}`
      ).toBeGreaterThan(0);
      expect(a).toBeLessThan(255);
    }
  });

  // COMBINED colormap + volumetric (lines): USE_COLORMAP sources the
  // colour from the LUT while LUXAR_VOLUMETRIC maps the per-element
  // alpha through w(a) into τ — both branches read the SAME single
  // texel fetch, so this is the case that breaks if either read
  // displaces the other. Both defines co-compile on GLSL; the TSL side
  // builds with {useColormap: true, blendingMode: 'volumetric'}.
  test('line-volumetric-colormap: combined USE_COLORMAP + LUXAR_VOLUMETRIC matches across backends', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-volumetric-colormap');
    const tslResult = await runTSL(page, 'line-volumetric-colormap');

    assertBothRendered(glslPixels, tslResult.pixels, 'line-volumetric-colormap');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-volumetric-colormap: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // Physical absorption alpha still sub-saturated + non-zero at the
    // centre on both backends (the volumetric branch survived the
    // colormap combination; the LUT path did not zero the density).
    const centerAlphaGLSL = glslPixels[(32 * 64 + 32) * 4 + 3];
    const centerAlphaTSL = tslResult.pixels[(32 * 64 + 32) * 4 + 3];
    for (const [backend, a] of [
      ['GLSL', centerAlphaGLSL],
      ['TSL', centerAlphaTSL],
    ] as const) {
      expect(
        a,
        `line-volumetric-colormap ${backend}: expected sub-saturated absorption alpha at center, got ${a}`
      ).toBeGreaterThan(0);
      expect(a).toBeLessThan(255);
    }
  });

  // Max-mode premultiplied RGB-contribution output: the GLSL side
  // compiles with LUXAR_MAX_RGB_CONTRIBUTION, the TSL side is built
  // with `blendingMode: 'max'`. The fragment must emit rgb·alpha (the
  // contribution MaxEquation compares) IDENTICALLY on both backends —
  // a premultiply divergence here would make max-mode brightness differ
  // between the GLSL and TSL renderers in production.
  for (const variant of ['point-max', 'line-max'] as const) {
    test(`${variant}: max-mode RGB-contribution premultiply parity`, async ({ page }) => {
      await bootHarness(page);

      const glslPixels = await runGLSL(page, variant);
      const tslResult = await runTSL(page, variant);

      assertBothRendered(glslPixels, tslResult.pixels, variant);
      expect(
        meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
        `${variant}: per-covered-pixel parity (footprint-invariant)`
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
        `${variant} parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.\nSamples:\n${samples}`
      ).toBeLessThan(2.0);
    });
  }

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

  test('line cap suppression renders fractional values instead of quantising to a flag', async ({
    page,
  }) => {
    await bootHarness(page);

    for (const backend of ['GLSL', 'TSL'] as const) {
      const render = async (name: string): Promise<number[]> =>
        backend === 'GLSL' ? runGLSL(page, name) : (await runTSL(page, name)).pixels;
      const zeroPixels = await render('line-cap-zero');
      const fractionalPixels = await render('line-cap-fractional');
      const fullPixels = await render('line-cap-full');
      const zero = capEndpointContribution(zeroPixels);
      const fractional = capEndpointContribution(fractionalPixels);
      const full = capEndpointContribution(fullPixels);

      expect(
        fractional,
        `${backend}: fractional cap must be visibly brighter than s=0 (got ${zero}, ${fractional}, ${full})`
      ).toBeGreaterThan(zero + 15);
      expect(
        full,
        `${backend}: s=1 cap must be visibly brighter than s=0.5 (got ${zero}, ${fractional}, ${full})`
      ).toBeGreaterThan(fractional + 15);
      expect(
        fractional,
        `${backend}: s=0.5 endpoint should stay near the midpoint of s=0 and s=1`
      ).toBeCloseTo((zero + full) * 0.5, -1);
    }
  });

  // Multi-row texture-orientation parity (one per geometry type): the
  // element renders from STORAGE SLOT 1 of a 2-row data texture whose
  // row 0 is a green decoy parked away from the viewport centre. A
  // Y-flip mismatch between the GLSL texelFetch and the TSL textureLoad
  // codegen (three wraps WebGL-fallback loads in `height − y − 1`)
  // would sample the decoy on one backend only → pixel parity fails.
  // The centre-pixel content assertion keeps the test non-vacuous: if
  // BOTH backends consistently sampled the wrong row, the centred real
  // element would be missing and the decoy's green would dominate.
  for (const variant of ['point-multirow', 'line-multirow', 'gsplat-multirow'] as const) {
    test(`${variant}: 2-row texture — both backends resolve the same storage row`, async ({
      page,
    }) => {
      await bootHarness(page);
      const glslPixels = await runGLSL(page, variant);
      const tslResult = await runTSL(page, variant);

      assertBothRendered(glslPixels, tslResult.pixels, variant);
      expect(
        meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
        `${variant}: per-covered-pixel parity (footprint-invariant)`
      ).toBeLessThan(2.0);

      // Non-vacuousness: the REAL element (slot 1) is centred, so the
      // centre pixel must be lit on both backends and must NOT be the
      // decoy's pure green (real color is red-dominant 1.0/0.5/0.25).
      for (const [label, px] of [
        ['GLSL', glslPixels],
        ['TSL', tslResult.pixels],
      ] as const) {
        const o = (32 * 64 + 32) * 4;
        const [r, g] = [px[o], px[o + 1]];
        expect(
          r,
          `${variant} ${label}: centre pixel unlit — wrong storage row sampled`
        ).toBeGreaterThan(10);
        expect(
          r,
          `${variant} ${label}: centre pixel is the green DECOY — storage row mismatch`
        ).toBeGreaterThan(g);
      }
    });
  }

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

  test('gsplat-thin-cov: 2D-covariance dilation on a near-degenerate splat matches across backends', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-thin-cov');
    const tslResult = await runTSL(page, 'gsplat-thin-cov');

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-thin-cov');
    // GLSL and TSL must apply the identical Σ_2D diagonal dilation.
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-thin-cov: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);
    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `GSplat-thin-cov parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.`
    ).toBeLessThan(3.0);
    // assertBothRendered above already guards that the dilated near-degenerate
    // splat produces a non-empty, non-uniform footprint on BOTH backends — so a
    // dilation regression that collapsed the sliver on one side (or produced NaN
    // via a near-singular Cholesky) would fail here, not pass vacuously.
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

  test('gsplat-volumetric: LUXAR_VOLUMETRIC emission–absorption matches TSL volumetric branch', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-volumetric');
    const tslResult = await runTSL(page, 'gsplat-volumetric');

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-volumetric');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-volumetric: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // The alpha channel must carry the PHYSICAL absorption 1 − e^(−τ):
    // with κ=1.5, opacity=0.7 the center τ is mid-range, so alpha is
    // strictly sub-saturated and non-zero on both backends (guards a
    // stale alpha=1 graph — exactly what the pre-fix TSL rebuild
    // predicate produced on additive→volumetric switches).
    const centerAlphaGLSL = glslPixels[(32 * 64 + 32) * 4 + 3];
    const centerAlphaTSL = tslResult.pixels[(32 * 64 + 32) * 4 + 3];
    for (const [backend, a] of [
      ['GLSL', centerAlphaGLSL],
      ['TSL', centerAlphaTSL],
    ] as const) {
      expect(
        a,
        `gsplat-volumetric ${backend}: expected sub-saturated absorption alpha at center, got ${a}`
      ).toBeGreaterThan(0);
      expect(a).toBeLessThan(255);
    }

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    // Same looser tolerance as gsplat-normal-premult — this variant also
    // exercises the sum-projection ray-integral path.
    expect(
      diff,
      `GSplat-volumetric parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.`
    ).toBeLessThan(3.0);
  });

  test('gsplat-volumetric-rgba: per-splat alpha → optical-depth fold matches across backends', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'gsplat-volumetric-rgba');
    const tslResult = await runTSL(page, 'gsplat-volumetric-rgba');

    assertBothRendered(glslPixels, tslResult.pixels, 'gsplat-volumetric-rgba');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'gsplat-volumetric-rgba: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // With uHasElementAlpha=1 and texel3.y = 0.5 the density scalar is
    // scaled by w = −ln(1 − 0.5) ≈ 0.693 on BOTH backends — the
    // α → optical-depth branch no other gsplat variant executes (they
    // all leave the gate at its 0 default). The center alpha must stay
    // sub-saturated absorption, exactly like the α = 1-identity twin.
    const centerAlphaGLSL = glslPixels[(32 * 64 + 32) * 4 + 3];
    const centerAlphaTSL = tslResult.pixels[(32 * 64 + 32) * 4 + 3];
    for (const [backend, a] of [
      ['GLSL', centerAlphaGLSL],
      ['TSL', centerAlphaTSL],
    ] as const) {
      expect(
        a,
        `gsplat-volumetric-rgba ${backend}: expected sub-saturated absorption alpha at center, got ${a}`
      ).toBeGreaterThan(0);
      expect(a).toBeLessThan(255);
    }

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    // Same looser tolerance as gsplat-volumetric (sum-projection
    // ray-integral path).
    expect(
      diff,
      `GSplat-volumetric-rgba parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.`
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

  // Surface-pick depth behaviour (S2, front-most-wins for normal-mode
  // gsplats). The harness scene is two overlapping splats along the view
  // axis: instance 0 NEARER but DIMMER (amplitude 0.3), instance 1
  // FARTHER but BRIGHTER (1.0), footprints coinciding at the
  // viewport-centre probe pixel (32,32). The pick buffer encodes
  // (nodeId, elementIdLow16, brightness, elementIdHigh16): in the RGBA8 readback nodeId 42
  // clamps to R=255 (coverage marker), elementId quantises to G=0
  // (instance 0) or G=255 (instance 1), and brightness lands in B
  // (≈77 for the dim splat, 255 for the bright one).
  //   - uSurfaceDepth=1 (surface / 'normal' mode): real projected depth
  //     → the NEARER, DIMMER splat must win (elementId 0).
  //   - uSurfaceDepth=0 (commutative modes): brightness-as-depth
  //     → the BRIGHTER, FARTHER splat must win (elementId 1).
  // Asserted on BOTH backends, plus the usual cross-backend parity.
  for (const { variant, wantNear } of [
    { variant: 'gsplat-pick-surface', wantNear: true },
    { variant: 'gsplat-pick-surface-off', wantNear: false },
  ] as const) {
    test(`${variant}: ${
      wantNear ? 'front-most (dimmer, near)' : 'brightest (farther)'
    } splat wins the overlap pixel on both backends`, async ({ page }) => {
      await bootHarness(page);

      const glslPixels = await runGLSL(page, variant);
      const tslResult = await runTSL(page, variant);

      assertBothRendered(glslPixels, tslResult.pixels, variant);
      expect(
        meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
        `${variant}: per-covered-pixel parity (footprint-invariant)`
      ).toBeLessThan(2.0);

      for (const [backend, pixels] of [
        ['GLSL', glslPixels],
        ['TSL', tslResult.pixels],
      ] as const) {
        const o = (32 * 64 + 32) * 4;
        const [r, g, b] = [pixels[o], pixels[o + 1], pixels[o + 2]];
        // Coverage marker: nodeId 42 clamps to 255 in the RGBA8 readback.
        expect(
          r,
          `${variant} (${backend}): probe pixel (32,32) not covered by any splat (r=${r})`
        ).toBe(255);
        if (wantNear) {
          // elementId 0 = the near, dim splat; its brightness 0.3 ≈ 77.
          expect(
            g,
            `${variant} (${backend}): expected NEAR splat (elementId 0) to win, got g=${g}`
          ).toBeLessThan(128);
          expect(
            b,
            `${variant} (${backend}): winner should carry the DIM brightness (~77), got b=${b}`
          ).toBeGreaterThan(40);
          expect(b).toBeLessThan(120);
        } else {
          // elementId 1 = the far, bright splat; its brightness 1.0 → 255.
          expect(
            g,
            `${variant} (${backend}): expected FAR bright splat (elementId 1) to win, got g=${g}`
          ).toBeGreaterThan(128);
          expect(
            b,
            `${variant} (${backend}): winner should carry FULL brightness (255), got b=${b}`
          ).toBeGreaterThan(200);
        }
      }
    });
  }

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

  test('point-sorted-permuted: aSortedIndex permutation + multi-row texel fetch parity', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'point-sorted-permuted');
    const tslResult = await runTSL(page, 'point-sorted-permuted');

    assertBothRendered(glslPixels, tslResult.pixels, 'point-sorted-permuted');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'point-sorted-permuted: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // All four permuted slots must land — one distinct-color sprite per
    // screen quadrant (world ±0.5 → pixels 16/48; the quadrant set is
    // symmetric, so probe COVERAGE per quadrant regardless of readback
    // row order; the parity assertions above pin the per-pixel colors).
    // A missing quadrant means the aSortedIndex → texel indirection
    // dropped or aliased a storage slot (e.g. a broken base / W row
    // computation reading row 0 for every point).
    for (const [x, y] of [
      [16, 16],
      [48, 16],
      [16, 48],
      [48, 48],
    ] as const) {
      for (const [backend, px] of [
        ['GLSL', glslPixels],
        ['TSL', tslResult.pixels],
      ] as const) {
        const o = (y * 64 + x) * 4;
        const covered =
          px[o] !== px[0] || px[o + 1] !== px[1] || px[o + 2] !== px[2] || px[o + 3] !== px[3];
        expect(
          covered,
          `point-sorted-permuted ${backend}: no sprite at (${x},${y}) — ` +
            'the sorted-index indirection dropped/aliased a storage slot'
        ).toBe(true);
      }
    }

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `point-sorted-permuted parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.`
    ).toBeLessThan(2.0);
  });

  test('line-sorted-permuted: aSortedIndex permutation + multi-row texel fetch parity', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-sorted-permuted');
    const tslResult = await runTSL(page, 'line-sorted-permuted');

    assertBothRendered(glslPixels, tslResult.pixels, 'line-sorted-permuted');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-sorted-permuted: per-covered-pixel parity (footprint-invariant)'
    ).toBeLessThan(2.0);

    // All four permuted slots must land — one distinct-color segment per
    // screen quadrant (midpoints at world ±0.5 → pixels 16/48; the
    // quadrant set is symmetric, so probe COVERAGE per quadrant
    // regardless of readback row order; the parity assertions above pin
    // the per-pixel colors). A missing quadrant means the
    // aSortedIndex → texel indirection dropped or aliased a storage
    // slot (e.g. a broken base / W row computation reading row 0 for
    // every segment).
    for (const [x, y] of [
      [16, 16],
      [48, 16],
      [16, 48],
      [48, 48],
    ] as const) {
      for (const [backend, px] of [
        ['GLSL', glslPixels],
        ['TSL', tslResult.pixels],
      ] as const) {
        const o = (y * 64 + x) * 4;
        const covered =
          px[o] !== px[0] || px[o + 1] !== px[1] || px[o + 2] !== px[2] || px[o + 3] !== px[3];
        expect(
          covered,
          `line-sorted-permuted ${backend}: no segment at (${x},${y}) — ` +
            'the sorted-index indirection dropped/aliased a storage slot'
        ).toBe(true);
      }
    }

    const diff = meanAbsDiff(glslPixels, tslResult.pixels);
    expect(
      diff,
      `line-sorted-permuted parity: mean abs diff ${diff.toFixed(2)} on 0-255 scale.`
    ).toBeLessThan(2.0);
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

  for (const variant of ['line-on-near-plane', 'line-pick-on-near-plane'] as const) {
    test(`${variant}: endpoint exactly at nearCull stays finite and renders`, async ({ page }) => {
      await bootHarness(page);
      const glslPixels = await runGLSL(page, variant);
      const tslResult = await runTSL(page, variant);

      assertBothRendered(glslPixels, tslResult.pixels, variant);
      expect(
        meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
        `${variant} parity`
      ).toBeLessThan(2.0);
      expect(
        nonUniformPixelCount(glslPixels),
        `${variant} GLSL must not cull the boundary`
      ).toBeGreaterThan(20);
      expect(
        nonUniformPixelCount(tslResult.pixels),
        `${variant} TSL must not cull the boundary`
      ).toBeGreaterThan(20);
    });
  }

  test('line-crossing-remap: clipped endpoint uses remapped color/width/sharpness', async ({
    page,
  }) => {
    await bootHarness(page);
    const glslPixels = await runGLSL(page, 'line-crossing-remap');
    const tslResult = await runTSL(page, 'line-crossing-remap');

    assertBothRendered(glslPixels, tslResult.pixels, 'line-crossing-remap');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-crossing-remap parity'
    ).toBeLessThan(2.0);
    for (const [backend, pixels] of [
      ['GLSL', glslPixels],
      ['TSL', tslResult.pixels],
    ] as const) {
      const stats = clippedSideStats(pixels);
      expect(
        stats.blue,
        `${backend}: clipped side must start from tEff=2/3 (blue-dominant), not raw red t=0; ${JSON.stringify(stats)}`
      ).toBeGreaterThan(stats.red * 1.25);
      expect(
        stats.maxRows,
        `${backend}: remapped near-side width must stay visibly thick; ${JSON.stringify(stats)}`
      ).toBeGreaterThanOrEqual(6);
    }
  });

  test('line-pick-crossing-remap: picking footprint uses remapped width/sharpness', async ({
    page,
  }) => {
    await bootHarness(page);
    const glslPixels = await runGLSL(page, 'line-pick-crossing-remap');
    const tslResult = await runTSL(page, 'line-pick-crossing-remap');

    assertBothRendered(glslPixels, tslResult.pixels, 'line-pick-crossing-remap');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-pick-crossing-remap parity'
    ).toBeLessThan(2.0);
    expect(
      clippedSideStats(glslPixels).maxRows,
      'pick GLSL remapped near-side width'
    ).toBeGreaterThanOrEqual(6);
    expect(
      clippedSideStats(tslResult.pixels).maxRows,
      'pick TSL remapped near-side width'
    ).toBeGreaterThanOrEqual(6);
  });

  test('line-crossing: camera-plane-crossing segment clips at nearCull — no wrapped-quad band', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-crossing');
    const tslResult = await runTSL(page, 'line-crossing');

    // The in-front part of the segment must render on both backends…
    assertBothRendered(glslPixels, tslResult.pixels, 'line-crossing');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-crossing parity'
    ).toBeLessThan(2.0);

    // …and the CONTENT assertion that fails pre-fix: the horizontal
    // segment's in-front half must render as a centered band on BOTH
    // backends (a drooping wedge = the wrapped-quad regression). See
    // `assertCrossingBandCentered`. This fixture exercises the tA clip
    // branch (start behind the camera).
    assertCrossingBandCentered(glslPixels, 'line-crossing GLSL');
    assertCrossingBandCentered(tslResult.pixels, 'line-crossing TSL');
  });

  test('line-crossing-reversed: end-behind (tB) clip branch keeps the band centered', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-crossing-reversed');
    const tslResult = await runTSL(page, 'line-crossing-reversed');

    // Same physical segment as `line-crossing` with the endpoints
    // swapped, so it drives the DISTINCT tB clip branch (end behind,
    // start in front) while the footprint is identical — reuse the same
    // parity + content assertions. A sign/ordering slip in tB on any
    // backend droops the band off the centerline.
    assertBothRendered(glslPixels, tslResult.pixels, 'line-crossing-reversed');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-crossing-reversed parity'
    ).toBeLessThan(2.0);
    assertCrossingBandCentered(glslPixels, 'line-crossing-reversed GLSL');
    assertCrossingBandCentered(tslResult.pixels, 'line-crossing-reversed TSL');
  });

  test('line-pick-crossing: picking near-plane segment clip keeps the band centered', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-pick-crossing');
    const tslResult = await runTSL(page, 'line-pick-crossing');

    // The ONLY fixture that reaches the PICKING near-plane segment clip
    // (`line-pick` is ortho, `line-pick-behind` is culled before the
    // clip). A pure GLSL↔TSL parity check is insufficient — both twins
    // could droop identically — so the content assertion is applied
    // per-backend: deleting the picking clip on EITHER backend droops the
    // wrapped-quad wedge and fails that side. Lit pixels are detected via
    // the summed RGB (nodeId=42 saturates the R channel of covered
    // pixels), identical to the visual crossing tests.
    assertBothRendered(glslPixels, tslResult.pixels, 'line-pick-crossing');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-pick-crossing parity'
    ).toBeLessThan(2.0);
    assertCrossingBandCentered(glslPixels, 'line-pick-crossing GLSL');
    assertCrossingBandCentered(tslResult.pixels, 'line-pick-crossing TSL');
  });

  test('line-pick-crossing-reversed: picking end-behind (tB) clip branch keeps the band centered', async ({
    page,
  }) => {
    await bootHarness(page);

    const glslPixels = await runGLSL(page, 'line-pick-crossing-reversed');
    const tslResult = await runTSL(page, 'line-pick-crossing-reversed');

    // Picking twin of `line-crossing-reversed`: same physical segment as
    // `line-pick-crossing` with the endpoints swapped, so it drives the
    // picking shaders' DISTINCT tB clip branch (end behind, start in
    // front) while the footprint is identical. Like `line-pick-crossing`
    // a pure parity check is insufficient — both twins could droop
    // identically — so the content assertion is applied per-backend:
    // deleting the picking tB clip on EITHER backend droops the
    // wrapped-quad wedge off the centerline and fails that side.
    assertBothRendered(glslPixels, tslResult.pixels, 'line-pick-crossing-reversed');
    expect(
      meanAbsDiffPerCoveredPixel(glslPixels, tslResult.pixels),
      'line-pick-crossing-reversed parity'
    ).toBeLessThan(2.0);
    assertCrossingBandCentered(glslPixels, 'line-pick-crossing-reversed GLSL');
    assertCrossingBandCentered(tslResult.pixels, 'line-pick-crossing-reversed TSL');
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
