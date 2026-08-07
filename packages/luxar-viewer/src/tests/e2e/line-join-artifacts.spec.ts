/**
 * Line-joint artifact measurement (#785 / #790).
 *
 * Renders `test_line_joints.luxar.zarr` — one face-on plane holding every joint
 * topology as a vertically separated band — and measures each band twice from a
 * SINGLE page load, flipping the live `uLineJoin` uniform between the two
 * states. Nothing reloads in between, so camera, DPR, layout and load order are
 * identical by construction; the only variable is the join style.
 *
 * ## Two metrics, because one of them is blind
 *
 * 1. **Local-median outliers.** Per band, compare every lit pixel against its
 *    5x5 local median and count deviations past a threshold. This is the #790
 *    metric: it catches the sharp one-to-two-pixel notch a wedge leaves on the
 *    convex edge of a bend, and the matching bright tick inside it.
 *
 * 2. **Axial flux ripple.** Sum each column of a band (the tube's total
 *    brightness at that x) and measure the ripple of that profile along the
 *    band.
 *
 * The second exists because the first CANNOT see the #780 bead-chain class: a
 * smooth 50% dip spread over a segment is exactly what a local median tracks,
 * so it is invisible to an outlier count while being a 40% brightness loss to
 * the eye. That is not hypothetical — the flux metric is what caught the
 * inverted cap-suppression seed during this work (`straight_1w` p05/p50 0.885,
 * `straight_4w` 0.699) on bands the outlier count called clean.
 *
 * ## The straight bands are the metrics' own null control
 *
 * Collinear quads tile exactly under any camera, and the miter reduces
 * algebraically to `R * perp` at a collinear joint. So both straight bands must
 * read clean in BOTH states, and — the sharper assertion — must be pixel-for-
 * pixel IDENTICAL between them. A metric that moves there is measuring
 * something other than the joint, and a join that moves there has broken its
 * own reduction.
 *
 * Pixels are decoded in-page from a screenshot data URL through a 2D canvas,
 * matching the existing `samplePixelsAt` helper — no image dependency.
 *
 * @module tests/e2e/line-join-artifacts.spec
 */

import { test, expect, type Page } from '@playwright/test';

/**
 * Repository root as served by the E2E static server. Overridable so the spec
 * can run in a multi-worktree checkout, where port 9000 may already be held by
 * a SIBLING worktree's server: `playwright.config.ts` rejects a foreign server
 * via its checkout-identity marker, but a hand-rolled config probing something
 * generic (a `package.json`) will happily reuse one and then 404 on this
 * fixture. Defaults to the same base every other spec hardcodes.
 */
const DATA_BASE = process.env.LUXAR_E2E_DATA_BASE ?? 'http://localhost:9000';
const DATASET = `${DATA_BASE}/packages/luxar-viewer/tests/fixtures/test_line_joints.luxar.zarr`;

/** Band order must match `generate_line_joints_test()` (top of frame → bottom). */
const BANDS = [
  'straight_4w',
  'straight_1w',
  'curve_smooth',
  'zigzag_90',
  'star_hub',
  'free_ends',
] as const;
type Band = (typeof BANDS)[number];

/**
 * Horizontal crop. The control rail and the first-run onboarding popover are
 * DOM overlays that land in an element screenshot, and they merged two bands
 * into one run the first time this was measured. Cropping to the clear middle
 * of the frame removes them from the row scan entirely.
 */
const CROP_X0 = 310;
const CROP_X1 = 1130;

interface BandStats {
  band: string;
  /** Lit pixels deviating > 25/255 from their 5x5 local median. */
  darkOutliers: number;
  brightOutliers: number;
  /** Worst signed deviation from the local median, on the 0-255 scale. */
  worstDark: number;
  worstBright: number;
  /** Axial flux: mean column sum, its coefficient of variation, and the
   *  5th-percentile column against the median one (1.0 = a flat tube). */
  meanFlux: number;
  fluxCV: number;
  fluxP05OverP50: number;
}

/**
 * Runs in the page. Decodes the screenshot, auto-detects each band as a run of
 * lit rows, and computes both metrics per band.
 */
async function analyze(
  page: Page,
  dataUrl: string,
  names: readonly string[]
): Promise<BandStats[]> {
  return page.evaluate(
    async ({ url, bandNames, x0, x1 }) => {
      const img = new Image();
      img.decoding = 'sync';
      await new Promise<void>((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error('screenshot decode failed'));
        img.src = url;
      });
      const W = img.naturalWidth;
      const H = img.naturalHeight;
      const cv = document.createElement('canvas');
      cv.width = W;
      cv.height = H;
      const ctx = cv.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(img, 0, 0);
      const px = ctx.getImageData(0, 0, W, H).data;

      const lum = new Float32Array(W * H);
      for (let i = 0, q = 0; i < lum.length; i++, q += 4) {
        lum[i] = 0.2126 * px[q] + 0.7152 * px[q + 1] + 0.0722 * px[q + 2];
      }

      // Auto-detect bands as maximal runs of rows with any lit pixel in the
      // crop. Runs shorter than 4 rows are rejected as stray AA, not a tube.
      const rowLit = new Uint8Array(H);
      for (let y = 0; y < H; y++) {
        for (let x = x0; x < x1; x++) {
          if (lum[y * W + x] > 18) {
            rowLit[y] = 1;
            break;
          }
        }
      }
      const runs: Array<[number, number]> = [];
      for (let y = 0; y < H; y++) {
        if (!rowLit[y]) continue;
        let end = y;
        while (end + 1 < H && rowLit[end + 1]) end++;
        if (end - y >= 3) runs.push([y, end]);
        y = end;
      }

      return runs.map(([yTop, yBot], k) => {
        let darkOutliers = 0;
        let brightOutliers = 0;
        let worstDark = 0;
        let worstBright = 0;
        const window: number[] = [];
        for (let y = yTop; y <= yBot; y++) {
          for (let x = x0; x < x1; x++) {
            const v = lum[y * W + x];
            // Only measure INSIDE the drawn tube: an unlit background pixel has
            // no cross-profile for a median to track.
            if (v <= 18) continue;
            window.length = 0;
            for (let dy = -2; dy <= 2; dy++) {
              const yy = y + dy;
              if (yy < 0 || yy >= H) continue;
              for (let dx = -2; dx <= 2; dx++) {
                const xx = x + dx;
                if (xx < 0 || xx >= W) continue;
                window.push(lum[yy * W + xx]);
              }
            }
            window.sort((a, b) => a - b);
            const med = window[window.length >> 1];
            const d = v - med;
            if (d < -25) {
              darkOutliers++;
              worstDark = Math.min(worstDark, d);
            } else if (d > 25) {
              brightOutliers++;
              worstBright = Math.max(worstBright, d);
            }
          }
        }

        // Axial flux: total brightness per column across the band.
        const flux: number[] = [];
        for (let x = x0; x < x1; x++) {
          let s = 0;
          for (let y = yTop; y <= yBot; y++) s += lum[y * W + x];
          flux.push(s);
        }
        const lit = flux.filter((f) => f > 40);
        const mean = lit.length ? lit.reduce((a, c) => a + c, 0) / lit.length : 0;
        const sd = lit.length
          ? Math.sqrt(lit.reduce((a, c) => a + (c - mean) ** 2, 0) / lit.length)
          : 0;
        const sorted = lit.slice().sort((a, b) => a - b);
        const p05 = sorted[Math.floor(sorted.length * 0.05)] ?? 0;
        const p50 = sorted[Math.floor(sorted.length * 0.5)] ?? 1;

        return {
          band: bandNames[k] ?? `run_${k}`,
          darkOutliers,
          brightOutliers,
          worstDark: Math.round(worstDark),
          worstBright: Math.round(worstBright),
          meanFlux: +mean.toFixed(1),
          fluxCV: +(sd / (mean || 1)).toFixed(4),
          fluxP05OverP50: +(p05 / (p50 || 1)).toFixed(4),
        };
      });
    },
    { url: dataUrl, bandNames: [...names], x0: CROP_X0, x1: CROP_X1 }
  );
}

/** Set `uLineJoin` on every line material and re-render. Returns how many it hit. */
async function setJoin(page: Page, value: number): Promise<number> {
  const touched = await page.evaluate((v) => {
    const dbg = window.__luxarDebug!;
    let n = 0;
    // `any` here matches the surrounding e2e specs: the debug surface is
    // deliberately untyped, and a structural cast fights THREE's Object3D
    // signature without buying anything.

    (dbg.scene as any).traverse((obj: any) => {
      const raw = obj.material;
      const mats = Array.isArray(raw) ? raw : raw ? [raw] : [];
      for (const m of mats as Array<{ uniforms?: Record<string, { value: unknown }> }>) {
        if (m?.uniforms?.uLineJoin) {
          m.uniforms.uLineJoin.value = v;
          n++;
        }
      }
    });
    dbg.renderOnce?.();
    return n;
  }, value);
  await page.waitForTimeout(500);
  return touched;
}

async function shoot(page: Page): Promise<string> {
  const canvas = await page.$('canvas');
  const buf = await canvas!.screenshot();
  return `data:image/png;base64,${buf.toString('base64')}`;
}

function byBand(rows: BandStats[]): Record<string, BandStats> {
  return Object.fromEntries(rows.map((r) => [r.band, r]));
}

function table(label: string, rows: BandStats[]): string {
  const head = `${label}\n  ${'band'.padEnd(14)}${'dark'.padStart(6)}${'bright'.padStart(8)}${'worstD'.padStart(8)}${'meanFlux'.padStart(10)}${'cv'.padStart(9)}${'p05/p50'.padStart(10)}`;
  const body = rows
    .map(
      (r) =>
        `  ${r.band.padEnd(14)}${String(r.darkOutliers).padStart(6)}${String(r.brightOutliers).padStart(8)}${String(r.worstDark).padStart(8)}${String(r.meanFlux).padStart(10)}${String(r.fluxCV).padStart(9)}${String(r.fluxP05OverP50).padStart(10)}`
    )
    .join('\n');
  return `${head}\n${body}`;
}

test.describe('line join artifacts (#785 / #790)', () => {
  test('the miter closes the joint wedge and leaves straight lines untouched', async ({ page }) => {
    // ?dpr=1 pins the device pixel ratio so band rows land at fixed
    // coordinates regardless of the host display.
    await page.goto(`/?src=${DATASET}&debug&dpr=1`);
    await page.waitForFunction(() => Boolean(window.__luxarDebug?.renderOnce), null, {
      timeout: 60000,
    });
    // The scene streams in; wait for the line materials to exist before the
    // first flip, or `setJoin` would touch nothing and both states would be
    // the same empty frame.
    await page.waitForFunction(
      () => {
        let n = 0;

        (window.__luxarDebug!.scene as any).traverse((o: any) => {
          const raw = o.material;
          const mats = Array.isArray(raw) ? raw : raw ? [raw] : [];
          for (const m of mats as Array<{ uniforms?: Record<string, unknown> }>) {
            if (m?.uniforms?.uLineJoin) n++;
          }
        });
        return n > 0;
      },
      null,
      { timeout: 60000 }
    );
    await page.waitForTimeout(2500);

    // --- Measure none -> miter -> none. The repeated `none` is the harness's
    // own control: if the two disagree, the measurement is untrustworthy and
    // no comparison between them means anything.
    const touchedNone = await setJoin(page, 0);
    const none = await analyze(page, await shoot(page), BANDS);
    const touchedMiter = await setJoin(page, 1);
    const miter = await analyze(page, await shoot(page), BANDS);
    await setJoin(page, 0);
    const none2 = await analyze(page, await shoot(page), BANDS);

    console.log(table('join=none', none));
    console.log(table('join=miter', miter));

    // The uniform must actually exist. On the WebGPU/TSL backend the style is
    // a BUILD-time graph variant, so there is no uniform to flip and this
    // whole A/B would silently compare a frame with itself.
    expect(
      touchedNone,
      'no line material exposed uLineJoin — the join style is a build-time graph variant on the TSL backend, so this live-flip A/B only applies to the WebGL path'
    ).toBeGreaterThan(0);
    expect(touchedMiter).toBe(touchedNone);

    // Band auto-detection must have found every band, or the labels below are
    // attached to the wrong runs.
    expect(
      none.map((r) => r.band),
      'band auto-detection'
    ).toEqual([...BANDS]);
    expect(miter.map((r) => r.band)).toEqual([...BANDS]);

    // --- Control: the two `none` measurements must agree.
    for (const b of BANDS) {
      const a = byBand(none)[b];
      const c = byBand(none2)[b];
      expect(
        Math.abs(a.darkOutliers - c.darkOutliers),
        `${b}: repeated join=none measurement disagreed (${a.darkOutliers} vs ${c.darkOutliers}) — the harness is not reproducible, so no other assertion here is meaningful`
      ).toBeLessThanOrEqual(2);
    }

    const N = byBand(none);
    const M = byBand(miter);

    // --- 1. The wedge is CLOSED on the bent bands. Both are measured at once
    // because the artifact is a pair: an uncovered sector outside the bend and
    // a double-covered lens inside it.
    //
    // Under ADDITIVE blending on black it is the INNER lens that dominates the
    // reading: double coverage sums to roughly twice the tube, a large positive
    // deviation, while the outer wedge merely removes coverage from a region
    // where the tube's own profile is already falling off — which a local
    // median tracks. Measured on this fixture, `none` reads 233 / 726 bright
    // and 0 dark. The dark bound below is therefore a REGRESSION guard rather
    // than the primary signal; the bright one is what actually moves.
    for (const b of ['curve_smooth', 'zigzag_90'] as Band[]) {
      expect(
        N[b].darkOutliers + N[b].brightOutliers,
        `${b}: the fixture must EXHIBIT the artifact under join=none, or this test proves nothing about the fix`
      ).toBeGreaterThan(20);
      expect(
        M[b].brightOutliers,
        `${b}: bright ticks along the concave edge must be gone (none=${N[b].brightOutliers}, miter=${M[b].brightOutliers})`
      ).toBeLessThanOrEqual(2);
      expect(
        M[b].darkOutliers,
        `${b}: no dark ticks along the convex edge (none=${N[b].darkOutliers} worst=${N[b].worstDark}, miter=${M[b].darkOutliers} worst=${M[b].worstDark})`
      ).toBeLessThanOrEqual(2);
    }

    // The miter ADDS coverage at a corner — it fills the sector the two quads
    // left open — so total brightness on a bent band must rise, never fall.
    // Measured: zigzag_90 meanFlux 6885 -> 7487 (+8.7%). A DROP would mean the
    // rotated end edge is cutting into the tube instead of extending it.
    for (const b of ['curve_smooth', 'zigzag_90'] as Band[]) {
      expect(
        M[b].meanFlux,
        `${b}: the miter fills the wedge, so flux must not drop (none=${N[b].meanFlux}, miter=${M[b].meanFlux})`
      ).toBeGreaterThanOrEqual(N[b].meanFlux * 0.995);
    }

    // --- 2. Bands with no degree-2 joint must not move at all. Both are FREE
    // ends: `star_hub`'s nine rays are nine separate nodes, so each is a
    // one-segment polyline whose two ends carry the free-end sentinel (the rays
    // merely coincide in space — nothing joins them, and the degree->=3 hub
    // sentinel is never produced here), and `free_ends` is isolated segments.
    // So this pins that a free end is untouched by the join; a change in either
    // means the partner decode is reaching something it should not.
    for (const b of ['star_hub', 'free_ends'] as Band[]) {
      expect(
        Math.abs(M[b].darkOutliers - N[b].darkOutliers),
        `${b}: has no mitrable joint and must be untouched by the join (none=${N[b].darkOutliers}, miter=${M[b].darkOutliers})`
      ).toBeLessThanOrEqual(2);
      expect(
        Math.abs(M[b].meanFlux - N[b].meanFlux) / (N[b].meanFlux || 1),
        `${b}: total brightness must be unchanged (none=${N[b].meanFlux}, miter=${M[b].meanFlux})`
      ).toBeLessThan(0.02);
    }

    // --- 3. THE ALGEBRAIC IDENTITY. At a collinear joint the miter reduces to
    // `R * perp`, so the straight bands must come out not merely similar but
    // the SAME. This is the strongest statement the harness can make about the
    // miter being exact rather than approximately right, and it is also the
    // metrics' null control — a nonzero reading here means the measurement is
    // picking up something other than the joint.
    for (const b of ['straight_4w', 'straight_1w'] as Band[]) {
      expect(
        N[b].darkOutliers + N[b].brightOutliers,
        `${b}: collinear quads tile exactly, so the metric must read clean even BEFORE the fix`
      ).toBe(0);
      expect(M[b].darkOutliers + M[b].brightOutliers, `${b}: still clean after`).toBe(0);
      expect(
        M[b].meanFlux,
        `${b}: the miter must reduce EXACTLY to R*perp at a collinear joint (none=${N[b].meanFlux}, miter=${M[b].meanFlux})`
      ).toBeCloseTo(N[b].meanFlux, 1);
    }

    // --- 4. AXIAL FLUX, the metric the outlier count is blind to. A dense
    // polyline whose endpoint caps are wrongly applied loses brightness in a
    // smooth periodic dip — exactly what a local median tracks, so the count
    // above stays clean while the tube visibly beads (#780). The straight
    // bands are the ones at risk: their segments are short, so a per-joint
    // dimming error repeats at every segment.
    for (const b of ['straight_4w', 'straight_1w'] as Band[]) {
      for (const [label, s] of [
        ['none', N[b]],
        ['miter', M[b]],
      ] as const) {
        expect(
          s.fluxP05OverP50,
          `${b} (${label}): the tube must be axially FLAT — p05/p50 ${s.fluxP05OverP50} means the dimmest 5% of columns carry that fraction of the median column's brightness. Well under 1.0 is the #780 bead chain, which the outlier count cannot see (dark=${s.darkOutliers}).`
        ).toBeGreaterThan(0.97);
      }
    }
  });
});
