/**
 * Render-on-change audit (#2944 A2).
 *
 * The animation loop renders a tick only when something changed since the
 * last rendered frame. That is safe exactly when every source of a drawn-state
 * change reports itself — a wake, a `requestRender`, a per-frame callback
 * returning `true`, or a camera / drawing-buffer change. `?renderAudit`
 * (debug only) checks it on the real pipeline: the loop renders EVERY tick,
 * and on each tick the scheduler would have skipped it compares a 64x64
 * readback against the last frame it really rendered. A difference is a
 * change nobody reported, counted in the perf counter `render.missedDirty`.
 *
 * Each scenario below keeps the loop ticking for its whole duration (a
 * continuous no-op per-frame callback), so every tick of the action AND of
 * its settling tail is audited — not just the ones the idle timer happens to
 * leave running — and then asserts no missed dirty, plus that the audit did
 * compare frames (a vacuous zero proves nothing).
 *
 * Runs on both renderer surfaces (`?renderer=webgpu` falls back to the
 * WebGPURenderer's WebGL2 backend where the browser has no adapter, which is
 * still the TSL material path).
 *
 * Fixtures: `pnpm test:generate-fixtures`.
 */

import type { Page } from '@playwright/test';
import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

const FIXTURES = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';
const POINTS = `${FIXTURES}/test_labelled_points.luxar.zarr`;
const FOUR_D = `${FIXTURES}/test_4d.luxar.zarr`;
const LOD = `${FIXTURES}/test_lod_group.luxar.zarr`;
/** Several `layer=True` point layers (the Layers panel has rows to drive). */
const LAYERS = `${FIXTURES}/test_points_blending_modes.luxar.zarr`;

/** Picking settle window (hover-tooltip.spec.ts); a hover pick lands after it. */
const HOVER_SETTLE_MS = 150;

const BACKENDS = ['webgl', 'webgpu'] as const;

interface AuditCounters {
  missedDirty: number;
  compares: number;
  renders: number;
  ticks: number;
  /** Every `render.*` counter, for the failure message. */
  render: Record<string, number>;
}

async function readCounters(page: Page): Promise<AuditCounters> {
  return page.evaluate(() => {
    const counters = (
      window as unknown as {
        __luxarDebug: { getPerf: () => { counters: Record<string, number> } };
      }
    ).__luxarDebug.getPerf().counters;
    return {
      missedDirty: counters['render.missedDirty'] ?? 0,
      compares: counters['render.auditCompares'] ?? 0,
      renders: counters['render.count'] ?? 0,
      ticks: counters['render.ticks'] ?? 0,
      render: Object.fromEntries(
        Object.entries(counters).filter(([k, v]) => k.startsWith('render.') && v !== 0)
      ),
    };
  });
}

/** Keep the loop ticking (every tick audited) until {@link stopTicking}. */
async function keepTicking(page: Page): Promise<void> {
  await page.evaluate(() => {
    const debug = (
      window as unknown as {
        __luxarDebug: {
          animationController: {
            addPerFrameCallback: (id: string, cb: () => void, o: { continuous: boolean }) => void;
            startAnimation: () => void;
          };
          resetPerfCounters: () => void;
        };
      }
    ).__luxarDebug;
    debug.animationController.addPerFrameCallback('render-audit-keepalive', () => {}, {
      continuous: true,
    });
    debug.animationController.startAnimation();
  });
}

async function stopTicking(page: Page): Promise<void> {
  await page.evaluate(() => {
    (
      window as unknown as {
        __luxarDebug: { animationController: { removePerFrameCallback: (id: string) => void } };
      }
    ).__luxarDebug.animationController.removePerFrameCallback('render-audit-keepalive');
  });
}

async function resetCounters(page: Page): Promise<void> {
  await page.evaluate(() =>
    (
      window as unknown as { __luxarDebug: { resetPerfCounters: () => void } }
    ).__luxarDebug.resetPerfCounters()
  );
}

/** Wait until nothing is loading (the audit then compares settled frames too). */
async function waitSettled(page: Page, timeout = 30000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (
        window as unknown as { __luxarDebug?: { getPerf?: () => { isSettled: boolean | null } } }
      ).__luxarDebug;
      return debug?.getPerf?.().isSettled === true;
    },
    null,
    { timeout }
  );
}

/** Audit comparisons required after the tail before a scenario is judged. */
const SETTLED_COMPARES = 10;

/**
 * Wait until the audit has compared `SETTLED_COMPARES` would-be-skipped ticks
 * since `from`. A tick is only audited once NOTHING asks for a render, and the
 * orbit damping tail can keep asking for a couple of seconds after a fast drag
 * (its change test is exact, so sub-pixel motion still reports a camera
 * change — and rightly renders), so a fixed tail can end before the first
 * audited tick. A scene that never settles fails the scenario: that was a real
 * bug (a ULP limit cycle in the damped orbit rotation kept the loop rendering
 * forever after ~1 in 8 drags; fixed in `luxar-orbit-controls/update.ts`).
 */
async function waitForAuditedTicks(page: Page, from: number, timeout: number): Promise<boolean> {
  try {
    await page.waitForFunction(
      (target) => {
        const counters = (
          window as unknown as {
            __luxarDebug: { getPerf: () => { counters: Record<string, number> } };
          }
        ).__luxarDebug.getPerf().counters;
        return (counters['render.auditCompares'] ?? 0) >= target;
      },
      from + SETTLED_COMPARES,
      { timeout }
    );
    return true;
  } catch (err) {
    if (err instanceof Error && /Timeout .*exceeded/.test(err.message)) return false;
    throw err;
  }
}

/**
 * Run `action` with the loop kept ticking, let it settle for `tailMs`, then
 * until enough settled ticks have been audited, and assert no missed dirty.
 */
async function audited(
  page: Page,
  label: string,
  action: () => Promise<void>,
  tailMs = 1500
): Promise<void> {
  await resetCounters(page);
  await keepTicking(page);
  try {
    await action();
    await page.waitForTimeout(tailMs);
    const settled = await waitForAuditedTicks(page, (await readCounters(page)).compares, 15_000);
    expect(
      settled,
      `${label}: the scene never settled (every tick rendered) ${JSON.stringify(await readCounters(page))}`
    ).toBe(true);
  } finally {
    await stopTicking(page);
  }
  const counters = await readCounters(page);
  expect(
    counters.compares,
    `${label}: the audit compared no frame ${JSON.stringify(counters.render)}`
  ).toBeGreaterThan(0);
  expect(counters.missedDirty, `${label}: ${JSON.stringify(counters)}`).toBe(0);
}

async function canvasBox(page: Page): Promise<{ x: number; y: number; w: number; h: number }> {
  const box = await page.locator('canvas').first().boundingBox();
  if (!box) throw new Error('no canvas');
  return { x: box.x, y: box.y, w: box.width, h: box.height };
}

/** First data (non-group) layer; the Layers panel builds its model when opened. */
async function firstLayerPath(page: Page): Promise<string> {
  return page.evaluate(() => {
    const app = (
      window as unknown as {
        __luxarDebug: {
          app: {
            layersPanel: { show: () => void };
            getLayers: () => Array<{ path: string; type?: string }>;
          };
        };
      }
    ).__luxarDebug.app;
    app.layersPanel.show();
    const layer = app.getLayers().find((l) => l.type !== 'group');
    if (!layer) throw new Error('no data layer');
    return layer.path;
  });
}

async function setLayer(page: Page, path: string, patch: Record<string, unknown>): Promise<void> {
  await page.evaluate(
    ([p, change]) => {
      (
        window as unknown as {
          __luxarDebug: { app: { setLayer: (path: string, patch: unknown) => void } };
        }
      ).__luxarDebug.app.setLayer(p as string, change);
    },
    [path, patch] as const
  );
}

for (const backend of BACKENDS) {
  test.describe(`render-on-change audit (${backend})`, () => {
    test('load, camera drag with damping, hover', async ({ page }) => {
      test.setTimeout(120_000);
      await page.goto(`/?src=${POINTS}&debug&renderAudit&renderer=${backend}`);
      await waitForLuxarReady(page);
      await waitForPointsLoaded(page);

      // The load itself: every tick from the first commit to settled was
      // audited (the loop ran through it); a miss there is counted too.
      await waitSettled(page);
      const load = await readCounters(page);
      expect(load.compares, 'load: the audit compared no frame').toBeGreaterThan(0);
      expect(load.missedDirty, `load: ${JSON.stringify(load)}`).toBe(0);

      const box = await canvasBox(page);
      const cx = box.x + box.w / 2;
      const cy = box.y + box.h / 2;

      await audited(page, 'camera drag with damping', async () => {
        await page.mouse.move(cx, cy);
        await page.mouse.down();
        for (let i = 1; i <= 8; i++) await page.mouse.move(cx + i * 12, cy + i * 4);
        await page.mouse.up();
      });

      await audited(page, 'hover', async () => {
        await page.mouse.move(cx, cy);
        await page.waitForTimeout(HOVER_SETTLE_MS + 200);
        await page.mouse.move(cx + 40, cy - 30);
        await page.waitForTimeout(HOVER_SETTLE_MS + 200);
      });
    });

    test('layer visibility toggle, appearance change, window resize', async ({ page }) => {
      test.setTimeout(120_000);
      await page.goto(`/?src=${LAYERS}&debug&renderAudit&renderer=${backend}`);
      await waitForLuxarReady(page);
      await waitForPointsLoaded(page);
      await waitSettled(page);

      const layer = await firstLayerPath(page);
      await audited(page, 'layer visibility toggle', async () => {
        await setLayer(page, layer, { visible: false });
        await page.waitForTimeout(300);
        await setLayer(page, layer, { visible: true });
      });

      await audited(page, 'colormap / appearance change', async () => {
        await setLayer(page, layer, { opacity: 0.5 });
        await page.waitForTimeout(300);
        await setLayer(page, layer, { gamma: 2 });
        await page.waitForTimeout(300);
        await setLayer(page, layer, { colormap: 'viridis' });
        await page.waitForTimeout(300);
        await setLayer(page, layer, { opacity: 1, gamma: 1 });
      });

      await audited(page, 'window resize', async () => {
        const size = page.viewportSize() ?? { width: 1280, height: 720 };
        await page.setViewportSize({ width: size.width - 200, height: size.height - 100 });
        await page.waitForTimeout(400);
        await page.setViewportSize(size);
      });
    });

    test('hidden-dimension playback', async ({ page }) => {
      test.setTimeout(90_000);
      await page.goto(`/?src=${FOUR_D}&debug&renderAudit&renderer=${backend}`);
      await waitForLuxarReady(page);
      await waitSettled(page);

      await audited(
        page,
        'hidden-dimension playback',
        async () => {
          const dim = await page.evaluate(() => {
            const debug = (
              window as unknown as {
                __luxarDebug: {
                  sceneDimsManager: { getDims: () => { ndim: number; displayed: number[] } };
                  inputHandler: {
                    animationManager: {
                      play: (d: number, o: { targetFPS: number; loopMode: string }) => boolean;
                    };
                  };
                };
              }
            ).__luxarDebug;
            const dims = debug.sceneDimsManager.getDims();
            const hidden = Array.from({ length: dims.ndim }, (_, i) => i).find(
              (i) => !dims.displayed.includes(i)
            );
            if (hidden === undefined) throw new Error('fixture has no hidden dimension');
            debug.inputHandler.animationManager.play(hidden, { targetFPS: 10, loopMode: 'loop' });
            return hidden;
          });
          await page.waitForTimeout(3000);
          await page.evaluate((d) => {
            (
              window as unknown as {
                __luxarDebug: {
                  inputHandler: { animationManager: { pause: (i: number) => void } };
                };
              }
            ).__luxarDebug.inputHandler.animationManager.pause(d);
          }, dim);
        },
        2000
      );

      // Playback commits drive the renders; the skipped ticks between them are
      // what render-on-change saves. (Audit mode renders every tick, so only
      // the scheduler's own counters would show the saving — not asserted.)
    });

    test('LOD zoom across levels', async ({ page }) => {
      test.setTimeout(90_000);
      await page.goto(`/?src=${LOD}&debug&renderAudit&renderer=${backend}`);
      await waitForLuxarReady(page);
      await waitSettled(page);
      const box = await canvasBox(page);

      await audited(
        page,
        'LOD zoom',
        async () => {
          await page.mouse.move(box.x + box.w / 2, box.y + box.h / 2);
          for (let i = 0; i < 12; i++) {
            await page.mouse.wheel(0, -240);
            await page.waitForTimeout(60);
          }
          await page.waitForTimeout(800);
          for (let i = 0; i < 18; i++) {
            await page.mouse.wheel(0, 240);
            await page.waitForTimeout(60);
          }
        },
        2500
      );
    });
  });
}

// The audit's own control: a drawn-state change that NOTHING reports (a mesh
// hidden behind every API's back) must be caught. Without this, a zero
// `missedDirty` above could be an audit that never sees anything.
test('the audit catches a silent drawn-state change', async ({ page }) => {
  await page.goto(`/?src=${POINTS}&debug&renderAudit`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
  await waitSettled(page);
  await resetCounters(page);
  await keepTicking(page);
  try {
    await page.waitForTimeout(300);
    const hidden = await page.evaluate(() => {
      const debug = (
        window as unknown as {
          __luxarDebug: {
            scene: {
              traverse: (cb: (o: { isMesh?: boolean; visible: boolean }) => void) => void;
            };
          };
        }
      ).__luxarDebug;
      let n = 0;
      debug.scene.traverse((o) => {
        if (o.isMesh && o.visible) {
          o.visible = false;
          n++;
        }
      });
      return n;
    });
    expect(hidden).toBeGreaterThan(0);
    await page.waitForTimeout(500);
  } finally {
    await stopTicking(page);
  }
  const counters = await readCounters(page);
  expect(counters.missedDirty, JSON.stringify(counters.render)).toBeGreaterThan(0);
});

test('without the audit, an idle tail skips its renders (render-on-change is live)', async ({
  page,
}) => {
  await page.goto(`/?src=${POINTS}&debug`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
  await waitSettled(page);
  await resetCounters(page);
  await keepTicking(page);
  await page.waitForTimeout(1000);
  await stopTicking(page);
  const counters = await page.evaluate(
    () =>
      (
        window as unknown as {
          __luxarDebug: { getPerf: () => { counters: Record<string, number> } };
        }
      ).__luxarDebug.getPerf().counters
  );
  // A second of ticks on an unchanged scene: the wake renders, the rest skip.
  expect(counters['render.ticks'] ?? 0).toBeGreaterThan(5);
  expect(counters['render.skippedTicks'] ?? 0).toBeGreaterThan(0);
  expect(counters['render.count'] ?? 0).toBeLessThan((counters['render.ticks'] ?? 0) / 2);
});
