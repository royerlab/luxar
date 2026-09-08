/**
 * Multi-touch gesture helpers for the mobile Playwright suite.
 *
 * Playwright's `page.touchscreen` only taps. Pinch, twist, multi-finger drag
 * and long-press are synthesised through the Chrome DevTools Protocol
 * (`Input.dispatchTouchEvent`), which is why the mobile projects run on
 * Chromium. Coordinates are CSS pixels in the viewport.
 */

import type { Page } from '@playwright/test';

interface TouchPoint {
  x: number;
  y: number;
  id: number;
}

async function cdp(page: Page) {
  return await page.context().newCDPSession(page);
}

async function dispatch(
  session: Awaited<ReturnType<typeof cdp>>,
  type: 'touchStart' | 'touchMove' | 'touchEnd' | 'touchCancel',
  touchPoints: TouchPoint[]
): Promise<void> {
  await session.send('Input.dispatchTouchEvent', { type, touchPoints });
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** One finger from `from` to `to` in `steps` moves (~16 ms apart). */
export async function dragFinger(
  page: Page,
  from: { x: number; y: number },
  to: { x: number; y: number },
  steps = 12
): Promise<void> {
  const s = await cdp(page);
  await dispatch(s, 'touchStart', [{ ...from, id: 1 }]);
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    await dispatch(s, 'touchMove', [{ x: lerp(from.x, to.x, t), y: lerp(from.y, to.y, t), id: 1 }]);
    await page.waitForTimeout(16);
  }
  await dispatch(s, 'touchEnd', []);
  await s.detach();
}

/**
 * Two fingers on a horizontal line through `centre`, from separation `r0` to
 * `r1` (half-distances), i.e. a pinch-out when `r1 > r0`.
 */
export async function pinch(
  page: Page,
  centre: { x: number; y: number },
  r0: number,
  r1: number,
  steps = 16
): Promise<void> {
  const s = await cdp(page);
  const pts = (r: number): TouchPoint[] => [
    { x: centre.x - r, y: centre.y, id: 1 },
    { x: centre.x + r, y: centre.y, id: 2 },
  ];
  await dispatch(s, 'touchStart', pts(r0));
  for (let i = 1; i <= steps; i++) {
    await dispatch(s, 'touchMove', pts(lerp(r0, r1, i / steps)));
    await page.waitForTimeout(16);
  }
  await dispatch(s, 'touchEnd', []);
  await s.detach();
}

/**
 * Two fingers at radius `r` around `centre`, rotating from angle `a0` to `a1`
 * (radians, screen space) at constant separation — a pure twist.
 */
export async function twist(
  page: Page,
  centre: { x: number; y: number },
  r: number,
  a0: number,
  a1: number,
  steps = 16
): Promise<void> {
  const s = await cdp(page);
  const pts = (a: number): TouchPoint[] => [
    { x: centre.x + r * Math.cos(a), y: centre.y + r * Math.sin(a), id: 1 },
    { x: centre.x - r * Math.cos(a), y: centre.y - r * Math.sin(a), id: 2 },
  ];
  await dispatch(s, 'touchStart', pts(a0));
  for (let i = 1; i <= steps; i++) {
    await dispatch(s, 'touchMove', pts(lerp(a0, a1, i / steps)));
    await page.waitForTimeout(16);
  }
  await dispatch(s, 'touchEnd', []);
  await s.detach();
}

/**
 * Two fingers down, then lift ONE and drag the survivor — the 2 → 1 transition
 * the orbit controls must continue as a one-finger rotate.
 */
export async function pinchThenDragSurvivor(
  page: Page,
  centre: { x: number; y: number },
  r: number,
  dragTo: { x: number; y: number },
  steps = 12
): Promise<void> {
  const s = await cdp(page);
  const a: TouchPoint = { x: centre.x - r, y: centre.y, id: 1 };
  const b: TouchPoint = { x: centre.x + r, y: centre.y, id: 2 };
  await dispatch(s, 'touchStart', [a, b]);
  await page.waitForTimeout(50);
  // Lift finger 2 — touchEnd lists the points being released.
  await dispatch(s, 'touchEnd', [b]);
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    await dispatch(s, 'touchMove', [
      { x: lerp(a.x, dragTo.x, t), y: lerp(a.y, dragTo.y, t), id: 1 },
    ]);
    await page.waitForTimeout(16);
  }
  await dispatch(s, 'touchEnd', []);
  await s.detach();
}

/** Hold one finger still for `ms`, then lift. */
export async function longPress(page: Page, at: { x: number; y: number }, ms = 700): Promise<void> {
  const s = await cdp(page);
  await dispatch(s, 'touchStart', [{ ...at, id: 1 }]);
  await page.waitForTimeout(ms);
  await dispatch(s, 'touchEnd', []);
  await s.detach();
}

/**
 * Two quick taps at the same spot.
 *
 * All four touch events are QUEUED on one CDP session without awaiting in
 * between: CDP executes them in order, so the browser sees the taps a few
 * milliseconds apart whatever the machine is doing. Awaiting each round trip
 * (or `page.touchscreen.tap` twice) put 300-700 ms between the two lifts on a
 * loaded Mac, past the viewer's 300 ms double-tap window, and the gesture read
 * as two single taps.
 */
export async function doubleTap(page: Page, at: { x: number; y: number }): Promise<void> {
  const s = await cdp(page);
  const pt: TouchPoint[] = [{ ...at, id: 1 }];
  await Promise.all([
    dispatch(s, 'touchStart', pt),
    dispatch(s, 'touchEnd', []),
    dispatch(s, 'touchStart', pt),
    dispatch(s, 'touchEnd', []),
  ]);
  await s.detach();
}

/** What the page's media queries and profile say about the emulated device. */
export async function inputProbe(page: Page): Promise<{
  coarse: boolean;
  hoverNone: boolean;
  anyHover: boolean;
  touchPoints: number;
  pageScale: number;
}> {
  return await page.evaluate(() => ({
    coarse: matchMedia('(pointer: coarse)').matches,
    hoverNone: matchMedia('(hover: none)').matches,
    anyHover: matchMedia('(any-hover: hover)').matches,
    touchPoints: navigator.maxTouchPoints,
    pageScale: window.visualViewport?.scale ?? 1,
  }));
}

/**
 * Camera pose snapshot from the debug surface.
 *
 * Controls apply input to the camera inside their per-frame `update()`, so a
 * pose read straight after a gesture can predate the frame that applies it —
 * under software GL with two workers a frame can take a second or more. Two
 * animation frames are awaited first, so the read follows an actual update.
 */
export async function cameraPose(page: Page): Promise<{
  position: [number, number, number];
  quaternion: [number, number, number, number];
  target: [number, number, number];
}> {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
      )
  );
  return await page.evaluate(() => {
    const debug = (
      window as unknown as {
        __luxarDebug: {
          camera: {
            position: { toArray: () => number[] };
            quaternion: { toArray: () => number[] };
          };
          controls: { getFocusTarget: () => { toArray: () => number[] } };
        };
      }
    ).__luxarDebug;
    return {
      position: debug.camera.position.toArray() as [number, number, number],
      quaternion: debug.camera.quaternion.toArray() as [number, number, number, number],
      target: debug.controls.getFocusTarget().toArray() as [number, number, number],
    };
  });
}

export function distance(a: [number, number, number], b: [number, number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function quaternionAngle(
  a: [number, number, number, number],
  b: [number, number, number, number]
): number {
  const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, dot));
}

/** Viewport centre of the WebGL canvas. */
export async function canvasCentre(page: Page): Promise<{ x: number; y: number }> {
  return await page.evaluate(() => {
    const canvas = (
      window as unknown as { __luxarDebug: { renderer: { domElement: HTMLCanvasElement } } }
    ).__luxarDebug.renderer.domElement;
    const r = canvas.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  });
}

/** Tap a rail button by its `data-rail-id`. */
export async function tapRail(page: Page, id: string): Promise<void> {
  await page.locator(`[data-rail-id="${id}"]`).tap();
}

/** Geometry of the first element matching `selector`, or null. */
export async function rectOf(
  page: Page,
  selector: string
): Promise<{
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
  vw: number;
  vh: number;
  scrollable: boolean;
} | null> {
  return await page.evaluate((sel) => {
    const el = document.querySelector(sel);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {
      left: r.left,
      top: r.top,
      right: r.right,
      bottom: r.bottom,
      width: r.width,
      height: r.height,
      vw: window.innerWidth,
      vh: window.innerHeight,
      scrollable: el.scrollHeight > el.clientHeight + 1,
    };
  }, selector);
}
