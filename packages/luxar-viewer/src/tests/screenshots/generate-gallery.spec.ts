/**
 * Gallery Generator — stills + orbit videos for every manifest demo.
 *
 * A single, manifest-driven harness that captures a still PNG plus a seamless
 * 360° orbit video for each demo — an animated WebP (README-inline on GitHub)
 * and a full-quality VP9 WebM master (click-through / archival) — in one pass:
 *
 *   - **Framing** — presses `F` (which restores the scene's authored
 *     `viewer_config` camera when present, else FOV-aware bounds-fit), then
 *     closed-loop dollies to a target screen coverage using an outlier-robust
 *     percentile bounding box of the lit pixels.
 *   - **Auto-exposure** — measures the composited frame (a Playwright
 *     screenshot, decoded in-page) and picks an exposure in three phases:
 *     (1) drive the lit foreground's p99 just below clipping (bright, not blown
 *     out); (2) if the lit histogram turns out to be NARROW (p99 − p10 <
 *     `NARROW_SPREAD_MAX` — a headlit shaded mesh, where p99 says nothing), re-
 *     target the lit MEDIAN to `TARGET_MID` so the surface keeps its colour
 *     instead of washing out in the ACES shoulder; (3) step down while the
 *     subject is blown white or the background is lifted to grey. The decision
 *     logic is `./exposure-policy` (pure, unit-tested); a per-demo `exposure`
 *     in the manifest overrides the whole thing.
 *   - **Seamless orbit** — ORBIT_FRAMES explicit per-angle screenshots of a
 *     small-angle SINUSOIDAL ROCK (±ORBIT_AMPLITUDE_DEG about the subject's up
 *     axis, one full period over the frame count, so the loop is continuous),
 *     assembled by ffmpeg into a WebM master + animated WebP. Explicit frames
 *     because headless Playwright video does not reliably record the viewer's
 *     rAF repaints.
 *
 * The demo list is `scripts/gallery/manifest.json` (shared with the Python
 * dataset generator). Demos whose dataset is absent are skipped (not failed).
 * Restrict to a subset with `GALLERY_ONLY=id1,id2 pnpm gallery`.
 *
 * Usage:
 *   pnpm gallery                 # capture every demo with a dataset on disk
 *   GALLERY_ONLY=lorenz pnpm gallery
 *
 * Prerequisites:
 *   - Datasets present:  hatch run python scripts/gallery/generate_gallery_datasets.py
 *   - ffmpeg installed (video conversion)
 */

import { test } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import {
  computeAutoExposure,
  CLIP_LUMA,
  CLIP_SAT_MAX,
  HI_PERCENTILE,
  LIT_THRESHOLD,
  type AutoExposureResult,
  type LumaStats,
} from './exposure-policy';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '../../../../..');
const MANIFEST_PATH = path.join(REPO_ROOT, 'scripts/gallery/manifest.json');
const OUTPUT_DIR = path.join(REPO_ROOT, 'docs/images/gallery');

// Dedicated ports (not the default 5173/9876) so a concurrent agent's dev
// server on the shared port can't poison the run (its file changes would
// trigger a mid-capture Vite reload that wipes window.__luxarDebug).
const DATA_SERVER = `http://localhost:${process.env.GALLERY_DATA_PORT ?? 9899}`;
const VIEWER_URL = `http://localhost:${process.env.GALLERY_VITE_PORT ?? 5199}`;

// Orbit video: captured as ORBIT_FRAMES explicit per-angle SCREENSHOTS (not
// Playwright's passive video, which does not reliably record the viewer's rAF
// repaints in headless — the recorded video came out frozen/choppy). ffmpeg
// assembles the frame sequence into a WebM master + animated WebP. i∈[0,N) with
// angle = i/N·2π makes the loop seamless (no duplicated 0/2π frame).
// Orbit motion: a small-angle SINUSOIDAL ROCK (±ORBIT_AMPLITUDE_DEG), not a full
// 360° turn. A full turn at an affordable frame count has a large, jarring
// inter-frame angle; a ±20° rock covers only ~4·amplitude of travel per cycle,
// so the SAME frames give a tiny (~1°) inter-frame angle → smooth, and it loops
// seamlessly (sin returns to start). Each frame is a screenshot, so orbit frames
// are captured at a reduced viewport (ORBIT_CAPTURE_PX) for speed; the still
// stays full-res.
// Capture ENOUGH real frames for one gentle rock cycle to play smoothly with NO
// motion interpolation. An earlier version captured 60 frames and used ffmpeg
// `minterpolate` (mci) to synthesize a 24 fps master — but motion-compensated
// interpolation invents motion vectors and WARPS fine structure (chromatin
// filaments, splat/point clouds smear and morph between real frames). So we now
// capture the full playback frame count as real screenshots and assemble them
// 1:1 at ORBIT_FPS — zero synthesized frames, zero warping. The rock is slow, so
// 12 fps already reads as smooth; mci was expensive, so dropping it roughly
// offsets the extra screenshots. Loops seamlessly (sin returns to start).
// real screenshots per cycle (played 1:1, no interp). GALLERY_ORBIT_FRAMES lets
// a quick smoke run (or a heavy software-GL demo) use fewer frames.
const ORBIT_FRAMES = Number(process.env.GALLERY_ORBIT_FRAMES ?? 120);
// Timelapse: number of DISTINCT timepoints sampled across the clip. Each is a
// separate gsplat-slice load, so loading one per orbit frame (120) is far too
// slow; ~40 distinct steps (each held ~3 frames) keeps the development legible
// while cutting loads 3×. The camera still rocks smoothly over all 120 frames.
const TL_STEPS = 40;
const ORBIT_AMPLITUDE_DEG = 20; // ± rock amplitude
const ORBIT_FPS = 12; // 120 frames ⇒ a 10 s cycle, played 1:1 (no interpolation)
const WEBM_WIDTH = 900; // VP9 master (archival / click-through)
const WEBM_CRF = 24; // lower = higher quality (was 34, too lossy)
// Animated WebP for the README (inline on GitHub). Small: downscaled.
// Inline README preview thumbnail: kept SMALL (26 tiles all load on the GitHub
// page). Dense rotating point clouds compress poorly, so we drop to 340px, 8fps
// and low quality — the full 900px/12fps detail lives in the WebM the tile links
// to. (512px/q60/12fps produced 6-9MB previews on star-field scenes.)
const WEBP_WIDTH = 340;
const WEBP_QUALITY = 32;
const WEBP_FPS = 8;

// Auto-exposure tuning constants and the exposure DECISION live in
// ./exposure-policy (pure + unit-tested); this file owns only the IO —
// applying an exposure and measuring the resulting frame.

// Fill-to-screen: dolly so the subject fills this fraction of the frame's
// smaller dimension. Measured from a screenshot using a PERCENTILE bounding
// box of the lit pixels (3rd–97th pctile) so a few stray outlier points don't
// keep the whole scene tiny (the Gaia/asteroid failure mode). Runs BEFORE
// exposure calibration — exposure depends on what's actually in frame.
const FILL_TARGET = 0.95; // subject fills ~95% of the min dimension (fill more)
const FILL_TOLERANCE = 0.04;
const FILL_ITERS = 7;
// Tighter percentile (3rd–97th) leans the measured extent toward the DENSE
// body and lets faint outskirts bleed off-frame — the subject reads bigger.
const BBOX_LO_PCTILE = 0.03;
const BBOX_HI_PCTILE = 0.97;

interface DemoEntry {
  id: string;
  title: string;
  geometry: string;
  category: string;
  script: string | null;
  dataset: string; // repo-relative
  exposure?: number; // per-demo exposure override (log2 stops); else auto
  // Skip the screenshot-heavy auto steps for very heavy scenes (e.g. the 3M-star
  // Gaia field, where ~20 measurement screenshots time out in software GL). When
  // false, the demo's BAKED camera + exposure (viewer_config) are used verbatim.
  autoFrame?: boolean; // default true — closed-loop fill-to-screen
  autoExpose?: boolean; // default true — percentile + clip-guard exposure
  // Per-demo fill target (fraction of frame). Default FILL_TARGET (0.95) is
  // right for compact subjects, but diffuse clouds / survey cones (DESI) want a
  // wider framing so the whole structure reads instead of zooming into the core.
  fillTarget?: number;
  dimensionNav?: { key: string; steps: number };
  // Initial view orientation (degrees), applied after F and before fill: orbit
  // the camera to this azimuth/elevation around the target so the subject is
  // seen from the right side (e.g. faces the head, looks down the z-axis).
  // elevation 90 = straight down (+/- along vertical), 0 = equator (side-on).
  viewAngle?: { azimuth?: number; elevation?: number };
  // World axis that is the subject's "up": the orbit rock revolves about it and
  // the camera up-vector uses it. Default 'y'. Set 'z' (or 'x') for a subject
  // whose long/vertical axis is world-Z — otherwise a world-Y yaw degenerates
  // into an in-plane roll (e.g. a supine CT body lying along Z).
  orbitUp?: 'x' | 'y' | 'z';
  // Multiplicative zoom applied AFTER fill-to-screen: a final dolly by 1/zoom.
  // zoom > 1 zooms IN (e.g. 3 = 3x closer), zoom < 1 zooms OUT (e.g. 0.8 = 20%
  // further). fillTarget sets coarse framing; zoom is the fine per-demo nudge
  // for subjects the closed-loop fill frames too tight or too loose.
  zoom?: number;
  // Absolute camera distance from the target (world units), applied after
  // viewAngle and INSTEAD of the closed-loop fill. Use for faint/diffuse
  // subjects the coverage-based fill can't frame (e.g. quantum orbitals).
  distance?: number;
  // Timelapse capture: for 4D time-series demos, ADVANCE the time dimension over
  // the clip while recording (instead of a static slice) so the subject evolves
  // as the camera gently rocks. Time is sampled on a coarse grid of TL_STEPS
  // distinct timepoints (each held a few frames — loading one slice per orbit
  // frame is far too slow); the STILL is framed at `framePoint` of the range.
  timelapse?: {
    // Fraction of the time range used to frame the STILL (default 0.85) — a
    // developmental series is empty at t0, so the poster wants a late timepoint.
    framePoint?: number;
  };
  // Force the finest LOD (adds ``&lod-finest``). Default TRUE — a coarse level
  // looks blurry in a hero still. Set FALSE for a very heavy scene (e.g. the
  // 2.3M-segment global rivers globe) where forcing all elements makes each
  // software-GL frame take minutes: the viewport-relative coverage LOD then
  // picks a lighter level sized to the framing, so the orbit video is feasible.
  lodFinest?: boolean;
  readme?: boolean;
  // Free-text human annotation carried in the manifest (why a demo is framed a
  // certain way, what still needs tuning). Declared so the manifest and this
  // interface agree; the capture code never reads it.
  note?: string;
}

function loadManifest(): DemoEntry[] {
  const raw = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf-8')) as { demos: DemoEntry[] };
  let demos = raw.demos;
  const only = process.env.GALLERY_ONLY;
  if (only) {
    const wanted = new Set(
      only
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    );
    demos = demos.filter((d) => wanted.has(d.id));
  }
  return demos;
}

async function waitForLuxarReady(page: any, timeout = 60000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      return debug && debug.getState && debug.getState().initialized;
    },
    { timeout }
  );
}

async function waitForDataLoaded(page: any, timeout = 60000): Promise<void> {
  // Geometry-agnostic: totalElements sums points + gsplats + lines + triangles,
  // so this works for all four geometry types and mixed scenes (totalPoints alone
  // stays 0 for a pure Lines demo like dipc_3d_genome, or a pure Mesh one).
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      if (!debug || !debug.getState) return false;
      const state = debug.getState();
      return state && !state.isLoading && state.totalElements > 0;
    },
    { timeout }
  );
}

/** Navigate a non-displayed dimension (e.g. to a populated timepoint). */
async function navigateDimension(page: any, key: string, steps: number): Promise<void> {
  await page.keyboard.press(key);
  await page.waitForTimeout(300);
  for (let i = 0; i < steps; i++) {
    await page.keyboard.press(']');
    await page.waitForTimeout(400);
  }
  await page.waitForTimeout(500);
}

/**
 * Set the time (first non-displayed) dimension to a fraction of its range and
 * await the loader commit. Used to frame a timelapse still at a content-rich
 * timepoint (keyboard stepping can't reach a deep timepoint in a 400-frame series).
 */
async function jumpTimeDimToFrac(page: any, frac: number): Promise<void> {
  await page.evaluate(async (f: number) => {
    const d = (window as any).__luxarDebug;
    const sdm = d?.sceneDimsManager;
    const dims = sdm?.getDims?.();
    const ranges = sdm?.getDimensionRanges?.();
    if (!sdm || !dims || !ranges) return;
    const displayed: number[] = dims.displayed ?? [];
    let ti = -1;
    for (let k = 0; k < dims.ndim; k++)
      if (!displayed.includes(k)) {
        ti = k;
        break;
      }
    if (ti < 0) return;
    const [mn, mx] = ranges[ti];
    sdm.setDimensionValue(ti, Math.round(mn + f * (mx - mn)));
    await sdm.waitForUpdate?.();
  }, frac);
  await page.waitForTimeout(600);
}

/**
 * The world axis the camera's current up-vector most nearly points along.
 *
 * Read AFTER framing, so it reflects whatever pose is actually on screen — the
 * demo's baked `viewer_config` up when it has one, else three.js's (0,1,0). The
 * rock axis has always been axis-aligned, so a non-axis-aligned baked up is
 * snapped to its dominant component rather than rejected.
 *
 * Falls back to `'y'` when the debug handle or camera is missing, which is the
 * historical default and keeps a partially-loaded page from throwing here.
 */
async function dominantCameraUpAxis(page: any): Promise<'x' | 'y' | 'z'> {
  const axis = await page.evaluate(() => {
    const cam = (window as any).__luxarDebug?.camera;
    if (!cam?.up) return 'y';
    const { x, y, z } = cam.up;
    const ax = Math.abs(x);
    const ay = Math.abs(y);
    const az = Math.abs(z);
    // Degenerate up (zero or non-finite): fall back to 'y' like the missing-camera
    // guard above, rather than letting the >= chain answer 'x' for an all-zero
    // vector — that would be an arbitrary axis dressed up as a measurement.
    if (!(ax + ay + az > 0)) return 'y';
    if (ax >= ay && ax >= az) return 'x';
    if (az >= ay) return 'z';
    return 'y';
  });
  return axis as 'x' | 'y' | 'z';
}

/**
 * Position the camera for a subject whose up-axis is `up` (default 'y'): look at
 * the target along the +B axis with up=U, preserving the current distance. This
 * is the base pose the orbit rock revolves around, so the still and the video
 * share orientation. For up='z' (CT lying along Z) this gives an upright coronal
 * view instead of the degenerate top-down one from F.
 */
async function positionForOrbitUp(page: any, up: string): Promise<void> {
  await page.evaluate((u: string) => {
    const d = (window as any).__luxarDebug;
    const cam = d?.camera;
    const c = d?.controls;
    if (!cam || !c?.getFocusTarget) return;
    const t = c.getFocusTarget();
    const r = Math.hypot(cam.position.x - t.x, cam.position.y - t.y, cam.position.z - t.z) || 1;
    const U =
      u === 'x' ? { x: 1, y: 0, z: 0 } : u === 'z' ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 };
    const B = u === 'z' ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
    cam.position.set(t.x - r * B.x, t.y - r * B.y, t.z - r * B.z);
    cam.up.set(U.x, U.y, U.z);
    cam.lookAt(t.x, t.y, t.z);
    cam.updateMatrixWorld?.();
    c.reinitialize?.();
    c.update?.();
    d.renderOnce?.();
  }, up);
  await page.waitForTimeout(300);
}

/** Press F to fit the scene (restores the authored camera, else bounds-fit). */
async function centerCamera(page: any): Promise<void> {
  await page.keyboard.press('f');
  await page.waitForTimeout(900);
}

/**
 * Orbit the camera to an absolute azimuth/elevation (degrees) around the
 * target, preserving the current distance. Elevation is measured from the
 * horizontal plane: 0 = side-on (equator), +90 = looking straight down the
 * scene-up axis. Used to face anatomical volumes correctly (e.g. front of the
 * head, or down the microscopy z-axis) instead of the default fit angle.
 */
async function setViewAngle(page: any, azimuthDeg: number, elevationDeg: number): Promise<void> {
  await page.evaluate(
    ({ az, el }: { az: number; el: number }) => {
      const debug = (window as any).__luxarDebug;
      const c = debug?.controls;
      const cam = debug?.camera;
      if (!cam || !c?.getFocusTarget) return;
      const t = c.getFocusTarget();
      const r = Math.hypot(cam.position.x - t.x, cam.position.y - t.y, cam.position.z - t.z);
      const a = (az * Math.PI) / 180;
      const e = (el * Math.PI) / 180;
      cam.position.set(
        t.x + r * Math.cos(e) * Math.sin(a),
        t.y + r * Math.sin(e),
        t.z + r * Math.cos(e) * Math.cos(a)
      );
      c.reinitialize?.();
      c.update?.();
      debug.renderOnce?.();
    },
    { az: azimuthDeg, el: elevationDeg }
  );
  await page.waitForTimeout(300);
}

/** Dolly the camera toward the orbit target by `factor` (<1 = closer). */
async function dolly(page: any, factor: number): Promise<void> {
  await page.evaluate((f: number) => {
    const debug = (window as any).__luxarDebug;
    const controls = debug?.controls; // ControlsManager
    const cam = debug?.camera; // the actual scene camera
    // NOTE: use debug.camera + controls.getFocusTarget(); the ControlsManager
    // does NOT expose `.object`/`.target` (those live on the inner controls),
    // so the old `controls.object` access silently no-oped the dolly.
    if (!cam || !controls?.getFocusTarget) return;
    const t = controls.getFocusTarget();
    const dx = cam.position.x - t.x;
    const dy = cam.position.y - t.y;
    const dz = cam.position.z - t.z;
    cam.position.set(t.x + dx * f, t.y + dy * f, t.z + dz * f);
    // MUST reinitialize after moving the camera externally — otherwise the
    // next update() re-derives from the stored orbit radius and snaps back.
    controls.reinitialize?.();
    controls.update?.();
    debug.renderOnce?.();
  }, factor);
  await page.waitForTimeout(250);
}

/**
 * Place the camera at an ABSOLUTE distance from the target along the current view
 * direction, bypassing the closed-loop fill. For faint/diffuse subjects (e.g. a
 * quantum-orbital density cloud) the coverage-based fill can't find the subject
 * and leaves it tiny; a hand-tuned absolute distance frames them deterministically.
 */
async function setDistance(page: any, dist: number): Promise<void> {
  await page.evaluate((d: number) => {
    const debug = (window as any).__luxarDebug;
    const controls = debug?.controls;
    const cam = debug?.camera;
    if (!cam || !controls?.getFocusTarget) return;
    const t = controls.getFocusTarget();
    const dx = cam.position.x - t.x;
    const dy = cam.position.y - t.y;
    const dz = cam.position.z - t.z;
    const r = Math.hypot(dx, dy, dz) || 1;
    const s = d / r;
    cam.position.set(t.x + dx * s, t.y + dy * s, t.z + dz * s);
    controls.reinitialize?.();
    controls.update?.();
    debug.renderOnce?.();
  }, dist);
  await page.waitForTimeout(250);
}

/**
 * Measure how much of the frame the subject fills, from a screenshot. Uses a
 * PERCENTILE bounding box of the lit pixels (BBOX_LO/HI_PCTILE) so a few stray
 * outlier points don't report a full-frame subject (the Gaia/asteroid failure).
 * Returns coverage = max(bboxW/frameW, bboxH/frameH) and the lit fraction.
 */
async function measureCoverage(page: any): Promise<{ coverage: number; litFraction: number }> {
  const shot = await page.screenshot({ type: 'jpeg', quality: 60 });
  const b64 = shot.toString('base64');
  return await page.evaluate(
    async ({
      b64img,
      litThreshold,
      loP,
      hiP,
    }: {
      b64img: string;
      litThreshold: number;
      loP: number;
      hiP: number;
    }) => {
      const blob = await (await fetch(`data:image/jpeg;base64,${b64img}`)).blob();
      const bmp = await createImageBitmap(blob);
      const w = 400;
      const h = Math.max(1, Math.round((bmp.height / bmp.width) * w));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(bmp, 0, 0, w, h);
      const data = ctx.getImageData(0, 0, w, h).data;
      const xs: number[] = [];
      const ys: number[] = [];
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = (y * w + x) * 4;
          const luma = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255;
          if (luma > litThreshold) {
            xs.push(x);
            ys.push(y);
          }
        }
      }
      if (xs.length < 10) return { coverage: 0, litFraction: xs.length / (w * h) };
      xs.sort((a, b) => a - b);
      ys.sort((a, b) => a - b);
      const q = (arr: number[], p: number) =>
        arr[Math.min(arr.length - 1, Math.max(0, Math.floor(p * arr.length)))];
      const bw = (q(xs, hiP) - q(xs, loP)) / w;
      const bh = (q(ys, hiP) - q(ys, loP)) / h;
      return { coverage: Math.max(bw, bh), litFraction: xs.length / (w * h) };
    },
    { b64img: b64, litThreshold: LIT_THRESHOLD, loP: BBOX_LO_PCTILE, hiP: BBOX_HI_PCTILE }
  );
}

/**
 * Dolly until the subject fills ~FILL_TARGET of the frame. Closed-loop on the
 * measured (outlier-robust) coverage, so it works whether the starting pose
 * came from an authored camera or a bounds fit, and regardless of scene scale.
 * Runs BEFORE exposure calibration (exposure depends on what's in frame).
 */
async function fillToScreen(page: any, target: number = FILL_TARGET): Promise<number> {
  let m = await measureCoverage(page);
  for (let iter = 0; iter < FILL_ITERS; iter++) {
    if (m.litFraction < 0.0005 || m.coverage <= 0) break; // nothing visible
    if (Math.abs(m.coverage - target) <= FILL_TOLERANCE) break;
    // factor < 1 moves closer (increases coverage). Clamp per-step so a wild
    // first estimate (huge bounds) can't overshoot past the subject.
    const factor = Math.max(0.35, Math.min(2.2, m.coverage / target));
    await dolly(page, factor);
    m = await measureCoverage(page); // re-measure so the return reflects reality
  }
  return m.coverage;
}

/**
 * Hide all viewer chrome for a clean hero shot: the control rail (+ its
 * hint/flyouts/popovers) and the DOM scene overlays (title/scalebar/info —
 * the README supplies its own captions). Injected as CSS so it persists
 * through the recorded orbit; the WebGL canvas is untouched.
 */
const CHROME_HIDER_CSS = `
  .luxar-control-rail,
  .luxar-control-rail-hint,
  .luxar-control-rail__flyout,
  .luxar-control-rail__popover,
  .luxar-gui,
  .luxar-dimensions-panel,
  .luxar-rendering-controls,
  .luxar-help-overlay,
  .luxar-resolution-indicator,
  .luxar-overlay { display: none !important; }
`;

/**
 * Register the chrome-hiding CSS as an init script so it is (re)applied on
 * EVERY document load — including a mid-session full reload (Vite's
 * "re-optimizing dependencies" reload drops a one-shot addStyleTag and was
 * leaving the rail/overlays visible). Must be called before `page.goto`.
 */
async function installChromeHider(page: any): Promise<void> {
  await page.addInitScript((css: string) => {
    const inject = () => {
      const s = document.createElement('style');
      s.setAttribute('data-gallery-chrome-hider', '');
      s.textContent = css;
      document.head?.appendChild(s);
    };
    if (document.head) inject();
    else window.addEventListener('DOMContentLoaded', inject);
  }, CHROME_HIDER_CSS);
}

/** Re-assert the chrome-hiding CSS in the live DOM (belt-and-braces). */
async function hideChrome(page: any): Promise<void> {
  await page.addStyleTag({ content: CHROME_HIDER_CSS }).catch(() => {});
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
}

async function setExposure(page: any, stops: number): Promise<void> {
  await page.evaluate((s: number) => {
    const debug = (window as any).__luxarDebug;
    debug?.app?.sceneManager?.updateExposure?.(s);
    debug?.renderOnce?.();
  }, stops);
  await page.waitForTimeout(200);
}

/**
 * Return luminance stats of the "lit" (foreground) pixels: the low/median/high
 * percentiles (the spread drives the flat-subject exposure pass) and the lit
 * fraction. Measured from a Playwright screenshot (the true composited frame)
 * rather than `gl.readPixels` — the renderer runs with
 * `preserveDrawingBuffer: false` and `renderOnce()` drives an async rAF loop,
 * so the default framebuffer is empty by the time an in-page readback runs.
 * The screenshot is decoded back inside the page (createImageBitmap → 2D
 * canvas at native resolution → getImageData) so no Node image dependency is
 * needed and the blown-fraction matches the final still.
 */
async function measureLuminance(page: any): Promise<LumaStats> {
  const shot = await page.screenshot({ type: 'jpeg', quality: 70 });
  const b64 = shot.toString('base64');
  return await page.evaluate(
    async ({
      b64img,
      litThreshold,
      hiPercentile,
      clipLuma,
      clipSatMax,
    }: {
      b64img: string;
      litThreshold: number;
      hiPercentile: number;
      clipLuma: number;
      clipSatMax: number;
    }) => {
      const blob = await (await fetch(`data:image/jpeg;base64,${b64img}`)).blob();
      const bmp = await createImageBitmap(blob);
      // Measure at NATIVE resolution: any downscale averages neighbouring
      // pixels and hides blown cores, so the clip guard would under-count and
      // stop above the target (spiral_galaxy read ≤5% at 600px but 19% at full
      // res). Matching the still's resolution makes the guard enforce the real
      // blown fraction (and agree with score_exposure.py).
      const w = bmp.width;
      const h = bmp.height;
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(bmp, 0, 0, w, h);
      const data = ctx.getImageData(0, 0, w, h).data;
      const total = w * h;
      // Histogram the lit-pixel luma (1024 bins over [0,1]) instead of pushing
      // every luma into an array and sorting it — a native-resolution frame can
      // be ~1M lit pixels and this runs up to 21× per demo (3 p99 iterations +
      // 1 spread probe + MID_EXPOSURE_ITERS + CLIP_GUARD_ITERS, less the probe
      // that the next phase reuses), so the array+sort was needless memory
      // churn. The histogram gives every percentile in O(n), no growth.
      const BINS = 1024;
      const hist = new Int32Array(BINS); // lit-pixel luma
      const allHist = new Int32Array(BINS); // ALL-pixel luma (for background level)
      let litCount = 0;
      let blown = 0;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i] / 255;
        const g = data[i + 1] / 255;
        const b = data[i + 2] / 255;
        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        allHist[Math.min(BINS - 1, (luma * BINS) | 0)]++;
        if (luma > litThreshold) {
          litCount++;
          hist[Math.min(BINS - 1, (luma * BINS) | 0)]++;
          const mx = Math.max(r, g, b);
          const mn = Math.min(r, g, b);
          const sat = mx > 0 ? (mx - mn) / mx : 0;
          // Blown = bright AND desaturated (a saturated bright colour is fine).
          if (luma > clipLuma && sat < clipSatMax) blown++;
        }
      }
      const pctile = (h: Int32Array, count: number, p: number): number => {
        const rank = Math.min(count, Math.max(1, Math.ceil(p * count)));
        let c = 0;
        for (let bin = 0; bin < BINS; bin++) {
          c += h[bin];
          if (c >= rank) return (bin + 0.5) / BINS;
        }
        return 1;
      };
      // bgLuma = 10th-percentile luma of the WHOLE frame — the "background"
      // level. A hero shot on black should keep this near 0; if exposure lifts
      // the background (bloom/haze) to grey, bgLuma rises and the guard pulls
      // exposure back down (the white-blown clip check alone misses grey bgs).
      const bgLuma = pctile(allHist, total, 0.1);
      if (litCount === 0)
        return { hiLuma: 0, loLuma: 0, midLuma: 0, litFraction: 0, clippedFrac: 0, bgLuma };
      // loLuma/midLuma come from the SAME histogram as hiLuma — no extra pass
      // over the pixels, no extra screenshot. hiLuma − loLuma is the lit
      // histogram's spread, which tells a headlit (flat) subject apart from an
      // emissive one; midLuma is the flat-subject exposure anchor.
      return {
        hiLuma: pctile(hist, litCount, hiPercentile),
        loLuma: pctile(hist, litCount, 0.1),
        midLuma: pctile(hist, litCount, 0.5),
        litFraction: litCount / total,
        clippedFrac: blown / litCount,
        bgLuma,
      };
    },
    {
      b64img: b64,
      litThreshold: LIT_THRESHOLD,
      hiPercentile: HI_PERCENTILE,
      clipLuma: CLIP_LUMA,
      clipSatMax: CLIP_SAT_MAX,
    }
  );
}

/**
 * Pick an exposure in three phases (the decision itself lives in
 * `./exposure-policy::computeAutoExposure`; this only wires it to the page):
 *   1. Percentile pass — converge so the lit foreground's high percentile hits
 *      TARGET_HI (bright but not clipped). Exposure is ~log-linear in
 *      luminance, so a few log2 corrections converge.
 *   2. Flat-subject pass — if the lit histogram's spread (p99 − p10) is under
 *      NARROW_SPREAD_MAX, the subject has no internal dynamic range (a headlit
 *      shaded mesh: N·L ≈ 1 everywhere) and p99 is a meaningless anchor, so
 *      re-target the lit MEDIAN to TARGET_MID instead. An EMPIRICAL gate — see
 *      NARROW_SPREAD_MAX for the measured margins; when it fires, the capture
 *      log says so ("flat subject") so a sweep can spot a false positive.
 *   3. Guard — step exposure DOWN while EITHER >5% of the subject is blown to
 *      white OR the background is lifted to grey (frame p10 above near-black).
 *      Phase 1 (p99 target) over-boosts sparse/bloomy scenes into a grey wash;
 *      the background term is what pulls those back to a black background.
 * Returns the chosen stops and whether phase 2 fired.
 */
async function autoExpose(page: any): Promise<AutoExposureResult> {
  return await computeAutoExposure({
    apply: (s: number) => setExposure(page, s),
    measure: () => measureLuminance(page),
  });
}

/**
 * Capture a seamless 360° orbit as ORBIT_FRAMES explicit per-angle screenshots.
 *
 * NOT Playwright's passive video: in headless it does not reliably record the
 * viewer's rAF repaints, so the recorded video came out frozen/choppy even
 * though the camera was moving. Instead we FREEZE the render loop + controls
 * (so nothing overrides the camera), then for each azimuth set the camera,
 * render one frame synchronously, and screenshot it — exactly the still-capture
 * path that already works. Returns the number of frames written to `framesDir`.
 */
async function captureOrbitFrames(page: any, framesDir: string, demo?: DemoEntry): Promise<number> {
  // Timelapse warm-up: for 4D time-series, PLAY the time dim a couple of cycles
  // first so the slice cache is primed (the built-in t+1 prefetch warms the next
  // timepoint), then we step it deterministically during capture. Returns the
  // discovered time-dimension index + range so we can pace the warm-up and steps.
  let tl: { timeIdx: number; min: number; max: number; lo: number } | null = null;
  if (demo?.timelapse) {
    const found = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const sdm = debug?.sceneDimsManager;
      const dims = sdm?.getDims?.();
      const ranges = sdm?.getDimensionRanges?.();
      if (!sdm || !dims || !ranges) return null;
      // Time dim = first NON-displayed dimension.
      const displayed: number[] = dims.displayed ?? [];
      let timeIdx = -1;
      for (let d = 0; d < dims.ndim; d++)
        if (!displayed.includes(d)) {
          timeIdx = d;
          break;
        }
      if (timeIdx < 0) return null;
      const [min, max] = ranges[timeIdx];
      return { timeIdx, min, max };
    });
    if (found) {
      const lo = Math.round(found.min + 0.12 * (found.max - found.min)); // skip near-empty start
      const tlv = { ...found, lo };
      tl = tlv;
      // Prime the slice cache: load each of the TL_STEPS distinct capture
      // timepoints once (much cheaper than the old 400-frame play warm-up). The
      // capture pass then largely hits cache.
      for (let k = 0; k < TL_STEPS; k++) {
        const v = Math.round(tlv.lo + ((tlv.max - tlv.lo) * k) / (TL_STEPS - 1));
        await page.evaluate(
          async ({ timeIdx, value }: { timeIdx: number; value: number }) => {
            const sdm = (window as any).__luxarDebug?.sceneDimsManager;
            sdm?.setDimensionValue?.(timeIdx, value);
            await sdm?.waitForUpdate?.();
          },
          { timeIdx: tlv.timeIdx, value: v }
        );
      }
    }
  }

  // Rock axis. An explicit `orbitUp` wins; otherwise DERIVE it from the camera's
  // own up-vector rather than assuming world-Y.
  //
  // This used to be a flat `?? 'y'`, and that default was silently wrong for any
  // demo whose `viewer_config` bakes a non-Y up. The still keeps the baked pose
  // (`F` restores it) but the loop below hard-sets `cam.up` every frame, so those
  // demos render an animation ROLLED away from their own poster, with the
  // turntable degenerating into an in-plane tumble — measured at a 130% swing in
  // subject aspect over one rock. Three demos bake a non-Y up and none had opted
  // in to `orbitUp`, because nothing told them they had to (#1377). Only one of
  // the three was visibly shipping the roll: another's checked-in dataset
  // predates its own CameraConfig and carries no camera at all, and the third's
  // camera was not orbiting in the first place (#1383).
  //
  // Deriving it means a baked up is honoured by default and `orbitUp` becomes a
  // true override. A demo that bakes nothing gets three.js's default camera up,
  // (0,1,0) -> 'y', so this is a no-op for every previously-correct demo.
  const upAxis = demo?.orbitUp ?? (await dominantCameraUpAxis(page));
  console.log(
    `[${demo?.id ?? 'orbit'}] orbitUp=${upAxis}` +
      (demo?.orbitUp ? ' (manifest)' : ' (derived from camera up)')
  );
  const ok = await page.evaluate((up: string) => {
    const debug = (window as any).__luxarDebug;
    const ac = debug?.animationController;
    const cam = debug?.camera;
    const c = debug?.controls;
    if (!ac || !cam || !c?.getFocusTarget) return false;
    ac.stopAnimation?.(); // freeze the rAF loop so it can't move/re-render the camera
    c.setEnabled?.(false); // and stop controls damping from touching it
    const t = c.getFocusTarget();
    const off = { x: cam.position.x - t.x, y: cam.position.y - t.y, z: cam.position.z - t.z };
    // The rock revolves the camera about the world `up` axis (U), in the plane of
    // the other two axes (A,B). Default 'y' reproduces the original XZ yaw. For a
    // subject whose long/vertical axis is world-Z (e.g. a supine CT body), 'z'
    // gives a proper turntable instead of an in-plane roll.
    const U =
      up === 'x' ? { x: 1, y: 0, z: 0 } : up === 'z' ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 };
    const A = up === 'x' ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
    const B = up === 'z' ? { x: 0, y: 1, z: 0 } : { x: 0, y: 0, z: 1 };
    const dot = (p: any, q: any) => p.x * q.x + p.y * q.y + p.z * q.z;
    const compA = dot(off, A);
    const compB = dot(off, B);
    (window as any).__orbit = {
      phi0: Math.atan2(compA, compB),
      radius: Math.hypot(compA, compB),
      compU: dot(off, U),
      A,
      B,
      U,
      tx: t.x,
      ty: t.y,
      tz: t.z,
    };
    return true;
  }, upAxis);
  if (!ok) return 0;

  fs.mkdirSync(framesDir, { recursive: true });
  const ampRad = (ORBIT_AMPLITUDE_DEG * Math.PI) / 180;
  let tlLastV = -1;
  for (let i = 0; i < ORBIT_FRAMES; i++) {
    // Timelapse: advance the time dim across the clip on a COARSE grid of
    // TL_STEPS distinct timepoints (each held ~ORBIT_FRAMES/TL_STEPS frames), so
    // the camera rock stays smooth over all frames while the (expensive) slice
    // loads are ~3× fewer. Only reload when the quantized timepoint changes.
    if (tl) {
      const k = Math.min(TL_STEPS - 1, Math.floor((i / ORBIT_FRAMES) * TL_STEPS));
      const v = Math.round(tl.lo + ((tl.max - tl.lo) * k) / (TL_STEPS - 1));
      if (v !== tlLastV) {
        tlLastV = v;
        await page.evaluate(
          async ({ timeIdx, value }: { timeIdx: number; value: number }) => {
            const sdm = (window as any).__luxarDebug?.sceneDimsManager;
            sdm?.setDimensionValue?.(timeIdx, value);
            await sdm?.waitForUpdate?.();
          },
          { timeIdx: tl.timeIdx, value: v }
        );
      }
    }
    await page.evaluate(
      async ({ idx, n, amp }: { idx: number; n: number; amp: number }) => {
        const debug = (window as any).__luxarDebug;
        const o = (window as any).__orbit;
        // Guard: a mid-run reload would wipe these; skip the frame rather than
        // throw (the run then just has a duplicate frame, not a crash).
        if (!debug?.camera || !o) return;
        const cam = debug.camera;
        // Sinusoidal rock: one seamless period over N frames, ±amp radians, in
        // the A,B plane about the up axis U (position = center + r·(sinθ·A +
        // cosθ·B) + compU·U). up = U keeps the subject upright.
        const theta = o.phi0 + amp * Math.sin((idx / n) * Math.PI * 2);
        const s = Math.sin(theta);
        const c = Math.cos(theta);
        cam.position.set(
          o.tx + o.radius * (s * o.A.x + c * o.B.x) + o.compU * o.U.x,
          o.ty + o.radius * (s * o.A.y + c * o.B.y) + o.compU * o.U.y,
          o.tz + o.radius * (s * o.A.z + c * o.B.z) + o.compU * o.U.z
        );
        cam.up.set(o.U.x, o.U.y, o.U.z);
        cam.lookAt(o.tx, o.ty, o.tz);
        cam.updateMatrixWorld();
        // The orbit stopped the rAF loop, so the per-frame depth-sort scheduler
        // is dead — re-sort depth-ordered nodes for THIS pose before rendering,
        // or every frame would show the permutation frozen at the pre-orbit pose
        // (order-dependent modes: normal / volumetric). Offline, so a full sort
        // per frame is affordable.
        await debug.resortDepthOrderingForCapture?.();
        debug.postProcessing?.render?.(); // synchronous render with the new camera
      },
      { idx: i, n: ORBIT_FRAMES, amp: ampRad }
    );
    await page.screenshot({
      path: path.join(framesDir, `f${String(i).padStart(4, '0')}.png`),
      type: 'png',
    });
  }
  return ORBIT_FRAMES;
}

/**
 * Warn when the still and the FIRST orbit frame disagree.
 *
 * They are the same nominal pose — the rock is `phi0 + amp*sin(0)` at frame 0 —
 * so they should render near-identically. When they do not, the orbit is showing
 * the subject from somewhere the poster never does, and since the README embeds
 * the ANIMATION while reviewers usually look at the still, that divergence ships
 * unnoticed. It already did: three demos bake a non-Y camera up, and the one
 * whose dataset actually carries that camera shipped an animation rolled ~90 deg
 * from its own poster (#1377). Nothing in this harness compared the two.
 *
 * Normalised cross-correlation on a 256x256 greyscale downscale — deliberately
 * not SSIM, which is punishing on high-frequency filamentary subjects (a
 * correctly-matched cosmic-web tile scores 0.57 while correlating at 0.98) and
 * would cry wolf. Measured separation is wide: 0.98-1.00 when consistent,
 * 0.52-0.54 when rolled, so 0.85 sits clear of both.
 *
 * WARNS rather than fails. A demo may legitimately reorient between still and
 * orbit (`orbitUp` + `viewAngle` do exactly that on purpose), so this is a "look
 * at this" signal, not a correctness gate.
 */
async function warnIfStillDisagreesWithOrbit(
  page: any,
  stillPath: string,
  frameZeroPath: string,
  demo: DemoEntry
): Promise<void> {
  // Timelapse demos are exempt: the still is deliberately framed at
  // `timelapse.framePoint` of the time range while orbit frame 0 sits at the
  // clip's start, so the two show DIFFERENT TIMEPOINTS and correlate ~0.14 even
  // when the camera agrees perfectly. Comparing them would warn on every
  // timelapse tile, which is how a guard gets ignored.
  //
  // Note this tests the DECLARED field, while the per-frame re-freeze in
  // `captureOrbitFrames` gates on `tl !== null` — whether time stepping is
  // actually happening. Deliberately not the same question: a demo that declares
  // `timelapse` but whose scene has no non-displayed dimension steps nothing, so
  // the freeze correctly stays off while this exemption still (harmlessly) skips
  // one warning. Do not "align" the two.
  if (demo.timelapse) return;
  if (!fs.existsSync(stillPath) || !fs.existsSync(frameZeroPath)) return;
  const [a, b] = [stillPath, frameZeroPath].map((f) => fs.readFileSync(f).toString('base64'));
  try {
    const corr = await page.evaluate(
      async ([s, f]: [string, string]) => {
        const grey = async (b64: string) => {
          const blob = await (await fetch('data:image/png;base64,' + b64)).blob();
          const img = await createImageBitmap(blob);
          const c = new OffscreenCanvas(256, 256);
          const ctx = c.getContext('2d')!;
          ctx.drawImage(img, 0, 0, 256, 256);
          const d = ctx.getImageData(0, 0, 256, 256).data;
          const out = new Float64Array(256 * 256);
          for (let i = 0, j = 0; i < d.length; i += 4, j++) {
            out[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
          }
          const mean = out.reduce((p, q) => p + q, 0) / out.length;
          let ss = 0;
          for (let i = 0; i < out.length; i++) {
            out[i] -= mean;
            ss += out[i] * out[i];
          }
          const sd = Math.sqrt(ss / out.length);
          if (sd > 0) for (let i = 0; i < out.length; i++) out[i] /= sd;
          return out;
        };
        const [ga, gb] = [await grey(s), await grey(f)];
        let acc = 0;
        for (let i = 0; i < ga.length; i++) acc += ga[i] * gb[i];
        return acc / ga.length;
      },
      [a, b]
    );
    if (corr < 0.85) {
      console.warn(
        `[${demo.id}] ⚠️  still and orbit frame 0 disagree (correlation ${corr.toFixed(3)}). ` +
          'They are the same nominal pose, so the animation is showing a different ' +
          "orientation than the poster — check 'orbitUp' against the demo's baked " +
          'viewer_config camera up (see #1377).'
      );
    }
  } catch (e) {
    // Diagnostic only — never let it break a capture. But say so: a guard that
    // fails silently is worse than no guard, because the absence of a warning
    // reads as "checked and fine". If `createImageBitmap`/`OffscreenCanvas` ever
    // go missing, this line is what stops #1377 regressing unnoticed.
    console.warn(`[${demo.id}] still-vs-orbit check could not run:`, e);
  }
}

/**
 * High-quality VP9 WebM master from the orbit frame sequence. Constant-quality
 * (CRF), looping by construction. GitHub renders a committed .webm with a player
 * on the file page; the README embeds the smaller WebP and links to this.
 */
function convertFramesToWebm(framesDir: string, output: string): void {
  // Assemble the real frames 1:1 at ORBIT_FPS — NO minterpolate. Every output
  // frame is a genuine screenshot, so there is no motion-vector warping of fine
  // structure. Only a spatial downscale to WEBM_WIDTH.
  execSync(
    `ffmpeg -y -framerate ${ORBIT_FPS} -i "${framesDir}/f%04d.png" ` +
      `-vf "scale=${WEBM_WIDTH}:-1" -c:v libvpx-vp9 -crf ${WEBM_CRF} -b:v 0 -pix_fmt yuv420p -an "${output}"`,
    { stdio: 'pipe' }
  );
}

/** Small animated WebP loop for the README (inline on GitHub) from the frames. */
function convertFramesToWebp(framesDir: string, output: string): void {
  // Same 1:1 assembly (no minterpolate) — real frames only, no warping.
  execSync(
    `ffmpeg -y -framerate ${ORBIT_FPS} -i "${framesDir}/f%04d.png" ` +
      `-vf "fps=${WEBP_FPS},scale=${WEBP_WIDTH}:-1" -vcodec libwebp -lossless 0 -compression_level 6 -q:v ${WEBP_QUALITY} -loop 0 -preset default -an -vsync 0 "${output}"`,
    { stdio: 'pipe' }
  );
}

const DEMOS = loadManifest();

test.beforeAll(() => {
  if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  console.log(`\n[Gallery] Output: ${OUTPUT_DIR}`);
  console.log(`[Gallery] ${DEMOS.length} demos in manifest\n`);
});

for (const demo of DEMOS) {
  test(`Capture ${demo.id} (${demo.title})`, async ({ page }, testInfo) => {
    const datasetOnDisk = fs.existsSync(path.join(REPO_ROOT, demo.dataset));
    if (!datasetOnDisk) {
      testInfo.skip(true, `dataset missing: ${demo.dataset} (run generate_gallery_datasets.py)`);
      return;
    }

    const dataUrl = `${DATA_SERVER}/${demo.dataset}`;
    // &lod-finest forces the finest LOD level regardless of screen coverage —
    // a coarse level looks blurry in a hero still even when the subject is small.
    // Opt out (lodFinest:false) for a very heavy scene where forcing every
    // element makes each software-GL frame take minutes (see DemoEntry.lodFinest).
    const finestParam = demo.lodFinest === false ? '' : '&lod-finest';
    const viewerUrl = `${VIEWER_URL}/?src=${dataUrl}&debug${finestParam}`;
    console.log(`[${demo.id}] ${dataUrl}`);

    await installChromeHider(page); // survives Vite reloads (must precede goto)
    await page.goto(viewerUrl, { waitUntil: 'networkidle' });
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
    // Settle: a Vite "re-optimizing deps" full reload can fire right after first
    // paint; wait out any in-flight reload and re-confirm the scene is loaded.
    await page.waitForTimeout(1500);
    await waitForDataLoaded(page);
    console.log(`[${demo.id}] data loaded`);

    if (demo.dimensionNav) {
      await navigateDimension(page, demo.dimensionNav.key, demo.dimensionNav.steps);
    }
    // Timelapse: frame the STILL at a content-rich timepoint (a developmental
    // series is nearly empty at t0). The orbit video still plays the full range.
    if (demo.timelapse) {
      const frac = demo.timelapse.framePoint ?? 0.85;
      await jumpTimeDimToFrac(page, frac);
      console.log(`[${demo.id}] timelapse still framePoint=${frac}`);
    }

    await hideChrome(page);
    // Frame FIRST (F restores the authored camera or bounds-fits), optionally
    // orient to a demo-specified view angle, THEN fill the screen — both the
    // fill and the exposure depend on what's actually in frame.
    await centerCamera(page);
    if (demo.orbitUp) {
      await positionForOrbitUp(page, demo.orbitUp);
    }
    if (demo.viewAngle) {
      await setViewAngle(page, demo.viewAngle.azimuth ?? 0, demo.viewAngle.elevation ?? 0);
    }
    if (typeof demo.distance === 'number') {
      await setDistance(page, demo.distance);
      console.log(`[${demo.id}] distance=${demo.distance} (absolute, fill bypassed)`);
    } else if (demo.autoFrame !== false) {
      const coverage = await fillToScreen(page, demo.fillTarget);
      console.log(`[${demo.id}] coverage=${(coverage * 100).toFixed(0)}%`);
    } else {
      console.log(`[${demo.id}] framing=baked (autoFrame off)`);
    }
    // Final per-demo zoom nudge (dolly by 1/zoom): zoom>1 closer, zoom<1 further.
    if (typeof demo.zoom === 'number' && demo.zoom > 0 && demo.zoom !== 1) {
      await dolly(page, 1 / demo.zoom);
      console.log(`[${demo.id}] zoom=${demo.zoom}x`);
    }

    if (typeof demo.exposure === 'number') {
      await setExposure(page, demo.exposure);
      console.log(`[${demo.id}] exposure=${demo.exposure.toFixed(2)} stops (override)`);
    } else if (demo.autoExpose !== false) {
      const { stops, flatSubject } = await autoExpose(page);
      const how = flatSubject ? 'auto, flat subject' : 'auto';
      console.log(`[${demo.id}] exposure=${stops.toFixed(2)} stops (${how})`);
    } else {
      console.log(`[${demo.id}] exposure=baked (autoExpose off)`);
    }

    // Let the forced-finest LOD stream/commit and a couple of frames render
    // before the still (fine gsplat/point levels load progressively).
    await page.waitForTimeout(2500);

    // Still.
    const pngPath = path.join(OUTPUT_DIR, `${demo.id}.png`);
    await page.screenshot({ path: pngPath, type: 'png' });
    console.log(`[${demo.id}] still → ${path.basename(pngPath)}`);

    // Orbit: capture explicit per-angle frames (reliable in headless), then
    // assemble the WebM master + animated WebP.
    const framesDir = path.join(OUTPUT_DIR, `_frames_${demo.id}`);
    const n = await captureOrbitFrames(page, framesDir, demo);
    await warnIfStillDisagreesWithOrbit(page, pngPath, path.join(framesDir, 'f0000.png'), demo);
    await page.close();
    if (n === 0) {
      console.error(`[${demo.id}] orbit capture failed (no camera/controls)`);
      return;
    }
    console.log(`[${demo.id}] captured ${n} orbit frames`);

    const webpPath = path.join(OUTPUT_DIR, `${demo.id}.webp`);
    const webmPathOut = path.join(OUTPUT_DIR, `${demo.id}.webm`);
    try {
      convertFramesToWebp(framesDir, webpPath); // README inline (GitHub)
      console.log(`[${demo.id}] webp → ${path.basename(webpPath)}`);
    } catch (e) {
      console.error(`[${demo.id}] webp failed:`, e);
    }
    try {
      convertFramesToWebm(framesDir, webmPathOut); // full-quality master
      console.log(`[${demo.id}] webm → ${path.basename(webmPathOut)}`);
    } catch (e) {
      console.error(`[${demo.id}] webm failed:`, e);
    }
    fs.rmSync(framesDir, { recursive: true, force: true }); // clean up frames
  });
}

test('Gallery summary', async () => {
  console.log('\n======== Gallery capture complete ========');
  console.log(`Output: ${OUTPUT_DIR}`);
  for (const demo of DEMOS) {
    const png = fs.existsSync(path.join(OUTPUT_DIR, `${demo.id}.png`));
    const webp = fs.existsSync(path.join(OUTPUT_DIR, `${demo.id}.webp`));
    const webm = fs.existsSync(path.join(OUTPUT_DIR, `${demo.id}.webm`));
    console.log(
      `  ${png ? '[png]' : '[---]'} ${webp ? '[webp]' : '[----]'} ${webm ? '[webm]' : '[----]'} ${demo.id}`
    );
  }
  console.log('');
});
