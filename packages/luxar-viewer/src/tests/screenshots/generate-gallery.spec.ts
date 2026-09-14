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
 *     `NARROW_SPREAD_MAX`, where p99 says nothing about mid-tone placement), re-
 *     target the lit MEDIAN to `TARGET_MID` so the surface keeps its colour
 *     instead of washing out in the ACES shoulder; (3) step down while the
 *     subject is blown white or the background is lifted to grey. The decision
 *     logic is `./exposure-policy` (pure, unit-tested); a per-demo `exposure`
 *     in the manifest overrides the whole thing.
 *   - **Under-fill check** — measures final still span + lit area on every
 *     framing path after all nudges. Warns below the measured gallery floors
 *     and never fails — see `./crop-policy`.
 *   - **Crop check** — counts LIT pixels on the frame's outermost row/column in
 *     the still and in orbit poses spread across the whole rock on a ~5° grid
 *     (plus the last frame of a timelapse, where a developing subject is
 *     largest). The fill loop's percentile bbox is
 *     blind to exactly this (what touches the edge IS the outliers it discards),
 *     so a tile can report a good fit while the subject runs off frame. Warns
 *     with a suggested knob and never fails — a non-zero count is common and
 *     often legitimate — see `./crop-policy`.
 *   - **Seamless orbit** — ORBIT_FRAMES explicit per-angle screenshots of a
 *     small-angle SINUSOIDAL ROCK (±ORBIT_AMPLITUDE_DEG about the subject's up
 *     axis, one full period over the frame count, so the loop is continuous),
 *     assembled by ffmpeg into a WebM master + animated WebP. Explicit frames
 *     because headless Playwright video does not reliably record the viewer's
 *     rAF repaints.
 *
 * The demo list is `scripts/gallery/manifest.json` (shared with the Python
 * dataset generator). Demos whose dataset is absent are skipped (not failed).
 * Restrict to manifest ids with `GALLERY_ONLY=id1,id2 pnpm gallery`, or use
 * `GALLERY_ONLY=readme` for the media embedded in the root README.
 *
 * Usage:
 *   pnpm gallery                 # capture every demo with a dataset on disk
 *   GALLERY_ONLY=lorenz pnpm gallery
 *   GALLERY_ONLY=readme pnpm gallery
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
  COMPARE_SIZE,
  DISAGREEMENT_THRESHOLD,
  normalizedCrossCorrelation,
} from './frame-similarity';
import { dominantSignedAxis, type SignedUpAxis } from './orbit-axis';
import {
  computeAutoExposure,
  CLIP_LUMA,
  CLIP_SAT_MAX,
  HI_PERCENTILE,
  LIT_THRESHOLD,
  type AutoExposureResult,
  type LumaStats,
} from './exposure-policy';
import {
  borderLitPercent,
  borderSampleFrames,
  evaluateBorderLit,
  evaluateUnderfill,
  selectCropVerdictSamples,
  type BorderSample,
  type CoverageMeasurement,
  type CropFraming,
} from './crop-policy';
import { mediaKeyIndex, requireMediaBaseUrl, resolveGalleryOnly } from './gallery-selection';
import { resolveTimelapseSettleMs, waitForTimelapseSliceSettled } from './gallery-timelapse-settle';
import {
  hasNonZeroDimensionStep,
  isGalleryDataReady,
  readBakedDimensionStep,
} from './gallery-dimension-readiness';
import {
  checkGalleryMediaSize,
  collectGalleryMediaSizeIssues,
  formatGalleryCaptureMetrics,
  formatMediaSize,
  galleryDroppedElementsWarning,
  skipGalleryMediaWhenRequested,
  summarizeGalleryMedia,
  type GalleryMediaFile,
} from './gallery-media-reporting';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '../../../../..');
const MANIFEST_PATH = path.join(REPO_ROOT, 'scripts/gallery/manifest.json');
const README_PATH = path.join(REPO_ROOT, 'README.md');
const MEDIA_MANIFEST_PATH = path.join(REPO_ROOT, 'scripts/gallery/media-manifest.json');
const OUTPUT_DIR = path.join(REPO_ROOT, 'docs/images/gallery');
const capturedMedia: GalleryMediaFile[] = [];

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
// seamlessly (sin returns to start). Each frame is a plain screenshot of the
// same square viewport as the still (1080², set in playwright.gallery.config.ts)
// — there is no separate orbit capture size.
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
// How long cache priming and capture wait for each timelapse slice's full
// progressive ladder to drain. `waitForUpdate()` resolves at the first geometry
// commit, while later rungs are still arriving; shooting during that drain is
// what bakes pop-in into the clip (measured: 15.7x frame-to-frame spike on
// gsplats_4d_zebrafish_timelapse vs <1.8x for a clean rock). Bounded so one
// stubborn slice cannot stall a 120-frame x ~90-demo sweep; invalid/non-positive
// overrides fall back to the finite default.
const TL_SETTLE_MS = resolveTimelapseSettleMs(process.env.GALLERY_TL_SETTLE_MS);
const ORBIT_AMPLITUDE_DEG = 20; // ± rock amplitude
const ORBIT_FPS = 12; // 120 frames ⇒ a 10 s cycle, played 1:1 (no interpolation)
const WEBM_WIDTH = 900; // VP9 master (archival / click-through)
const WEBM_CRF = 24; // lower = higher quality (was 34, too lossy)
// Animated WebP for the README (inline on GitHub). Small: downscaled.
// Inline README preview thumbnail: kept SMALL (28 tiles all load on the GitHub
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
  // Dataset credit, copied VERBATIM from the demo's DEMO_META citation.short (a
  // Python test cross-checks the two). Present only when the demo declares a
  // real credit: an ABSENT key means "unknown or not yet recorded", which is not
  // the same claim as "nothing to credit", so a procedural demo carries no
  // `citation` either. Declared so the manifest and this interface agree; the
  // capture code never reads it (the on-scene footer is what a tile renders).
  citation?: string;
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
  // Relative override applied after the scene reaches its baked opening step.
  dimensionNav?: { key: string; steps: number };
  // Initial view orientation (degrees), applied after F and before fill: orbit
  // the camera to this azimuth/elevation around the target so the subject is
  // seen from the right side (e.g. faces the head, looks down the z-axis).
  // elevation 90 = straight down (+/- along vertical), 0 = equator (side-on).
  viewAngle?: { azimuth?: number; elevation?: number };
  // World axis that is the subject's "up": the orbit rock revolves about it and
  // the camera up-vector uses it — the orbit loop hard-SETS `cam.up` from this
  // on every frame. DEFAULT = derived from the camera's own up-vector after
  // framing (`dominantCameraUpAxis`), so a baked `viewer_config` up is honoured
  // without being declared here; a demo that bakes nothing gets three.js's
  // (0,1,0) -> 'y', the historical default.
  // Set it only to OVERRIDE that — e.g. to rock about something other than the
  // scene's own up, or to give a subject whose long/vertical axis is world-Z an
  // upright turntable instead of the in-plane roll a world-Y yaw degenerates
  // into (a supine CT body lying along Z) — and note that setting it also runs
  // `positionForOrbitUp`, which re-parks the camera on the axis and discards the
  // baked framing, so it usually wants a `viewAngle` beside it (that parked pose
  // can land edge-on).
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
  // Capture a STATIC single-frame tile and skip the orbit video entirely, for a
  // subject whose apparent extent changes sharply with view angle — a row of
  // objects foreshortening, a flat wall going edge-on — where the rock reads as
  // flashing rather than motion. gsplats_lod_embryo_line swings 13x in mean
  // luminance twice per loop at the default ±20°, and still 8.2x at ±6°, so no
  // amplitude fixes it.
  //
  // Note the `.webp` IS the animated loop (build_gallery_data uses it as the
  // tile's `still`, and has_media is `video or still`), so skipping only the
  // `.webm` would leave the pulsing in place — hence the static webp.
  noOrbitVideo?: boolean;
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
    // The README's gallery media is content-addressed and hosted, so its URLs
    // carry no demo id; media-manifest.json is what maps key -> id.
    const hasMediaManifest = fs.existsSync(MEDIA_MANIFEST_PATH);
    const mediaManifest = hasMediaManifest
      ? (JSON.parse(fs.readFileSync(MEDIA_MANIFEST_PATH, 'utf-8')) as Parameters<
          typeof mediaKeyIndex
        >[0])
      : { tiles: {}, base_url: undefined };
    const selection = resolveGalleryOnly(
      only,
      fs.readFileSync(README_PATH, 'utf-8'),
      demos.map((demo) => demo.id),
      mediaKeyIndex(mediaManifest),
      hasMediaManifest ? requireMediaBaseUrl(mediaManifest) : undefined
    );
    if (selection.unknownTokens.length > 0) {
      console.warn(
        `[gallery] GALLERY_ONLY did not match manifest ids: ${selection.unknownTokens.join(', ')}`
      );
    }
    demos = demos.filter((demo) => selection.wantedIds.has(demo.id));
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

async function waitForDataLoaded(
  page: any,
  {
    timeout = 60000,
    requireElements = true,
    expectedDimensionStep = null,
  }: {
    timeout?: number;
    requireElements?: boolean;
    expectedDimensionStep?: number[] | null;
  } = {}
): Promise<void> {
  // Geometry-agnostic: totalElements sums points + gsplats + lines + triangles,
  // so this works for all four geometry types and mixed scenes (totalPoints alone
  // stays 0 for a pure Lines demo like dipc_3d_genome, or a pure Mesh one).
  //
  // `requireElements = false` permits an empty initial slice while still waiting
  // for the authored dimension step. A scene whose index-zero slice is empty may
  // not have elements until its baked opening step or dimensionNav override runs.
  await page.waitForFunction(
    isGalleryDataReady,
    { requireElements, expectedDimensionStep },
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
 * The SIGNED world axis the camera's current up-vector most nearly points along.
 *
 * Read AFTER framing, so it reflects whatever pose is actually on screen — the
 * demo's baked `viewer_config` up when it has one, else three.js's (0,1,0).
 *
 * The page does the READ only; the classification (and its degenerate-input
 * fallback to +Y, the historical default) is `dominantSignedAxis` in
 * `./orbit-axis`, so it can be unit-tested outside a browser — this is the fix
 * for #1377 itself, and a sign slip in it ships a rolled animation.
 */
async function dominantCameraUpAxis(page: any): Promise<SignedUpAxis> {
  const up = await page.evaluate(() => {
    const cam = (window as any).__luxarDebug?.camera;
    // Return plain numbers, not the Vector3: only x/y/z survive the CDP hop.
    return cam?.up ? { x: cam.up.x, y: cam.up.y, z: cam.up.z } : null;
  });
  return dominantSignedAxis(up);
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
 * A retained PNG can be supplied when measuring a frame the harness already
 * captured; framing probes otherwise take a smaller JPEG screenshot.
 */
async function measureCoverage(page: any, pngShot?: Buffer): Promise<CoverageMeasurement> {
  const shot = pngShot ?? (await page.screenshot({ type: 'jpeg', quality: 60 }));
  const b64 = shot.toString('base64');
  return await page.evaluate(
    async ({
      b64img,
      imageType,
      litThreshold,
      loP,
      hiP,
    }: {
      b64img: string;
      imageType: 'jpeg' | 'png';
      litThreshold: number;
      loP: number;
      hiP: number;
    }) => {
      const blob = await (await fetch(`data:image/${imageType};base64,${b64img}`)).blob();
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
    {
      b64img: b64,
      imageType: pngShot ? 'png' : 'jpeg',
      litThreshold: LIT_THRESHOLD,
      loP: BBOX_LO_PCTILE,
      hiP: BBOX_HI_PCTILE,
    }
  );
}

/** Failure-tolerant final coverage diagnostic for an already-captured PNG still. */
async function measureCoverageOrNull(
  page: any,
  shot: Buffer,
  demoId: string
): Promise<CoverageMeasurement | null> {
  try {
    return await measureCoverage(page, shot);
  } catch (error) {
    console.warn(`[${demoId}] coverage measurement skipped:`, error);
    return null;
  }
}

/** Failure-tolerant final renderer-capacity diagnostic. */
async function measureDroppedElementsOrZero(page: any, demoId: string): Promise<number> {
  try {
    return await page.evaluate(() => {
      const value = (window as any).__luxarDebug?.getState?.()?.totalDroppedElements;
      return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : 0;
    });
  } catch (error) {
    console.warn(`[${demoId}] dropped-elements measurement skipped:`, error);
    return 0;
  }
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
      // be ~1M lit pixels and this runs up to ~50× per demo (3 p99 iterations +
      // 1 spread probe + MID_EXPOSURE_ITERS + up to 37 range-derived guard
      // iterations, less a probe reused by the next phase), so the array+sort
      // was needless memory churn. The histogram gives every percentile in
      // O(n), no growth.
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
      // histogram's spread, which tells a flat subject apart from an
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
 * Count the LIT pixels on the outermost row/column of an already-captured frame
 * — the measurement half of the crop check (the decision is
 * `./crop-policy::evaluateBorderLit`). Lit content on the frame edge means the
 * subject is running off frame, which `measureCoverage`'s percentile bbox
 * structurally cannot see.
 *
 * Takes the PNG buffer of a screenshot the harness already had to take (the
 * still, and the sampled orbit frames), so the check costs no extra screenshot;
 * it is decoded back inside the page exactly the way `measureLuminance` does
 * (createImageBitmap → 2D canvas at NATIVE resolution → getImageData).
 *
 * PNG, not the JPEG probes the coverage/exposure passes use, and that matters
 * here: JPEG ringing next to a bright edge spills energy into neighbouring
 * blocks, which can push a genuinely black border pixel over the 0.04 lit cutoff
 * and fake a crop. A lossless source makes a non-zero count mean something.
 *
 * Each border pixel is counted ONCE: the full top and bottom rows, then the
 * left/right columns excluding those corners — hence `borderPixels = 2w + 2h − 4`
 * (which over-counts a degenerate 1-px-wide/tall frame, so that case reports
 * `w·h` instead and the reported denominator always matches what was counted).
 *
 * The pixel loop is INLINE inside `page.evaluate` and cannot be extracted into an
 * imported helper for unit testing: the callback is serialized to the browser,
 * where this module's imports do not exist. Only the DECISION half
 * (`./crop-policy`) is testable, and that is deliberate.
 */
async function measureBorderLit(page: any, shot: Buffer, label: string): Promise<BorderSample> {
  const b64 = shot.toString('base64');
  const counted = await page.evaluate(
    async ({ b64img, litThreshold }: { b64img: string; litThreshold: number }) => {
      const blob = await (await fetch(`data:image/png;base64,${b64img}`)).blob();
      const bmp = await createImageBitmap(blob);
      const w = bmp.width;
      const h = bmp.height;
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(bmp, 0, 0, w, h);
      const data = ctx.getImageData(0, 0, w, h).data;
      const lit = (x: number, y: number): boolean => {
        const i = (y * w + x) * 4;
        const luma = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 255;
        return luma > litThreshold;
      };
      let borderLit = 0;
      for (let x = 0; x < w; x++) {
        if (lit(x, 0)) borderLit++;
        if (h > 1 && lit(x, h - 1)) borderLit++;
      }
      // Columns without the corners the rows already counted.
      for (let y = 1; y < h - 1; y++) {
        if (lit(0, y)) borderLit++;
        if (w > 1 && lit(w - 1, y)) borderLit++;
      }
      const borderPixels = w > 1 && h > 1 ? 2 * w + 2 * h - 4 : w * h;
      return { borderLit, borderPixels };
    },
    { b64img: b64, litThreshold: LIT_THRESHOLD }
  );
  return { label, borderLit: counted.borderLit, borderPixels: counted.borderPixels };
}

/**
 * `measureBorderLit` that never throws — a diagnostic must not cost the capture (a
 * mid-run Vite reload or a page crash rejects the in-page decode, and throwing
 * would lose the video conversion and the frames-dir cleanup). Says WHY the pose
 * was dropped, because the alternative is a silently short sample set: the log
 * line's `poses=<measured>/<attempted>` says one went missing, and this says
 * whether it was a dead page or a broken decode.
 */
async function measureBorderLitOrNull(
  page: any,
  shot: Buffer,
  label: string,
  demoId: string
): Promise<BorderSample | null> {
  return await measureBorderLit(page, shot, label).catch((e: unknown) => {
    console.warn(`[${demoId}] border-lit pose "${label}" not measured: ${e}`);
    return null;
  });
}

/**
 * Pick an exposure in three phases (the decision itself lives in
 * `./exposure-policy::computeAutoExposure`; this only wires it to the page):
 *   1. Percentile pass — converge so the lit foreground's high percentile hits
 *      TARGET_HI (bright but not clipped). Exposure is ~log-linear in
 *      luminance, so a few log2 corrections converge.
 *   2. Flat-subject pass — if the lit histogram's spread (p99 − p10) is under
 *      NARROW_SPREAD_MAX, the subject has no internal dynamic range and p99 is
 *      a meaningless anchor, so
 *      re-target the lit MEDIAN to TARGET_MID instead. An EMPIRICAL gate — see
 *      NARROW_SPREAD_MAX for the measured margins; when it fires, the capture
 *      log says so ("flat subject") so a sweep can spot a false positive.
 *   3. Guard — step exposure DOWN while EITHER >5% of the subject is blown to
 *      white OR the background is lifted to grey (frame p10 above near-black).
 *      Phase 1 (p99 target) over-boosts sparse/bloomy scenes into a grey wash;
 *      the background term is what pulls those back to a black background.
 * Returns the chosen stops, whether phase 2 fired, and whether phase 3 exhausted.
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
 * path that already works.
 *
 * Returns the number of frames written to `framesDir` plus border-lit samples for
 * the poses `borderSampleFrames` picks — a ~5° angular grid across the whole rock
 * — and, for a timelapse, the final timepoint. The measuring
 * happens HERE rather than in the caller because the per-frame screenshot buffers
 * are not retained past the frame loop (and the page is closed on return) — but
 * AFTER that loop rather than inside it: deferring costs nothing and keeps the
 * capture loop free of measurement work, so no in-page decode is interleaved with
 * the frames. (It is not a guarantee of an undisturbed orbit — a timelapse already
 * awaits a slice load every few frames, which dwarfs a decode.)
 *
 * `borderPosesAttempted` is how many poses were RETAINED for measurement, so the
 * caller can report a short sample set: a measurement that fails is skipped, and
 * without the attempted count a partial set would print an ordinary-looking
 * verdict computed from fewer poses than it claims.
 */
async function captureOrbitFrames(
  page: any,
  framesDir: string,
  demo?: DemoEntry
): Promise<{ frames: number; borderSamples: BorderSample[]; borderPosesAttempted: number }> {
  // Timelapse warm-up: for 4D time-series, PLAY the time dim a couple of cycles
  // first so the slice cache is primed (the built-in t+1 prefetch warms the next
  // timepoint), then we step it deterministically during capture. Returns the
  // discovered time-dimension index + range so we can pace the warm-up and steps.
  let tl: { timeIdx: number; min: number; max: number; lo: number; demoId: string } | null = null;
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
      const tlv = { ...found, lo, demoId: demo.id };
      tl = tlv;
      // Prime the slice cache: fully load each of the TL_STEPS distinct capture
      // timepoints once (much cheaper than the old 400-frame play warm-up), so
      // the capture pass can restore the complete ladder from cache.
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
        await waitForTimelapseSliceSettled(page, tlv.demoId, TL_SETTLE_MS);
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
  // subject aspect over one rock. FIVE manifest demos bake a non-Y up; only two
  // of them (the CT atlas and the mesh tile) declare `orbitUp`, so THREE were
  // left on the wrong axis because nothing told them they had to opt in
  // (#1377). TWO of those three were measurably shipping the roll — the
  // asteroids/cosmicflows pairs pinned in
  // `src/tests/unit/gallery-frame-similarity.test.ts`. The third is a 4D
  // timelapse whose camera was barely orbiting in the first place (#1383).
  //
  // Deriving it means a baked up is honoured by default and `orbitUp` becomes a
  // true override. A demo that bakes nothing gets three.js's default camera up,
  // (0,1,0) -> +y, so this is a no-op for every previously-correct demo.
  const derived = demo?.orbitUp ? null : await dominantCameraUpAxis(page);
  const upAxis: 'x' | 'y' | 'z' = demo?.orbitUp ?? derived?.axis ?? 'y';
  // An explicit `orbitUp` names a POSITIVE world axis — `positionForOrbitUp` has
  // already re-parked the camera on it with `up = +U` — so a sign can only come
  // from the derived case.
  const upSign: 1 | -1 = derived?.sign ?? 1;
  console.log(
    `[${demo?.id ?? 'orbit'}] orbitUp=${upSign < 0 ? '-' : '+'}${upAxis}` +
      (demo?.orbitUp ? ' (manifest)' : ' (derived from camera up)')
  );
  const ok = await page.evaluate(
    ({ up, sign }: { up: string; sign: number }) => {
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
      //
      // `sign` is U's direction (-1 for a baked up like (0,0,-1)). It reaches the
      // pose ONLY through `cam.up`: the out-of-plane offset is stored as
      // `compU = dot(off, U)` and re-applied as `compU·U`, so negating U leaves
      // every frame's camera POSITION bit-identical, and phi0/radius come from
      // A,B alone. It flips the rock's direction of travel, which a symmetric
      // ±amp sine covers either way.
      const U =
        up === 'x'
          ? { x: sign, y: 0, z: 0 }
          : up === 'z'
            ? { x: 0, y: 0, z: sign }
            : { x: 0, y: sign, z: 0 };
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
    },
    { up: upAxis, sign: upSign }
  );
  if (!ok) return { frames: 0, borderSamples: [], borderPosesAttempted: 0 };

  fs.mkdirSync(framesDir, { recursive: true });
  const ampRad = (ORBIT_AMPLITUDE_DEG * Math.PI) / 180;
  // Crop check: sample poses on a ~BORDER_SAMPLE_STEP_DEG grid across the WHOLE
  // ±ORBIT_AMPLITUDE_DEG rock (`borderSampleFrames`), not just its two endpoints —
  // the pose of greatest projected extent is often an intermediate angle, and an
  // endpoints-only check reads zero on exactly those subjects.
  // When the time dimension was found, also sample the LAST frame — the final
  // timepoint at (essentially) the BASE pose, since sin(2π·(N−1)/N) ≈ 0. The rock
  // grid spans the clip's first ~80% of the time range, so for a late-framed still
  // (`framePoint` 0.87 on gsplats_4d_celegans_tracking) the largest, final
  // timepoint would otherwise never be measured. Keyed on `tl`, not on
  // `demo.timelapse`, so a demo whose time dimension was NOT discovered does not
  // contribute a near-duplicate of the still under a misleading label.
  // Deduped and range-checked because GALLERY_ORBIT_FRAMES can be tiny in a
  // smoke run (at N=4 the timelapse last frame IS the −20° extreme; at N=1 the
  // out-of-range indices are filtered away).
  const borderMeasureAt = new Set(
    [
      ...borderSampleFrames(ORBIT_FRAMES, ORBIT_AMPLITUDE_DEG),
      ...(tl ? [ORBIT_FRAMES - 1] : []),
    ].filter((i) => i >= 0 && i < ORBIT_FRAMES)
  );
  // Buffers only — every measurement is deferred until after the frame loop.
  const borderShots: { label: string; shot: Buffer }[] = [];
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
        // `waitForUpdate()` resolves at the first commit, before progressive
        // refinement releases the loader's broad update lock. Wait through the
        // ladder drain; the explicit synchronous render below then draws it at
        // the orbit pose immediately before the screenshot.
        await waitForTimelapseSliceSettled(page, tl.demoId, TL_SETTLE_MS);
      }
    }
    await page.evaluate(
      async ({
        idx,
        n,
        amp,
        isTimelapse,
      }: {
        idx: number;
        n: number;
        amp: number;
        isTimelapse: boolean;
      }) => {
        const debug = (window as any).__luxarDebug;
        const o = (window as any).__orbit;
        // Guard: a mid-run reload would wipe these; skip the frame rather than
        // throw (the run then just has a duplicate frame, not a crash).
        if (!debug?.camera || !o) return;
        const cam = debug.camera;
        // TIMELAPSE ONLY: re-freeze the rAF loop every frame.
        //
        // Setup froze it once, but a timelapse slice load restarts it (the data
        // path reaches `startAnimation`), and a live loop runs
        // `controls.update()`, which re-derives the camera from the controls'
        // stored spherical state and snaps it back to the pre-orbit pose —
        // AFTER this evaluate returns, so re-applying the pose inside here does
        // not help (measured: identical output). Before this, the camera sat at
        // the baked pose in 6 of 8 frames and a 4D tile barely orbited at all
        // while its time dimension advanced (#1383).
        //
        // Gated on `isTimelapse` deliberately, even though the freeze measured
        // harmless on the heaviest non-timelapse tile (1.5M asteroids: output
        // byte-identical either way). The restart it defends against is a
        // timelapse-specific event, so there is no reason to change behaviour
        // for captures that were never broken, and one demo is thin evidence
        // for the rest. Non-timelapse captures are byte-identical to before —
        // verified against this file's pre-change version run back-to-back.
        if (isTimelapse) {
          debug.animationController?.stopAnimation?.();
          debug.controls?.setEnabled?.(false);
        }
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
      { idx: i, n: ORBIT_FRAMES, amp: ampRad, isTimelapse: tl !== null }
    );
    const shot = await page.screenshot({
      path: path.join(framesDir, `f${String(i).padStart(4, '0')}.png`),
      type: 'png',
    });
    if (borderMeasureAt.has(i)) {
      // Retain the buffer and label the pose by its actual rock angle (rounding
      // for a tiny frame count can land slightly off the ±amplitude extreme),
      // naming the timelapse end-of-range frame explicitly.
      const deg = ORBIT_AMPLITUDE_DEG * Math.sin((i / ORBIT_FRAMES) * Math.PI * 2);
      const pose = `rock ${deg >= 0 ? '+' : ''}${deg.toFixed(0)}°`;
      borderShots.push({
        label: tl && i === ORBIT_FRAMES - 1 ? `${pose}, last timepoint` : pose,
        shot,
      });
    }
  }
  // Capture loop done — now decode the retained frames. A failed decode skips
  // that pose (evaluateBorderLit copes with fewer, or zero, samples) and is
  // reported by measureBorderLitOrNull; the returned attempted count lets the
  // caller flag the short set.
  const borderSamples: BorderSample[] = [];
  for (const { label, shot } of borderShots) {
    const sample = await measureBorderLitOrNull(page, shot, label, demo?.id ?? 'orbit');
    if (sample) borderSamples.push(sample);
  }
  return { frames: ORBIT_FRAMES, borderSamples, borderPosesAttempted: borderShots.length };
}

/**
 * Warn when the still and the FIRST orbit frame disagree.
 *
 * They are the same nominal pose — the rock is `phi0 + amp*sin(0)` at frame 0 —
 * so they should render near-identically. When they do not, the orbit is showing
 * the subject from somewhere the poster never does, and since the README embeds
 * the ANIMATION while reviewers usually look at the still, that divergence ships
 * unnoticed. It already did: five demos bake a non-Y camera up, three of them
 * did not declare `orbitUp`, and two of those three measurably shipped
 * animations rolled ~90 deg from their own posters (#1377). Nothing in this
 * harness compared the two.
 *
 * Normalised cross-correlation on a greyscale downscale — deliberately not
 * SSIM, which is punishing on high-frequency filamentary subjects (a
 * correctly-matched cosmic-web tile scores 0.57 while correlating at 0.98) and
 * would cry wolf. The maths, the size and the threshold live in
 * `frame-similarity.ts`, where they are unit-tested.
 *
 * WARNS rather than fails. The still and frame 0 can differ for reasons that are
 * not a roll — a progressively-streamed scene keeps filling in between the two
 * captures, and the in-page decode could go missing — and neither should abandon
 * a 61-tile media run that takes hours. So this is a "look at this" signal, not
 * a correctness gate.
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
  try {
    // Read inside the try: an exists/read race or an unreadable file must warn
    // like any other failure here, not reject and take the whole capture down.
    const [a, b] = [stillPath, frameZeroPath].map((f) => fs.readFileSync(f).toString('base64'));
    // The page does DECODING only — it is the one place with an image decoder —
    // and hands back plain integer greyscale buffers. The comparison itself runs
    // in Node against `frame-similarity.ts`, so the arithmetic is unit-tested
    // rather than trapped inside a `page.evaluate` string.
    const [ga, gb] = await page.evaluate(
      async ([s, f, size]: [string, string, number]) => {
        const grey = async (b64: string) => {
          const blob = await (await fetch('data:image/png;base64,' + b64)).blob();
          const img = await createImageBitmap(blob);
          const c = new OffscreenCanvas(size, size);
          const ctx = c.getContext('2d')!;
          ctx.drawImage(img, 0, 0, size, size);
          const d = ctx.getImageData(0, 0, size, size).data;
          const out: number[] = new Array(size * size);
          for (let i = 0, j = 0; i < d.length; i += 4, j++) {
            // Kept in step with `luma8` in frame-similarity.ts — this side
            // cannot import it (the body is serialised into the page).
            out[j] = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
          }
          return out;
        };
        return [await grey(s), await grey(f)];
      },
      [a, b, COMPARE_SIZE]
    );
    const corr = normalizedCrossCorrelation(ga, gb);
    if (corr < DISAGREEMENT_THRESHOLD) {
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

/**
 * Single-frame WebP from the curated still, for demos that opt out of an orbit
 * clip. The animated `.webp` is what the gallery uses as a tile's `still`, so a
 * subject that should not move needs a STATIC one here — dropping only the
 * `.webm` would leave the animation in place.
 */
function convertFrameToStaticWebp(input: string, output: string): void {
  execSync(
    `ffmpeg -y -i "${input}" ` +
      `-vf "scale=${WEBP_WIDTH}:-1" -vcodec libwebp -lossless 0 ` +
      `-compression_level 6 -q:v ${WEBP_QUALITY} -frames:v 1 "${output}"`,
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

function statGalleryMedia(demoId: string, filePath: string): GalleryMediaFile | null {
  const stats = fs.statSync(filePath, { throwIfNoEntry: false });
  if (!stats) return null;
  return {
    demoId,
    fileName: path.basename(filePath),
    sizeBytes: stats.size,
  };
}

function recordGalleryMedia(demoId: string, filePath: string): GalleryMediaFile {
  const media = statGalleryMedia(demoId, filePath);
  if (!media) throw new Error(`[${demoId}] encoded media is missing: ${filePath}`);
  try {
    const { warning } = checkGalleryMediaSize(media);
    if (warning) console.warn(warning);
  } catch (error) {
    fs.rmSync(filePath, { force: true });
    throw error;
  }
  capturedMedia.push(media);
  return media;
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
    const bakedDimensionStep = await readBakedDimensionStep(dataUrl);
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
    // An nD demo may be empty at index zero. Permit that until the baked opening
    // step (or a subsequent dimensionNav override) has had a chance to populate it.
    const needsNav = Boolean(demo.dimensionNav);
    const opensAwayFromDefault = hasNonZeroDimensionStep(bakedDimensionStep);
    const initialLoadOptions = {
      requireElements: !(needsNav || opensAwayFromDefault),
      expectedDimensionStep: bakedDimensionStep,
    };
    await waitForDataLoaded(page, initialLoadOptions);
    // Settle: a Vite "re-optimizing deps" full reload can fire right after first
    // paint; wait out any in-flight reload and re-confirm the scene is loaded.
    await page.waitForTimeout(1500);
    await waitForDataLoaded(page, initialLoadOptions);
    console.log(`[${demo.id}] data loaded`);

    if (demo.dimensionNav) {
      await navigateDimension(page, demo.dimensionNav.key, demo.dimensionNav.steps);
      // NOW the scene must have content: the navigation is what fills it, and a
      // still of an empty canvas is worse than a failed capture.
      await waitForDataLoaded(page);
      console.log(
        `[${demo.id}] dimensionNav key=${demo.dimensionNav.key} steps=${demo.dimensionNav.steps}`
      );
    }
    // Timelapse: frame the STILL at a content-rich timepoint (a developmental
    // series is nearly empty at t0). The orbit video still plays the full range.
    if (demo.timelapse) {
      const frac = demo.timelapse.framePoint ?? 0.85;
      await jumpTimeDimToFrac(page, frac);
      await waitForTimelapseSliceSettled(page, demo.id, TL_SETTLE_MS);
      console.log(`[${demo.id}] timelapse still framePoint=${frac}`);
    }

    // Camera fitting and exposure must measure a fixed radius, not a live dolly phase.
    await page.evaluate(() => (window as any).__luxarDebug?.controls?.setAutoDolly?.(false));
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
      const { stops, flatSubject, guardExhausted } = await autoExpose(page);
      const how = `auto${flatSubject ? ', flat subject' : ''}${guardExhausted ? ', guard exhausted' : ''}`;
      console.log(`[${demo.id}] exposure=${stops.toFixed(2)} stops (${how})`);
      if (guardExhausted) {
        console.warn(
          `[${demo.id}] exposure guard exhausted at ${stops.toFixed(2)} stops — still over the clip/background limit at the floor`
        );
      }
    } else {
      console.log(`[${demo.id}] exposure=baked (autoExpose off)`);
    }

    // Let the forced-finest LOD stream/commit and a couple of frames render
    // before the still (fine gsplat/point levels load progressively).
    await page.waitForTimeout(2500);

    // Still.
    const pngPath = path.join(OUTPUT_DIR, `${demo.id}.png`);
    // Keep the buffer as well as writing the file: the crop check measures this
    // very frame, so it costs no extra screenshot.
    const stillShot = await page.screenshot({ path: pngPath, type: 'png' });
    const pngMedia = recordGalleryMedia(demo.id, pngPath);
    console.log(
      `[${demo.id}] still → ${path.basename(pngPath)} (${formatMediaSize(pngMedia.sizeBytes)})`
    );

    const noOrbit = demo.noOrbitVideo === true;
    const webpPath = path.join(OUTPUT_DIR, `${demo.id}.webp`);
    const webmPathOut = path.join(OUTPUT_DIR, `${demo.id}.webm`);
    if (noOrbit) {
      try {
        // Subject should not be rocked (see noOrbitVideo in the manifest):
        // a static tile rather than a clip that pulses as the view changes.
        convertFrameToStaticWebp(pngPath, webpPath);
        const webpMedia = recordGalleryMedia(demo.id, webpPath);
        console.log(
          `[${demo.id}] webp → ${path.basename(webpPath)} (${formatMediaSize(webpMedia.sizeBytes)})`
        );
      } catch (e) {
        fs.rmSync(webpPath, { force: true });
        console.error(`[${demo.id}] webp failed:`, e);
      }
      skipGalleryMediaWhenRequested(noOrbit, webmPathOut);
      console.log(`[${demo.id}] noOrbitVideo — static tile, skipping webm`);
    }

    // Orbit: capture explicit per-angle frames (reliable in headless), then
    // assemble the WebM master + animated WebP.
    const framesDir = path.join(OUTPUT_DIR, `_frames_${demo.id}`);
    const {
      frames: n,
      borderSamples,
      borderPosesAttempted,
    } = await captureOrbitFrames(page, framesDir, demo);
    // Only when this run actually produced frames — framesDir is not cleaned
    // between runs, so on a failed capture f0000.png can be a stale leftover and
    // comparing the fresh still against it would warn about nothing. Must happen
    // before page.close() (the check decodes in the page), hence the guard here
    // rather than after the n === 0 bail-out below.
    if (n > 0) {
      await warnIfStillDisagreesWithOrbit(page, pngPath, path.join(framesDir, 'f0000.png'), demo);
    }
    // Measure the still LAST (but while the page is still open): decoding a
    // retained buffer doesn't care where the camera ended up, and doing it here
    // keeps both still measurements from inserting seconds between the still
    // screenshot and the orbit's rAF freeze — progressive LOD is still streaming
    // there, so a diagnostic must not change what the media pipeline captures.
    // Failure-tolerant for the same reason as the orbit samples.
    const coverageMeasurement = await measureCoverageOrNull(page, stillShot, demo.id);
    const stillSample = await measureBorderLitOrNull(page, stillShot, 'still', demo.id);
    const totalDroppedElements = await measureDroppedElementsOrZero(page, demo.id);
    await page.close();

    console.log(
      `[${demo.id}] ${formatGalleryCaptureMetrics(coverageMeasurement, totalDroppedElements)}`
    );
    const droppedWarning = galleryDroppedElementsWarning(demo.id, totalDroppedElements);
    if (droppedWarning) console.warn(droppedWarning);
    if (coverageMeasurement) {
      const underfill = evaluateUnderfill({ demoId: demo.id, measurement: coverageMeasurement });
      if (underfill.underfilled) console.warn(underfill.message);
    }

    // Crop check (before the orbit-failure return — the still sample alone is
    // worth reporting). ALWAYS log the count: it is a per-tile regression signal
    // that should not grow between captures, and a non-zero value is common
    // enough that only the number, not the boolean, is informative. Warn (never
    // fail) when it clears the floor — see ./crop-policy.
    const framing: CropFraming = {
      fillTarget: demo.fillTarget ?? FILL_TARGET,
      zoom: demo.zoom,
      distance: demo.distance,
      autoFrame: demo.autoFrame,
    };
    // Keep every measurement in the log, but judge only poses that reach the
    // published media. Still first, so a tie names the easiest pose to reproduce.
    const samples = [...(stillSample ? [stillSample] : []), ...borderSamples];
    const verdictSamples = selectCropVerdictSamples(stillSample, borderSamples, noOrbit);
    const verdict = evaluateBorderLit({ demoId: demo.id, samples: verdictSamples, framing });
    // ALWAYS report measured/attempted poses: the verdict is the worst of whatever
    // could be measured, so a skipped pose (each one warned about above) would
    // otherwise print an ordinary-looking count that silently misses a crop
    // confined to the dropped pose.
    const posesAttempted = 1 + borderPosesAttempted; // the still, plus the orbit poses
    const poses = `poses=${samples.length}/${posesAttempted}`;
    if (verdict.worst) {
      const w = verdict.worst;
      console.log(
        `[${demo.id}] border-lit=${w.borderLit}/${w.borderPixels} ` +
          `(${borderLitPercent(w).toFixed(1)}%) worst=${w.label} ${poses}`
      );
    } else {
      // Say so loudly: with every measurement swallowed, a broken in-page decode
      // would otherwise just stop printing the count for every demo — silence
      // indistinguishable from the check not existing.
      console.log(`[${demo.id}] border-lit=unmeasured (${poses})`);
    }
    if (verdict.cropped) console.warn(verdict.message);

    if (n === 0) {
      console.error(`[${demo.id}] orbit capture failed (no camera/controls)`);
      return;
    }
    console.log(`[${demo.id}] captured ${n} orbit frames`);

    try {
      if (!noOrbit) {
        let webpEncoded = false;
        try {
          convertFramesToWebp(framesDir, webpPath); // README inline (GitHub)
          webpEncoded = true;
        } catch (e) {
          fs.rmSync(webpPath, { force: true });
          console.error(`[${demo.id}] webp failed:`, e);
        }
        if (webpEncoded) {
          const webpMedia = recordGalleryMedia(demo.id, webpPath);
          console.log(
            `[${demo.id}] webp → ${path.basename(webpPath)} (${formatMediaSize(webpMedia.sizeBytes)})`
          );
        }
        let webmEncoded = false;
        try {
          convertFramesToWebm(framesDir, webmPathOut); // full-quality master
          webmEncoded = true;
        } catch (e) {
          fs.rmSync(webmPathOut, { force: true });
          console.error(`[${demo.id}] webm failed:`, e);
        }
        if (webmEncoded) {
          const webmMedia = recordGalleryMedia(demo.id, webmPathOut);
          console.log(
            `[${demo.id}] webm → ${path.basename(webmPathOut)} (${formatMediaSize(webmMedia.sizeBytes)})`
          );
        }
      }
    } finally {
      fs.rmSync(framesDir, { recursive: true, force: true }); // clean up frames
    }
  });
}

test('Gallery summary', async () => {
  console.log('\n======== Gallery capture complete ========');
  console.log(`Output: ${OUTPUT_DIR}`);
  const checkedMediaFiles = new Set(capturedMedia.map((media) => media.fileName));
  const allMedia: GalleryMediaFile[] = [];
  const uncheckedMedia: GalleryMediaFile[] = [];
  for (const demo of DEMOS) {
    const png = statGalleryMedia(demo.id, path.join(OUTPUT_DIR, `${demo.id}.png`));
    const webp = statGalleryMedia(demo.id, path.join(OUTPUT_DIR, `${demo.id}.webp`));
    const webm = statGalleryMedia(demo.id, path.join(OUTPUT_DIR, `${demo.id}.webm`));
    const demoMedia = [png, webp, webm].filter(
      (media): media is GalleryMediaFile => media !== null
    );
    allMedia.push(...demoMedia);
    for (const media of demoMedia) {
      if (!checkedMediaFiles.has(media.fileName)) uncheckedMedia.push(media);
    }
    console.log(
      `  ${png ? '[png]' : '[---]'} ${webp ? '[webp]' : '[----]'} ${webm ? '[webm]' : '[----]'} ${demo.id}`
    );
  }
  const mediaIssues = collectGalleryMediaSizeIssues(uncheckedMedia);
  for (const warning of mediaIssues.warnings) console.warn(warning);
  const mediaSummary = summarizeGalleryMedia(allMedia);
  console.log(mediaSummary.totalLine);
  if (mediaSummary.largestLines.length > 0) {
    console.log('Largest media:');
    for (const line of mediaSummary.largestLines) console.log(line);
  }
  console.log('');
  if (mediaIssues.errors.length > 0) throw new Error(mediaIssues.errors.join('\n'));
});
