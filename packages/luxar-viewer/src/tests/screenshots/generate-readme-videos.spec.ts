/**
 * README Video Generator
 *
 * Generates animated videos/GIFs of Luxar demos for README documentation.
 * Uses Playwright video recording with programmatic camera rotation.
 *
 * Usage:
 *   pnpm readme-videos
 *
 * Prerequisites:
 *   - Run demo generators first: make run-demos
 *   - ffmpeg installed for format conversion
 */

import { test } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';
import { execSync } from 'child_process';

// Output directory for README videos
const OUTPUT_DIR = path.resolve(__dirname, '../../../../../docs/images/readme');

// Server URLs
const DATA_SERVER = 'http://localhost:9876';
const VIEWER_URL = 'http://localhost:5173';

// Video settings
const VIDEO_DURATION_MS = 7000; // 7 seconds
const ROTATION_SPEED = 0.015; // Radians per frame
const FRAME_INTERVAL_MS = 50; // 20 FPS for rotation updates

// Demo configurations for videos
interface VideoDemoConfig {
  name: string;
  datasetPath: string;
  filename: string; // Base filename without extension
  zoomClicks?: number;
  exposure?: number; // Log2 stops
  rotationAxis?: 'y' | 'x' | 'both';
}

const VIDEO_DEMOS: VideoDemoConfig[] = [
  {
    name: 'Lorenz Attractor',
    datasetPath: 'datasets/demos/lorenz.luxar.zarr',
    filename: 'lorenz-demo',
    zoomClicks: 3,
    exposure: 1.3, // ~2.5x
    rotationAxis: 'y',
  },
  {
    name: 'Spiral Galaxy',
    datasetPath: 'datasets/demos/spiral_galaxy.luxar.zarr',
    filename: 'spiral-galaxy-demo',
    zoomClicks: 4,
    exposure: 1.6, // ~3x
    rotationAxis: 'y',
  },
  {
    name: 'Rainbow Sphere',
    datasetPath: 'datasets/demos/rainbow_sphere.luxar.zarr',
    filename: 'rainbow-sphere-demo',
    zoomClicks: 17,
    exposure: 4.6, // ~25x
    rotationAxis: 'y',
  },
];

/**
 * Wait for Luxar to initialize
 */
async function waitForLuxarReady(page: any, timeout = 60000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      return debug && debug.getState && debug.getState().initialized;
    },
    { timeout }
  );
}

/**
 * Wait for data to load
 */
async function waitForDataLoaded(page: any, timeout = 45000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      if (!debug || !debug.getState) return false;
      const state = debug.getState();
      return state && !state.isLoading && state.totalPoints > 0;
    },
    { timeout }
  );
}

/**
 * Center camera on scene
 */
async function centerCamera(page: any): Promise<void> {
  await page.keyboard.press('f');
  await page.waitForTimeout(800);
}

/**
 * Zoom using scroll wheel
 */
async function zoom(page: any, clicks: number): Promise<void> {
  if (clicks === 0) return;

  const canvas = await page.$('canvas');
  if (!canvas) return;

  const box = await canvas.boundingBox();
  if (!box) return;

  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;

  const absClicks = Math.abs(clicks);
  const direction = clicks > 0 ? 100 : -100;

  for (let i = 0; i < absClicks; i++) {
    await page.mouse.move(x, y);
    await page.mouse.wheel(0, direction);
    await page.waitForTimeout(100);
  }
  await page.waitForTimeout(300);
}

/**
 * Set exposure (log2 stops)
 */
async function setExposure(page: any, exposureStops: number): Promise<void> {
  await page.evaluate((stops: number) => {
    const debug = (window as any).__luxarDebug;
    const sceneManager = debug?.app?.sceneManager;
    if (sceneManager?.updateExposure) {
      sceneManager.updateExposure(stops);
    }
    if (debug?.renderOnce) {
      debug.renderOnce();
    }
  }, exposureStops);
  await page.waitForTimeout(500);
}

/**
 * Hide UI panels
 */
async function hideUI(page: any): Promise<void> {
  await page.keyboard.press('Escape');
  await page.waitForTimeout(100);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
}

/**
 * Rotate camera around the scene
 */
async function rotateCamera(page: any, durationMs: number, rotationSpeed: number): Promise<void> {
  const startTime = Date.now();
  let angle = 0;

  while (Date.now() - startTime < durationMs) {
    angle += rotationSpeed;

    await page.evaluate((theta: number) => {
      const debug = (window as any).__luxarDebug;
      const controls = debug?.controls;

      if (controls && controls.object) {
        // Get current camera position relative to target
        const camera = controls.object;
        const target = controls.target;

        // Calculate distance from target
        const dx = camera.position.x - target.x;
        const dz = camera.position.z - target.z;
        const distance = Math.sqrt(dx * dx + dz * dz);

        // Rotate around Y axis
        camera.position.x = target.x + distance * Math.cos(theta);
        camera.position.z = target.z + distance * Math.sin(theta);

        // Update controls
        controls.update();
      }

      // Trigger render
      if (debug?.renderOnce) {
        debug.renderOnce();
      }
    }, angle);

    await page.waitForTimeout(FRAME_INTERVAL_MS);
  }
}

/**
 * Convert WebM to GIF using ffmpeg
 */
function convertToGif(inputPath: string, outputPath: string): void {
  // Two-pass approach for better quality GIF
  const paletteCmd = `ffmpeg -y -i "${inputPath}" -vf "fps=15,scale=800:-1:flags=lanczos,palettegen=stats_mode=diff" -y /tmp/palette.png`;
  const gifCmd = `ffmpeg -y -i "${inputPath}" -i /tmp/palette.png -lavfi "fps=15,scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -y "${outputPath}"`;

  console.log('Generating palette...');
  execSync(paletteCmd, { stdio: 'pipe' });

  console.log('Creating GIF...');
  execSync(gifCmd, { stdio: 'pipe' });
}

/**
 * Convert WebM to WebP using ffmpeg
 */
function convertToWebp(inputPath: string, outputPath: string): void {
  const cmd = `ffmpeg -y -i "${inputPath}" -vf "fps=15,scale=800:-1" -vcodec libwebp -lossless 0 -compression_level 6 -q:v 70 -loop 0 -preset default -an -vsync 0 "${outputPath}"`;
  execSync(cmd, { stdio: 'pipe' });
}

// Ensure output directory exists
test.beforeAll(async () => {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  console.log(`\n[README Videos] Output directory: ${OUTPUT_DIR}\n`);
});

// Configure test to record video
test.use({
  video: {
    mode: 'on',
    size: { width: 1280, height: 720 },
  },
});

// Generate videos for each demo
for (const demo of VIDEO_DEMOS) {
  test(`Generate ${demo.name} video`, async ({ page }, _testInfo) => {
    const dataUrl = `${DATA_SERVER}/${demo.datasetPath}`;
    const viewerUrl = `${VIEWER_URL}/?src=${dataUrl}&debug`;

    console.log(`[${demo.name}] Loading from ${dataUrl}`);

    // Navigate to viewer
    await page.goto(viewerUrl, { waitUntil: 'networkidle' });

    // Wait for initialization
    await waitForLuxarReady(page);
    console.log(`[${demo.name}] Viewer initialized`);

    // Wait for data
    await waitForDataLoaded(page);
    console.log(`[${demo.name}] Data loaded`);

    // Center camera
    await centerCamera(page);
    console.log(`[${demo.name}] Camera centered`);

    // Zoom if needed
    if (demo.zoomClicks && demo.zoomClicks !== 0) {
      await zoom(page, demo.zoomClicks);
      const direction = demo.zoomClicks > 0 ? 'out' : 'in';
      console.log(`[${demo.name}] Zoomed ${direction}`);
    }

    // Hide UI
    await hideUI(page);

    // Set exposure
    if (demo.exposure) {
      await setExposure(page, demo.exposure);
      console.log(`[${demo.name}] Exposure set to ${demo.exposure} stops`);
    }

    // Wait a moment for everything to settle
    await page.waitForTimeout(500);

    // Rotate camera for the video duration
    console.log(`[${demo.name}] Recording rotation...`);
    await rotateCamera(page, VIDEO_DURATION_MS, ROTATION_SPEED);

    // Get the video object before closing the page
    const videoObj = page.video();

    // Close page to finalize video recording
    await page.close();

    // Get the recorded video path - need to wait for it to be available
    const webmPath = videoObj ? await videoObj.path() : null;

    if (webmPath) {
      // Wait a moment for the file to be fully written
      await new Promise((resolve) => setTimeout(resolve, 500));
      const gifPath = path.join(OUTPUT_DIR, `${demo.filename}.gif`);
      const webpPath = path.join(OUTPUT_DIR, `${demo.filename}.webp`);

      console.log(`[${demo.name}] Converting to GIF...`);
      try {
        convertToGif(webmPath, gifPath);
        console.log(`[${demo.name}] GIF saved: ${gifPath}`);
      } catch (e) {
        console.error(`[${demo.name}] GIF conversion failed:`, e);
      }

      console.log(`[${demo.name}] Converting to WebP...`);
      try {
        convertToWebp(webmPath, webpPath);
        console.log(`[${demo.name}] WebP saved: ${webpPath}`);
      } catch (e) {
        console.error(`[${demo.name}] WebP conversion failed:`, e);
      }
    } else {
      console.error(`[${demo.name}] No video recorded - check video configuration`);
    }
  });
}

// Summary
test('Video Summary', async () => {
  console.log('\n========================================');
  console.log('README Video Generation Complete');
  console.log('========================================');
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log('\nVideos:');
  for (const demo of VIDEO_DEMOS) {
    const gifExists = fs.existsSync(path.join(OUTPUT_DIR, `${demo.filename}.gif`));
    const webpExists = fs.existsSync(path.join(OUTPUT_DIR, `${demo.filename}.webp`));
    console.log(
      `  ${gifExists ? '[GIF]' : '[---]'} ${webpExists ? '[WebP]' : '[----]'} ${demo.filename}`
    );
  }
  console.log('');
});
