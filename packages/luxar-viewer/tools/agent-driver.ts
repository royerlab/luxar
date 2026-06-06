/**
 * Luxar Agent Driver - Browser Automation for AI-Assisted Development
 *
 * This script allows Claude Code (or any AI agent) to "see" and "debug"
 * the Luxar viewer without a physical monitor. It captures console logs,
 * inspects Three.js scene state, and takes screenshots.
 *
 * Usage:
 *   tsx tools/agent-driver.ts
 *   tsx tools/agent-driver.ts --url http://localhost:5173/?src=/data/demo.zarr
 *   tsx tools/agent-driver.ts --headless=false  # Watch in real browser
 *
 * Output:
 *   - Terminal: All browser console logs, errors, and state dumps
 *   - debug-view.png: Screenshot of current state
 *   - error-state.png: Screenshot on failure
 */

import { chromium, type Browser, type Page } from '@playwright/test';

// ============================================================================
// CONFIGURATION
// ============================================================================

interface DriverConfig {
  url: string;
  headless: boolean;
  waitTime: number;
  screenshotPath: string;
  errorScreenshotPath: string;
}

function parseArgs(): DriverConfig {
  const args = process.argv.slice(2);
  const config: DriverConfig = {
    url: process.env.APP_URL || 'http://localhost:5173/?debug',
    headless: true,
    waitTime: 5000,  // Time to wait for scene initialization (ms)
    screenshotPath: 'test-results/debug/debug-view.png',
    errorScreenshotPath: 'test-results/debug/error-state.png'
  };

  for (const arg of args) {
    if (arg.startsWith('--url=')) {
      config.url = arg.substring(6);
    } else if (arg.startsWith('--headless=')) {
      config.headless = arg.substring(11) === 'true';
    } else if (arg.startsWith('--wait=')) {
      config.waitTime = parseInt(arg.substring(7), 10);
    }
  }

  // Ensure debug mode is enabled
  if (!config.url.includes('debug')) {
    config.url += (config.url.includes('?') ? '&' : '?') + 'debug';
  }

  return config;
}

// ============================================================================
// BROWSER EVENT HANDLERS
// ============================================================================

function setupConsoleLogging(page: Page): void {
  // Mirror browser console logs to terminal
  // This is CRITICAL - it allows Claude to see what's happening in the browser
  page.on('console', msg => {
    const type = msg.type().toUpperCase();
    const text = msg.text();

    // Color code by type for better readability
    const prefix = `[BROWSER-CONSOLE-${type}]`;

    switch (msg.type()) {
      case 'error':
        console.error(`\x1b[31m${prefix}\x1b[0m ${text}`);
        break;
      case 'warning':
        console.warn(`\x1b[33m${prefix}\x1b[0m ${text}`);
        break;
      case 'info':
        console.info(`\x1b[36m${prefix}\x1b[0m ${text}`);
        break;
      default:
        console.log(`${prefix} ${text}`);
    }
  });

  // Catch uncaught errors (JavaScript runtime errors)
  page.on('pageerror', err => {
    console.error('\x1b[31m[BROWSER-CRASH]\x1b[0m', err.message);
    if (err.stack) {
      console.error('  Stack:', err.stack);
    }
  });

  // Catch failed network requests
  page.on('requestfailed', request => {
    const failure = request.failure();
    console.error(
      `\x1b[31m[NETWORK-FAIL]\x1b[0m ${request.url()}`,
      failure ? `- ${failure.errorText}` : ''
    );
  });

  // Log successful navigation
  page.on('response', response => {
    const status = response.status();
    if (status >= 400) {
      console.error(
        `\x1b[31m[HTTP-ERROR]\x1b[0m ${status} ${response.url()}`
      );
    }
  });
}

// ============================================================================
// LUXAR STATE INSPECTION
// ============================================================================

async function extractLuxarState(page: Page): Promise<object> {
  return await page.evaluate(() => {
    // Access the global debug object that Luxar exposes
    const debug = (window as any).__luxarDebug;

    if (!debug) {
      return {
        error: 'Debug mode not enabled',
        hint: 'Add ?debug to URL to enable debug mode'
      };
    }

    // Extract scene information
    const sceneInfo = debug.scene ? {
      totalChildren: debug.scene.children.length,
      pointClouds: debug.scene.children.filter((obj: any) => obj.type === 'Points').length,
      groups: debug.scene.children.filter((obj: any) => obj.type === 'Group').length,
      pointCloudDetails: debug.scene.children
        .filter((obj: any) => obj.type === 'Points')
        .map((obj: any) => ({
          name: obj.name || 'unnamed',
          pointCount: obj.geometry?.attributes?.position?.count || 0,
          visible: obj.visible,
          hasColors: !!obj.geometry?.attributes?.color,
          hasRadii: !!obj.geometry?.attributes?.radius,
        }))
    } : null;

    // Extract camera information
    const cameraInfo = debug.camera ? {
      position: {
        x: debug.camera.position.x,
        y: debug.camera.position.y,
        z: debug.camera.position.z
      },
      fov: debug.camera.fov,
      near: debug.camera.near,
      far: debug.camera.far,
      aspect: debug.camera.aspect
    } : null;

    // Extract renderer information
    const rendererInfo = debug.renderer ? (() => {
      try {
        // Create a proper Vector2-like object for getSize
        const sizeVec = new (window as any).THREE.Vector2();
        debug.renderer.getSize(sizeVec);

        return {
          pixelRatio: debug.renderer.getPixelRatio(),
          size: { width: sizeVec.x, height: sizeVec.y },
          capabilities: {
            maxTextureSize: debug.renderer.capabilities.maxTextureSize,
            maxTextures: debug.renderer.capabilities.maxTextures,
          }
        };
      } catch (error) {
        return {
          pixelRatio: debug.renderer.getPixelRatio(),
          size: { width: 0, height: 0 },
          error: String(error)
        };
      }
    })() : null;

    // Extract performance stats (if available)
    const performanceInfo = debug.getState ? debug.getState() : {
      note: 'Performance stats not available - getState() not implemented'
    };

    return {
      timestamp: new Date().toISOString(),
      scene: sceneInfo,
      camera: cameraInfo,
      renderer: rendererInfo,
      performance: performanceInfo,
    };
  });
}

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main() {
  const config = parseArgs();

  console.log('\n' + '='.repeat(80));
  console.log('🤖 Luxar Agent Driver - Browser Automation for AI Development');
  console.log('='.repeat(80));
  console.log(`\nConfiguration:`);
  console.log(`  URL: ${config.url}`);
  console.log(`  Headless: ${config.headless}`);
  console.log(`  Wait Time: ${config.waitTime}ms`);
  console.log(`  Screenshot: ${config.screenshotPath}`);
  console.log('\n' + '-'.repeat(80) + '\n');

  let browser: Browser | null = null;
  let page: Page | null = null;

  try {
    // Launch browser with WebGL/GPU acceleration flags
    console.log('[AGENT] Launching Chromium with GPU acceleration...');
    browser = await chromium.launch({
      headless: config.headless,
      args: [
        '--use-gl=egl',                          // Force GPU acceleration
        '--ignore-gpu-blocklist',                // Ignore GPU blacklist
        '--enable-webgl-developer-extensions',   // Enable WebGL extensions
        '--enable-webgl-draft-extensions',       // Enable draft extensions
        '--disable-web-security',                // Allow CORS for local testing
      ]
    });

    page = await browser.newPage();

    // Setup event handlers
    setupConsoleLogging(page);

    // Navigate to app
    console.log(`[AGENT] Navigating to ${config.url}...`);
    await page.goto(config.url, {
      waitUntil: 'networkidle',  // Wait for all network requests to finish
      timeout: 60000              // 60 second timeout
    });

    console.log('[AGENT] Page loaded, waiting for Three.js scene initialization...');
    await page.waitForTimeout(config.waitTime);

    // Extract and log Luxar state
    console.log('\n' + '='.repeat(80));
    console.log('📊 LUXAR STATE INSPECTION');
    console.log('='.repeat(80) + '\n');

    const luxarState = await extractLuxarState(page);
    console.log(JSON.stringify(luxarState, null, 2));

    console.log('\n' + '-'.repeat(80) + '\n');

    // Take screenshot
    console.log(`[AGENT] Taking screenshot...`);
    await page.screenshot({
      path: config.screenshotPath,
      fullPage: false,  // Canvas doesn't scroll
      timeout: 10000
    });
    console.log(`[AGENT] ✅ Screenshot saved to: ${config.screenshotPath}`);

    // Additional checks
    const hasErrors = await page.evaluate(() => {
      return (window as any).__luxarDebug?.errors?.length > 0;
    });

    if (hasErrors) {
      console.warn('\n⚠️  WARNING: Errors detected in Luxar state!');
    }

    console.log('\n' + '='.repeat(80));
    console.log('✅ Agent driver completed successfully');
    console.log('='.repeat(80) + '\n');

  } catch (error) {
    console.error('\n' + '='.repeat(80));
    console.error('❌ AGENT DRIVER ERROR');
    console.error('='.repeat(80));
    console.error('\nExecution failed:', error);

    // Take error screenshot if page exists
    if (page) {
      try {
        await page.screenshot({
          path: config.errorScreenshotPath,
          fullPage: false
        });
        console.error(`\n📸 Error screenshot saved to: ${config.errorScreenshotPath}`);
      } catch (screenshotError) {
        console.error('Failed to capture error screenshot:', screenshotError);
      }
    }

    console.error('\n' + '='.repeat(80) + '\n');
    process.exit(1);

  } finally {
    // Cleanup
    if (browser) {
      await browser.close();
    }
  }
}

// Run the driver
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
