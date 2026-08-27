/**
 * Browser coverage for HTTP-ranged `.zarr.zip` scene loading.
 *
 * The unit tests pin the store and reader pieces, but only a real browser load
 * catches the routing failure where an archive can reach `initialized` with an
 * empty scene. This spec packages an existing generated fixture, loads both
 * forms through the normal E2E data server, and requires identical element
 * counts. It also emulates a host that ignores Range and pins the loud startup
 * failure instead of an initialized empty scene. The DEFLATE fixture is smaller
 * than unzipit's end-of-directory search window, so it is retained in one read;
 * the STORED case covers windowed reads.
 */

import { fileURLToPath } from 'url';
import * as fs from 'fs';
import * as path from 'path';
import { zipSync } from 'fflate';
import { test, expect, ALLOW_CONSOLE_ERRORS, type Page } from './fixtures';
import {
  getLuxarState,
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForSpatialQueryOrThrow,
} from './helpers';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const viewerRoot = path.resolve(__dirname, '../../..');
const projectRoot = path.resolve(viewerRoot, '../..');
const fixtureName = 'test_extend_to_all_4d.luxar.zarr';
const fixturePath = path.join(viewerRoot, 'tests/fixtures', fixtureName);
const dataBaseURL = 'http://127.0.0.1:9000';

const archiveFormats = [
  { name: 'STORED', level: 0, compressionMethod: 0 },
  { name: 'DEFLATE', level: 6, compressionMethod: 8 },
] as const;

function collectArchiveEntries(root: string): Record<string, Uint8Array> {
  const entries: Record<string, Uint8Array> = {};

  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      for (const [name, bytes] of Object.entries(collectArchiveEntries(fullPath))) {
        entries[path.posix.join(entry.name, name)] = bytes;
      }
    } else if (entry.isFile()) {
      entries[entry.name] = fs.readFileSync(fullPath);
    }
  }

  return entries;
}

function fixtureURL(filePath: string): string {
  const relative = path.relative(projectRoot, filePath).split(path.sep).join('/');
  return `${dataBaseURL}/${relative}`;
}

async function loadElementCounts(page: Page, source: string) {
  await page.goto(`/?src=${encodeURIComponent(source)}&debug`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
  await waitForSpatialQueryOrThrow(page);

  const state = await getLuxarState(page);
  return {
    totalPoints: state.totalPoints,
    totalLines: state.totalLines,
    totalGSplats: state.totalGSplats,
    totalTriangles: state.totalTriangles,
    totalElements: state.totalElements,
  };
}

for (const format of archiveFormats) {
  test(`a ${format.name} zipped scene converges on the same element counts as its directory twin`, async ({
    page,
  }) => {
    const archivePath = path.join(
      viewerRoot,
      'test-results/zipped-store-loading',
      `test-extend-to-all-4d-${format.name.toLowerCase()}.luxar.zarr.zip`
    );
    fs.mkdirSync(path.dirname(archivePath), { recursive: true });
    const archiveBytes = zipSync(collectArchiveEntries(fixturePath), { level: format.level });
    const archiveView = new DataView(
      archiveBytes.buffer,
      archiveBytes.byteOffset,
      archiveBytes.byteLength
    );
    expect(archiveView.getUint16(8, true)).toBe(format.compressionMethod);
    fs.writeFileSync(archivePath, archiveBytes);

    const directoryCounts = await loadElementCounts(page, fixtureURL(fixturePath));
    const archiveCounts = await loadElementCounts(page, fixtureURL(archivePath));

    expect(directoryCounts.totalPoints).toBeGreaterThan(0);
    expect(directoryCounts.totalLines).toBeGreaterThan(0);
    expect(directoryCounts.totalGSplats).toBeGreaterThan(0);
    expect(archiveCounts).toEqual(directoryCounts);
  });
}

test('a host that ignores Range fails loudly without initializing an empty scene', async ({
  page,
}, testInfo) => {
  testInfo.annotations.push({
    type: ALLOW_CONSOLE_ERRORS,
    description: 'The console error is the user-visible behavior under test.',
  });

  const archiveBytes = zipSync(collectArchiveEntries(fixturePath), { level: 0 });
  const archiveURL = `${dataBaseURL}/range-ignored.luxar.zarr.zip`;
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  page.on('pageerror', (error) => pageErrors.push(error.message));
  await page.route(archiveURL, async (route) => {
    await route.fulfill({
      status: 200,
      headers: {
        'access-control-allow-origin': '*',
        'content-length': String(archiveBytes.byteLength),
        'content-type': 'application/zip',
      },
      body: route.request().method() === 'HEAD' ? undefined : Buffer.from(archiveBytes),
    });
  });

  await page.goto(`/?src=${encodeURIComponent(archiveURL)}&debug`);

  await expect
    .poll(() => consoleErrors.join('\n'), { timeout: 10_000 })
    .toMatch(/honours HTTP Range requests/);
  await expect(page.locator('#luxar-error-message')).toBeVisible();
  expect(pageErrors).toEqual([]);
  expect(
    await page.evaluate(() => ({
      initialized: window.__luxarDebug?.app.initialized,
      runtimeReady: window.__luxarDebug?.runtimeReady,
      hasGetState: typeof window.__luxarDebug?.getState === 'function',
    }))
  ).toEqual({ initialized: false, runtimeReady: undefined, hasGetState: false });
});
