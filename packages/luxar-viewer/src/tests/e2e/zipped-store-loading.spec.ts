/**
 * Positive browser coverage for HTTP-ranged `.zarr.zip` scene loading.
 *
 * The unit tests pin the store and reader pieces, but only a real browser load
 * catches the routing failure where an archive can reach `initialized` with an
 * empty scene. This spec packages an existing generated fixture, loads both
 * forms through the normal E2E data server, and requires identical element
 * counts.
 */

import { fileURLToPath } from 'url';
import * as fs from 'fs';
import * as path from 'path';
import { zipSync } from 'fflate';
import { test, expect, type Page } from './fixtures';
import {
  getLuxarState,
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForSpatialQueryOrThrow,
} from './helpers';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const viewerRoot = path.resolve(__dirname, '../../..');
const projectRoot = path.resolve(viewerRoot, '../..');
const fixtureName = 'test_mixed.luxar.zarr';
const fixturePath = path.join(viewerRoot, 'tests/fixtures', fixtureName);
const archivePath = path.join(
  viewerRoot,
  'test-results/zipped-store-loading',
  `${fixtureName}.zip`
);
const dataBaseURL = 'http://127.0.0.1:9000';

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

test('a zipped scene converges on the same element counts as its directory twin', async ({
  page,
}) => {
  fs.mkdirSync(path.dirname(archivePath), { recursive: true });
  fs.writeFileSync(archivePath, zipSync(collectArchiveEntries(fixturePath), { level: 0 }));

  const directoryCounts = await loadElementCounts(page, fixtureURL(fixturePath));
  const archiveCounts = await loadElementCounts(page, fixtureURL(archivePath));

  expect(directoryCounts.totalElements).toBeGreaterThan(0);
  expect(archiveCounts).toEqual(directoryCounts);
});
