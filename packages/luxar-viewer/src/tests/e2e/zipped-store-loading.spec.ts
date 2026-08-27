/**
 * Positive browser coverage for HTTP-ranged `.zarr.zip` scene loading.
 *
 * The unit tests pin the store and reader pieces, but only a real browser load
 * catches the routing failure where an archive can reach `initialized` with an
 * empty scene. This spec packages an existing generated fixture, loads both
 * forms through the normal E2E data server, and requires identical element
 * counts. The DEFLATE fixture is smaller than unzipit's end-of-directory search
 * window, so it is retained in one read; the STORED case covers windowed reads.
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
const fixtureName = 'test_image_overlay.luxar.zarr';
const fixturePath = path.join(viewerRoot, 'tests/fixtures', fixtureName);
const dataBaseURL = 'http://127.0.0.1:9000';
const overlayName = 'archive-image';
const endOfDirectorySearchWindowBytes = 65_557;

const archiveFormats = [
  { name: 'STORED', level: 0, compressionMethod: 0, windowedRead: true },
  { name: 'DEFLATE', level: 6, compressionMethod: 8, windowedRead: false },
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

async function loadSceneState(page: Page, source: string) {
  await page.goto(`/?src=${encodeURIComponent(source)}&debug`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
  await waitForSpatialQueryOrThrow(page);

  const image = page.locator(`[data-overlay-name="${overlayName}"] img`);
  await expect(image).toBeVisible();
  await expect
    .poll(() =>
      image.evaluate((element: HTMLImageElement) => ({
        complete: element.complete,
        width: element.naturalWidth,
        height: element.naturalHeight,
      }))
    )
    .toEqual({ complete: true, width: 1, height: 1 });

  const state = await getLuxarState(page);
  return {
    elements: {
      totalPoints: state.totalPoints,
      totalLines: state.totalLines,
      totalGSplats: state.totalGSplats,
      totalTriangles: state.totalTriangles,
      totalElements: state.totalElements,
    },
    overlay: await image.evaluate((element: HTMLImageElement) => ({
      width: element.naturalWidth,
      height: element.naturalHeight,
      sourceKind: element.src.startsWith('blob:') ? 'blob' : 'http',
    })),
  };
}

for (const format of archiveFormats) {
  test(`a ${format.name} zipped scene matches its directory twin, including image overlays`, async ({
    page,
  }) => {
    const archivePath = path.join(
      viewerRoot,
      'test-results/zipped-store-loading',
      `test-image-overlay-${format.name.toLowerCase()}.luxar.zarr.zip`
    );
    fs.mkdirSync(path.dirname(archivePath), { recursive: true });
    const archiveBytes = zipSync(collectArchiveEntries(fixturePath), { level: format.level });
    const archiveView = new DataView(
      archiveBytes.buffer,
      archiveBytes.byteOffset,
      archiveBytes.byteLength
    );
    expect(archiveView.getUint16(8, true)).toBe(format.compressionMethod);
    expect(archiveBytes.byteLength > endOfDirectorySearchWindowBytes).toBe(format.windowedRead);
    fs.writeFileSync(archivePath, archiveBytes);

    const directoryState = await loadSceneState(page, fixtureURL(fixturePath));
    const archiveState = await loadSceneState(page, fixtureURL(archivePath));

    expect(directoryState.elements.totalPoints).toBeGreaterThan(0);
    expect(directoryState.elements.totalLines).toBeGreaterThan(0);
    expect(directoryState.elements.totalGSplats).toBeGreaterThan(0);
    expect(archiveState.elements).toEqual(directoryState.elements);
    expect(directoryState.overlay).toEqual({ width: 1, height: 1, sourceKind: 'http' });
    expect(archiveState.overlay).toEqual({ width: 1, height: 1, sourceKind: 'blob' });
  });
}
