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
const fixtureName = 'test_extend_to_all_4d.luxar.zarr';
const fixturePath = path.join(viewerRoot, 'tests/fixtures', fixtureName);
const dataBaseURL = 'http://127.0.0.1:9000';
const overlayName = 'archive-image';
const overlayImageName = 'image.png';
const overlayImageBytes = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2n3sAAAAASUVORK5CYII=',
  'base64'
);

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

function createImageOverlayFixture(destination: string): void {
  fs.cpSync(fixturePath, destination, { recursive: true });

  const overlayAttrs = {
    type: 'overlay_image',
    position: [0.02, 0.02],
    image_file: overlayImageName,
    size: [0.05, 0.05],
    opacity: 1,
    anchor: 'top-left',
    transition: 'none',
    transition_duration: 0.3,
    interactive: false,
    z_index: 0,
  };
  const overlaysMetadata = { attributes: {}, zarr_format: 3, node_type: 'group' };
  const overlayMetadata = { attributes: overlayAttrs, zarr_format: 3, node_type: 'group' };

  const rootMetadataPath = path.join(destination, 'zarr.json');
  const rootMetadata = JSON.parse(fs.readFileSync(rootMetadataPath, 'utf8'));
  rootMetadata.consolidated_metadata.metadata.overlays = overlaysMetadata;
  rootMetadata.consolidated_metadata.metadata[`overlays/${overlayName}`] = overlayMetadata;
  fs.writeFileSync(rootMetadataPath, `${JSON.stringify(rootMetadata, null, 2)}\n`);

  const overlayPath = path.join(destination, 'overlays', overlayName);
  fs.mkdirSync(overlayPath, { recursive: true });
  fs.writeFileSync(
    path.join(destination, 'overlays', 'zarr.json'),
    `${JSON.stringify(overlaysMetadata)}\n`
  );
  fs.writeFileSync(path.join(overlayPath, 'zarr.json'), `${JSON.stringify(overlayMetadata)}\n`);
  fs.writeFileSync(path.join(overlayPath, overlayImageName), overlayImageBytes);
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
    const stagedFixturePath = path.join(
      viewerRoot,
      'test-results/zipped-store-loading',
      `${format.name.toLowerCase()}-${fixtureName}`
    );
    const archivePath = path.join(
      viewerRoot,
      'test-results/zipped-store-loading',
      `test-extend-to-all-4d-${format.name.toLowerCase()}.luxar.zarr.zip`
    );
    fs.rmSync(stagedFixturePath, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(stagedFixturePath), { recursive: true });
    createImageOverlayFixture(stagedFixturePath);
    const archiveBytes = zipSync(collectArchiveEntries(stagedFixturePath), { level: format.level });
    const archiveView = new DataView(
      archiveBytes.buffer,
      archiveBytes.byteOffset,
      archiveBytes.byteLength
    );
    expect(archiveView.getUint16(8, true)).toBe(format.compressionMethod);
    fs.writeFileSync(archivePath, archiveBytes);

    const directoryState = await loadSceneState(page, fixtureURL(stagedFixturePath));
    const archiveState = await loadSceneState(page, fixtureURL(archivePath));

    expect(directoryState.elements.totalPoints).toBeGreaterThan(0);
    expect(directoryState.elements.totalLines).toBeGreaterThan(0);
    expect(directoryState.elements.totalGSplats).toBeGreaterThan(0);
    expect(archiveState.elements).toEqual(directoryState.elements);
    expect(directoryState.overlay).toEqual({ width: 1, height: 1, sourceKind: 'http' });
    expect(archiveState.overlay).toEqual({ width: 1, height: 1, sourceKind: 'blob' });
  });
}
