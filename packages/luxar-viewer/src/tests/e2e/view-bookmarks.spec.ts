/** A named bookmark restores the live view and the copied link survives a reload. */
import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';
import { readFile } from 'node:fs/promises';

const DATASET = 'http://localhost:9000/datasets/examples/layers_test_example.luxar.zarr';

test('bookmark panel captures, revisits, exports, and opens a view link', async ({ page }) => {
  await page.goto(`/?src=${DATASET}&debug&renderer=webgl`);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);

  const expected = await page.evaluate(() => {
    const app = (window as any).__luxarDebug.app;
    const pose = app.getCameraPose();
    const moved = { ...pose, position: [pose.position[0] + 5, pose.position[1], pose.position[2]] };
    app.setCameraPose(moved);
    const layer = app.getLayers()[0];
    app.setLayer(layer.path, { visible: false, opacity: 0.4 });
    return { position: app.getCameraPose().position, path: layer.path };
  });

  await page.locator('[data-rail-id="bookmarks"]').click();
  await page.getByRole('textbox', { name: 'Bookmark name' }).fill('Cells at T7');
  await page.getByRole('button', { name: 'Add bookmark' }).click();
  const saved = page.getByRole('button', { name: 'Cells at T7' });
  const url = await saved.getAttribute('title');
  expect(url).toContain('view=');

  await page.evaluate((path) => {
    const app = (window as any).__luxarDebug.app;
    app.recenterCamera();
    app.setLayer(path, { visible: true, opacity: 1 });
  }, expected.path);
  await saved.click();
  await expect(page.getByRole('status')).toHaveText('Opened Cells at T7');
  const revisited = await page.evaluate((path) => {
    const app = (window as any).__luxarDebug.app;
    return {
      position: app.getCameraPose().position,
      layer: app.getLayers().find((l: any) => l.path === path),
    };
  }, expected.path);
  for (let i = 0; i < 3; i++) expect(revisited.position[i]).toBeCloseTo(expected.position[i], 5);
  expect(revisited.layer.visible).toBe(false);
  expect(revisited.layer.opacity).toBeCloseTo(0.4);

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('luxar-bookmarks.txt');
  expect(await readFile(await download.path(), 'utf8')).toBe(`Cells at T7\t${url}\n`);
  await page.getByRole('button', { name: 'Clear', exact: true }).click();
  await expect(saved).toHaveCount(0);

  await page.goto(url!);
  await waitForLuxarReady(page);
  await waitForPointsLoaded(page);
  const opened = await page.evaluate((path) => {
    const app = (window as any).__luxarDebug.app;
    return {
      position: app.getCameraPose().position,
      layer: app.getLayers().find((l: any) => l.path === path),
    };
  }, expected.path);
  for (let i = 0; i < 3; i++) expect(opened.position[i]).toBeCloseTo(expected.position[i], 5);
  expect(opened.layer.visible).toBe(false);
  expect(opened.layer.opacity).toBeCloseTo(0.4);
});
