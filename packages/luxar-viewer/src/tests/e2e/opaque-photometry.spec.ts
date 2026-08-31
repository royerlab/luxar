import { test, expect, type Page } from './fixtures';
import { captureElementScreenshot, waitForLuxarReady } from './helpers';

const POINTS_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_points_blending_modes.luxar.zarr';
const LINES_FIXTURE =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lines_blending_modes.luxar.zarr';

async function meanCanvasLinearLuminance(page: Page): Promise<number> {
  const png = await captureElementScreenshot(page, 'canvas#app', 'framebuffer');
  const dataUrl = `data:image/png;base64,${png.toString('base64')}`;

  return page.evaluate(async (url) => {
    const img = new Image();
    img.decoding = 'sync';
    const loaded = new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('opaque-photometry: failed to decode screenshot'));
    });
    img.src = url;
    await loaded;

    const offscreen = document.createElement('canvas');
    offscreen.width = img.naturalWidth;
    offscreen.height = img.naturalHeight;
    const context = offscreen.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('opaque-photometry: 2D context unavailable');
    context.drawImage(img, 0, 0);

    const data = context.getImageData(0, 0, offscreen.width, offscreen.height).data;
    const toLinear = (value: number): number => {
      const channel = value / 255;
      return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
    };

    let sum = 0;
    for (let index = 0; index < data.length; index += 4) {
      sum +=
        0.2126 * toLinear(data[index]) +
        0.7152 * toLinear(data[index + 1]) +
        0.0722 * toLinear(data[index + 2]);
    }
    return sum / (data.length / 4);
  }, dataUrl);
}

async function stableCanvasLinearLuminance(page: Page, label: string): Promise<number> {
  const captures: number[] = [];
  let stableCaptures = 0;
  for (let attempt = 0; attempt < 8; attempt++) {
    await page.evaluate(() => (window as any).__luxarDebug.renderOnce?.());
    await page.waitForTimeout(attempt === 0 ? 100 : 1000);
    const current = await meanCanvasLinearLuminance(page);
    captures.push(current);
    const previous = captures.at(-2);
    if (previous === undefined) {
      stableCaptures = 1;
      continue;
    }
    const tolerance = Math.max(1e-8, Math.max(Math.abs(previous), Math.abs(current)) * 1e-4);
    stableCaptures = Math.abs(current - previous) <= tolerance ? stableCaptures + 1 : 1;
    if (stableCaptures >= 3) return current;
  }
  throw new Error(`${label}: canvas luminance did not settle (${captures.join(', ')})`);
}

async function isolateOpaqueGeometry(page: Page, nodeType: 'points' | 'lines'): Promise<number> {
  return page.evaluate((type) => {
    let opaqueCount = 0;
    const debug = (window as any).__luxarDebug;
    debug.scene.traverse((object: any) => {
      if (object.userData?.nodeType !== type || !object.material) return;
      const isOpaque = object.material.userData?.blendingMode === 'opaque';
      object.visible = isOpaque;
      if (isOpaque && (object.geometry?.instanceCount ?? 0) > 0) opaqueCount++;
    });
    debug.renderOnce?.();
    return opaqueCount;
  }, nodeType);
}

async function setOpaqueOpacity(page: Page, nodeType: 'points' | 'lines', opacity: number) {
  return page.evaluate(
    ({ type, value }) => {
      let updated = 0;
      const debug = (window as any).__luxarDebug;
      debug.scene.traverse((object: any) => {
        if (
          object.userData?.nodeType !== type ||
          object.material?.userData?.blendingMode !== 'opaque'
        )
          return;
        const uniform = object.material.uniforms?.uOpacity;
        if (!uniform) return;
        uniform.value = value;
        updated++;
      });
      debug.renderOnce?.();
      return updated;
    },
    { type: nodeType, value: opacity }
  );
}

async function setBlendingModeVisible(
  page: Page,
  nodeType: 'points' | 'lines',
  blendingMode: 'opaque' | 'luminous',
  visible: boolean
) {
  return page.evaluate(
    ({ type, mode, value }) => {
      let updated = 0;
      const debug = (window as any).__luxarDebug;
      debug.scene.traverse((object: any) => {
        if (object.userData?.nodeType !== type || object.material?.userData?.blendingMode !== mode)
          return;
        object.visible = value;
        updated++;
      });
      debug.renderOnce?.();
      return updated;
    },
    { type: nodeType, mode: blendingMode, value: visible }
  );
}

async function overlapOpaqueInFrontOfLuminous(
  page: Page,
  nodeType: 'points' | 'lines'
): Promise<{
  opaqueCameraDistance: number;
  luminousCameraDistance: number;
  projectedBoundsIntersect: boolean;
} | null> {
  return page.evaluate((type) => {
    const debug = (window as any).__luxarDebug;
    let opaque: any;
    let luminous: any;
    debug.scene.traverse((object: any) => {
      if (object.userData?.nodeType !== type || !object.material) return;
      const mode = object.material.userData?.blendingMode;
      object.visible = mode === 'opaque' || mode === 'luminous';
      if (mode === 'opaque') opaque = object;
      if (mode === 'luminous') luminous = object;
    });
    if (!opaque || !luminous) return null;

    const worldCenter = (object: any) => {
      const boundingBox = object.geometry.boundingBox;
      if (!boundingBox) throw new Error(`${type}: geometry has no authored bounding box`);
      const center = boundingBox.getCenter(object.position.clone());
      return object.localToWorld(center);
    };
    const worldRadius = (object: any) => {
      const boundingSphere = object.geometry.boundingSphere;
      if (!boundingSphere) throw new Error(`${type}: geometry has no authored bounding sphere`);
      const scale = object.getWorldScale(object.position.clone());
      return (
        boundingSphere.radius * Math.max(Math.abs(scale.x), Math.abs(scale.y), Math.abs(scale.z))
      );
    };
    const projectedBounds = (object: any) => {
      const center = worldCenter(object);
      const radius = worldRadius(object);
      const projectedCenter = center.clone().project(debug.camera);
      const cameraRight = center
        .clone()
        .setFromMatrixColumn(debug.camera.matrixWorld, 0)
        .normalize()
        .multiplyScalar(radius);
      const cameraUp = center
        .clone()
        .setFromMatrixColumn(debug.camera.matrixWorld, 1)
        .normalize()
        .multiplyScalar(radius);
      const projectedRight = center.clone().add(cameraRight).project(debug.camera);
      const projectedUp = center.clone().add(cameraUp).project(debug.camera);
      const radiusX = Math.abs(projectedRight.x - projectedCenter.x);
      const radiusY = Math.abs(projectedUp.y - projectedCenter.y);
      return {
        minX: projectedCenter.x - radiusX,
        maxX: projectedCenter.x + radiusX,
        minY: projectedCenter.y - radiusY,
        maxY: projectedCenter.y + radiusY,
      };
    };
    debug.scene.updateMatrixWorld(true);
    const opaqueCenterInParent = opaque.parent.worldToLocal(worldCenter(opaque).clone());
    const luminousCenterInParent = opaque.parent.worldToLocal(worldCenter(luminous).clone());
    opaque.position.add(luminousCenterInParent.sub(opaqueCenterInParent));

    debug.scene.updateMatrixWorld(true);
    const cameraPosition = debug.camera.getWorldPosition(opaque.position.clone());
    const alignedOpaqueCenter = worldCenter(opaque);
    const nudgeDistance = Math.min(worldRadius(opaque), worldRadius(luminous));
    const targetWorldCenter = alignedOpaqueCenter
      .clone()
      .add(
        cameraPosition.clone().sub(alignedOpaqueCenter).normalize().multiplyScalar(nudgeDistance)
      );
    const alignedCenterInParent = opaque.parent.worldToLocal(alignedOpaqueCenter.clone());
    const targetCenterInParent = opaque.parent.worldToLocal(targetWorldCenter);
    opaque.position.add(targetCenterInParent.sub(alignedCenterInParent));

    debug.scene.updateMatrixWorld(true);
    const opaqueCenter = worldCenter(opaque);
    const luminousCenter = worldCenter(luminous);
    const opaqueBounds = projectedBounds(opaque);
    const luminousBounds = projectedBounds(luminous);
    debug.renderOnce?.();
    return {
      opaqueCameraDistance: cameraPosition.distanceTo(opaqueCenter),
      luminousCameraDistance: cameraPosition.distanceTo(luminousCenter),
      projectedBoundsIntersect:
        opaqueBounds.maxX >= luminousBounds.minX &&
        luminousBounds.maxX >= opaqueBounds.minX &&
        opaqueBounds.maxY >= luminousBounds.minY &&
        luminousBounds.maxY >= opaqueBounds.minY,
    };
  }, nodeType);
}

for (const [nodeType, fixture] of [
  ['points', POINTS_FIXTURE],
  ['lines', LINES_FIXTURE],
] as const) {
  test(`opaque ${nodeType} preserve fragment photometry`, async ({ page }) => {
    test.slow();
    await page.goto(`/?src=${fixture}&debug&dpr=1&no-opfs`);
    await waitForLuxarReady(page);
    await page.waitForFunction(
      (type) => {
        let committed = 0;
        (window as any).__luxarDebug.scene.traverse((object: any) => {
          if (object.userData?.nodeType === type && (object.geometry?.instanceCount ?? 0) > 0)
            committed++;
        });
        return committed >= 6;
      },
      nodeType,
      { timeout: 60000 }
    );

    expect(await isolateOpaqueGeometry(page, nodeType)).toBeGreaterThan(0);
    await stableCanvasLinearLuminance(page, `${nodeType}: opaque warm-up capture`);

    expect(await setBlendingModeVisible(page, nodeType, 'opaque', false)).toBeGreaterThan(0);
    const withoutOpaque = await stableCanvasLinearLuminance(
      page,
      `${nodeType}: first background capture`
    );

    expect(await setBlendingModeVisible(page, nodeType, 'opaque', true)).toBeGreaterThan(0);
    const full = await stableCanvasLinearLuminance(page, `${nodeType}: full opaque capture`);

    expect(await setOpaqueOpacity(page, nodeType, 0.1)).toBeGreaterThan(0);
    const dimmed = await stableCanvasLinearLuminance(page, `${nodeType}: dimmed opaque capture`);

    expect(await setBlendingModeVisible(page, nodeType, 'opaque', false)).toBeGreaterThan(0);
    const withoutOpaqueAfter = await stableCanvasLinearLuminance(
      page,
      `${nodeType}: second background capture`
    );
    const backgroundDrift = Math.abs(withoutOpaqueAfter - withoutOpaque);
    const backgroundTolerance = Math.max(1e-8, withoutOpaque * 1e-3);
    expect(
      backgroundDrift,
      `${nodeType}: background changed between captures (before=${withoutOpaque}, after=${withoutOpaqueAfter}, drift=${backgroundDrift})`
    ).toBeLessThanOrEqual(backgroundTolerance);

    // The clear-colour floor remains when the opaque node is hidden; compare only its signal.
    const background = (withoutOpaque + withoutOpaqueAfter) / 2;
    const fullSignal = full - background;
    const dimmedSignal = dimmed - background;
    const dimmedRatio = dimmedSignal / fullSignal;
    const measurements = `full=${full}, withoutOpaque=${withoutOpaque}, withoutOpaqueAfter=${withoutOpaqueAfter}, dimmed=${dimmed}, fullSignal=${fullSignal}, dimmedSignal=${dimmedSignal}, ratio=${dimmedRatio}`;

    expect(
      fullSignal,
      `${nodeType}: isolated opaque geometry must dominate the background; ${measurements}`
    ).toBeGreaterThan(Math.max(1e-4, background * 10));
    // Nominal opacity is 0.1; measured ratios are 0.11910 (points) and 0.08684 (lines).
    expect(
      dimmedRatio,
      `${nodeType}: opaque framebuffer output must retain measurable fragment alpha; ${measurements}`
    ).toBeGreaterThan(0.07);
    expect(
      dimmedRatio,
      `${nodeType}: opaque framebuffer output must dim with fragment alpha; ${measurements}`
    ).toBeLessThan(0.15);
  });

  test(`dim opaque ${nodeType} do not erase luminous geometry behind`, async ({ page }) => {
    test.slow();
    await page.goto(`/?src=${fixture}&debug&dpr=1&no-opfs`);
    await waitForLuxarReady(page);
    await page.waitForFunction(
      (type) => {
        const modes = new Set<string>();
        (window as any).__luxarDebug.scene.traverse((object: any) => {
          if (object.userData?.nodeType === type && (object.geometry?.instanceCount ?? 0) > 0) {
            modes.add(object.material?.userData?.blendingMode);
          }
        });
        return modes.has('opaque') && modes.has('luminous');
      },
      nodeType,
      { timeout: 60000 }
    );

    const overlap = await overlapOpaqueInFrontOfLuminous(page, nodeType);
    expect(overlap, `${nodeType}: opaque and luminous fixtures must both exist`).not.toBeNull();
    if (!overlap) throw new Error(`${nodeType}: overlap setup failed`);
    expect(
      overlap.opaqueCameraDistance,
      `${nodeType}: opaque geometry must be closer to the camera than luminous geometry`
    ).toBeLessThan(overlap.luminousCameraDistance);
    expect(
      overlap.projectedBoundsIntersect,
      `${nodeType}: opaque and luminous projected bounds must overlap`
    ).toBe(true);
    await stableCanvasLinearLuminance(page, `${nodeType}: overlap warm-up capture`);
    expect(await setBlendingModeVisible(page, nodeType, 'opaque', false)).toBeGreaterThan(0);
    expect(await setBlendingModeVisible(page, nodeType, 'luminous', false)).toBeGreaterThan(0);
    const background = await stableCanvasLinearLuminance(
      page,
      `${nodeType}: first luminous background capture`
    );

    expect(await setBlendingModeVisible(page, nodeType, 'luminous', true)).toBeGreaterThan(0);
    const luminousOnly = await stableCanvasLinearLuminance(page, `${nodeType}: luminous capture`);

    expect(await setBlendingModeVisible(page, nodeType, 'opaque', true)).toBeGreaterThan(0);
    expect(await setOpaqueOpacity(page, nodeType, 0.00005)).toBeGreaterThan(0);
    const withDimOpaque = await stableCanvasLinearLuminance(
      page,
      `${nodeType}: dim opaque over luminous capture`
    );

    expect(await setBlendingModeVisible(page, nodeType, 'opaque', false)).toBeGreaterThan(0);
    expect(await setBlendingModeVisible(page, nodeType, 'luminous', false)).toBeGreaterThan(0);
    const backgroundAfter = await stableCanvasLinearLuminance(
      page,
      `${nodeType}: second luminous background capture`
    );
    const backgroundDrift = Math.abs(backgroundAfter - background);
    expect(
      backgroundDrift,
      `${nodeType}: luminous background changed between captures (before=${background}, after=${backgroundAfter}, drift=${backgroundDrift})`
    ).toBeLessThanOrEqual(Math.max(1e-8, background * 1e-3));

    const averageBackground = (background + backgroundAfter) / 2;
    const luminousSignal = luminousOnly - averageBackground;
    const withDimOpaqueSignal = withDimOpaque - averageBackground;
    const visibleRatio = withDimOpaqueSignal / luminousSignal;
    const measurements = `background=${background}, backgroundAfter=${backgroundAfter}, luminousOnly=${luminousOnly}, withDimOpaque=${withDimOpaque}, luminousSignal=${luminousSignal}, withDimOpaqueSignal=${withDimOpaqueSignal}, ratio=${visibleRatio}`;
    expect(
      luminousSignal,
      `${nodeType}: luminous geometry behind must dominate the background; ${measurements}`
    ).toBeGreaterThan(Math.max(1e-4, averageBackground * 5));
    expect(
      visibleRatio,
      `${nodeType}: negligible opaque contributions must not stamp the depth buffer; ${measurements}`
    ).toBeGreaterThan(0.9);
  });
}
