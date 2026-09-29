import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

import { ViewContextProvider, type ViewSize } from '../../../scene/view-context';

function perspectiveAt(position: THREE.Vector3, target: THREE.Vector3): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 500);
  camera.position.copy(position);
  camera.lookAt(target);
  camera.updateProjectionMatrix();
  return camera;
}

function provider(
  getCamera: () => THREE.Camera,
  css: ViewSize | null = { width: 800, height: 450 },
  buffer: ViewSize | null = { width: 1600, height: 900 }
): ViewContextProvider {
  return new ViewContextProvider({
    getCamera,
    getViewportCss: () => css,
    getDrawingBuffer: () => buffer,
  });
}

describe('ViewContextProvider', () => {
  it('derives every field from the camera independently of the live matrices', () => {
    const camera = perspectiveAt(new THREE.Vector3(3, 4, 12), new THREE.Vector3(1, 0, 0));
    // A stale matrixWorld (never updated since the pose was written) must not leak in.
    const ctx = provider(() => camera).get();

    const world = new THREE.Matrix4().compose(
      camera.position,
      camera.quaternion,
      new THREE.Vector3(1, 1, 1)
    );
    const view = world.clone().invert();
    expect(ctx.viewMatrix.elements).toEqual(view.elements);
    expect(ctx.cameraWorldPosition.toArray()).toEqual([3, 4, 12]);
    const toTarget = new THREE.Vector3(1, 0, 0).sub(camera.position).normalize();
    expect(ctx.viewDirection.distanceTo(toTarget)).toBeLessThan(1e-12);
    expect(ctx.projectionMatrix.elements).toEqual(camera.projectionMatrix.elements);
    const projView = camera.projectionMatrix.clone().multiply(view);
    expect(ctx.projView.elements).toEqual(projView.elements);
    const frustum = new THREE.Frustum().setFromProjectionMatrix(projView);
    ctx.frustum.planes.forEach((plane, i) => {
      expect(plane.normal.toArray()).toEqual(frustum.planes[i].normal.toArray());
      expect(plane.constant).toBe(frustum.planes[i].constant);
    });
    expect(ctx.isOrtho).toBe(false);
    expect(ctx.viewportCss).toEqual({ width: 800, height: 450 });
    expect(ctx.drawingBuffer).toEqual({ width: 1600, height: 900 });
  });

  it('reports a parallel projection as ortho', () => {
    const camera = new THREE.OrthographicCamera(-8, 8, 4.5, -4.5, 0.1, 100);
    camera.position.set(0, 0, 20);
    camera.updateProjectionMatrix();
    expect(provider(() => camera).get().isOrtho).toBe(true);
  });

  it('uses the world pose of a parented camera', () => {
    const rig = new THREE.Group();
    rig.position.set(10, 0, 0);
    rig.rotation.y = Math.PI / 2;
    const camera = perspectiveAt(new THREE.Vector3(0, 0, 5), new THREE.Vector3(0, 0, 0));
    rig.add(camera);
    rig.updateMatrixWorld(true);

    const ctx = provider(() => camera).get();
    const expected = new THREE.Vector3();
    camera.getWorldPosition(expected);
    expect(ctx.cameraWorldPosition.distanceTo(expected)).toBeLessThan(1e-12);
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    expect(ctx.viewDirection.distanceTo(dir)).toBeLessThan(1e-12);
  });

  it('builds once while the camera is unchanged, then rebuilds when invalidated', () => {
    const camera = perspectiveAt(new THREE.Vector3(0, 0, 10), new THREE.Vector3());
    let css: ViewSize = { width: 800, height: 450 };
    const getViewportCss = vi.fn(() => css);
    const views = new ViewContextProvider({
      getCamera: () => camera,
      getViewportCss,
      getDrawingBuffer: () => null,
    });
    const first = views.get();
    expect(first.cameraWorldPosition.z).toBe(10);
    expect(views.get()).toBe(first); // the same reused object
    expect(getViewportCss).toHaveBeenCalledTimes(1); // no rebuild on an unchanged camera

    // Sizes are only re-read on invalidate() (the frame loop does it every frame).
    css = { width: 400, height: 225 };
    expect(views.get().viewportCss).toEqual({ width: 800, height: 450 });
    views.invalidate();
    const second = views.get();
    expect(second).toBe(first);
    expect(second.viewportCss).toEqual({ width: 400, height: 225 });
  });

  it('follows a camera move or projection change without an invalidate', () => {
    // A read outside the frame loop (e.g. a load-time consumer) must not see
    // the pose of the last tick.
    const camera = perspectiveAt(new THREE.Vector3(0, 0, 10), new THREE.Vector3());
    const views = provider(() => camera);
    expect(views.get().cameraWorldPosition.z).toBe(10);

    camera.position.set(0, 0, 20); // matrixWorld not updated yet
    expect(views.get().cameraWorldPosition.z).toBe(20);

    camera.near = 1;
    camera.updateProjectionMatrix();
    expect(views.get().projectionMatrix.elements).toEqual(camera.projectionMatrix.elements);
  });

  it('follows a camera swap', () => {
    let camera: THREE.Camera = perspectiveAt(new THREE.Vector3(0, 0, 10), new THREE.Vector3());
    const views = provider(() => camera);
    expect(views.get().camera).toBe(camera);
    const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
    camera = ortho;
    views.invalidate();
    expect(views.get().camera).toBe(ortho);
    expect(views.get().isOrtho).toBe(true);
  });

  it('reports a collapsed canvas or an empty buffer as null', () => {
    const camera = perspectiveAt(new THREE.Vector3(0, 0, 10), new THREE.Vector3());
    const ctx = provider(() => camera, { width: 0, height: 450 }, null).get();
    expect(ctx.viewportCss).toBeNull();
    expect(ctx.drawingBuffer).toBeNull();
  });
});
