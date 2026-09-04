import * as THREE from 'three';
import { afterEach, describe, expect, it } from 'vitest';

import {
  ProjectedDensityTracker,
  getProjectedDensityTracker,
  projectSphereAreaPx,
  snapshotProjectedDensity,
} from '../../../scene/projected-density';
import { setCommittedData } from '../../../types/committed-data';

/** A committed gsplats node the tracker can measure. */
function node(path: string, radius: number, splats: number, position = new THREE.Vector3()) {
  const geometry = new THREE.BufferGeometry();
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), radius);
  const mesh = new THREE.Mesh(geometry);
  mesh.name = path;
  mesh.position.copy(position);
  mesh.userData.nodeType = 'gsplats';
  mesh.userData.visibleSplatCount = splats;
  setCommittedData(mesh, {});
  return mesh;
}

function perspective(width: number, height: number, distance: number): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
  camera.position.set(0, 0, distance);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  camera.updateProjectionMatrix();
  return camera;
}

function tracker(root: THREE.Object3D, camera: THREE.Camera, width = 1600, height = 1000) {
  const t = new ProjectedDensityTracker();
  let enabled = true;
  t.configure({
    enabled: () => enabled,
    getRoot: () => root,
    getCamera: () => camera,
    getDrawingBufferSize: () => ({ width, height }),
  });
  return { t, setEnabled: (v: boolean) => (enabled = v) };
}

describe('projectSphereAreaPx', () => {
  it('a sphere filling half the vertical FOV covers ~π/4 of a square-ish buffer', () => {
    // fov 60°, distance 10: half-height at the sphere = 10·tan(30°) = 5.77.
    // A radius of 5.77 spans the full NDC height → r_ndc_y = 1, r_ndc_x = 1/aspect.
    const camera = perspective(1000, 1000, 10);
    const r = 10 * Math.tan(Math.PI / 6);
    const { areaPx, onScreen } = projectSphereAreaPx({ x: 0, y: 0, z: -10 }, r, camera, 1000, 1000);
    expect(onScreen).toBe(true);
    expect(areaPx / (1000 * 1000)).toBeCloseTo(Math.PI / 4, 2);
  });

  it('scales with the buffer: half the resolution → a quarter of the pixels', () => {
    const camera = perspective(1600, 1000, 100);
    const a = projectSphereAreaPx({ x: 0, y: 0, z: -100 }, 5, camera, 1600, 1000).areaPx;
    const b = projectSphereAreaPx({ x: 0, y: 0, z: -100 }, 5, camera, 800, 500).areaPx;
    expect(b).toBeCloseTo(a / 4, 6);
  });

  it('is off-screen when the ellipse misses the NDC square, full-buffer when the camera is inside', () => {
    const camera = perspective(1600, 1000, 100);
    expect(projectSphereAreaPx({ x: 500, y: 0, z: -100 }, 5, camera, 1600, 1000)).toEqual({
      areaPx: 0,
      onScreen: false,
    });
    expect(projectSphereAreaPx({ x: 0, y: 0, z: -1 }, 5, camera, 1600, 1000)).toEqual({
      areaPx: 1_600_000,
      onScreen: true,
    });
  });

  it('handles an orthographic camera without the depth division', () => {
    const camera = new THREE.OrthographicCamera(-10, 10, 5, -5, 0.1, 100);
    camera.updateProjectionMatrix();
    // radius 5 → NDC radii 0.5 (x) and 1.0 (y) → ellipse π·(0.5·800)·(1·500)
    const { areaPx } = projectSphereAreaPx({ x: 0, y: 0, z: -50 }, 5, camera, 1600, 1000);
    expect(areaPx).toBeCloseTo(Math.PI * 400 * 500, 3);
  });
});

describe('ProjectedDensityTracker', () => {
  afterEach(() => getProjectedDensityTracker().reset());

  it('measures elements per drawing-buffer pixel for committed, named, visible meshes only', () => {
    const root = new THREE.Scene();
    const dense = node('/dense', 1, 1_000_000);
    root.add(dense);
    const uncommitted = new THREE.Mesh(new THREE.BufferGeometry());
    uncommitted.name = '/uncommitted';
    uncommitted.geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1);
    root.add(uncommitted);
    root.updateMatrixWorld(true);
    const camera = perspective(1600, 1000, 100);
    const { t } = tracker(root, camera);

    expect(t.evaluate()).toBe(true);
    const rec = t.get('/dense');
    expect(rec).toBeDefined();
    expect(rec!.onScreen).toBe(true);
    expect(rec!.elements).toBe(1_000_000);
    // radius 1 at distance 100 with fov 60: r_ndc_y = 1/(100·tan30°) = 0.0173 →
    // ellipse ≈ π·(0.0173/1.6·800)·(0.0173·500) ≈ 235 px
    expect(rec!.areaPx).toBeGreaterThan(200);
    expect(rec!.areaPx).toBeLessThan(270);
    expect(rec!.elementsPerPixel).toBeCloseTo(1_000_000 / rec!.areaPx, 6);
    expect(t.get('/uncommitted')).toBeUndefined();
  });

  it('a hidden node reads off-screen with zero density, and disposed nodes are pruned', () => {
    const root = new THREE.Scene();
    const a = node('/a', 1, 1000);
    const b = node('/b', 1, 1000);
    root.add(a, b);
    root.updateMatrixWorld(true);
    const { t } = tracker(root, perspective(1600, 1000, 100));
    t.evaluate();
    expect(t.get('/a')!.onScreen).toBe(true);

    a.visible = false;
    t.evaluate();
    expect(t.get('/a')).toMatchObject({
      onScreen: false,
      areaPx: 0,
      elementsPerPixel: 0,
      elements: 1000,
    });

    root.remove(b);
    t.evaluate();
    expect(t.get('/b')).toBeUndefined();
    expect(Object.keys(t.snapshot())).toEqual(['/a']);
  });

  it('applies the mesh world scale to the sphere radius', () => {
    const root = new THREE.Scene();
    const small = node('/small', 1, 100);
    const scaled = node('/scaled', 1, 100, new THREE.Vector3(3, 0, 0));
    scaled.scale.setScalar(2);
    root.add(small, scaled);
    root.updateMatrixWorld(true);
    const { t } = tracker(root, perspective(1600, 1000, 100));
    t.evaluate();
    expect(t.get('/scaled')!.areaPx / t.get('/small')!.areaPx).toBeCloseTo(4, 1);
  });

  it('does nothing when disabled or without a root/camera/buffer, and the singleton snapshot follows', () => {
    const root = new THREE.Scene();
    root.add(node('/a', 1, 10));
    root.updateMatrixWorld(true);
    const camera = perspective(1600, 1000, 100);
    const { t, setEnabled } = tracker(root, camera);
    setEnabled(false);
    expect(t.evaluate()).toBe(false);
    expect(t.get('/a')).toBeUndefined();

    const empty = new ProjectedDensityTracker();
    expect(empty.evaluate()).toBe(false);

    const s = getProjectedDensityTracker();
    s.configure({
      enabled: () => true,
      getRoot: () => root,
      getCamera: () => camera,
      getDrawingBufferSize: () => ({ width: 1600, height: 1000 }),
    });
    s.evaluate();
    expect(Object.keys(snapshotProjectedDensity())).toEqual(['/a']);
    // Snapshot records are copies.
    const snap = snapshotProjectedDensity();
    snap['/a'].elements = -1;
    expect(snapshotProjectedDensity()['/a'].elements).toBe(10);
  });
});
