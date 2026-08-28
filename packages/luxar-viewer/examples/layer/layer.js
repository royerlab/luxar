import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LuxarLayer } from '../../src/index.ts';

const DEFAULT_SRC =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_lod_group.luxar.zarr';

const canvas = document.getElementById('layer-canvas');
const viewport = document.getElementById('viewport');
const status = document.getElementById('status');
const src = new URLSearchParams(window.location.search).get('src') || DEFAULT_SRC;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setClearColor(0x050812, 1);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
camera.position.set(8, 8, 8);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

let loaded = false;
let frameRequested = true;

const layer = new LuxarLayer({
  renderer,
  scene,
  getCamera: () => camera,
  getViewportSize: () => ({ width: viewport.clientWidth, height: viewport.clientHeight }),
  requestRender: () => {
    frameRequested = true;
  },
});
layer.onDatasetFault(({ src: faultSrc, error }) => {
  status.textContent = `Dataset fault for ${faultSrc}: ${error.message}`;
});

function resize() {
  const width = Math.max(1, viewport.clientWidth);
  const height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  layer.resize();
  frameRequested = true;
}

function frameLayer() {
  const bounds = layer.getBounds();
  if (!bounds) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  camera.position
    .copy(center)
    .add(new THREE.Vector3(1, 1, 1).normalize().multiplyScalar(size * 1.6));
  camera.near = Math.max(size / 1000, 0.001);
  camera.far = Math.max(size * 100, 100);
  camera.updateProjectionMatrix();
  controls.update();
  layer.resize();
}

function isEffectivelyVisible(object) {
  let current = object;
  while (current) {
    if (!current.visible) return false;
    current = current.parent;
  }
  return true;
}

function getVisibleGeometry() {
  let visibleSplatCount = 0;
  const visibleLodLevels = [];
  layer.root?.traverse((object) => {
    if (!isEffectivelyVisible(object)) return;
    const count = object.userData.visibleSplatCount;
    if (typeof count === 'number') visibleSplatCount += count;
    const match = object.name.match(/\/child_(\d+)$/);
    if (match) visibleLodLevels.push(Number(match[1]));
  });
  return { visibleSplatCount, visibleLodLevels: visibleLodLevels.sort((a, b) => a - b) };
}

function getState() {
  const dimensions = layer.getDimensions();
  return {
    loaded,
    ...getVisibleGeometry(),
    dimensions: dimensions
      ? { names: layer.getDimensionNames(), currentStep: [...dimensions.currentStep] }
      : null,
  };
}

function setCameraDistance(distance) {
  const target = controls.target.clone();
  const direction = camera.position.clone().sub(target).normalize();
  camera.position.copy(target).add(direction.multiplyScalar(distance));
  camera.updateMatrixWorld(true);
  controls.update();
  layer.resize();
  frameRequested = true;
}

async function setDimensionValue(index, value) {
  await layer.setDimensionValue(index, value);
  frameRequested = true;
}

async function dispose() {
  await layer.dispose();
  loaded = false;
  frameRequested = true;
}

window.__luxarLayerExample = { dispose, getState, setCameraDistance, setDimensionValue };

controls.addEventListener('change', () => {
  frameRequested = true;
});
window.addEventListener('resize', resize);
window.addEventListener('pagehide', () => void dispose(), { once: true });
canvas.addEventListener('webglcontextlost', (event) => {
  event.preventDefault();
  layer.handleContextLost();
});
canvas.addEventListener('webglcontextrestored', () => {
  layer.handleContextRestored();
  frameRequested = true;
});

function animate() {
  requestAnimationFrame(animate);
  const controlsChanged = controls.update();
  layer.update();
  if (!frameRequested && !controlsChanged) return;
  renderer.render(scene, camera);
  frameRequested = false;
}

resize();
animate();

try {
  await layer.load(src);
  frameLayer();
  loaded = true;
  status.textContent = `Loaded ${new URL(src).pathname.split('/').pop()}`;
  frameRequested = true;
} catch (error) {
  status.textContent = `Load failed: ${error instanceof Error ? error.message : String(error)}`;
  throw error;
}
