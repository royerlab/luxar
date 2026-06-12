/**
 * Minimal embed example for the Luxar viewer.
 *
 * Demonstrates the embed surface:
 * - import LuxarApp from the built library and mount it on a host-owned canvas
 *   inside a host container (overlays/panels scope to the container)
 * - drive it with the programmatic API (recenterCamera, screenshot)
 * - react to events (dataset-loaded / dataset-error / dimensions-changed / selection)
 * - dispose it cleanly when done
 *
 * To run:
 *   cd packages/luxar-viewer
 *   pnpm build:lib
 *   python -m http.server 8765   # from this directory
 *   open http://localhost:8765/
 *
 * The page proves CSS isolation (host fonts/margins survive viewer mount)
 * and dispose cleanliness (host page is unchanged after teardown).
 */

import { LuxarApp } from '../../dist/lib/luxar-viewer.js';

// Default dataset: the package-local 4D test fixture, resolved relative to
// this module so it works wherever the package root is mounted. Generate it
// once with `pnpm test:generate-fixtures` if missing; replace with your own
// zarr URL (no trailing slash) for real data.
const DEFAULT_SRC = new URL('../../tests/fixtures/test_4d.zarr', import.meta.url).href;

const canvas = document.getElementById('luxar-canvas');
const viewerFrame = document.getElementById('viewer-frame');
const disposeBtn = document.getElementById('dispose-btn');
const reinitBtn = document.getElementById('reinit-btn');
const recenterBtn = document.getElementById('recenter-btn');
const screenshotBtn = document.getElementById('screenshot-btn');
const statusEl = document.getElementById('status');

let app = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function mount() {
  if (app) return;
  app = new LuxarApp();

  // Programmatic event surface — subscribe before/after init; the per-app
  // emitter outlives init/dispose cycles.
  app.on('dataset-loaded', ({ src }) => setStatus(`Loaded: ${src || '(none)'}`));
  app.on('dataset-error', ({ src, error }) => setStatus(`Error loading ${src}: ${error.message}`));
  app.on('dimensions-changed', (dims) => {
    if (dims.ndim > 0) setStatus(`Dimensions: [${dims.currentStep.join(', ')}]`);
  });
  app.on('selection', (sel) => {
    if (sel) setStatus(`Picked ${sel.nodeName} #${sel.elementIndex}`);
  });

  await app.init({
    canvas,
    // Mount every viewer overlay/panel/toast/dialog inside our framed box
    // rather than document.body. The frame is position:relative, so the
    // viewer's fixed overlays are scoped to it (and dispose() removes the
    // whole subtree cleanly). The viewer also attaches a ResizeObserver to
    // the canvas, so resizing the frame re-fits the scene automatically.
    container: viewerFrame,
    // Embed-friendly defaults: don't rewrite the host page's URL when the
    // user picks a dataset from the viewer's browser UI.
    updateBrowserUrl: false,
    // Load the bundled 4D fixture so the demo shows a scene immediately.
    // Replace with your own URL — or call app.switchDataset(url) later.
    src: DEFAULT_SRC,
  });
}

function teardown() {
  if (!app) return;
  app.dispose();
  app = null;
  setStatus('Disposed.');
}

disposeBtn.addEventListener('click', teardown);
reinitBtn.addEventListener('click', () => {
  teardown();
  void mount();
});
recenterBtn.addEventListener('click', () => app?.recenterCamera());
screenshotBtn.addEventListener('click', async () => {
  if (!app) return;
  try {
    const blob = await app.screenshot({ format: 'png' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'luxar-screenshot.png';
    a.click();
    URL.revokeObjectURL(url);
    setStatus('Screenshot downloaded.');
  } catch (err) {
    setStatus(`Screenshot failed: ${err.message}`);
  }
});

void mount().catch((err) => {
  console.error('[embed] LuxarApp.init failed:', err);
  setStatus(`Init failed: ${err.message}`);
});
