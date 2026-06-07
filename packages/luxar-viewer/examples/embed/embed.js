/**
 * Minimal embed example for the Luxar viewer.
 *
 * Demonstrates the full v1 embed surface in ~30 lines:
 * - import LuxarApp from the built library
 * - mount it on a host-owned canvas
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

const canvas = document.getElementById('luxar-canvas');
const viewerFrame = document.getElementById('viewer-frame');
const disposeBtn = document.getElementById('dispose-btn');
const reinitBtn = document.getElementById('reinit-btn');

let app = null;

async function mount() {
  if (app) return;
  app = new LuxarApp();
  await app.init({
    canvas,
    // Mount every viewer overlay/panel/toast/dialog inside our framed box
    // rather than document.body. The frame is position:relative, so the
    // viewer's fixed overlays are scoped to it (and dispose() removes the
    // whole subtree cleanly).
    container: viewerFrame,
    // Embed-friendly defaults: don't rewrite the host page's URL when the
    // user picks a dataset from the viewer's browser UI.
    updateBrowserUrl: false,
    // Use a dataset that's known to work with the demo server. Replace with
    // your own URL.
    src: '',
  });
}

function teardown() {
  if (!app) return;
  app.dispose();
  app = null;
}

disposeBtn.addEventListener('click', teardown);
reinitBtn.addEventListener('click', () => {
  teardown();
  void mount();
});

void mount().catch((err) => {
  console.error('[embed] LuxarApp.init failed:', err);
});
