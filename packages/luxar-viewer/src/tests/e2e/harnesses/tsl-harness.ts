/**
 * TSL ↔ GLSL parity harness — page entry.
 *
 * Loaded by `tsl-harness.html` (Vite serves it at `/tsl-harness.html`).
 * Exposes `window.__tslHarness` with primitives that render a fullscreen
 * pass through both backends and return the result for pixel-diffing in
 * a Playwright spec.
 *
 * The harness internals live in `./tsl-harness/`, split by shader family:
 * - `types.ts` — the `RegistryEntry` registry contract
 * - `shared.ts` — cross-family camera / colormap-LUT helpers
 * - `post-processing.ts` — const-rgb, fxaa, bloom-threshold, mega* (8 entries)
 * - `points.ts` — point + point-pick variants (18 entries)
 * - `lines.ts` — line + line-pick variants (32 entries)
 * - `gsplats.ts` — gsplat + gsplat-pick variants (18 entries)
 * - `mesh.ts` — mesh + mesh-pick variants (8 entries)
 * - `render.ts` — the `renderGLSL` / `renderTSL` executors
 * - `index.ts` — merges the families into the 84-entry `SHADER_REGISTRY`
 *
 * Why a dedicated page rather than reusing the main viewer:
 * - Construction order is explicit and minimal — no app/state machine
 *   to wait on, no scene graph to mock around.
 * - Both backends (WebGL2 via `THREE.WebGLRenderer`, WebGPU-via-WebGL2
 *   via `WebGPURenderer({ forceWebGL: true })`) live side by side; the
 *   test toggles between them per call rather than per page load.
 * - The TSL path drives `GLSLNodeBuilder` directly so the generated
 *   GLSL strings are recoverable for snapshot-diff.
 *
 * Not in scope: real WebGPU dispatch. That requires Chrome stable +
 * `?webgpu=1` and runs in a separate spec. This harness validates
 * the WebGL2 fallback parity, which is what `forceWebGL: true` covers.
 *
 * @module tests/e2e/harnesses/tsl-harness
 */

import { SHADER_REGISTRY, renderGLSL, renderTSL } from './tsl-harness/index';

declare global {
  interface Window {
    __tslHarness?: {
      ready: Promise<void>;
      renderGLSL: (shaderName: string) => Uint8Array;
      renderTSL: (
        shaderName: string
      ) => Promise<{ pixels: Uint8Array; vertexShader: string; fragmentShader: string }>;
      listShaders: () => string[];
    };
  }
}

const status = document.getElementById('status');
const setStatus = (msg: string) => {
  if (status) status.textContent = msg;
};

const ready = (async () => {
  setStatus('tsl-harness ready');
})();

window.__tslHarness = {
  ready,
  renderGLSL,
  renderTSL,
  listShaders: () => Object.keys(SHADER_REGISTRY),
};
