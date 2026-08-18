/**
 * The one dynamic edge above the TSL / WebGPU cone.
 *
 * `loadTslMaterials()` is the only place that imports
 * `rendering/tsl/registry`, and it does so with `await import()`.
 * Because that is the sole path from the entry point to the `*-tsl` modules,
 * rolldown emits them — and the `three-webgpu` chunk they pull in — as a lazy
 * chunk instead of a static dependency of `index-*.js`. A WebGL session never
 * calls this, so it never pays for ~173 kB gzipped of the three.js node system
 * (issue #1679).
 *
 * The nine TSL material classes deliberately keep their ordinary
 * `extends NodeMaterial` shape. A class body cannot be evaluated before an
 * `await` resolves, which makes it tempting to convert them all into
 * `createXTSLMaterialClass()` factories — but that is unnecessary here and
 * would push `async` through ~20 unit test files for no payload benefit. What
 * matters is that nothing reaches those modules *except* through this one
 * dynamic import; their class bodies then evaluate when the lazy chunk loads,
 * which is after the await.
 *
 * Ordering contract: {@link loadTslMaterials} must resolve before the first
 * material is constructed on the WebGPU path. `SceneManager.init` already
 * satisfies this — it awaits `setupRenderer()` (where the install happens)
 * before `setupPostProcessing()`, which builds the first material of the whole
 * app. That is why `buildMaterial` can stay synchronous.
 *
 * Synchronous consumers read the loaded registry from `./slot`, which is a
 * zero-import leaf so the GLSL shader modules can reach it without closing a
 * dependency cycle — see that module's header.
 *
 * @module rendering/tsl/load
 */

import { log } from '../../utils/log';

import { areTslMaterialsLoaded, clearTslMaterialsForTests, setTslMaterials } from './slot';

import type { TslRegistry } from './registry';

/**
 * The in-flight load, cached so concurrent callers share one chunk fetch.
 * Cleared on rejection so a transient chunk-load failure can be retried
 * (the same shape as `data/loaders/once-init.ts`, which lives a layer above
 * `rendering/` and so cannot be imported here).
 */
let inFlight: Promise<TslRegistry> | null = null;

/** The resolved registry, kept alongside the slot for the fast path. */
let loaded: TslRegistry | null = null;

/**
 * Load the TSL / WebGPU material cone, fetching its chunk on first call.
 *
 * Idempotent and concurrency-safe: repeat calls return the cached registry (or
 * join the in-flight load) rather than re-importing.
 */
export async function loadTslMaterials(): Promise<TslRegistry> {
  if (loaded) return loaded;
  if (inFlight) return inFlight;

  inFlight = (async () => {
    // A literal specifier, deliberately: a computed one would defeat
    // rolldown's static analysis (no lazy chunk) and make knip report the
    // registry as an unused file.
    const mod = await import('./registry');
    loaded = mod.TSL_REGISTRY;
    setTslMaterials(loaded);
    log.load('TSL', 'WebGPU material registry loaded (three/webgpu chunk fetched)');
    return loaded;
  })();

  try {
    return await inFlight;
  } catch (error) {
    inFlight = null;
    throw error;
  }
}

/**
 * Drop the cached registry so the next `loadTslMaterials()` re-imports.
 *
 * For tests that need to observe the unloaded state. Production never unloads.
 */
export function resetTslMaterialsForTests(): void {
  loaded = null;
  inFlight = null;
  clearTslMaterialsForTests();
}

export { areTslMaterialsLoaded };
