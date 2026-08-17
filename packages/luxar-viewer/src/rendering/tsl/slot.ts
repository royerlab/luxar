/**
 * The holder for the lazily-loaded TSL registry — a zero-runtime-import leaf.
 *
 * Split from `load.ts` for one structural reason: the `ShaderSource.webgpu`
 * closures in the GLSL shader modules need `requireTslMaterials()`, and if that
 * lived in `load.ts` those modules would gain a graph edge to
 * `load.ts → registry.ts → *-tsl.ts → *-glsl.ts` and close a cycle. The cycle
 * is broken at runtime by the dynamic import, but `no-circular` in
 * `.dependency-cruiser.cjs` is an error-severity rule and counts a dynamic edge
 * like any other — correctly, since a cycle that only works because of import
 * timing is exactly the kind of thing that rule exists to stop people relying
 * on.
 *
 * So the accessor lives here, importing nothing at runtime, and `load.ts` is
 * the only module that both reaches `registry.ts` and writes this slot. Same
 * shape and same motivation as `material-manager/soft-dispose-flag.ts`.
 *
 * @module rendering/tsl/slot
 */

import type { TslRegistry } from './registry';

/** Resolved registry, or `null` before the first successful load. */
let registry: TslRegistry | null = null;

/** Publish the loaded registry. Called only by `load.ts`. */
export function setTslMaterials(loaded: TslRegistry): void {
  registry = loaded;
}

/**
 * The already-loaded registry, for synchronous consumers.
 *
 * Throws when the WebGPU path is active but `loadTslMaterials()` has not
 * resolved yet. This **must not** degrade to the GLSL implementations: a
 * `ShaderMaterial` does not render at all under `WebGPURenderer` (see
 * `BROWSER_SUPPORT_POLICY.md`), so a silent fallback would paint blank quads
 * and read as a rendering bug rather than the wiring bug it is.
 */
export function requireTslMaterials(): TslRegistry {
  if (!registry) {
    throw new Error(
      'requireTslMaterials(): the TSL/WebGPU material registry has not been ' +
        'loaded. It is installed by `await loadTslMaterials()` in ' +
        'SceneManager.setupWebGPURenderer(), which runs before any material ' +
        'is constructed. Reaching this means a WebGPU-path material was built ' +
        'without that await — call `await loadTslMaterials()` first (tests ' +
        'that construct TSL materials directly need it in a beforeAll).'
    );
  }
  return registry;
}

/** Whether the registry has been loaded. */
export function areTslMaterialsLoaded(): boolean {
  return registry !== null;
}

/**
 * Drop the cached registry.
 *
 * For tests that need to observe the unloaded state (the `requireTslMaterials`
 * throw, or a fresh `loadTslMaterials` call). Production never unloads: the
 * chunk stays in memory for the life of the page.
 */
export function clearTslMaterialsForTests(): void {
  registry = null;
}
