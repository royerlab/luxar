/**
 * The single list of WASM exports added after the initial kernel set.
 *
 * A stale gitignored `public/wasm/` build can still import and initialise
 * successfully while missing them, otherwise failing later at first use — where
 * the symptom is an opaque "x is not a function" rather than "your WASM build
 * is old".
 *
 * - `compute_joint_codes` — added with the line cap-suppression kernel.
 * - `mesh_vertex_visibility_mask` / `compact_visible_faces` — added with the
 *   mesh culling kernels.
 *
 * Add a name here when you add a kernel, so a stale build is diagnosed rather
 * than silently half-working.
 *
 * ## Why this lives in its own module
 *
 * Two consumers need it and neither may duplicate it — a second copy would rot
 * independently, which is the exact failure this list exists to prevent:
 *
 * 1. {@link ../wasm/index.ts} asserts the names at load time, so a stale module
 *    enters the TypeScript fallback path (correct, slower) instead of throwing
 *    at first use.
 * 2. `src/tests/global-setup.ts` treats a build MISSING one of these names as
 *    equivalent to no build at all, and rebuilds it. Test setup runs in plain
 *    Node before any browser environment exists, so it cannot import
 *    `wasm/index.ts` — hence a module whose only import is a type (erased at
 *    runtime, so nothing browser-shaped is pulled in).
 *
 * The `satisfies` clause is what makes the list trustworthy: a name that is not
 * a real `WasmModule` member fails to compile here rather than silently never
 * matching at runtime.
 */
import type { WasmModule } from './types';

export const REQUIRED_WASM_EXPORTS = [
  'compute_joint_codes',
  'mesh_vertex_visibility_mask',
  'compact_visible_faces',
] as const satisfies readonly (keyof WasmModule)[];
