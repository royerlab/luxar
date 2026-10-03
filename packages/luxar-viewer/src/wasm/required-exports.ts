/**
 * The single list of WASM exports added — or re-signatured — after the initial
 * kernel set.
 *
 * A stale gitignored `public/wasm/` build can still import and initialise
 * successfully while missing them (or exporting an older signature), otherwise failing later at first use — where
 * the symptom is an opaque "x is not a function" rather than "your WASM build
 * is old".
 *
 * - `compute_joint_codes` — added with the line cap-suppression kernel.
 * - `mesh_vertex_visibility_mask` / `compact_visible_faces` — added with the
 *   mesh culling kernels.
 * - `project_gsplats_nd_to_3d` — present from the start, but it gained the
 *   trailing `out_source_indices` parameter, so its parameter count is checked
 *   too ({@link REQUIRED_WASM_ARITIES}).
 *
 * Add a name here when you add a kernel, and its parameter count to
 * {@link REQUIRED_WASM_ARITIES} when you change a kernel's signature, so a
 * stale build is diagnosed rather than silently half-working — and REMOVE it
 * when you rename or delete that kernel, or every consumer reports a freshly
 * built artifact as "stale" and sends the reader to rebuild it in a loop. Only free functions belong here:
 * the names are matched against the raw `.wasm` exports as well as the shim's,
 * and wasm-bindgen mangles anything else (a struct method exports as
 * `<struct>_<method>`).
 *
 * ## Why this lives in its own module
 *
 * Two consumers need it and neither may duplicate it — a second copy would rot
 * independently, which is the exact failure this list exists to prevent:
 *
 * 1. `./index.ts` asserts the names at load time, so a stale module enters the
 *    TypeScript fallback path (correct, slower) instead of throwing at first
 *    use.
 * 2. `../tests/global-setup.ts` treats a build MISSING one of these names as
 *    equivalent to no build at all, and rebuilds it. Test setup runs in plain
 *    Node before any browser environment exists, so it cannot import
 *    `./index.ts` — hence a module whose only import is a type (erased at
 *    runtime, so nothing browser-shaped is pulled in).
 */
import type { WasmModule } from './types';

/**
 * WASM exports added after the initial kernel set, which a stale build may not
 * have.
 *
 * `initWasm` asserts these at load time and `global-setup` rebuilds when one is
 * absent — see the module comment above for why the list lives here rather than
 * in either consumer.
 *
 * Add a name here when you add a kernel. The `satisfies` clause makes that safe:
 * a name that is not a real {@link WasmModule} member fails to compile rather
 * than silently never matching at runtime.
 */
export const REQUIRED_WASM_EXPORTS = [
  'compute_joint_codes',
  'mesh_vertex_visibility_mask',
  'compact_visible_faces',
  'project_gsplats_nd_to_3d',
] as const satisfies readonly (keyof WasmModule)[];

type RequiredExport = (typeof REQUIRED_WASM_EXPORTS)[number];

/** The parameter count of `WasmModule[K]`'s signature. */
type Arity<K extends keyof WasmModule> = WasmModule[K] extends (...args: infer P) => unknown
  ? P['length']
  : never;

/**
 * Parameter count of each required export whose SIGNATURE changed after it
 * first shipped. A stale build still exports the name, and its JS wrapper
 * silently ignores an argument it does not declare: an old
 * `project_gsplats_nd_to_3d` never writes `out_source_indices`, so picking
 * reads zeros. wasm-bindgen emits one named JS parameter per Rust parameter,
 * so the wrapper's `Function.length` (and the parameter list of its
 * `export function` declaration) is the count to check — on the shim only; the
 * raw `.wasm` export splits each slice into a pointer and a length.
 *
 * The `satisfies` clause ties every count to the {@link WasmModule} signature:
 * changing the signature without updating the count fails to compile.
 */
export const REQUIRED_WASM_ARITIES = {
  project_gsplats_nd_to_3d: 18,
} as const satisfies { readonly [K in RequiredExport]?: Arity<K> };

/** The parameter count `name` must declare, or `undefined` when only presence is checked. */
export function requiredWasmArity(name: RequiredExport): number | undefined {
  return (REQUIRED_WASM_ARITIES as Partial<Record<RequiredExport, number>>)[name];
}
