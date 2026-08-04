/**
 * Global constants shared across viewer subsystems.
 *
 * Anything here is a value that must stay literally equal across multiple
 * TS modules AND the Rust/WASM implementation. Prefer the AppConfig in
 * `./index.ts` for tunables that have a single owner.
 */

/**
 * Maximum number of nD dimensions supported by WASM-accelerated paths.
 *
 * Rust mirrors this as `MAX_SUPPORTED_DIMS` in
 * `src/wasm/rust/src/common.rs`. The bound comes from fixed-size stack
 * arrays in the Rust kernels (effective radii, marginal Cholesky,
 * Mahalanobis distance). When a dataset reports ndim > MAX_SUPPORTED_DIMS,
 * the TypeScript fallback path in `src/wasm/typescript/` handles it.
 *
 * Changing this value REQUIRES rebuilding the WASM module.
 */
export const MAX_SUPPORTED_DIMS = 16;

/**
 * GSplat truncation radius `T`, in sigmas: the Mahalanobis distance beyond
 * which a splat's shifted Gaussian is exactly zero. The kernel is
 * `max(0, exp(-D²/2) - C) / (1 - C)` with `C = exp(-T²/2)`, so `T` sets both
 * the support and the normalization `1 / (1 - C)`.
 *
 * Used as the fallback when a node carries no `truncation_radius` attribute
 * and as the default for materials constructed without one. A fitted dataset
 * always stamps its own value, which wins.
 *
 * MIRROR: `DEFAULT_TRUNCATION_RADIUS` in
 * `packages/luxar/src/luxar/typing_utils/constants.py` must hold the same
 * value. Both sides are pinned by tests that name each other.
 */
export const GSPLAT_DEFAULT_TRUNCATION_RADIUS = 2.75;

/**
 * Largest vertex count a mesh node may declare, `2^27`.
 *
 * This is the pick vote-key stride: mesh's pick `elementId` is `gl_VertexID`
 * — the one geometry type whose element ordinal is not bounded by the
 * element-texture capacity — and the largest ordinal a node contributes is
 * `n_vertices - 1`, so `n_vertices <= 2^27` is the exact alias-free bound.
 * One vertex more and the vote key wraps into a neighbouring node's range,
 * which resolves picks to the wrong node with nothing to signal it.
 *
 * Enforced by the loader's Stage-1 metadata preflight (see
 * `data/mesh/mesh-preflight.ts`), i.e. before any CHUNK is fetched, so an
 * oversized declaration costs no geometry allocation.
 *
 * Precisely "no chunk", not "no allocation": the preflight has to read `.zarray`
 * and `.zattrs` to learn the shapes and encodings it decides on, and a hostile
 * store's `.zattrs` is not itself bounded by anything here — a `lut_uint16`
 * encoding legitimately carries its lookup table inline as a JSON list, so that
 * blob could be made arbitrarily large. Bounding a metadata response is a
 * store-layer concern (every sibling loader reads `.zattrs` the same way), not
 * something this gate can do, so the claim is scoped to what it actually buys.
 *
 * MIRROR: `MAX_MESH_VERTICES` in
 * `packages/luxar/src/luxar/typing_utils/constants.py` must hold the same
 * value — it is the write-time twin of this gate, so that `add_mesh` cannot
 * emit a store Luxar's own viewer then refuses. Both sides are pinned by
 * tests that name each other.
 */
export const MAX_MESH_VERTICES = 134217728;

/**
 * Per-node ceiling, in bytes, on what a mesh node is allowed to *declare*
 * before the loader will fetch any of it. Default 512 MiB.
 *
 * Viewer-only: there is no Python twin, because the write side cannot know
 * what a tab can survive. The viewer loads arbitrary `?src=` URLs, and the
 * mesh loader is whole-node (`docs/specs/MESH_NODE_SPEC.md` §7) — it fetches
 * and decodes every array in full — so a hostile or corrupt store could
 * otherwise exhaust tab memory before the per-node `LoaderError` containment
 * is ever reachable.
 *
 * Three quantities are checked against this, all from `.zarray` metadata alone:
 * every array's stored footprint, what each one DECODES to, and each array's
 * single largest per-chunk allocation.
 *
 * Charging the decoded term is not belt-and-braces — the stored side can be
 * arbitrarily smaller than what gets allocated, so a stored-only budget is wrong
 * in the dangerous direction. A broadcast array stores one row and expands to
 * `n_elements` rows, so a ~12-byte declaration can materialize gigabytes; every
 * decoder-routed array yields a `Float32Array`, so a `uint8` store decodes at 4x
 * its stored bytes; and `faces` widens to u32 whatever narrow dtype the INDEX
 * encoder chose. The decoded term is also the only thing bounding `ndim`, which
 * has no cap of its own.
 *
 * The per-chunk term is likewise not redundant: zarr v2 does not require
 * `chunks <= shape`, so a `"shape": [100, 3]` array declaring
 * `"chunks": [268435456, 3]` would slip a multi-gigabyte first-chunk allocation
 * past a shape-only budget.
 *
 * Be precise about what this bounds: the SUM of stored bytes, decoded bytes, and
 * the largest single chunk buffer — the three are added together, not checked
 * independently, because a chunk buffer exists during decode alongside the arrays.
 * (An oversized chunk is ALSO rejected on its own, so a single array cannot exceed
 * the ceiling even where the sum would fit.) This is still not the loader's whole
 * transient peak: the admission path also holds the extracted display-space
 * `position` and the driver-side GPU upload, so the real peak is a modest multiple
 * of this value.
 * 512 MiB keeps that comfortably inside a 64-bit tab — which is why this must
 * never be raised toward "what a tab survives": the tab has to survive the
 * multiple, not the ceiling.
 *
 * The ceiling is per NODE. N nodes can still sum to N x budget; the guarantee
 * it buys is "one node lost, not the scene", not an aggregate cap.
 */
export const MESH_DECODE_BUDGET_BYTES = 512 * 1024 * 1024;
