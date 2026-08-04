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
