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
