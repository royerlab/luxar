/**
 * Shared TSL helper functions used by the geometry-material factories.
 *
 * The "sanitise" helpers reproduce the GLSL `sanitizePositive` /
 * `sanitizeNonNegative` shape — guard against NaN/Inf produced by
 * upstream data loaders and fall back to a sensible scalar default.
 * They are pure TSL builder calls, so the same body works under any
 * NodeBuilder (WebGL2 or WebGPU).
 *
 * @module rendering/tsl-helpers
 */

/**
 * Loosely-typed TSL node alias. TSL's typed overloads return many
 * mutually-incompatible inner constructor types; relaxing at helper
 * boundaries lets the runtime TSL builder do the real type checking
 * when it compiles to GLSL/WGSL.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TSLNode = any;

/** Sanitise a positive scalar. Mirrors GLSL `sanitizePositive`. */
export function sanitizePositive(value: TSLNode, fallback: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  const isPositive = value.greaterThan(0.0);
  return isFinite.and(isPositive).select(value, fallback);
}

/** Sanitise a non-negative scalar. Mirrors GLSL `sanitizeNonNegative`. */
export function sanitizeNonNegative(value: TSLNode, fallback: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  const isNonNeg = value.greaterThanEqual(0.0);
  return isFinite.and(isNonNeg).select(value, fallback);
}
