/**
 * Shared TSL helper functions used by the geometry-material factories.
 *
 * The "sanitise" helpers reproduce the GLSL `sanitizePositive` /
 * `sanitizeNonNegative` shape — guard against NaN/Inf produced by
 * upstream data loaders and fall back to a sensible scalar default.
 * They are pure TSL builder calls, so the same body works under any
 * NodeBuilder (WebGL2 or WebGPU).
 *
 * `proxyIUniform` bridges the THREE `IUniform`-shaped public API
 * (`material.uniforms.uX.value = Y`) directly onto a TSL
 * `UniformNode.value` getter/setter. It replaces the old
 * `uniform(value).onUpdate(() => iuniform.value, 'render')` bridge,
 * which read the host IUniform's value into the node before every
 * render. That callback was structural noise — the IUniform record
 * could *be* the node, just dressed in the IUniform shape. The proxy
 * forwards reads and writes directly to the node so there is no
 * per-frame callback overhead and no two-source-of-truth class.
 *
 * @module rendering/tsl-helpers
 */

import type { IUniform } from 'three';

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

/**
 * Boolean TSL node: true when `value` is NaN or +/-Inf. Mirrors GLSL
 * `isInvalidFloat`. TSL has no direct `isnan`/`isinf` exposed across
 * backends, so we approximate via the finite-range test that
 * `sanitizePositive` already uses: any value outside (-1e30, 1e30) is
 * treated as non-finite. The same pattern is used by the visual
 * point/shader-tsl.ts sharpness-compensation guard.
 */
export function invalidFloatTSL(value: TSLNode): TSLNode {
  return value.lessThan(1e30).and(value.greaterThan(-1e30)).not();
}

/**
 * Wrap a TSL `UniformNode` in an `IUniform`-shaped getter/setter so
 * callers can keep using `material.uniforms.uX.value = Y` while the
 * read/write is routed straight to `node.value`. See the module
 * preamble for why this replaces the older `.onUpdate('render')`
 * bridge.
 *
 * The node argument is typed loosely as `TSLNode` because TSL's
 * typed-overload surface returns many concrete inner classes; the
 * runtime contract is just that `node.value` is a readable & writable
 * property of type `T`.
 */
export function proxyIUniform<T>(node: TSLNode): IUniform<T> {
  return {
    get value(): T {
      return node.value as T;
    },
    set value(v: T) {
      node.value = v;
    },
  } as IUniform<T>;
}
