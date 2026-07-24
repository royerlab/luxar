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
import { float, int, smoothstep } from 'three/tsl';

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
 * Per-element opacity sanitizer. Mirrors GLSL `sanitizeAlpha`: NaN/Inf
 * route to the 1.0 opaque identity (corruption stays LOUD), finite
 * values clamp to [0, 1] — alpha is opacity, never HDR (Python pins the
 * range at write; this guards hand-crafted zarr). The clamp keeps the
 * zero boundary CONTINUOUS (a -1e-4 epsilon vanishes like +0.0 renders,
 * instead of jumping to full opacity) and keeps the value
 * mediump-varying-safe.
 */
export function sanitizeAlpha(value: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  return isFinite.select(value.max(0.0).min(1.0), float(1.0));
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

/**
 * Unified perspective near-plane fade (runtime-uniform variant, for
 * the point/gsplat graphs whose ortho flag is the `uIsOrtho` uniform).
 * Mirrors GLSL `perspectiveNearFade` in glsl-lib.ts: perspective =
 * 0.0 behind the camera, smoothstep across [nearCull, 2*nearCull];
 * ortho = 1.0 always (NDC clipping is the sole cull authority).
 * Callers reject the vertex when the result < 0.01 and multiply the
 * surviving amplitude/alpha/brightness by it.
 */
export function perspectiveNearFadeTSL(
  uIsOrtho: TSLNode,
  viewZ: TSLNode,
  uNearCull: TSLNode
): TSLNode {
  const fade = smoothstep(uNearCull, uNearCull.mul(2.0), viewZ.negate());
  const persp = viewZ.greaterThanEqual(0.0).select(float(0.0), fade);
  return int(uIsOrtho).equal(int(1)).select(float(1.0), persp);
}

/**
 * Compile-time-ortho variant for the line graphs (their ortho flag is
 * the factory `config.isOrtho`, baked into the graph): ortho variants
 * carry NO fade/cull code at all.
 */
export function perspectiveNearFadeStaticTSL(
  isOrtho: boolean,
  viewZ: TSLNode,
  uNearCull: TSLNode
): TSLNode {
  if (isOrtho) return float(1.0);
  return viewZ
    .greaterThanEqual(0.0)
    .select(float(0.0), smoothstep(uNearCull, uNearCull.mul(2.0), viewZ.negate()));
}
