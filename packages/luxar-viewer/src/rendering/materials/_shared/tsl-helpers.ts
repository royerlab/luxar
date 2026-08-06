/**
 * Shared TSL helper functions used by the geometry-material factories.
 *
 * `sanitizeNonNegative` reproduces the GLSL function of the same name —
 * guard against NaN/Inf produced by upstream data loaders and fall back to a
 * sensible scalar default. It is a pure TSL builder call, so the same body
 * works under any NodeBuilder (WebGL2 or WebGPU). (`glsl-lib.ts` also defines
 * a `sanitizePositive`; no TSL shader calls it, so there is no TSL mirror.)
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
import { attribute, float, int, smoothstep } from 'three/tsl';

/**
 * Loosely-typed TSL node alias. TSL's typed overloads return many
 * mutually-incompatible inner constructor types; relaxing at helper
 * boundaries lets the runtime TSL builder do the real type checking
 * when it compiles to GLSL/WGSL.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TSLNode = any;

/**
 * Double-buffered draw-slot → storage-slot index — the TSL twin of
 * `GLSL_SORTED_INDEX` (depth-sorting spec §2.1 tier 3).
 *
 * A permutation must swap ATOMICALLY: a half-applied ordering is not a
 * reordering but a corrupt permutation (elements drawn twice / not at
 * all). Each new ordering therefore streams into the INACTIVE attribute
 * across frames, and `uSortedIndexSlot` flips only once that buffer holds
 * the whole permutation.
 *
 * BOTH attributes are referenced unconditionally, which is required, not
 * merely tidy: the WebGPU backend only uploads graph-referenced
 * attributes, so an unreferenced back buffer would never receive its
 * chunked uploads — and `RenderObject` dereferences a referenced
 * attribute before its undefined guard, so every geometry must carry
 * both. `attachElementStorage` allocates them as two DISTINCT buffers
 * together — never aliased, never added later, since the WebGPU vertex
 * layout is cached from the attribute set at first draw and a set that
 * grows afterwards renders the scene black.
 *
 * `uSortedIndexSlot` must stay a RUNTIME uniform: a compile-time flag
 * would rebuild the graph on every swap. (The lines material treats
 * `uIsOrtho` as compile-time — deliberately NOT the pattern here.)
 *
 * BRANCHLESS on purpose. `.select()` emits an if/else STATEMENT, and the
 * pick factories consume this index at `varying(float(...))` — evaluated
 * OUTSIDE their `Fn()` body, where a statement cannot legally land. That
 * produced a shader which built and code-generated fine but rendered
 * zero pixels (the same vacuous-black failure mode the gsplat harness
 * comment in `tests/e2e/harnesses/tsl-harness/gsplats.ts` warns about).
 * A pure arithmetic mix is position-independent, so the one helper is
 * safe both inside and outside an `Fn()`. The multiplier is a uniform,
 * so there is no per-vertex divergence to save by branching anyway.
 *
 * Returns an INT node (not uint): the downstream `int(...)` wraps and
 * `float(...)` casts accept it unchanged.
 */
export function sortedIndexNode(uSortedIndexSlot: TSLNode): TSLNode {
  const slot: TSLNode = int(uSortedIndexSlot);
  const a: TSLNode = int(attribute<'uint'>('aSortedIndex', 'uint'));
  const b: TSLNode = int(attribute<'uint'>('aSortedIndexB', 'uint'));
  return a.mul(int(1).sub(slot)).add(b.mul(slot));
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
 * backends, so we approximate via the same finite-range test
 * `sanitizeNonNegative` uses: any value outside (-1e30, 1e30) is
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

/**
 * TSL counterpart of `GLSL_LINE_JOINT_CODE`'s
 * `luxarLineJointCapSuppression` — the endpoint cap multiplier implied by a
 * per-endpoint joint code (`texel4.yz`; see that GLSL block for the encoding
 * and for why a slot-bearing code must suppress rather than keep the cap).
 *
 * Emitted as a node expression rather than a TSL `Fn()` so it composes inside
 * the line factories' single traced vertex body, where a structural branch is
 * deliberately avoided.
 */
export function tslLineJointCapSuppression(jointCode: TSLNode): TSLNode {
  const isFreeEnd = jointCode.greaterThan(-0.5).and(jointCode.lessThan(0.5));
  const isHub = jointCode.lessThan(-1.5).and(jointCode.greaterThan(-2.5));
  return isFreeEnd.or(isHub).select(float(0.0), float(1.0));
}
