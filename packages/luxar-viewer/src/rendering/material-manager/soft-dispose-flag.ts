/**
 * The soft-dispose sentinel, isolated in a leaf module.
 *
 * `invalidateRenderObjectFor` tags a mesh's material with this flag and
 * then dispatches a `dispose` event on it to evict Three's cached WebGPU
 * `RenderObject`; the `MaterialManager` lifecycle listener checks for the
 * flag and skips its own registry teardown (the material is being kept
 * alive).
 * Dispatcher (`data/scene-loader/commit/invalidate-render-object`) and
 * listener (`material-manager/lifecycle`) stay name-coupled through this
 * single definition.
 *
 * It lives on its own so the dispatcher — reachable from the renderer
 * bootstrap — can import the sentinel without pulling in the material
 * factory graph (the TSL materials import `three/webgpu`, which the
 * renderer-setup unit test intentionally stubs).
 *
 * Symbol-keyed so the flag can't collide with Three's internal
 * properties or with userspace `userData` keys, and so it's invisible
 * to enumeration / serialization.
 *
 * @module rendering/material-manager/soft-dispose-flag
 */

export const SOFT_DISPOSE_FLAG = Symbol.for('luxar.invalidateRenderObject.softDispose');
