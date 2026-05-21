# Fullscreen rendering primitives

Shared infrastructure used by every post-processing effect. A fullscreen pass
needs two pieces — a triangle that covers the entire NDC region with a single
draw call, and a `Scene + Camera + Mesh` triplet that wraps a material and
exposes a `render(renderer)` method. Both are caps-aware: the `UV` attribute on
the geometry inverts on WebGPU so the same shader code samples the same texel.

| File          | Role                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------- |
| `geometry.ts` | `createFullscreenTriangleGeometry(caps)` — single triangle covering `[-1,-1]` → `[3,3]` NDC  |
| `pass.ts`     | `FullscreenPass` class — owns a `Scene + OrthographicCamera + Mesh(geometry, material)` triplet, exposes `setMaterial()`, `render()`, `dispose()` |

Consumers: `bloom/chain.ts`, `fxaa/pass.ts`, `post-processing-manager.ts` (mega pass),
and `post-processing-manager/resource-lifecycle.ts` (builds the mega pass at boot).
