# Materials — Per-Geometry Material Stacks

> The per-geometry material subtree: parallel Point / Line / GSplat / Mesh stacks plus the shared infrastructure that keeps them symmetric. Every visual shader in Luxar ships from this folder as a GLSL3 `ShaderMaterial` + TSL `NodeMaterial` pair behind a single `ShaderSource`.

## Overview

This is a pure container — no `.ts` files of its own. Its job is to organise
the **eight visual shader wrappers** (4 geometries × {GLSL, TSL}) into a
one-for-one symmetric layout: each geometry kind owns a leaf folder with the
**same four-file shape** (`shader-glsl.ts`, `shader-tsl.ts`, `material-glsl.ts`,
`material-tsl.ts`), implements the same `ColormapAwareMaterial` contract, exposes
the same public update surface (`updateOpacity`, `updateGamma`,
`applyBlendingMode`, `clone`, …), and is paired across backends by a single
`ShaderSource` registry entry. The cross-cutting helpers live one level down in
`_shared/`.

`CameraAwareMaterial` is the one contract all four implement, but mesh takes only
**half** of it: Point / Line / GSplat compute a screen-space sprite extent and
need fov/resolution/ortho broadcast to them every time the camera changes, while
a mesh's size _is_ its geometry, so it ignores fov/resolution and consumes only
`isOrtho`/`nearCull` — the two inputs of the shared near fade, which applies to a
surface exactly as it does to a sprite (#1431). All four therefore live in the
manager's camera-broadcast `registeredMaterials`; `staticMaterials` survives only
as the fallback for a non-camera-aware `register()` caller, which no geometry
material is any more.

The dual-backend pattern is the load-bearing structural choice: `WebGLRenderer`
dispatches the GLSL3 strings via `THREE.ShaderMaterial`; `WebGPURenderer`
dispatches the TSL factory via `NodeMaterial`. `_shared/material-builder.ts`
branches on `RendererCapabilities.apiSurface` so callers (`MaterialManager`,
`NodeFactory`, `LayersPanel`) never see the divergence. The two backends are
kept byte-for-byte equivalent by `tsl-shader-parity.spec.ts`; per project memory
("Keep GLSL shaders as reference"), the GLSL3 sources are **never deleted** —
they remain the readable spec for the rendering math.

The picking counterparts in `../picking/{point,line,gsplat}/` mirror this same
layout one-for-one (four-file per-geometry trio implementing
`CameraAwareMaterial`), and reuse `_shared/`'s math helpers
(`camera-uniforms.ts`, `glsl-lib.ts`) so screen-space hit tests match what the
user sees.

## Layout

```
materials/
├── _shared/      # Cross-cutting helpers: ShaderSource, buildMaterial,
│                 #   CameraAwareMaterial / ColormapAwareMaterial contracts,
│                 #   camera-uniforms math, GLSL/TSL sanitisers, proxyIUniform
├── point/        # Point sprites — 4 files: shader-{glsl,tsl}, material-{glsl,tsl}
├── line/         # Thick lines — 6 files (screen-space quad + capsule primitive pairs)
├── gsplat/       # Gaussian splats — 4 files + math.ts (ray-integral helper)
└── mesh/         # Triangle surfaces — 4 files + appearance.ts (defaults + mode map)
```

## Subpackages

- [`_shared/`](./_shared/README.md) — Geometry-agnostic infrastructure: the
  `ShaderSource` GLSL/TSL parity registry, `buildMaterial(source, config, caps)`
  backend dispatch, `CameraAwareMaterial` / `ColormapAwareMaterial` marker
  interfaces broadcast by `MaterialManager`, the in-shader projection helpers
  (`luxarProjectionSizeScale` / `luxarIsOrthoProjection` in `glsl-lib.ts`, their
  TSL twins in `tsl-helpers.ts`) with their CPU mirror `projection-math.ts`,
  `GLSL_SANITIZE_FUNCTIONS` and
  their TSL counterparts, and `proxyIUniform(node)` — the wrapper that lets
  `material.uniforms.uX.value = Y` land on a TSL `UniformNode` without a
  per-render callback bridge.

- [`point/`](./point/README.md) — Soft-edged sprite shader for Luxar's
  **Points** geometry. One instanced unit-quad per point, world-space FOV-
  independent sizing, per-point sharpness mapped to a shifted-truncated
  super-Gaussian falloff exponent (`β = 2^(6s − 2)`, no size compensation —
  the kernel truncates at the sprite edge), per-node Gain/Offset/Gamma,
  optional `USE_COLORMAP` LUT branch, and zero-radius-discard nD slicing.

- [`line/`](./line/README.md) — Thick-line material with instanced screen-space
  quad expansion, a shifted-truncated super-Gaussian perpendicular
  cross-section `max(exp(−K·p^β) − C, 0)/(1 − C)` (sharpness a `[0, 1]` knob,
  `β = 2^(6s − 2)`, default `s = 0.5 → β = 2` Gaussian), and the cap-factor
  joint trick that makes adjacent segments sum to exactly `1.0` under additive
  blending. Four variant defines (`USE_COLORMAP`, `LUXAR_GAMMA_ONE`,
  `LUXAR_NO_GOG`, `LUXAR_MAX_RGB_CONTRIBUTION`) drive fast paths.

- [`gsplat/`](./gsplat/README.md) — Volumetric Gaussian-splat material with
  oriented instanced quads, full 3D covariance via Cholesky factors,
  perspective-Jacobian projection to 2D screen-space covariance, and a
  shifted Gaussian (`exp(−½·r²) − C`) with C⁰ continuity at truncation. Sum
  vs max projection branches; backend-divergent additive blending documented
  in detail.

- [`mesh/`](./mesh/README.md) — The **shaded** one, and the only stack that
  draws a plain indexed `BufferGeometry` rather than instanced quads. A
  light-free view-anchored offset key with wrapped diffuse and a subtle additive Blinn–Phong highlight over stored
  view-space normals or a screen-space-derivative fallback, chosen as a
  compile-time variant; per-vertex alpha as the sole coverage term; and a
  three-way emission (hard cutout for the `opaque` default, premultiplied for
  `max`, alpha-weighted otherwise). `appearance.ts` holds the defaults and the
  mode→emission map both backends read.

## Four-Geometry Symmetry

Per project memory ("Three-geometry symmetry rule", now four), Points / Lines /
GSplats / Mesh share **the same names, the same decomposition, the same shared
helpers, and parallel tests**. The materials subtree is one of the strongest
expressions of that rule — and where Mesh diverges, it does so by NAME rather
than by omission (each bullet below records its own exception):

- **Same file names per geometry**: each leaf is exactly
  `{shader,material}-{glsl,tsl}.ts` (plus a fifth module only where the geometry
  has standalone helpers — `gsplat/math.ts`, `mesh/appearance.ts`).
- **Same public surface**: `updateOpacity`, `updateGamma`, `updateIntensity`,
  `updateOffset`, `applyBlendingMode`, `clone`, `setColormapTexture`,
  `setScalarRange` — plus `updateCameraParams` on the three instanced-quad
  types, and `updateFlatNormal` on Mesh alone (nothing else shades).
- **Same shared helpers** from `_shared/`: every wrapper consumes
  `clampGamma`, the sanitiser snippets, `CameraAwareMaterial`, and
  `ColormapAwareMaterial`. Points, Lines and GSplats read their projection
  terms (size scale, ortho test, focal length / Jacobian) in shader from the
  projection matrix three binds per draw; Point additionally consumes
  `computeMaxPointSize` from `camera-uniforms.ts`.
- **Parallel picking counterparts** in `../picking/{point,line,gsplat,mesh}/`
  with the same four-file shape — except Mesh, a six-file variant: its element
  ordinal is `gl_VertexID` rather than an element-texture texel, which adds
  `provoking-vertex.ts` (the two backends read a `flat` varying from different
  triangle corners) and `pick-mode.ts` (a surface's cutout and depth
  consequences derived in one place) (MESH_NODE_SPEC.md §6.5).
- **Single parity harness** (`tsl-shader-parity.spec.ts`) renders the same
  scene through both backends and pixel-compares for every geometry.

When adding a new geometry kind, this is the structural template — populate a
new leaf folder with the four canonical files, implement the same interfaces,
delegate cross-cutting concerns to `_shared/`, and add the parity test case.

## See Also

- `../README.md` — Rendering package overview; materials are components 2–4
  ("Point Material", "Line Material", "GSplat Material") of the larger pipeline
- `../material-manager.ts` — `getPointMaterial` / `getLineMaterial` /
  `getGSplatMaterial` / `getMeshMaterial` dispatch on `caps.apiSurface`, and the
  first three broadcast camera updates via the `CameraAwareMaterial` interface
- `../node-factory/` — `create-{points,lines,gsplats}-node.ts` pair these
  materials with their `InstancedBufferGeometry` siblings;
  `create-mesh-node.ts` pairs the mesh material with a plain indexed
  `BufferGeometry` and owns the shading-variant decision
- `../picking/` — Picking subsystem mirrors this layout one-for-one
- `../blending-state.ts` — `getCompleteBlendingState` /
  `applyBlendingStateToMaterial`, the single source of truth for THREE blending
  state that every `applyBlendingMode` routes through
- `../shaders/` — Barrel re-exporting per-geometry GLSL constants for the TSL
  parity harness's import path
- `../../tests/e2e/tsl-shader-parity.spec.ts` — the GLSL↔TSL parity harness
  this folder's `ShaderSource` pattern exists to serve
