# Materials — Per-Geometry Material Stacks

> The three-geometry material subtree: parallel Point / Line / GSplat stacks plus the shared infrastructure that keeps them symmetric. Every visual shader in Luxar ships from this folder as a GLSL3 `ShaderMaterial` + TSL `NodeMaterial` pair behind a single `ShaderSource`.

## Overview

This is a pure container — no `.ts` files of its own. Its job is to organise
the **six visual shader wrappers** (3 geometries × {GLSL, TSL}) into a
one-for-one symmetric layout: each geometry kind owns a leaf folder with the
**same four-file shape** (`shader-glsl.ts`, `shader-tsl.ts`, `material-glsl.ts`,
`material-tsl.ts`), implements the same `CameraAwareMaterial` /
`ColormapAwareMaterial` contracts, exposes the same public update surface
(`updateOpacity`, `updateGamma`, `applyBlendingMode`, `clone`, …), and is paired
across backends by a single `ShaderSource` registry entry. The cross-cutting
helpers live one level down in `_shared/`.

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
├── line/         # Thick lines — 4 files (instanced quad expansion in pixel space)
└── gsplat/       # Gaussian splats — 4 files + math.ts (ray-integral helper)
```

## Subpackages

- [`_shared/`](./_shared/README.md) — Geometry-agnostic infrastructure: the
  `ShaderSource` GLSL/TSL parity registry, `buildMaterial(source, config, caps)`
  backend dispatch, `CameraAwareMaterial` / `ColormapAwareMaterial` marker
  interfaces broadcast by `MaterialManager`, shared `pointSizeFactor` /
  `focalLength` math in `camera-uniforms.ts`, `GLSL_SANITIZE_FUNCTIONS` and
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

## Three-Geometry Symmetry

Per project memory ("Three-geometry symmetry rule"), Points / Lines / GSplats
share **the same names, the same decomposition, the same shared helpers, and
parallel tests**. The materials subtree is one of the strongest expressions of
that rule:

- **Same file names per geometry**: each leaf is exactly
  `{shader,material}-{glsl,tsl}.ts` (plus `math.ts` only where the geometry
  has standalone numerical helpers — currently `gsplat/`).
- **Same public surface**: `updateOpacity`, `updateGamma`, `updateIntensity`,
  `updateOffset`, `updateCameraParams`, `applyBlendingMode`, `clone`,
  `setColormapTexture`, `setScalarRange`.
- **Same shared helpers** from `_shared/`: every wrapper consumes
  `clampGamma`, the sanitiser snippets, `CameraAwareMaterial`, and
  `ColormapAwareMaterial`. Point and GSplat additionally consume
  `computePointSizeFactor` / `computeMaxPointSize` / `computeFocalLength`
  from `camera-uniforms.ts`.
- **Parallel picking counterparts** in `../picking/{point,line,gsplat}/` with
  the same four-file shape.
- **Single parity harness** (`tsl-shader-parity.spec.ts`) renders the same
  scene through both backends and pixel-compares for every geometry.

When adding a new geometry kind, this is the structural template — populate a
new leaf folder with the four canonical files, implement the same interfaces,
delegate cross-cutting concerns to `_shared/`, and add the parity test case.

## See Also

- `../README.md` — Rendering package overview; materials are components 2–4
  ("Point Material", "Line Material", "GSplat Material") of the larger pipeline
- `../material-manager.ts` — `getPointMaterial` / `getLineMaterial` /
  `getGSplatMaterial` dispatch on `caps.apiSurface` and broadcast camera
  updates to every registered material via the `CameraAwareMaterial` interface
- `../node-factory/` — `create-{points,lines,gsplats}-node.ts` are the callers
  that pair these materials with their `InstancedBufferGeometry` siblings
- `../picking/` — Picking subsystem mirrors this layout one-for-one
- `../blending-state.ts` — `getCompleteBlendingState` /
  `applyBlendingStateToMaterial`, the single source of truth for THREE blending
  state that every `applyBlendingMode` routes through
- `../shaders/` — Barrel re-exporting per-geometry GLSL constants for the TSL
  parity harness's import path
- `../../tests/e2e/tsl-shader-parity.spec.ts` — the GLSL↔TSL parity harness
  this folder's `ShaderSource` pattern exists to serve
