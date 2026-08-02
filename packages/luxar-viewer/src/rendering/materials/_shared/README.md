# Shared Material Infrastructure

> Cross-cutting helpers used by the Point / Line / GSplat material stacks (both GLSL `ShaderMaterial` and TSL `NodeMaterial` variants) and their picking counterparts.

This folder holds the small, geometry-agnostic pieces that the per-geometry
material wrappers in `../point/`, `../line/`, and `../gsplat/` compose. Nothing
here renders a pixel on its own — each module either declares a contract the
materials implement (`CameraAwareMaterial`, `ColormapAwareMaterial`, `ShaderSource`),
encapsulates math/string content that would otherwise be copy-pasted across six
material wrappers, or branches the dual-stack WebGL2 / WebGPU dispatch in one
place (`buildMaterial`).

## Module map

| File                         | Role                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shader-source.ts`           | `ShaderSource` registry type: `{ name, webgl?: { vertex, fragment }, webgpu?: factory }` plus `requireWebGLSources()` narrowing helper                                                                                                                                                                                                                                                         |
| `material-builder.ts`        | `buildMaterial(source, config, caps)` — branches on `caps.apiSurface` to return either a `THREE.ShaderMaterial` (GLSL3) or a TSL `NodeMaterial`                                                                                                                                                                                                                                                |
| `camera-aware-material.ts`   | `CameraAwareMaterial` interface + `isCameraAwareMaterial` guard. The contract `updateCameraParams(fov, resolution, isOrtho?, nearCull?)` that `MaterialManager` broadcasts to every registered visual + picking material                                                                                                                                                                       |
| `colormap-aware-material.ts` | `ColormapAwareMaterial` interface + guard. Two setters (`setColormapTexture`, `setScalarRange`) so the colormap helpers never reach into `material.uniforms` directly                                                                                                                                                                                                                          |
| `camera-uniforms.ts`         | Pure math shared by visual + picking materials: `computePointSizeFactor`, `computeMaxPointSize`, `computeFocalLength`. Branches on `isOrtho` so callers don't special-case projection                                                                                                                                                                                                          |
| `uniform-helpers.ts`         | `clampGamma(g)` — single source of truth for the `Math.max(0.001, g ?? 1.0)` clamp used in every material constructor; `isGammaOne(g)` — `abs(g - 1) < 1e-4` fast-path test that gates the `LUXAR_GAMMA_ONE` define (GLSL) / `gammaOne` flag (TSL) so the shader skips `pow(color, 1/gamma)` when gamma is unity                                                                               |
| `glsl-lib.ts`                | `GLSL_SANITIZE_FUNCTIONS` GLSL3 snippet (`isInvalidFloat`, `sanitizePositive`, `sanitizeNonNegative`) prepended to every Point / Line / GSplat visual _and_ picking GLSL shader, plus `GLSL_NEAR_FADE_FUNCTIONS` (`perspectiveNearFade` — the unified near handling all three geometry types share)                                                                                            |
| `tsl-helpers.ts`             | TSL counterparts to the GLSL sanitisers (`sanitizePositive`, `sanitizeNonNegative`, `invalidFloatTSL`), the near-fade twins (`perspectiveNearFadeTSL` runtime-uniform / `perspectiveNearFadeStaticTSL` compile-time-ortho), plus `proxyIUniform(node)` — wraps a TSL `UniformNode` in an `IUniform`-shaped getter/setter so the `material.uniforms.uX.value = Y` API works under both backends |

## The `ShaderSource` GLSL/TSL parity pattern

Every Luxar shader (the 12 production materials plus FXAA / bloom / mega-shader)
is exported as a `ShaderSource` rather than as a pair of inline strings on its
material. The registry carries both backends side-by-side:

```typescript
export const pointShaderSource: ShaderSource = {
  name: 'point',
  webgl: { vertex: POINT_VERTEX_GLSL, fragment: POINT_FRAGMENT_GLSL },
  webgpu: (uniforms) => buildPointNodeMaterial(uniforms),
};
```

The material wrapper then asks `buildMaterial` for "whichever backend the
active renderer dispatches", and the same constructor body works under both
`THREE.WebGLRenderer` and `WebGPURenderer`:

```typescript
const material = buildMaterial(
  pointShaderSource,
  { uniforms, defines, blending: THREE.AdditiveBlending, toneMapped: false },
  rendererCapabilities
);
```

Both `webgl` and `webgpu` fields are optional in the type so a future shader
can ship single-backend, but the **runtime invariant is that the active
backend's source must be present**. `buildMaterial` throws a fix-it-here
error otherwise — it does **not** silently fall through to the other backend.
Under `WebGPURenderer` specifically, a `ShaderMaterial` fallback would render
blank quads (the renderer cannot dispatch `ShaderMaterial` even when running
on its internal WebGL2 backend; see `BROWSER_SUPPORT_POLICY.md`), so the
explicit throw is load-bearing.

The parallel GLSL/TSL sources are kept in sync by the
`tsl-shader-parity.spec.ts` harness — see the user-feedback note "Keep GLSL
shaders as reference" in `CLAUDE.md` / project memory: GLSL3 sources are
never deleted, and the parity check runs on every PR.

## The marker-interface pattern

`CameraAwareMaterial` and `ColormapAwareMaterial` are not base classes —
materials don't extend a shared abstract class. They implement these as flat
interfaces, and the manager uses the `is*` type guards to discover capability
at runtime:

- **`CameraAwareMaterial`**: implemented by every visual material (Point,
  Line, GSplat) **and** their picking counterparts. `MaterialManager`
  iterates its registry and calls `updateCameraParams` on every member.
- **`ColormapAwareMaterial`**: implemented by visual materials only. Picking
  materials deliberately do **not** implement it — picking shaders don't
  sample colormaps. The two setters are the **only** public surface for
  `material-colormap-helpers.ts`; nothing outside reaches into
  `material.uniforms.uColormapTex` directly. This separation is part of the
  WebGPU prep — under `NodeMaterial` the `.uniforms` shape goes away, so
  confining writes to setters means the WebGPU port rewrites the setter
  bodies and nothing outside changes.

## Why the math helpers live here

`camera-uniforms.ts` is the smallest module here but the most consequential:
the perspective↔ortho `pointSizeFactor` / `focalLength` formulas were
previously inlined in each of the four materials that need them
(`PointMaterial`, `PointPickingMaterial`, `GSplatMaterial`,
`GSplatPickingMaterial`). When picking drifts out of sync with rendering,
the user's screen-space hit test stops matching what they see. Centralising
the formulas guarantees byte-for-byte identical math and makes them
unit-testable in one place.

Similarly, `clampGamma` exists only because the `Math.max(0.001, gamma ?? 1.0)`
clamp was repeated in six material constructors (3 geometries × {GLSL, TSL});
the GLSL `pow(color, 1.0 / gamma)` divides by zero when `gamma == 0`, and
keeping the bound in one place lets us change the clamp once if the policy
ever tightens. Its sibling `isGammaOne` is the same idea for the fast path:
when gamma is within `1e-4` of `1.0` the per-fragment `pow(color, 1/gamma)`
is a no-op (`pow(x, 1) == x`), so the threshold gates the `LUXAR_GAMMA_ONE`
GLSL define / `gammaOne` TSL config flag — and sharing the threshold keeps
that fast-path boundary byte-for-byte identical across all six files.

## Adding a new shared helper

Treat this folder as a magnet, not a dumping ground. A helper belongs here
only if:

1. It is **already duplicated** across two or more material wrappers (or its
   absence would force the duplication immediately), and
2. It is **geometry-agnostic** — anything Point-specific lives under
   `../point/`, not here.

The `uniform-helpers.ts` audit that introduced `clampGamma` explicitly
rejected a wider "uniform-spec registry" because abstract spec-driven
construction was documentation disguised as code. Keep the modules here
small and concrete.

## See Also

- `../README.md` — Materials subtree overview; sibling per-geometry material folders share this README's vocabulary
- `../../README.md` — Rendering package overview and how materials fit into the pipeline
- `../../renderer-capabilities.ts` — defines `RendererCapabilities.apiSurface` that `buildMaterial` branches on
- `../../material-manager.ts` — owns the camera-broadcast loop that drives `CameraAwareMaterial`
- `../../material-colormap-helpers.ts` — the only consumer of `ColormapAwareMaterial`
- `../../picking/` — picking materials use the same shared helpers (camera-uniforms in particular) so screen-space hit tests match rendering exactly
- `../../../tests/e2e/tsl-shader-parity.spec.ts` — the GLSL↔TSL parity harness `ShaderSource` was designed for
