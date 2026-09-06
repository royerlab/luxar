# Shared Material Infrastructure

> Cross-cutting helpers used by the Point / Line / GSplat / Mesh material stacks (both GLSL `ShaderMaterial` and TSL `NodeMaterial` variants) and their picking counterparts.

This folder holds the small, geometry-agnostic pieces that the per-geometry
material wrappers in `../point/`, `../line/`, `../gsplat/` and `../mesh/`
compose. Nothing
here renders a pixel on its own — each module either declares a contract the
materials implement (`CameraAwareMaterial`, `ColormapAwareMaterial`, `ShaderSource`),
encapsulates math/string content that would otherwise be copy-pasted across eight
material wrappers, or branches the dual-stack WebGL2 / WebGPU dispatch in one
place (`buildMaterial`). Mesh takes only a subset — it consumes just half of the
`CameraAwareMaterial` contract (the near-fade inputs; there is no screen-space size
to recompute) and none of the per-element falloff machinery — so each module below
records which types actually reach it.

## Module map

| File                         | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shader-source.ts`           | `ShaderSource` registry type: `{ name, webgl?: { vertex, fragment }, webgpu?: factory }` plus `requireWebGLSources()` narrowing helper                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `material-builder.ts`        | `buildMaterial(source, config, caps)` — branches on `caps.apiSurface` to return either a `THREE.ShaderMaterial` (GLSL3) or a TSL `NodeMaterial`                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `camera-aware-material.ts`   | `CameraAwareMaterial` interface + `isCameraAwareMaterial` guard. The contract `updateCameraParams(fov, resolution, isOrtho?, nearCull?, pixelRatio?)` that `MaterialManager` broadcasts to every registered visual + picking material; `pixelRatio` is the current render target's physical pixels per CSS pixel, including supersampling                                                                                                                                                                                                                                                                 |
| `colormap-aware-material.ts` | `ColormapAwareMaterial` interface + guard. Two setters (`setColormapTexture`, `setScalarRange`) so the colormap helpers never reach into `material.uniforms` directly                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `camera-uniforms.ts`         | Pure math shared by visual + picking materials: `computePointSizeFactor`, `computeMaxPointSize`, `computeFocalLength`. Branches on `isOrtho` so callers don't special-case projection                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `uniform-helpers.ts`         | `clampGamma(g)` — single source of truth for the `Math.max(0.001, g ?? 1.0)` clamp used in every material constructor; `isGammaOne(g)` — `abs(g - 1) < 1e-4` fast-path test that gates the `LUXAR_GAMMA_ONE` define (GLSL) / `gammaOne` flag (TSL) so the shader skips `pow(color, 1/gamma)` when gamma is unity                                                                                                                                                                                                                                                                                          |
| `glsl-lib.ts`                | `GLSL_SANITIZE_FUNCTIONS` GLSL3 snippet (`isInvalidFloat`, `sanitizePositive`, `sanitizeNonNegative`) prepended to every visual _and_ picking GLSL shader of all four geometry types, plus `GLSL_NEAR_FADE_FUNCTIONS` (`perspectiveNearFade` — the unified near handling **all four** types share, visual and pick; see the stage table below for where each one evaluates it)                                                                                                                                                                                                                            |
| `falloff.ts`                 | Shared constants for the Points/Lines shifted super-Gaussian sprite falloff (`FALLOFF_FLOOR`, `FALLOFF_K`, `INV_ONE_MINUS_FALLOFF_FLOOR`, `GAUSSIAN_EQUIVALENT_TRUNCATION`) — one `toFixed`-stable literal source for the GLSL strings, the TSL graphs, and the codegen snapshots                                                                                                                                                                                                                                                                                                                         |
| `line-capsule.ts`            | Capsule line primitive (#1352): the calibration constants (`CAPSULE_SUPPORT_SIGMA`, `CAPSULE_RADIUS_PER_QUAD_HALFWIDTH`, `CAPSULE_JOINT_PACKET_MIN_RADIUS_PX`, …), the sharpness map `capsuleProfileExponent`, the joint-end stencil reach rule (`capsuleJointStencilReach` — full disc once a deficit packet exists, #1488), and the CPU reference profile + joint-composition model (`capsuleProfile`, `capsuleJointCompositionError`) the unit tests pin, which mirrors the fragment math AND the vertex stencil so a short reach chops it — single literal source for the GLSL strings and TSL graphs |
| `volumetric.ts`              | Shared constants for the emission–absorption (`volumetric`) output branch (`ALPHA_CLAMP`, the `S(τ)` series coefficients and threshold) used identically by the point/line/gsplat GLSL and TSL twins                                                                                                                                                                                                                                                                                                                                                                                                      |
| `erf.ts`                     | The viewer's single erf source: `erfRef` (A&S 7.1.26, CPU reference — consumed by `gsplat/math.ts`) plus the shader polynomial (`erfPoly` TS mirror, `GLSL_ERF_FUNCTIONS`) — degree-13, no `exp`/division, generated from one coefficient array; backend agreement is pinned by the `erf` parity fixture and codegen snapshot. Free of `three/tsl` on purpose: `gsplat/math.ts` reaches `erfRef` on the GLSL path, and one TSL import here pulled the whole WebGPU cone into the eager bundle (#1679)                                                                                                     |
| `erf-tsl.ts`                 | The TSL half of the same polynomial (`erfPolyTSL`), split out of `erf.ts` so that file stays `three/tsl`-free; same coefficients, so the parity fixture still pins both against `erfRef`                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `scalar-range.ts`            | Single source for the colormap scalar-range → uniform mapping shared by all six materials (`DEGENERATE_SCALAR_RANGE_EPS`, `computeScalarRangeUniforms`): the shaders' `t = clamp((scalar − uScalarMin) · uScalarScale, 0, 1)` was inlined 12 times, and a DEGENERATE range (min === max) sent every element to `t = 0`; the shared form maps a constant scalar to the LUT midpoint                                                                                                                                                                                                                        |
| `tsl-helpers.ts`             | TSL counterparts to the GLSL sanitisers (`sanitizeNonNegative`, `sanitizeAlpha`, `invalidFloatTSL` — `sanitizePositive` has no TSL twin, nothing calls it there), the near-fade twins (`perspectiveNearFadeTSL` runtime-uniform / `perspectiveNearFadeStaticTSL` compile-time-ortho), plus `proxyIUniform(node)` — wraps a TSL `UniformNode` in an `IUniform`-shaped getter/setter so the `material.uniforms.uX.value = Y` API works under both backends                                                                                                                                                  |

## Where each geometry type evaluates `perspectiveNearFade`

All four types share the one helper (`GLSL_NEAR_FADE_FUNCTIONS` /
`perspectiveNearFadeTSL`), in both the visual and the picking shader, so nothing
pops or clips hard against the near plane. **Which stage** evaluates it is not a
style choice — it follows from what the primitive is:

| Type        | Stage            | Reject                                                      | Why that stage                                                                                                                                                                                                                                                                                                                                |
| ----------- | ---------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Points**  | **Per vertex**   | Hard, `< 0.01` — collapses the quad off-screen              | The whole instanced quad shares one center depth, so a per-vertex value is exact for the sprite. Rejecting in the vertex stage costs the fragments nothing.                                                                                                                                                                                   |
| **GSplats** | **Per vertex**   | Hard, `< 0.01` — collapses the quad off-screen              | Same: one center depth per instance. Also guards the `1/z` Jacobian the projection would otherwise evaluate at a near-singular depth.                                                                                                                                                                                                         |
| **Lines**   | **Per fragment** | None at 0.01 — the fade multiplies into the intensity chain | A segment SPANS depth, so either endpoint's value would be wrong along it. The vertex stage instead culls a both-endpoints-near segment and clips a half-near one onto the plane.                                                                                                                                                             |
| **Mesh**    | **Per fragment** | Hard, `< 0.01` — `discard`                                  | A triangle spans depth too, and per-vertex would smear the ramp across a large one. The reject must be per fragment because the mode may WRITE depth — `opaque` always, `normal` at opacity ≥ 0.99 (`normalModeDepthWrite`) — and it is unconditional rather than gated on that, since gating would cost a uniform to save a discard (#1431). |

The reject threshold and the band are one shared contract, not four: the near-plane
floor in `scene/scene-manager/clipping/bounds-math.ts` derives `MAX_NEAR_FAR_RATIO`
from `1.0589 · nearCull` (the depth at which `smoothstep` reaches 0.01), and a
property test pins it against these sources.

## The `ShaderSource` GLSL/TSL parity pattern

Every Luxar shader (the 10 production materials plus FXAA / bloom / mega-shader)
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

That call shape is illustrative, not a census: the only `buildMaterial` callers
today are the post-processing passes (`post-processing/bloom/chain.ts` ×3 and
`post-processing/fxaa/pass.ts`). The geometry materials build their
`ShaderMaterial` directly or extend the `*TSLMaterial` classes from
`rendering/tsl/registry.ts`, and `POINT_SOURCE` & friends are consumed by the
E2E parity harness.

For the passes that do call it, the render state in the config is
**authoritative on both branches**: `blending`, `depthTest`, `depthWrite`,
`transparent`, `toneMapped` and `side` are resolved once (same defaults either
way — note `toneMapped` defaults to `false`, the opposite of Three's own) and
then applied to the `ShaderMaterial` _or_ the `NodeMaterial`, overriding whatever
the TSL factory set on itself. Only `defines` is WebGL-only, because
`NodeMaterial` has no such field — a TSL factory takes its compile-time flags
through its own arguments. The WebGPU branch used to forward `uniforms` alone and
drop the rest, which silently disabled the bloom upsample pass's
`AdditiveBlending` and turned WebGPU bloom into a flat dim wash (#2563).

Two boundaries follow, and the example above sits on top of both. `toneMapped` is
threaded for symmetry but is only **observed** under `THREE.WebGLRenderer` —
three's node/WebGPU renderers never read `material.toneMapped`; tone mapping
there is an output pass driven by `renderer.toneMapping`, which
`post-processing-manager.ts` pins to `NoToneMapping`. And the config cannot
express `CustomBlending`'s factors (there is no `blendEquation` / `blendSrc` /
`blendDst`), which `getCompleteBlendingState` in `../../blending-state.ts`
returns for the `max` and `opaque` modes. So a factory that derives its own
**complete** blending state — `pointWebGPUFactory` ends by calling
`applyBlendingStateToMaterial`, and re-does it on every `rebuildGraph` — must not
be routed through `buildMaterial` with a partial config: the resolved defaults
would clobber the `depthTest` / `depthWrite` / `transparent` the factory chose
(`additive` wants `false / false / true`), and the next graph rebuild would put
them back, so the state would appear and disappear rather than fail outright.

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
