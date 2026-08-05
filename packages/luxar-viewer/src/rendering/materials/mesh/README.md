# Mesh Material — Shaded Triangle Surfaces

> The fourth per-geometry material stack, and the first one that **shades**. A
> light-free view-anchored headlight over an indexed `THREE.BufferGeometry`, with
> the stored-normal / derivative-normal choice made as a compile-time shader
> variant and per-vertex alpha as the sole coverage term.

## Overview

The other three geometry types are purely **emissive** instanced quads: their
fragment stage computes a soft falloff and emits colour, with no notion of a
surface orientation. A triangle has one, and a mesh drawn flat is an unreadable
silhouette — so this is the stack where lighting enters Luxar, and with it the
two hazards that come from reading a normal at all: degenerate or corrupt stored
normals, and back faces.

It is also the first stack whose per-vertex data arrives in **vertex
attributes** rather than an RGBA32F element texture. There is no `texelFetch`
prologue, no `aSortedIndex` indirection and no quad expansion: `position` and
`normal` are three's own auto-declared attributes and the draw is an ordinary
indexed `drawElements`.

Full design rationale: `docs/specs/MESH_NODE_SPEC.md` §6.

## Files

| File               | Role                                                                                                        |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| `shader-glsl.ts`   | Hand-written GLSL3 vertex + fragment pair, and the readable reference for the shading math                  |
| `shader-tsl.ts`    | `meshWebGPUFactory` — the `NodeMaterial` twin, plus `buildMeshTSLNodesFromUniforms` for the harness         |
| `material-glsl.ts` | `MeshMaterial` (`THREE.ShaderMaterial`) — uniforms, defines, `applyBlendingMode`, `clone`                   |
| `material-tsl.ts`  | `MeshTSLMaterial` (`NodeMaterial`) — same surface, with graph rebuilds where the GLSL twin toggles a define |
| `appearance.ts`    | `MESH_DEFAULTS`, the supported-mode list, and the mode → emission-shape map both backends read              |

## The shading model (§6.2)

```
shade = mix(uAmbient, 1.0, pow(saturate(dot(N, V) * 0.5 + 0.5), uShadeExponent))
```

`V` is the fixed view-space axis `(0, 0, 1)` — a camera headlight — so
`dot(N, V)` reduces to the view-space normal's z. Nothing is added to the scene
graph and no light node exists; `uAmbient = 1.0` collapses the term entirely and
reproduces the emissive look of the other three types.

The shade factor multiplies **RGB only**. It must never enter the coverage, or a
silhouette fragment would also turn transparent — and, under the `opaque`
cutout, dissolve.

### Three rules about the normal, in this order

1. **The derivative normal is computed unconditionally**, before any
   guard-dependent branch. The epsilon guard reads an interpolated varying, so
   branching on it is non-uniform control flow, where GLSL leaves `dFdx`/`dFdy`
   _undefined_. Only the cheap `gl_FrontFacing` flip may sit behind the guard.
2. **The epsilon guard is affirmative**: the stored normal is used only when
   `dot(N, N) >= eps` is positively true. `NaN` fails every comparison, so a
   corrupt store lands in the fallback rather than normalizing into NaN shading.
3. **The two-sided flip applies to the stored normal only.** Without it a
   back-facing fragment has `dot(N, V) < 0`, the wrap term lands in `[0, 0.5)`,
   and the back side shades with a dimmed inverted gradient collapsing toward
   `uAmbient` — visible immediately, because `double_sided` defaults true and the
   whole-triangle cull exposes a sliced isosurface's interior faces. The
   derivative normal needs no flip, and flipping it would _reintroduce_ that
   inverted shade exactly at the degenerate vertices the guard exists to rescue.
   So the exemption is per **fragment**, not per variant.

### Why the derivative normal is forced viewer-facing

`cross(dFdx(P), dFdy(P))` carries the sign of the fragment-space y axis, and the
two backends disagree about it: GLSL's `dFdy` is with respect to a **bottom-up**
window coordinate, WGSL's `dpdy` a **top-down** one. Left alone, the flat variant
shades correctly on WebGL and collapses to `uAmbient` everywhere on WebGPU — and
the parity harness (which compiles TSL _to GLSL_) could never see it. Since `V`
is `(0, 0, 1)`, "faces the viewer" is exactly `z >= 0`, so one sign flip makes
the fallback convention-independent.

## Variants

| Define / config flag         | Effect                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `USE_COLORMAP`               | read `aScalar` + LUT instead of the `color` attribute                         |
| `LUXAR_GAMMA_ONE`            | skip the gamma `pow()`                                                        |
| `LUXAR_NO_GOG`               | skip the intensity/offset mul-add-clamp chain                                 |
| `LUXAR_MESH_FLAT_NORMAL`     | derivative-only normal; the `normal` attribute and varying disappear entirely |
| `LUXAR_MESH_ALPHA_CUTOUT`    | `opaque`: hard, order-independent alpha cutout                                |
| `LUXAR_MAX_RGB_CONTRIBUTION` | `max`: premultiply RGB by coverage                                            |

The stored-normal-vs-flat choice is a **compile-time variant**, not a runtime
branch, because a declared-but-unbound `normal` attribute reads `(0, 0, 0, 1)`
rather than "absent" — there is no runtime value meaning "no normals". It is
decided once per node in `create-mesh-node.ts` and handed to both backends, and
re-applied per epoch by `applyMeshShading` because its
`normal_dims == displayDims` half is view-dependent.

## Where mesh diverges from its three siblings — by name

| Divergence                                                  | Why                                                                                                                                                                                                                                        |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Default blending mode is `opaque`, not `additive`           | The only mode unconditionally correct without per-triangle depth sorting (§6.3), and what a surface should look like. Applied **viewer-side** in `create-mesh-node.ts`; stamping it in the writer would override an ancestor group's mode. |
| `volumetric` degrades to `opaque` with a warning            | Emission–absorption integrates over a path length through a medium, and a triangle is zero-thickness. A warning rather than a failure, because the mode can be **inherited** from an ancestor the mesh knows nothing about.                |
| No `CameraAwareMaterial`                                    | A mesh has no screen-space size to recompute. An empty `updateCameraParams` would be a lie that also costs a per-frame call per node.                                                                                                      |
| No `uAbsorption` / `uHasElementAlpha`                       | Both exist solely to serve the volumetric mode.                                                                                                                                                                                            |
| No `radiusScale` / `truncationRadius`                       | Both normalize a per-element extent; a triangle's extent is its own vertices.                                                                                                                                                              |
| Coverage is `vAlpha * uOpacity`, not `intensity * uOpacity` | Mesh has no per-element intensity/amplitude/falloff scalar (§2.2).                                                                                                                                                                         |
| `vAlpha` interpolates (no `flat` qualifier)                 | A splat's alpha is a per-**instance** constant, so `flat` is free there. A mesh vertex is not, so its opacity must vary across the face — like the line shader's `vAlpha` along a segment.                                                 |
| `updateFlatNormal`                                          | Nothing else shades, so nothing else has a normal-source variant.                                                                                                                                                                          |

## What `opaque` does to `opacity`

Node opacity folds into the coverage that is compared against `uAlphaCutoff`, so
under the default mode `opacity` does **not** dim a mesh — it sweeps the cutout
threshold. On an RGB mesh (`vAlpha ≡ 1`) that is a hard step at the cutoff; with
authored per-vertex alpha the surface **erodes** as more vertices fall below it.
A smooth opacity fade means selecting `normal` and accepting its unsorted-
translucency caveat (§6.3).

## Testing

- `tests/unit/rendering/materials/mesh/` — the define/state lifecycle on **both**
  backends from one table, and the appearance/mode map.
- `tests/unit/rendering/node-factory/create-mesh-node.test.ts` — the variant
  decision, the placeholder attribute set, the viewer-side mode defaults.
- `tests/e2e/tsl-shader-parity.spec.ts` — pixel parity per variant, plus the two
  anti-vacuity assertions: the flat variant is **lit** (not collapsed to
  ambient), and flat and smooth **differ** on a fixture whose stored normals fan
  away from its geometric normal.
- `tests/__codegen__/mesh*.glsl.txt` — the generated-shader snapshots (five
  variants, ten files).

## See Also

- `../README.md` — the four-geometry material subtree and its symmetry rules
- `../../mesh-geometry.ts` — the attribute layout this shader reads, including
  the §6.1.1 WebGPU dtype rules
- `../../node-factory/create-mesh-node.ts` — the caller, and the owner of the
  shading-variant decision
- `docs/specs/MESH_NODE_SPEC.md` §6 — the full design rationale
