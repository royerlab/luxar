# Mesh Material — Shaded Triangle Surfaces

> The fourth per-geometry material stack, and the first one that **shades**. A
> light-free view-anchored offset key light over an indexed `THREE.BufferGeometry`, with
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

| File               | Role                                                                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shader-glsl.ts`   | Hand-written GLSL3 vertex + fragment pair, and the readable reference for the shading math                                                         |
| `shader-tsl.ts`    | `meshWebGPUFactory` — the `NodeMaterial` twin, plus `buildMeshTSLNodesFromUniforms` for the harness                                                |
| `material-glsl.ts` | `MeshMaterial` (`THREE.ShaderMaterial`) — uniforms, defines, `applyBlendingMode`, `clone`                                                          |
| `material-tsl.ts`  | `MeshTSLMaterial` (`NodeMaterial`) — same surface, with graph rebuilds where the GLSL twin toggles a define                                        |
| `appearance.ts`    | `MESH_DEFAULTS`, the normal-validity epsilon, the supported-mode list, and the mode → emission-shape map — every value both backends must agree on |

## The shading model (§6.2)

```
L = normalize(vec3(-0.35, 0.55, 0.75))
H = normalize(L + vec3(0.0, 0.0, 1.0))
shade = mix(uAmbient, 1.0, pow(saturate(dot(N, L) * 0.5 + 0.5), uShadeExponent))
spec = uSpecular * pow(max(dot(N, H), 0.0), uShininess)
rgb = rgb * shade + vec3(spec)
```

`L` is a fixed above-left view-space key and `V` remains the fixed view axis `(0, 0, 1)`, making `H` constant too. Nothing is added to the scene graph and no light node exists. `uAmbient = 1.0` removes the diffuse gradient, while `uSpecular = 0.0` removes the highlight.

The shade factor multiplies **RGB only**. It must never enter the coverage, or a
silhouette fragment would also turn transparent — and, under the `opaque`
cutout, dissolve.

### Three rules about the normal, in this order

1. **The derivative normal is computed unconditionally**, before any
   guard-dependent branch. The epsilon guard reads an interpolated varying, so
   branching on it is non-uniform control flow, where GLSL leaves `dFdx`/`dFdy`
   _undefined_. Only the cheap `gl_FrontFacing` flip may sit behind the guard.
2. **The epsilon guard is affirmative and two-sided**: the stored normal is used
   only when `eps <= dot(N, N) < 1e30` is positively true. `NaN` fails every
   comparison, so a corrupt store lands in the fallback rather than normalizing into
   NaN shading — and the upper bound matters for the same reason: an INFINITE normal
   component satisfies `>= eps`, and `inf * inversesqrt(inf)` is `inf * 0` = NaN,
   i.e. the guard's own failure mode arriving from the other end.
3. **The two-sided flip applies to the stored normal only.** Without it the stored
   back normal shades on the wrong side of its gradient, producing an inverted
   result that can collapse toward `uAmbient`. This is visible immediately because
   `double_sided` defaults true and the whole-triangle cull exposes a sliced
   isosurface's interior faces. The
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

## The near fade, and what it does in `opaque` (#1431)

Mesh evaluates the shared `perspectiveNearFade` **per fragment** — a triangle
spans depth, so a per-vertex value would interpolate the ramp across the face —
and rejects below 0.01 in every blending mode. The stage table for all four
geometry types is in `../_shared/README.md`.

**In the default `opaque` mode the fade darkens rather than dissolves, and that
is accepted rather than overlooked.** `opaque` emits `vec4(rgb, 1.0)`: there is
no alpha left to fade, so the fade ramps the shaded RGB instead. Over a black
background that reads as a dissolve; over a lit one the near shell goes visibly
**black** for the width of the band before the 0.01 reject removes it. Every
other mode folds the fade into coverage and dissolves properly.

Two things keep it a non-issue in practice. The band is
`[nearCull, 2·nearCull]` with `nearCull = 1e-3 · scene diagonal`, so the
darkening sits 0.1–0.2% of the scene diagonal in front of the eye and is about
0.1% thick — a distance the camera crosses in a frame or two of any real
approach. And the alternatives are each worse in their own way. The only way to
dissolve here that keeps the fragment DETERMINISTIC is to let the fade move the
cutout comparison (`a * nearFade < uAlphaCutoff`), which would dissolve the
surface's authored holes OPEN as the camera closed in — a distance effect
rewriting an authored mask, and non-monotone besides. A stochastic reject
(`discard` when `nearFade < hash(gl_FragCoord.xy)`) would dissolve properly
without touching the mask, which is the standard trick for exactly this
situation, and it is declined rather than overlooked: it costs a hash plus a
codegen variant, and a non-deterministic fragment would turn the parity
harness's exact-factor lock (`mesh-near-fade` is asserted at precisely
`0.15625 ×` its un-faded reference, per pixel) into a much weaker
coverage-fraction test. Dropping the RGB ramp and keeping only the `< 0.01`
reject is the fourth option, and is simply the hard clip this change set out to
remove. Partial transparency is self-contradictory in a depth-writing mode drawn
without per-triangle sorting (§6.3), so `opaque` has no honest coverage to fade;
the fix, if this ever needs one, is to select `normal`.

This is the same structural question `scene/lod-fade.ts` answers, and it answers
it the same way: its `BLENDABLE_MODES` is `additive`/`luminous`/`volumetric`
only, so an `opaque`/`normal`/`max` layer keeps a HARD LOD swap rather than a
cross-fade — a mode with no linear opacity knob does not get a fake one. The
near fade differs only in that it must still do _something_ at the near plane,
so it ramps the one channel it honestly can.

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

| Divergence                                                  | Why                                                                                                                                                                                                                                                                                |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Default blending mode is `opaque`, not `additive`           | The only mode unconditionally correct without per-triangle depth sorting (§6.3), and what a surface should look like. Applied **viewer-side** in `create-mesh-node.ts`; stamping it in the writer would override an ancestor group's mode.                                         |
| `volumetric` degrades to `opaque` with a warning            | Emission–absorption integrates over a path length through a medium, and a triangle is zero-thickness. A warning rather than a failure, because the mode can be **inherited** from an ancestor the mesh knows nothing about.                                                        |
| `CameraAwareMaterial`, half-consumed                        | `resolution` / `isOrtho` are ignored — a mesh has no screen-space size to recompute, and the near fade's ortho test reads three's `isOrthographic` — but `nearCull` is read, because the shared near fade applies to a surface too. See the stage table in `../_shared/README.md`. |
| No `uAbsorption` / `uHasElementAlpha`                       | Both exist solely to serve the volumetric mode.                                                                                                                                                                                                                                    |
| No `radiusScale` / `truncationRadius`                       | Both normalize a per-element extent; a triangle's extent is its own vertices.                                                                                                                                                                                                      |
| Coverage is `vAlpha * uOpacity`, not `intensity * uOpacity` | Mesh has no per-element intensity/amplitude/falloff scalar (§2.2).                                                                                                                                                                                                                 |
| `vAlpha` interpolates (no `flat` qualifier)                 | A splat's alpha is a per-**instance** constant, so `flat` is free there. A mesh vertex is not, so its opacity must vary across the face — like the line shader's `vAlpha` along a segment.                                                                                         |
| `updateFlatNormal`                                          | Nothing else shades, so nothing else has a normal-source variant.                                                                                                                                                                                                                  |

## The appearance knobs are author-reachable, and clamped

`ambient`, `shade_exponent`, `specular`, `shininess`, and `alpha_cutoff` are read from the mesh leaf attrs. The writer never stamps them — they arrive only when passed through
`add_mesh(**attrs)`. This authoring path and the material reads landed together, so
an accepted value always affects the rendered mesh rather than becoming dead metadata.

All five are clamped at the material boundary because they are fractions or exponents, not gains:

| Knob             | Clamp      | Why it is not merely tidiness                                                                                                                                |
| ---------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ambient`        | `[0, 1]`   | It is the shade floor. `1e9` multiplies the surface to white.                                                                                                |
| `specular`       | `[0, 1]`   | It is an additive white term. `1e9` overwhelms the surface colour and clips the result.                                                                      |
| `alpha_cutoff`   | `[0, 1]`   | Compared against a coverage already in `[0, 1]`. `1e9` discards every fragment — the mesh vanishes with no diagnostic.                                       |
| `shade_exponent` | `>= 0.001` | `pow(wrap, 0)` where `wrap` is exactly 0 (any face-away fragment) is **undefined** GLSL — driver-dependent 1, 0 or NaN. Same hazard `clampGamma` exists for. |
| `shininess`      | `>= 0.001` | `pow(max(dot(N, H), 0), 0)` is likewise undefined where the half-vector term is 0; negative values can diverge there.                                        |

NaN/Inf route to the documented default rather than to a range boundary, matching the
sibling shaders' sanitizer policy: corruption resolves loudly, not to a value that
looks deliberate.

## The other family: `material="physical"`

This shader is the mesh's DEFAULT material, not its only one. `material="physical"` on
`add_mesh` hands a mesh to three's own physically based material instead, lit by a lazily
built scene environment — metals, lacquer, a pearlescent shell, a true view-relative
Fresnel rim, glass that refracts the background and other meshes, none of which the
fixed key above can express. It lives in
[`../mesh-physical/`](../mesh-physical/README.md) as its own `VISUAL_FACTORIES.meshPhysical`
entry; nothing in this directory branches on it. The five knobs above, `blending_mode`,
colormaps and textures are refused at authoring under that material, because none of
them means anything there. Design: `docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md`.

## What `opaque` does to `opacity`

Node opacity folds into the coverage that is compared against `uAlphaCutoff`, so
under the default mode `opacity` does **not** dim a mesh — it sweeps the cutout
threshold. On an RGB mesh (`vAlpha ≡ 1`) that is a hard step at the cutoff; with
authored per-vertex alpha the surface **erodes** as more vertices fall below it.
A smooth opacity fade means selecting `normal` and accepting its unsorted-
translucency caveat (§6.3) — which the commit path now warns about once per node
(`data/scene-loader/commit/commit-mesh-geometry.ts`), so the tradeoff is stated at
the console rather than only here.

Note the warning's two arms are not the same condition. The opacity arm keys on
`normalModeDepthWrite` (`>= 0.99`), the threshold where `normal` stops depth-writing.
Depth-writing does not make the compositing exact above it — a translucent fragment
that writes depth still drops whatever is behind it — but that arm only fires with no
per-vertex alpha, so every fragment is at least `opacity` opaque and the dropped term
is bounded by `1 − opacity`. Below the threshold nothing bounds it: unsorted
alpha-over swaps almost the whole contribution of two overlapping faces.

The per-vertex-alpha arm fires at **any** opacity, because at `opacity = 1` a
translucent fragment still writes depth and rejects whatever is behind it — dropout
rather than mis-ordering, and unbounded, since one vertex's alpha says nothing about
the rest. Both arms key on what is observable rather than on what was authored: an
RGBA colour array whose alpha is uniformly opaque composites like an RGB one, so it
stays silent.

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
