# Volumetric Blending Mode — Emission–Absorption Compositing

> **Status**: Proposed — not implemented. Design settled 2026-07-19 (mode name,
> κ semantics, opacity-scales-density rule, per-splat weights spec'd-but-deferred,
> gsplats-first phasing). All file/line references verified against main
> `38c6eb19`.
> **Scope**: A 6th blending mode, `volumetric`, spanning Python (enum, validation,
> node attr, default stamping), the viewer (mode SSOT, blend state, composition,
> shaders GLSL+TSL, depth-sort gating, layers-panel UI), and — in later phases —
> the `.gsplats.zarr` format (optional per-splat absorption weights) and the
> Points/Lines geometry types.
> **Goal**: Physically grounded emission-with-occlusion rendering: each element
> adds its emitted light to the pixel AND exponentially attenuates everything
> behind it, composited back-to-front. One continuous knob (κ, `absorption`)
> spans the whole range from today's `additive` (κ = 0, exactly) to a dense
> self-occluding medium.
> **Non-goals**: WebGPU compute sorting / order-independent transparency;
> scattering, shadowing, or any multi-bounce light transport; per-splat κ in
> phase 1 (spec'd in §5.4, built in phase 2); Points/Lines in phase 1 (phases
> 3–4); skipping the depth sort when κ = 0 (mode gates sorting — predicates stay
> simple).

Related reading: `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` (the sorting
infrastructure this mode rides on), `packages/luxar-viewer/src/rendering/README.md`,
`packages/luxar-viewer/src/types/blending.ts` (mode SSOT),
`docs/specs/GSPLATS_ZARR_FORMAT.md` (format home of the deferred per-splat array).

---

## 1. Motivation and optical model

Luxar's five blending modes currently occupy the two ends of the classical
volume-rendering spectrum:

- **Pure emission** — `additive`/`luminous`: every element adds light, nothing
  occludes. Commutative, unsorted, physically the κ → 0 limit of radiative
  transfer. The right model for sparse fluorescence, but dense scenes wash out:
  a bright background shines *through* foreground structure, and depth ordering
  is unreadable.
- **Pure occlusion** — `normal`/`opaque`: front surfaces hide back ones
  (`normal` = premultiplied alpha-over with a clamped coverage alpha, sorted
  back-to-front; `opaque` = depth-tested overwrite). The right model for
  surfaces, but it discards the volumetric nature of the data — a splat is
  treated as a screen-aligned film, not a glowing medium with thickness.

The `volumetric` mode fills the middle with the standard **emission–absorption
model** of direct volume rendering (Max 1995, *Optical Models for Direct Volume
Rendering*): a medium with density ρ(x) both emits light proportionally to ρ and
absorbs the light passing through it with optical cross-section κ per unit
density. Along a view ray, the radiance reaching the camera is

    L = ∫ ε(s) · T(s) ds,        T(s) = exp(−κ ∫₀ˢ ρ(u) du)

where ε ∝ ρ is emission and T is transmittance. This is exactly the model NeRF
composites with (its per-sample weight is α = 1 − e^(−σδ)), and the model 3DGS
approximates (§2). For Luxar it is unusually apt: fitted gsplat amplitudes *are*
physical densities (fluorophore concentration), not learned opacities — so κ has
a real interpretation ("turbidity of the sample") rather than being a rendering
hack.

**What the user gets**: one slider (κ) that morphs a layer continuously from
X-ray-like additive glow (κ = 0 — bit-identical to today's `additive`, §4.3
invariant I1) through attenuated projection (small κ: front structures pop,
occluded ones dim — depth cueing for dense timelapses) to a dense
smoke/ink-like medium (large κ). Together with `max` (MIP) and `normal`/`opaque`
(surface), Luxar then covers every classical volume-rendering regime.

### 1.1 Mode taxonomy after the change

| mode | alpha source | emission | projection | commutative | sorted | depthWrite |
|---|---|---|---|---|---|---|
| `additive` | — (One/One, α ignored) | ray integral | sum | yes | no | never |
| `luminous` | — (One/One, α ignored) | ray integral | sum | yes | no | never |
| `max` | — (MaxEquation) | peak value | peak | yes | no | never |
| **`volumetric`** | **physics: 1 − e^(−τ), τ = κ·∫ρ** | **ray integral × screening** | **sum** | **no** | **yes** | **never** |
| `normal` | clamped coverage: min(intensity·opacity, 1) | peak value | peak | no | yes | gsplats never; points/lines at opacity ≥ 0.99 |
| `opaque` | — (opaque overwrite) | peak value | peak | no (depth-tested) | no (z-buffer) | always |

`volumetric` deliberately breaks the previous alignment *sum-projection ⇒
commutative ⇒ unsorted*: it is emissive for the projection taxonomy
(`usesPeakProjection(mode) === false`, unchanged —
`rendering/blending-state.ts:93`) but **ordered** for compositing. That split is
why §5.2 introduces a `needsDepthSort(mode)` predicate distinct from
`usesPeakProjection`.

---

## 2. Relation to NeRF and 3DGS (facts, to prevent drift)

All three — Luxar `normal`, 3DGS, and `volumetric` — use the **same compositing
operator**: back-to-front premultiplied "over", framebuffer state
`One / OneMinusSrcAlpha`. They differ only in where alpha comes from:

- **NeRF**: α = 1 − e^(−σδ) — the exact emission–absorption weight.
  `volumetric` is this, with the integral in closed form for Gaussians (§3).
- **3DGS**: α = o·G₂D(x) with a learned per-splat opacity `o`, clamped ≈ 0.99 —
  a heuristic that saturates by clamping instead of exponentially.
- **Luxar `normal`** (`shader-glsl.ts:436-445`): RGB carries the full unclamped
  HDR contribution, alpha carries `clamp(intensity·opacity, 0, 1)` — an
  emitter-with-occlusion model that equals 3DGS compositing bit-for-bit in the
  LDR regime (intensity·opacity ≤ 1) and diverges above it, where 3DGS clamps
  emission and occlusion together and Luxar lets emission keep going.

So `volumetric` sits at the NeRF/radiative-transfer end: alpha derived from
ray-integrated optical depth, saturating exponentially, view- and
orientation-consistent (an elongated splat seen end-on absorbs more than seen
side-on — a stored per-splat alpha cannot express that).

---

## 3. Mathematics

### 3.1 Per-splat quantities (gsplats)

For splat *i* with amplitude Aᵢ, covariance Σᵢ, and a pixel ray with unit
direction r, the fragment shader's sum-projection path **already computes the
exact line integral** of the anisotropic 3D Gaussian
(`rendering/materials/gsplat/shader-glsl.ts:229-269`):

    σ_ray = 1 / sqrt(rᵀ Σ⁻¹ r)                       (shader-glsl.ts:265)
    m(x)  = Aᵢ · G₂D(x) · σ_ray · c_T                 ("ray mass" at pixel x)

where G₂D is the shifted-truncated 2D Gaussian evaluated per fragment and c_T is
the truncated-Gaussian integral factor `√(2π)·erf(T/√2) − 2T·exp(−½T²)` ≈ 2.433
at T = 3, precomputed by `computeRayIntegralFactor`
(`rendering/materials/gsplat/math.ts:35-49`). `m(x)` is exactly what the
`additive` fragment emits today (times color and opacity). Volumetric mode
reuses it verbatim and adds:

    τ(x)      = κ_eff · m(x)                              optical depth
    T(x)      = exp(−τ)                                   transmittance
    α(x)      = 1 − exp(−τ)                               absorption alpha
    S(τ)      = (1 − exp(−τ)) / τ                         self-screening, S(0) = 1
    emission  = color · m(x) · S(τ)                       self-absorbed emission

with the effective absorption coefficient

    κ_eff = absorption (composed, §4.2) · opacity (composed) · wᵢ (per-splat weight, §5.4; 1 in phase 1)

and the fragment output

    fragColor = vec4(emission, α)        blended One / OneMinusSrcAlpha, back-to-front.

**Why opacity multiplies τ too**: `opacity` means "how much of this layer is
there" — it scales the *density*, hence emission and absorption together.
Emission also carries the plain `opacity` factor it has today (`m(x)` enters the
emission via the existing `finalColor = gammaColor · intensity · uOpacity` path,
`shader-glsl.ts:434`). If opacity scaled only emission, fading a layer to 0
would leave an invisible fog that still darkens everything behind it; with
density scaling, opacity → 0 removes both glow and occlusion, so layer fades
and LOD cross-fades stay well defined. Unlike `normal` mode there is no
depthWrite cliff at opacity ≥ 0.99 (`normalModeDepthWrite`,
`blending-state.ts:142`) — everything is smooth in both sliders.

**Why the screening factor S(τ)**: the front of a splat absorbs the emission of
its own back. For emission and absorption both proportional to density the
closed form of ∫ ε·T ds across one splat is exactly `color · m · S(τ)`. Without
it, a thick splat viewed end-on over-emits relative to the same mass split into
thin splats, and invariant I2 below fails.

### 3.2 Numerical form

S(τ) = (1 − e^(−τ))/τ is 0/0 at τ = 0. Required implementation:

- α via `-expm1(-tau)` where available (TS/TSL); in GLSL use
  `1.0 - exp(-tau)` guarded by the branch below (float32 is adequate here — the
  error of `1-exp(-τ)` at τ ≈ 1e-4 is ~1e-8, invisible at 8–10 bpc output).
- S(τ): for τ < 1e-3 use the series `S(τ) ≈ 1 − τ/2 + τ²/6` (relative error
  < 1e-10 at the cutoff); else `α/τ`. One branch, warp-coherent (τ varies
  smoothly per pixel).
- κ = 0 must short-circuit to α = 0, S = 1 exactly (invariant I1) — the series
  gives this for free.

### 3.3 Invariants (testable laws)

- **I1 — additive limit**: κ_eff = 0 ⇒ α = 0, S = 1, fragColor =
  vec4(emission, 0). Under `One/OneMinusSrcAlpha`, dst factor = 1 − 0 = 1 ⇒
  identical framebuffer arithmetic to `additive`'s `One/One`. A `volumetric`
  node at absorption 0 renders **pixel-identical** to the same node in
  `additive` mode (E2E pixel-compare test, §8).
- **I2 — split-splat multiplicativity**: one splat with ray mass m ≡ its two
  ray-wise halves (mass m/2 each) composited back-to-front. Proof sketch: back
  half contributes L_b = (c/κ)(1 − e^(−κm/2)), T_b = e^(−κm/2); front over back
  gives L_f + T_f·L_b = (c/κ)(1 − e^(−κm)) = the whole splat. Exact — only
  because absorption is exponential and emission is screened. This is what
  keeps LOD merges/splits and the additive streaming ladder visually consistent,
  and it is the core unit test (§8).
- **I3 — order independence in the limit**: as κ → 0 the compositing operator
  degenerates continuously to commutative addition; artifacts from an imperfect
  sort vanish proportionally to κ.

---

## 4. Format and Python API

### 4.1 Enum, validation, node property

- `BlendingMode` gains `VOLUMETRIC = "volumetric"`
  (`packages/luxar/src/luxar/typing_utils/enums.py:36-40`); the depth-behavior
  docstring (L20-30) gains one line (never depth-writes, requires sorting).
  `validate_blending_mode` (`validation/types.py:408-437`) derives its set from
  the enum and updates automatically.
- New `validate_absorption` modeled on `validate_opacity`
  (`validation/types.py:257-282`): float coercion, range **[0, ∞)** (no upper
  bound — κ is a physical coefficient; NaN/Inf rejected).
- New `Node.absorption` property modeled on `Node.opacity`
  (`core/node/node.py:552-574`): getter `self.attrs.get("absorption", 1.0)`,
  setter validates then `self._persist_attr("absorption", ...)`.

### 4.2 Default stamping and composition

`absorption` is **identity-valued** (multiplicative identity 1.0) — per the
blending-modes campaign doctrine it is therefore safe to default-stamp,
following the `opacity` precedent exactly:

- Stamp `absorption = 1.0` where absent in `apply_default_render_attrs`
  (`io/_compiler/node_common.py:56-63`) **and** the gsplat twin
  `apply_gsplat_group_attrs` (`io/_compiler/gsplat_assembly.py:430-438`).
  (Contrast: `blending_mode` has *no* identity value and is deliberately never
  stamped — `node_common.py:46-52`.)
- Viewer composition (§5.3): multiplicative down the chain like
  opacity/gamma/intensity — ancestors scale it, κ = 0 at any level zeroes
  absorption for the subtree.
- Default 1.0 (not 0) so that switching a layer to `volumetric` immediately
  *looks* volumetric; dragging the slider to 0 recovers the additive look.

### 4.3 Scope

κ is read **only** by the volumetric shader branch — inert in every other mode
(precedent: coverage alpha only matters in `normal`; `truncation_radius` only
for gsplats). Setting it on a group and A/B-flipping the mode between `additive`
and `volumetric` preserves the tuning.

---

## 5. Viewer design

### 5.1 Mode SSOT and blend state

- `BLENDING_MODES` tuple gains `'volumetric'`
  (`packages/luxar-viewer/src/types/blending.ts:22`). The layers-panel dropdown
  (`ui/layers/layer-controls.ts:208-213` iterates the tuple),
  `normalizeBlendingMode` (`rendering/blending-state.ts:124-133`), and TS-side
  validation all grow automatically — the campaign's SSOT paying off.
- New predicate `isVolumetricMode` beside the others (`blending-state.ts:42-76`).
- `getCompleteBlendingState` (`blending-state.ts:189-266`) gains a `volumetric`
  branch: `CustomBlending`, `AddEquation`, `blendSrc One`, `blendDst
  OneMinusSrcAlpha`, `transparent true`, `depthTest true`, `depthWrite false`
  **unconditionally** (no `normalModeDepthWrite` coupling), `shaderOutputMode:
  'premultiplied-alpha'`. This is the same framebuffer state as
  `getGSplatNormalBlendingState` (`blending-state.ts:303-315`) — share or
  generalize rather than duplicate, but note the semantic difference lives in
  the fragment shader, not the blend state.
- `usesPeakProjection` (`blending-state.ts:93-95`) **unchanged** — volumetric is
  sum-projected. The TSL rebuild boundary keyed off `usesPeakProjection`
  therefore does not fire on additive ↔ volumetric switches (correct: same
  projection graph; only the output branch and uniforms differ — see §5.5 for
  what *does* have to change on a mode switch).

### 5.2 Depth-sort gating: `needsDepthSort(mode)`

New predicate in `blending-state.ts`:

    needsDepthSort(mode) = isNormalMode(mode) || isVolumetricMode(mode)

replacing `isNormalMode` at every order-dependence gate:

- `rendering/depth-sort-coordinator.ts:242` (sort dispatch),
  `:519` (renderOrder-bias clearing), `:610-618`
  (`noteGSplatsBlendingModeSwitch` transition logic — switching *to* a sorted
  mode clears committed data + reprocesses; switching *away* invalidates
  in-flight sorts). The additive↔volumetric transition thus reuses the exact
  machinery normal↔additive already exercises.
- `rendering/depth-sort-coordinator/render-order.ts` — the cross-node
  renderOrder pass collects "normal-mode gsplat meshes" (docs L2-18, collection
  around L123/L166): volumetric meshes join the same global back-to-front
  domain.
- The commit path (`data/scene-loader/commit/commit-gsplats-geometry.ts:110-112`)
  is deliberately un-gated (identity ordering is a no-op for commutative modes;
  the sort corrects ordered modes afterwards) — no change.

Picking: phase 1 keeps additive-style brightness-as-depth picking for
volumetric (it is emissive; the `setSurfacePickDepth(isNormalMode || isOpaqueMode)`
front-most rule in `rendering/picking/picking-system.ts` stays as-is).
Front-most picking beyond a τ threshold is a possible follow-up, not phase 1.

### 5.3 Attr composition and the uniform

- `ComposableAttrs`/`EffectiveAttrs` gain `absorption`
  (`data/attrs-composer.ts:25-42`); `composeAttrs` (L54-75) multiplies it like
  opacity/gamma/intensity (identity 1.0), clamped to `max(0, ·)`. Consumers
  (`data/scene-loader/view-state/effective-attrs.ts:25-38`, the three
  `loader-factory.ts` sites at L191/258/321) thread it into material config.
- New uniform `uAbsorption` following the `uOpacity` pattern end-to-end:
  GLSL `rendering/materials/gsplat/material-glsl.ts` (uniform type L111, init
  L172, setter L282), TSL `material-tsl.ts` (L92/128/287), material-manager
  config pass-through (`rendering/material-manager.ts:238/285/337`).

### 5.4 Fragment shader (GLSL + TSL twins)

A third output branch beside `LUXAR_NORMAL_PREMULT`
(`shader-glsl.ts:436-451`), guarded by a new define `LUXAR_VOLUMETRIC`:

```glsl
#ifdef LUXAR_VOLUMETRIC
// 'volumetric' mode: emission–absorption (Max 1995). rayMass is the
// SUM-projection ray integral the additive path already computes; κ_eff
// couples the node absorption knob with opacity (density scaling).
float tau   = uAbsorption * uOpacity * rayMass;          // × wᵢ in phase 2
float alpha = 1.0 - exp(-tau);
float screen = (tau < 1e-3) ? 1.0 - 0.5*tau + tau*tau/6.0 : alpha / tau;
fragColor = vec4(finalColor * screen, alpha);
#endif
```

Key constraints:

- The branch lives on the **sum-projection** vertex path (`uProjectionMode = 0`,
  `shader-glsl.ts:222-273`) — unlike `LUXAR_NORMAL_PREMULT`, which pairs with
  peak projection. `applyBlendingMode` (`material-glsl.ts:450-510`) manages the
  define + `uProjectionMode` + blend state per mode; the volumetric case sets
  `LUXAR_VOLUMETRIC`, projection 0, and the shared One/OneMinusSrcAlpha state.
- `finalColor` already contains `gammaColor · intensity · uOpacity`
  (`shader-glsl.ts:434`) — i.e. emission's density scaling by opacity is
  inherited; only τ needs the explicit `uOpacity` factor.
- TSL twin: mirror as a runtime branch on `config.blendingMode` in
  `shader-tsl.ts` (the normal-mode branch is L587-602; blending-state selection
  L611), keeping 1:1 math with the GLSL via shared helpers in
  `rendering/materials/gsplat/math.ts` where anything is precomputed. Parity is
  enforced by `tsl-shader-parity.spec.ts` and the codegen snapshots (§8).

**Per-splat weights wᵢ (spec'd here, built in phase 2)**: optional
`.gsplats.zarr` per-splat array `absorption_weights` (float32, shape (N,),
values ≥ 0, **absent ⇒ wᵢ = 1 and no viewer buffer is allocated** — the common
fitted-microscopy case pays nothing). When present: one extra interleaved
attribute (52 → 56 B/splat) threaded like `aAmplitude`, τ gains the `wᵢ`
factor, and the classical-splat importer maps learned 3DGS opacity `o` to
`wᵢ ∝ −ln(1 − o)` (solving 1 − e^(−τ_peak) = o at the splat center) so imported
scenes get genuinely per-splat occlusion. LOD merging must aggregate wᵢ
mass-weighted (τ is linear in wᵢ·Aᵢ, so the merged weight is the
amplitude-weighted mean of the children's — preserves total optical depth to
first order). Format details land in `docs/specs/GSPLATS_ZARR_FORMAT.md` when
phase 2 is built.

### 5.5 Layers-panel UI

- κ slider cloned from the opacity-slider block
  (`ui/layers/layer-controls.ts:179-197`): `LabeledSlider`, range 0–10 with
  step 0.05 (log-feeling coverage of the useful range; the attr itself is
  unbounded), `layer.absorption` state field, `applyAbsorption` on
  `LayerApplyEngine` (`ui/layers/layer-apply.ts`, beside `applyOpacity` L208),
  `updateAbsorption` on the `LuxarMaterial` interface
  (`ui/layers/luxar-material.ts:24`).
- Visibility: the slider is shown/enabled **only when the selected layer's
  effective mode is `volumetric`** — κ is inert elsewhere and the UI should say
  so. Sync with the existing dropdown-change handler
  (`layer-controls.ts:214-224`).
- Layer init reads the composed value (`getEffectiveAttrs(...).absorption`) —
  same rule the campaign established for `blending_mode`; multiplicative attrs
  init from raw like opacity does today is also acceptable, but κ has no
  per-node slider-bounds logic, so composed is simpler and matches what
  renders.

---

## 6. Interactions with existing systems

- **LOD / streaming**: substitutive levels pin total mass per barrier group by
  default, and τ ∝ mass along the ray ⇒ absorption strength survives LOD
  switches without popping. The additive-ladder energy compensation 1/e(k)
  (PR #541) scales amplitudes — and hence τ — consistently while a ladder
  streams. Chunks arrive in energy order, not depth order: fine, the sort
  worker re-sorts on every commit (Phase-2 sorting contract), and I3 bounds the
  transient error.
- **LOD cross-fade** (`scene/lod-fade.ts`, `BLENDABLE_MODES` =
  additive/luminous today): volumetric is a *candidate* for inclusion since an
  opacity fade is ghost-free (opacity scales τ — §3.1), unlike `normal` where
  depthWrite complicates fading. Open follow-up, not phase 1.
- **Tone mapping / HDR**: pure additive accumulates without bound and can blow
  out under ACES; volumetric bounds accumulated radiance near c/κ, improving
  tone-mapped appearance on dense scenes. Emission remains unclamped HDR — a
  splat can emit more than it occludes (same deliberate asymmetry `normal`
  mode has; a property, not a bug — §9).
- **Inter-node overlap**: within a node, exact back-to-front per splat; across
  nodes, the global renderOrder pass orders whole meshes — interleaved
  splats of *different* nodes composite approximately. Same caveat `normal`
  already carries; invisible at small κ.
- **>16D / WASM**: the sort kernel and the ray-integral math are unaffected by
  dimensionality concerns (both operate on the 3 displayed dims); no new WASM
  kernel is needed — τ/α/S are per-fragment shader math.

---

## 7. Phased implementation plan

**Phase 1 — gsplats, node-level κ** (the core; independently shippable)
1. Python: enum member, `validate_absorption`, `Node.absorption`, default
   stamps (§4), CLI help text (`cli/scene_commands.py` mode list), docs mode
   lists (§8).
2. Viewer: tuple entry, `isVolumetricMode`, `needsDepthSort`, blend-state
   branch, composer + uniform plumbing, GLSL + TSL fragment branches,
   `applyBlendingMode` cases, sort-gate replacements, renderOrder inclusion,
   panel slider (§5).
3. Tests per §8.
   Exit criteria: I1 pixel-compare E2E green; I2 unit test green; codegen
   snapshot `gsplat-volumetric` committed; parity suite green; per-mode
   blend-state E2E extended to 6 modes; full unit + targeted E2E green.

**Phase 2 — per-splat `absorption_weights` + 3DGS import mapping** (§5.4):
format array + `GSPLATS_ZARR_FORMAT.md`, viewer attribute (present-only),
LOD-merge aggregation, importer mapping, `gsplat info`/`filter` awareness.
Exit criteria: imported 3DGS scene renders with per-splat occlusion; absent-array
path allocates nothing (assert in a unit test).

**Phase 3 — points**: extend the depth-sort infrastructure to point nodes
(centers sort directly — the same kernel input shape as splat centers); chord
integral through the existing super-Gaussian radial profile
(`rendering/materials/point/shader-glsl.ts:189-198`; for a Gaussian-profile
point the math is the isotropic special case of §3.1). Per the three-geometry
symmetry rule: same mode name, same attr, shared `needsDepthSort`.
Showcase + exit criterion: switch the **mandelbulb demo**
(`packages/luxar/src/luxar/demos/demo_mandelbulb.py:252-260`) to
`blending_mode="volumetric"` — today it must dim its colors ×0.1 (L221) to keep
the dense fractal surface from blowing out under additive; volumetric's bounded
accumulation removes that workaround and adds real depth cueing to the surface.

**Phase 4 — lines**: segment-midpoint depth sort (standard approximation;
artifacts only when long segments interleave — subdivision if ever needed);
cross-section chord through the transverse super-Gaussian profile
(`rendering/materials/line/shader-glsl.ts:344-351`).

---

## 8. Test plan

Every SSOT list that must grow for a 6th mode (inventory from the 2026-07
blending campaign):

- `src/tests/e2e/blending-expected-state.ts` — `EXPECTED_BLEND_STATE` gains
  `volumetric` (CustomBlending 5, AddEquation 100, One 201,
  OneMinusSrcAlpha 205, depthWrite false).
- Codegen SHADERS lists: `tsl-codegen-snapshot.spec.ts:118` + harness
  registries (`tests/e2e/harnesses/tsl-harness/gsplats.ts:191`; points/lines in
  phases 3–4) — new `gsplat-volumetric` variant, snapshot committed.
- `tests/unit/rendering/materials/gsplat/blending-mode.test.ts` — state +
  define + projection-mode asserts for the new branch, GLSL and TSL.
- Python `typing_utils/tests/test_enums.py:22-30` — hard-coded member count
  5 → 6; validation error-message test (lists all 6).
- Docs mode lists: `docs/guides/user/LUXAR_ZARR_FORMAT.md:237,489`,
  `docs/specs/GSPLATS_ZARR_FORMAT.md:367`, CLI help.

Invariant and behavior tests:

- **I1 E2E pixel-compare**: same deterministic fixture rendered once
  `additive`, once `volumetric` with absorption 0 → identical pixels
  (tolerance 0; same blend arithmetic). Then absorption > 0 → measured
  darkening behind a front splat (discriminator sampling per the
  blending-modes E2E pattern, `?dpr=1`).
- **I2 unit test**: closed-form check — composite two half-mass fragments
  back-to-front in TS using the shader formulas, compare to the single-splat
  formula to 1e-6.
- **Ordering E2E**: reuse the front-to-back reversed-overlap fixture pattern
  (`test_gsplats_normal_overlap_reversed`) with volumetric mode — asserts the
  sort actually engages for the new mode (fail-first against a build without
  the `needsDepthSort` change).
- New deterministic fixture(s) in `tests/fixtures/generate_test_data.py`
  (auto-generated by the vitest globalSetup manifest).
- Numerical: S(τ) series/branch cross-check at τ ∈ {0, 1e-6, 1e-3, 1, 10}.

---

## 9. Risk register

1. **Sorting cost now applies to more scenes.** Any volumetric node pays the
   SortWorker path (dispatch hysteresis, per-commit re-sorts). Mitigated by the
   Phase-2/3 sorting contract already meeting 10M+ splats; explicitly NOT
   mitigated by skipping the sort at κ = 0 (mode gates sorting; predicates stay
   simple — a user wanting free additive uses `additive`).
2. **Numerics at τ → 0.** S(τ) is 0/0; the series branch (§3.2) is mandatory,
   and I1's exactness depends on it. Covered by unit tests.
3. **HDR asymmetry.** Emission is unclamped while absorption saturates at 1 —
   a bright splat brightens more than it occludes. Inherent to HDR scientific
   rendering (shared with `normal`); documented, not "fixed".
4. **Taxonomy drift.** `volumetric` is the first sum-projected *sorted* mode;
   any future code that infers "sorted ⇒ peak" or "sum ⇒ commutative" from the
   old alignment is wrong. `needsDepthSort` is the only sanctioned
   order-dependence predicate; `usesPeakProjection` the only projection one.
5. **Per-splat weights format churn** (phase 2): +4 B/splat when present,
   importer/LOD/merge surface. Deferred by design; the absent-array fast path
   keeps phase 1 format-neutral.
6. **Mode-switch state machine.** `applyBlendingMode` now manages three
   define/projection/blend combinations (plain, NORMAL_PREMULT, VOLUMETRIC).
   The campaign's mutation-tested branch tests must grow with it; the TSL
   rebuild boundary (keyed on `usesPeakProjection`) is already correct for
   additive↔volumetric (no rebuild needed) and normal↔volumetric (rebuild —
   projection changes).

---

## 10. Changelog

- **2026-07-19** — Initial spec. Design decisions settled with the user:
  mode name `volumetric`; κ as node-level composable `absorption` attr
  (multiplicative, identity/default 1.0, default-stamped like opacity);
  opacity scales density (emission AND τ); per-splat `absorption_weights`
  spec'd but deferred to phase 2 with the 3DGS opacity mapping; phasing
  gsplats → per-splat → points → lines.
