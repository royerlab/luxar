# Volumetric Blending Mode — Emission–Absorption Compositing

> **Status**: **Phase 3 IMPLEMENTED** (2026-07-24): points render the real
> emission–absorption math. The point fragment computes the ISOTROPIC
> special case of the §3.1 ray integral — rayMass = falloff · R·√(π/K)
> (the line integral of the Gaussian-profile ball; K = ln 100 truncates at
> T = √(2K) ≈ 3.03σ, nearly the gsplat T = 3, so point and gsplat κ scales
> agree; `POINT_CHORD_SCALE` in `rendering/materials/point/math.ts`) —
> with τ = κ·density·rayMass where density = falloff·opacity·sizeScale²·
> nearFade (every "how much of this point is there" factor scales τ, so
> fades leave no ghost fog). Points also gained RGBA colors: the alpha
> column rides texel2.y, active in every mode, mapped through w(a) under
> volumetric exactly like gsplats (§5.4.1), gated by `uHasElementAlpha`.
> `effectiveGeometryMode` now falls back to additive for LINES only
> (phase 4); the depth sort engages for points volumetric through the
> existing `needsDepthSort(effectiveGeometryMode(...))` gates with zero
> coordinator change (the phase-B chokepoint pins inverted as designed).
> Showcase: the mandelbulb demo runs volumetric at full-strength colors
> (the ×0.1 anti-blowout dimming is gone).
>
> **Status**: **Phase 2 IMPLEMENTED** (2026-07-20): per-element opacity via
> the color ALPHA channel (RGBA colors) for gsplats — see §5.4.1. Alpha is
> active in EVERY blending mode (linear contribution scale; volumetric maps
> it into optical depth w = −ln(1−a)); the classical importer now stores
> learned 3DGS opacity in alpha (amplitudes := 1) so imported scenes occlude
> correctly. The original per-splat `absorption_weights` array plan is
> SUPERSEDED by this. Points/lines RGBA + volumetric remain phases 3–4.
>
> **Status**: **Phase 1 IMPLEMENTED** (2026-07-19): gsplats +
> node-level κ, exactly per §4/§5 with the pre-implementation corrections
> below. Implementation deltas vs the text: (a) the layers-panel κ slider is
> additionally gated to gsplat/group layers (not just the volumetric mode) so
> points/lines never show a dead control; (b) `LabeledSlider` grew a
> `setVisible()` for the mode-conditional control; (c) both material
> `clone()`s round-trip `absorption` (the panel clones on ANY first
> interaction — a clone that reset κ to 1.0 was caught in review); (d) the
> S(τ) quotient divisor is guarded `max(τ, 1e-20)` on BOTH backends (GPU
> selects evaluate both lanes); (e) E2E I1 compares sampled pixels within one
> page session at per-sample tolerance ≤6/765 (TAA/dither headroom), not
> bit-exact screenshots. Phases 2–4 (per-splat weights, points, lines) remain
> open. Design settled 2026-07-19 (mode name, κ semantics,
> opacity-scales-density rule, per-splat weights spec'd-but-deferred,
> gsplats-first phasing). File/line references were verified against main
> `38c6eb19` at design time. Pre-implementation review corrections
> (2026-07-19): the TSL output branch is BUILD-TIME, so additive↔volumetric
> DOES require a graph rebuild (§5.4, risk #6); layer-state κ inits from the
> RAW attr (§5.5); phase-1 points/lines get an additive-state fallback (§5.1);
> the additive-ladder energy compensation does NOT apply to volumetric in
> phase 1 (§6); E2E expected blend state needs a per-geometry split (§8).
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
  identical framebuffer RGB arithmetic to `additive`'s `One/One` (the
  destination-ALPHA accumulation differs, invisible on the `alpha:false`
  canvas). A `volumetric`
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
  sum-projected. But the TSL rebuild boundary must NOT be keyed on
  `usesPeakProjection` alone: the fragment **output branch is chosen at graph
  build time** (a JS conditional on `config.blendingMode`,
  `shader-tsl.ts` (the build-time normal-mode output branch)), so an additive ↔ volumetric switch changes the graph
  even though the projection doesn't. The rebuild predicate
  (`material-tsl.ts` (the `applyBlendingMode` rebuild predicate)) generalizes its `premultChanged` term to an
  `outputBranchChanged` term covering BOTH `isNormalMode` and `isVolumetricMode`
  crossings (§5.4).
- **Points/lines in phase 1**: the shared mode tuple means the panel dropdown
  offers `volumetric` for every geometry type, and Python accepts it on any
  node. Point/line materials intercept it in `applyBlendingMode` and apply the
  **additive** state instead (the exact κ = 0 limit of volumetric), keeping the
  requested mode in `userData.blendingMode` so stored scenes upgrade
  automatically when phases 3–4 implement the real math. Without this, an
  unhandled mode falls through to normal-mode alpha-over state — silently wrong.

### 5.2 Depth-sort gating: `needsDepthSort(mode)`

New predicate in `blending-state.ts`:

    needsDepthSort(mode) = isNormalMode(mode) || isVolumetricMode(mode)

replacing `isNormalMode` at every order-dependence gate:

- `rendering/depth-sort-coordinator.ts:242` (sort dispatch),
  `:519` (renderOrder-bias clearing), `:610-618`
  (`noteDepthSortBlendingModeSwitch` transition logic — switching *to* a sorted
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
  config pass-through — the GSPLAT factory call only in phase 1
  (`rendering/material-manager.ts`); the point/line factory configs
  deliberately omit it (their materials have no `uAbsorption` until
  phases 3–4).

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
- TSL twin: mirror as a **build-time JS branch** on `config.blendingMode` in
  `shader-tsl.ts` (the normal-mode output branch is the template; TSL
  `.select()` is deliberately avoided for structural branches because it
  materializes both sides), keeping 1:1 math with the GLSL. Because the branch
  is build-time, `material-tsl.ts`'s rebuild predicate must fire on
  any `isVolumetricMode` crossing:

  ```ts
  const outputBranchChanged =
    previousMode === undefined ||
    isNormalMode(previousMode) !== isNormalMode(mode) ||
    isVolumetricMode(previousMode) !== isVolumetricMode(mode);
  ```

  (replacing the old `premultChanged`; `projectionChanged` stays). Parity is
  enforced by `tsl-shader-parity.spec.ts` and the codegen snapshots (§8).
- **Discard interactions**: the color discard (`max(adjusted.rgb) < 1e-4` —
  the zero-color discards in `shader-glsl.ts` and the TSL twin) must be bypassed when τ is significant —
  a black splat still absorbs (a pure-ink occluder via gain → 0 must keep its
  optical depth). Under `LUXAR_VOLUMETRIC`, discard only when the color AND τ
  are both negligible. The earlier intensity discard (`:409`) stays: the τ it
  can drop is bounded by κ·opacity·1e-4 per fragment — invisible at slider
  κ ≤ 10 (risk #7).

### 5.4.1 Per-element opacity via the color ALPHA channel (Phase 2 — IMPLEMENTED)

Phase 2 does NOT add a parallel `absorption_weights` array (the original plan,
superseded 2026-07-20). Instead the `colors` attribute widens from strictly
`(N, 3)` RGB to optionally `(N, 4)` RGBA, and the **alpha column is per-element
opacity aᵢ ∈ [0, 1]** — one new concept, no new parameter, and it rides inside
`colors` so almost every gsplat op carries it for free (mask/permute/concat).

**Per-mode consumption** — alpha is active in EVERY blending mode, each
consuming it the way it consumes node opacity (a splat's rendered mass is A·aᵢ,
so the two decouple emission from opacity):

| mode | how aᵢ enters |
|------|---------------|
| additive / luminous / max / opaque | `intensity *= aᵢ` (linear contribution scale) |
| normal | coverage-alpha × aᵢ (true per-element alpha compositing) |
| volumetric | `intensity *= w(aᵢ)` **before** τ, where `w(aᵢ) = −ln(1 − min(aᵢ, 1−1/512))` |

The volumetric mapping makes a splat's peak rendered alpha reproduce aᵢ exactly
(3DGS-faithful) and self-screens emission to ≈ c·aᵢ. Dilute limit: `w ≈ a` as
a → 0, so volumetric and additive agree there (the same κ→0 coherence carried to
per-splat alpha); at large a volumetric is intentionally denser
(optical-depth semantics). Mid-alpha renders therefore differ between modes —
documented, not a bug.

**Storage / encoding**: `colors` shape `(N, 4)`; alpha in `[0, 1]`, validated
(finite, bounded) and never HDR (the SDR/HDR autodetect and display-range scan
look at RGB only, `dataset_writers/colors.py`). Codecs are channel-agnostic
(`rgb_uint8` element-wise; `geolog_perchannel` derives column count from data),
so **no format-version bump** — old readers that hardcode 3 are the only ones
affected, and Luxar's own readers key off the array shape. `absent ⇒ aᵢ = 1`
(the writer stamps 1.0 into texel3.y unconditionally — pool textures are reused;
a full RGB dataset allocates no wider buffer).

**Gate**: a uniform `uHasElementAlpha` (0/1, from the loaded color layout, set
per-commit) gates ONLY the volumetric w-mapping — RGB data carries the identity
alpha 1.0, which must NOT map to w ≈ 6.24. The linear per-mode factor needs no
gate. `uHasElementAlpha` is a plain uniform, deliberately NOT a shader define,
so toggling it never triggers a TSL graph rebuild.

**Restriction**: only direct-color splats get per-element opacity. Intensity/
colormap (CLUT) splats have no stored color, so alpha falls back to the node
dials — correct for fitted microscopy, where τ ∝ amplitude is already the right
model. (LUT-alpha ramps for CLUT mode: a coherent future extension, out of
scope.)

**Import / export**: the classical importer stops folding opacity into
amplitude — `alpha := o`, `amplitudes := 1`, and `stats["interop"].
opacity_in_alpha = True` marks the provenance. Additive renders stay visually
identical (c·a vs the old baked c·o); normal and volumetric become *correct*
(dark solid surfaces occlude). INRIA PLY export reads alpha verbatim into both
data-driven opacity policies, so the round-trip is lossless. Mass-ranked ops
(LOD-ladder scorers, culling, `gsplat info`) switch to the alpha-effective
amplitude A·aᵢ (`gsplats/utils/alpha.py::effective_amplitudes`) so imported
scenes keep a meaningful energy order.

**LOD merge**: a substitutive reduction aggregates the alpha column in
**w-space** — the mass-weighted mean of −ln(1−aᵢ), mapped back through 1−e^(−w)
(`_substitutive/kmeans_lloyd.py`). Optical depth composes linearly; opacity does
not, so an o-space mean would over-report transmittance when a bin mixes opaque
and translucent members. A uniform-alpha bin is a fixed point.

The `ALPHA_CLAMP = 1 − 1/512` literal is shared between Python
(`gsplats/utils/alpha.py`) and both viewer shaders (GLSL + TSL) so aggregation
and rendering agree.

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
- Layer init reads the **raw** node attr (`node.attrs.absorption ?? 1.0`),
  exactly like opacity — NOT the composed value. The panel's
  `composeEffective` substitutes each layer's live values per ancestry node, so
  a composed init would multiply ancestor κ in twice. (The composed-init rule
  applies only to nearest-setter-wins attrs like `blending_mode`;
  multiplicative attrs must stay raw.)

---

## 6. Interactions with existing systems

- **LOD / streaming**: substitutive levels pin total mass per barrier group by
  default, and τ ∝ mass along the ray ⇒ absorption strength survives LOD
  switches without popping. **Correction (phase 1)**: the additive-ladder
  energy compensation 1/e(k) (PR #541) is gated by `BLENDABLE_MODES` =
  {additive, luminous} (`scene/lod-fade.ts:49,142`), so a volumetric leaf
  streaming a partial ladder gets NO compensation — the brightening pop the
  mechanism removes returns for volumetric. Since opacity scales τ (§3.1),
  applying it would be first-order correct, but `BLENDABLE_MODES` also gates
  the cross-fade (deferred below), so enabling one without the other means
  splitting that predicate — a **documented follow-up**, not phase 1. Chunks
  arrive in energy order, not depth order: fine, the sort worker re-sorts on
  every commit (Phase-2 sorting contract), and I3 bounds the transient error.
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
   stamps (§4), CLI (`cli/gsplat_ops/scene_commands.py` — mode help +
   `--absorption` threaded into both `add_gsplats_*` call sites), docs mode
   lists (§8). Points/lines materials get the additive-state fallback (§5.1).
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

**Phase 3 — points (IMPLEMENTED 2026-07-24)**: extend the depth-sort infrastructure to point nodes
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

- `src/tests/e2e/blending-expected-state.ts` — **per-geometry split**:
  `EXPECTED_BLEND_STATE.volumetric` = the gsplat premultiplied state
  (CustomBlending 5, AddEquation 100, One 201, OneMinusSrcAlpha 205,
  depthWrite false), plus an explicit points/lines expectation = the ADDITIVE
  row (the phase-1 fallback, §5.1) consumed by the two per-mode loops
  (`blending-modes.spec.ts` points loop, `lines-blending-modes.spec.ts`).
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
   define/projection/blend combinations (plain, NORMAL_PREMULT, VOLUMETRIC),
   and every non-volumetric branch must clear `LUXAR_VOLUMETRIC` (including
   the normal branch — a volumetric→normal switch must not strand the define).
   The campaign's mutation-tested branch tests must grow with it. The TSL
   rebuild boundary is NOT already correct: additive↔volumetric crosses a
   build-time output branch without crossing `usesPeakProjection` or
   `isNormalMode`, so the predicate gains the `outputBranchChanged` term
   (§5.4); a fail-first rebuild-boundary test pins it.
7. **Discard-threshold τ loss.** The intensity early-discard drops fragments
   whose τ ≤ κ·opacity·1e-4 — invisible at slider range (κ ≤ 10 ⇒ α ≲ 0.1%),
   lossy only for extreme Python-set κ (~10⁴). Documented at the discard site;
   the COLOR discard, by contrast, is bypassed in volumetric (§5.4) because a
   black splat must still absorb.

---

## 10. Changelog

- **2026-07-24** — Phase 3 (points) implemented: isotropic chord-integral
  rayMass (`POINT_CHORD_SCALE = √(π/K)`), `LUXAR_VOLUMETRIC` output branch
  in both point shader backends, `uAbsorption`/`uHasElementAlpha` on both
  point materials, points RGBA colors end-to-end (validator `channels=(3,4)`,
  accumulator/loader/projection/texel-writer stride threading, alpha in
  texel2.y), `effectiveGeometryMode` narrowed to a lines-only fallback,
  panel κ slider shown for points layers, mandelbulb demo switched to
  volumetric (×0.1 dimming removed).
- **2026-07-19 (later)** — Pre-implementation review corrections: TSL output
  branch is build-time ⇒ additive↔volumetric requires a rebuild
  (`outputBranchChanged` predicate, §5.4/risk #6); phase-1 points/lines
  additive-state fallback (§5.1); layer-state κ inits raw, not composed
  (§5.5); energy compensation does not fire for volumetric in phase 1 (§6);
  per-geometry E2E expected-state split (§8); volumetric color-discard bypass
  + intensity-discard τ-loss bound (§5.4, risk #7); CLI path erratum.
- **2026-07-19** — Initial spec. Design decisions settled with the user:
  mode name `volumetric`; κ as node-level composable `absorption` attr
  (multiplicative, identity/default 1.0, default-stamped like opacity);
  opacity scales density (emission AND τ); per-splat `absorption_weights`
  spec'd but deferred to phase 2 with the 3DGS opacity mapping; phasing
  gsplats → per-splat → points → lines.
