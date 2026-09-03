# Layers Panel

Napari-inspired per-layer control panel for Luxar scenes.

## Overview

The Layers panel exposes scene graph nodes marked with `layer=True` (set in the Python API) as controllable layers in the viewer. Data nodes (`points`, `lines`, `gsplats`, `mesh`) and container `group` nodes may both be exposed as layers; for groups, controls apply to every data descendant. Specialized groups (`kind: 'lod'`, `kind: 'partition'`) appear under their resolved `display_type` rather than as `group`, and carry an extra badge (and, for LOD groups, an inline level selector). Each layer provides:

- **Visibility toggle** (eye icon) — initial state taken from the node's `visible` attr (default `true`)
- **Display range** / **Colour range** [min, max] — windows scalar data across the colormap or maps direct RGB input to the full output range, respectively
- **Gamma** correction
- **Opacity**
- **Blending mode** (additive, volumetric, normal, max, opaque, luminous)
- **Layer order** — the authored cross-layer draw order (`docs/guides/specs/LAYER_ORDER_SPEC.md`). A number field rather than a slider, because the value is a signed JavaScript safe integer and must be able to be **blank**: empty (placeholder `auto`) means _unset_, while any explicit value, **0 included**, remains distinguishable as author intent for the panel and diagnostics. Both unset and authored 0 resolve to band 0; different band values are what override the inferred bounding-sphere containment order. Higher draws nearer the camera, like a CSS `z-index`. Layers on different levels never interleave whatever the camera does, which is the whole point: it converts an inferred, geometry-dependent order into a stated one. Sparse values (10/20/30) leave room to insert a layer later. Unlike every other control here the level is not a material uniform but a cross-node **sort key**, so `applyLayerOrder` writes `userData.layerOrder` on each affected leaf and wakes the render loop rather than going through `applyComposed` — there is no `mat.updateX` to call. It cannot reorder across the opaque/transparent split (an opaque layer always draws before a transparent one, and an authored opaque level at or above a transparent level warns), and it cannot make interpenetrating concave layers _correct_ — it buys stability, not correctness.
- **Absorption** (κ) — only shown when the layer's effective blending mode is `volumetric`; all three geometry types implement the emission–absorption math, and κ = 0 is exactly the additive limit. Since the 2026-08-02 ray-mass unification τ = κ · rayMass with rayMass the same peak-alpha-normalised quantity the additive branch emits, so κ is dimensionless and comparable across points, lines, gsplats and scene scales — one fixed **logarithmic** track (nominally 0.001–10) serves every layer; the former per-layer geometry-derived bounds are gone. Both ends still move (within hard clamps) to keep an AUTHORED κ outside the nominal span on the track, so the readout always shows a value the thumb can express. Position 0 is a dedicated stop for exactly κ = 0 (the geometric span starts one step in, so the floor round-trips). On a log track the component mirrors the readout into `aria-valuetext`, since the input's native value is a position. See `absorption-range.ts`.
- **Ambient** / **Shade falloff** / **Specular** / **Shininess** — shown only for a mesh whose resolved shading mode is `smooth` or `flat`. Hidden for `shading="none"` because the unlit shader reads none of the four uniforms, and hidden for points/lines/gsplats because those types are emissive per-element sprites with no surface orientation. Ambient is the wrapped diffuse term's shade floor (what keeps a face-away silhouette readable rather than black; `1.0` removes the diffuse gradient), while shade falloff shapes that gradient from the fixed view-space light direction; specular and shininess control the additive Blinn–Phong highlight. Linear tracks, unlike κ's log one: the bounded fractions and small exponents have meaningful midpoints, rather than being scale-free coefficients spanning decades. The exponent tracks start at the material's `0.001` clamp rather than 0, avoiding undefined `pow(0, 0)` at a fragment with zero wrapped diffuse response.
- **Alpha cutoff** — mesh only, AND only in `opaque` mode (the type gate plus a mode gate, the narrowest condition in the panel): that is the one mode whose fragment stage applies the hard cutout, so in any other mesh mode the threshold is read by no branch of the shader. The drag also reaches the mesh's PICK material, because the pick pass applies the identical cutout (§6.5) — a threshold that moved on screen but not in the pick buffer would leave a freshly-dissolved region still hoverable.
- **Colormap** (for gsplats with scalars/amplitudes, scalar-backed points/lines/mesh, and groups that fan out to such descendants)
- **Classes** (GSplats with `label_vocabulary`) — switch between authored and categorical colours, or isolate one exact class; the filter also reaches the pick material so hidden classes are not hoverable. Hidden when no vocabulary is available
- **Active level** (LOD groups, and partitions wrapping LOD groups) — `auto` or lock to a specific level

A mesh layer's `blendingMode` is stored **resolved**, not as composed: `volumetric` has no meaning for a zero-thickness surface, so the mesh material maps it to `opaque` and stamps the resolved mode — and `resolveLayerBlendingMode` applies the same mapping before the value reaches `LayerInfo`. Without that, the panel disagreed with the render in two visible ways at once: it showed **Absorption** (which no mesh shader reads) and hid **Alpha cutoff** precisely when the cutout was active. Resolving at the point of storage rather than at each display gate means every consumer — the Blend dropdown's own displayed value included — sees the mode that renders, so explicitly picking `volumetric` on a mesh snaps back to `opaque`.

The five mesh appearance values are the one control group that does **not** compose along the ancestry, and are applied through their own `applyMeshAppearance` rather than through `applyComposed`: a shade floor is a per-surface appearance choice with no composition rule (multiplying two ambients would mean nothing), and the writer never stamps them on a group. A group layer over meshes therefore does not offer them, since a group control would have to mean "set all descendants" — a different verb from every other control here.

Rendering attributes compose along the scene graph per the Luxar composition spec: `opacity`, `absorption`, `gamma`, and `intensity` multiply through ancestors; `offset` adds; `blending_mode` takes the nearest ancestor's choice — except inside the edited layer's own subtree, where the layer's single Blend control wins (see [Blending mode inside a layer's subtree](#blending-mode-inside-a-layers-subtree)). Every panel mutation recomposes the effective attributes for each affected data-leaf (the layer itself, or every data descendant of a group layer) using live panel state for `layer=true` nodes and authoring-time zarr attrs for the rest. `colormap` composes nearest-setter-wins too (#1600), so a palette authored on a group reaches every descendant that can use one; the panel's own colormap control still fans out **imperatively** to each affected leaf material rather than going through composition, because a live dropdown change has no authored attr to compose from.

Edits made in the panel are viewer-only and not persisted back to the zarr store; reload the page to return to the authored state.

### Specialized groups (LOD / partition)

`group` nodes carrying a `kind` attr of `'lod'` or `'partition'` are surfaced
specially (see `LayerKind` in `layer-state.ts`):

- They render under their resolved `display_type` attr (one of `points` /
  `lines` / `gsplats`), never as `group`.
- A `kind=lod` layer gets a **`N LODs`** badge and an inline **Active level**
  dropdown. Selecting a level calls `LODGroupRegistry.setSelectorMode(path, …)`
  with either `'auto'` or `{ lockLevel: i }`; the status span shows the
  currently-rendering level 1-based (`L{i}/{n}`), matching the data-monitor
  chip and the dropdown labels, with an `(off-screen)` suffix when the
  frustum gate is holding the group at its coarsest level. A per-frame
  callback (`layers-lod-status`) keeps the readout live, so it tracks
  `auto`-mode swaps driven by camera motion — not only state changes.
- A `kind=partition` layer gets a **`N parts`** badge. When it wraps nested
  `lod_group` descendants, the badge combines counts as **`N parts × M LODs`**
  (M = max child count across the nested ladders) and the Active-level dropdown
  broadcasts the chosen mode to every nested `lod_group` path (clamped per-group
  by `setSelectorMode` on ragged ladders). The readout aggregates the live
  level across all nested groups as `L{i}/{n} · {N} groups`, widening to a
  range `L{min}–{max}/{n}` when parts diverge under `auto` (each part picks
  its own level by its own on-screen size); `{n}` is the max ladder depth, so
  it agrees with the dropdown's option count.

The LOD registry is looked up lazily via
`SceneLoaderManager.getInstance().getDefaultLoader()?.lodGroupRegistry` so the
panel doesn't import `scene/` directly (respecting the data → ui layer
direction). Locking a level wakes the animation loop (`requestRender`) so the
new active level paints even when the camera and slice are idle.

### Load-failure badge

When a node's loader throws (corrupt data, network failure, the "Vertex index N
not found" path), that failure used to surface only in the console and the
collapsed data monitor, reading as a blank-canvas camera/shader problem. The
panel now shows a per-row **error badge** (`.luxar-layer-row__error`, a
warning-triangle SVG with `role="img"` and an `aria-label` naming the reason)
plus an `--error` row tint (`luxar-layer-row--error`).

- The app injects an equivalent `FailedLoadsProviderPort` over the same live
  failure set the data monitor reads (each `SceneLoader.getFailedLoadsProvider()`
  call returns a new object, but all close over the loader's one `failedLoaders`
  map) via `LayersPanel.setFailedLoadsProvider(provider)`, wired in
  `core/app/dataset/load-dataset.ts` right AFTER `initFromScene` (whose `clear()`
  resets any prior provider first). Cleared on dispose.
- A row is in error if its own path failed OR any descendant leaf failed
  (`failedPath === layer.path || failedPath.startsWith(layer.path + '/')`), so a
  failure inside a `kind=lod` / `kind=partition` group lights up the group's row.
- The tooltip prefers the provider's per-path reason (`getFailedReason`, from the
  loader's `error.message` / classified kind), falls back to a generic message,
  and appends `(N parts failed)` when more than one descendant failed.
- Refresh is signature-gated (mirroring the data monitor's
  `lastFailedLoadsSignature`): the per-frame `layers-lod-status` callback (gated
  on panel visibility) only touches the DOM when the failed set — folded with
  each path's reason — changes. The signature reset sentinel is `null`, so an
  empty set / `setFailedLoadsProvider(null)` still clears badges. `renderList()`
  invalidates the signature so a row rebuild (e.g. `resetAllLayers`) re-applies.

This covers per-node LOADER failures only; render-thread texture-capacity
truncation is a separate follow-up.

## Usage

### Python (scene authoring)

```python
from luxar import LuxarZarrCompiler, Dimensions

with LuxarZarrCompiler("scene.luxar.zarr") as c:
    scene = c.create_scene(dimensions=dims)
    scene.add_points("GFP", positions=..., colors=..., layer=True)
    scene.add_points("mCherry", positions=..., colors=..., layer=True)
    scene.add_group("_internal")  # Not a layer — no panel entry
```

### Viewer

Press **L** to toggle the Layers panel (Escape closes when focus is inside the panel).
Keys used by panel controls (arrows, Home/End/Page Up/Page Down, Enter/Space, and
menu keys) stay inside the panel, while unrelated viewer shortcuts remain available.

- **Click** a layer to select it
- **Ctrl/Cmd+Click** to toggle additional layers
- **Shift+Click** for range selection
- **Arrow Up / Arrow Down** move the keyboard focus through rows (and select on simple navigation)
- **Enter / Space** select the focused row (honouring Ctrl/Cmd/Shift modifiers)
- The bound labels on either side of the display-range slider are click-to-edit and scroll-to-adjust (hold **Shift** for finer increments)
- Controls below the list (display range, gamma, opacity, absorption, the five mesh appearance sliders, blend, colormap) apply to all selected layers; the absorption, mesh-appearance, colormap, and **Active level** controls auto-hide when the primary selected layer doesn't support them

## Architecture

```
layer-state.ts     Pure data model, selection logic (re-exports the min/max ↔ intensity/offset math from rendering/display-range.ts)
layers-panel.ts    DOM panel (list + lifecycle), event handling; facade over the two below
layer-controls.ts  LayerControls — the controls section (sliders, blend/colormap/LOD selects, LOD readout)
layer-apply.ts     LayerApplyEngine — attr composition + scene/material application
luxar-material.ts  LuxarMaterial contract + colormap-vs-direct routing helpers
range-slider.ts    Dual-thumb [min, max] slider (click-to-edit + scroll-adjust bounds)
labeled-slider.ts  Single-thumb labeled slider (gamma, opacity, absorption, mesh shading); linear or log track
absorption-range.ts κ slider log-track bounds (fixed nominal span, widened onto authored κ) + κ readout format
attrs-utils.ts     Pure helpers: clampGamma, blending-state mapping, liveLayerAttrs
```

The public entrypoint is `../layers.ts` (parent file); it re-exports only
`LayersPanel`. The rest — `LayerStateManager`, the `computeUniforms` /
`computeDisplayRange` math and the `LayerInfo` / `DisplayUniforms` /
`SelectionMode` types (`./layers/layer-state`), and `RangeSlider`
(`./layers/range-slider`) — are imported directly from their leaf modules.

## Display Range Mapping

The UI shows one [min, max] slider with mode-specific wording:

- **Display range** for a colormapped layer is a scalar data window mapped across the colormap.
- **Colour range** for a direct-colour layer is the input RGB window mapped to the full output range. It is a live gain/offset control, not a report of the layer's data extents.

For direct colours, the window maps to the existing shader gain/offset uniforms:

```
intensity = 1 / (max - min)
offset    = -min / (max - min)
```

For colormapped layers, the window instead controls the LUT lookup through
`uScalarMin` / `uScalarScale`; the post-LUT colour gain and offset stay at the
identity so the window is not applied twice. Gain/offset remain the internal
composition currency used to recover and propagate the window.

### Which window a layer STARTS at

The window maps the **rendered value** to `[0, 1]`, so the default depends on
what that value is (`layer-state.ts::initialDisplayRange`):

| Layer renders                                                           | Starting window                                                                                                                | Why                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| through an effective colormap (own, descendant, or consumable ancestor) | `scalar_data_range`, else `amplitude_data_range`, else a descendant leaf's (`deriveScalarRangeFromDescendants`), else `[0, 1]` | the value is a scalar; gsplat amplitudes are heavily right-skewed, so a linear `[0, 1]` window renders near-black (#522)                                                                                                |
| direct RGB colours                                                      | `[0, 1]` — the identity                                                                                                        | the value IS authored colour. Windowing it on `color_data_range` is an unrequested contrast stretch: a uniform grey `(0.72, 0.74, 0.78)` has range `[0.72, 0.78]` → gain 16.7 / offset −12 → renders **saturated blue** |

`color_data_range` therefore never sets the starting window — it only widens the
**slider bounds** for direct-colour layers, so stretching authored colours stays
a one-drag operation.

The "or a descendant" walk stops at a nested `layer=true` node: that node is its
own row with its own colormap control, so a palette derived from it would be a
snapshot that goes stale on the first inner edit. Writers route `layer` onto the
wrapper only, so a `kind=partition` / `kind=lod` layer never has layer
descendants and is unaffected.

An ancestor palette counts only when the layer can actually consume it:
gsplats always can, while points / lines / mesh require scalar data. A wrapper
group inherits only when at least one data descendant meets the same rule.

Bounds are the union of the starting window, the recovered authored
`intensity`/`offset` window, and (direct colour only) `color_data_range` —
`[min(dataMin, displayMin), max(dataMax, displayMax)]` — so the `<input>` never
silently clamps the thumb on first render.

#### Toggling the colormap

On an off↔on MODE flip, `setColormapWindow` re-defaults the window AND the
bounds to the new mode — a window carried over from the other mode is
meaningless, and merely widening the bounds would leave the useful window as an
unusable sliver (an amplitude window of `[1e-4, 0.02]` inside `[0, 1]` bounds
is 2% of the track). Both land exactly where a natively authored layer of that
mode inits, which is why `LayerInfo` keeps `colorDataRange` alongside
`scalarDataRange`. Switching between two active palettes is NOT a mode flip —
the rendered value stays the same scalar, so a user-adjusted window survives.

Two things the select handler must do that are easy to miss:

- **Re-render.** It runs with `controlsInteracting = true`, which suppresses the
  state-change re-render, and `RangeSlider` emits values parsed from its own
  `<input>` elements. Without an explicit `render()` the thumbs keep the old
  window and the first drag writes it back, reverting the re-default.
- **Honour the fail-closed guard.** `applyColormap` returns whether any leaf
  actually took the LUT. The C1 guard suppresses it on leaves with no scalar
  data bound (for example, an explicitly colormapped group over scalar-less points);
  such a layer keeps rendering direct colour, so the handler puts the identity
  window back rather than applying a scalar range as a colour gain.

For a MIXED group layer (some leaves accept the LUT, some are suppressed), the
layer keeps the scalar window for its colormapped leaves, and `applyComposed`
routes per leaf: a leaf whose material is not colormap-active while the layer's
window is a scalar one (`LayerInfo.scalarWindow`) gets the identity window
instead, so the scalar range is never applied to authored RGB as a colour gain.

#### The window each LEVEL actually receives

The panel composes ONE window per layer, but `getAffectedDataLeaves` fans it out
over **every** data descendant of a group layer, and those leaves need not share
a scalar range. Across a `kind=lod` ladder that is a genuine problem: gsplat LOD
merging **sums** amplitudes, so a coarsened level is the same physical signal at
a different numeric SCALE, and the producer
(`packages/luxar/src/luxar/io/_compiler/gsplat_assembly.py`) computes
`amplitude_data_range = [min, p99.9]` over each splat set's OWN amplitudes.
The layer's reference range is merely whichever descendant
`deriveScalarRangeFromDescendants` picked: the one with the largest `n_splats`,
or — since points/lines/mesh write `n_points` and never `n_splats`, and gsplat
siblings can tie — simply the first descendant visited. Pushing the composed
window verbatim into every level rendered the whole layer on the reference
level's window: per-level differentiation was discarded, and a level streamed in
after the last panel commit kept its own window until the next one, so load
order decided how a level looked (#1753).

So for a **colormap-active** leaf under a LOD ladder, `applyComposed`
re-expresses the layer's window in that leaf's own range before pushing it,
preserving its relative position
(`rendering/display-range.ts::remapWindowToLeafRange`): the middle 40% of the
layer's signal stays the middle 40% of each level's signal. The window is
returned **unchanged** when either range is missing, when either is
non-remappable — degenerate, inverted, or non-finite (`[x, x]` is legitimate:
every classical splat import has `amplitudes = 1`; a sub-`1e-10` reference span
means the layer's window was seeded from a single point and `t₀`/`t₁` come out
as `0/0`; and a zero-span _leaf_
range collapses the window to a point, which `computeScalarRangeUniforms`
answers with the **LUT midpoint** (#631), i.e. one flat neutral colour for the
whole leaf) — or when the two ranges are equal, the common case, short-circuited
so an ordinary single-range layer keeps a bit-exact window.

How much this moves depends on the ladder. On a plain gsplat one it is modest:
for a default `luxar gsplat lod --recipe levels -K 4 -L 3` store the four levels'
`amplitude_data_range` top ends are 0.43884 / 0.44417 / 0.43637 / 0.39260 — a
≤13% spread, since the range is `[min, p99.9]` and mass conservation pins the
peak (the bottom ends differ more, ~8x: 7.06e-05 vs 5.60e-04). "LOD merging sums
amplitudes" describes the mechanism, not the magnitude. The dramatic case is the
MIXED lift ladder — `add_points(..., scalars=…, colormap=…, layer=True,
substitutive_lod=True)` lifts its coarse levels to gsplats (direct colour,
`amplitude_data_range` + `n_splats`) and leaves the finest level as colormapped
`points` (`scalar_data_range`, no `n_splats`). The reference is then a lifted
sibling's AMPLITUDE range, a different physical quantity from the one
colormapped leaf's scalar: on a real store an amplitude window of
`[0.037, 2.817]` went into a scalar spanning `1.035 … 280.862`, so everything
above 2.817 — 99.4% of that span — clamped to the top of the LUT and rendered as
one flat colour.

##### Only across a LOD ladder — never across a partition

A LOD _level_ and a partition _part_ are not the same kind of sibling, and only
one of them is a scale change. Levels are alternative representations of the
WHOLE object, so re-expressing the window per level is exactly what makes every
level render one physical value identically. Parts are disjoint **spatial**
subsets of one field at the **same** scale, differing only by content — so a
per-part window is per-tile auto-contrast: the same value renders as a different
colour in different tiles and the colormap goes non-monotone, with a visible
discontinuity at every BSP seam. On the repo's own
`packages/luxar-viewer/tests/fixtures/test_partition_layer.luxar.zarr`
(a `layer=True, kind=partition` gsplats wrapper — and `layer=True` is the default
of `luxar gsplat convert`), `tiles/part_0` is `[0.5000, 0.7455]` and
`tiles/part_1` `[0.7542, 0.9998]`: remapping would ramp black→white across both,
stepping the field 0.7455 (white) → 0.7542 (black) at the seam. Saturated but
monotone is the correct failure.

`composedWindowIsInReferenceBasis` therefore requires every GROUP node from the
edited layer down to (excluding) the leaf to be `kind === 'lod'`. The
consequences are all deliberate:

| Layer shape                                                    | Remaps?                                                     |
| -------------------------------------------------------------- | ----------------------------------------------------------- |
| `kind=lod` over levels                                         | yes — this is #1753's case                                  |
| `kind=partition` over parts                                    | no — one shared window, as before                           |
| `overview` (lod cap + nested partition branch)                 | no-op in practice: the cap is eligible but IS the reference |
| `adaptive` (partition of per-tile lod groups)                  | no, anywhere                                                |
| a plain group layer over several colormapped leaves (channels) | no                                                          |

`adaptive` is conservative on purpose: the correct reference for a tile's ladder
would be that TILE's own finest level, not the layer's, and building that is a
bigger change than #1753 asks for. A plain group over two channels declines for
the same reason a partition does — two different physical fields are not one
field at two scales.

`overview` deserves the same honesty. Its tiles decline on the nested partition,
and its coarse cap, while structurally eligible, changes nothing: on a measured
`luxar gsplat lod --recipe overview` store the cap and all four parts carry 432
splats each, and `deriveScalarRangeFromDescendants` breaks that tie with a strict
`count > bestCount` while visiting the cap FIRST — so the cap IS the reference
and the equality short-circuit hands its window straight back. Net, this change
does nothing at all on an `overview` tree, and the fine parts still render on the
cap's window: `part_2`'s own `[0.00059, 0.19962]` on the cap's
`[0.000116, 0.44551]`, so its brightest splat lands at LUT 0.45 rather than 1.0.
Correcting that needs a per-branch reference — the same missing piece `adaptive`
would need.

Once #1691 / PR #1752 lands (it harmonizes `amplitude_data_range` across a gsplat
structure so siblings **share** a window) partition parts will carry equal ranges
and the equality short-circuit would make the remap a no-op there anyway. The
`kind` gate is the safety net until then, and for every store already written.

Mesh ladders never reach any of this, and the reason is the **producer's**, not
the panel's: `core/group/adders/mesh.py::_shared_scalar_window` derives ONE
window from the whole field before any split or decimation and stamps it on every
part and every level, precisely because "a level or a part that stamps its own
subset min/max renders the same value as a different colour". So mesh siblings
have equal ranges and the equality short-circuit fires. That is also why gsplat
LOD is the principled exception rather than an inconsistency: mesh decimation
drops vertices without touching the scalars they carry, whereas gsplat LOD
merging changes the amplitude SCALE, and only a scale change is something a
window has to be re-expressed across.

##### …and only from a window in the reference basis

`LayerApplyEngine.composedWindowIsInReferenceBasis` decides this over the same
ancestry chain `composeEffective` walks (walked once per leaf and shared between
them — `collectAncestorNodes` resolves each step with a linear `children.find`,
so a fan-out over a P-part wrapper is O(P²) and `applyComposed` runs on every
slider tick). `intensity` multiplies and `offset` sums over that whole chain with
live panel state substituted for `layer=true` nodes, so a window contributed at
or below the edited layer puts the composed one in a different basis, and
remapping it would corrupt a window that was already correct. Four arms decline:

- the edited layer is not on the leaf's ancestry;
- the edited layer AUTHORED a gain (`walkSceneGraph` then seeds `displayMin/Max`
  from `computeDisplayRange(intensity, offset)`, a normalized-gain window
  unrelated to `scalarDataRange`);
- a node strictly below the layer is itself **tracked as a layer** — regardless
  of its gain. A nested layer owns its own window and its own panel row, full
  stop. Testing its gain instead was a false negative for any child whose range
  is exactly `[0, 1]`: `computeUniforms(0, 1)` is `{1, -0}`, indistinguishable
  from "no window authored", and `[0, 1]` is the ordinary range of a normalized
  scalar, probability, mask or fraction;
- a non-layer node strictly below the layer authored an `intensity`/`offset`,
  which mirrors `resolveColormapWindow`'s "non-identity raw leaf gain ⇒ the
  composed gain IS the window" rule.

Ancestors _above_ the layer are not gated — but they are not remapped through
either. What gets re-expressed is the layer's OWN window (`displayMin` /
`displayMax`), the only one actually stated in reference units, and the ancestor
gain is re-applied to the result. Remapping the _composed_ window instead would
read a position that already carries that gain as if it were a reference-basis
position; the error cancels when `ref₀/refSpan == leaf₀/leafSpan` — notably when
both ranges start at 0, as gsplat amplitude ranges nearly always do — and not
otherwise. With `ref = [1, 3]`, `leaf = [0, 8]` and an ancestor `intensity = 2`
it gives `[-2, 2]`, a quarter of the LUT spent below the leaf's own minimum,
where node creation gives `[0, 4]`. Swapping the layer's contribution reproduces
`resolveColormapWindow`'s ancestor-only branch identically for every range pair,
which is what the two positive controls in the test file pin.

One asymmetry this does **not** close: a leaf declaring no range at all keeps the
composed window from the panel, while `create-gsplats-node.ts` gives the same
leaf `[0, 1]` at creation (`?? [0, 1]`). The two disagree, as they did before.

Direct-colour leaves are untouched by all of this: they have no scalar window at
all, and take the `identityLayerWindow` route above instead. `applyColormap`'s own
`updateScalarRange` write goes through the same per-leaf helper so the two
places that push a window cannot drift apart.

##### What the colormap legend says

`ui/colormap-legend.ts` labels the gradient with `layer.displayMin` /
`layer.displayMax` — the panel's one composed window. With per-level windows,
only the REFERENCE level renders exactly those numbers; a coarser level renders
the same relative band of its own range. For a `kind=lod` ladder that is
arguably more correct than before rather than less: a coarse level's amplitude 2
IS the fine level's 0.5 (the same physical structure, summed), so a
reference-basis label is the right statement about the SIGNAL. And before the
change the coarse levels rendered fully clipped while the legend advertised a
window nothing on screen was actually using. No code changed here.

### Blending mode inside a layer's subtree

`blending_mode` composes nearest-setter-wins, but a layer exposes exactly ONE
Blend control for its whole subtree. So within a layer, the **layer's** mode
wins: `LayerApplyEngine.composeEffective` ignores a `blending_mode` authored on
a descendant that is not itself a layer (a nested layer keeps its own live
value — it has its own control). Without this, a `kind=partition` /
`kind=lod` layer whose parts carry their own stamped mode had an inert Blend
control.

## Files

| File                                       | Purpose                                                                                                                          |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| `layer-state.ts`                           | `LayerStateManager`, selection logic; re-exports `computeUniforms` / `computeDisplayRange` (now in `rendering/display-range.ts`) |
| `layers-panel.ts`                          | `LayersPanel` class — panel/list DOM + lifecycle; facade over controls + apply                                                   |
| `layer-controls.ts`                        | `LayerControls` — controls-section DOM (sliders, selects, live LOD readout)                                                      |
| `layer-apply.ts`                           | `LayerApplyEngine` — attr composition + material application per data-leaf                                                       |
| `luxar-material.ts`                        | `LuxarMaterial` interface, `isColormapActive` / `applyColorAdjustments` routing                                                  |
| `range-slider.ts`                          | `RangeSlider` — dual-thumb input component with editable / scrollable bound labels                                               |
| `labeled-slider.ts`                        | `LabeledSlider` — single-thumb labeled input component (gamma, opacity, absorption); `linear` or `log` track                     |
| `absorption-range.ts`                      | `absorptionSliderRange` / `formatAbsorption` — κ track bounds (fixed log span, widened onto authored κ) + readout format         |
| `attrs-utils.ts`                           | `clampGamma`, `getBlendingState`, `liveLayerAttrs` — pure helpers (no DOM)                                                       |
| `../layers.ts`                             | Public entrypoint — re-exports `LayersPanel`                                                                                     |
| `../../styles/components/layers-panel.css` | Themed CSS styles                                                                                                                |
