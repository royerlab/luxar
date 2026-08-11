# Line picking

Per-geometry picking sources for lines. Self-contained — no cross-geometry imports.

| File                     | Role                                                                                                                                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `material.ts`            | GLSL3 `THREE.ShaderMaterial` wrapper (`LinePickingMaterial`). Selects the screen-space, volumetric, or capsule shader pair from the same `?linePrimitive=` resolution the visual material uses. Recompiles on `needsUpdate` |
| `material-tsl.ts`        | WebGPU `NodeMaterial` counterpart (`LinePickingTSLMaterial`). Same `CameraAwareMaterial` contract; every rebuild (ortho flip, texture rebind, clone) dispatches to the factory matching the resolved primitive              |
| `pick.tsl.ts`            | TSL node factory for the screen-space primitive (`linePickWebGPUFactory` + `buildLinePickTSLNodesFromUniforms`). Used by the WebGPU material above and by the GLSL/TSL parity harness                                       |
| `pick-volumetric.tsl.ts` | TSL node factory for the volumetric primitive (#1352, `volumetricLinePickWebGPUFactory`): the visual stadium-stencil vertex with pick IDs, and the PEAK capsule fragment used unconditionally                               |
| `shaders.ts`             | Screen-space GLSL3 vertex/fragment source strings + `LINE_PICK_SOURCE: ShaderSource`. Uses full pick width (lines are already narrow); cap factor in fragment matches the visual shader                                     |
| `shaders-volumetric.ts`  | Volumetric GLSL3 sources + `VOLUMETRIC_LINE_PICK_SOURCE: ShaderSource` — the GLSL twin of `pick-volumetric.tsl.ts`                                                                                                          |
| `pick-capsule.tsl.ts`    | TSL node factory for the capsule primitive (#1352, `capsuleLinePickWebGPUFactory`): the capsule stencil vertex (bisector cuts + fold-cap rule) with pick IDs, quartic-profile brightness fragment                           |
| `shaders-capsule.ts`     | Capsule GLSL3 sources + `CAPSULE_LINE_PICK_SOURCE: ShaderSource` — the GLSL twin of `pick-capsule.tsl.ts`                                                                                                                   |

## The volumetric pick pass (#1352, behind `?linePrimitive=volumetric`)

The pick pass mirrors the visual volumetric primitive's stadium stencil verbatim
(bisector-cut overhang, depth-tilt disc reach, coverage fade), so the pick
footprint rasterizes exactly where the eye sees the line — end-on included,
which is the degenerate case the volumetric primitive exists to fix. The
fragment is the visual shader's **peak capsule lane, unconditionally**: a pick
buffer needs a brightness _ordering_ (hotspot on the centerline), not
radiometry, so none of the sum lanes' integral machinery (erf windows, mixed
splits, near-plane clip algebra) is present. Output contract is unchanged:
`vec4(nodeId, elementId-low16, brightness, elementId-high16)`,
`gl_FragDepth = 1 − brightness`.

Footprint agreement is pinned in `tsl-shader-parity.spec.ts`: **exact** (up to
the 1-px quantisation ribbon) against the peak-mode visual footprint (side-on
AND the V joint, which puts the bisector-cut ends under the contract), the
end-on additive disc, and **pick ⊆ visible** against the side-on additive
footprint — the additive family's separable radial·axial coverage keeps dim
(< ~10% brightness) corner crescents beyond the endpoints that no capsule
reaches, so those corners are visible-but-unpickable by design. The invariant
that must never break is the other direction: nothing is pickable where
nothing is visible. A separate half-space test pins the property none of the
above can see when it breaks on both backends at once: with the bisector cuts
active, each side of a joint decodes to exactly its own segment id.

## The capsule pick pass (#1352, behind `?linePrimitive=capsule`)

Same dispatch, third variant. The capsule pick shaders duplicate the visual
capsule's vertex stage exactly — stencil-local corners, 2D bisector cuts,
the width-gated fold-cap rule — and the fragment shades the same quartic
profile of the 2D point-to-segment distance, so `brightness = profile ×
fade` tracks the visible pixels one-for-one (the capsule is peak-shaped by
construction; there is no separate peak lane to select). Per-element alpha
and node opacity are ignored, matching the pick contract of the other two
primitives. Output contract and depth encoding are unchanged. Footprint
agreement is pinned in `tsl-shader-parity.spec.ts` against the fat visual
footprint (1-px quantisation ribbon).
