# lod

LOD display-policy configuration slice. Owns how a change of a `kind=lod`
group's displayed level reaches the screen; which level to select is the
registry's business (`scene/lod-group-registry.ts`, `scene/lod-selector-math.ts`).

- `data.ts` — `lodConfig: LodConfig`. `fadeMs: 250`, `preloadBandFraction: 0.4`.
- `types.ts` — `LodConfig`.
- `validate.ts` — `validateLod`: `fadeMs` must be finite and ≥ 0;
  `preloadBandFraction` must be in [0, 0.5) — strictly below the 0.5 exit band,
  or the entry and exit bands coincide and a camera at the edge restarts a
  visit (and a reload) per wobble.

`fadeMs` is the duration of the dissolve between the outgoing and the incoming
level when a blendable (additive / luminous / volumetric) group changes level.
The dissolve is driven by time since the change, not by the camera's distance
to a threshold, so a parked camera always settles on a single level; a retarget
mid-dissolve continues from the current opacities. `0` makes every change a
hard swap, and `?noLodFade` turns the dissolve off for a session.

`preloadBandFraction` sizes the band around each level threshold inside which
the level across it is loaded in the background, hidden, so crossing the
threshold starts the dissolve at once rather than after that level's load. The
half-width is this fraction of the smaller adjacent inter-threshold gap (0.4 is
the band the retired distance cross-fade drew both levels in). It applies only
while the dissolve is on and never during playback; `0` turns it off. A level
released under VRAM pressure while the camera stays in the band is not reloaded
until the metric leaves the wider exit band (half-width 0.5 × the gap) and comes
back.
