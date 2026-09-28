# lod

LOD display-policy configuration slice. Owns how a change of a `kind=lod`
group's displayed level reaches the screen; which level to select is the
registry's business (`scene/lod-group-registry.ts`, `scene/lod-selector-math.ts`).

- `data.ts` — `lodConfig: LodConfig`. `fadeMs: 250`.
- `types.ts` — `LodConfig`.
- `validate.ts` — `validateLod`: `fadeMs` must be finite and ≥ 0.

`fadeMs` is the duration of the dissolve between the outgoing and the incoming
level when a blendable (additive / luminous / volumetric) group changes level.
The dissolve is driven by time since the change, not by the camera's distance
to a threshold, so a parked camera always settles on a single level; a retarget
mid-dissolve continues from the current opacities. `0` makes every change a
hard swap, and `?noLodFade` turns the dissolve off for a session.
