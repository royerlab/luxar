# lod

LOD display-policy configuration slice. Owns how a change of a `kind=lod`
group's displayed level reaches the screen; which level to select is the
registry's business (`scene/lod-group-registry.ts`, `scene/lod-selector-math.ts`).

- `data.ts` — `lodConfig: LodConfig` (defaults below).
- `types.ts` — `LodConfig`; each field's doc says why it has its value.
- `validate.ts` — `validateLod`: every field finite and in range (table in
  the file); `preloadBandFraction` strictly below `preloadExitBandFraction`
  (else the entry and exit bands coincide and a camera at the edge restarts a
  visit, and a reload, per wobble); `playbackKeepBudgetFraction` ≥
  `playbackLoadBudgetFraction` (the keep budget is the admission budget's
  hysteresis).

Registry tunables (were module constants of `scene/lod-group-registry.ts` /
`scene/lod-selector-math.ts`; same values):

| field | default | what it tunes |
|---|---|---|
| `preloadExitBandFraction` | 0.5 | exit half-width of the preload band |
| `fineReloadSettleMs` | 130 | settle debounce before a stale fine level reloads |
| `playbackLoadBudgetFraction` | 0.8 | share of the playback period a finer level's reload must fit to be admitted |
| `playbackKeepBudgetFraction` | 1.0 | share the current aspiration (and levels below) must fit to be kept |
| `loadEwmaAlpha` | 0.3 | weight of a new load sample in the reload EWMA |
| `playbackProbeIntervalMs` | 1000 | how often a capped level is re-measured during playback |
| `staleHoldMs` | 250 | how long a stale finer level may stay up while the aspiration re-commits |
| `staleHoldMinRatio` | 0.5 | how much coarser the fresh fallback must be before that hold is worth it |
| `failedRetryMs` | 2000 | failure cooldown before a lazy level is retried |
| `lazyActivationRequestTimeoutMs` | 2000 | how long a deferred part's activation request waits for a pass |
| `partitionFrustumMargin` | 0.1 | screen-space pad of the partition frustum gate |
| `hysteresisRatio` | 0.1 | downgrade hysteresis of the threshold and footprint picks (mirrored in `luxar.io.lod_screening`) |
| `maxMedianFootprintPx` | 1.5 | GSplat footprint pick's median-sigma limit (CSS px) |

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
until the metric leaves the wider exit band (half-width
`preloadExitBandFraction` × the gap) and comes back.
