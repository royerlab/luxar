# Progressive Loader Shared Helpers

> Cross-geometry building blocks for the additive-LOD progressive loaders.

## Overview

The three progressive (additive-LOD) loaders —
`../../points/points-progressive-loader.ts`,
`../../lines/lines-progressive-loader.ts`, and
`../../gsplats/gsplats-progressive-loader.ts` — each load a sequence of LOD
levels and concatenate the per-LOD typed-array fields into a single
merged payload. Their whole per-view lifecycle is ONE engine here,
`additive-ladder-core.ts`; the geometry loaders are thin faces over it that
carry only their geometry-specific wrinkles (segment-index offsetting, Lines'
sharpness default, the Points picking map, GSplats' deeper lookahead
planner).

This folder holds no loader of its own — it is a pure helper module
imported by the geometry loaders two directories up (in
`src/data/points/`, `src/data/lines/`, and `src/data/gsplats/`).
`src/data/mesh/mesh-progressive-loader.ts` — the reveal ladder — shares
`concat-helpers.ts`, `streaming-policy.ts` and `pass-rollback.ts` too, plus
`measureLodBytes` from `slice-cache-helper.ts`; the rest of the SliceCache
helpers stay points/lines/gsplats-only, since a mesh is whole-node resident
and has no per-slice payload to cache.

## File Structure

```
progressive/
├── concat-helpers.ts      # Generic typed-array field concatenation across LOD parts
├── additive-ladder-core.ts # AdditiveLadderCore — the shared per-view ladder lifecycle
├── child-signal.ts        # createChildController — a per-read child of a pass signal that
│                          # inherits its priority/origin tags (a pinned pass's concurrent rungs)
├── constants.ts           # CACHE_HIT_THRESHOLD_MS — the shared streaming threshold
├── lookahead-signal.ts    # 'lookahead'-origin, 'speculative'-priority controller for the
│                          # next-rung prefetch (free navigation only; linked to the
│                          # scheduling update's abort signal)
├── pass-rollback.ts       # Shared retry/truncate/full-unwind decision for failed passes
├── streaming-policy.ts    # Per-pass LOD streaming decisions (playback / prefetch / refine)
├── slice-cache-helper.ts  # Shared SliceCache key/clone/restore/store logic, including separate
│                          # logical ladder depth for folded Points/Lines/GSplats payloads; also used by plain
│                          # spatial-index loaders caching a decoded slice as a 1-element ladder
│                          # — plus the in-flight shadow-store handoff (beginShadowStore /
│                          # awaitShadowStore) a foreground pass uses to adopt a running shadow
└── view-state-equal.ts    # viewStatesEqual — the memoized-noop / generation-reset linchpin
                           # (formerly three byte-identical copies, one per geometry loader)
```

## Components

### `additive-ladder-core.ts`

`AdditiveLadderCore<TData, TRung>` owns everything a Points / Lines / GSplats
progressive loader does per view; the loaders construct one and delegate their
public surface to it. A geometry supplies a `LadderGeometry` (timing-key kind,
log vocabulary, `countOf`, `concat(parts, { payloadLevelStarts, loadedLevels })`,
`empty()`, GPU bytes per element, whether a restored result is stamped with its
slice-cache origin) and optionally a `LookaheadPlanner` (default: warm the next
rung; GSplats plans up to three within L0 headroom). One pass:

1. records the playback budget / pinned depth (before the restore branch);
2. on a view change: aborts the old view's speculation (keeping a read-ahead of
   the slice being entered), DEPARTURE-stores the outgoing ladder, adopts an
   in-flight shadow store, then restores a cached ladder (a FULL one ends the
   pass) or resets;
3. streams rungs under `streaming-policy.ts`, warming rung k+1's index at
   refinement priority while rung k loads (B6); a pinned pass starts all its
   rungs at once under per-rung `child-signal.ts` controllers, and anything it
   does not commit is aborted and its accumulator released (A20);
4. logs a verbose summary, schedules the signal-linked speculative lookahead
   and the B5 ±1-slice read-ahead of rung 0 (free navigation only), and stores
   the ladder unless the pass was torn down.

The memoized concat (same reference for an unchanged view, prefix-lineage
parent per fresh result), `rollbackToPassStart` (`pass-rollback.ts`),
`ladderResidency`, `committedEnergyFraction`, `hasMoreLODs` and the
predicted-view `prefetchChunks` / `prefetchChunkBoundary` (which warm the
coarse rung) are all here too.

### `concat-helpers.ts`

Structurally-generic concatenation over any typed array `A`
(Float32Array, Uint8/16/32Array, Float16Array, …). The output dtype is
preserved by constructing from the first part's array.

| Symbol                                                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `interface ConcatTypedArray`                          | Minimal structural shape (`length`, `set(array, offset?)`) common to every typed array being concatenated.                                                                                                                                                                                                                                                                                                                                                                                                         |
| `concatRequiredField(parts, get, countOf, perItem?)`  | Concatenate a **required** field. Allocates `sum(countOf) * perItem` elements and copies each part at a running offset.                                                                                                                                                                                                                                                                                                                                                                                            |
| `concatOptionalField(parts, get, countOf, perItem?)`  | Concatenate an **optional** field with **all-or-nothing** policy: returns `undefined` unless _every_ part carries the field — except that **zero-row parts abstain** (see below).                                                                                                                                                                                                                                                                                                                                  |
| `concatColorsWhiteFilled(parts, get, countOf, label)` | The shared **colour** policy of all three emissive ladders: when any part carries colours the result does, and a part without them is filled with white (full scale for the dtype; RGBA alpha opaque). Each coloured part is first checked against its own declared layout (`assertColorLayout`, so an RGBA buffer omitting `colorComponents` throws instead of mis-striding); dtype and layout (RGB/RGBA) then come from the first coloured part, and a mismatch throws naming the level. Zero-row parts abstain. |

`get` extracts the field from a part, `countOf` returns a part's element
count (rows), and `perItem` is the components per element (e.g. `3` for
positions, `1` for scalar widths/radii). `parts` is assumed to have
length ≥ 1.

```typescript
import { concatRequiredField, concatOptionalField } from '../loaders/progressive/concat-helpers';

// Required: positions are ndim components per row.
const positions = concatRequiredField(parts, (p) => p.positions, count, ndim);

// Optional: radii only survive if every LOD part has them.
const radii = concatOptionalField(parts, (p) => p.radii as ScalarArray, count, 1);
```

**Zero-row parts abstain.** A part with `countOf(p) === 0` contributes no rows,
so it gets no vote on which optional attributes the merged result carries: it is
excluded from the presence gate, the copy, and the dtype comparison. This is
load-bearing because the canonical empty payloads **omit** their optional fields
rather than emitting zero-length arrays — so without the rule, one slice-culled
level of an additive ladder stripped those fields from every _other_ level too
and the node adapters substituted constant fills (issue #1456). What each
geometry stood to lose differs:

| Geometry | Fields routed through `concatOptionalField` | Lost without the rule                          |
| -------- | ------------------------------------------- | ---------------------------------------------- |
| Points   | `radii`, `sharpness`, `scalars`             | radius 0.5, sharpness 0.5, colormap suppressed |
| Lines    | `scalars` only                              | colormap suppressed (`hasScalars` clears)      |
| GSplats  | none                                        | nothing — unaffected                           |

Colours go through `concatColorsWhiteFilled` for all three geometries (its
zero-row parts abstain the same way). Lines has no `radii` field at all, and its
`sharpness` is present-but-`null` on the empty payload and merges via the
bespoke fill-with-default (0.5) path in `concatenateLinesData`. GSplats' empty
payload carries zero-length `Float32Array`s for its required fields.

When _every_ part is zero-row the gate falls back to all the parts, so a wholly
empty ladder yields exactly what it always did.

### `constants.ts`

```typescript
export const CACHE_HIT_THRESHOLD_MS = 15;
```

The time threshold (ms) below which a LOD load is treated as a likely
cache hit. It is consumed by `streaming-policy.ts` (below) — the `refine`
and `playback` passes keep streaming while levels load faster than this and
stop at the first slower/cold one. Deliberately a single cross-geometry
threshold, not a per-geometry tuning knob.

### `pass-rollback.ts`

Computes the state change required after a progressive pass appends levels but
fails before its result is committed. `planLadderRollback` clamps the pass-start
watermark, reports how many levels must be discarded, and invalidates a
concatenation memo only when it covers one of those discarded levels.

### `streaming-policy.ts`

Pure decisions that drive each progressive loader's LOD streaming loop, so
the two loops (`AdditiveLadderCore` and Mesh's reveal ladder) stay identical by construction. A pass is classified from
three facts (is a per-frame budget active? is this a background prefetch? is a
ladder depth pinned?):

| Pass       | When                                                    | Behavior                                                                                                                                |
| ---------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `playback` | foreground, budgeted (playing)                          | Stream cache-resident levels within budget and stop at the first cold/slow one. Only an empty ladder gets an unconditional first level. |
| `prefetch` | background shadow pass                                  | **Deepen toward the full ladder** — keep one new level after a restore, never stop on a cache miss, then obey the pass budget + abort.  |
| `refine`   | foreground, unbudgeted (paused)                         | Stream cache-resident levels; stop at the first cold/slow one (the `CACHE_HIT_THRESHOLD_MS` rule).                                      |
| `pinned`   | Numeric `ViewState.ladderDepth`, or unbudgeted `'auto'` | **Load exactly the pinned rung count**, cold or not. Foreground or shadow; takes precedence over the other three.                       |

During budgeted playback, `'auto'` uses the first rung reaching
`autoEnergyThreshold` as a guaranteed floor, then follows `playback` or
`prefetch` rules toward the full ladder. The energy rung alone ignores the
deadline. The last foreground frame's depth can carry as a preferred cold
prefix, but it still obeys the deadline. A new cold rung is probed only when
the previous frame had enough time left for its average rung cost, preventing
steady playback from alternating between deep overruns and shallow frames.
Unstamped ladders remain time-budgeted.

When the refinement sweep supplies a residency allowance, it overrides all
four disciplines: the level that spends the allowance is kept, then the pass
stops before appending another cached level.

```typescript
const pinnedDepth = resolveLadderDepth(viewState.ladderDepth, nLods, energyTable, threshold);
const autoFloor =
  viewState.ladderDepth === 'auto' && budgetDeadline !== null ? (pinnedDepth ?? 0) : 0;
const preferredDepth = autoFloor; // A measured foreground carry may raise this.
const pass = classifyStreamingPass(
  budgetDeadline !== null,
  viewState.prefetch === true,
  pinnedDepth !== null && autoFloor === 0
);
for (let level = startLevel; level < (autoFloor > 0 ? nLods : (pinnedDepth ?? nLods)); level++) {
  if (
    shouldStopBeforeLevel(
      pass,
      level,
      startLevel,
      { nowMs: now(), deadlineMs: budgetDeadline },
      autoFloor
    )
  )
    break;
  // ... load level ...
  const additionalResidentBytes = Math.max(
    0,
    ladderResidentBytes(this.ladderResidency()) - residentBytesAtPassStart
  );
  if (
    shouldStopAfterLevel(
      pass,
      level,
      startLevel,
      allResident,
      elapsed,
      additionalResidentBytes,
      residencyAllowanceBytes,
      preferredDepth
    )
  )
    break;
}
```

This is what keeps timelapse playback responsive (coarse-but-fast first
loop) while the background prefetch fills the SliceCache toward full
ladders so later loops are higher-quality — still fast.

**Playback detail (`pinned`).** The time-budgeted discipline trades quality
for cadence: a frame shows whatever was resident within its tick, so a heavy
time-lapse (millions of splats per timepoint) flickers between coarse and fine
frames. Pinning a ladder depth (`DimensionAnimationManager.setLadderDepth`, the
**Detail** section of the dimension context menu, `play(dim, { ladderDepth })`, or
the scene's authored `viewer_config.playback_lod_depth`) flips the trade: every
frame is drawn at exactly that many rungs and the tick waits for the data, so
quality is constant and the frame rate adapts. The default, `'auto'`, guarantees
each ladder the first rung at or above the configured energy threshold (0.9),
then spends spare time on deeper rungs. A fast committed depth carries to the
next tick; slow commits fall back to the energy floor. The shadow pass warms
deeper rungs in the t+1 SliceCache entry. Ladders without stamps keep the
time-budgeted behaviour.
Scrubbing (a slider drag, a keyboard step) is pinned the same way, and an
unpinned settle pass follows once the scrub is quiet so a paused view still
refines to its full ladder. The pinned
depth rides the same per-pass directive path as `frameBudgetMs`
(`updateAllNDNodes` → `updateSceneForDimensions` → handler ctx → derived view
state) and is forwarded to the t+1 shadow prefetch, so the next frame's S-cache
entry already carries the pinned prefix. A pinned loader reports
`hasMoreLODs === false` (the pinned prefix IS the target) exactly like a
budgeted one; the refine-on-pause pass, which carries neither directive,
resumes the ladder from the prefix.

**An empty level never ends the loop.** These loaders exist only for
_additive_ ladders, whose levels are disjoint increments of one permutation
(`additive_0` is a small subset — a few thousand elements under
`-b stream:C` / `--target-ms`), not coarse-to-fine resamplings of the same
elements. A hidden-dimension slice that no LOD-0 element lands on says
nothing whatever about levels 1..n-1, which may hold plenty of geometry
right there. Each loader therefore streams its whole ladder, and a slice is
known to be empty only once every level has been looked at. (Breaking out on
an empty LOD 0 — which all three loaders used to do — left such a slice
permanently blank: issue #1456.) Nothing needs to latch that verdict: a fully
streamed ladder already reports `hasMoreLODs === false` on its level count, and
the prefetch and cache store both self-guard on the same count.

## Consumers

| File                                          | Uses                                                           |
| --------------------------------------------- | -------------------------------------------------------------- |
| `../../points/points-progressive-loader.ts`   | `concatRequiredField`, `concatOptionalField`, streaming-policy |
| `../../lines/lines-progressive-loader.ts`     | `concatRequiredField`, `concatOptionalField`, streaming-policy |
| `../../gsplats/gsplats-progressive-loader.ts` | `concatRequiredField`, streaming-policy                        |
| `streaming-policy.ts`                         | `CACHE_HIT_THRESHOLD_MS`                                       |

## See Also

- [`../README.md`](../README.md) — unified loader infrastructure.
- [`../../README.md`](../../README.md) — the Luxar Data package overview.
