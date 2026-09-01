# Progressive Loader Shared Helpers

> Cross-geometry building blocks for the additive-LOD progressive loaders.

## Overview

The three progressive (additive-LOD) loaders —
`../../points/points-progressive-loader.ts`,
`../../lines/lines-progressive-loader.ts`, and
`../../gsplats/gsplats-progressive-loader.ts` — each load a sequence of LOD
levels and concatenate the per-LOD typed-array fields into a single
merged payload. The bits that were identical across all three live here
so the geometry loaders only carry their own geometry-specific wrinkles
(segment-index offsetting, colour fill-with-white, Cholesky factor
sizing).

This folder holds no loader of its own — it is a pure helper module
imported by the geometry loaders two directories up (in
`src/data/points/`, `src/data/lines/`, and `src/data/gsplats/`).
`src/data/mesh/mesh-progressive-loader.ts` — the reveal ladder — shares
`concat-helpers.ts` and `streaming-policy.ts` too; only the SliceCache
helpers stay points/lines/gsplats-only, since a mesh is whole-node
resident and has no per-slice payload to cache.

## File Structure

```
progressive/
├── concat-helpers.ts      # Generic typed-array field concatenation across LOD parts
├── constants.ts           # CACHE_HIT_THRESHOLD_MS — the shared streaming threshold
├── pass-rollback.ts       # Failed-pass ladder truncation and concat-cache retention plan
├── streaming-policy.ts    # Per-pass LOD streaming decisions (playback / prefetch / refine)
├── slice-cache-helper.ts  # Shared SliceCache key/clone/restore/store logic, including separate
│                          # logical ladder depth for folded Lines payloads; also used by plain
│                          # spatial-index loaders caching a decoded slice as a 1-element ladder
└── view-state-equal.ts    # viewStatesEqual — the memoized-noop / generation-reset linchpin
                           # (formerly three byte-identical copies, one per geometry loader)
```

## Components

### `concat-helpers.ts`

Structurally-generic concatenation over any typed array `A`
(Float32Array, Uint8/16/32Array, Float16Array, …). The output dtype is
preserved by constructing from the first part's array.

| Symbol                                               | Description                                                                                                                                                                       |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `interface ConcatTypedArray`                         | Minimal structural shape (`length`, `set(array, offset?)`) common to every typed array being concatenated.                                                                        |
| `concatRequiredField(parts, get, countOf, perItem?)` | Concatenate a **required** field. Allocates `sum(countOf) * perItem` elements and copies each part at a running offset.                                                           |
| `concatOptionalField(parts, get, countOf, perItem?)` | Concatenate an **optional** field with **all-or-nothing** policy: returns `undefined` unless _every_ part carries the field — except that **zero-row parts abstain** (see below). |

`get` extracts the field from a part, `countOf` returns a part's element
count (rows), and `perItem` is the components per element (e.g. `3` for
positions, `1` for scalar widths/radii). `parts` is assumed to have
length ≥ 1.

```typescript
import { concatRequiredField, concatOptionalField } from '../loaders/progressive/concat-helpers';

// Required: positions are ndim components per row.
const positions = concatRequiredField(parts, (p) => p.positions, count, ndim);

// Optional: colours only survive if every LOD part has them.
const colors = concatOptionalField(parts, (p) => p.colors as ColorArray, count, 3);
```

**Zero-row parts abstain.** A part with `countOf(p) === 0` contributes no rows,
so it gets no vote on which optional attributes the merged result carries: it is
excluded from the presence gate, the copy, and the dtype comparison. This is
load-bearing because the canonical empty payloads **omit** their optional fields
rather than emitting zero-length arrays — so without the rule, one slice-culled
level of an additive ladder stripped those fields from every _other_ level too
and the node adapters substituted constant fills (issue #1456). What each
geometry stood to lose differs:

| Geometry | Fields routed through `concatOptionalField` | Lost without the rule                                            |
| -------- | ------------------------------------------- | ---------------------------------------------------------------- |
| Points   | `colors`, `radii`, `sharpness`, `scalars`   | all four → white, radius 0.5, sharpness 0.5, colormap suppressed |
| Lines    | `scalars` only                              | colormap suppressed (`hasScalars` clears)                        |
| GSplats  | none                                        | nothing — unaffected                                             |

Lines has no `radii` field at all, and its `colors` / `sharpness` are
present-but-`null` on the empty payload and merge via the bespoke find-first +
fill-for-missing path in `concatenateLinesData`, not this helper. GSplats'
empty payload carries zero-length `Float32Array`s for its required fields and
takes the same fill-for-missing path for colours.

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
the three loops stay identical by construction. A pass is classified from
two facts (is a per-frame budget active? is this a background prefetch?):

| Pass       | When                            | Behavior                                                                                                                                |
| ---------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `playback` | foreground, budgeted (playing)  | Stream cache-resident levels within budget and stop at the first cold/slow one. Only an empty ladder gets an unconditional first level. |
| `prefetch` | background shadow pass          | **Deepen toward the full ladder** — keep one new level after a restore, never stop on a cache miss, then obey the pass budget + abort.  |
| `refine`   | foreground, unbudgeted (paused) | Stream cache-resident levels; stop at the first cold/slow one (the `CACHE_HIT_THRESHOLD_MS` rule).                                      |

```typescript
const pass = classifyStreamingPass(budgetDeadline !== null, viewState.prefetch === true);
for (let level = startLevel; level < nLods; level++) {
  if (shouldStopBeforeLevel(pass, level, startLevel, now(), budgetDeadline)) break;
  // ... load level ...
  if (shouldStopAfterLevel(pass, level, startLevel, allResident, elapsed)) break; // playback/refine cold-or-slow stop
}
```

This is what keeps timelapse playback responsive (coarse-but-fast first
loop) while the background prefetch fills the SliceCache toward full
ladders so later loops are higher-quality — still fast.

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
