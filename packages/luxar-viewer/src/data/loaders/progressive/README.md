# Progressive Loader Shared Helpers

> Cross-geometry building blocks for the additive-LOD progressive loaders.

## Overview

The three progressive (additive-LOD) loaders —
`points/points-progressive-loader.ts`,
`lines/lines-progressive-loader.ts`, and
`gsplats/gsplats-progressive-loader.ts` — each load a sequence of LOD
levels and concatenate the per-LOD typed-array fields into a single
merged payload. The bits that were identical across all three live here
so the geometry loaders only carry their own geometry-specific wrinkles
(segment-index offsetting, colour fill-with-white, Cholesky factor
sizing).

This folder holds no loader of its own — it is a pure helper module
imported by the geometry loaders one directory up.

## File Structure

```
progressive/
├── concat-helpers.ts      # Generic typed-array field concatenation across LOD parts
├── concat-arena.ts        # Growable ARENA fields + append-span registry (amortized ladder concat)
├── constants.ts           # CACHE_HIT_THRESHOLD_MS — the shared streaming threshold
├── streaming-policy.ts    # Per-pass LOD streaming decisions (playback / prefetch / refine)
├── slice-cache-helper.ts  # Shared SliceCache key/clone/restore/store logic (3-loader symmetry;
│                          # also used by the plain spatial-index loaders — a plain leaf caches
│                          # its decoded slice as a 1-element ladder under the same key contract)
└── view-state-equal.ts    # viewStatesEqual — the memoized-noop / generation-reset linchpin
                           # (formerly three byte-identical copies, one per geometry loader)
```

## Components

### `concat-helpers.ts`

Structurally-generic concatenation over any typed array `A`
(Float32Array, Uint8/16/32Array, Float16Array, …). The output dtype is
preserved by constructing from the first part's array.

| Symbol                                               | Description                                                                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `interface ConcatTypedArray`                         | Minimal structural shape (`length`, `set(array, offset?)`) common to every typed array being concatenated.                   |
| `concatRequiredField(parts, get, countOf, perItem?)` | Concatenate a **required** field. Allocates `sum(countOf) * perItem` elements and copies each part at a running offset.      |
| `concatOptionalField(parts, get, countOf, perItem?)` | Concatenate an **optional** field with **all-or-nothing** policy: returns `undefined` unless _every_ part carries the field. |

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

### `concat-arena.ts`

The **amortized** counterpart of `concat-helpers.ts`. The reference
concat rebuilds the whole ladder every time a level lands — appending
level _k_ re-copies levels `0..k`, i.e. O(k·N) across a ladder, measured
as a frame-stall source. The arena keeps ONE capacity-managed buffer per
field per loader **reset generation** and appends only the new level's
bytes: O(N_total) for the whole ladder.

| Symbol                                                          | Description                                                                                                                                                                                                                     |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `class ArenaField<A>`                                           | One growable typed-array field. `append` / `appendWith` / `appendFill`, `ensureCapacity(total, exact?)`, `trimToFit()`, `view()`. Tracks length in ELEMENTS (rows of `perItem` entries) and carries `stats` copy-work counters. |
| `class OptionalLadderField<A>`                                  | An **all-or-nothing** optional attribute (the `concatOptionalField` twin). `plan()` validates (throws), `apply()` mutates — so an arena can validate a whole batch before writing anything.                                     |
| `validateLadderFieldDtype(parts, from, get, ctor, label)`       | Ladder-dtype fail-fast with `concatRequiredField`'s exact message.                                                                                                                                                              |
| `setAppendSpan` / `getAppendSpan`                               | Identity-keyed `WeakMap` (same pattern as `types/prefix-lineage.ts`) recording the `AppendSpan` each concat result adds relative to its prefix parent.                                                                          |
| `interface AppendSpan`                                          | `{ fromElement, elementCount }` (+ lines-only `fromVertex` / `vertexCount`). Element = the geometry's commit unit: splats / points / segments.                                                                                  |
| `ARENA_GROWTH_FACTOR` (1.5), `ARENA_TRIM_SLACK_FRACTION` (0.02) | Capacity policy constants.                                                                                                                                                                                                      |

**Capacity policy** — per-level sizes are spatial-query results, unknown
before each level loads (no manifest total to pre-size from), so: exact
size on first allocation; `max(needed, ceil(capacity × 1.5))` on growth;
**exact** growth for the ladder's final level; and a trim-to-exact once
the ladder completes when slack exceeds 2%. Steady-state memory therefore
equals the old exact-size concat (0% overhead), with ≤ 50% transient
slack mid-ladder.

**Aliasing contract (view-with-copy-on-grow)** — snapshots are
`subarray(0, used)` VIEWS, not per-snapshot copies (a copy per level
would reinstate the very cost this removes). Safe because appends only
write past every handed-out view's length, growth re-allocates (leaving
old buffers and their views intact), and nothing downstream mutates or
**transfers** a loader-result buffer — the worker projection
structured-clones its inputs, and `noteDepthSortCommit`, which _does_
transfer, is only ever handed freshly-allocated arrays. See the module
doc for the full argument.

The three per-geometry arenas (`GSplatsLadderArena`, `PointsLadderArena`,
`LinesLadderArena`) live beside their loaders' reference `concatenate*Data`
functions and are pinned byte-for-byte against them by
`tests/unit/data/loaders/progressive/concat-arena-equivalence.test.ts`.

### `constants.ts`

```typescript
export const CACHE_HIT_THRESHOLD_MS = 15;
```

The time threshold (ms) below which a LOD load is treated as a likely
cache hit. It is consumed by `streaming-policy.ts` (below) — the `refine`
pass keeps streaming while levels load faster than this and stops at the
first slower/cold one. Deliberately a single cross-geometry threshold, not
a per-geometry tuning knob.

### `streaming-policy.ts`

Pure decisions that drive each progressive loader's LOD streaming loop, so
the three loops stay identical by construction. A pass is classified from
two facts (is a per-frame budget active? is this a background prefetch?):

| Pass       | When                            | Behavior                                                                                                         |
| ---------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `playback` | foreground, budgeted (playing)  | Commit the restored cached prefix + a **LOD-0 first-paint floor**; never block on fine levels — stay responsive. |
| `prefetch` | background shadow pass          | **Deepen toward the full ladder** — never stop on a cache miss; bounded by the pass budget + abort.              |
| `refine`   | foreground, unbudgeted (paused) | Stream cache-resident levels; stop at the first cold/slow one (the `CACHE_HIT_THRESHOLD_MS` rule).               |

```typescript
const pass = classifyStreamingPass(budgetDeadline !== null, viewState.prefetch === true);
for (let level = startLevel; level < nLods; level++) {
  if (!shouldLoadLevel(pass, level, startLevel)) break; // playback floor
  // ... load level ...
  if (shouldStopAfterLevel(pass, level, startLevel, allResident, elapsed)) break; // refine stop
}
```

This is what keeps timelapse playback responsive (coarse-but-fast first
loop) while the background prefetch fills the SliceCache toward full
ladders so later loops are higher-quality — still fast.

## Consumers

| File                                       | Uses                                                                         |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| `../points/points-progressive-loader.ts`   | `concatRequiredField`, `concatOptionalField`, concat-arena, streaming-policy |
| `../lines/lines-progressive-loader.ts`     | `concatRequiredField`, `concatOptionalField`, concat-arena, streaming-policy |
| `../gsplats/gsplats-progressive-loader.ts` | `concatRequiredField`, concat-arena, streaming-policy                        |
| `streaming-policy.ts`                      | `CACHE_HIT_THRESHOLD_MS`                                                     |

## See Also

- [`../README.md`](../README.md) — unified loader infrastructure.
- [`../../README.md`](../../README.md) — the Luxar Data package overview.
