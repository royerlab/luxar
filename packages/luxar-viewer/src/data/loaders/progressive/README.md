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
├── concat-helpers.ts   # Generic typed-array field concatenation across LOD parts
└── constants.ts        # CACHE_HIT_THRESHOLD_MS — the shared streaming threshold
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

### `constants.ts`

```typescript
export const CACHE_HIT_THRESHOLD_MS = 15;
```

The time threshold (ms) below which a LOD load is treated as a likely
cache hit. Each progressive loader keeps streaming to the next LOD while
levels load faster than this; once a level takes longer (or not every
level is cache-resident), it stops and lets the refinement loop pick up
the rest:

```typescript
if (level > startLevel && (!allResident || elapsed > CACHE_HIT_THRESHOLD_MS)) {
  // stop streaming; refinement loop handles remaining LODs
}
```

It is deliberately a single cross-geometry threshold, not a per-geometry
tuning knob — Points, Lines, and GSplats all import the same constant.

## Consumers

| File                                       | Uses                                                                   |
| ------------------------------------------ | ---------------------------------------------------------------------- |
| `../points/points-progressive-loader.ts`   | `concatRequiredField`, `concatOptionalField`, `CACHE_HIT_THRESHOLD_MS` |
| `../lines/lines-progressive-loader.ts`     | `concatRequiredField`, `concatOptionalField`, `CACHE_HIT_THRESHOLD_MS` |
| `../gsplats/gsplats-progressive-loader.ts` | `concatRequiredField`, `CACHE_HIT_THRESHOLD_MS`                        |

## See Also

- [`../README.md`](../README.md) — unified loader infrastructure.
- [`../../README.md`](../../README.md) — the Luxar Data package overview.
