# Test Data Builders

> Fluent factory helpers for constructing test fixtures — geometry
> data, dimension configurations, Three.js scenes, and mock zarr
> arrays — without scattering boilerplate across the unit tests.

The builders here back the recommended fixture pattern from
[`../README.md`](../README.md#using-test-builders): a chained
`withX().withY().build()` API that returns plain typed arrays plus
counts. Tests that need a thousand 4D points with colours and radii,
or an isotropic-Gaussian gsplat batch, or a stub `zarr.Location` for
the spatial-index loader, build them here instead of hand-rolling
`Float32Array` arithmetic in every spec.

## File Structure

```
builders/
├── test-data-builders.ts       # Fluent builders for Points / Lines / GSplats /
│                               #   Dimensions / Scene / Chunk / MockZarrArray.
└── spatial-loader-fixtures.ts  # Mock zarr.Location, chunk-bounds arrays, and
                                #   default visible-range fixtures for the
                                #   spatial-index loader tests.
```

## Builders

All builders follow the same shape: zero-argument constructor, chained
`withX()` setters that return `this`, and a final `build()` that
returns either a plain object of typed arrays + counts, a typed
config, or a constructed Three.js object. Required attributes are
auto-populated with sensible defaults when `build()` runs without an
explicit setter call.

The three geometry builders (`PointsBuilder`, `LinesBuilder`,
`GSplatsBuilder`) are intentionally parallel — same setter names,
same parameter conventions, same default behaviour — per the
project's three-geometry symmetry rule.

### `PointsBuilder`

Builds points fixtures: `numPoints * dimensions` flat positions plus
optional `colors`, `radii`, and `sharpness`. `build()` returns the
typed arrays alongside `numPoints` and `dimensions`. `colors` are
RGB floats (3 components per point); `radii` and `sharpness` are
per-point scalars.

```typescript
import { PointsBuilder } from '../builders/test-data-builders';

const points = new PointsBuilder()
  .withPoints(1000)
  .withDimensions(4)
  .withRandomPositions([-10, 10])
  .withColors()
  .withRadii(0.5)
  .build();
```

### `LinesBuilder`

Builds line-segment fixtures. Vertices are stored flat as
`(numSegments * 2) * dimensions` floats — two endpoints per segment.
`widths` is required (defaults to uniform 1.0 if unset). `segments`
defaults to the canonical packing `[0, 1, 2, …, 2N-1]` — every pair
of consecutive vertex indices forms one segment. `colors` are
per-vertex RGB; `sharpness` is per-vertex.

### `GSplatsBuilder`

Builds Gaussian-splat fixtures with required `centers`,
`amplitudes`, and `choleskyFactors`. The Cholesky factor is packed
lower-triangular, `ndim * (ndim + 1) / 2` floats per splat —
matching the rest of the viewer (for `ndim=3`:
`[L00, L10, L11, L20, L21, L22]`).
`withIsotropicCovariance(sigma)` is the most common path: it writes
`sigma` on the diagonals and zero elsewhere. `colors` are per-splat
RGB.

### `DimensionsBuilder`

Builds a `SimpleDims` object: `ndim`, `displayed` (max three
indices), `currentStep` (slice position in non-displayed dims), and
`metadata` (per-dimension name / unit / range / display flag /
discrete / step). Convenience setters:

- `withSpatialDimensions(unit, range)` — populates x / y / z.
- `withTimeDimension(range, step)` — adds a continuous time
  dimension at index 3 (expanding `ndim` to 4 if needed).
- `withChannelDimension(numChannels)` — adds a discrete channel
  dimension at index 4 (expanding `ndim` to 5 if needed).

### `SceneBuilder`

Builds a `THREE.Scene` for tests that need a real scene graph.
`withPoints(positions, { colors, dims })` adds a `THREE.Points`
node with the given attributes; `withGroup(name)` adds a top-level
group; `withNestedStructure(levels)` chains nested groups. The
underlying object array is exposed via `getObjects()` for direct
assertions.

### `ChunkBuilder`

Builds zarr chunk fixtures: `shape`, `chunkShape`, `dtype`, the
chunk `data` (random `Float32Array` or `Uint8Array` per dtype), and
the derived `chunkGrid` (per-axis chunk counts).

### `MockZarrArrayBuilder`

Builds a minimal zarr-array stub with `shape`, `chunks`, `dtype`,
arbitrary `metadata`, and a vitest-mocked `get()` that resolves to
an empty `Float32Array` of the right size. Useful when a test needs
something that looks like an opened zarr array without standing up
the full mock store.

## Spatial-Loader Fixtures

`spatial-loader-fixtures.ts` carries the shared bits between the
three spatial-index loader test files
(`unit/data/points/spatial-index-loader.test.ts`,
`unit/data/lines/spatial-index-loader.test.ts`,
`unit/data/gsplats/spatial-index-loader.test.ts`). The mock-symmetry rule
means each of these tests needs the same `zarr.Location` stub,
the same chunk-bounds array shape, and the same default visible
ranges from a mocked `SpatialQueryBuilder.execute()`.

Note: `vi.mock(...)` calls themselves still live in each test file,
because `vi.mock` paths resolve relative to the calling file. Only
the helper functions and constants are hoisted here.

| Export                                     | Purpose                                                                                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `makeMockZarrLocation(prefix)`             | Stub `zarr.Location` whose `resolve(path)` returns `"${prefix}://${path}"` so `zarr.open` mocks can match on path substring.                |
| `makeChunkBoundsArray(chunkCount, ndim)`   | Descriptor for the chunk-bounds zarr array — shape `[chunkCount, ndim, 2]` (one min/max pair per chunk per dimension), `float32`, no attrs. |
| `makeMockZarrArray(shape, dtype?, attrs?)` | Minimal `{ shape, dtype, attrs }` descriptor for the zarr arrays the loaders touch.                                                         |
| `makeChunkBoundsBuffer(chunkCount, ndim)`  | Zero-filled `Float32Array` of length `chunkCount * ndim * 2` for `zarr.get`'s `data` field — every chunk intersects every query.            |
| `DEFAULT_VISIBLE_RANGES`                   | `[{start: 0, end: 100}, {start: 200, end: 300}]` — two ranges so the loader's range-merge / accumulator logic gets exercised.               |

## Conventions

- **Required vs optional attributes** match the data model: Points
  require positions only; Lines require vertices and widths; GSplats
  require centers, amplitudes, and Cholesky factors. When a required
  attribute is omitted, `build()` fills it with a reasonable default
  (random data, uniform 1.0, isotropic σ=0.1).
- **Defaults are deterministic in shape, random in content.** Tests
  that need reproducibility should pass explicit data via
  `withPositions` / `withVertices` / `withCenters`.
- **Typed-array returns, not domain objects.** Builders return raw
  `Float32Array` / `Uint32Array` buffers plus counts — they don't
  construct `Points` / `Lines` / `GSplats` runtime objects. Tests
  that need the runtime objects build them from the buffers.
- **Three-geometry symmetry**: when adding a setter to one geometry
  builder, mirror it on the other two if the concept applies.

## Related

- [`../README.md`](../README.md) — overall test suite organisation and
  the "Using Test Builders" usage snippet.
- [`../unit/builders/builders.test.ts`](../unit/builders/builders.test.ts)
  — covering tests for the builders themselves.
- [`../mocks/`](../mocks/) — global mock infrastructure (WebGL,
  THREE.js, OPFS, browser APIs) that runs before tests under
  `setup.ts`. Builders here construct test data; mocks there
  substitute environment APIs.
