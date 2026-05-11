# Zarr Facade

This package is Luxar's boundary around the current Zarr reader backend.
Production viewer code should import Zarr primitives from `data/zarr` rather than
importing `zarrita` or `@zarrita/storage` directly.

## Purpose

The facade keeps Luxar code dependent on a small Luxar-owned API:

- create/open stores with consolidated metadata support
- resolve dataset-relative locations
- open groups and arrays
- read full arrays or selected slices
- create slice selections
- expose the codec registry needed during bootstrap

The implementation currently delegates to `zarrita`, but backend API changes,
local workarounds, or a future reader replacement should be isolated here.

## Example

```ts
import * as zarr from '../data/zarr';

const rawStore = zarr.createFetchStore('https://example.org/scene.zarr');
const store = await zarr.openStore(rawStore);
const root = zarr.root(store);
const group = await zarr.openGroup(root);
const positions = await zarr.openArray(root.resolve('points/positions'));
const chunk = await zarr.readArray(positions, [zarr.slice(0, 100), zarr.slice(null)]);
```

## Backend policy

Luxar is pre-release, so this facade targets the current backend API rather than
preserving historical zarrita versions. If a backend quirk or patch is needed,
add it here and keep callers on the facade API.
