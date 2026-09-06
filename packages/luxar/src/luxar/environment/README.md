# `luxar.environment` — baked scene environments

The scene environment is what lights `material="physical"` meshes
(`docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3). A live viewer can
capture it from the scene itself (`viewer_config.environment.source = "scene"`:
a `CubeCamera` renders the scene from a probe point, and metals and glass reflect
the data they sit in). This package moves that capture off the viewer for a
published scene: `luxar env bake` drives the viewer headlessly, and
`luxar env attach` stores the six captured cube faces in the store as a
root-level `environment/` sidecar group the viewer prefilters at load in
milliseconds.

## Quick start

```bash
luxar env bake scene.luxar.zarr                      # capture at the scene centre, 128 px faces
luxar env bake scene.luxar.zarr --probe node:clusters/shell_3 --resolution 256
luxar env attach scene.luxar.zarr scene.env.bin      # the manual half of the same loop
```

From Python:

```python
from luxar.environment import attach_environment, pack, unpack

report = attach_environment("scene.luxar.zarr", "scene.env.bin")
print(report.status, report.array_name)   # attached faces-3f9a1c02

header, faces = unpack(open("scene.env.bin", "rb").read())
faces.view("float16")                     # (6, H, W, 4) radiance, px nx py ny pz nz
```

## The store contract

- **Group `environment/` at the store root, with NO `type` and NO `kind` attr.**
  The viewer's node discovery skips exactly such groups as metadata sidecars,
  and every Python walker consults `RESERVED_ROOT_GROUPS`
  (`luxar.typing_utils.constants`), so the group never shows up as a node. The
  compiler refuses a user node under that name.
- **Excluded from the scene `content_hash`.** The map is derived from the scene
  and records the digest it was baked against (`scene_content_hash`). That guard
  is only exact if attaching the map leaves the root digest alone, so both
  hashing walks (compile-time and `luxar optimise`'s streaming twin) stamp the
  group's own `content_hash` but do not fold it into the root's. A bake never
  invalidates a visitor's warm cache.
- **Array `environment/faces-<xxh64[:8]>`**, shape `(6, H, W, 4)`, dtype
  `uint16` holding IEEE half-float bits (`sample_format: "half-float-bits"`),
  one chunk per face, zstd-9 without byte shuffle. The digest suffix makes a
  re-bake a NEW path, so a caching viewer cannot serve stale faces; the group
  attr `faces` names the live array and stale siblings are removed. Half bits
  as `uint16` because the GPU readback yields them, three's `HalfFloatType` cube
  texture consumes them, and a zarr `uint16` array needs no `Float16Array` on
  the reading engine.
- **Attrs** carry the container header (`format`, `face_order`,
  `coordinate_system`, `probe`, `resolution`, `scene_content_hash`,
  `appearance`, `baked_at`, `viewer_version`) plus `faces`, `sample_format`,
  `shape` and the group's own `content_hash`.
- **Idempotent.** Attaching the same faces twice writes nothing; a different
  bake replaces the array and reports what it removed. A map whose
  `scene_content_hash` is not the store's current digest is refused unless
  `--force` (the viewer would ignore it as stale anyway).

## Modules

| Module | What it is |
| --- | --- |
| `container.py` | The blob the viewer hands back: `LXENV001` magic, u32 header length, JSON header, raw `uint16` faces. `pack` / `unpack` are the single definition, mirrored by `packages/luxar-viewer/src/rendering/environment/bake.ts`. |
| `attach.py` | `attach_environment(store, faces, force=False) -> AttachReport`: validation, the digest-named array, the hash guard, stale-sibling removal, consolidation. |
| `bake.py` | `bake_environment(...)`: serves the store and the built viewer, drives `packages/luxar-viewer/scripts/bake-env.mjs` (Node Playwright), then attaches. |

## Testing

`env/tests/test_attach.py` pins the container round-trip and refusals, the
digest-named array, the unchanged scene digest under BOTH hashing walks,
idempotency and replacement, the stale-bake refusal and `--force`, and that the
group is invisible to `LuxarScene.nodes`, `luxar info` and refused as a node name
by the compiler; `luxar optimise` copies it verbatim.
