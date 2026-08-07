# Formats & Migration

Luxar has **two distinct on-disk formats** with **independent version numbers**.
This page distinguishes them, documents the versioning policy, and explains how to
migrate legacy inputs. For the exhaustive field-level specs, see
[Luxar zarr format](./LUXAR_ZARR_FORMAT.md) (scenes) and
[GSplats zarr format](../../specs/GSPLATS_ZARR_FORMAT.md) (standalone splats).

## Two on-disk formats

| Format | Extension | What it holds | Version attr | Current version |
|---|---|---|---|---|
| Scene container | `.luxar.zarr` | The scene graph (points / lines / gsplats / mesh nodes, groups, transforms, dimensions) | `luxar_version` | **v0.1** |
| Standalone gsplats | `.gsplats.zarr` | A detached Gaussian-splat node-tree (leaf, additive ladder, `kind=lod`, `kind=partition`, nested) | `format_version` (with `format_type="gsplats_zarr"`) | **v3.3** |

The two version numbers are **unrelated** — the scene container being at v0.1 says
nothing about the gsplats node-tree being at v3.3, and vice versa. A
`.gsplats.zarr` is structurally identical to the gsplats node a scene already
contains; it can be **grafted into a scene** with `luxar gsplat convert` or, from
Python, `scene.add_gsplats_from_file(...)`.

## Versioning policy

Both version numbers and their supported ranges are single-sourced in
[`format-contract/contract.yaml`](../../../format-contract/contract.yaml). That
file is projected to Python (`typing_utils/_format_contract.py`) and TypeScript
(`luxar-viewer/src/types/format-contract.ts`) by `make gen-contract`, and
`hatch run check-contract` gates against drift, so the writer and the viewer always
agree on the vocabulary.

The contract declares the supported/reserved ladders:

- **Scene (`.luxar.zarr`):** current **0.1**; **0.2** and **0.3** are declared in
  the supported-version set, reserved for future revisions.
- **Gsplats (`.gsplats.zarr`):** current **3.3**; **3.0–3.3** are all readable.

### Why the scene format is still v0.1

The scene container has **not needed a breaking change** since its introduction:
the spatial-index layout, transform convention, and node schema have been additive
only. The version number is intentionally conservative and is bumped only when a
change would break existing readers. The contract already reserves **0.2** and
**0.3** so a future revision has a declared, forward-supported slot rather than an
ad-hoc bump. In short, v0.1 is not a placeholder — it is an accurate statement that
no incompatible scene-format revision has been required yet.

**Reader behavior on a version mismatch:** the reader does not hard-fail; it
**warns** and continues (`io/reader.py`). The warning fires on *any* version other
than the current one — including the reserved 0.2/0.3 — so newer-than-expected data
stays loadable while the mismatch is surfaced in the console.

## Supported legacy inputs & migration

`luxar gsplat migrate-format` upgrades an older `.gsplats.zarr` layout to the
current v3.3 node-tree. It **auto-detects** the input shape:

| Legacy layout | How it is recognized | Migrates to |
|---|---|---|
| v1.0 (flat single-LOD) | root `format_version` attr = `"1.0"` | v3.3 leaf |
| v1.1 (additive ladder) | root `format_version` attr = `"1.1"` | v3.3 leaf + additive ladder |
| pre-v2.0 substitutive directory | a directory whose `manifest.json` has `lod_kind: "substitutive"` | v3.3 `kind=lod` group |
| v2.0 (matrix) | root `format_version` attr = `"2.0"` | v3.3 tree |
| v3.0 / v3.1 with legacy LOD attrs | a current node tree whose `kind=lod` groups still carry the pre-v3.2 `pixel_size` selector attrs | v3.3 (`selector: "coverage"` + derived `coverage_fraction`) |

```bash
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr             # single file (AUTO encoding)
luxar gsplat migrate-format old_pyr/ v3.gsplats.zarr                        # substitutive directory
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr --lossless  # preserve float32 Cholesky exactly
```

By default the output adopts the AUTO encoding policy (certified uint8→uint16
quantization of the Cholesky factors). Pass `--lossless` to keep the factors in
float32 for archival fidelity.

### migrate-format vs reencode

Both write a new copy and never mutate the input, but they solve different
problems:

| | `luxar gsplat migrate-format` | `luxar gsplat reencode` |
|---|---|---|
| Input | **Legacy** layout (v1.0 … v3.1) | **Current-format** dataset |
| Output | Current v3.3 layout | Same structure, re-quantized |
| Changes | Layout **and** encoding (AUTO; `--lossless` for float32) | **Only** the on-disk Cholesky encoding |
| Encoding choices | float32 vs AUTO (certified u8→u16 ladder) via `--lossless` | Full ladder: `-e memory` (uint8) / `precision` (float32) / `auto` |
| Structure | May restructure legacy tree | Structure-preserving (leaf/lod/partition/nested; the `fitting`/`provenance`/`pipeline` groups carry over for directory stores) |

```bash
luxar gsplat reencode fit.gsplats.zarr fit_u8.gsplats.zarr -e memory      # uint8 (smallest)
luxar gsplat reencode fit.gsplats.zarr fit_f32.gsplats.zarr -e precision  # float32 (exact/archival)
```

Decode is always to float32, so the viewer, GPU, and WASM paths are unaffected by
the on-disk encoding choice.

Both are distinct from the two **export** commands, which produce entirely
different artifacts:

- `luxar export` — a **scene** (`.luxar.zarr`) → a standalone offline viewer folder.
- `luxar gsplat export` — a `.gsplats.zarr` → a classical **INRIA PLY** file.

## LOD terminology map

The level-of-detail **algorithms** and their **on-disk semantics** keep their
original names in the API and data model. The CLI `--recipe` flag uses
**intent-first** names for the same concepts. Both vocabularies are current — the
API terms are **not** deprecated; they name the algorithm and the `kind=lod`
semantics, while the recipe names describe what you get.

| API / data-model term | CLI `--recipe` name | Meaning |
|---|---|---|
| additive | `stream` | One leaf + a progressive prefix-sum ladder (fast first paint) |
| substitutive / pyramid | `levels` | Coarse→fine replacement levels (zoom across scales) |
| partitioned | `tiles` | Spatial BSP tiles, culled and streamed per tile |
| multiscale | `overview` | Instant coarse overview level + fine tiles on zoom |
| mosaic | `adaptive` | Tiles where each tile picks its own detail level |

(The pre-2026-07 recipe names — additive, substitutive, pyramid, partitioned,
multiscale, mosaic — still error on the CLI with a pointer to the new name; stored
batch manifests translate silently.)

## See also

- [Luxar zarr format](./LUXAR_ZARR_FORMAT.md) — full scene-container spec
- [GSplats zarr format](../../specs/GSPLATS_ZARR_FORMAT.md) — full standalone-splats spec, including the complete version history
- [CLI reference](./CLI_REFERENCE.md) — every command grouped and linked
