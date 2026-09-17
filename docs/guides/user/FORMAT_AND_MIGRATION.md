# Formats & Migration

Luxar has **two distinct on-disk formats** with **independent version numbers**.
This page distinguishes them, documents the versioning policy, and explains how to
migrate legacy inputs. For the exhaustive field-level specs, see
[Luxar zarr format](./LUXAR_ZARR_FORMAT.md) (scenes) and
[GSplats zarr format](../../specs/GSPLATS_ZARR_FORMAT.md) (standalone splats).

## Two on-disk formats

| Format | Extension | What it holds | Version attr | Current version |
|---|---|---|---|---|
| Scene container | `.luxar.zarr` | The scene graph (points / lines / gsplats / mesh nodes, groups, transforms, dimensions) | `format_version` (with `format_type="luxar_zarr"`; 0.1 stores carry the legacy `luxar_version` instead) | **v0.2** |
| Standalone gsplats | `.gsplats.zarr` | A detached Gaussian-splat node-tree (leaf, additive ladder, `kind=lod`, `kind=partition`, nested) | `format_version` (with `format_type="gsplats_zarr"`) | **v3.4** |

The two version numbers are **unrelated** — the scene container being at v0.2 says
nothing about the gsplats node-tree being at v3.4, and vice versa. A
`.gsplats.zarr` is structurally identical to the gsplats node a scene already
contains; it can be **grafted into a scene** with `luxar gsplat convert` or, from
Python, `scene.add_gsplats_from_file(...)`.

## Versioning policy

What readers do with a version they do not know, and how long deprecated names keep working, is the [Compatibility & Deprecation Policy](./COMPATIBILITY_POLICY.md).

Both version numbers and their supported ranges are single-sourced in
[`format-contract/contract.yaml`](../../../format-contract/contract.yaml). That
file is projected to Python (`typing_utils/_format_contract.py`) and TypeScript
(`luxar-viewer/src/types/format-contract.ts`) by `make gen-contract`, and
`hatch run check-contract` gates against drift, so the writer and the viewer always
agree on the vocabulary.

The contract declares what each build can read:

- **Scene (`.luxar.zarr`):** current **0.2**; **0.1** and **0.2** are readable.
- **Gsplats (`.gsplats.zarr`):** current **3.4**; **3.0–3.4** are all readable.

There are no reserved future slots: a newer version is handled by the policy
below, not by pre-declaring it.

### One version-check rule, both formats, both languages

Every reader — `LuxarScene.load` and the validator in Python, the gsplats
loaders (`load_gsplats`, `inspect_gsplats`, `luxar gsplat doctor`, the batch
tile validator), and the viewer's scene loader — applies the same rule, from
`luxar/typing_utils/format_version.py` and its mirror
`luxar-viewer/src/data/format-version.ts`. The two are pinned by one shared case
table (same ids in both test suites):

| Arm | Example on disk | Outcome |
|---|---|---|
| **supported** | scene `0.1` / `0.2`; gsplats `3.0` … `3.4` | Load silently. |
| **newer-minor** | scene `0.3`; gsplats `3.5` | Load, but **warn** (Python `UserWarning`; viewer console warning + toast). A minor bump is additive by policy, so the reader still makes sense of the store — it just cannot see what the newer writer added. |
| **older-unsupported** | scene `0.0`; gsplats `2.0` | **Refuse.** The error names the version, the supported set and the remedy: rebuild a scene with the current release; `luxar gsplat migrate-format <in> <out>` for a gsplats store. |
| **newer-major** | `9.9` | **Refuse** — upgrade Luxar / the viewer. |
| **unparsable** | `abc`, `0.2.1` | **Refuse** — a version must be `MAJOR.MINOR`. |
| **missing, with `format_type`** | `{format_type: "luxar_zarr"}` and no version | **Refuse** — a 0.2+ header with its version stripped is corrupt, not legacy. |
| **missing, without `format_type`** | `{type: "scene"}` only | Tolerated — a hand-written / pre-header store. |
| **legacy key** | `{luxar_version: "0.1"}` | Supported — read exactly as `format_version: "0.1"`. |

In Python a refusal is `luxar.typing_utils.format_version.UnsupportedFormatVersionError`
(a `ValueError`), or a `ValidationError` from `validate_zarr_attributes`; in the
viewer the loader throws and the error overlay names the version and the fix.

### Scene 0.1 → 0.2

0.2 changed only the **root header**: `format_version` + `format_type:
"luxar_zarr"` replace `luxar_version`, and `luxar_software_version` records the
writing release (provenance only — **excluded from `content_hash`** by both the
compile-time hasher and `luxar optimize`'s streaming twin, so a restamp under a
newer release never invalidates a viewer cache). Node schema, spatial index and
encodings are identical. Nothing needs migrating: every 0.1 store, including the
published Zenodo records and the `datasets/**` stores, loads through the
`legacy key` arm above. Recompiling a scene at 0.2 does change its
`content_hash` (the header keys fold into the digest) — expected, one-time.

Two stores, two keys, one fact: a standalone `.gsplats.zarr` still records the
writing release as `luxar_gsplats_version`. That asymmetry is deliberate —
renaming a 3.4 header key without bumping the gsplats format would make "3.4"
mean two shapes, and every published gsplats record carries the old key. It is
scheduled to unify on `luxar_software_version` at the next gsplats bump.

## Supported legacy inputs & migration

`luxar gsplat migrate-format` upgrades an older `.gsplats.zarr` layout to the
current v3.4 node-tree. It **auto-detects** the input shape:

| Legacy layout | How it is recognized | Migrates to |
|---|---|---|
| v1.0 (flat single-LOD) | root `format_version` attr = `"1.0"` | v3.4 leaf |
| v1.1 (additive ladder) | root `format_version` attr = `"1.1"` | v3.4 leaf + additive ladder |
| pre-v2.0 substitutive directory | a directory whose `manifest.json` has `lod_kind: "substitutive"` | v3.4 `kind=lod` group |
| v2.0 (matrix) | root `format_version` attr = `"2.0"` | v3.4 tree |
| v3.0 / v3.1 with legacy LOD attrs | a current node tree whose `kind=lod` groups still carry the pre-v3.2 `pixel_size` selector attrs | v3.4 (`selector: "screen-area"` + derived per-child `coverage_fraction`) |

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
| Output | Current v3.4 layout | Same structure, re-quantized |
| Changes | Layout **and** encoding (AUTO; `--lossless` for float32) | The on-disk Cholesky encoding, **and** under `auto`/`memory` the centers too (per-axis uint16 fixed-point — on ordinary spatial data centers are bit-exact only under `precision`; a *gridded* axis such as a stacked `sigma=0` time axis keeps uint16 but has its grid snapped onto the data's own spacing, and a *non-gridded* axis whose grid would displace splats past their own σ for **more than 0.1% of the splats** falls back to float32 — so a gridded axis is exact in every mode, while a smaller degenerate population on a non-gridded axis is quantized away with no warning) |
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
