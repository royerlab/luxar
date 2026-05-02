# luxar.gsplats.batch - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-02

## Purpose

`luxar.gsplats.batch` plans, submits, monitors, and merges Slurm batch jobs for fitting Gaussian splats to large OME-Zarr or nD Zarr datasets. It converts a dataset into a deterministic grid of fitting tasks over timepoints, flat channel combinations, and spatial tiles.

---

## Core Concepts

### Batch Manifest

`BatchManifest` is the single source of truth for a batch run. It stores input paths, selected timepoints/channels, folded channel-axis metadata, spatial tiling, fitting parameters, Slurm resources, denoising configuration, submitted job IDs, and the expected output files.

### Flat Channel Indexing

Datasets may have more than one non-spatial, channel-like axis, such as `camera` and `channel`. Batch planning folds those axes into one flat channel task index using row-major order.

Example for `channel_shape=(2, 4)`:

```text
flat 0 -> (0, 0)
flat 1 -> (0, 1)
flat 4 -> (1, 0)
flat 7 -> (1, 3)
```

The manifest records both `channel_shape` and per-job `channel_coords` so generated scripts, status checks, and merge operations can agree on the intended dataset slice.

### Task Encoding

A flat task ID maps to `(timepoint_sequence, channel_sequence, tile_index)`:

```text
task_id = t_seq * (n_channels * n_tiles) + c_seq * n_tiles + tile_index
```

When timepoint/channel slicing is used, sequence indices map to real dataset indices through `timepoint_indices` and `channel_indices`.

---

## Data Structures

### BatchJob

```text
BatchJob:
  task_id: int
  timepoint: int        # real dataset timepoint index
  channel: int          # real flat channel index
  tile_index: int
  output_filename: str
  estimated_wall_seconds: float
  channel_coords: tuple[int, ...]
```

**Invariants**:
- `task_id` is unique within the manifest.
- `output_filename` uses real dataset indices, not sequence indices.
- `channel_coords == decode_flat_channel_index(channel, channel_shape)` when channel axes exist.

### BatchManifest

```text
BatchManifest:
  input_path: str
  output_dir: str
  array_key: str | None
  n_timepoints: int       # selected count
  n_channels: int         # selected flat-channel count
  channel_axes: list[str]
  channel_shape: tuple[int, ...]
  spatial_shape: tuple[int, ...]
  n_tiles: int
  total_tasks: int
  jobs: list[BatchJob]
```

**Invariants**:
- `total_tasks == n_timepoints * n_channels * n_tiles`.
- `len(jobs) == total_tasks` after planning.
- If `timepoint_indices` is present, it contains real dataset indices for selected timepoints.
- If `channel_indices` is present, it contains real flat channel indices for selected channel combinations.

---

## Algorithms

### Decode Task ID

**Purpose**: Convert a Slurm array task ID to sequence indices.

```text
n_c = manifest.n_channels
n_k = manifest.n_tiles
t_seq = task_id // (n_c * n_k)
r = task_id % (n_c * n_k)
c_seq = r // n_k
tile_index = r % n_k
```

**Complexity**: O(1) time and space.

### Slurm Fit Script Generation

**Purpose**: Generate an sbatch array script that runs one or more fitting tasks per Slurm job.

**Algorithm**:
1. Compute output filename widths from real selected indices.
2. Emit index mapping arrays for sliced timepoints/channels.
3. Decode each task ID into sequence indices.
4. Map sequence indices to real dataset indices.
5. Run `luxar gsplat fit` into `${OUTPUT}.tmp`.
6. Atomically rename `.tmp` to final output.
7. Skip existing outputs to support restart/requeue.

### 3-Level Merge

**Purpose**: Merge completed tile outputs into one final `.gsplats.zarr`.

**Algorithm**:
1. Merge tiles per `(timepoint, channel)`.
2. Stack timepoints per channel as a new dimension.
3. Merge channels, optionally applying channel colors.

---

## Validation Rules

- Timepoint and channel slices must select at least one index.
- Selected timepoint indices must be in `[0, n_timepoints_full)`.
- Selected flat channel indices must be in `[0, n_channels_full)`.
- Tile output names must be derived from real dataset indices.
- Existing final outputs are skipped; incomplete `.tmp` outputs are removed before rerun.

---

## Related Specifications

- `luxar.gsplats` - Gaussian splat data model and fitting overview (`../SPECIFICATIONS.md`)
- `luxar.gsplats.rendering` - volume rendering of splats (`../rendering/SPECIFICATIONS.md`)
- `luxar.gsplats.preprocessing` - NLM denoising and preprocessing pipeline (`../preprocessing/SPECIFICATIONS.md`)

---

## Changelog

- **v1.0.0** (2026-05-02): Initial specification.
