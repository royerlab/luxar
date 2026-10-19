# Authoring a BUNDLED demo: data shipping and the cinematic camera gate

Loaded from the `luxar-visualization` skill. Only relevant when adding or editing a
demo under `packages/luxar/src/luxar/demos/` in a source checkout; a user building
their own scene can skip it.

## Shipping a demo's data

`ensure_dataset("<key>")` — also from the `luxar.demos` barrel — resolves a demo's
payload cache → in-repo (git-LFS) → Zenodo, verifying the manifest sha256 at every
step, and returns the paths. The key comes from
`packages/luxar/src/luxar/demos/data_manifest.json`. It only resolves `zenodo`
keys: a `local-compute` / `regenerate` one raises `LocalComputeDataset` instead,
so those demos have to build their data themselves.

**That manifest is GENERATED — never hand-edit it.** Declare the dataset in
`scripts/gen_data_manifest.py` (bucket, record, license, source, attribution) and
run `make gen-data-manifest`; in-repo files are measured from the tree, while a
`zenodo` entry's `sha256` and byte count pin the published record. Legacy
`hosted_sha256` metadata is carried forward when present because it cannot be
derived from the tree.
A hand-edited metadata field always fails
`test_committed_manifest_matches_generator`; a hand-edited checksum does not when
the payload directory is absent or empty in your checkout, since the generator
then reuses the committed file list verbatim rather than wiping it. Re-serializing
it yourself with the wrong `indent` rewrites all ~600 lines into diff noise (the
generator uses `indent=2`). To replace a `zenodo` payload, upload the new bytes to
the record first, update its `sha256`/`bytes`, append the outgoing digest to
`superseded_sha256`, then run `make gen-data-manifest` and inspect the focused
diff. Copying a replacement over an LFS-tracked file applies only to the retained
Dip-C payload; regenerate afterwards and verify its `sha256`/`bytes` change.

## The cinematic camera in a bundled demo

**In a bundled demo this exact call fails required tests.** Demos run under the
cinematic 35 mm preset, and `test_demos_cinematic_mode.py` refuses an authored
pose that pins `fov`/`fov_preset` or that is not visibly composed for the 63°
lens. It also requires literal `cinematic_mode=True` on every `ViewerConfig`
built under `luxar/demos/`. Leave the FOV unset, derive the distance at
`CINEMATIC_FOV_DEG` directly, and show the full wrapper:

```python
import math

from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG

half_fov = math.radians(CINEMATIC_FOV_DEG) / 2
d = (
    max(
        (W / fill) / (2 * math.tan(half_fov) * aspect),
        (H / fill) / (2 * math.tan(half_fov)),
    )
    + depth / 2
)
camera = CameraConfig(position=(cx, cy, cz + d), target=(cx, cy, cz))
scene = compiler.create_scene(
    dimensions=dims,
    viewer_config=ViewerConfig(
        cinematic_mode=True, tone_mapping="ACES", camera=camera
    ),
)
```

The guard also recognises an inline `position=pull_in(...)` when carrying over an
empirically tuned distance from another FOV. For a derived distance, importing
`CINEMATIC_FOV_DEG` or `framing_scale` vouches for the module and lets the pose use
a local variable normally.
