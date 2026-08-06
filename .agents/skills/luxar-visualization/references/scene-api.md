# Luxar scene-authoring API — reference

Exact signatures (from `packages/luxar/src/luxar/core/`). `Scene` inherits from
`Group`, so every `add_*` method below is available on both a `Scene` and any
`Group` returned by `scene.add_group(...)`.

## Entry point

```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension, transforms

with LuxarZarrCompiler(output_path) as compiler:        # io/compiler.py
    scene = compiler.create_scene(dimensions=dims)      # -> Scene
    # ... add nodes ...
# context exit finalizes the .luxar.zarr store
```

Never instantiate `Scene` directly — always go through `LuxarZarrCompiler`. Data is
written progressively as you call `add_*`; the node objects returned are lightweight
metadata handles.

## Dimensions (REQUIRED for create_scene)

```python
@dataclass
class Dimension:
    name: str
    unit: str = ""
    range: Optional[Tuple[float, float]] = None
    step: Optional[float] = None
    display: bool = True       # include in the 3D viewport
    discrete: bool = False     # integer-only (time, frames)
    cyclic: bool = False
    scale: float = 1.0
    spatial: Optional[bool] = None
    categories: list | None = None   # for categorical axes
    description: str = ""

Dimensions([...])              # list of Dimension
```

Convenience constructors: `Dimensions.default_2d()`, `default_3d()`,
`default_timeseries(n_timepoints=100, time_unit="s")`,
`default_multichannel(n_channels=3)`, `from_positions(positions, names=None)`.

The first 2-3 `display=True` dims are the viewport axes; non-displayed dims (time,
channel, z-stack) become nD slider axes navigated with keys `1`-`9` and `[` / `]`.

## add_points

```python
scene.add_points(
    name, positions,             # positions: (N, D) float32
    colors=None,                 # (N,3) | (3,) | tuple; Uint8 or Float32 (HDR)
    radii=None,                  # (N,) | scalar
    sharpness=None,              # (N,) | scalar  (edge softness)
    scalars=None,                # (N,) -> colormap input
    labels=None, image_labels=None,
    parent=None,                 # a Group, for hierarchy
    extend_to_all=None,          # broadcast across listed non-displayed dims
    dim_order=None, fill=None,
    **attrs,                     # opacity, intensity, gamma, blending_mode,
)                                #   colormap, layer, visible, transform
```

## add_lines

```python
scene.add_lines(
    name, vertices,              # (N, D) float32
    widths,                      # (N,) | scalar  (REQUIRED)
    colors=None, sharpness=None, scalars=None,
    labels=None, image_labels=None,
    indices=None,                # for line_type="indexed"
    line_type="polyline",        # "segments" | "polyline" | "loop" | "indexed"
    parent=None, extend_to_all=None, dim_order=None, fill=None,
    **attrs,
)
```

## GSplats (4 ways)

```python
# raw arrays — cholesky_factors packed lower-triangular; 3D=(N,6) [L00,L10,L11,L20,L21,L22]
scene.add_gsplats(name, centers, amplitudes, cholesky_factors, colors=None, **attrs)

# from a fitted .gsplats.zarr file (auto-lowers multi-level LOD to a kind=lod group)
scene.add_gsplats_from_file(name, path, **attrs)

# from an in-memory GSplatData (output of fit_gaussian_splats / make_additive_lod)
scene.add_gsplats_from_data(name, result, **attrs)

# fit in one step from a volume
scene.add_gsplats_from_volume(name, volume, seeds=None, n_iters=1000,
                              device=None, progressive=False, **fit_kwargs)
```

Isotropic Gaussian of std σ in 3D: `cholesky_factors = [1/σ, 0, 1/σ, 0, 0, 1/σ]`.

## Groups & transforms

```python
group = scene.add_group("organ", **attrs)   # returns a Group with all add_* methods
group.add_points("nuclei", pts)              # nested -> composed transforms

from luxar import transforms              # core/transforms.py — all return 4x4 float32
transforms.identity()
transforms.translate(x=0, y=0, z=0)
transforms.scale(x=1, y=1, z=1, uniform=None)
transforms.rotate_x(degrees) / rotate_y(degrees) / rotate_z(degrees)
transforms.rotate(degrees, axis)            # axis: "x"|"y"|"z" | (x,y,z) | ndarray
transforms.compose(t1, t2, t3)              # applies t1 FIRST, then t2, then t3
transforms.inverse(t)
transforms.look_at(eye, center, up)

# attach via the transform= attr on any node
scene.add_points("pts", positions, transform=transforms.compose(rot, trans))
```

`transform` is the 4x4 spatial matrix (composed parent→child). It is separate from
`nd_transform`, which acts per-dimension on non-displayed dims (see
`docs/guides/specs/ND_TRANSFORMS_SPEC.md`).

## Common `**attrs`
`opacity` (float), `intensity` (HDR multiplier), `gamma`, `blending_mode`
(`additive` | `normal` | `max` | `opaque` | `luminous`), `colormap` (name),
`layer`, `visible` (bool), `transform` (4x4 ndarray).

## Overlays (on-screen 2D UI)

```python
scene.add_text(text, position=(0.02, 0.02), font_size=0.025,
               anchor="top-left", color="white", blend_mode="normal")   # -> Overlay
```
`position` is normalized `(x, y)` in `[0,1]²`.

## Serialize & serve

The store is written progressively; the context-manager exit finalizes it. To copy
the finalized store elsewhere: `scene.to_zarr(path)` (`core/scene/scene.py`). Serve
with the CLI:

```bash
luxar serve scene.luxar.zarr --viewer --open     # data server + viewer + browser
luxar export scene.luxar.zarr -o out/ --open     # standalone offline folder
```

From a demo script, the canonical one-liner is `launch_viewer(output_path)` (it shells
out to `luxar serve ... --viewer`).
