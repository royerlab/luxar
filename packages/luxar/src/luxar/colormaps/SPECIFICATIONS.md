# Colormaps — Specifications

**Version**: 1.0.0
**Last Updated**: 2026-03-31

## Changelog

- **v1.0.0** (2026-03-31): Initial version header added; content unchanged.

---

## LUT Format

All colormaps are normalized to a standard format:
- Shape: `(256, 3)` — 256 entries, 3 channels (RGB)
- Dtype: `uint8` — values in [0, 255]
- Order: Row-major, channel order R, G, B
- Index 0 maps to scalar minimum, index 255 maps to scalar maximum

## Built-in Storage

Built-in colormaps are stored as raw byte literals in `builtins.py`:
- Each colormap: 768 bytes (256 entries x 3 channels x 1 byte)
- Total: ~12KB for 16 colormaps
- Lazy conversion to numpy arrays via `_bytes_to_lut()`
- Cached after first access; `get_builtin_lut()` returns a copy to prevent mutation

## Resolution Algorithm

`resolve_colormap(colormap)` follows this logic:

```
if isinstance(colormap, str):
    1. Check BUILTIN_COLORMAP_NAMES → return built-in LUT
    2. Try matplotlib.pyplot.get_cmap(name) → sample at 256 points → uint8
    3. Try colorcet.palette[name] → parse hex → uint8
    4. Raise ValueError with list of available built-ins
elif isinstance(colormap, np.ndarray):
    1. Validate shape (N, 3) with N >= 2
    2. Convert float [0,1] → uint8, or accept uint8 directly
    3. Resample to 256 entries if N != 256
else:
    Raise TypeError
```

## Resampling Algorithm

When a custom LUT has N != 256 entries, linear interpolation is used per channel:

```python
src_t = np.linspace(0, 1, N)       # Source sample positions
dst_t = np.linspace(0, 1, 256)     # Target sample positions
for each channel c in [R, G, B]:
    result[:, c] = np.interp(dst_t, src_t, lut[:, c])
```

This preserves endpoints exactly (first and last colors unchanged).

## Zarr Storage

### Named colormaps
Stored as a string attribute on the node group:
```
group.attrs["colormap"] = "viridis"
```

### Custom colormaps
Stored as a dataset + marker attribute:
```
group.attrs["colormap"] = "custom"
group["colormap_lut"] = uint8 array of shape (256, 3)
```

### Scalar data
For Points and Lines, scalar values for colormap lookup are stored as:
```
group["scalars"] = float32 array of shape (N,)
group.attrs["scalar_data_range"] = [min, max]
```

For GSplats, the existing `amplitudes` array serves as the scalar source,
and `amplitude_data_range` provides the normalization range.

## Linear Ramp Colormaps

Microscopy linear ramps map `t ∈ [0, 1]` to `(t*R, t*G, t*B)`:

| Name | End color (R, G, B) |
|------|-------------------|
| green | (0, 255, 0) |
| magenta | (255, 0, 255) |
| cyan | (0, 255, 255) |
| red | (255, 0, 0) |
| blue | (0, 0, 255) |
| yellow | (255, 255, 0) |
| gray | (255, 255, 255) |

All start at black (0, 0, 0).

## Viewer-Side Application

The viewer applies colormaps in the vertex shader (one LUT lookup per vertex/splat):

```glsl
float t = clamp((scalar - uScalarMin) * uScalarScale, 0.0, 1.0);
vec3 color = texture(uColormapTex, vec2(t, 0.5)).rgb;
```

The existing GOG (Gain-Offset-Gamma) pipeline then operates on the resulting RGB color.
For GSplats, the Gaussian falloff modulates the looked-up color in the fragment shader.
