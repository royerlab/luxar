# luxar.utils - Technical Specification

**Version**: 1.0.1
**Last Updated**: 2025-11-27

## Purpose

The `utils` package provides utility functions for array manipulation, broadcasting, and demo scene generation. These are helper functions used throughout Luxar.

**Related Specifications**:
- `luxar.core` - Data structures using these utilities (see `core/SPECIFICATIONS.md`)
- `luxar.validation` - Uses array utilities for validation (see `validation/SPECIFICATIONS.md`)

---

## Array Utilities (array.py)

### Broadcasting Specification

**Purpose**: Expand single values or small arrays to match point cloud size

#### broadcast_color_to_points(colors, n_points)

**Inputs**:
- `colors`: None | tuple(3) | list(3) | array(3,) | array(n_points, 3)
- `n_points`: Target number of points

**Behavior**:
- None → None (no colors)
- Single RGB (3 values) → tile to (n_points, 3)
- Full array (n_points, 3) → return as-is (validate shape)

**Output**: None or (n_points, 3) float32 array

**Errors**: ValueError if shape doesn't match (n_points, 3) or (3,)

#### broadcast_scalar_to_points(values, n_points, name, require_positive)

**Generic scalar broadcasting for radii, sharpness, or any per-point attribute**

**Inputs**:
- `values`: None | scalar | array(n_points,)
- `n_points`: Target number of points
- `name`: Attribute name for error messages
- `require_positive`: Whether to enforce > 0

**Behavior**:
- None → None
- Scalar → full array of that value
- Array → validate shape and requirements

**Output**: None or (n_points,) float32 array

**Validation**:
- If require_positive: all values must be > 0
- Shape must be (n_points,) exactly

#### Specialized Broadcasters

- `broadcast_radii_to_points()`: Calls broadcast_scalar with require_positive=True
- `broadcast_sharpness_to_points()`: Calls broadcast_scalar with require_positive=False

### Array Utilities

#### ensure_float32(array)
**Purpose**: Convert any array to float32 dtype
**Behavior**: No-op if already float32, otherwise astype(np.float32)

#### validate_array_shape(array, expected_shape, name)
**Purpose**: Validate array has expected shape(s)
**Input**: expected_shape can be single tuple or list of acceptable tuples
**Error**: ValueError with clear message showing expected vs actual

---

## Demo Scene Generators (demos.py)

### create_lorenz_attractor(store_path, n_points, seed)

**Purpose**: Generate aesthetically pleasing demo with Lorenz attractor

**Algorithm**:
1. Initialize Lorenz system at (0.1, 0, 0) with optional random perturbation
2. Integrate Lorenz equations using Euler method:
   ```
   dx/dt = σ(y - x)
   dy/dt = x(ρ - z) - y
   dz/dt = xy - βz

   where σ=10, ρ=28, β=8/3, dt=0.01
   ```
3. Generate n_points trajectory points
4. Scale positions by 0.1 for comfortable viewing
5. Center at center of mass

**Color Generation** (Vectorized HSV→RGB):
1. Create time parameter: t = linspace(0, 1, n_points)
2. Hue cycles: hue = (t * 2) % 1.0 (cycles twice through color wheel)
3. Convert HSV to RGB using vectorized algorithm:
   ```
   c = v * s (chroma)
   h_prime = hue * 6.0 (hue in [0, 6) range)
   x = c * (1 - |h_prime % 2 - 1|) (intermediate value)
   m = v - c (match value)

   sector = floor(h_prime)
   For each sector (0-5), assign RGB based on hue position
   Final RGB = (r+m, g+m, b+m)
   ```
4. Use boolean masks for vectorization (no loops)

**Radii**: Linear progression from 0.01 to 0.02 (growing effect)

**Output**: Zarr store with single "LorenzAttractor" points group

### create_random_spheres(store_path, n_spheres, points_per_sphere, seed)

**Purpose**: Generate multiple random colored spheres

**Algorithm**:
For each sphere:
1. Random center position: uniform(-10, 10) in 3D
2. Random radius: uniform(0.5, 2.0)
3. Generate points using spherical coordinates:
   ```
   φ = uniform(0, 2π)
   cos(θ) = uniform(-1, 1)
   u = uniform(0, 1)

   θ = arccos(cos(θ))
   r = radius * u^(1/3)  # Uniform volume distribution

   x = r * sin(θ) * cos(φ) + center_x
   y = r * sin(θ) * sin(φ) + center_y
   z = r * cos(θ) + center_z
   ```
4. Random HDR color: uniform(0.5, 2.0) in RGB
5. Fixed point radius: 0.05
6. Write each sphere as separate points group

### create_time_series_demo(store_path, n_timepoints, n_points_per_time, seed)

**Purpose**: Generate 4D time series (expanding sphere over time)

**Dimension Specification**:
```python
Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True, range=(0, n_timepoints-1))
])
```

**Algorithm**:
For each timepoint t:
1. Sphere radius grows: radius = 1.0 + t * 0.5
2. Generate random points in expanding sphere
3. Add time coordinate: positions_4d = [x, y, z, t]
4. Color transitions: interpolate from red→purple over time
5. Concatenate all timepoints into single 4D array
6. Write as single points group

---

## Utility Constants

**From array.py**:
- Broadcasting uses np.tile() for efficiency
- All outputs are float32 for GPU compatibility
- Warnings use Python's warnings module

**Demo Defaults**:
- Lorenz: σ=10, ρ=28, β=8/3, dt=0.01
- Sphere volume: u^(1/3) for uniform distribution
- HDR colors: values > 1.0 for vibrant visualization

---

## HSV to RGB Vectorization

**Critical Algorithm** (30-100x faster than loop):

```python
# Input: hue array in [0, 1]
# Output: RGB in [0, 1] (or higher for HDR)

# Convert to 6-sector representation
h_prime = hue * 6.0
sector = floor(h_prime)  # Integer in [0, 5]

# Calculate chroma and intermediate
c = v * s  # Chroma (v=1, s=1 for vibrant)
x = c * (1 - |h_prime % 2 - 1|)  # Intermediate value
m = v - c  # Match value

# Initialize RGB arrays
r, g, b = zeros(n), zeros(n), zeros(n)

# Apply values per sector using boolean masks
sector==0: (r,g,b) = (c, x, 0)  # Red → Yellow
sector==1: (r,g,b) = (x, c, 0)  # Yellow → Green
sector==2: (r,g,b) = (0, c, x)  # Green → Cyan
sector==3: (r,g,b) = (0, x, c)  # Cyan → Blue
sector==4: (r,g,b) = (x, 0, c)  # Blue → Magenta
sector==5: (r,g,b) = (c, 0, x)  # Magenta → Red

# Add match value
RGB = (r+m, g+m, b+m)
```

**Performance**: O(n) with vectorized operations, no Python loops

---

## Error Handling

**Validation Errors**:
- Use ValidationError from validation/base.py
- Include context in error messages
- Provide fix suggestions

**Demo Generation**:
- Check for performance warnings (large n_points)
- Use arbol for structured console output
- Handle seed for reproducibility

---

## This specification provides sufficient detail to re-implement the utility functions and demo generators.

---

## Changelog

- **v1.0.1** (2025-11-27): Removed sharpness warning
  - Removed arbitrary sharpness "typical range" warning (was unhelpful)

- **v1.0.0** (2025-11-27): Initial versioned specification
  - Documented array utilities and broadcasting
  - Specified demo generation functions
  - Defined error handling patterns
