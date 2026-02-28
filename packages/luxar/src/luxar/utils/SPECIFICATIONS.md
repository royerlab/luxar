# luxar.utils - Technical Specification

**Version**: 1.2.0
**Last Updated**: 2026-02-28

## Purpose

The `utils` package provides utility functions for array manipulation and demo scene generation. These are helper functions used throughout Luxar.

**Related Specifications**:
- `luxar.core` - Data structures using these utilities (see `core/SPECIFICATIONS.md`)
- `luxar.validation` - Uses array utilities for validation (see `validation/SPECIFICATIONS.md`)
- `luxar.encoding` - Handles scalar broadcasting since v1.4.0 (see `encoding/SPECIFICATIONS.md`)

---

## Array Utilities (array.py)

**Note**: As of Luxar v1.4.0, scalar broadcasting (e.g., `colors=(1,0,0)`, `radii=0.5`) is handled by `ArrayEncoder` in `luxar.encoding`. The previous `broadcast_*_to_points()` functions have been removed.

### Current Functions

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
4. Scale positions by 0.1 for initial normalization
5. Center at center of mass (subtract mean position)
6. Final scaling and offset: positions * 100.0 - 50.0 for optimal viewer framing

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

**Radii**: Uniform radius 2.0 for consistent visibility

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

## Constants

**Demo Defaults**:
- Lorenz: σ=10, ρ=28, β=8/3, dt=0.01
- Sphere volume: u^(1/3) for uniform distribution
- HDR colors: values > 1.0 for vibrant visualization

**Array Utilities**:
- `ensure_float32()`: Returns float32 for GPU compatibility
- `validate_array_shape()`: Raises ValueError with clear messages

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

## Path Utilities (paths.py)

### Purpose

Provides consistent path resolution for all generated datasets, ensuring outputs go to a centralized `datasets/` folder at the project root regardless of where scripts are run from.

### Functions

#### get_project_root() -> Path

**Purpose**: Find the Luxar project root directory by traversing up from the module file looking for `pyproject.toml`.

**Behavior**:
- Starts from `Path(__file__).resolve()` and walks up through parent directories
- Returns the first parent directory containing a `pyproject.toml` file
- Result is cached via `@lru_cache(maxsize=1)` for performance

**Returns**: `Path` to the project root

**Raises**: `RuntimeError` if no parent directory contains `pyproject.toml`

#### get_datasets_dir() -> Path

**Purpose**: Get the `datasets/` directory at project root.

**Behavior**: Calls `get_project_root() / "datasets"` and creates the directory if it does not exist (`mkdir(exist_ok=True)`).

**Returns**: `Path` to `<project_root>/datasets/`

#### get_examples_output_dir() -> Path

**Purpose**: Get the output directory for example scripts.

**Behavior**: Calls `get_datasets_dir() / "examples"` and creates the directory if it does not exist.

**Returns**: `Path` to `<project_root>/datasets/examples/`

#### get_demos_output_dir() -> Path

**Purpose**: Get the output directory for demo scripts.

**Behavior**: Calls `get_datasets_dir() / "demos"` and creates the directory if it does not exist.

**Returns**: `Path` to `<project_root>/datasets/demos/`

### Directory Layout

```
<project_root>/
  pyproject.toml          # Sentinel for root detection
  datasets/               # Created by get_datasets_dir()
    examples/             # Created by get_examples_output_dir()
    demos/                # Created by get_demos_output_dir()
```

---

## Download Utilities (download.py)

### Purpose

Production-grade download functionality for large datasets with automatic retry on failure, partial download resume via HTTP Range requests, progress tracking with ETA, and integrity verification.

### Functions

#### robust_download(url, output_path, ...)

**Purpose**: Download a file with automatic retry, resume capability, and progress tracking.

**Parameters**:
- `url` (`str`): URL to download from
- `output_path` (`Path`): Where to save the downloaded file
- `max_retries` (`int`, default `3`): Maximum number of retry attempts
- `timeout` (`int`, default `300`): Timeout in seconds for initial connection
- `chunk_size` (`int`, default `1048576`): Size of download chunks in bytes (1 MB)
- `verify_size` (`bool`, default `True`): Whether to verify final file size matches Content-Length
- `expected_size` (`Optional[int]`, default `None`): Expected file size in bytes for validation

**Returns**: `Path` to downloaded file

**Raises**:
- `requests.HTTPError`: If HTTP error occurs after all retries
- `requests.ConnectionError`: If connection fails after all retries
- `ValueError`: If downloaded file size does not match expected size

**Behavior**:
1. Skips download if file already exists and matches `expected_size`
2. Sets up `requests.Session` with exponential backoff retry strategy (2s, 4s, 8s...)
3. Retries on HTTP status codes: 429, 500, 502, 503, 504
4. If a partial file exists, attempts resume via HTTP Range header
5. Falls back to full download if server returns 200 instead of 206
6. Reports progress every 100 MB with download speed and ETA
7. Verifies final file size against Content-Length when `verify_size=True`
8. On HTTP errors, deletes partial file; on connection errors, keeps partial file for future resume

#### verify_file_checksum(file_path, expected_md5=None, expected_sha256=None)

**Purpose**: Verify file integrity using MD5 and/or SHA256 checksums.

**Parameters**:
- `file_path` (`Path`): Path to file to verify
- `expected_md5` (`Optional[str]`): Expected MD5 hex digest
- `expected_sha256` (`Optional[str]`): Expected SHA256 hex digest

**Returns**: `True` if file matches all provided checksums, `False` otherwise (or if file does not exist)

**Behavior**: Reads file in 1 MB chunks for memory-efficient hashing. Checks MD5 first, then SHA256 (if both provided). Returns `False` on first mismatch.

#### download_with_checksum(url, output_path, expected_md5=None, expected_sha256=None, **kwargs)

**Purpose**: Combine `robust_download()` with checksum verification for data integrity.

**Parameters**:
- `url` (`str`): URL to download from
- `output_path` (`Path`): Where to save the file
- `expected_md5` (`Optional[str]`): Expected MD5 hash
- `expected_sha256` (`Optional[str]`): Expected SHA256 hash
- `**kwargs`: Additional arguments passed to `robust_download()`

**Returns**: `Path` to downloaded and verified file

**Raises**: `ValueError` if checksum verification fails (file is deleted automatically on failure)

**Behavior**: Calls `robust_download()` first, then `verify_file_checksum()`. If verification fails, the corrupted file is deleted and a `ValueError` is raised.

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

## Changelog

- **v1.2.0** (2026-02-28): Added path and download utility documentation
  - **ADDED**: Path Utilities (paths.py) section documenting `get_project_root()`, `get_datasets_dir()`, `get_examples_output_dir()`, `get_demos_output_dir()`
  - **ADDED**: Download Utilities (download.py) section documenting `robust_download()`, `verify_file_checksum()`, `download_with_checksum()`
  - Removed stray heading-level concluding sentence

- **v1.1.1** (2025-11-30): Documentation cleanup
  - Removed obsolete "Utility Constants" section that referenced removed np.tile() broadcasting
  - Renamed section to "Constants" with current accurate content

- **v1.1.0** (2025-11-29): Removed obsolete broadcast functions
  - **BREAKING**: Removed `broadcast_color_to_points()`, `broadcast_radii_to_points()`,
    `broadcast_sharpness_to_points()`, `broadcast_scalar_to_points()`
  - These functions are now handled by `ArrayEncoder` in `luxar.encoding`
  - Added reference to encoding package in Related Specifications
  - Updated documentation to reflect current API

- **v1.0.1** (2025-11-27): Removed sharpness warning
  - Removed arbitrary sharpness "typical range" warning (was unhelpful)

- **v1.0.0** (2025-11-27): Initial versioned specification
  - Documented array utilities and broadcasting
  - Specified demo generation functions
  - Defined error handling patterns
