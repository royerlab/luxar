# Luxar Scripts

This directory contains standalone scripts for generating data and validating project documentation.

## Documentation Quality Checker

### `check_documentation.py`

Validates documentation completeness and quality across the Luxar project. Run this before commits or in CI/CD to ensure documentation standards.

**Purpose:**
- Ensure all Python packages have README.md files
- Check TypeScript packages for consistent documentation
- Validate Markdown syntax
- Maintain documentation standards across the project

**Usage:**

```bash
# Check all documentation
hatch run python scripts/check_documentation.py

# With verbose output
hatch run python scripts/check_documentation.py --verbose

# Auto-fix issues (if supported)
hatch run python scripts/check_documentation.py --fix
```

**What it checks:**
- Python packages have README.md with key classes and usage examples
- TypeScript packages have README.md in src/{package}/
- Documentation is syntactically valid Markdown

---

## Gaia DR3 Data Generator

## Gaia DR3 Data Generator

### `generate_galaxy_simple.py`

Fetches real star data from ESA's Gaia DR3 archive and saves it as a **raw zarr table** (NOT Luxar format).

**Purpose:**
- Generates raw astronomical data for use in demos
- Output is consumed by `demo_gaia_milky_way_3m.py`

**Additional Dependencies:**
```bash
pip install astroquery astropy
```

These are **NOT** part of luxar's core dependencies because they're large astronomy-specific packages only needed for data generation.

**Usage:**

```bash
# Install dependencies (one-time)
hatch run pip install astroquery astropy

# Generate 3M stars (used for demo, ~90 minutes)
hatch run python scripts/generate_galaxy_simple.py --count 3000000 --output packages/luxar/src/luxar/demos/data/milky_way_gaia_3m.zarr

# Test with smaller datasets
hatch run python scripts/generate_galaxy_simple.py --count 10000    # 10k stars (~30 sec)
hatch run python scripts/generate_galaxy_simple.py --count 100000   # 100k stars (~2 min)
```

**Output Format:**

Raw zarr table (NOT Luxar format) with arrays:
- `x_kpc` - Galactocentric X coordinate (float32)
- `y_kpc` - Galactocentric Y coordinate (float32)
- `z_kpc` - Galactocentric Z coordinate (float32)
- `phot_g_mean_mag` - G-band magnitude (float32)
- `bp_rp` - BP-RP color index (float32)

Plus metadata: `num_stars`, `magnitude_range`, `description`, `data_source`

**Query Parameters:**
- `parallax > 0.1 mas` → distances up to ~10 kpc from Sun
- `parallax_over_error > 5` → high-quality measurements only
- `ORDER BY phot_g_mean_mag ASC` → sorted by brightness

**Coordinate Transform:**
- Uses Astropy's Galactocentric frame
- R₀ = 8.122 kpc (Sun-GC distance, GRAVITY 2018)
- Filters to stars within 30 kpc of Galactic Center

**To visualize the data:**
```bash
# The raw data CANNOT be viewed directly
# Use the demo which converts to Luxar format:
python packages/luxar/src/luxar/demos/demo_gaia_milky_way_3m.py
```

---

**Data Attribution:**

ESA/Gaia/DPAC - Gaia Data Release 3 (2022)

Citation:
Gaia Collaboration, Vallenari et al. (2023)
"Gaia Data Release 3: Summary of the content and survey properties"
Astronomy & Astrophysics, 674, A1
DOI: 10.1051/0004-6361/202243940

---

**See also:**
- `demo_gaia_milky_way_3m.py` - Converts raw data to Luxar format and visualizes
- `demo_gaia_milky_way_8m.py` - Larger 8M star dataset demo
- ESA Gaia Archive: https://gea.esac.esa.int/archive/
