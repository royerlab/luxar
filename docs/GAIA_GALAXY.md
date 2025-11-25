Perfect — here’s a clear, copy-pasteable playbook you can hand to **Claude Code** to generate working code that pulls **the brightest 10,000,000 Milky Way stars** (positions + color + brightness), does a **Galactocentric transform**, and enforces a **bounding sphere centered on the Galactic Center**.

I include (1) the resources you’ll need, (2) two robust workflows that people actually use, (3) ready-to-use ADQL building blocks, and (4) the exact post-processing steps (with parameter choices and references).

---

# 0) What you’ll need (Python + services)

**Python packages**

* `astroquery` (TAP access to ESA Gaia Archive)
* `pyvo` (TAP access to GAVO/ARI services)
* `astropy` (coords/units; Galactocentric transform)
* `pandas`, `pyarrow` (I/O: Parquet/Feather/CSV)

```bash
pip install astroquery pyvo astropy pandas pyarrow
```

**Endpoints you’ll hit**

* **ESA Gaia Archive TAP+**: `https://gea.esac.esa.int/tap-server/tap` (full DR3, ADQL, GAIA_HEALPIX_INDEX) ([cosmos.esa.int][1])
* **GAVO (Heidelberg) TAP**: `https://dc.g-vo.org/tap` (fast DR3 “lite” + **GeDR3 distances** table) ([dc.zah.uni-heidelberg.de][2])

**Docs you may want handy**

* DR3 contents & column names (e.g., `phot_g_mean_mag`, `bp_rp`) ([cosmos.esa.int][3])
* DSC (probabilities for galaxy/quasar; optional filter) is documented in **astrophysical_parameters** for DR3. ([gea.esac.esa.int][4])
* ADQL “how-to” & examples (ESA) ([cosmos.esa.int][1])
* HEALPix tips & server-specific functions (ESA: `GAIA_HEALPIX_INDEX`, GAVO/ARI: `ivo_healpix_index`) ([gaia.ari.uni-heidelberg.de][5])
* Astropy **Galactocentric** usage and defaults (so you can set **R0**) ([docs.astropy.org][6])

---

# 1) Choose your workflow

You have two good, production-style paths. Both end with the **same** post-step (Galactocentric transform + spherical cut). In practice, teams often prefer **A** (distances from Bailer-Jones) because it avoids naïve parallax inversion at low S/N.

## A) (Recommended) Use **GAVO** + **GeDR3 distances (Bailer-Jones)**

* Table: `gaia.dr3lite` (trimmed DR3, fast) + join with `gedr3dist.main` (robust geometric distances). ([gaia.ari.uni-heidelberg.de][5])
* Why: the BJ distances incorporate priors; safer than 1/π, especially for faint stars. ([bailer-jones.www3.mpia.de][7])
* How people do it: tile the sky by **HEALPix**, fetch only the columns you need, **order by brightness** inside each tile, then merge and keep the **global top 10M**. (Avoids a single massive ORDER BY across ~1.8B sources.) HEALPix partitioning is explicitly recommended by ARI and VO docs. ([gaia.ari.uni-heidelberg.de][5])

## B) Use **ESA Gaia Archive** only (DR3 + optional DSC filters)

* Table: `gaiadr3.gaia_source` (+ optional left join to `gaiadr3.astrophysical_parameters` to down-weight galaxies/quasars with `classprob_dsc_*`). ([cosmos.esa.int][3])
* How people do it: same HEALPix tiling; ESA provides `GAIA_HEALPIX_INDEX`. ([cosmos.esa.int][1])

---

# 2) Bounding sphere definition (what to filter on)

You want “**within a bounding sphere centered on the Galactic Center**.” Do that **after** download:

1. Convert (α, δ, distance) → **Galactocentric** Cartesian (x, y, z) with **Astropy**. Set **R₀** (Sun–GC distance) explicitly; a commonly used modern value is **R₀ ≈ 8.122 kpc** (GRAVITY 2018). ([arXiv][8])
2. Keep stars with ( \sqrt{x^2+y^2+z^2} \le R_{\mathrm{max}} ) where you pick **Rmax** (e.g., 30–50 kpc depending on how much halo you want).
3. Since you also want the **brightest** 10M, you’ll (a) pull many more than 10M using tile-wise brightness ordering, (b) concatenate, (c) take **global top-10M by `phot_g_mean_mag`**, then (d) apply the **Galactocentric sphere** cut. (Either order works; if you sphere-cut first, make sure you still have ≥10M to rank.)

Astropy examples show exactly how to set the Galactocentric frame and its defaults. ([docs.astropy.org][6])

---

# 3) ADQL building blocks (Claude can parameterize these)

### 3.1 Columns to select (minimal 3D + color/brightness)

* `source_id, ra, dec, parallax, parallax_over_error, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, bp_rp` (DR3 standard) ([cosmos.esa.int][3])
* **Workflow A** also selects **`d.rgeo`** (geometric distance in pc) from `gedr3dist.main`. ([dc.g-vo.org][9])

### 3.2 HEALPix tiling

* **GAVO/ARI**: `ivo_healpix_index(order, ra, dec)` (e.g., order 5–7 for manageable tiles). ([gaia.ari.uni-heidelberg.de][5])
* **ESA**: `GAIA_HEALPIX_INDEX(order, ra, dec)` or using `source_id`-based helpers. ([cosmos.esa.int][10])

### 3.3 “One tile” query (Workflow A: GAVO + BJ distances)

Ask Claude to loop over `hpx IN (...)` and run this per tile, then write to disk:

```sql
SELECT
  g.source_id, g.ra, g.dec,
  g.parallax, g.parallax_over_error, g.pmra, g.pmdec,
  g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
  d.rgeo AS dist_pc
FROM gaia.dr3lite AS g
JOIN gedr3dist.main AS d USING (source_id)
WHERE ivo_healpix_index(6, g.ra, g.dec) = {HPX}
  AND d.rgeo < 50000            -- heliocentric < 50 kpc (generous MW bound)
  AND g.phot_bp_mean_mag IS NOT NULL
  AND g.phot_rp_mean_mag IS NOT NULL
ORDER BY g.phot_g_mean_mag ASC
LIMIT {PER_TILE_LIMIT};
```

* **Order 6** has 49,152 tiles; use a **subset** (e.g., cycle through all, or choose coarser order like 5 = 12,288 tiles). ARI recommends HEALPix constraints to scale. ([gaia.ari.uni-heidelberg.de][5])
* Pick `{PER_TILE_LIMIT}` so that the **sum across tiles comfortably exceeds 10M** (e.g., 400 per tile at order-6 gives ~20M rows, which you’ll reduce later).

### 3.4 (Optional) “One tile” query (Workflow B: ESA + DSC filter)

```sql
SELECT
  g.source_id, g.ra, g.dec,
  g.parallax, g.parallax_over_error, g.pmra, g.pmdec,
  g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp
FROM gaiadr3.gaia_source AS g
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
  ON ap.source_id = g.source_id
WHERE GAIA_HEALPIX_INDEX(6, g.ra, g.dec) = {HPX}
  AND g.phot_bp_mean_mag IS NOT NULL
  AND g.phot_rp_mean_mag IS NOT NULL
  AND COALESCE(ap.classprob_dsc_combmod_galaxy, 0) < 0.01
  AND COALESCE(ap.classprob_dsc_combmod_quasar, 0) < 0.01
ORDER BY g.phot_g_mean_mag ASC
LIMIT {PER_TILE_LIMIT};
```

* DSC probabilities are documented in DR3; adjust thresholds if you want stricter purity. Note DR3 DSC completeness/purity caveats vs galaxies/quasars. ([gea.esac.esa.int][4])

---

# 4) What Claude should generate (Python outline)

Have Claude produce a single script with these **numbered sections** (so it’s easy for you to review):

### (1) Config

* `SERVICE = "gavo"` or `"esa"`
* `TAP_URL = "https://dc.g-vo.org/tap"` **or** `"https://gea.esac.esa.int/tap-server/tap"`
* `HEALPIX_ORDER = 6`
* `PER_TILE_LIMIT = 400`  *(tune to end near 15–25M rows before final top-10M cut)*
* `OUT_DIR = "gaia_tiles"`
* `RMAX_KPC = 30.0`  *(or 50.0 if you want more halo)*
* `R0_KPC = 8.122`  *(Sun–GC distance)* ([arXiv][8])

### (2) TAP client

* If `SERVICE=="gavo"`, use **`pyvo`** to submit ADQL (async, with retry on HTTP 5xx/timeouts).
* If `SERVICE=="esa"`, use **`astroquery.gaia.Gaia.launch_job_async`** (async jobs are standard). ([GitHub][11])

### (3) Tile driver

* Build list of HEALPix indices for the chosen order.
* For each `{HPX}`:

  * Fill the **ADQL template** (A or B).
  * Submit async job; download as CSV or FITS to `OUT_DIR/hpx_{h}.csv`.
  * Log rows returned.

*(Tiling is a best-practice for performance and fairness on TAP; both ESA and ARI recommend using HEALPix constraints rather than single monolithic queries.)* ([gaia.ari.uni-heidelberg.de][5])

### (4) Consolidate & take global “top-10M by brightness”

* Stream-read the per-tile files with `pandas`.
* Concatenate into a single Arrow/Parquet dataset (chunked).
* Use a **two-pass** strategy:

  1. Keep a **running quantile** estimate of `phot_g_mean_mag` to infer a global magnitude threshold that yields ~10M rows.
  2. Re-scan files, **filter to `G <= G_thresh`**, and if still >10M, take the 10M brightest by a final sort.

*(This avoids materializing, say, 25M rows in RAM.)*

### (5) Distances

* **Workflow A**: distance is `dist_pc = rgeo` from BJ (already in table, in parsecs). ([dc.g-vo.org][9])
* **Workflow B**: compute `dist_pc = 1000.0 / parallax_mas` **after** applying a parallax quality cut, or (better) join the ESA-hosted **external** BJ distances table named `external.gaiaedr3_distance` (slightly different name at ESA) to avoid raw inversion. ([bailer-jones.www3.mpia.de][7])

### (6) Galactocentric transform & sphere cut

* Use **Astropy**:

```python
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

# RA/Dec in deg, distance in pc -> convert once to kpc
dist_kpc = (df["dist_pc"].values * u.pc).to(u.kpc)
c_icrs = SkyCoord(ra=df["ra"].values*u.deg,
                  dec=df["dec"].values*u.deg,
                  distance=dist_kpc)

gc = c_icrs.transform_to(Galactocentric(galcen_distance=R0_KPC*u.kpc))
r_gc = (gc.x**2 + gc.y**2 + gc.z**2)**0.5
sel = r_gc <= (RMAX_KPC*u.kpc)
df_gc = df.loc[sel]
```

Astropy’s Galactocentric examples & defaults are documented here. ([docs.astropy.org][6])

### (7) Save final products

* Save both **spherical** (α, δ, distance, mags) and **Cartesian** (**x,y,z** in kpc) tables:

  * `milky_way_top10M_dr3.parquet` (spherical)
  * `milky_way_top10M_galcen.parquet` (x,y,z,G,BP_RP)
* Include a tiny **README.md** with:

  * the ADQL you ran,
  * the HEALPix order & per-tile limit,
  * R0 and Rmax values, and
  * the date + TAP service used (for reproducibility).

---

# 5) Guardrails & good defaults (Claude should bake these in)

* **Only select needed columns** (avoid `SELECT *`) — DR3 tables are wide; you’ll waste bandwidth. ([gaia.ari.uni-heidelberg.de][5])
* **Asynchronous TAP** jobs with retries & backoff. (TAP servers queue large jobs.) ([cosmos.esa.int][1])
* **Batching via HEALPix** is normal and recommended by ARI/ESA docs for large pulls. ([gaia.ari.uni-heidelberg.de][5])
* **Distances**: prefer **Bailer-Jones GeDR3** (`gedr3dist.main`) rather than raw parallax inversion. Also available at ESA under “external catalogues” (`external.gaiaedr3_distance`). ([dc.g-vo.org][9])
* **Extragalactic contamination**: optional DSC probability cut (e.g., `< 0.01`). DR3 papers note completeness/purity trade-offs; don’t over-rely on DSC alone. ([gea.esac.esa.int][4])
* **R0 setting**: Set Galactocentric **`galcen_distance=8.122 kpc`** unless you intentionally test alternatives (Astropy lets you change defaults). ([arXiv][8])

---

# 6) Ready-to-paste prompts for Claude Code

### Prompt A (GAVO + BJ distances; HEALPix tiling)

> Write a Python 3 script that:
>
> 1. Uses **pyvo** to connect to `https://dc.g-vo.org/tap`.
> 2. Iterates over **HEALPix order 6** indices (49,152 tiles) but only queries the **top N=400 brightest** per tile using this ADQL (fill `{HPX}`):
>
>    ```
>    SELECT
>      g.source_id, g.ra, g.dec,
>      g.parallax, g.parallax_over_error, g.pmra, g.pmdec,
>      g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp,
>      d.rgeo AS dist_pc
>    FROM gaia.dr3lite AS g
>    JOIN gedr3dist.main AS d USING (source_id)
>    WHERE ivo_healpix_index(6, g.ra, g.dec) = {HPX}
>      AND d.rgeo < 50000
>      AND g.phot_bp_mean_mag IS NOT NULL
>      AND g.phot_rp_mean_mag IS NOT NULL
>    ORDER BY g.phot_g_mean_mag ASC
>    LIMIT 400;
>    ```
> 3. Runs **asynchronous TAP jobs** with retry/backoff, writes each tile to `gaia_tiles/hpx_{HPX}.csv`.
> 4. Concatenates all CSVs **streaming**, estimates a global brightness threshold for **top ~10M** by `phot_g_mean_mag`, and writes a filtered file.
> 5. Converts to **Galactocentric** (Astropy) with `galcen_distance=8.122 kpc`, computes `r_gc`, filters to `r_gc <= 30 kpc`.
> 6. Outputs two Parquet files: spherical columns and Cartesian (**x,y,z,kpc,G,BP_RP**), plus a short `README.md` with config and ADQL.
>    Make the code idempotent, resumable (skip tiles already downloaded), and memory-safe (no giant in-RAM concatenations).

### Prompt B (ESA only; optional DSC filter)

> Same as Prompt A **but** use **astroquery.gaia** (ESA TAP at `https://gea.esac.esa.int/tap-server/tap`) and this per-tile ADQL:
>
> ```
> SELECT
>   g.source_id, g.ra, g.dec,
>   g.parallax, g.parallax_over_error, g.pmra, g.pmdec,
>   g.phot_g_mean_mag, g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp
> FROM gaiadr3.gaia_source AS g
> LEFT JOIN gaiadr3.astrophysical_parameters AS ap ON ap.source_id = g.source_id
> WHERE GAIA_HEALPIX_INDEX(6, g.ra, g.dec) = {HPX}
>   AND g.phot_bp_mean_mag IS NOT NULL
>   AND g.phot_rp_mean_mag IS NOT NULL
>   AND COALESCE(ap.classprob_dsc_combmod_galaxy, 0) < 0.01
>   AND COALESCE(ap.classprob_dsc_combmod_quasar, 0) < 0.01
> ORDER BY g.phot_g_mean_mag ASC
> LIMIT 400;
> ```
>
> Keep the rest identical, but either (a) invert parallax with a quality cut, or (b) add a **second pass** to join ESA’s **external** BJ distances (`external.gaiaedr3_distance`).

---

# 7) Why this mirrors community practice

* **ADQL + TAP + HEALPix tiling** is the standard way to pull large Gaia subsets efficiently; both ESA and ARI/GAVO recommend it. ([gaia.ari.uni-heidelberg.de][5])
* **GeDR3 distances** are widely used to avoid biases from raw parallax inversion; they’re hosted at GAVO and also mirrored in ESA’s archive (as an “external catalogue”). ([dc.g-vo.org][9])
* **Astropy Galactocentric** is the canonical tool for the frame transform and lets you fix **R0** explicitly. ([docs.astropy.org][6])

---

If you want, I can also hand you a **single consolidated prompt** that asks Claude to produce the *exact* script (including CLI args and logging) — just say which workflow (A vs B), your **Rmax** (30 or 50 kpc), and your preferred **output format** (Parquet/Feather/CSV).

[1]: https://www.cosmos.esa.int/web/gaia-users/archive/writing-queries?utm_source=chatgpt.com "How to write ADQL queries for Gaia data - Gaia Users - Cosmos"
[2]: https://dc.zah.uni-heidelberg.de/browse/gaia/q3?utm_source=chatgpt.com "Information on resource 'Selections from Gaia Data Release 3 (DR3)'"
[3]: https://www.cosmos.esa.int/web/gaia/dr3?utm_source=chatgpt.com "Gaia Data Release 3 contents summary - Gaia - Cosmos"
[4]: https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html?utm_source=chatgpt.com "20.2.1 astrophysical_parameters‣ 20.2 Astrophysical parameter tables ..."
[5]: https://gaia.ari.uni-heidelberg.de/tap.html?utm_source=chatgpt.com "TAP · ARI's Gaia Services"
[6]: https://docs.astropy.org/en/stable/coordinates/example_gallery_plot_galactocentric_frame.html?utm_source=chatgpt.com "Transforming positions and velocities to and from a Galactocentric frame"
[7]: https://bailer-jones.www3.mpia.de/gedr3_distances.html?utm_source=chatgpt.com "GeDR3 distances - Max Planck Society"
[8]: https://arxiv.org/pdf/1808.09435?utm_source=chatgpt.com "arXiv:1808.09435v1 [astro-ph.GA] 28 Aug 2018"
[9]: https://dc.g-vo.org/tableinfo/gedr3dist.main?tapinfo=True&utm_source=chatgpt.com "Table information for 'gedr3dist.main'"
[10]: https://www.cosmos.esa.int/web/gaia-users/archive/faq-old?utm_source=chatgpt.com "FAQ old - Gaia Users - Cosmos"
[11]: https://github.com/astropy/astroquery/issues/2276?utm_source=chatgpt.com "Gaia.launch_job_async output_file parameter enhancement"
