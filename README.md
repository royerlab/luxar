# Luxar

[![CI](https://github.com/royerlab/luxar/actions/workflows/ci.yml/badge.svg)](https://github.com/royerlab/luxar/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/luxar?label=PyPI)](https://pypi.org/project/luxar/)
[![npm](https://img.shields.io/npm/v/%40luxar%2Fviewer?label=npm%20%40luxar%2Fviewer)](https://www.npmjs.com/package/@luxar/viewer)
[![Docs](https://img.shields.io/badge/docs-royerlab.github.io%2Fluxar-blue)](https://royerlab.github.io/luxar/)
[![Live demos](https://img.shields.io/badge/live%20demos-demos.luxarviewer.dev-7c3aed)](https://demos.luxarviewer.dev)

**High-performance n-dimensional scientific visualization.**

![Luxar: n-dimensional scientific data, compiled and explored in the browser. Points, lines, Gaussian splats, meshes.](https://data.luxarviewer.dev/media/558d5db88635bec2.png)


Luxar makes large scientific datasets explorable in a web browser. You describe a
scene in Python — **points**, **lines**, **Gaussian splats**, and **triangle
meshes**, in as many dimensions as your data actually has — and Luxar *compiles*
it into a chunked, spatially indexed Zarr archive. A GPU viewer then streams that
archive and renders it. The expensive work happens once, at compile time; what
remains during exploration is bounded by your graphics card, your screen and your
network link, not by the size or format of the file.

Three ideas carry most of the design:

- **Compile, don't load.** Elements are reordered along a space-filling curve,
  chunked to ~64 KB and compressed, so the viewer fetches only the chunks the
  current view actually intersects — and starts drawing before the rest arrives.
- **n dimensions are first class.** Time, channel, camera, or any abstract axis is
  a real dimension with units and extent, not a folder of frames. Geometry is
  sliced by nD proximity, and every non-displayed axis becomes a navigable
  control — a slider, or a toggle or dropdown for categorical axes.
- **Represent, don't rasterize.** Image volumes are *fitted* to sparse oriented
  Gaussians instead of shipped as voxel grids — which is what lets a 3.3 GB
  light-sheet stack, or a 400-timepoint timelapse, travel over a network at all.

**[▶ Try it in your browser](https://demos.luxarviewer.dev)** — 88 live demos (of
the 90 bundled), no install. Or open your own data in the hosted viewer:
[luxarviewer.dev](https://luxarviewer.dev)`?src=<url-to-your-scene>`.

[Live Demos](https://demos.luxarviewer.dev) | [Docs Site](https://royerlab.github.io/luxar/) | [Quick Start](#quick-start) | [Volume Rendering](#volume-rendering-with-gaussian-splats) | [Gallery](#gallery) | [Documentation](#documentation) | [API Reference](#api-reference)

<p align="center">
  <a href="https://demos.luxarviewer.dev/d/gsplats_4d_drosophila_embryogenesis"><img src="https://data.luxarviewer.dev/media/4a0c1a68d18e092e.webp" alt="Drosophila gastrulation: 500 light-sheet timepoints fitted as Gaussian splats, streamed and played in the Luxar viewer" width="100%"></a>
  <br>
  <sub>A <em>Drosophila</em> embryo through gastrulation: 500 light-sheet timepoints, 256K Gaussian splats each, streamed from static hosting and played in the browser (<a href="https://demos.luxarviewer.dev/d/gsplats_4d_drosophila_embryogenesis">open it live</a>; recording by the Keller lab, HHMI Janelia).</sub>
</p>

---

## Why Luxar?

Most scientific viewers are limited by how much work the application must do per
frame. Luxar moves that work out of the interaction loop: spatial ordering,
chunking, compression, level-of-detail, and — for image volumes — the Gaussian fit
itself all happen ahead of time, so the browser is left with little to do but draw.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://data.luxarviewer.dev/media/345632ca3a0ddb55.png">
  <img alt="How Luxar works: describe a scene in Python, compile it once into a chunked, indexed .luxar.zarr archive, host the archive on any static file server, explore it in any browser" src="https://data.luxarviewer.dev/media/91044ca3c0ca2b00.png" width="100%">
</picture>

Because the archive is self-describing and chunked, the same output serves every
consumer: `luxar serve` for local exploration, a static file host for sharing, or a
standalone export a colleague opens with no Luxar install. And because the viewer's
remaining work is drawing, rendering is GPU-bound rather than application-bound —
real-world speed scales with your graphics card (headless software-GL, with no GPU,
is the slow exception).

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **[Four geometries](#geometry-types)** | Points, lines, Gaussian splats and triangle meshes share one attribute model — nD positions, colors and opacity (points, lines and splats add a per-element size; a triangle takes its extent from its own vertices) |
| **[n-Dimensional](#n-dimensional-visualization)** | 3D, 4D, 5D and beyond: named axes with physical units, radius-based slicing, keyboard navigation, per-axis transforms |
| **Massive scale** | 100K to 10M+ primitives at interactive frame rates, with level-of-detail and progressive streaming from local files or remote servers |
| **[Volume rendering](#volume-rendering-with-gaussian-splats)** | Image volumes fitted to oriented Gaussians — gigabytes of voxels become megabytes of streamable, GPU-native geometry, timelapses included |
| **Appearance** | HDR 16-bit float color, bloom and tone mapping, matplotlib/colorcet colormaps, and six blending modes from additive to physically-based [emission–absorption](#emission-and-absorption-not-just-glow) |
| **Fitting pipeline** | `cal → fit → lod` with cross-validated splat budgets, optional CUDA, and whole-timelapse fitting across many GPUs or a Slurm cluster |
| **Interoperable** | Reads classical 3D-Gaussian-splatting captures (INRIA, `.splat`, `.spz`, SuperSplat, PlayCanvas SOG); writes INRIA PLY |
| **Shareable** | `luxar export` produces a standalone offline folder, or a native bundle — a double-clickable macOS `.app`, a portable Linux folder — that opens with no Luxar install |

> **Requirements:** The Luxar viewer needs a browser with **WebGL2** support. Chromium, Firefox and WebKit are all tested — see [Browser Compatibility](#browser-compatibility) for what was measured and what was not. Touch input is implemented (one- and two-finger orbit/fly gestures, tap-to-pick, long-press menus, a coarse-pointer layout) and exercised by a Chromium-emulated iPhone/iPad/Pixel Playwright suite; performance on real phones and tablets has not been benchmarked, so treat mobile as supported but unmeasured.

---

## Quick Start

### Prerequisites

- Python 3.12 or newer
- Modern browser with WebGL 2.0
- **Gaussian-splat _fitting_** runs on the CPU; it is much faster on an NVIDIA **CUDA GPU** (see `make build-cuda`). Apple **MPS** is also supported, but neither GPU backend is required. _Viewing_ splats in the browser needs no GPU setup at all.
- **Ubuntu/Debian only**: `sudo apt-get install -y pipx && pipx ensurepath`

### Install and Run Demos

```bash
git clone https://github.com/royerlab/luxar.git
cd luxar
make setup-dev          # Auto-installs Node.js, pnpm, Hatch (no sudo)
hatch shell             # Activate the environment `setup-dev` created
luxar demo              # Browse the 90 bundled demos
luxar demo run cloud    # Run one — generates the data and opens the viewer
```

Once a release is published on PyPI, `pip install luxar` installs the viewer,
compiler and CLI without a checkout (add `"luxar[gsplats]"` to fit volumes), and
the same `luxar demo` commands apply. Some demos need extra packages;
`luxar demo deps --install` fetches what a demo reports missing.

`luxar demo run cloud` builds a convective cloud, a 4D point cloud that evolves over
a time axis, and opens it in the viewer, where the time slider plays it back:

<p align="center">
  <img src="https://data.luxarviewer.dev/media/b6d1359e5db8850e.webp" alt="Quick start recording: pip install luxar and luxar demo run cloud typed in a terminal, then the evolving cloud playing in the Luxar viewer" width="100%">
  <br>
  <sub>The post-release flow in one take: <code>pip install luxar</code>, then <code>luxar demo run cloud</code>. From a checkout today, the commands above do the same.</sub>
</p>

#### Browsing the catalogue

`luxar demo` prints every bundled demo, grouped by category — index, key,
geometry, what it needs, and whether you have already built it:

```
🎬 90 Luxar demos  ·  75 built  ·  8 cached  ·  7 not generated yet

 ASTRONOMY ──────────────────────────────────────────────────────────── 6 demos
 ✓  2  asteroids_solar_system                  points+lines  300 MB
 ✓ 12  cosmicflows_laniakea                    points+lines  25 MB
 • 15  desi_galaxies                           points        73 MB

 MEDICAL ────────────────────────────────────────────────────────────── 4 demos
 ✓ 17  dmri_tractography                       lines         588 MB
 • 29  gsplats_2d_cmu1_pathology               gsplats       150 MB GPU?

 SYNTHETIC ─────────────────────────────────────────────────────────── 18 demos
 ✓ 10  cloud                                   points
   11  collision                               points
 ...
 ✓ 22  exotic_surfaces                         points

 ✓ built   • inputs cached   (blank) not generated yet
 GPU/GPU? = required/optional     git-lfs kaggle manual = data you supply

 Run      luxar demo run <key|#>                e.g. luxar demo run cloud
 Details  luxar demo info <key|#>
 Filter   luxar demo list -c astronomy -g gsplats
 Deps     luxar demo deps                       --install fixes what it reports
 Caches   luxar demo cache list                 cache clear <key> to reclaim
 Stop     luxar demo stop                       frees ports of forgotten runs
```

The **left rail** is what you already have: `✓` the scene is built, `•` the
inputs are downloaded, blank means neither — so a second pass over the catalogue
shows exactly what is on disk. The **last column** is what a demo costs you
before you commit to it: the download size, `GPU` (required) or `GPU?`
(optional), plus `git-lfs`, `kaggle`, or `manual` when it needs Git-LFS data,
Kaggle credentials, or a file you supply.

#### The `luxar demo` commands

| Command | What it does |
|---------|--------------|
| `luxar demo` | Browse the catalogue (same as `luxar demo list`) |
| `luxar demo list -c microscopy` | Filter by category: `synthetic`, `microscopy`, `embeddings`, `photogrammetry`, `astronomy`, `structural`, `networks`, `medical`, `geoscience`, `genomics`, `connectome` |
| `luxar demo list -g gsplats` | Filter by geometry: `points`, `gsplats`, `lines`, `mesh`, `points+lines`, `mixed` |
| `luxar demo info <key\|#>` | Requirements, caches, outputs, and how to run one demo |
| `luxar demo run <key\|#>` | Run by key (e.g. `luxar demo run lorenz`) or by the index shown in the table — keys are stable, indices shift as demos are added |
| `luxar demo run <key> -- ARGS` | Forward arguments to the demo script, e.g. `luxar demo run gsplats_3d_tribolium_embryo -- --recompute --no-serve` |
| `luxar demo run-all` | Build every eligible demo's dataset unattended. Manual/Kaggle-data demos are always skipped; GPU-required demos, downloads over 200 MB, and already-built outputs are skipped by default (`--include-gpu`, `--max-download-mb 0`, `--force` lift these) |
| `luxar demo cache list` | Inventory the demo caches under `~/.cache/luxar/`, with sizes and orphans |
| `luxar demo cache clear <keys>` | Reclaim space — `--all` for everything, `--outputs` to drop generated scenes too, `--dry-run` to preview |

Demos run as subprocesses, so `Ctrl-C` tears down the demo *and* the viewer it
spawned. Every tile in the [Gallery](#gallery) below is one of these demos — pick
a key from `luxar demo` and `luxar demo run <key>` reproduces it locally.

---

## Your First Visualization

The demos show what Luxar can render; this is how you build a scene from your own
data. Define a coordinate system, add geometry, and the compiler writes a
`.luxar.zarr` archive.

```python
import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

# Define coordinate system
dims = Dimensions([
    Dimension("x", unit="um", display=True),
    Dimension("y", unit="um", display=True),
    Dimension("z", unit="um", display=True),
])

# Create and save visualization
with LuxarZarrCompiler("my_data.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Your data as NumPy arrays
    positions = np.random.randn(100_000, 3).astype(np.float32) * 50
    colors = np.random.rand(100_000, 3).astype(np.float32)

    scene.add_points("MyPoints", positions, colors=colors)

# View it — one command starts the data server + viewer and opens the browser:
# luxar serve my_data.luxar.zarr --viewer --open
```

### Viewing and sharing it

```bash
# Serve the scene and open the viewer
luxar serve my_data.luxar.zarr --viewer --open

# Inspect what you compiled
luxar info my_data.luxar.zarr --stats

# Test how it behaves on a slow link
luxar serve my_data.luxar.zarr --profile 3g --viewer

# Package it so a colleague can open it without installing Luxar
# (from a source checkout, run `make build-viewer` once first)
luxar export my_data.luxar.zarr -o my_export/
```

### Hosting a scene on the web

A compiled scene is a folder of static files (or a single `.luxar.zarr.zip`).
Any host that serves static files over HTTPS can publish it; there is no server
code to run. Upload the store, then share
`https://luxarviewer.dev/?src=<URL of the store>`. The host needs two things:

- **CORS.** The viewer page comes from `luxarviewer.dev` while the data comes from
  your host, so the host must send `Access-Control-Allow-Origin` (`*` is fine for
  public data). Without it the scene stays empty and the browser console shows
  requests blocked by CORS policy, not a 404.
- **Byte ranges, for zipped stores only.** A `.luxar.zarr.zip` is read in place with
  `Range` requests, so the host must honour them, allow the `Range` request header
  and expose `Content-Range`, `Content-Length`, `Accept-Ranges` and `ETag`. A directory
  store uses plain GETs and needs only CORS.

Typical options, roughly from most to least convenient for scenes of a few GB:

| Host | Why it works well | Watch out for |
|---|---|---|
| **Cloudflare R2** (what serves the demo corpus at `data.luxarviewer.dev`) | Object storage with no egress fees, a free tier of about 10 GB, public buckets, custom domains, and a CORS policy pasted in the dashboard. Reads are billed per request, so chunk size matters (see below). | Custom domain needs a Cloudflare-managed DNS zone. Check the free tier before publishing terabytes. |
| **Amazon S3 / Google Cloud Storage** | Same static-object model; the S3 CORS JSON in the [Viewer Guide](docs/guides/user/VIEWER_GUIDE.md#your-host-must-allow-cross-origin-reads) applies to S3 and R2 as is, and GCS takes the equivalent through `gsutil cors set`. | Egress is billed per GB, which is the cost that grows with popularity. |
| **GitHub Pages** | Free, versioned, zero setup for small scenes: commit the store to a `gh-pages` branch. CORS and byte ranges work out of the box. | Files above 100 MB are rejected and Git LFS objects are not served, so it suits scenes under a few hundred MB in total. Drop a `.nojekyll` file at the site root, or Jekyll strips the dotfiles a zarr-v2 store and `.luxar-index.json` depend on. |
| **Your lab's web server** (nginx, Apache) | Data stays on infrastructure you control; an nginx snippet is in the [Viewer Guide](docs/guides/user/VIEWER_GUIDE.md#configuration-for-common-hosts). | You add the CORS and range headers yourself; institutional proxies sometimes strip `Range`. |
| **No host at all** | `luxar export scene.luxar.zarr -o out/` writes the viewer plus a stdlib-only `serve.py`; from a source checkout, after `make build-launchers`, `--native macos` gives a double-clickable app ([Distributing scenes](docs/tutorials/distributing_scenes.rst)). | The recipient runs it locally; nothing is shareable as a link. |

An archive of record such as Zenodo is the right place to deposit a scene for
citation; whether it can also serve it to the viewer depends on its CORS and range
headers, so test before linking to it.

Object stores bill and throttle per request, and most generated stores land far
below the 64 KB chunk target (the demo corpus averages 5 KB per file), so re-chunk
before publishing with `luxar optimize` and pick the profile by access pattern:
`local` (64 KB) when the viewer will slice into a large node, `hosting` or
`archive` when it loads the node whole. Re-chunking
one demo to the 64 KB profile cut a cold load from 9,390 requests to 2,348. The
re-chunked store carries a new content hash, so publish it under a new URL prefix
rather than over the old one, or warm viewer caches will serve stale chunks; the
profile guidance is in the [CLI Reference](docs/guides/user/CLI_REFERENCE.md#luxar-optimize).
To verify a host, request one chunk with
`curl -sI -H "Origin: https://luxarviewer.dev" <URL>` and look for the
`access-control-allow-origin` header; for a zipped store, make a ranged GET,
`curl -s -o /dev/null -D - -H "Range: bytes=0-0" <URL>`, and expect a `206`
(a HEAD may legitimately answer `200`). This is how the demo corpus is hosted: a
public bucket of directory stores behind a CORS policy, with its full setup,
checks and failure modes in the
[Demo Site Runbook](docs/guides/developer/DEMO_SITE_RUNBOOK.md). A folder of
scenes on a static host becomes browsable in the viewer's Dataset Browser through
a hand-written `.luxar-index.json`
([Viewer Guide](docs/guides/user/VIEWER_GUIDE.md#directory-listings-and-luxar-indexjson)),
and [Distributing scenes](docs/tutorials/distributing_scenes.rst) covers exports,
native bundles and sharing across operating systems.

From here: [Geometry Types](#geometry-types) for points, lines, splats, and meshes;
[n-Dimensional Visualization](#n-dimensional-visualization) for 4D and beyond; and
[Volume Rendering](#volume-rendering-with-gaussian-splats) if your data is an image
volume rather than a point set.

---

## Gallery

A cross-section of Luxar's built-in demos — all four geometry types (**Points**, **Lines**, **Gaussian Splats**, **Mesh**) across real scientific datasets. Each tile is an animated preview: **click the image** for the full-resolution video, or **click the title** to open that exact scene live and interactive in your browser at [demos.luxarviewer.dev](https://demos.luxarviewer.dev). Media are generated by the gallery harness (`make generate-gallery`): square framing, auto-exposed on a black background, gentle orbit turntables.

Each data tile carries a short credit; full citations, licences and links are in
[Acknowledgments → Datasets & scientific data](#datasets--scientific-data). Tiles
with no credit are synthetic — generated by Luxar itself, with no upstream source.

> **These are videos. The real thing is interactive:
> [demos.luxarviewer.dev](https://demos.luxarviewer.dev)** hosts the published
> demo corpus as live scenes you can orbit, slice and navigate in the browser —
> the same compiled archives this README describes, streamed from object storage.

### Gaussian splats — microscopy, medical & astronomy

| [![Cells3D — multichannel fluorescence](https://data.luxarviewer.dev/media/99b6a58b7ab89e70.webp)](https://data.luxarviewer.dev/media/cc4e61c71ee45989.webm) | [![Mouse Blastocyst — multichannel nuclei](https://data.luxarviewer.dev/media/68a7d25d2fcaa240.webp)](https://data.luxarviewer.dev/media/69c59ecd170fae0b.webm) | [![Zebrafish Neuromast — 4D timelapse](https://data.luxarviewer.dev/media/7d6cc6a57c5a5584.webp)](https://data.luxarviewer.dev/media/4e27c8ba7018ad0a.webm) |
|:--:|:--:|:--:|
| **[Cells3D](https://demos.luxarviewer.dev/d/gsplats_3d_cells3d_multichannel)**<br>multichannel fluorescence<br><sub>Allen Institute for Cell Science</sub> | **[Mouse Blastocyst](https://demos.luxarviewer.dev/d/gsplats_3d_blastocyst_multichannel)**<br>Lamin B1 + DAPI<br><sub>Blin et al. 2019</sub> | **[Zebrafish Neuromast](https://demos.luxarviewer.dev/d/gsplats_4d_neuromast_2ch)**<br>4D timelapse<br><sub>Jacobo lab, CZ Biohub</sub> |
| [![Tribolium Embryo — light-sheet](https://data.luxarviewer.dev/media/e42b46b168783747.webp)](https://data.luxarviewer.dev/media/535f397a960bd108.webm) | [![CT Anatomy Atlas — TotalSegmentator](https://data.luxarviewer.dev/media/9334f8298377e7cd.webp)](https://data.luxarviewer.dev/media/77c1fed5db8e70af.webm) | [![Milky Way Dust — galactic dust clouds](https://data.luxarviewer.dev/media/32beb7d243228050.webp)](https://data.luxarviewer.dev/media/f1921c6b3c604bb5.webm) |
| **[Tribolium Embryo](https://demos.luxarviewer.dev/d/gsplats_3d_tribolium_embryo)**<br>light-sheet<br><sub>Barry et al. 2022</sub> | **[CT Anatomy Atlas](https://demos.luxarviewer.dev/d/gsplats_3d_ct_totalsegmentator)**<br>TotalSegmentator<br><sub>Wasserthal et al. 2023</sub> | **[Milky Way Dust](https://demos.luxarviewer.dev/d/gsplats_3d_milky_way_dust)**<br>galactic dust clouds<br><sub>Leike et al. 2020</sub> |

### The living cell & genome

| [![FlyWire Connectome — fly-brain neurons](https://data.luxarviewer.dev/media/8a4163b0ec90b5d6.webp)](https://data.luxarviewer.dev/media/6323bf2395b9e871.webm) | [![Single-Cell 3D Genome — Dip-C chromosomes](https://data.luxarviewer.dev/media/54580524ec54c072.webp)](https://data.luxarviewer.dev/media/77ced36306375c34.webm) | [![C. elegans — 4D nuclei-tracking timelapse](https://data.luxarviewer.dev/media/8e7ade5dbe3f2a10.webp)](https://data.luxarviewer.dev/media/32357e2574e1ee44.webm) |
|:--:|:--:|:--:|
| **[FlyWire Connectome](https://demos.luxarviewer.dev/d/flywire_connectome)**<br>fly-brain neurons<br><sub>Dorkenwald et al. 2024</sub> | **[Single-Cell 3D Genome](https://demos.luxarviewer.dev/d/dipc_3d_genome)**<br>Dip-C chromosomes<br><sub>Tan et al. 2018</sub> | **[C. elegans](https://demos.luxarviewer.dev/d/gsplats_4d_celegans_tracking)**<br>4D nuclei-tracking timelapse<br><sub>Santella et al. 2022</sub> |
| [![ATP Synthase — molecular machine](https://data.luxarviewer.dev/media/d9d1994630f8b126.webp)](https://data.luxarviewer.dev/media/955f21cfd7c54c43.webm) | [![Tabula Sapiens — human cell atlas](https://data.luxarviewer.dev/media/c2c0a1e64e13d76e.webp)](https://data.luxarviewer.dev/media/f7ac0cc31aff917d.webm) | [![Human Multiome — ATAC-peak UMAP](https://data.luxarviewer.dev/media/0537e5927e4f7813.webp)](https://data.luxarviewer.dev/media/fde5da5b8e088552.webm) |
| **[ATP Synthase](https://demos.luxarviewer.dev/d/atp_synthase)**<br>molecular machine<br><sub>Zhou et al. 2015</sub> | **[Tabula Sapiens](https://demos.luxarviewer.dev/d/tabula_sapiens)**<br>human cell atlas<br><sub>Tabula Sapiens Consortium 2022</sub> | **[Human Multiome](https://demos.luxarviewer.dev/d/human_multiome_peak_umap)**<br>ATAC-peak UMAP<br><sub>Domcke et al. 2020</sub> |
| [![Human White-Matter Tractography — 87 dMRI tracts](https://data.luxarviewer.dev/media/1a6726e9b58e4c8a.webp)](https://data.luxarviewer.dev/media/a5ab670e3390f45b.webm) | [![Cells3D Isosurfaces — shaded triangle meshes](https://data.luxarviewer.dev/media/a1c3a92470d26a69.webp)](https://data.luxarviewer.dev/media/63c090e9c9e15cfd.webm) | |
| **[White-Matter Tractography](https://demos.luxarviewer.dev/d/dmri_tractography)**<br>87 human dMRI tracts<br><sub>Yeh 2022</sub> | **[Cells3D Isosurfaces](https://demos.luxarviewer.dev/d/mesh_isosurface_cells3d)**<br>shaded membrane + nuclei surfaces<br><sub>Allen Institute for Cell Science</sub> | |

### Earth & geoscience

| [![Global Earthquakes — USGS on the globe](https://data.luxarviewer.dev/media/dc4b3e44dcac6de8.webp)](https://data.luxarviewer.dev/media/d05c55e7ede7ca65.webm) | [![Rivers of Earth — topography + river networks](https://data.luxarviewer.dev/media/62f7ac63f2a7055a.webp)](https://data.luxarviewer.dev/media/acc224755930426d.webm) |
|:--:|:--:|
| **[Global Earthquakes](https://demos.luxarviewer.dev/d/earthquakes)**<br>USGS on the globe<br><sub>USGS catalog; NASA Blue Marble</sub> | **[Rivers of Earth](https://demos.luxarviewer.dev/d/global_rivers_earth)**<br>topography + river networks<br><sub>HydroSHEDS + NOAA NCEI</sub> |

### Networks & embeddings

| [![CAIDA — Internet AS topology](https://data.luxarviewer.dev/media/c3370e8b29dd5522.webp)](https://data.luxarviewer.dev/media/4d397bcf0404748e.webm) | [![HuRI — human interactome](https://data.luxarviewer.dev/media/8feb73b6cd217f65.webp)](https://data.luxarviewer.dev/media/129fa8a15eec01c1.webm) | [![Protein Landscape — CAFA5 embeddings](https://data.luxarviewer.dev/media/397b1162496bb4e9.webp)](https://data.luxarviewer.dev/media/44971e188a905b8c.webm) |
|:--:|:--:|:--:|
| **[CAIDA](https://demos.luxarviewer.dev/d/caida_as_topology)**<br>Internet AS topology<br><sub>CAIDA, UC San Diego</sub> | **[HuRI](https://demos.luxarviewer.dev/d/huri_interactome)**<br>human interactome<br><sub>Luck et al. 2020</sub> | **[Protein Landscape](https://demos.luxarviewer.dev/d/protein_landscape)**<br>CAFA5 embeddings<br><sub>CAFA5; Elnaggar et al. 2022</sub> |
| [![Spotify — audio-feature embedding](https://data.luxarviewer.dev/media/531b5a277960ccae.webp)](https://data.luxarviewer.dev/media/c9e0e6b051b4e28f.webm) | [![Zebrahub — multiome embedding](https://data.luxarviewer.dev/media/773fce285f464bbc.webp)](https://data.luxarviewer.dev/media/3560cd22b27a75da.webm) |
| **[Spotify](https://demos.luxarviewer.dev/d/spotify_tracks)**<br>audio-feature embedding<br><sub>Spotify Web API</sub> | **[Zebrahub](https://demos.luxarviewer.dev/d/zebrahub_multiome)**<br>multiome embedding<br><sub>Kim et al. 2024</sub> |

### Synthetic & mathematical

| [![Lorenz Attractor — chaotic dynamics](https://data.luxarviewer.dev/media/59b7714301144d07.webp)](https://data.luxarviewer.dev/media/73917f691e02cf9a.webm) | [![Rainbow Sphere — HDR Fibonacci sphere](https://data.luxarviewer.dev/media/75006268b34f0730.webp)](https://data.luxarviewer.dev/media/6962f8d833a8f20b.webm) | [![Quantum Orbitals — hydrogen 2p_z](https://data.luxarviewer.dev/media/4c98b8733cfe67c3.webp)](https://data.luxarviewer.dev/media/9adc484ae6b8ee3f.webm) |
|:--:|:--:|:--:|
| **[Lorenz Attractor](https://demos.luxarviewer.dev/d/lorenz)**<br>chaotic dynamics | **[Rainbow Sphere](https://demos.luxarviewer.dev/d/rainbow_sphere)**<br>HDR Fibonacci sphere | **[Quantum Orbitals](https://demos.luxarviewer.dev/d/quantum_orbitals)**<br>hydrogen 2p_z |
| [![Spiral Galaxy — barred multi-armed disk](https://data.luxarviewer.dev/media/c412a7b78b54f22a.webp)](https://data.luxarviewer.dev/media/8c78f4437882452c.webm) | [![Galaxy Simulation — density-wave spiral](https://data.luxarviewer.dev/media/9770a88b525190b5.webp)](https://data.luxarviewer.dev/media/6dea0c44152aa88f.webm) | [![Hilbert Curve — 3D space-filling](https://data.luxarviewer.dev/media/fd6783ee4a5019b8.webp)](https://data.luxarviewer.dev/media/ac0ca6cb3852e9c1.webm) |
| **[Spiral Galaxy](https://demos.luxarviewer.dev/d/spiral_galaxy)**<br>barred multi-armed disk | **[Galaxy Simulation](https://demos.luxarviewer.dev/d/galaxy_simulation)**<br>density-wave spiral | **[Hilbert Curve](https://demos.luxarviewer.dev/d/hilbert_curve_3d)**<br>3D space-filling |
| [![Particle Collision — physics event](https://data.luxarviewer.dev/media/f69a603488d5897e.webp)](https://data.luxarviewer.dev/media/5041a2035cbed16d.webm) | [![Ocean — bioluminescent jellyfish](https://data.luxarviewer.dev/media/da7d144cab71ac33.webp)](https://data.luxarviewer.dev/media/ca062349f7ed8b8d.webm) |
| **[Particle Collision](https://demos.luxarviewer.dev/d/collision)**<br>physics event | **[Ocean](https://demos.luxarviewer.dev/d/ocean)**<br>bioluminescent jellyfish |

> The full curated set lives in `scripts/gallery/manifest.json`. A few very large point clouds (the DESI cosmic web) and heavy volumes render too slowly under headless software-GL to include as videos here; `make generate-gallery` on a machine with a GPU regenerates them. The Gaia 3M-star tile is absent for a different reason: the catalogue is CC BY-NC, so it is not shipped, and the demo reads it from `~/.cache/luxar/milky_way_gaia_3m/milky_way_gaia_3m.zarr.zip`, which you build yourself (`luxar demo run gaia_milky_way -- --build-catalog` queries the ESA archive). The harness reports such a demo as `manual-data` instead of failing the whole build.
>
> These demos visualize openly-shared scientific datasets — see [Acknowledgments → Datasets & scientific data](#datasets--scientific-data) for full sources and citations.

### Reproducing these locally

Each tile is a bundled demo — run it by key with the
[`luxar demo` commands](#the-luxar-demo-commands) shown in Quick Start:

```bash
luxar demo                          # Find the key for any tile above
luxar demo run galaxy_simulation    # Build and view one
luxar demo run-all                  # Build every eligible demo (skips manual/Kaggle, GPU-required, >200MB, already-built)

make generate-gallery               # Stage replacement stills + orbit videos for review and publishing
```

**Prefer to just look?** Nothing to install — click any tile title above, or browse all of them at **[demos.luxarviewer.dev](https://demos.luxarviewer.dev)**, and orbit, slice and navigate the same scenes in your browser.

---

## Volume Rendering with Gaussian Splats

Classical volume rendering ships **voxels**: to display a 3D image you upload the
grid to the GPU as a 3D texture and ray-march it. That model does not survive
contact with modern microscopy — a single light-sheet timepoint can be several
gigabytes, a timelapse multiplies that by the number of timepoints, and none of it
fits through a browser.

Luxar takes the other route. It **fits** the volume with a sparse mixture of
oriented 3D Gaussians and renders those instead. The size of the representation
tracks how much *structure* a sample contains rather than the grid it happened to
be sampled on — so empty space costs nothing, and what is left is small enough to
stream:

| Dataset | Source volume | Fitted representation |
|---------|---------------|-----------------------|
| **Tribolium embryo** — light-sheet, 1 timepoint | 965 × 1871 × 991 = 1.8 G voxels (3.3 GB as TIFF) | 296,559 splats · **2.0 MB** |
| **C. elegans embryo** — confocal, 400 timepoints | 400 × 41 × 512 × 512 = 4.3 G voxels | 5.64M splats · **81 MB** (about 200 KB per timepoint) |

The *C. elegans* fit is fetched from the permissively licensed Zenodo demo record
(see [Where the demo data lives](#where-the-demo-data-lives)); the Tribolium fit is
produced locally because its source is not redistributable.
Both use full source resolution, and the single-file Tribolium fit works out to
about 7 bytes per splat on disk. They then
render in any WebGL2 browser: no 3D textures, no ray-marching, and no CUDA
on the viewing machine.

<p align="center">
  <img src="https://data.luxarviewer.dev/media/3f4df30e7f4b4a4a.webp" alt="From a light-sheet stack to splats: the raw volume in napari, luxar gsplat fit converging in a terminal with its PSNR curve, a short Python script, and the fitted embryo opening in the browser" width="100%">
  <br>
  <sub>A raw <em>Drosophila</em> light-sheet stack in napari, <code>luxar gsplat fit</code> converging (training PSNR plotted as it runs), a dozen lines of Python to compile the scene, and the result opening in the browser. Excerpts of <a href="https://demos.luxarviewer.dev/v/01/">Supplementary Video 1</a>.</sub>
</p>

This is lossy, so fidelity is measured rather than asserted. Across a 17-volume
microscopy benchmark (4–107 M voxels; spinning-disk, confocal, light-sheet and iSIM),
fits at each volume's cross-validated splat budget land between 26 and 67 dB PSNR at
6–340× compression (median 99×), measured as the bytes of the source volume at its
stored bit depth (uint16 for most, uint8 or float32 where that is how the source is
stored) over the bytes of the stored splat archive (manuscript in preparation). The
single-volume Tribolium fit above is a gigavoxel source well beyond the benchmark's
per-volume range, so its compression exceeds the quoted ceiling; the *C. elegans* figure
instead aggregates 400 timepoints of 10.7 M voxels each. `luxar gsplat compare` reports
PSNR/SSIM/MSE for your own data.

### Emission and absorption, not just glow

The `volumetric` blending mode implements the standard emission–absorption model of
direct volume rendering (Max 1995) in closed form for Gaussians — the same
radiative transfer NeRF composites with. One per-layer knob, absorption κ, morphs
the render continuously:

| κ | Look | Good for |
|---|------|----------|
| `0` | pure emission — bit-identical to `additive` | sparse fluorescence, X-ray-like projection |
| small | attenuated projection — near structure pops, occluded structure dims | depth cueing in dense timelapses |
| large | dense smoke- or ink-like medium | opaque tissue, anatomy |

Because fitted amplitudes are background-relative image intensities — proportional
to the detected fluorescence after floor subtraction and normalisation, not a
calibrated fluorophore concentration — rather than learned opacities, κ is still
interpretable as an effective turbidity of the sample instead of being an arbitrary
rendering constant. Absorption is also orientation-consistent —
an elongated splat seen end-on absorbs more than the same splat seen side-on, which
a stored per-splat opacity cannot express. Points, lines and Gaussian splats all render
the same physics, on both the WebGL/GLSL and WebGPU/TSL backends.

```bash
luxar gsplat convert fit.gsplats.zarr scene.luxar.zarr \
    --blending-mode volumetric --absorption 4 --colormap plasma --tone-mapping ACES
```

The other modes cover the rest of the classical spectrum: `additive`/`luminous`
(pure emission — `additive` is the default), `max` (maximum-intensity projection), and
`normal`/`opaque` (surfaces). Full derivation and invariants in the
[Volumetric Blending Spec](docs/guides/specs/VOLUMETRIC_BLENDING_SPEC.md).

### Timelapses are one dataset, not a folder of frames

Each timepoint is fitted in 3D and the results are stacked onto a time axis: every
splat gains a time coordinate and a corresponding covariance entry (width zero for a
discrete axis, or a real extent if you want temporal spread), so a complete 4D
acquisition — or 5D, adding channel or camera — lives in a **single**
`.gsplats.zarr`. The viewer's time slider is then ordinary nD slice navigation, and
because splats are stored timepoint-major, each frame's splats are contiguous on disk:
scrubbing fetches the chunks of the current frame and nothing else (at most the one
chunk on a frame boundary carries a few splats of its neighbour).

Coarsening treats the time and channel axes as **hard barriers**. Coarse splats are
never merged across them and mass is conserved per barrier group, so a timepoint
keeps its exact brightness at every level of detail and scrubbing never smears one
frame into the next.

Fitting a whole timelapse is one command, on whatever hardware you have:

```bash
# Every GPU in the box, planned over T×C, resumable
luxar gsplat batch-fit run movie.zarr out/ --gpus auto

# Or a Slurm array job on a cluster
luxar gsplat batch-fit submit movie.zarr out/ -p gpu --tiling content --cal cal.json
```

Both plan tiles once across all timepoints and channels, fit each tile as an
independent task, then **stream-merge** the results — peak memory is one tile
region, never the whole movie. `status`, `validate`, `merge`, and `cancel` are
shared by both backends. Pass `--merge-recipe` to give each spatial part its own
LOD ladder as it streams.

### Scaling: pick a topology, stream the rest

`luxar gsplat lod --recipe` turns a fitted dataset into a level-of-detail topology.
The recipes are named by intent and ordered by dataset scale:

| Recipe | Structure | Use when |
|--------|-----------|----------|
| `flat` | one bare leaf | tiny data, debugging |
| `stream` | one leaf + progressive ladder | small data, fast first paint |
| `levels` | coarse→fine replacement levels | zooming across scales |
| `tiles` | spatial tiles, each with its own ladder | large scene at one scale |
| `overview` | instant coarse overview, fine tiles on zoom | huge scene, "see everything first" |
| `adaptive` | tiles where every tile picks its own level | largest scenes, locally adaptive |

<p align="center">
  <img src="https://data.luxarviewer.dev/media/8b1592c0e9cc72fa.webp" alt="The six LOD recipes built from one Tribolium fit, shown side by side as six embryos, then up close: the levels column swapping from its coarse level to its finest as the camera approaches, the tiles column, and the adaptive column" width="100%">
  <br>
  <sub>One <em>Tribolium</em> fit, six topologies side by side (<code>flat</code>, <code>stream</code>, <code>levels</code>, <code>tiles</code>, <code>overview</code>, <code>adaptive</code>), each column coloured by the part or level it is drawn from; the data-loading monitor counts what is resident as the camera dollies in. <a href="https://demos.luxarviewer.dev/d/gsplats_recipes_tribolium">Open the demo</a> or watch <a href="https://demos.luxarviewer.dev/v/07/">Supplementary Video 7</a>.</sub>
</p>

Apart from `flat`, every recipe carries a progressive streaming ladder by default:
splats are reordered so that early prefixes carry as much of the signal as possible,
which means the first chunk to arrive is already a meaningful picture and later
chunks only refine it. Where levels replace each other, the viewer picks between
them by screen occupancy: each level's `coverage_fraction` is a literal
**screen-area fraction** — the node's projected bounding-box rect area over the
viewport area — and the ladder is derived by halving it, so the finest level
shows while the object occupies at least half the screen and every halving of
occupied area steps one level coarser. (`adaptive` and `overview` are
partition-bound and anchor one step higher, at area `1.0`.) Being a viewport
fraction rather than an absolute pixel count, the metric needs no per-resolution
or DPI tuning, though occupancy does move with viewport aspect.

The canonical end-to-end pipeline is three commands:

```bash
# 1. Choose the splat budget K* by blind-spot cross-validation
luxar gsplat cal volume.tiff cal.json

# 2. Fit at K*
luxar gsplat fit volume.tiff fit.gsplats.zarr --seeds <K*>

# 3. Build the streaming topology
luxar gsplat lod fit.gsplats.zarr scene.gsplats.zarr --recipe stream
```

Step 1 earns its place: `cal` sweeps K, finds where *held-out* PSNR peaks, and
reports the dataset's noise floor — so the splat count is picked by
cross-validation against the data rather than by guesswork.

### Photogrammetric splats, too

The same renderer reads classical 3D-Gaussian-splatting captures. `luxar gsplat
import` auto-sniffs INRIA `.ply`, antimatter15 `.splat`, Niantic/Scaniverse `.spz`,
SuperSplat compressed `.ply`, and PlayCanvas SOG, then feeds them through the same
LOD and streaming path — the largest interop demo is a 13.6M-Gaussian aerial city
scene. `luxar gsplat export` writes INRIA PLY back out.

See the [Gaussian Splatting Guide](packages/luxar/src/luxar/gsplats/README.md) for
the fitting model, calibration, and LOD algorithms in full.

---

## Geometry Types

### Points

Render collections of nD points as soft-edged spheres.

```python
scene.add_points(
    "ParticleCloud",
    positions,           # (N, D) float32 - nD coordinates
    colors=colors,       # (N, 3) float32 - RGB (0-1, HDR supported)
    radii=radii,         # (N,) float32 - per-point size
    sharpness=sharpness, # (N,) float32 - edge falloff (0-1, normalized)
    opacity=0.8,         # Global opacity
    # "additive" (default), "volumetric", "normal", "max", "opaque", "luminous"
    blending_mode="additive",
)
```

### Lines

Connected line segments with per-vertex attributes.

```python
scene.add_lines(
    "Branches",
    vertices,      # (N, D) float32 - nD vertex positions
    widths=widths, # (N,) float32 - per-vertex width
    colors=colors, # (N, 3) float32 - per-vertex colors
)
```

### Gaussian Splats

Oriented Gaussian functions — Luxar's volume-rendering primitive. Fitting needs the
`gsplats` extra (`pip install -e ".[gsplats]"` from a checkout, `pip install
"luxar[gsplats]"` once released); viewing does not.

```python
from luxar.gsplats import fit_gaussian_splats

# Fit splats to your volume
result = fit_gaussian_splats(volume, n_iters=1000)

# Add to scene, rendered as an absorbing medium
scene.add_gsplats_from_data(
    "Reconstruction", result, blending_mode="volumetric", absorption=4.0
)
```

See [Volume Rendering with Gaussian Splats](#volume-rendering-with-gaussian-splats)
for the scaling and timelapse story, or the
[Gaussian Splatting Guide](packages/luxar/src/luxar/gsplats/README.md) for the
fitting model in detail.

### Mesh

Triangle surfaces — isosurfaces, segmentation boundaries, cortical and organ
meshes. The other three primitives are soft and emissive; a mesh is the one
*connected, shaded* type, lit by a view-anchored offset key so shape reads from
shading rather than from density.

```python
scene.add_mesh(
    "Nuclei",
    vertices,               # (V, D) float32 - nD vertex positions
    faces,                  # (F, 3) uint32  - triangle vertex indices
    normals=normals,        # (V, 3) float32 - optional, needs normal_dims
    normal_dims=[0, 1, 2],  # which three dims the normals describe
    colors=colors,          # (V, 3|4) - optional per-vertex RGB(A)
)
```

Two differences from the other three worth knowing up front. A mesh has **no
per-element size** — a triangle's extent comes from its own vertices, so it adds
no padding to scene bounds. And it renders with `opaque` blending by default
rather than `additive`, which is what makes it depth-correct without sorting.
That default is applied at load time rather than written into the node, so an
unset `blending_mode` can still be inherited from an ancestor group.

Normals are optional: omit them and the shader derives a flat per-face normal
from screen-space derivatives. `normal_dims` is required whenever you *do* pass
them, because in nD there is no implicit "first three dimensions".

Already have a mesh **file**? Skip the Python entirely — `luxar mesh import`
reads PLY, OBJ, STL, VTP and glTF/GLB with no extra dependencies, and a
directory of `T<number>`-indexed files stacks into a mesh timelapse:

```bash
luxar mesh import bunny.ply bunny.luxar.zarr
luxar mesh import frames/ frames.luxar.zarr --pattern '*.ply'
luxar mesh lod bunny.luxar.zarr bunny_lod.luxar.zarr -L 4   # coarse levels
luxar serve bunny_lod.luxar.zarr --viewer
```

Spatial partitioning **is** supported — `add_mesh(partition=…)` splits a large
surface into frustum-cullable parts, though each part still loads whole.
Substitutive LOD (`add_mesh(substitutive_lod=…)`, decimated by
`luxar.mesh.decimate`) and a spatially coherent *reveal* additive ladder
(`add_mesh(additive_lod=…)`, `method="radial"`) both ship. What is still
refused — and neither degrades silently — is an additive ladder over an
*arbitrary* order (a prefix of an arbitrarily ordered index buffer is a holed
surface, not a coarser one) and `volumetric` blending: `add_mesh` raises on
both, and a `volumetric` that reaches the viewer by *inheritance* from an
ancestor group logs a warning naming the node and falls back to `opaque`. See
[the mesh spec](docs/specs/MESH_NODE_SPEC.md) §9 for why, per exclusion.

---

## n-Dimensional Visualization

Luxar natively supports datasets with more than 3 dimensions.

### Defining Dimensions

```python
dims = Dimensions([
    # Displayed dimensions (shown in 3D viewer)
    Dimension("x", unit="um", range=(-100, 100), display=True),
    Dimension("y", unit="um", range=(-100, 100), display=True),
    Dimension("z", unit="um", range=(-50, 50), display=True),

    # Non-displayed dimensions (navigated via sliders)
    Dimension("time", unit="s", range=(0, 60), step=1.0, display=False),
    Dimension("channel", unit="", categories=["DAPI", "GFP", "mCherry"], display=False),
])
```

### How nD Slicing Works

Points in nD space are treated as **hyperspheres**. When viewing a 3D slice:

1. Points whose hypersphere intersects the current hyperplane are visible
2. Larger radius = visible across more slices only for dimensions declared `spatial=True` (non-displayed dimensions default to non-spatial)
3. For those spatial dimensions, effective radius shrinks with distance from slice: `r_eff = sqrt(r² - d²)`

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `1-9` | Select a non-displayed dimension to navigate |
| `[` / `]` | Step backward/forward in selected dimension |
| `N` | Toggle dimension panel |

### Per-Axis Transforms

The 4×4 `transform` moves geometry through the three displayed dimensions. The
separate `nd_transform` attribute does the same for the *non-displayed* ones — so
two datasets recorded on different clocks, sampling rates, or channel orders can be
aligned in one scene instead of being resampled first.

Each axis takes the operation its domain allows: continuous and discrete axes take
an affine `scale`/`offset`, categorical axes take a `permutation`. Like spatial
transforms, they compose down the scene graph.

```python
group = scene.add_group(
    "DatasetB",
    transform=transforms.translate(10, 0, 0),      # displayed dims (4x4)
    nd_transform={
        "time": {"scale": 0.001, "offset": 50.0},  # ms → s, shifted
        "channel": {"permutation": [2, 1, 0]},     # reorder channels
    },
)
group.add_points("cells", positions_5d)            # children inherit it
```

The viewer applies these by inverse-transforming the *query* — the slice position
and tolerance — from world to local space once per view change, rather than
transforming millions of element coordinates. See the
[nD Transforms Spec](docs/guides/specs/ND_TRANSFORMS_SPEC.md) for the domain rules
and composition semantics.

---

## Viewer Controls

### Navigation Modes

| Key | Action |
|-----|--------|
| `V` | Cycle control mode (Orbit / Fly / Ortho) |
| `F` | Recenter camera on scene |
| `Space` | Toggle fullscreen |

### Orbit Mode (default)

| Input | Action |
|-------|--------|
| Left drag / right drag | Rotate around the scene / pan. On macOS the default is left = rotate, right = pan ("natural drag"); elsewhere the reverse. Swap it in the Navigation panel (right-click, or long-press, the Navigation button in the rail) |
| Scroll | Zoom |
| Shift + scroll | Roll around the viewing axis |
| Ctrl/Cmd + scroll | Change field of view |

### Fly Mode

| Input | Action |
|-------|--------|
| `W/S` | Forward/backward |
| `A/D` | Strafe left/right |
| `Alt+W/S` | Up/down |
| `Q/E` or Shift + scroll | Roll around the viewing axis |
| Arrows | Look around |
| `I` | Toggle inertia |

### Interface Panels

| Key | Panel |
|-----|-------|
| `R` | Rendering controls (bloom, exposure, AA) |
| `N` | nD dimension sliders |
| `P` | Performance statistics |
| `H` | Help overlay |
| `O` | Dataset browser |
| `Ctrl+L` | Debug console |

---

## Transforms

Luxar provides a full transform system for hierarchical scene organization.

```python
from luxar import transforms

# Basic transforms
t = transforms.translate(10, 0, 0)
r = transforms.rotate_z(45)
s = transforms.scale(2, 2, 2)

# Compose (applied left-to-right)
combined = transforms.compose(t, r, s)

# Apply to groups
group = scene.add_group("Cluster", transform=combined)
scene.add_points("Points", positions, parent=group)

# Hierarchical transforms
parent = scene.add_group("Robot")
parent.transform = transforms.translate(100, 0, 0)

arm = parent.add_group("Arm")
arm.transform = transforms.rotate_y(45)  # Relative to parent
```

---

## Data Format

Luxar uses Zarr for chunked, compressed storage optimized for streaming. New
stores use Zarr format 3 by default:

```
scene.luxar.zarr/
├── zarr.json               # Scene attributes and consolidated metadata
└── node_name/
    ├── zarr.json           # Node attributes (type, transform, rendering)
    ├── positions/          # (N, D) coordinates (quantised on disk, decoded as float32)
    ├── colors/             # (N, 3) RGB values
    ├── radii/              # (N,) point sizes
    └── chunk_bounds/       # Spatial index for efficient queries
```

See the [format compatibility notes](docs/guides/user/LUXAR_ZARR_FORMAT.md#compatibility-notes)
for the format-2 container layout.

### Performance Characteristics

Measured on an **NVIDIA RTX 3070 at 1280×720**, adaptive DPR pinned to 1.0 for
measurement, at the reference primitive size (4 px for points and splats, 1.5 px for
lines), with the default HDR pipeline. Each cell is the latency of one synchronised
render-and-readback call, in milliseconds; the vsync budget at 60 FPS is 16.7 ms:

| Elements | Lines | Points | Gaussian splats |
|----------|-------|--------|-----------------|
| 100K | 1.21 | 1.23 | 1.72 |
| 1M | 3.31 | 4.98 | 8.64 |
| 10M | — | 18.8 | 19.4 |

At ten million elements the call runs 12–16% over the budget; the viewer's adaptive
DPR (on by default, and pinned for these measurements) buys the frame rate back by
downscaling the render buffer. Typical scenes are lighter than these synthetic
sweeps: the fifteen demo scenes of the same study all hold 60 FPS, the heaviest, a
2.2M-splat time-lapse frame, in 5.6 ms. Whole-slide and other very large splat
scenes are overdraw-bound at full resolution, and the projected-density guard (on
by default) thins them: before it shipped, the 29.6M-splat CMU-1 slide measured about
108 ms per frame at DPR 1.0 on an Apple M4 Max at a 1600×1000 canvas
([viewer performance audit](docs/guides/developer/VIEWER_PERFORMANCE_AUDIT_2026_09.md)). Frame rate
is GPU-, resolution- and geometry-dependent, so treat
these as one reference point rather than a guarantee. Load time is dominated by
transfer and decode, so it tracks your link and cache state rather than element
count alone.

### Spatial Indexing

Data is reordered using space-filling curves (Hilbert/Morton) for:
- **Spatial locality**: Nearby elements stored together
- **Efficient queries**: Only load chunks intersecting current view
- **Progressive loading**: Stream data as needed

See [Zarr Format Specification](docs/guides/user/LUXAR_ZARR_FORMAT.md) for complete details.

---

## Architecture

Luxar is two code bases that never import each other. The Python package
authors, fits and compiles; the TypeScript viewer streams and renders; the
`.luxar.zarr` archive is the only contract between them. Anything that can
write that archive (today: the Python compiler) feeds anything that can read
it (the hosted viewer, an offline export, an embedded `@luxar/viewer`, a native
launcher).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://data.luxarviewer.dev/media/7be8aedb71f7d2dc.png">
  <img alt="Luxar architecture: the Python package (scene graph, annotations, Gaussian-splat and mesh fitting, compiler and encoding, CLI, remote control, demos), the .luxar.zarr archive in the middle served by any static host, and the TypeScript viewer (LuxarApp, interface, navigation, rendering, compute, streaming, configuration, sound) plus the export folder and native launcher" src="https://data.luxarviewer.dev/media/e6f5d569fc19c1d6.png" width="100%">
</picture>

**Python side.** `core/` is the scene graph a user builds: nodes for the four
geometry types, groups with transforms, `Dimensions` that declare every axis
with its unit and extent, per-node appearance, and annotations (text, image,
video and HTML overlays, sound, story waypoints). `gsplats/` turns image volumes
into that geometry: cross-validated calibration, the fitter with its CUDA and
Apple-MPS kernels, tiled and multi-GPU batch fitting, and the LOD recipes.
`mesh/` imports and decimates surfaces. `io/` is the compiler: it orders every
node along a space-filling curve, chunks it to about 64 KB, writes the nD
spatial index and the LOD ladders, and emits Zarr format 3 (reading both 2 and
3); `encoding/` decides how each attribute is quantised and compressed. `cli/`
exposes all of it, `control/` drives a running viewer from Python, and `demos/`
holds the 90 bundled demos whose data is pinned by digest and fetched on demand.

**The archive.** A directory of small chunk files, or a single zip read by byte
range. It carries its own metadata (dimensions, node tree, chunk bounds, ladder
energies), so no server code is needed: `luxar serve` for a laptop, a lab web
server, object storage, GitHub Pages or an exported folder all serve it the same
way, and the viewer opens it from a URL. The format is specified in
[LUXAR_ZARR_FORMAT.md](docs/guides/user/LUXAR_ZARR_FORMAT.md); the fitted-splat
container in [GSPLATS_ZARR_FORMAT.md](docs/specs/GSPLATS_ZARR_FORMAT.md).

**Viewer side.** From the bottom up: `data/` and `cache/` are the Zarr client,
the spatial queries that turn a view into chunk requests, prefetch, and four
cache tiers (in-memory S-cache, L0, L1 and an OPFS-backed L2 that survives a
reload). `workers/` decode off the main thread and call Rust kernels in `wasm/`
for the nD projection, effective radii and Mahalanobis tests, with a TypeScript
reference implementation for scenes above 16 dimensions. `rendering/` and
`scene/` draw with Three.js on WebGL2 or WebGPU through an HDR pipeline with six
blending modes, choosing levels of detail by projected screen area. `controls/`
and `input/` implement orbit, fly and ortho cameras for mouse, keyboard and
touch; `ui/` and `themes/` are the rail and its panels; `config/` holds the URL
parameters and settings, including the density guard and adaptive resolution;
`audio/` plays sound cued by the hidden dimensions. `core/` ties these into
`LuxarApp`, the object a page embeds and scripts. Two more consumers sit beside
the browser: `luxar export` writes the viewer and the data as an offline folder,
and `packages/luxar-launcher` (Go) wraps that folder as a double-clickable
macOS or Linux app.

---

## Development

### Setup

```bash
make setup-dev     # Complete environment (Node.js, pnpm, Hatch)
make check-deps    # Verify installation
```

### Commands

| Task | Command |
|------|---------|
| Run all tests | `make test-all` |
| Quality checks | `make check-all` |
| Format all code | `make format-all` |
| Clean all artifacts | `make clean-all` |
| Start viewer | `make viewer` |
| Build viewer | `make build-viewer` |
| Run examples | `make run-examples` |
| Generate README images | `make generate-readme-images` |

### Python Development

```bash
hatch run test              # Run tests
hatch run test-cov          # With coverage
hatch run python script.py  # Run in environment
```

### TypeScript Development

```bash
cd packages/luxar-viewer
pnpm dev           # Dev server (port 5173)
pnpm build         # Production build
pnpm test --run    # Unit tests
pnpm test:e2e      # E2E tests (Playwright)
```

### WASM (Optional)

The viewer works without WASM, but Rust acceleration improves performance:

```bash
make install-rust  # Install Rust + wasm-pack
make build-wasm    # Build WASM module
```

---

## API Reference

Full generated reference for Python and TypeScript:
[Luxar documentation](https://royerlab.github.io/luxar/).

### Python API

```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension, transforms

# Create compiler
with LuxarZarrCompiler("output.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Add geometry
    scene.add_points(name, positions, colors=..., radii=..., ...)
    scene.add_lines(name, vertices, widths=..., colors=...)
    scene.add_mesh(name, vertices, faces, normals=..., normal_dims=..., colors=...)
    scene.add_group(name, transform=..., opacity=...)

    # Gaussian splatting (embedding an existing fit needs no extra;
    # producing one — `luxar gsplat fit` — needs the gsplats extra)
    scene.add_gsplats_from_data(name, gsplat_result)
    scene.add_gsplats_from_file(name, "file.gsplats.zarr")
```

### CLI Reference

```bash
luxar demo                             # List / run / manage bundled demos
luxar demo run <key|#> [-- ARGS]       # Run a demo (forwards ARGS to it)
luxar serve PATH [OPTIONS]              # Serve Zarr dataset
luxar viewer [--data PATH] [OPTIONS]    # Serve viewer only or viewer + data
luxar info PATH [--stats]               # Dataset information (--stats also reports the chunk layout)
luxar optimize SRC DST [--profile ...]  # Re-chunk an existing store for streaming (values stay bit-identical)
luxar restamp-lod STORE [--dry-run]     # Re-derive LOD thresholds in place (attrs only)
luxar export SOURCE -o DIR              # Export standalone folder (Python 3 + browser)
luxar export SOURCE -o DIR --native macos|linux-amd64|linux-arm64
                                        # Double-clickable native bundle (.app / portable folder)
                                        # Needs `make build-launchers` FIRST — the bundler looks for the
                                        # host-platform binary in cli/_launchers/ and errors without it
luxar profiles                          # List network simulation profiles
luxar gsplat <subcommand> [OPTIONS]     # Gaussian splatting tools (fit, cal, lod --recipe {flat,stream,levels,tiles,overview,adaptive}, migrate-format, convert, render, merge, ...)
luxar gsplat flatten IN OUT             # Collapse a gsplat tree (LOD/partition) to one flat leaf
luxar mesh import FILE|DIR OUT          # PLY/OBJ/STL/VTP/glTF (a directory stacks a mesh timelapse)
luxar mesh lod IN OUT [-L N]            # Decimated coarse levels, or --recipe reveal
luxar env bake SCENE                    # Bake the scene environment map for physical materials
```

See [`docs/tutorials/distributing_scenes.rst`](docs/tutorials/distributing_scenes.rst) for the full distribution story (folder export, native bundles, sharing across OSes, Gatekeeper handling).

**Serve Options:**
- `--viewer` - Launch viewer alongside server
- `--open` - Open browser automatically
- `--port PORT` - Data server port (default: 8000)
- `--viewer-port PORT` - Viewer port (default: 5173)
- `--profile NAME` - Network simulation profile (3g, 4g, satellite, etc.)
- `--bandwidth VALUE` - Custom bandwidth limit (e.g., "500kbps")
- `--latency VALUE` - Custom latency (e.g., "200ms")

---

## AI Agent Skills

Luxar ships **Agent Skills** — reusable, model-invoked instructions that teach an AI
coding agent (Claude Code, OpenAI Codex, Gemini CLI, Cursor, …) how to drive Luxar.
They use the cross-tool [`SKILL.md`](https://agentskills.io) open standard, so the
same skill works across compatible agents.

They are committed to the repo, so **cloning is the only install step** — agents
auto-discover them. The canonical copies live in [`.agents/skills/`](.agents/skills/)
(the vendor-neutral path) and are symlinked into `.claude/skills/` so Claude Code
finds them too.

| Skill | What it teaches |
|-------|-----------------|
| [`luxar-install`](.agents/skills/luxar-install/SKILL.md) | Install Luxar on a machine — user-install vs full dev setup, per profile: modest laptop (CPU/MPS), NVIDIA-GPU desktop (CUDA extension), and Slurm/HPC cluster (no-sudo bootstrap, `build-cuda SLURM=1`). |
| [`luxar-visualization`](.agents/skills/luxar-visualization/SKILL.md) | Build a `.luxar.zarr` scene from a dataset — Points/Lines/GSplats, Dimensions, transforms, hierarchy, serve/export — grounded in the demos and examples. |
| [`luxar-gsplat-pipeline`](.agents/skills/luxar-gsplat-pipeline/SKILL.md) | Fit Gaussian splats to an nD image: the `cal → fit → lod` pipeline, the full CLI option surface, tiling, the Python fitting API, and adding a gsplat node to a scene. |
| [`luxar-hpc-batch-fit`](.agents/skills/luxar-hpc-batch-fit/SKILL.md) | Fit a whole nD timelapse at scale — local multi-GPU (`batch-fit run`) or Slurm/Bruno (`batch-fit submit`), plus status/validate/merge/cancel and the GPU benchmark profile. |
| [`luxar-gsplat-edit`](.agents/skills/luxar-gsplat-edit/SKILL.md) | Post-fit toolbox on a `.gsplats.zarr`: slice, transform, cull, filter, partition, merge, convert, migrate-format, and inspect (info/render/compare/view/napari). |
| [`luxar-data-loading`](.agents/skills/luxar-data-loading/SKILL.md) | Load an nD image/volume (`.zarr`/OME-Zarr/`.tiff`/`.npy`/`.npz`) — channel/timepoint/array-key selection and `--axes` overrides, with the RAM/axes pitfalls. |
| [`luxar-export`](.agents/skills/luxar-export/SKILL.md) | Package a scene for sharing — a standalone offline folder (viewer + data + `serve.py`) or a native macOS/Linux app bundle. |

**Invoke** — Claude Code: `/<skill-name>` or automatically; Codex: `$<skill-name>`,
`/skills`, or automatically. Implicit selection is driven by each skill's
`description`. See [`.agents/skills/README.md`](.agents/skills/README.md) for the
layout and how to add a new skill.

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Documentation site](https://royerlab.github.io/luxar/)** | Tutorials, guides, format specs, and the generated Python + TypeScript API reference |
| **[Live demo gallery](https://demos.luxarviewer.dev)** | 88 demos as interactive scenes in the browser |
| **[Hosted viewer](https://luxarviewer.dev)** | Open any reachable scene: `luxarviewer.dev/?src=<url>` |
| [Demo Site Runbook](docs/guides/developer/DEMO_SITE_RUNBOOK.md) | How the two sites above are hosted and published |
| [Python Package README](packages/luxar/README.md) | Full Python API documentation |
| [Viewer Guide](docs/guides/user/VIEWER_GUIDE.md) | Navigating a scene: camera, nD slicing, panels, keyboard |
| [CLI Reference](docs/guides/user/CLI_REFERENCE.md) | Every `luxar` command, what it is for and where its guide lives; `--help` lists the flags |
| [CLI package README](packages/luxar/src/luxar/cli/README.md) | Per-command options and runnable examples for the fitting pipeline (`cal`, `fit`, `lod`, `batch-fit`) |
| [Viewer README](packages/luxar-viewer/README.md) | Viewer features and configuration |
| [Zarr Format Spec](docs/guides/user/LUXAR_ZARR_FORMAT.md) | Complete data format specification |
| [GSplats Format Spec](docs/specs/GSPLATS_ZARR_FORMAT.md) | The `.gsplats.zarr` fitted-splat container and its LOD node tree |
| [Format and Migration](docs/guides/user/FORMAT_AND_MIGRATION.md) | Zarr format 2 vs 3, `LUXAR_ZARR_FORMAT`, migrating older stores |
| [Compatibility Policy](docs/guides/user/COMPATIBILITY_POLICY.md) | What stays stable across releases and how breaking changes are announced |
| [Distributing scenes](docs/tutorials/distributing_scenes.rst) | Static hosting, folder exports, native bundles, sharing across OSes |
| [Mesh Spec](docs/specs/MESH_NODE_SPEC.md) | The shaded triangle-mesh node: attributes, LOD, what it refuses and why |
| [HDR Guide](docs/guides/user/HDR_GUIDE.md) | HDR color workflow |
| [Gaussian Splatting](packages/luxar/src/luxar/gsplats/README.md) | n-Dimensional Gaussian fitting |
| [Volumetric Blending Spec](docs/guides/specs/VOLUMETRIC_BLENDING_SPEC.md) | Emission–absorption compositing: the optical model, the κ maths, and the testable invariants |
| [nD Transforms Spec](docs/guides/specs/ND_TRANSFORMS_SPEC.md) | Per-axis transforms on non-displayed dimensions — domains, composition, inverse-query design |
| [Agent Skills](.agents/skills/README.md) | Cross-tool AI agent skills (Claude Code, Codex, …) shipped with Luxar |
| [Build System](docs/guides/developer/BUILD_SYSTEM_SPEC.md) | Development environment setup |
| [Project Statistics](stats/PROJECT_STATS.md) | Codebase size, language mix, test coverage, git activity (see [`project_stats.html`](stats/project_stats.html) for the styled report) |
| [Contributing](CONTRIBUTING.md) | How to contribute |
| [Code of Conduct](CODE_OF_CONDUCT.md) | Contributor Covenant 2.1 |
| [Security Policy](SECURITY.md) | How to report a vulnerability privately |

---

## GPU Acceleration (Optional)

Gaussian splat fitting runs on CPU by default. For much faster fitting (often orders of magnitude, GPU-dependent), install with GPU support:

```bash
# Install gsplats dependencies (PyTorch, scipy, etc.)
pip install -e ".[gsplats]"     # from a checkout; pip install "luxar[gsplats]" once released

# For NVIDIA CUDA acceleration (optional, requires CUDA toolkit + GPU):
make build-cuda
```

**What gets accelerated:**

| Component | CPU | CUDA GPU | Apple Metal (MPS) |
|-----------|-----|----------|-------------------|
| Splat fitting | Supported (slow) | Much faster (often orders of magnitude, GPU-dependent) | Supported |
| NLM denoising | Supported | Faster with `make build-cuda` | Supported (PyTorch backend) |
| Seeding | Supported | Much faster for large volumes (GPU-dependent) | Supported |
| **Viewer rendering** | N/A | N/A | N/A |

> **Note:** The viewer renders through **WebGL2** (or opt-in WebGPU) on your browser's GPU — no CUDA needed.
> CUDA acceleration is only for the Python-side fitting pipeline.
> The CUDA extension (`.so`) is compiled locally for your specific GPU architecture
> and cannot be pre-built or distributed in the wheel.

See `docs/guides/developer/BUILD_SYSTEM_SPEC.md` for HPC/Slurm build instructions.

---

## Troubleshooting

### Common Issues

**White screen in viewer**
- Check browser console for errors
- Prefer the canonical Zarr dataset URL spelling without a trailing slash
- Ensure CORS headers if serving cross-domain (see [Hosting a scene on the web](#hosting-a-scene-on-the-web))

**ImportError: No module named 'luxar'**
```bash
make setup-dev    # Set up environment
# Or: hatch shell  # Activate environment
```

**Poor rendering performance**
- Reduce element count (target <1M elements)
- Lower bloom quality in rendering panel
- Disable MSAA/SSAA, use FXAA instead

**CUDA splatting backend not compiled**
```bash
make check-cuda-deps  # Check what's installed/missing
make build-cuda       # Build for local GPU
# On HPC clusters:
make build-cuda SLURM=1 SLURM_PARTITION=gpu
```
If you see "GPU fitting will use slower PyTorch fallback", fitting still works — just slower.

**nD navigation not working**
- Verify `scene_dimensions` is defined in the scene attributes
- Check dimension count matches position array shape
- Ensure non-displayed dimensions have valid `range` and `step`

### Browser Compatibility

Luxar needs **WebGL 2.0**, which is the default backend. WebGPU is opt-in via
`?renderer=webgpu` and falls back to an internal WebGL2 backend when no adapter
is available. The viewer builds with Vite's `target: 'esnext'` and declares no
`browserslist`, so nothing is downlevelled and no version floor is derived from
the toolchain.

What is verified, by running the E2E smoke subset (13 tests across basic
rendering, viewer initialisation, and geometry types including GSplats) on
2026-09-04, macOS arm64, Playwright's bundled engines:

| Engine   | Smoke subset | Notes                                                |
| -------- | ------------ | ---------------------------------------------------- |
| Chromium | 13/13 pass   | L2 (OPFS) disk cache initialises                      |
| Firefox  | 13/13 pass   | L2 (OPFS) disk cache initialises                      |
| WebKit   | 13/13 pass   | **runs without the L2 disk cache** — the OPFS store's init / write probe fails, so chunk data is not persisted between sessions |

Reproduce from `packages/luxar-viewer/` after running `pnpm
test:generate-fixtures` and `pnpm exec playwright install firefox webkit`, then
run `pnpm test:e2e:browsers`. The checked-in visual snapshot corpus is
Chromium-only, so this command ignores snapshot assertions and compares
functional behavior rather than pixels.

Not verified: the full E2E suite on any engine but Chromium; real phones and
tablets, including touch behaviour in iOS Safari; **Safari and Edge themselves**
— Playwright's WebKit is a WebKit build, not Safari, and Edge is Chromium-based
but untested; and any performance comparison between engines.
WebKit lacks main-thread `FileSystemFileHandle.createWritable()`, so Safari and
the native WKWebView launcher fall back to L1-only caching; see the
[`opfs-unavailable` cache badge](packages/luxar-viewer/src/cache/README.md#cache-status-badges).

### Viewer URL Parameters

| Parameter | Description |
|-----------|-------------|
| `?src=<url>` | Data source URL (Zarr store) |
| `?theme=light` | Set UI theme (`dark`, `light`, `frosted-glass` or `liquid-glass`) |
| `?debug` | Enable debug mode (`window.__luxarDebug`) |
| `?noCache` | Disable all caching tiers (S-cache + L0/L1/L2) |
| `?cacheDebug` | Show cache hit/miss statistics |
| `?clearCache` | Clear the OPFS persistent cache on load |
| `?noPrefetch` | Disable predictive chunk prefetching |
| `?renderer=webgl\|webgpu` | Select WebGLRenderer + GLSL (production default) or opt into WebGPURenderer + TSL |
| `?webgpuForceWebgl` | With `?renderer=webgpu`, keep WebGPURenderer + TSL but force Three.js's internal WebGL2 backend for diagnostics |

---

## Acknowledgments

### Software

Built with:
- [Three.js](https://threejs.org/) - WebGL/WebGPU rendering
- [Zarr](https://zarr.readthedocs.io/) / [Zarrita](https://github.com/manzt/zarrita.js) - Chunked array storage
- [NumPy](https://numpy.org/) - Numerical computing
- [PyTorch](https://pytorch.org/) - Gaussian-splat fitting
- [FastAPI](https://fastapi.tiangolo.com/) - Data serving
- [Vite](https://vitejs.dev/) - Frontend tooling

### Datasets & scientific data

The gallery and demos visualize openly-available scientific datasets — with
gratitude to the authors, labs, and consortia who produced and shared them.
Luxar only *renders* these data; all rights and credit remain with the original
providers, under their respective licenses. Each demo script's docstring carries
the full citation.

**Microscopy & cell biology**
- **Cells3D** — fluorescence microscopy provided by the [Allen Institute for Cell Science](https://www.allencell.org/), distributed as scikit-image sample data (`skimage.data.cells3d`); scikit-image itself: van der Walt et al. (2014), *PeerJ* 2:e453, [doi:10.7717/peerj.453](https://doi.org/10.7717/peerj.453).
- **Mouse Blastocyst** — Blin et al. (2019), *PLOS Biol.* 17:e3000388, [doi:10.1371/journal.pbio.3000388](https://doi.org/10.1371/journal.pbio.3000388) (CC BY 4.0), via the [Image Data Resource](https://idr.openmicroscopy.org/) (IDR idr0062; Williams et al. 2017, *Nat. Methods*, [doi:10.1038/nmeth.4326](https://doi.org/10.1038/nmeth.4326)).
- **Drosophila Embryogenesis** — acquired in Philipp J. Keller's lab at HHMI Janelia Research Campus, where L. A. Royer was then a postdoctoral fellow; used with permission. Imaged on the SiMView instrument described by Royer et al. (2016), *Nat. Biotechnol.* 34:1267-1278, [doi:10.1038/nbt.3708](https://doi.org/10.1038/nbt.3708).
- **Zebrafish h2afva** — Royer lab, CZ Biohub San Francisco; light-sheet histone timelapse from the Zebrahub imaging corpus. Please cite Lange et al. (2024), *Cell*, [doi:10.1016/j.cell.2024.09.047](https://doi.org/10.1016/j.cell.2024.09.047).
- **Zebrafish Neuromast** — Adrian Jacobo lab (CZ Biohub SF / Rockefeller); unpublished iSIM, deconvolved 4D timelapse. Related biology: Erzberger et al. (2020), *Mechanochemical symmetry breaking during morphogenesis of lateral-line sensory organs*, *Nature Physics* 16:949-957, [doi:10.1038/s41567-020-0894-9](https://doi.org/10.1038/s41567-020-0894-9).
- **Tribolium Embryo** — Barry et al. (2022), *J. Cell Sci.* 135, jcs259511, [doi:10.1242/jcs.259511](https://doi.org/10.1242/jcs.259511), whose supplemental stack ([Zenodo 5270323](https://zenodo.org/records/5270323)) re-hosts a [Cell Tracking Challenge](https://celltrackingchallenge.net/) recording; Maška et al. (2023), *Nat. Methods* 20:1010, [doi:10.1038/s41592-023-01879-y](https://doi.org/10.1038/s41592-023-01879-y).
- **C. elegans nuclei tracking** — Santella, Kovacevic, Bao & Hirsch (2022), 3D+time confocal nuclei dataset, [Zenodo 6460303](https://doi.org/10.5281/zenodo.6460303) (CC BY 4.0).
- **Kidney (3-channel)** — scikit-image sample data `skimage.data.kidney()` (G. Buckley, 2018; CC0 1.0).
- **OpenCell MAP4** and **cytoself Protein Landscape** — [OpenCell](https://opencell.sf.czbiohub.org/) (CZ Biohub SF): Cho, Bhatt et al. (2022), *Science* 375:eabi6983, [doi:10.1126/science.abi6983](https://doi.org/10.1126/science.abi6983); CytoSelf embeddings: Kobayashi et al. (2022), *Nat. Methods*, [doi:10.1038/s41592-022-01541-z](https://doi.org/10.1038/s41592-022-01541-z); [AWS Open Data: czb-opencell](https://registry.opendata.aws/czb-opencell/) (CC BY-SA 4.0).
- **FlyLight MCFO** (whole brain 63× tile, and the FISBe neurons) — FlyLight Project Team, HHMI Janelia Research Campus, [Gen1 MCFO](https://gen1mcfo.janelia.org): Meissner et al. (2023), *eLife* 12:e80660, [doi:10.7554/eLife.80660](https://doi.org/10.7554/eLife.80660); MCFO: Nern, Pfeiffer & Rubin (2015), *PNAS* 112(22), [doi:10.1073/pnas.1506763112](https://doi.org/10.1073/pnas.1506763112); FISBe v1.0: Mais et al. (2024), *CVPR*, [Zenodo 10875063](https://doi.org/10.5281/zenodo.10875063) (CC BY 4.0).
- **STORM 3D Microtubules** — COS-7 α-tubulin (Alexa 647) 3D STORM localisations; Sieben (2019), [Zenodo 3547521](https://doi.org/10.5281/zenodo.3547521) (CC BY 4.0).
- **CODEX Pancreas** — 12-plex CODEX human pancreas section; Björklund et al. (2023), [Zenodo 7742474](https://doi.org/10.5281/zenodo.7742474) (CC BY 4.0).
- **Acto3D Heart** — E13.5 mouse heart, Zeiss Lightsheet 7, [Acto3D sample data](https://github.com/Acto3D/Acto3D); Takeshita et al. (2024), *Development*, [doi:10.1242/dev.202550](https://doi.org/10.1242/dev.202550). The sample data states no reuse terms, so it is fitted locally and never redistributed.
- **Zebrafish Gastrulation (endoderm, 4D)** — Pia Aanstad (2018), confocal timelapse, [Zenodo 1211599](https://doi.org/10.5281/zenodo.1211599) (CC BY-SA 4.0).
- **Cell Tracking (Kaggle)** — [Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development): zebrafish light-sheet crops and lineages, CZ Biohub SF (CC0 1.0).

**Medical & anatomy**
- **CT Anatomy Atlas** — [TotalSegmentator](https://zenodo.org/records/10047263) (102-subject subset, v2.0.1, CC BY 4.0); Wasserthal et al. (2023), *Radiology: AI*, [doi:10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024).
- **Visible Human Head** — U.S. National Library of Medicine [Visible Human Project](https://www.nlm.nih.gov/research/visible/visible_human.html) (Male, axial colour cryosections); Spitzer et al. (1996), *JAMIA*, [doi:10.1136/jamia.1996.96236280](https://doi.org/10.1136/jamia.1996.96236280) (public domain; no NLM endorsement implied).
- **CMU-1 Pathology Slide** — [OpenSlide test data](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/), Aperio `CMU-1.svs`; Goode et al. (2013), *J. Pathol. Inform.*, [doi:10.4103/2153-3539.119005](https://doi.org/10.4103/2153-3539.119005) (CC0 1.0).

**Astronomy & geoscience**
- **Milky Way Dust** — Leike, Glatzle & Enßlin (2020), *A&A*, [doi:10.1051/0004-6361/202038169](https://doi.org/10.1051/0004-6361/202038169); data [Zenodo](https://doi.org/10.5281/zenodo.3993082) (CC BY 4.0).
- **Gaia Milky Way (3M stars)** — ESA/Gaia/DPAC, [Gaia DR3](https://gea.esac.esa.int/archive/); Gaia Collaboration, Vallenari et al. (2023), *A&A* 674:A1, [doi:10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940) (CC BY-NC 3.0 IGO, hence built locally and not redistributed).
- **DESI DR1 Cosmic Web** — [DESI DR1 LSS clustering catalogues](https://data.desi.lbl.gov/public/dr1/); DESI Collaboration (2026), *Data Release 1 of the Dark Energy Spectroscopic Instrument*, [arXiv:2503.14745](https://doi.org/10.48550/arXiv.2503.14745) (CC BY 4.0; DESI's [acknowledgment text](https://data.desi.lbl.gov/doc/acknowledgments/) applies).
- **Cosmicflows-4 / Laniakea** — Tully et al. (2023), *ApJ*, [doi:10.3847/1538-4357/ac94d8](https://doi.org/10.3847/1538-4357/ac94d8); Laniakea: Tully et al. (2014), *Nature*; distances from the Extragalactic Distance Database; pipeline after Conradi & De Domenico ([manlius/laniakea](https://github.com/manlius/laniakea)).
- **IllustrisTNG Cosmic Web** — [TNG300-3-Dark](https://www.tng-project.org/data/), snapshot 99; Nelson et al. (2019), *Comput. Astrophys. Cosmol.* 6:2, [doi:10.1186/s40668-019-0028-x](https://doi.org/10.1186/s40668-019-0028-x) (access requires registration, so the fit is built locally).
- **Solar System Asteroids** — NASA/JPL [Small-Body Database](https://ssd.jpl.nasa.gov/), courtesy NASA/JPL-Caltech (public domain).
- **Ocean Currents of Earth** — [HYCOM + NCODA GOFS 3.1](https://www.hycom.org/dataserver/gofs-3pt1/analysis) global 1/12° analysis; Chassignet et al. (2007), *J. Mar. Syst.* 65, [doi:10.1016/j.jmarsys.2005.09.016](https://doi.org/10.1016/j.jmarsys.2005.09.016); Earth texture: NASA Blue Marble NG.
- **NEXRAD Supercell (El Reno, 2013)** — NOAA/NWS WSR-88D Level II, KTLX, via [NOAA Open Data Dissemination](https://registry.opendata.aws/noaa-nexrad) (public domain; a regridded, Gaussian-fitted derived product, not original NOAA data); read with MetPy ([doi:10.5065/D6WW7G29](https://doi.org/10.5065/D6WW7G29)).
- **Global Earthquakes** — [USGS Earthquake Catalog](https://earthquake.usgs.gov/) (real-time feed); Earth texture: NASA Blue Marble.
- **Rivers of Earth** — [HydroRIVERS v10](https://www.hydrosheds.org/products/hydrorivers) (HydroSHEDS; Lehner & Grill 2013, *Hydrol. Process.* 27(15), CC BY 4.0) + [ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model) global relief (NOAA NCEI, [doi:10.25921/fd45-gt74](https://doi.org/10.25921/fd45-gt74), public domain).
- **Biodiversity at Planetary Scale** — [GBIF](https://www.gbif.org/) occurrence records ([GBIF.org occurrence snapshot](https://registry.opendata.aws/gbif/) on the AWS Open Data registry; a CC BY 4.0 / CC0 1.0 subset) + CC0 animal-migration tracks from [Movebank](https://www.movebank.org/); Earth texture: NASA [Blue Marble](https://visibleearth.nasa.gov/collection/1484/blue-marble).

**Connectomes, structures & networks**
- **FlyWire Connectome** — FlyWire whole-brain connectome, public release 783; Dorkenwald et al. (2024) & Schlegel et al. (2024), *Nature*; [annotations](https://github.com/flyconnectome/flywire_annotations), [connectivity (Zenodo)](https://zenodo.org/records/10676866), [Codex](https://codex.flywire.ai/) (CC BY 4.0).
- **Single-Cell 3D Genome** — Dip-C; Tan et al. (2018), *Science* 361:924, [doi:10.1126/science.aat5641](https://doi.org/10.1126/science.aat5641); GEO GSE117876.
- **Human White-Matter Tractography** — [HCP-1065 population-averaged tractography atlas](https://brain.labsolver.org/hcp_trk_atlas.html); Yeh, F-C. (2022), *Nat. Commun.* 13:4933, [doi:10.1038/s41467-022-32595-4](https://doi.org/10.1038/s41467-022-32595-4) (CC BY-SA 4.0). Derived from the Human Connectome Project, WU-Minn Consortium (PIs David Van Essen & Kamil Ugurbil; 1U54MH091657), funded by the 16 NIH Institutes and Centers supporting the NIH Blueprint for Neuroscience Research, and by the McDonnell Center for Systems Neuroscience at Washington University; used under the [WU-Minn HCP open-access data-use terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
- **ATP Synthase** — PDB [5DN6](https://www.rcsb.org/structure/5DN6), Zhou et al. (2015), *eLife*, [doi:10.7554/eLife.10180](https://doi.org/10.7554/eLife.10180); RCSB [PDB-101](https://pdb101.rcsb.org/motm/72).
- **Nuclear Pore Complex** — human NPC scaffold, PDB [7R5K](https://www.rcsb.org/structure/7R5K) (constricted) and 7R5J (dilated); Mosalaganti et al. (2022), *Science* 376:eabm9506, [doi:10.1126/science.abm9506](https://doi.org/10.1126/science.abm9506); coordinates from the RCSB PDB.
- **PBCV-1 Capsid (cryo-EM)** — EMDB [EMD-5384](https://www.ebi.ac.uk/emdb/EMD-5384); Zhang et al. (2011), *PNAS* 108:14837, [doi:10.1073/pnas.1107847108](https://doi.org/10.1073/pnas.1107847108).
- **CAIDA AS topology** — [CAIDA](https://asrank.caida.org/) AS-relationships, AS-organizations & AS Rank (UC San Diego / CAIDA); Luckie et al. (2013), [doi:10.1145/2504730.2504735](https://doi.org/10.1145/2504730.2504735).
- **HuRI interactome** — Luck et al. (2020), *Nature* 580:402, [doi:10.1038/s41586-020-2188-x](https://doi.org/10.1038/s41586-020-2188-x); [Human Reference Interactome](https://interactome-atlas.org/).

**Single-cell atlases & embeddings**
- **Tabula Sapiens** — Tabula Sapiens Consortium (2022), *Science*, [doi:10.1126/science.abl4896](https://doi.org/10.1126/science.abl4896); accessed via [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/) (CC BY 4.0).
- **Zebrahub** — [CZ Biohub Zebrahub](https://zebrahub.org); RNA-velocity embedding: Lange et al. (2024), *Cell*, [doi:10.1016/j.cell.2024.09.047](https://doi.org/10.1016/j.cell.2024.09.047); Zebrahub-Multiome: Kim et al. (2024), [bioRxiv:2024.10.18.618987](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v2); 3D embeddings by Yang-Joon Kim (CZ Biohub SF).
- **Protein Landscape** — CAFA5 protein embeddings via ProtT5 (Elnaggar et al. 2022, *IEEE TPAMI*, [doi:10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)); [CAFA5 challenge](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction).
- **Spotify Tracks** — [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (Hugging Face), derived from the Spotify Web API audio features.
- **CELLxGENE Census UMAP** — 3D UMAP of the [CZ CELLxGENE Discover Census](https://cellxgene.cziscience.com/) scVI latent space; CZI Cell Science Program et al. (2025), *Nucleic Acids Res.* 53:D886, [doi:10.1093/nar/gkae1142](https://doi.org/10.1093/nar/gkae1142) (CC BY 4.0; the contributing studies are listed in the Census release).
- **ESM protein landscapes** — [UniProt/Swiss-Prot](https://www.uniprot.org/) sequences (CC BY 4.0) embedded with EvolutionaryScale ESM C / ESM3-open; **ESM Protein Universe** — the [ESM Atlas](https://biohub.ai/esmc/atlas) cluster map, Candido et al. (2026), bioRxiv, [doi:10.64898/2026.06.03.729735](https://doi.org/10.64898/2026.06.03.729735), shown with permission.
- **arXiv Papers** — 3.29M preprint embeddings from Kaggle [tomtum/openai-arxiv-embeddings](https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings); titles and categories from the Cornell University arXiv metadata snapshot.
- **Bird Plumage Colour Space** — [BirdColorBase](https://github.com/BirdColorBase/home), 360,432 reflectance spectra over 2,632 species (MIT).
- **Human Multiome** — peak-accessibility UMAP of human single-cell ATAC-seq; data from Domcke et al. (2020), *A human cell atlas of fetal chromatin accessibility*, *Science* 370:eaba7612, [doi:10.1126/science.aba7612](https://doi.org/10.1126/science.aba7612); peak-UMAP analysis from Zebrahub-Multiome, Kim et al. (2024), [bioRxiv:2024.10.18.618987](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v2).
- **Mouse Multiome** — peak-accessibility UMAP of a mouse single-cell multiome; data from Argelaguet et al. (2022), *Decoding gene regulation in the mouse embryo using single-cell multi-omics*, [bioRxiv:2022.06.15.496239](https://doi.org/10.1101/2022.06.15.496239); peak-UMAP analysis from Zebrahub-Multiome, Kim et al. (2024), [bioRxiv:2024.10.18.618987](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v2).

**Photogrammetry & 3DGS captures**
- **Cluster Fly** (macro photogrammetry) — Dany Bittel, [danybittel.ch/macro](https://danybittel.ch/macro) (CC BY 4.0).
- **Observatories** (Rubin, Gemini South) — Gaussian-splat captures by khyron, [khyron/Gaussian-Splatting](https://github.com/khyron/Gaussian-Splatting) (CC BY 4.0).
- **Scaniverse SPZ samples** — Niantic Labs [spz](https://github.com/nianticlabs/spz) reference captures (MIT).
- **INRIA "bonsai"** and **Mip-NeRF 360 "garden"** — Kerbl et al. (2023), *ACM TOG* 42(4), [doi:10.1145/3592433](https://doi.org/10.1145/3592433); captures from Mip-NeRF 360, Barron et al. (2022), *CVPR*, [doi:10.1109/CVPR52688.2022.00539](https://doi.org/10.1109/CVPR52688.2022.00539). Research-use terms, so both are fetched at run time and never redistributed.
- **MatrixCity aerial** — Li et al. (2023), *ICCV*, [doi:10.1109/iccv51070.2023.00297](https://doi.org/10.1109/iccv51070.2023.00297); 3DGS reconstruction hosted on [SuperSplat](https://superspl.at/scene/ace6e5b0).

Synthetic, procedurally generated demos — **Lorenz Attractor**, **Spiral Galaxy**, **Galaxy Simulation**, **Rainbow Sphere**, **Quantum Orbitals**, **Hilbert Curve**, **Bioluminescent Ocean**, **Particle Collision** (static and animated, inspired by CERN LHC events), **4D Fractals**, **Mandelbulb**, **Quasicrystal**, **Turing Patterns**, **Evolving Cloud**, **Cubic Array**, **L-System Forest**, the **nD Transform Test Bench**, and **Exotic Surfaces** (nodal surfaces after von Schnering & Nesper 1991, *Z. Phys. B* 83:407, [doi:10.1007/BF01313411](https://doi.org/10.1007/BF01313411)) — use no external data. The two **ChromaTrace** demos read a bundle you supply and are not hosted.

### Where the demo data lives

The Python package does not ship these artifacts. The demos consume **derived
products** — Gaussian-splat fits and point catalogues computed from the datasets
credited above — archived on Zenodo in four demo-data records. The ShareAlike
files need a separate record because a Zenodo record carries a single licence
field; the two large timelapses each have their own record so the data collector
is credited on the recording itself:

| Record | Contents | Cite |
|---|---|---|
| Permissively licensed (CC-BY, CC0, public domain) | 46 files, 1.2 GiB | [10.5281/zenodo.21912279](https://doi.org/10.5281/zenodo.21912279) |
| ShareAlike (CC BY-SA 4.0) | 3 files, 20 MiB | [10.5281/zenodo.21912281](https://doi.org/10.5281/zenodo.21912281) |
| Zebrafish histone timelapse (253 + 51 timepoints) | 2 files, 6.5 GiB | [10.5281/zenodo.21912283](https://doi.org/10.5281/zenodo.21912283) |
| *Drosophila* embryogenesis (500 timepoints) | 1 file, 1.1 GiB | [10.5281/zenodo.22118694](https://doi.org/10.5281/zenodo.22118694) |

These records cover 25 of the 31 demo datasets. Those 31 are the datasets tracked
by the data manifest; the other six (Gaia, IllustrisTNG, Acto3D, Tribolium, FlyLight
MCFO, Dip-C) are built locally because redistribution is not permitted or not yet
arranged, or because regeneration is cheap. The remaining demos fetch their
catalogues straight from the upstream providers credited above.

`luxar demo run <name>` resolves only what that demo needs, caches it under
`~/.cache/luxar/`, and verifies every file against a SHA-256 recorded in
[`data_manifest.json`](packages/luxar/src/luxar/demos/data_manifest.json). Files
fetched from Zenodo must match the hosted digest or the download is deleted.
Existing cache and packaged copies are checksum-checked the same way.

The DOIs above are *concept* DOIs: they always resolve to the newest version of
a record. Each individual version also has its own DOI, which is what to use
when a result needs to be reproducible against exact bytes.

**Citing these**: the Gaussian-splat fits are lossy representations built for
interactive visualisation, not substitutes for source imagery in quantitative
work. The coordinate, label, and point-catalogue files are derived analysis
outputs. If you use either kind, please cite the upstream dataset credited
above as well; each record's description names it per dataset.

---

## License

BSD-3-Clause License. See [LICENSE](LICENSE) for details.

---

## Citation

Two things can be cited. For the **software** (this repository, whichever
version you used):

```bibtex
@software{luxar2026,
  title = {Luxar: High-Performance n-Dimensional Scientific Visualization},
  author = {Royer, Lo{\"i}c A. and the Luxar contributors},
  year = {2026},
  url = {https://github.com/royerlab/luxar}
}
```

For the **method and the benchmarks** (Gaussian-splat fitting of microscopy
volumes, blind-spot cross-validation of the splat budget, the streaming viewer),
cite the preprint:

```bibtex
@article{royer2026luxar,
  title = {Luxar: Gaussian splatting for microscopy and scalable interactive web
           visualisation of multidimensional scientific data},
  author = {Royer, Lo{\"i}c A.},
  journal = {bioRxiv},
  year = {2026},
  url = {https://github.com/royerlab/luxar}
}
```

If you use the **demo datasets**, please cite the Zenodo record they came from
as well as the upstream data — see
[Where the demo data lives](#where-the-demo-data-lives) for the DOIs.

---

<p align="center">
Built for the scientific visualization community.
<br>
<a href="https://github.com/royerlab/luxar/issues">Report Issues</a> |
<a href="https://github.com/royerlab/luxar/discussions">Discussions</a>
</p>
