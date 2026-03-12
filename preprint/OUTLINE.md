# Luxar Preprint Outline

**Target**: BioRXiv preprint
**Focus**: Gaussian splatting as a compact, interactive representation for microscopy volumes
**Audience**: Microscopy / bio-imaging community

---

## Thesis

Gaussian splats provide a powerful, compact, and interactive representation for microscopy volumes — enabling web-based exploration of multi-channel, multi-timepoint 3D data that would otherwise require desktop software and large file transfers.

---

## Feature Inventory (What We Have)

### Python Pipeline
- **Fitting**: Per-splat Adam with dynamic relocation, 3 presets (draft/standard/hifi), GPU acceleration (CUDA/MPS)
- **Tiled fitting** (implemented): Cosine-apodized overlapping tiles for arbitrarily large volumes, enabling parallel fitting and removing GPU memory ceiling
- **Slurm batch fitting** (planned): `luxar gsplat batch` for HPC cluster fitting of 3D+t OME-Zarr via Slurm array jobs, with auto-merge
- **Generalized Gaussians**: Learnable sharpness parameter s in exp(-1/2 ||y||^s), where s=2 is the standard Gaussian (draft/standard presets use fixed s=2.0; hifi enables learnable s)
- **Loss functions**: L1, MSE, Poisson (with asymmetric penalties)
- **Seeding**: 4 methods (edges, decomposition, grid, auto) with GPU acceleration
- **Post-processing**: Pruning (3 methods), filtering (7 criteria: bbox, volume, amplitude, eccentricity, sharpness, mass, per-axis sigma), merging
- **Multi-channel**: Per-channel color assignment, boolean toggles
- **Multi-timepoint**: Stack as new dimension (discrete or continuous)
- **CLI**: Full pipeline (fit, convert, render, merge, split, slice, prune, filter, info, napari, view)
- **OME-Zarr input**: Full 5D TCZYX support via `--channel` and `--timepoint` flags, auto-detects OME-Zarr layout
- **CLAHE**: Contrast enhancement preprocessing for heterogeneous backgrounds

### Viewer
- **HDR rendering**: 16-bit float pipeline, 7 tone mapping modes, ACES filmic
- **Per-node GOG**: Gain-offset-gamma per node (intensity, offset, gamma)
- **Global EOG**: Exposure-offset-gamma post-processing
- **nD navigation**: Hypersphere slicing, dimension sliders, keyboard navigation
- **Post-processing**: Bloom, DOF, detector noise, chromatic aberration, vignette, AO, lens distortion
- **Anti-aliasing**: FXAA, MSAA, SMAA, SSAA
- **Caching**: 3-level (L0 decompressed 200MB, L1 compressed 100MB, L2 OPFS 2GB)
- **WASM acceleration**: Up to 16D, TypeScript fallback beyond
- **Progressive loading**: Chunked, spatial-indexed, streamed over HTTP
- **3 themes**: Dark, Light, Frosted Glass

### Data Model
- **Scene graph**: Scene > Group > (Points, Lines, GSplats), hierarchical transforms
- **nD transforms**: Per-dimension affine and categorical, inverse-query O(1) design
- **Encoding**: Semantic-aware (7 types, 4 modes), LUT, broadcast, quantization
- **Spatial indexing**: Compound ordering (discrete dims + Morton spatial)
- **Physical units**: nm, um, mm, cm, m, km, inch, foot, px, au (+ aliases metre, meter)

---

## Feature Gaps (Needed for Paper)

### Paper-Blocking
| Gap                    | Status   | Impact                                |
|------------------------|----------|---------------------------------------|
| Screenshot export      | DONE #11 | Cannot generate figures without it    |
| Scale bar overlay      | DONE #13 | Every microscopy figure requires one  |
| PSNR/SSIM `--compare`  | DONE #16 | Quantitative backbone of the paper    |
| Tiled fitting          | DONE #18 | Credible "handles real data" claim    |

### Strongly Recommended
| Gap                            | Status   | Impact                          |
|--------------------------------|----------|---------------------------------|
| Video export                   | DONE #12 | Supplementary materials         |
| Colorbar / channel legend      | TODO #14 | Multi-channel figures need labels |
| Orthographic projection        | TODO #6  | Microscopists expect ortho      |
| Layers panel (per-node toggles)| TODO #5  | Multi-channel UX                |

### Nice-to-Have
| Gap                   | Status   | Impact                       |
|-----------------------|----------|------------------------------|
| Viewer-side colormaps | TODO #15 | Microscopy convention        |
| Slurm batch fitting   | TODO #19 | HPC scalability for 3D+t     |

---

## Figure Plan

### Fig 1: Hero — Multi-Channel 3D Organoid
**Demo**: `demo_gsplats_3d_organoid_multichannel_from_idr.py`

**Panels**:
- (a) Original volume max-projection (per channel + composite)
- (b) GSplat reconstruction (same viewpoints)
- (c) Interactive web viewer screenshot with channels visible
- (d) Compression ratio callout (raw volume size vs gsplat file size)

**Point**: "Here's what Luxar does" — the hero figure showing the full pipeline on real public data (IDR).

### Fig 2: Quality & Compression — Systematic Benchmark
**Demo**: NEW — systematic quality benchmark across tissue types

**Panels**:
- (a) PSNR/SSIM vs compression ratio curves for 3 presets (draft/standard/hifi)
- (b) PSNR/SSIM vs seed count (under-fit to over-fit)
- (c) Visual comparison strip: original vs draft vs standard vs hifi for 2-3 tissue types
- (d) Table: compression ratios, fitting time, quality metrics

**Tissue types** (diversity matters):
- Nuclei (DAPI) — round, well-separated blobs
- Membrane / cytoplasm — diffuse, low-contrast
- Filaments (microtubules or actin) — thin, elongated structures

**Point**: "It's quantitatively good" — the evidence figure. Reviewers will scrutinize this most.

**Requires**: `luxar gsplat info --compare` (DONE #16)

### Fig 3: 4D Time-Lapse — Zebrafish Embryo
**Demo**: `demo_gsplats_4d_zebrafish_timelapse.py`

**Panels**:
- (a) Timeline strip of key frames (t=0, t=N/4, t=N/2, t=3N/4, t=N)
- (b) Compression ratio: raw 4D volume vs gsplats with discrete time dimension
- (c) Viewer screenshot showing time slider and nD navigation
- (d) Quality metric per timepoint (PSNR vs time — does quality degrade?)

**Point**: "It handles time-lapse data" — the nD story. Storing 50+ timepoints as volumes is massive; as gsplats with `sigma=0` on the time dimension, dramatically smaller.

### Fig 4: Multi-Channel with Toggles — Kidney or Cells3D
**Demo**: `demo_gsplats_4d_kidney_multichannel_toggles.py` or `demo_gsplats_4d_cells3d_multichannel_toggles.py`

**Panels**:
- (a) Composite view (all channels)
- (b-d) Individual channels isolated (one per panel)
- (e) Viewer screenshot showing boolean toggles in dimension panel

**Point**: "It handles complex multi-modal microscopy" — 6D data (x,y,z + 3 boolean channel toggles) with independent per-channel visibility control, exactly matching biologists' expectations. The kidney demo has DAPI (nuclei), WGA (glomeruli/tubules), and Phalloidin (actin) — all eight on/off combinations are navigable.

### Fig 5: Tracking + Mixed Geometry — C. elegans
**Demo**: `demo_gsplats_4d_celegans_tracking.py`

**Panels**:
- (a) GSplats (nuclear volumes) + polyline overlays (tracks) at selected timepoints
- (b) Close-up showing track-gsplat correspondence
- (c) Full 4D trajectory view

**Point**: "It's a scene graph, not just a volume renderer" — composing different geometry types (splats + lines) in a single nD scene. Tracking is a hot topic in bio-imaging.

### Fig 6: Scalability — Tiled Fitting for Large Volumes
**Demo**: NEW — tiled fitting on a large light-sheet volume (512^3+)

**Panels**:
- (a) Schematic: volume split into overlapping tiles with cosine apodization windows
- (b) Per-tile fitting (independent, parallelizable) -> concatenated splats
- (c) Seamless reconstruction: close-up of tile boundary region showing no visible seams
- (d) Quality vs tile size: PSNR curves for different tile sizes (128, 256, 512) and overlap fractions
- (e) Scaling: fitting time vs volume size (tiled vs monolithic), memory usage comparison

**Point**: "It scales to real microscopy data" — the practical scalability story. Without tiling, fitting is limited by GPU memory (volume + splat parameters + optimizer state + gradients must all fit); for large seed counts on consumer GPUs, this can cap out well below full light-sheet resolution. With tiled fitting, arbitrarily large volumes (2048x2048x500 light-sheet stacks) become tractable, and tiles can be fit in parallel across GPUs.

**Requires**: Tiled fitting implementation (DONE #18)

### Fig 7: Architecture & Pipeline Diagram
**Not a demo — a schematic**

**Panels**:
- (a) Pipeline: Volume data -> Seed generation -> GSplat fitting -> .gsplats.zarr -> Convert to Luxar scene -> HTTP serve -> Web viewer
- (b) Data flow: Python (fitting, encoding) -> HTTP (chunked streaming) -> Browser (WASM decode, WebGL render)
- (c) Scene graph structure: Scene > Groups > (Points, Lines, GSplats) with transforms and dimensions

**Point**: "Here's how it works" — system architecture for the methods section.

### Supplementary Figures

**Supp 1: Web Delivery & Performance**
- Loading time vs dataset size (100K -> 10M splats)
- Cache hit rates (L0/L1/L2) over repeated navigation
- Network simulation: performance under 3G, 4G, WiFi conditions
- Comparison: downloading raw volume vs streaming gsplats

**Supp 2: Preset Visual Comparison Gallery**
- Grid: rows = tissue types, columns = draft / standard / hifi
- Include PSNR, file size, and fitting time per cell

**Supp 3: Generalized Gaussian Sharpness**
- Show effect of learnable sharpness on reconstruction quality
- Compare fixed s=2.0 (draft/standard presets) vs learnable s (hifi preset)
- Visualize per-splat sharpness distribution for different tissue types
- Note: draft/standard default to fixed s=2.0; hifi enables learning via sharpness_range=(min, max)

**Supp 4: Dynamic Relocation**
- Convergence curves with and without dynamic ops
- Residual maps showing where relocated splats end up
- Quality improvement from relocation (PSNR difference)

**Supp 5: GPU Acceleration Benchmarks**
- Fitting time: CPU vs CUDA vs MPS for various volume sizes
- Seeding time: CPU vs GPU for edge-based seeding
- Rendering time: CPU vs GPU for volume reconstruction

---

## Demos: Existing vs Needed

### Existing Demos (Ready to Use)
| Demo | Figure | Notes |
|------|--------|-------|
| `demo_gsplats_3d_organoid_multichannel_from_idr.py` | Fig 1 | Real IDR data, full pipeline |
| `demo_gsplats_3d_organoid_multichannel_precomputed.py` | — | Quick-start variant |
| `demo_gsplats_3d_organoid_dapi_nuclei_from_idr.py` | Fig 2 (nuclei) | Single-channel DAPI |
| `demo_gsplats_3d_tribolium_embryo.py` | Fig 2 (alt) / Fig 6 | Light-sheet volume (965x1871x991, 0.381 um isotropic) — also a natural candidate for tiled fitting |
| `demo_gsplats_4d_zebrafish_timelapse.py` | Fig 3 | Confocal time-lapse |
| `demo_gsplats_4d_cells3d_multichannel_toggles.py` | Fig 4 (alt) | 5D with toggles |
| `demo_gsplats_4d_kidney_multichannel_toggles.py` | Fig 4 | 6D kidney |
| `demo_gsplats_4d_celegans_tracking.py` | Fig 5 | Tracking + polylines |

### New Demos Needed
| Demo | Figure | Description | Effort |
|------|--------|-------------|--------|
| **Quality benchmark script** | Fig 2 | Systematic PSNR/SSIM across presets, seed counts, and tissue types. Generates comparison tables and curves. | Medium (uses `luxar gsplat compare`, DONE #16) |
| **Filament/membrane dataset** | Fig 2 | Need a volumetric dataset with thin structures (actin, microtubules) to test quality on challenging morphologies. The kidney Phalloidin channel (actin filaments) could serve, or source a dedicated filament volume from a public repository (e.g., STORM microtubules from OpenCell, Allen Cell Explorer). | Small (data sourcing) |
| **Tiled fitting demo** | Fig 6 | Large volume fit with cosine-apodized tiling. Show seamless reconstruction, scaling curves, and tile boundary quality. The Tribolium embryo (965x1871x991) is already large enough, or use a cleared tissue / whole-brain dataset. | Medium (tiled fitting DONE #18, needs demo script) |
| **Sharpness ablation** | Supp 3 | Fit same data with fixed s=2.0 vs learnable sharpness, compare quality. | Small |
| **Dynamic ops ablation** | Supp 4 | Fit with and without dynamic relocation, compare convergence and quality. | Small |
| **GPU benchmark script** | Supp 5 | Systematic timing across devices and volume sizes. | Small |

---

## Suggested Paper Structure

### Title
*Luxar: Gaussian Splatting for Interactive Web-Based Visualization of Microscopy Volumes*

(or: *...for Compact Representation and Streaming of Microscopy Data*)

### Abstract (~250 words)
- Problem: Microscopy generates massive 3D/4D/nD volumes; sharing and exploring interactively is hard
- Solution: Represent volumes as Gaussian splats — compact, streamable, GPU-renderable
- Key results: X-fold compression, PSNR of Y, web-based with Z FPS
- Availability: Open source, pip-installable, web viewer

### Introduction
- Microscopy data volumes are growing (light-sheet, confocal, expansion microscopy)
- Current solutions: napari (desktop), neuroglancer (web but voxel-based), BigDataViewer (Java)
- Gap: No compact, streamable, interactive web solution for volumetric microscopy
- Gaussian splatting: originally from computer graphics (3DGS), we adapt for microscopy
- Key innovations: generalized Gaussians (learnable sharpness), nD support, OME-Zarr input, web streaming

### Results
- **Fig 1**: Multi-channel 3D organoid visualization (the demo)
- **Fig 2**: Systematic quality analysis (the evidence)
- **Fig 3**: 4D time-lapse handling (the nD story)
- **Fig 4**: Complex multi-channel with toggles (the real-world use case)
- **Fig 5**: Mixed geometry — tracking overlays (the composability story)
- **Fig 6**: Tiled fitting for large volumes (the scalability story)

### Methods
- **Fitting pipeline**: Seeds -> Adam optimization -> dynamic relocation -> post-processing
- **Generalized Gaussians**: a * exp(-1/2 ||Sigma^(-1/2)(x-mu)||^s) with learnable sharpness s
- **Tiled fitting**: Cosine-apodized overlapping tiles -> independent fitting -> seamless concatenation
- **nD representation**: Discrete dimensions (time, channel) + continuous (spatial)
- **Web architecture**: Zarr encoding -> spatial indexing -> chunked HTTP -> WASM decode -> WebGL render
- **Scene graph**: Hierarchical transforms, per-node rendering attributes

### Discussion
- Comparison with existing approaches (napari, neuroglancer, Vaa3D, 3DGS)
- Limitations: quality ceiling for thin structures, no deconvolution, tile boundary artifacts at very small overlap
- Future work: Slurm batch fitting for HPC clusters, viewer-side colormaps, real-time streaming, VR

### Data & Code Availability
- GitHub repo
- PyPI package (`pip install luxar[gsplats]`)
- Live demo URL

---

## Implementation Priority (What to Build First)

1. **DONE #18**: Tiled fitting for large volumes (unlocks Fig 6, credibility for real data)
2. **DONE #16**: PSNR/SSIM `--compare` in CLI (unlocks Fig 2)
3. **DONE #11**: Screenshot export (unlocks all figures)
4. **DONE #13**: Scale bar overlay (required for all microscopy figures)
5. **Quality benchmark script** (generates Fig 2 data)
6. **Tiled fitting demo on large volume** (generates Fig 6 data)
7. **TODO #12**: Video export (supplementary materials)
8. **TODO #14**: Colorbar / channel legend (multi-channel figures)
9. **Sharpness + dynamic ops ablation scripts** (Supp 3-4)
10. **GPU benchmark script** (Supp 5)

---

## Open Questions

- **Title**: Should it emphasize "Gaussian splatting" (novel method) or "web visualization" (practical impact)?
- **Scope**: Include the full Luxar system (points, lines, gsplats) or focus purely on gsplats for microscopy?
- **Comparison**: Do we want a head-to-head with napari/neuroglancer, or just benchmark against raw volumes?
- **Data sources**: Which public datasets give us the strongest story? IDR is good for reproducibility.
- **Large-scale demo**: Do we have access to a 512^3+ cleared tissue or whole-brain volume?
- **Tiled fitting**: What tile size and overlap fraction give the best quality/speed tradeoff? Is 10-20% overlap sufficient with cosine apodization?
- **Parallelism**: Should tiled fitting support multi-GPU out of the box, or is that a follow-up?
