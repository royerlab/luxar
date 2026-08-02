# Luxar Demos

This folder contains self-contained demo scripts that showcase Luxar's capabilities. Each demo is a complete, runnable Python script that generates data, launches the viewer, and cleans up automatically.

## Purpose

These demos are:
- **Didactic**: Easy to understand and learn from
- **Self-contained**: All generation code in one file
- **Complete**: Generate → Serve → View → Cleanup workflow
- **Copy-pasteable**: Can be used as templates for your own visualizations

## Running Demos (the `luxar demo` CLI)

The easiest way to run any demo is the CLI (equivalent to the per-demo
`hatch run python ...` commands listed below):

```bash
luxar demo                  # list all demos (key, index, requirements)
luxar demo info lorenz      # metadata for one demo
luxar demo run lorenz       # run by key or index; forwards -- args
luxar demo run lorenz -- --no-serve --points=10000
luxar demo run-all          # generate demo datasets (--no-serve; skips GPU/large-download by default)
luxar demo cache list       # inventory demo caches under ~/.cache/luxar/
luxar demo deps             # which optional dependencies are missing?
luxar demo deps --install   # install the extras that provide them
```

## Optional dependencies

The core install deliberately excludes the heavyweight packages some demos need
(`torch`, `umap-learn`, `esm`, `cellxgene-census`, `nibabel`, …), so a fresh
checkout can list every demo but not run every demo.

`_dependencies.py` is the single source of truth. `INSTALL_SPECS` maps each
**import** name (`skimage`, not `scikit-image`) to the constrained requirement,
the Luxar extra that provides it, and — where the bound or the skippability is
load-bearing — why. Two consumers read it, which is what keeps them honest:

| Consumer | Role |
|---|---|
| `require_module("x")` | The runtime gate. Raises `MissingDependencyError` naming the *constrained* spec and its extra. |
| `luxar demo deps` | The installer/report. Surveys the table with `find_spec` (no imports), flags an installed-but-below-pin package `OUTDATED`, and exits 1 if anything is missing or out of date. |

Two rules govern the gate, both learned from real bugs:

1. **Gate at the point of use, never at the entry point.** A demo whose
   expensive artifact is already cached must run *without* the dependency that
   produced it. `tests/test_no_entrypoint_dependency_preflight.py` fails the
   build if an entry-point preflight reappears.
2. **Advertise the constrained requirement.** A bare `pip install anndata`
   resolves to 0.13+, which needs `zarr>=3.1` and would silently upgrade Luxar
   past its `zarr<3.0` pin, breaking every store on disk.

`tests/test_demos_dependencies.py` enforces both: every spec must accept exactly
the versions its `pyproject.toml` pin accepts, every module passed to
`require_module` must exist in the table, and no demo source — runtime message or
docstring alike — may spell out `pip install <pkg>` for a package the table
bounds without carrying that bound.

Installing everything:

```bash
make install-demo-deps      # hatch env: the demos + gsplats + io extras
luxar demo deps --install   # same thing from the CLI, any environment
```

Some demos need something a package manager can't supply — a Kaggle credential,
a manual download, a `git lfs pull`, or a GPU. Those show up in the `NEEDS`
column of `luxar demo` and in `luxar demo info <key>`, not here.

## The DEMO_META registry

Every `demo_*.py` file carries a top-level `DEMO_META` dict literal —
machine-readable metadata (key, title, description, category, geometry,
requirements incl. GPU/download/local-data, plus its cache dirs and output
scene stems). `registry.py` discovers demos by globbing this folder and
AST-parses each file's `DEMO_META` (no import), so the registry can never
drift from the files on disk; `tests/test_demo_meta.py` schema-validates all
of them (a demo without valid `DEMO_META`, or one that ignores `--no-serve`,
fails the suite). The `luxar demo` sub-app is driven entirely by this
registry — there is no second metadata list to keep in sync.

## Available Demos

### Scientific Visualization Demos

#### demo_lorenz.py - Lorenz Attractor
Beautiful chaotic attractor with rainbow color gradient.

**Run**: `luxar demo run lorenz [-- --points=100000]`

**Demonstrates**: Differential equation integration, vectorized HSV-to-RGB color conversion, time-based color gradients, progressive writing, chaotic systems visualization.

---

#### demo_rainbow_sphere.py - Fibonacci Spiral Sphere
Dense sphere (200k points) with perfect distribution and rainbow colors.

**Run**: `luxar demo run rainbow_sphere [-- --points=100000]`

**Demonstrates**: Fibonacci (golden angle) spiral for optimal sphere coverage, smooth rainbow gradient using phase-shifted sine waves, automatic point spacing calculation, high density visualization.

---

#### demo_volumetric_cloud.py - Fractal Cloud Structure
Realistic cloud using multi-octave fractal noise and varying point sizes.

**Run**: `luxar demo run cloud [-- --points=1000000]`

**Demonstrates**: Self-contained Perlin-like fractal noise, multi-octave detail at multiple scales, volumetric density filtering, varying point sizes based on local density, soft cloud-like appearance (low sharpness 0.2-0.35 on the normalized [0, 1] knob).

---

#### demo_cubic_array.py - 3D Cubic Grid with Star Background
Dense 100-cubed grid (1M points) with 500k background stars.

**Run**: `luxar demo run cubic_array`

**Demonstrates**: Regular grids via meshgrid, multi-layer scenes (foreground + background), very high point density (1.5M total), different blending modes (additive vs normal), semi-transparent background layers.

---

#### demo_mandelbulb.py - 3D Mandelbulb Fractal
Stunning volumetric representation of the famous Mandelbulb 3D fractal.

**Run**: `luxar demo run mandelbulb [-- --resolution=128] [-- --power=8]`

**Demonstrates**: 3D fractal mathematics (extension of Mandelbrot set), distance estimation for surface detection, iteration-based coloring, adaptive point sizing, spherical coordinate transformation, escape-time algorithm in 3D.

---

#### demo_spiral_galaxy.py - Realistic Multi-Armed Spiral Galaxy
Beautiful astronomical simulation of a barred spiral galaxy.

**Run**: `luxar demo run spiral_galaxy [-- --stars=500000] [-- --arms=4]`

**Demonstrates**: Logarithmic spiral arm generation, realistic stellar population distributions, color variation (blue young stars in arms, red/yellow old stars in bulge), central galactic bulge and stellar halo modeling.

---

#### demo_quantum_orbitals.py - Quantum Atomic Orbitals (Gaussian splats)
Hydrogen-atom electron probability density |ψ|² for eight quantum states. Each orbital is evaluated on a 3D voxel grid and then fitted with oriented Gaussians, so it renders as a translucent volumetric cloud instead of a thresholded point cloud. Splats are tinted by the **sign of ψ** (warm = positive, cool = negative), which is what makes the nodal structure — the radial node of 2s, the plane between the 2p lobes, the alternating 3d cloverleaf lobes — actually visible. All eight states share one frame at true relative scale, so 1s really is several times smaller than 3d.

**Run**: `luxar demo run quantum_orbitals` — optionally with one `--` separator followed by all script args, e.g. `luxar demo run quantum_orbitals -- --grid 128 --seeds 40000 --recompute`

**Requires**: No download. The eight fits take ~2 min the first time (measured end to end at the defaults — 96³ voxels, 25k seeds, 1500 iters — on Apple-silicon MPS; a CUDA card is faster, CPU much slower) and are cached under `~/.cache/luxar/quantum_orbitals/`; later runs load instantly. The cache is keyed by `--grid`/`--seeds`/`--iters` **and a digest of the whole fit recipe** (the orbital table, the tuning constants baked into the splats, and a `FIT_RECIPE_VERSION`), so changing any of them refits rather than silently reusing the old result.

**Demonstrates**: Analytic volume → `fit_gaussian_splats` → `volumetric` blending; **real** (tesseral) spherical harmonics, so `2px` is a genuine dumbbell along x rather than the torus a complex `Y₁¹` would give; per-splat phase coloring; stacking independent 3D fits into a navigable categorical axis with `GSplatData.combine_as_new_dimension`; a baked 3/4 opening camera via `ViewerConfig(camera=...)`.

---

#### demo_atp_synthase.py - ATP Synthase Molecular Turbine
Visualizes the complete ATP Synthase rotary motor structure with F1 catalytic head and F0 membrane rotor, all subunits color-coded.

**Run**: `luxar demo run atp_synthase`

**Demonstrates**: Molecular machine visualization, multi-subunit protein complex, color-coded structural components (alpha, beta, gamma, c-ring), biological energy production machinery (~600 kDa enzyme), depth-sorted `normal` blending for surface-like atomic structures, `layer=True` for live blending / opacity / display-range control in the Layers panel (press **L**; the absorption slider appears once the layer is switched to `volumetric`).

---

#### demo_nuclear_pore_complex.py - Nuclear Pore Complex
Downloads real Nup107-160 subcomplex structure from PDB and applies perfect 8-fold rotational symmetry to visualize the nuclear gateway.

**Run**: `luxar demo run nuclear_pore_complex`

**Requires**: Internet access (downloads PDB structure).

**Demonstrates**: PDB structure download and parsing, all-atom rendering (`--representation=calpha` for a C-alpha backbone trace instead), 8-fold rotational symmetry application, van der Waals radii for atomic sizes, CPK element colors (`--color=spoke` for one color per spoke), depth-sorted `normal` blending for surface-like atomic structures, `layer=True` for live blending / opacity / display-range control in the Layers panel (press **L**; the absorption slider appears once the layer is switched to `volumetric`).

---

#### demo_quasicrystal_3d.py - 3D Aperiodic Quasicrystal
3D quasicrystal with icosahedral symmetry using the cut-and-project method from 6D to 3D.

**Run**: `luxar demo run quasicrystal`

**Demonstrates**: Cut-and-project method (6D lattice to 3D), icosahedral symmetry (5-fold), golden ratio projection matrices, aperiodic tiling (never repeats but ordered), color-coded by perpendicular space coordinates.

---

#### demo_turing_patterns.py - 2D Turing Reaction-Diffusion Patterns
Gray-Scott reaction-diffusion system creating organic patterns (spots, stripes, spirals, labyrinths) with temporal evolution.

**Run**: `luxar demo run turing_patterns [-- --size=N] [-- --steps=N]`

**Demonstrates**: Gray-Scott model simulation, multiple pattern types via different (F, k) parameters, temporal evolution of pattern formation, categorical navigation between pattern types, emergent complexity from simple rules.

---

#### demo_bioluminescent_ocean.py - Bioluminescent Ocean
Ethereal underwater visualization with animated jellyfish, flowing tentacles, glowing plankton, and bioluminescent deep-sea atmosphere.

**Run**: `luxar demo run ocean`

**Demonstrates**: Jellyfish bell and tentacle geometry, bioluminescent color schemes (GFP-inspired), marine organism animation, atmospheric underwater effects.

---

#### demo_flywire_connectome.py - FlyWire Adult Drosophila Connectome
First complete wiring diagram of an adult animal brain: ~139k neurons as Points at their soma positions (in µm) and top-N neuron-to-neuron connections as tapered Lines inside the real brain envelope.

**Run**: `luxar demo run flywire_connectome [-- --min-synapses=20] [-- --max-edges=300000]`

**Requires**: Internet access on first run (auto-downloads ~850 MB of connectivity data from Zenodo to `~/.cache/luxar/flywire/`; subsequent runs are offline).

**Demonstrates**: Biological **network visualization** (Points + Lines in the same scene), real EM-reconstructed 3D soma coordinates, dominant-presynaptic-neurotransmitter inference via argmax of per-NT probability columns, **one toggleable Lines layer per neurotransmitter** (acetylcholine / GABA / glutamate / dopamine / serotonin / octopamine) so users can isolate excitatory, inhibitory, or neuromodulatory circuits from the viewer's Layers panel, width-tapered directed connections, super-class legend with 10 neuron categories, large spatially-embedded graph (~139k nodes, up to 300k edges). Data: FlyWire 783 release ([Schlegel et al. 2024 Nature](https://github.com/flyconnectome/flywire_annotations), [Dorkenwald et al. Zenodo 10676866](https://zenodo.org/records/10676866)).

---

#### demo_particle_collision.py - Particle Collision Detector
Realistic visualization of particle physics collisions inspired by CERN's ATLAS and CMS detectors, showing helical particle tracks, jets, and energy deposits.

**Run**: `luxar demo run collision`

**Demonstrates**: Helical trajectories from Lorentz force (F = qv x B), transverse momentum and track curvature, detector geometry (barrel + endcap), particle jets and energy deposits, magnetic field effects on charged particles.

---

#### demo_particle_collision_animated.py - Animated Particle Collision Detector
Time-animated version of the particle collision demo. Watch particle tracks grow outward from the collision vertex as time advances.

**Run**: `luxar demo run particle_collision_animated`

**Demonstrates**: Same physics as `demo_particle_collision.py` plus time dimension animation, event unfolding in real-time, 4D (XYZ + time) navigation.

---

### nD and Multi-Dimensional Demos

#### demo_4d_fractals.py - 4D Geometric Fractal Explorer
Interactive exploration of 6 different 4D geometric fractals with categorical dimension.

**Run**: `luxar demo run fractals_4d [-- --grid=64]`

**Demonstrates**: 4D spatial navigation (XYZ + W dimension, every W slider stop shows structure), categorical dimension (select between 6 fractal types), large dataset (~7M points, up to ~1.5M per fractal), XOR Fractal, Menger Sponge 4D, Sierpinski 4D, Cantor Dust 4D, Checkerboard and Diamond patterns.

---

#### demo_5d_spiral_galaxy.py - 5D Spiral Galaxy with Time Evolution
Large-scale 5D data with millions of points: multiple spiral arms evolving over time with channel-based coloring.

**Run**: `luxar demo run spiral_galaxy_5d [-- --points=N]`

**Demonstrates**: 5D data (X, Y, Z, Time, Channel), continuous and discrete dimension navigation, time-based animation of spiral arm rotation, channel variation representing different wavelengths, logarithmic spiral arm mathematics.

---

#### demo_nd_transforms.py - nD Transform Test Bench
An instrument, not a picture: a calibrated bench that makes `nd_transform` readable tick by tick. Everything sits against a ruler along X where one tick = one frame index. A cyan cursor column (plain untransformed geometry tagged `Frame = k`) marks the WORLD index on the slider; each row is a group carrying ONE `nd_transform`, and its markers are 3D point-font digits that print their own LOCAL index — so whichever digit lights up IS the local frame the inverse-query resolved to, and the gap between digit and cursor, counted in ruler ticks, IS the transform. Faint always-on ghost digits (`extend_to_all`) mark every slot a row could light, so a dark row reads as "no local frame maps to this T" rather than "the row failed to load". A `visible_range`-gated readout prints the EXPECTED local index for every row at the current slice, computed from the same definitions that placed the data — render vs readout disagreement is the failure signal.

Rows: identity, `offset +5`, `offset -3`, `scale ×2`, `scale ×2 offset +2`, a nested `×2`-then-`+1` pair that must land on the same tick as the flat row above it (a real composition-**order** test — the wrong order lights the opposite parity), and a reversing `scale -1 offset +15`. A second section does categorical permutations with colour-coded letters that carry their own local channel identity.

**Run**: `luxar demo run nd_transforms`

**Demonstrates**: `nd_transform` affine on a discrete ordinal dimension (offset, scale, negative scale, scale+offset), categorical permutation, hierarchical composition through nested groups, a spatial 4x4 `transform` and an `nd_transform` on the same group, `extend_to_all` for static furniture and per-section pinning, `visible_range` overlays as a live expected-value readout. Doubles as the visual regression harness for the viewer's no-preimage rule (see `packages/luxar-viewer/src/data/transforms/README.md`).

---

### Feature Showcase Demos

#### demo_sharpness_showcase.py - Point Sharpness Showcase
Comprehensive showcase of the point sharpness feature: gradient from peaky (0.0) to hard-edged (1.0) on the normalized knob, fixed sharpness comparison rows, mixed sharpness cloud, and sinusoidal wave pattern.

**Run**: `luxar demo run sharpness_showcase [-- --points N]`

**Demonstrates**: Sharpness parameter control (normalized [0, 1] knob mapping to a super-Gaussian falloff exponent beta=2^(6s-2); 0.5 = Gaussian), soft/peaky points (low s) vs sharp disc-like points (high s), color-coded sharpness values, multiple visualization patterns.

---

#### demo_lsystem_forest.py - L-System Tree Forest (Lines Demo)
Beautiful procedural forest using L-system grammars to showcase the **Lines** node type.

**Run**: `luxar demo run forest [-- --iterations=6] [-- --trees=16]`

**Demonstrates**: **Lines node type** as an indexed branching network with shared joint vertices, L-system grammar expansion and interpretation, width tapering (thick trunk to thin twigs), color gradients (bark to foliage), 3D branching, multiple tree varieties (elegant, fractal, willow, bush, cherry), seasonal color schemes.

---

#### demo_hilbert_curve_3d.py - 3D Hilbert Space-Filling Curve (Lines Demo)
The 3D Hilbert curve — a continuous, self-similar polyline that visits every cell of a 2^n × 2^n × 2^n grid exactly once with consecutive cells always sharing a face. Vectorized Skilling algorithm produces orders 1 through max_order (default 6 = 262k vertices), each rendered as a single thin polyline with HSV hue swept along the traversal index. A slider on the `order` dim steps through the recursion so you can watch each level subdivide and rotate.

**Run**: `luxar demo run hilbert_curve_3d [-- --max-order=6]`

**Demonstrates**: Single ultra-long `polyline` Lines node (262k+ vertices, 262k segments), thin constant width with color gradient along traversal, runtime-verified Hamiltonian path on the integer lattice, slider-driven recursion exploration (each order on its own slot of a non-displayed `order` dim), Skilling's vectorized 3D Hilbert algorithm.

---

#### demo_network_performance.py - Network Performance Testing
Large multi-cluster dataset (1M points) for testing viewer performance under network constraints.

**Run**:
```bash
luxar demo run network_performance
luxar demo run network_performance -- --profile 3g
luxar demo run network_performance -- --points=2000000 --profile satellite
```

**Demonstrates**: Network simulation (bandwidth throttling, latency, jitter, packet loss), progressive loading behavior with limited bandwidth, cache effectiveness under bandwidth constraints, multi-cluster particle systems (1M+ points).

---

### Embedding and UMAP Demos

#### demo_arxiv_paper_embeddings.py - ArXiv Paper Embeddings (Semantic Scholar)
Visualizes scientific papers in 3D embedding space. Papers cluster by topic, colored by research field, sized by citation count.

**Run**: `luxar demo run arxiv_papers`

**Requires**: Internet access, `sentence-transformers` and `umap-learn` packages. Embeds abstracts with Sentence-BERT (all-MiniLM-L6-v2) and reduces to 3D with UMAP.

**Demonstrates**: Semantic embedding of text, UMAP dimensionality reduction (768D to 3D), citation-based sizing, research field clustering, Semantic Scholar API usage.

---

#### demo_arxiv_embeddings_kaggle.py - ArXiv Paper Embeddings (Kaggle / OpenAI)
Visualizes arXiv papers using pre-computed OpenAI embeddings from the Kaggle "openai-arxiv-embeddings" dataset (2M+ papers, 3072D).

**Run**: `luxar demo run arxiv_papers_kaggle`

**Requires**: Internet access, Kaggle account (for first download, ~30 GB cached to `~/.cache/mlcroissant/`), `mlcroissant` and `umap-learn` packages. First run takes 8-15 minutes; subsequent runs use cache.

**Demonstrates**: Large-scale embedding visualization (2M+ papers), pre-computed OpenAI text-embedding-3-large, UMAP reduction, Kaggle dataset integration via mlcroissant.

---

#### demo_protein_embeddings_cafa5.py - Protein Function Landscape (ProtT5 Embeddings)
Visualizes 142k proteins from the CAFA5 challenge in 3D embedding space, showing how proteins with similar functions cluster together. The 14 landscape regions carry **derived names** rather than `Cluster 0` ... `Cluster 13`: each is named after the UniProt keyword most over-represented among its members (`Mitochondrion`, `Transit peptide`, `Transducer`, `Cell inner membrane`, ...), and a region with no keyword clearly above background reads `Mixed` instead of being given a name the data does not support. Hovering a point shows its UniProt accession plus its region.

**Run**: `luxar demo run protein_landscape`

**Requires**: Internet access, `umap-learn` package. Uses ProtT5 1024D embeddings reduced to 3D with UMAP. Naming looks ~22k accessions up in the UniProt REST API on the first run (a few minutes, then cached in `~/.cache/luxar/protein_embeddings/uniprot_keywords.json`); if UniProt is unreachable the demo still runs and falls back to generic `Cluster N` labels. Note the Kaggle bundle's own `CAFA1_train_terms.tsv` is *not* usable for annotation — it covers 1,387 PDB-style entries with zero overlap with the 142,246 UniProt accessions in `train_ids.npy`, which is why annotation is fetched rather than read from disk.

**Demonstrates**: Protein language model embeddings (ProtT5), UMAP dimensionality reduction, k-means landscape segmentation, enrichment-based cluster naming against the UniProt keyword vocabulary, hover labels, CAFA5 challenge dataset.

---

#### demo_esm3_protein_landscape.py - ESM-3 Protein Landscape (Swiss-Prot)
~572k Swiss-Prot proteins embedded with ESM-3 (or ESM C 300M) and projected to 3D with UMAP. Each point is a protein, colored by taxonomic kingdom, with hover labels showing protein name, organism, and kingdom.

**Run**: `luxar demo run esm3_protein_landscape [-- --no-serve] [-- --sample=100000] [-- --model=esmc-300m]`

**Requires**: Internet access (downloads Swiss-Prot from UniProt), `esm>=3.0.0` (in the `demos` extra); a CUDA GPU to compute embeddings (first run computes ESM embeddings + UMAP, ~5h). Subsequent runs load cached results — and load them without `torch`, `esm` or `umap-learn` installed at all, because each is demanded only at the point where the corresponding uncached computation happens rather than as an entry-point preflight. `torch` and `scipy` are optional-with-degradation on the cached path: if either one is missing, the scene is built as flat Points instead of the substitutive Points LOD ladder (that path imports `luxar.gsplats.lod`, which needs both), so it stays fully viewable — `pip install 'luxar[gsplats]'` to rebuild with LOD. If an earlier run left a quarantined `*.corrupt` artifact in `~/.cache/luxar/esm3_swissprot/`, the demo reports its path and size up front — re-download the complete file or delete the quarantined copy, otherwise the run starts over from scratch.

**Demonstrates**: Protein language model embeddings (ESM-3 / ESM C), large-scale embedding visualization (~572k proteins), UMAP dimensionality reduction, taxonomic-kingdom coloring, hover labels.

---

#### demo_cellxgene_census_umap.py - CZ CELLxGENE Census single-cell 3D UMAP (LOD stress test)
A very large 3D UMAP of human single cells from the CZ CELLxGENE Census, embedded from their **precomputed scVI latent** (50-d) with **cuML UMAP**, rendered with **substitutive Points LOD** + a categorical `coloring` dimension (cell type / tissue / disease). The shipped default builds a 3M-element scene from a 1M-cell cache; the pipeline scales to **10M cells (30M elements, a 7-level LOD ladder)** — the largest UMAP demo in the repo, built to exercise the LOD machinery.

**Run**: `luxar demo run cellxgene_census_umap [-- --no-serve]`

**Requires**: nothing extra for the default (ships a 1M-cell coords cache via Git LFS). Regenerating at scale needs a CUDA GPU with `cellxgene-census` + `cuml` (RAPIDS) — see `scripts/gen_census_umap.py` (≈96.6M primary human cells available; ~140s scVI fetch + ~14min cuML UMAP for 10M). Point the demo at a larger cache via `CENSUS_UMAP_CACHE` / `CENSUS_UMAP_MAX_CELLS` / `CENSUS_UMAP_DEVICE`.

**Demonstrates**: precomputed scVI single-cell embeddings, cuML UMAP at 10M scale, substitutive Points LOD (coarse levels as mass-preserving Gaussian splats), the `coarsen_dims` barrier (coarse splats stay pure per coloring), categorical-dimension colour switching, large-scale LOD streaming.

---

#### demo_cytoself_protein_landscape.py - CytoSelf Protein Localization 3D UMAP
~114k per-image CytoSelf embeddings from the OpenCell dataset as a 3D UMAP point cloud. Each point is a single fluorescence microscopy crop of an endogenously tagged protein, colored by subcellular localization or protein identity.

**Run**: `luxar demo run cytoself_protein_landscape [-- --no-serve] [-- --recompute]`

**Requires**: Internet access (downloads embeddings from Google Drive), `umap-learn`, `pandas`, `requests`. First run computes 3D UMAP (~10-30 min); subsequent runs load cached results.

**Demonstrates**: Self-supervised image embeddings (CytoSelf VQ-VAE-2, 9,216-dim), subcellular localization landscape, ~1,311 OpenCell proteins, categorical attribute switching (localization vs protein), UMAP dimensionality reduction.

---

#### demo_zebrahub_multiome.py - Zebrahub Integrated Cells 3D UMAP
Visualizes integrated single cells from zebrafish with categorical attribute navigation between Cell Type and Timepoint views.

**Run**: `luxar demo run zebrahub_multiome`

**Requires**: Internet access (downloads the integrated 3D-UMAP parquet from the CZ Biohub public store).

**Demonstrates**: Remote data loading, 3D UMAP embedding of single-cell data, categorical dimension navigation (cell type vs timepoint), real scientific dataset from Zebrahub.

---

#### demo_zebrahub_velocity_streamlines.py - Zebrahub 3D RNA-Velocity UMAP + Streamlines
Turns the Zebrahub VeloCyto AnnData (spliced/unspliced counts + precomputed 3D RNA-velocity UMAP embedding) into a luminous scene: cells as soft Points (colored by anatomy ontology) and RK4-integrated streamlines as Lines tracing the velocity field through UMAP space.

**Run**: `luxar demo run zebrahub_velocity_streamlines [-- --preset preview] [-- --no-serve] [-- --h5ad /path/to/zebrahub_velocity.h5ad]`

**Requires**: Internet access on first run (auto-downloads the `.h5ad` from the shared Zebrahub Google Drive into `~/.cache/luxar/zebrahub_velocity/`), `anndata>=0.10,<0.13`, `h5py`, `gdown`, `scipy`. The anndata upper bound is load-bearing: anndata >= 0.13 requires `zarr>=3.1`, which conflicts with Luxar's `zarr>=2.16,<3.0` pin — installing `luxar[demos]` applies the constraint for you.

**Demonstrates**: RNA-velocity visualization (Points + Lines together), per-cell velocity binned into a regularized smoothed cubic vector field, vectorized RK4 streamline integration through UMAP space, categorical anatomy-ontology coloring, stratified streamline seeding.

---

#### demo_human_multiome_peak_umap.py - Human Multiome Peak 3D UMAP
3D UMAP embedding of ~1M single-cell ATAC-seq peaks, color-coded by cell type, lineage, timepoint, and other attributes.

**Run**: `luxar demo run human_multiome_peak_umap`

**Requires**: Local parquet data file, `pandas` package. `luxar[gsplats]` (torch + scipy) is optional — without it the scene builds as a flat, fully viewable point cloud instead of the Points LOD ladder.

**Demonstrates**: Large-scale single-cell visualization (~1M points), multiple categorical attributes (cell type, lineage, timepoint, peak type, chromosome), ATAC-seq chromatin accessibility data.

---

#### demo_mouse_multiome_peak_umap.py - Mouse Multiome Peak 3D UMAP
3D UMAP embedding of ~192k single-cell ATAC-seq peaks from mouse embryonic development (E7.5-E8.75), color-coded by cell type, lineage, and timepoint.

**Run**: `luxar demo run mouse_multiome_peak_umap`

**Requires**: Local parquet data file, `pandas` package. `luxar[gsplats]` (torch + scipy) is optional — without it the scene builds as a flat, fully viewable point cloud instead of the Points LOD ladder.

**Demonstrates**: Single-cell ATAC-seq visualization, embryonic developmental timepoints, multiple categorical attribute navigation, lineage-based coloring.

---

#### demo_zebrahub_multiome_peak_umap.py - Zebrahub Multiome Peak 3D UMAP
3D UMAP of 640k single-cell chromatin accessibility peaks from the Zebrahub project, with 30 cell types across 6 developmental stages.

**Run**: `luxar demo run zebrahub_multiome_peak_umap`

**Requires**: Internet access (downloads from CZ Biohub public zarr store). `luxar[gsplats]` (torch + scipy) is optional — without it the scene builds as a flat, fully viewable point cloud instead of the Points LOD ladder.

**Demonstrates**: Remote zarr data loading, 640k points with 30 cell types, 7 categorical attributes (cell type, chromosome, lineage, peak type, etc.), 6 developmental timepoints.

---

#### demo_chromatrace_choir_umap.py - Chromatrace CHOIR 3D UMAP (Cell-Type Atlas)
3D UMAP of ~60k single cells annotated with CHOIR bio-terms (88 fine-grained cell types) and bio-groups (8 broad lineages: Neuroectoderm, Neural Crest, Craniofacial Mesenchyme, Cardiac/Vascular, Mesoderm, Endoderm, Epidermis, Other). Two runtime-switchable color views with curated CHOIR palette and grouped HTML legend.

**Run**: `luxar demo run chromatrace_choir_umap [-- --data /path/to/zip-or-folder]`

**Requires**: User-provided `choir_umap_3d_viewer.zip` bundle (parquet + colormap JSON) — auto-detected in `~/Downloads/` or `~/.cache/luxar/chromatrace/`.

**Demonstrates**: Categorical attribute switching with two color views, curated upstream palette mapping, grouped two-column HTML legend, custom hover overlay (suppresses default top-right tooltip), large categorical dimension (88 unique cell types).

---

#### demo_tabula_sapiens.py - Tabula Sapiens Human Single-Cell Atlas UMAP
The Tabula Sapiens first-draft human cell atlas (~500k cells from 24 tissues of 15 donors) as a 3D UMAP embedding. Each point is a cell, colored by organ of origin, with hover labels showing cell type and tissue.

**Run**: `luxar demo run tabula_sapiens [-- --no-serve] [-- --sample=50000]`

**Requires**: Internet access (downloads from CZ CELLxGENE Discover), `pandas`, `umap-learn`, `scipy` (installed via `luxar[demos]`).

**Demonstrates**: Human single-cell transcriptomics atlas, 3D UMAP embedding, organ-of-origin categorical coloring, cell type / tissue hover labels, CELLxGENE Discover data integration.

---

#### demo_spotify_tracks.py - Spotify Tracks 3D UMAP of Audio Features
~114k Spotify tracks embedded into 3D via UMAP on 9 audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo). Colored by genre, sized by popularity, with track/artist/genre hover labels.

**Run**: `luxar demo run spotify_tracks [-- --no-serve] [-- --sample=50000]`

**Requires**: Internet access (downloads dataset CSV from Hugging Face), `pandas`, `umap-learn` (installed via `luxar[demos]`).

**Demonstrates**: UMAP on tabular audio features, genre-based categorical coloring, popularity-based point sizing, hover labels, open Hugging Face dataset integration.

---

#### demo_chromatrace_choir_umap_sequence.py - Chromatrace CHOIR UMAP — Cell-Type Walkthrough
Variant of the Chromatrace demo with an 89-step slider stepping through each fine-grained CHOIR bio-term one at a time. Within each bio-group, cell types are reordered (nearest-neighbor + 2-opt TSP on UMAP centroids) so consecutive slots are spatially adjacent. Backdrop layer keeps the full UMAP outline visible across all slots; highlight layer shows only the active type in its CHOIR palette color.

**Run**: `luxar demo run chromatrace_choir_umap_sequence [-- --no-tsp] [-- --data /path/to/zip-or-folder]`

**Requires**: Same `choir_umap_3d_viewer.zip` bundle as `demo_chromatrace_choir_umap.py`.

**Demonstrates**: `extend_to_all` for shared backdrop across slider slots (60k points instead of 89×60k), per-slot `visible_range` overlays, two-layer toggleable scene (Backdrop + Highlight), explicit `CameraConfig` to bypass auto-fit on a high-cardinality non-spatial dim, TSP-optimized intra-group ordering for smooth cross-slot transitions.

---

### Data-Driven Demos (External Datasets)

#### demo_global_rivers_earth.py - Rivers of Earth
A topographic ETOPO globe (Points) plus every HydroRIVERS reach (Lines) in geographic 3D.

**Run**: `luxar demo run global_rivers_earth`

**Demonstrates**: Mixed Points+Lines geometry in one scene, geographic (lat/lon/elevation) coordinate mapping, large real-world datasets with local caching (~1 GB download on first run).

---

#### demo_ocean_currents_earth.py - Ocean Currents of Earth
HYCOM surface-current streamlines (220k connected ribbons, coloured by speed) draped over a jittered-Fibonacci NASA Blue Marble globe — a "Perpetual Ocean"-style visualization of the Gulf Stream, Kuroshio, and Antarctic Circumpolar Current.

**Run**: `luxar demo run ocean_currents_earth`

**Requires**: Internet access on first run (~72 MB: HYCOM GLBy0.08 surface u/v + Blue Marble texture; cached under `~/.cache/luxar/ocean_currents_earth/`).

**Demonstrates**: Mixed Points+Lines geometry, fixed-arc-length RK4 streamline advection with along-segment land masking, per-vertex RGBA comet-tail fading, indexed Lines topology under the viewer's per-node vertex ceiling, `stream:` additive LOD for fast first paint on an 8M-point globe.

---

#### demo_gaia_milky_way_3m.py - Milky Way Stars (Gaia DR3, 3M Stars)
Real Milky Way stars from Gaia DR3: top 3M brightest stars with real photometric colors (BP-RP index), galactocentric coordinates, and reference markers (Sun, Betelgeuse, Rigel).

**Run**: `luxar demo run galaxy`

**Requires**: Internet access (Gaia DR3 TAP query or cached data).

**Demonstrates**: Real astronomical data (Gaia space telescope), 3M star dataset, BP-RP photometric color-to-RGB conversion, galactocentric coordinate system, magnitude-dependent point radii.

---

#### demo_earthquakes_3d.py - Global Earthquake Visualization
Real-time earthquake data from USGS plotted on a 3D Earth sphere with vertical spikes showing magnitude and color-coded by time.

**Run**: `luxar demo run earthquakes`

**Requires**: Internet access (USGS earthquake API).

**Demonstrates**: Real-time data from USGS API, spherical Earth projection (lat/lon to XYZ), magnitude-based spike height (logarithmic Richter scale), time-based coloring, plate boundary visualization.

---

#### demo_cosmicflows_laniakea.py - Cosmicflows-4 Laniakea Flow Field
Recreates the Cosmicflows-4 / Laniakea visualization by Simone Conradi and Manlio De Domenico: 55,486 local-universe galaxies plus colored streamlines tracing matter flow through basins of attraction.

**Run**: `luxar demo run cosmicflows_laniakea [-- --preset preview|full]`

**Requires**: Internet access on first run (downloads ~26 MB from the public `manlius/laniakea` GitHub data cache; reused from `~/.cache/luxar/laniakea/`). The full preset reproduces the reported 29,555 valid streamlines and writes a large line dataset; `--preset preview` is faster for iteration.

**Demonstrates**: Real astronomical catalogs, supergalactic coordinates, basin-of-attraction coloring, vectorized RK4 streamline integration, large indexed Lines layers, HDR additive rendering, layer toggles per basin. Data: Cosmicflows-4 / EDD and the open [`manlius/laniakea`](https://github.com/manlius/laniakea) pipeline.

---

#### demo_desi_galaxies.py - DESI DR1: The Cosmic Web in 3D (~9.75M galaxies & quasars)
The large-scale structure of the Universe as a point cloud from the Dark Energy Spectroscopic Instrument's first data release. Each point is a real galaxy or quasar with a measured spectroscopic redshift; the redshift becomes a comoving distance so sky position + depth give true 3D Cartesian coordinates in megaparsecs. You sit at the observer's origin looking out at the two DESI footprint caps fanning into filaments, voids, and the baryon-acoustic shells. Two colorings toggle in the Layers panel: **by tracer** (BGS/LRG/ELG/QSO populations, naturally layered by distance) and **by redshift** (continuous depth colormap).

**Run**: `luxar demo run desi_galaxies [-- --recompute]`

**Requires**: Nothing extra by default — ships a compact precomputed point cloud (quantized XYZ + redshift + tracer id) via Git LFS. With `--recompute` (or if the LFS asset isn't pulled) it auto-downloads the ~1 GB of DR1 LSS clustering catalogs to `~/.cache/luxar/desi_galaxies/` (resumable), reads them with `astropy`, and converts (RA, Dec, z) → comoving Mpc. Adds `astropy` to the `demos` extra. The built scene (with substitutive LOD) is cached in the demos output dir, so only the first launch pays the LOD-build cost. If the DESI data host is unavailable, check `https://data.desi.lbl.gov/`, run `git lfs pull`, and rerun without `--recompute` to use the shipped scene without downloading the source catalogs.

**Demonstrates**: Real spectroscopic-survey catalogs → a 3D cosmic-web Points cloud, `(RA, Dec, redshift)` → comoving-Mpc conversion via `astropy.cosmology` (DESI fiducial ΛCDM), substitutive Points LOD at ~9.75M points, dual coloring (categorical tracer vs. continuous redshift colormap) via layer toggles, HDR additive rendering, self-contained download → convert → cache-processed bootstrap. Data: [DESI DR1](https://data.desi.lbl.gov/doc/releases/dr1/) (DESI Collaboration 2025, arXiv:2503.14745; CC BY 4.0).

---

#### demo_asteroids_solar_system.py - The Solar System (~1.5M Real Asteroids, JPL SBDB)
Every catalogued minor planet placed in real 3D space by propagating its measured Keplerian orbit to a common epoch: ~1.5M asteroids as Points colored by semi-major axis, plus the eight planets, the Sun, and the planets' orbit ellipses (Lines). The main belt, Kirkwood gaps, Hilda triangle, and Jupiter Trojan clouds all emerge from the real orbital-element distribution. The asteroid `intensity` is reserved for the colormap's `[0, ~11 AU]` scalar window, while density-compensated opacity keeps aggregate additive weight stable across full, `--max-asteroids`, and animated builds. The opening camera orbits the Sun and frames the outer planets instead of fitting sparse distant-object outliers.

**Run**: `luxar demo run asteroids_solar_system [-- --animate] [-- --max-asteroids N]`

**Requires**: Internet access on first run (downloads the JPL SBDB catalog, ~140 MB, to `~/.cache/luxar/asteroids/`; the parsed catalog is then cached as `.npz`, so subsequent runs are offline).

**Demonstrates**: Real planetary-science data (NASA/JPL Small-Body Database), on-the-fly orbital mechanics (vectorized Kepler solve, element→Cartesian, mean-anomaly propagation — no astropy), ~1.5M-point cloud with `scalars`+colormap coloring and per-point hover labels, Points + Lines (planet orbits) in one scene, optional time-dimension animation (`--animate`) advancing every body along its orbit. Data: [JPL SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html) (public domain).

---

#### demo_dipc_3d_genome.py - Single-Cell 3D Genome (Dip-C)
The folded 3D structure of one human cell's genome from Dip-C (Tan et al. 2018, Science): the chromosomes coil through the nucleus as Lines, colored by chromosome, with the maternal and paternal genomes exposed as two toggleable Layers (press **L**) to isolate one copy or overlay both.

**Run**: `luxar demo run dipc_3d_genome [-- --recompute]`

**Requires**: Nothing extra by default — ships a small precomputed structure via Git LFS. With `--recompute` (or if the LFS asset isn't pulled) it auto-downloads the GEO archive (`GSE117876_RAW.tar`, ~4.7 GB) to `~/.cache/luxar/dipc_genome/`, extracts one cell's `.3dg`, and caches the processed `.npz`.

**Demonstrates**: 3D-genomics visualization (nothing else renders single-cell genome folding interactively), Lines with per-vertex color and hover labels, one Lines node per haplotype exposed as a **Layers-panel** toggle (a hard visibility on/off — the reliable way to isolate Lines, since a non-displayed dimension can't cull already-loaded polylines), self-contained download → extract → cache-processed bootstrap. Data: [Tan et al. 2018](https://doi.org/10.1126/science.aat5641), GEO GSE117876; [dip-c format](https://github.com/tanlongzhi/dip-c).

---

#### demo_dmri_tractography.py - Human White-Matter Tractography (HCP-1065)
The wiring of the human brain: all 87 named white-matter tracts of the HCP-1065 population atlas as 252,978 streamlines (7.08M vertices, 6.83M segments) in ICBM 2009a space, coloured by the standard diffusion-MRI direction convention (red = left-right, green = anterior-posterior, blue = inferior-superior). Each tract is its own Lines node exposed as a **Layers-panel** toggle (press **L**), so any bundle can be isolated — the corticospinal tract, the corpus callosum, the arcuate.

**Run**: `luxar demo run dmri_tractography [-- --per-bundle 6000 --points 28]`

**Requires**: Internet access on first run — downloads `hcp1065_avg_tracts_trk.zip` (588 MB) to `~/.cache/luxar/dmri_tractography/` and caches the decoded bundles as an `.npz`. Needs `nibabel` (in the `demos` extra). No GPU. Building the scene takes ~25 min because every node lifts its segments to Gaussian beads for the substitutive LOD; it is a one-time cost, paid again only on `--recompute`.

**Demonstrates**: Lines as the *native* geometry for data that is already made of curves — no conversion, unlike every volumetric demo. Arc-length resampling (the source is ~0.4 mm-sampled, ~10x finer than any rendered line width); `line_type="indexed"` with per-streamline contiguous vertices so thick tubes render seamless joints; 87 separate nodes, each sized to stay under both the un-laddered-leaf gate (200K vertices) and the per-node element-texture segment bound; per-node **substitutive LOD** at `compression_factor=256` — thin lines need a far larger K than the default, because the bead lift is driven by arc-length ÷ width rather than by segment count (at K=4 the "coarse" level comes out 5.6x heavier than the fine one, and the scene balloons to 2.1 GB); and `blending_mode="additive"` at low opacity with a display window, which is order-independent and so cannot pop as the camera orbits — unlike `normal`, whose per-object transparent-pass sort flips between overlapping bundles. Data: [Yeh 2022](https://doi.org/10.1038/s41467-022-32595-4), [HCP-1065 atlas](https://brain.labsolver.org/hcp_trk_atlas.html) (CC BY-SA 4.0; WU-Minn HCP data-use terms).

---

#### demo_storm_3d_microtubules.py - 3D STORM Super-Resolution Microscopy
Microtubule cytoskeleton at nanometer resolution using real STORM super-resolution microscopy localizations as Gaussian splats.

**Run**: `luxar demo run storm_3d_microtubules`

**Requires**: Internet access (downloads STORM localization data).

**Demonstrates**: STORM/PALM super-resolution data (~20 nm resolution), localization uncertainty as Gaussian splat size, 3D astigmatism-based z encoding, photon count-based coloring, microtubule cytoskeleton structure.

---

#### demo_huri_interactome.py - HuRI Human Reference Interactome (Network)
~8k human proteins as Points and ~50k direct Y2H protein-protein interactions as Lines, laid out in 3D by graph structure alone (spectral embedding + UMAP). Nodes colored by Louvain community or HGNC chromosome; intra-community edges colored by community, cross-community "bridge" edges in dim gray. Rich edge hovers show the pair, intra/cross label, and — when available — the shared CORUM complex.

**Run**: `luxar demo run huri_interactome`

**Requires**: Internet access (downloads ~35 MB one-time: HuRI + HGNC + optional CORUM), `networkx>=3.0`, `umap-learn`, `scipy`, `pandas`, `requests`.

**Demonstrates**: First-class **network** visualization in Luxar (Points + Lines together), graph-derived layout (normalized Laplacian eigenvectors → UMAP), Louvain community detection, categorical attribute switching between two color views, edge-level hover labels with biological context (CORUM shared complexes), symmetric edge rendering (undirected, equal both endpoints).

---

#### demo_ppi_flow_field.py - HuRI PPI Flow Field (Signed Network Streamlines)
Turns the HuRI protein-protein interaction graph into a continuous 3D flow landscape. PageRank orients each interaction from lower-centrality to higher-centrality protein; UMAP embeds a signed sparse adjacency / flow-profile matrix so upstream and downstream roles separate; a cubic vector field blends nearby oriented edges with a regularized `1/d^3` weight; Gaussian-smoothed streamlines are seeded from proteins and advected forward through the volume.

**Run**: `luxar demo run ppi_flow_field [-- --preset preview|full]`

**Requires**: Internet access on first run (reuses the HuRI/HGNC cache in `~/.cache/luxar/huri/`), `networkx>=3.0`, `umap-learn`, `scipy`, `pandas`, `requests`. The default `full` preset builds the requested 256³ vector field and caches a larger `.npz` file; use `--preset preview` for a faster 128³ authoring run.

**Demonstrates**: Directed biological **network flow** visualization, PageRank-oriented interactions, signed sparse adjacency UMAP, KD-tree-accelerated point-to-edge vector-field construction, regularized inverse-cubic edge weighting, Gaussian vector-field smoothing, vectorized RK4 streamline advection, optional tapered directed PPI edge layer, HDR additive Lines rendering.

---

#### demo_caida_as_topology.py - CAIDA AS Topology (The Internet as a Graph)
~80k Autonomous Systems as Points and ~350k BGP relationships as Lines, laid out in 3D from graph structure alone. Auto-fetches the latest CAIDA serial-2 snapshot. Edges are styled by RELATIONSHIP TYPE — warm amber tapered lines for provider-customer (thick at provider, thin at customer, encoding direction), cool cyan uniform lines for peers. Nodes colored by Louvain community or country (CAIDA as2org). Tier-1 ASes (no upstream providers) get a size bump so the backbone pops. Edge hovers narrate the relationship ("Tier-1 transit" / "Tier-1 peering" / "Provider → Customer" / etc.) with both endpoints' org names and countries.

**Run**: `luxar demo run caida_as_topology`

**Requires**: Internet access (auto-downloads ~60 MB from CAIDA on first run), `networkx>=3.0`, `umap-learn`, `scipy`, `pandas`, `requests`.

**Demonstrates**: Large-scale **directed network** visualization (~80k nodes), aesthetic-tuned graph layout pipeline (50-dim spectral embedding → UMAP with n_neighbors=30, metric='cosine', spread=2.0 → PCA realignment for consistent orientation), edges styled by relationship kind (tapered-directional vs symmetric), tier-1 backbone detection from graph topology, per-edge narrative hover labels, auto-discovery of latest upstream dataset snapshots.

---

### GSplats Demos (Gaussian Splatting)

#### demo_gsplats_3d_tng_cosmic_web.py - IllustrisTNG Cosmic Web
IllustrisTNG TNG300 dark-matter density (244M particles) as Gaussian splats — the cosmic web.

**Run**: `luxar demo run gsplats_3d_tng_cosmic_web`

**Demonstrates**: Fetch-raw-and-process-locally for access-gated data (IllustrisTNG requires an account + API key; nothing is redistributed), GPU gsplat fitting of a cosmological density field, LOD topologies for very large splat counts.

---

#### demo_gsplats_2d_codex_pancreas.py - 2D CODEX Pancreas (12-Channel Multiplexed Fluorescence)
12-channel multiplexed immunofluorescence image of human pancreas tissue as 2D Gaussian splats with per-channel colors, using tiled fitting.

**Run**: `luxar demo run gsplats_2d_codex_pancreas`

**Requires**: Internet access (downloads from Zenodo), GPU recommended. 476 megapixels per channel (25,816 x 18,440 px).

**Demonstrates**: Tiled 2D Gaussian splatting with Hann cosine apodization, 12-channel CODEX multiplexed fluorescence, per-channel biologically meaningful colors, tile merging for large images, Zenodo data download.

---

#### demo_gsplats_2d_cmu1_pathology.py - 2D Whole-Slide Pathology Image (CMU-1)
Gigapixel H&E-stained whole-slide pathology image as 2D Gaussian splats with per-channel (R/G/B) colors, using tiled fitting.

**Run**: `luxar demo run gsplats_2d_cmu1_pathology`

**Requires**: Internet access (downloads from OpenSlide test data, ~169 MB), GPU recommended. 1.5 gigapixels (46,000 x 32,914 px).

**Demonstrates**: Tiled 2D Gaussian splatting on gigapixel data, brightfield H&E histology (brightness inversion), R/G/B channel splitting, Aperio SVS whole-slide image format.

---

#### demo_gsplats_3d_organoid_dapi_nuclei.py - 3D Organoid DAPI Nuclei
Gaussian splatting compression of real 3D confocal microscopy data (DAPI-stained cell nuclei) from Image Data Resource (IDR).

**Run**: `luxar demo run gsplats_3d_organoid_dapi_nuclei`

**Requires**: Internet access (downloads OME-ZARR from IDR). Supports Metal (MPS) acceleration on Apple Silicon.

**Demonstrates**: 3D Gaussian splat fitting to real microscopy volumes, 20-50x compression vs raw voxels, oriented ellipsoids capturing elongated nuclear shapes, IDR/OME-ZARR data loading, `plasma` colormap + ACES tone-mapping.

---

#### demo_gsplats_3d_organoid_multichannel.py - Multi-Channel 3D Organoid (IDR Pipeline)
Multi-channel 3D microscopy data as Gaussian splats, with full compute pipeline from Image Data Resource (IDR). Uses precomputed gsplats from Git LFS by default.

**Run**: `luxar demo run gsplats_3d_organoid_multichannel [-- --recompute]`

**Requires**: Git LFS data (default) or internet access + GPU (with `--recompute`).

**Demonstrates**: Multi-channel Gaussian splatting with distinct colors per channel, additive blending so the two superimposed channels mix instead of occluding each other, full pipeline (fetch from IDR, fit per channel, merge), precomputed gsplats via Git LFS for fast demo.

---

#### demo_gsplats_3d_cells3d_multichannel.py - 3D Multi-Channel Cells (Layers + BOP LUTs)
Two-channel scikit-image `cells3d` fluorescence volume (membranes + nuclei) fitted per channel as 3D splats and shown as toggleable **layers**, each coloured by a BOP (Blue-Orange-Purple) microscopy LUT (`bop_orange` membranes, `bop_blue` nuclei). Fully self-contained (skimage downloads the sample) — the cheapest gsplat demo to run from scratch. Uses precomputed gsplats from Git LFS by default.

**Run**: `luxar demo run gsplats_3d_cells3d_multichannel [-- --recompute]`

**Requires**: Git LFS data (default) or `scikit-image` + GPU (with `--recompute`).

**Demonstrates**: Per-channel `layer=True` gsplats nodes with built-in BOP LUTs applied at display time (interactive colormap switching in the Layers panel, press L), shared amplitude-weighted centroid alignment, volumetric blending, ACES tone-mapping. The lightweight, no-download sibling of `organoid_multichannel` and `kidney_multichannel_layers`.

---

#### demo_gsplats_3d_kidney_multichannel_toggles.py - 3D Multi-Channel Kidney (Boolean Toggles)
Three-channel confocal mouse kidney tissue (nuclei, WGA, actin) with independent boolean toggle dimensions for each channel.

**Run**: `luxar demo run gsplats_6d_kidney_multichannel_toggles`

**Requires**: `scikit-image` package (for `kidney()` sample data, FluoCells Prepared Slide #3).

**Demonstrates**: Three-channel boolean toggles (8 visibility combinations), `extend_to_all` for cross-channel visibility, confocal fluorescence microscopy data, comparison with Layers panel approach.

---

#### demo_gsplats_3d_kidney_multichannel_layers.py - 3D Multi-Channel Kidney (Layers Panel)
Same kidney dataset as the toggles variant, but uses the **Layers panel** (`layer=True`) instead of boolean dimensions for per-channel visibility control.

**Run**: `luxar demo run gsplats_3d_kidney_multichannel_layers`

**Requires**: `scikit-image` package.

**Demonstrates**: Layers panel (press **L**) with visibility toggle, display range, gamma, and blending mode per channel. Keeps the scene at 3D (no extra nD dimensions) while providing rich per-channel controls.

---

#### demo_gsplats_3d_acto3d_heart.py - 3D Mouse Embryo Heart (Acto3D / Zeiss Lightsheet 7)
Three-channel light-sheet microscopy volume of an E13.5 mouse embryo heart as Gaussian splats with per-channel colors.

**Run**: `luxar demo run gsplats_3d_acto3d_heart`

**Requires**: Internet access (downloads ~1.65 GB TIFF from Google Drive), GPU recommended.

**Demonstrates**: Multi-channel light-sheet microscopy, 960 x 960 x 597 voxel volume, per-channel Gaussian splatting with distinct colors, Zeiss Lightsheet 7 data.

---

#### demo_gsplats_3d_opencell_map4.py - OpenCell MAP4 Cytoskeleton
Two-channel confocal z-stack of endogenously tagged MAP4 (microtubule-associated protein 4) in HEK293T cells from the OpenCell project, with Hoechst nuclear stain.

**Run**: `luxar demo run gsplats_3d_opencell_map4`

**Requires**: Internet access (downloads from OpenCell, ~70 MB).

**Demonstrates**: Multi-channel Gaussian splatting (MAP4-GFP + Hoechst), spinning-disk confocal microscopy, OpenCell project data, cytoskeleton visualization, 51 x 600 x 600 voxel volume.

---

#### demo_gsplats_3d_tribolium_embryo.py - 3D Tribolium Embryo (Cell Tracking Challenge)
Large isotropic 3D light-sheet volume of a developing beetle (*Tribolium castaneum*) embryo using Gaussian splatting. Near-isotropic at 0.381 um per voxel.

**Run**: `luxar demo run gsplats_3d_tribolium_embryo`

**Requires**: Internet access (downloads ~2.6 GB from Zenodo), GPU recommended. 965 x 1871 x 991 voxels.

**Demonstrates**: Large-volume Gaussian splatting, isotropic light-sheet microscopy, Zenodo/Cell Tracking Challenge data, Zeiss LightSheet Z.1 data, `normal` (peak-projection) blending to keep this volume's heavy diffuse background out of the way instead of accumulating it along every ray.

---

#### demo_gsplats_3d_milky_way_dust.py - 3D Interstellar Dust of the Solar Neighborhood (Leike & Enßlin 2020)
Gaussian-splats a real 3D reconstruction of the Milky Way's interstellar dust around the Sun — the splat pipeline applied to astrophysics rather than microscopy. The "volume" is a cube of *space*: a 740 × 740 × 540 pc reconstruction of dust extinction density at 1 pc resolution, fit into glowing 3D fog revealing the Local Bubble and the Orion / Taurus / Perseus molecular clouds.

**Run**: `luxar demo run gsplats_3d_milky_way_dust [-- --recompute]`

**Requires**: Nothing extra by default — ships a precomputed **full-resolution** fit via Git LFS (~8 MB: the native 740×740×540 cube fit to ~675k splats, PSNR ~35 dB). With `--recompute` (or if the LFS asset isn't pulled) it auto-downloads the 2.4 GB reconstruction (`mean_std.h5`) to `~/.cache/luxar/gsplats_milkyway_dust/` (resumable) and refits. The `--recompute` default reproduces the shipped full-res fit and needs a large-VRAM GPU (~40 GB+); on a smaller card pass `--target-size 256 --max-splats 200000` for a lighter downscaled refit.

**Demonstrates**: Real *volumetric astronomy* → Gaussian splats (the same `cal → fit → convert` pipeline used for microscopy, on a dust-density cube), 20–50× compression, `inferno` colormap + ACES tone-mapping + light volumetric HDR rendering (absorption κ=0.3, so near dust softly occludes far dust), self-contained download → fit → cache-processed bootstrap. Data: [Leike, Glatzle & Enßlin 2020](https://doi.org/10.1051/0004-6361/202038169), A&A 639, A138 (Zenodo record 3993082, CC BY 4.0).

---

#### demo_gsplats_3d_visible_human_head.py - Visible Human Head (Real-Color Anatomy)
The human head Gaussian-splatted in **true photographic color** from the NLM Visible Human Project cryosections — actual photographs of a frozen cadaver sliced at 1 mm, so brain, skull, muscle and vasculature appear in natural anatomical color (not a false-color transfer function). The splat pipeline applied to real photographic volumetric anatomy.

**Run**: `luxar demo run gsplats_3d_visible_human_head [-- --recompute]`

**Requires**: Nothing extra by default — ships a precomputed fit + per-splat colors via Git LFS. With `--recompute` (or if the LFS assets aren't pulled) it auto-downloads the 377 color head slices (~1.1 GB) to `~/.cache/luxar/gsplats_visible_human_head/`, builds the masked RGB volume, fits luminance (GPU), and samples per-splat colors.

**Demonstrates**: True-color volumetric anatomy → Gaussian splats via a **single luminance fit + per-splat color sampling** (one fit, real photographic color — vs. the scalar-intensity-plus-colormap microscopy demos), warm-vs-blue tissue masking to drop the frozen-gel background, ACES tone-mapping, self-contained download → mask → fit → cache-processed bootstrap. Data: [NLM Visible Human Project](https://www.nlm.nih.gov/research/visible/visible_human.html) (Male color cryosections, head subset; public domain).

---

#### demo_gsplats_3d_cryoem_virus.py - Cryo-EM Giant Virus Capsid (Structural Biology)
Gaussian-splats a real cryo-electron-microscopy density map from the EMDB: the icosahedral capsid of *Paramecium bursaria* chlorella virus 1 (PBCV-1), a giant virus. The reconstructed electron-density volume — a hollow ~1650 Å shell tiled with capsomers — is exactly what the Luxar splat fitter eats, so this is the microscopy splat pipeline applied to structural biology (no isosurface threshold needed).

**Run**: `luxar demo run gsplats_3d_cryoem_virus [-- --recompute]`

**Requires**: Nothing extra by default — ships a precomputed fit via Git LFS (~12 MB: the 700³ EMDB map downsampled to 512³, fit to ~1.0M splats, PSNR ~28 dB). With `--recompute` (or if the LFS asset isn't pulled) it auto-downloads the 1.3 GB EMDB map (`emd_5384.map.gz`) to `~/.cache/luxar/gsplats_cryoem_virus/` (resumable), reads it with `mrcfile`, downsamples to 512³, and fits Gaussian splats on the GPU. Adds `mrcfile` to the `demos` extra.

**Demonstrates**: Real *structural-biology* electron density → Gaussian splats (the same `cal → fit → convert` pipeline used for microscopy, on an EMDB MRC/CCP4 map), solvent clipping + percentile normalization, `inferno` colormap (starts at black, so empty space stays black) + ACES tone-mapping + volumetric HDR rendering (absorption κ=5, so the near shell occludes the far one), self-contained download → read → fit → cache-processed bootstrap. Data: [EMDB EMD-5384](https://www.ebi.ac.uk/emdb/EMD-5384) (Zhang et al. 2011, PNAS 108(36):14837; public domain / CC0).

---

#### demo_gsplats_3d_ct_totalsegmentator.py - CT Anatomical Atlas (neck-to-pelvis, Organs in Color)
Gaussian-splats a real clinical CT scan — a neck-to-pelvis study (the fullest coverage routine CT offers; the head and distal limbs are outside the scan) — and colors every splat by the anatomical structure it belongs to, using the TotalSegmentator dataset's 117-organ segmentation. The result is a glowing, rotatable atlas — white skeleton, red great vessels, cyan lungs, and colored abdominal organs in their true 3D positions. The same **one-fit + per-splat label sampling** idea as the Visible Human head demo, but the sampled organ label drives the color (a tissue-grouped palette), a **hover tooltip** (the specific structure name — all 117 tissue types), and a split into **toggle-able Layers-panel groups** (Skeleton / Organs / Vessels & heart / Nervous system / Muscles, the last a faint boosted context) — the splat pipeline applied to clinical radiology.

**Run**: `luxar demo run gsplats_3d_ct_totalsegmentator [-- --recompute]`

**Requires**: Nothing extra by default — ships a precomputed fit + per-splat organ labels via Git LFS (~8 MB: a neck-to-pelvis subject at 1.5 mm, fit to ~0.66M splats, PSNR ~43 dB; colors, layers and hover tooltips are all derived from the labels at scene build). With `--recompute` (or if the LFS assets aren't pulled) it auto-downloads the 3.2 GB TotalSegmentator subset to `~/.cache/luxar/gsplats_ct_totalsegmentator/` (resumable), extracts one subject, combines its 117 organ masks with `nibabel`, windows + fits on the GPU, and samples the per-splat organ label. Adds `nibabel` to the `demos` extra.

**Demonstrates**: Real *clinical CT* (neck-to-pelvis) + multi-organ segmentation → colored Gaussian splats, combining per-structure NIfTI masks into one label volume, Hounsfield windowing, per-splat organ-label sampling driving a tissue-grouped color palette + **per-splat hover tooltips** (specific structure names) + a split into **toggle-able tissue layers** (Skeleton/Organs/Vessels & heart/Nervous system/Muscles), cubic-voxel resampling, ACES tone-mapping + volumetric HDR rendering, self-contained download → combine → fit → cache-processed bootstrap. Data: [TotalSegmentator](https://zenodo.org/records/10047263) (Wasserthal et al. 2023, Radiology: AI; CC BY 4.0).

---

#### demo_gsplats_lod_tribolium.py - Adaptive Level of Detail on the Tribolium Embryo
Takes the precomputed Tribolium embryo fit and builds an **adaptive Level of Detail (LOD)** pyramid on it — the embryo is stored at several resolutions, and the viewer shows the simplest one that still looks right at the current zoom. This demo uses *substitutive* LOD (each coarser level *replaces* the finer one with fewer, larger splats), and ships it with per-level debug colors (green→amber→red, finest→coarsest) so the viewer's `coverage_fraction` level-switching is visible as you zoom. The scaled-up companion to `examples/gsplats_lod_example.py`.

**Run**: `luxar demo run gsplats_lod_tribolium`

**Requires**: Precomputed Tribolium splats (Git LFS); no network/GPU needed for the default path. `--recompute` re-fits from Zenodo (network + GPU).

**Demonstrates**: Substitutive LOD via `make_substitutive_lod` (`kmeans_lloyd`), `add_gsplats_from_data(lod_group=True)`, auto level-count from base splat count (#levels scales as log_K(N)), per-level debug coloring, `coverage_fraction` LOD selection in the viewer. Options: `--levels=N`, `--factor=K`, `--method=NAME`, `--serve-only`.

---

#### demo_gsplats_lod_embryo_line.py - Near-Unlimited Scaling with Adaptive Level of Detail
Lays out `--count` (default 100) copies of the single adaptive-detail Tribolium embryo along a straight line, drops the camera near the middle of the line, and lets you fly down it. **Level of Detail (LOD)** is the idea that makes this scale: each embryo is kept at several resolutions, and the viewer picks — per object, every frame, by viewport-relative `coverage_fraction` (how much of the screen it covers) — the simplest version that still looks right. Near embryos render fine (green) while distant ones collapse to a few big splats (red), so the detail actually drawn stays roughly bounded by what the screen can resolve, no matter how long the line. All copies share **byte-identical splat arrays** (per-embryo orientation/jitter lives only in scene-graph transforms), so the encoder's `array_ref` deduplication stores the geometry once: 100 embryos cost ~55 MB on disk instead of ~1.3 GB.

**Run**: `luxar demo run gsplats_lod_embryo_line`

**Requires**: Precomputed Tribolium splats (Git LFS); no network/GPU needed for the default path. `--recompute` re-fits from Zenodo (network + GPU).

**Demonstrates**: Per-object `coverage_fraction` LOD selection at scale, scene-graph transforms (`add_group(transform=...)`, `transforms.compose`/`rotate`/`translate`) for placement so splat arrays stay identical, automatic `array_ref` array deduplication in the encoder, and initial-camera setup via `ViewerConfig(camera=CameraConfig(...))`. Options: `--count=N`, `--levels=N`, `--factor=K`, `--method=NAME`, `--serve-only`.

---

#### demo_gsplats_recipes_tribolium.py - LOD `--recipe` gallery (flat / stream / levels / tiles / overview / adaptive)
Runs the unified `luxar gsplat lod --recipe` pipeline on the **one** precomputed Tribolium fit to build the six scale-ordered representation topologies and lays them out side by side for direct comparison: **flat** (one leaf) → **stream** (one leaf + a prefix-sum ladder) → **levels** (a `kind=lod` of *substitutive* levels — coarse↔fine replacement, one shown at a time) → **tiles** (a spatial BSP `kind=partition` where every part carries its own additive ladder) → **overview** (an *unbalanced-by-design* `kind=lod`: a cheap coarse substitutive cap for the far view, above a `tiles` fine branch for close-up — detail only where you look) → **adaptive** (a BSP `kind=partition` where every part is its own *substitutive* lod group — per-part coarse↔fine replacement, so each cell culls AND picks its own level by its own on-screen size). Structure is colour-coded so the topologies are legible: each BSP part gets a distinct colour, the overview coarse cap is red ("far") above cool-coloured fine parts ("near"), and the stream ladder runs blue→cyan coarse→fine. The reference demo for the `cal → fit → lod --recipe → convert → serve` workflow; each recipe build prints the equivalent CLI command.

**Run**: `luxar demo run gsplats_recipes_tribolium`

**Requires**: Precomputed Tribolium splats (Git LFS); no network/GPU needed for the default path. `--recompute` re-fits from Zenodo (network + GPU).

**Demonstrates**: The `lod --recipe` engine (`build_recipe`/`RecipeParams`) and the three novel topologies — `tiles` (per-part additive ladders), `overview` (coarse cap over a `tiles` fine branch), and `adaptive` (per-part substitutive lod groups); writing each recipe via the CLI's exact path (`GSplatData.save` for matrix recipes, `write_gsplats_tree` for composed node trees) and grafting them with `add_group(transform=...)` + `add_gsplats_from_file`. Options: `--max-elements=N`, `--factor=K`, `--serve-only`.

---

#### Classical Gaussian-splat interop demos (import from the photogrammetric ecosystem)

These six demos download **pre-captured** Gaussian-splat scenes from the classical/photogrammetric 3DGS ecosystem and import them into Luxar via `luxar gsplat import` — no fitting. Each prints a data-provenance/licence notice before downloading; Luxar redistributes none of the data. They render best with `blending_mode="normal"`: normal-mode gsplats are depth-sorted (with cross-mesh renderOrder) for correct alpha-over compositing.

##### demo_gsplats_interop_spz_scaniverse.py - Scaniverse SPZ captures (Niantic)
Downloads the two official Niantic `spz` sample scans (a horned lizard, a raccoon-family sculpture — phone captures) and imports them with a streaming LOD ladder.
**Run**: `luxar demo run gsplats_interop_spz_scaniverse`
**Requires**: Network (~18 + 24 MB). No GPU.
**Data**: [nianticlabs/spz](https://github.com/nianticlabs/spz) samples — **MIT**.

##### demo_gsplats_interop_mipnerf_garden.py - Mip-NeRF 360 garden / bicycle (`.splat`)
Downloads the iconic Mip-NeRF 360 *garden* (or `--scene bicycle`) scene as an antimatter15 `.splat` (~187/196 MB) and imports it with a **tiled** LOD (spatial BSP + per-tile streaming ladders) since it is ~5 M splats.
**Run**: `luxar demo run gsplats_interop_mipnerf_garden [-- --scene garden|bicycle]`
**Requires**: Network (~187 MB). No GPU.
**Data**: [cakewalk/splat-data](https://huggingface.co/cakewalk/splat-data); Mip-NeRF 360 (Barron et al. 2022) trained to 3DGS (Kerbl et al. 2023) — **research use** (Google), fetched at runtime, not redistributed.

##### demo_gsplats_interop_observatory.py - Astronomical observatories (SuperSplat compressed PLY)
Downloads a Gaussian-splat capture of the Vera C. Rubin Observatory (`--scene gemini-south` for Gemini South) as a SuperSplat *compressed* PLY and imports it with a streaming LOD.
**Run**: `luxar demo run gsplats_interop_observatory [-- --scene rubin|gemini-south]`
**Requires**: Network (~41/83 MB). No GPU.
**Data**: [khyron/Gaussian-Splatting](https://github.com/khyron/Gaussian-Splatting) — **CC BY 4.0**.

##### demo_gsplats_interop_inria_garden.py - Full-quality INRIA garden (HTTP-Range zip extraction)
Extracts just the ~1.45 GB `point_cloud.ply` member from INRIA's 14.7 GB `models.zip` over **HTTP Range requests** (Zip64 central-directory parsing — the member sits at byte offset ~5.6 GB), then imports it with a tiled LOD. The full-training-fidelity counterpart of the `.splat` demo.
**Run**: `luxar demo run gsplats_interop_inria_garden`
**Requires**: Network (~1.45 GB range-extracted from a 14.7 GB archive; server must honor byte ranges — INRIA's does). No GPU.
**Data**: [INRIA 3DGS pretrained models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/); INRIA Gaussian-Splatting license (**research / non-commercial**); Mip-NeRF 360 scenes (Google research use). Fetched at runtime, not redistributed.

##### demo_gsplats_interop_macro_clusterfly.py - Macro cluster fly (Dany Bittel)
Range-extracts just the ~68 MB `cluster fly L.ply` member (~300 k splats) from Dany Bittel's ~1 GB macro-photogrammetry release archive on GitHub — a real cluster fly (*Pollenia*) captured under a macro rig and 3DGS-trained — then imports it with a streaming LOD. A millimetre-scale "fly in digital amber": the smallest, most detailed subject of the interop set, and the one that exercises the range extractor against a **redirecting** host (GitHub release assets → signed CDN).
**Run**: `luxar demo run gsplats_interop_macro_clusterfly`
**Requires**: Network (~68 MB range-extracted from a ~1 GB archive; server must honor byte ranges — GitHub's asset CDN does). No GPU.
**Data**: [Dany Bittel macro splats](https://danybittel.ch/macro) — **CC BY 4.0** (attributed in-scene). Fetched at runtime, not redistributed.

##### demo_gsplats_interop_sog_matrixcity.py - MatrixCity aerial (PlayCanvas SOG, ~13.6M — the largest)
Fetches the aerial "small city" MatrixCity scene — **13,589,514 Gaussians** — from SuperSplat in PlayCanvas's compact **SOG** (Spatially Ordered Gaussians) format (`meta.json` + lossless WebP images, ~15–20× smaller than PLY), decodes it with Luxar's SOG reader, and builds an **overview** LOD: one coarse global level (instant whole-city first paint) over a BSP partition of stream-laddered fine tiles. The largest scene in the interop set (~45× the cluster fly) and the city-scale stress test for Luxar's LOD.
**Run**: `luxar demo run gsplats_interop_sog_matrixcity`
**Requires**: Network (~156 MB SOG bundle), several GB RAM for the import + LOD build (workstation recommended). No GPU required. `Pillow` (WebP) from the `demos` extra.
**Data**: [MatrixCity](https://city-super.github.io/matrixcity/) (Li et al. 2023), 3DGS via FriendlySplat on [SuperSplat](https://superspl.at/scene/ace6e5b0) — research/education; fetched at runtime, not redistributed.

**Demonstrates**: the classical-splat import path (all five dialects incl. **SOG** / WebP-compressed), the `download_zip_member` HTTP-Range/Zip64 extractor (including redirect-following HEAD for signed-CDN hosts), and applying `stream`/`tiles`/`overview` LOD recipes to imported scenes via `build_recipe`.

---

#### demo_gsplats_4d_zebrafish_timelapse.py - 4D Zebrafish Embryo Time-Lapse
4D (3D + time) confocal recording of a living zebrafish embryo during gastrulation, with per-timepoint Gaussian splatting and a time dimension slider.

**Run**: `luxar demo run gsplats_4d_zebrafish_timelapse`

**Requires**: Internet access (downloads ~2.1 GB LSM from Zenodo), GPU recommended.

**Demonstrates**: 4D Gaussian splatting (3D + time), per-timepoint independent fitting, `dim_order` + `fill` for time coordinate assignment, zebrafish gastrulation imaging, Zeiss LSM format.

---

#### demo_gsplats_4d_neuromast_2ch.py - 4D Two-Channel Zebrafish Neuromast Time-Lapse
100-timepoint iSIM recording of a developing zebrafish lateral-line **neuromast** (`she:GFP; cldnb:lyn-mScarlet`), shown as **two independently-toggleable Gaussian-splat layers**: membranes (mScarlet, `bop_blue` LUT) + nuclei (GFP, `bop_orange` LUT). Both channels are co-registered and share one 4D coordinate space, so they animate together — play the **Time** slider to scrub development, press **L** for the Layers panel to control each channel. The bundled gsplats are pre-fit (per-channel calibration K*=64k → `batch-fit` → Z×2.5 anisotropy correction → redundancy cull), so no fitting/GPU is needed to view.

**Run**: `luxar demo run gsplats_4d_neuromast_2ch`

**Requires**: The two pre-fit `.gsplats.zarr` (~220 MB) in a local store (`~/luxar_demo_data/gsplats_neuromast_2ch/`, or `$LUXAR_NEUROMAST_DATA_DIR`). ⚠️ **Not bundled/hosted yet** — this is the outstanding follow-up (upload to the demo data host and switch to `load_precomputed_gsplats`, like the other gsplat demos). No network/GPU needed once the store is populated.

**Demonstrates**: 4D + multi-channel gsplats, per-channel `layer=True` + named colormaps (`bop_blue`/`bop_orange`) for the Layers panel, `add_gsplats_from_file` grafting of pre-fit multi-LOD (`stream`, 8 LODs) nodes, Z-anisotropy correction baked via `transform --scale`, redundancy-based culling, near-zero volumetric absorption (κ=0.05) so the two superimposed channels barely occlude each other. Options: `--no-serve`, `--serve-only`.

---

#### demo_gsplats_4d_celegans_tracking.py - 4D C. elegans Nuclei Tracking with Lines
4D confocal time-series of a developing *C. elegans* embryo: Gaussian splats for volume rendering combined with polylines for tracked cell nuclei trajectories.

**Run**: `luxar demo run gsplats_4d_celegans_tracking`

**Requires**: Internet access (downloads ~26.1 GB from Zenodo), GPU recommended.

**Demonstrates**: Combined GSplats + Lines in 4D, cell lineage tracking as 4D polylines (X, Y, Z, Time), `extend_to_all` for persistent trajectory visibility, StarryNite tracking data, volume rendering + track overlay.

## Demo Pattern

Each demo follows this self-contained pattern:

```python
#!/usr/bin/env python3
"""Demo: Description

What this demonstrates...
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
from arbol import aprint, asection
from luxar import LuxarZarrCompiler, Dimension, Dimensions

def generate_my_data(output_path: Path, **params) -> None:
    """Generate the dataset - ALL CODE IN THIS FUNCTION.

    Args:
        output_path: Where to write zarr
        **params: Generation parameters
    """
    with asection("Generating Data"):
        # 1. Generate positions, colors, radii, etc.
        # ALL generation logic here - no external functions!
        positions = ...
        colors = ...

        # 2. Write to zarr
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points('name', positions, colors=colors, ...)

def main():
    """Entry point - generate and serve."""
    with tempfile.TemporaryDirectory(prefix="luxar_demo_") as tmpdir:
        output_path = Path(tmpdir) / "demo.luxar.zarr"

        # Generate
        generate_my_data(output_path)

        # Serve using the launch_viewer helper
        from luxar.demos import launch_viewer
        launch_viewer(output_path)

    # Auto-cleanup
    aprint("✓ Cleanup complete")

if __name__ == "__main__":
    main()
```

## Key Principles

### 1. Self-Contained
**All generation code must be in the demo file itself.**

✅ **GOOD**:
```python
def generate_my_data(output_path):
    # Generate positions here
    x = np.linspace(0, 10, 1000)
    positions = ...
    # Write to zarr here
    with LuxarZarrCompiler(output_path) as compiler:
        ...
```

❌ **BAD**:
```python
def generate_my_data(output_path):
    # Calls external function - not self-contained!
    from my_utils import create_positions
    positions = create_positions()  # ← External dependency!
```

### 2. Use Temporary Directory
Always use `tempfile.TemporaryDirectory()` to ensure cleanup:

```python
with tempfile.TemporaryDirectory(prefix="luxar_demo_myname_") as tmpdir:
    output = Path(tmpdir) / "data.luxar.zarr"
    generate_data(output)
    serve_and_view(output)
# Automatic cleanup when context exits
```

### 3. Use launch_viewer Helper
Don't reimplement server logic - use the `launch_viewer` helper:

```python
from luxar.demos import launch_viewer

# At the end of your demo, after generating data:
launch_viewer(output_path)
```

Benefits:
- Uses `sys.executable -m luxar` so it works regardless of how the demo was run
- Works with hatch, conda, or any Python environment where luxar is installed
- Handles all error cases (KeyboardInterrupt, missing viewer, etc.)
- Reuses tested server code
- Handles CORS, directory listing, etc.
- Stops cleanly on Ctrl+C

### 4. Running Demos

Demos should be run through hatch to ensure luxar is available:

```bash
# From project root
hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py

# Or activate hatch shell first
hatch shell
python packages/luxar/src/luxar/demos/demo_lorenz.py
```

### 5. Use arbol for Output
Structure console output with arbol for clarity:

```python
from arbol import aprint, asection

with asection("Generating Data"):
    aprint("Creating grid...")
    # ... generation code ...
    aprint(f"✓ Created {n_points:,} points")

with asection("Writing to Zarr"):
    # ... writing code ...
    aprint(f"✓ Written to {path}")
```

### 6. Use the shared helpers (don't hand-roll caching)

`luxar.demos` re-exports a small set of helpers — prefer them over hand-rolling
`Path.home() / ".cache" / ...` logic, `sys.argv` scanning, or HSV→RGB:

```python
from luxar.demos import (
    launch_viewer,       # serve + open viewer (serve_args=[...] to pass e.g. --profile)
    cached_download,     # download once into ~/.cache/luxar/<name>/, skip-if-present
    cache_computed,      # cache an expensive result (UMAP, field) — versioned, param-keyed
    require_local_data,  # gate LFS-tracked local data (clear "git lfs pull" message)
    require_module,      # gate an OPTIONAL dependency at its point of use (see #7)
    parse_demo_flags,    # --recompute / --no-serve / --serve-only
    parse_int_arg,       # --points=N / --sample N integer flags
    hsv_to_rgb,          # vectorized rainbow / hue-ramp colouring
    detect_device, warn_if_no_cuda_gpu,          # GPU/MPS/CPU
    load_precomputed_gsplats, load_precomputed_bundle,  # LFS-shipped gsplat data
)

# Download once, reused on every later run:
csv = cached_download("https://…/data.csv", "mydemo", "data.csv")

# Cache an expensive UMAP — the KEY must include every param that changes the
# output (sample size, feature set, …) so a stale cache is never reused:
positions = cache_computed(
    "mydemo", f"umap3d_n{n}_f{len(FEATURES)}", lambda: run_umap(features), version=1
)
```

`cache_computed` writes atomically and quarantines a corrupt cache to `.corrupt`
instead of crashing. A quarantined file is never reused, so
`luxar.utils.download.warn_if_quarantined()` (called from `robust_download()`,
the shared download chokepoint) reports its path and size before a re-fetch
starts — a demo that pulls a multi-gigabyte artifact should not silently restart
the download. Use `find_quarantined_files()` / `format_quarantine_notice()` when
a demo needs the same information inside its own error message.

Sibling demos are importable normally
(`from luxar.demos.demo_x import helper`) — no `importlib` file-path tricks.

### 7. Gate optional dependencies at the point of use, never at the entry point

Demos may need heavyweight extras (`umap-learn`, `sentence-transformers`,
`torch`, `esm`, `anndata`, …) that the core package deliberately does not
require. Demand them with `require_module`, **immediately before the work that
needs them**:

```python
from luxar.demos import require_module

def _compute_umap3d(features):
    # Gated here, not in main(): a warm cache never needs UMAP.
    UMAP = require_module("umap").UMAP
    return UMAP(n_components=3).fit_transform(features)
```

Do **not** write a preflight in `main()`:

```python
# ❌ WRONG — refuses to run on a machine that has every artifact it needs
def main():
    try:
        import umap  # noqa: F401
    except ImportError:
        aprint("Missing dependency: umap-learn")
        sys.exit(1)
```

Because these demos cache their expensive results, a warm cache never touches
the dependency that produced it — so a preflight rejects a perfectly usable
machine. It is not merely redundant: the ESM-3 demo printed advice ("a complete
cached embeddings file skips the model entirely") that its own preflight made
impossible to follow. Deferring costs nothing on a cold run either, because the
raw download is cached too: re-running after the install resumes instead of
refetching. `tests/test_no_entrypoint_dependency_preflight.py` fails the build if
a preflight reappears.

Version constraints live in one table, `INSTALL_SPECS` in
[`_dependencies.py`](_dependencies.py) — never advertise a bare `pip install
<pkg>`. See [Optional dependencies](#optional-dependencies) above for the table's
two consumers, what the tests enforce, and the `anndata` cautionary case; adding
a new optional dependency means pinning it in `pyproject.toml` **and** tabling it
there, or the build fails.

Soft checks that *degrade a feature* rather than refuse to run are fine and are
not flagged — e.g. disabling image thumbnails when `Pillow` is absent.

## Creating New Demos

1. **Copy a template** (demo_lorenz.py or demo_cubic_array.py)
2. **Rename** to demo_yourname.py
3. **Update docstring** with what it demonstrates
4. **Implement generation** in the generate_* function (keep everything in that function!)
5. **Test** by running: `hatch run python demo_yourname.py`
6. **Ctrl+C** to stop and verify cleanup works

## Tips

### For Large Datasets
Use progressive writing to avoid memory issues:

```python
# DON'T load all data at once if very large
# DO generate and write in chunks

dims = Dimensions.default_3d()
with LuxarZarrCompiler(output) as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Write in batches
    for i in range(num_batches):
        batch = generate_batch(i)
        scene.add_points(f'batch_{i}', batch)
```

### For nD Demos
Specify dimensions with proper display flags:

```python
dims = Dimensions([
    Dimension('x', unit='um', display=True),
    Dimension('y', unit='um', display=True),
    Dimension('z', unit='um', display=True),
    Dimension('time', unit='s', display=False, discrete=True, range=(0, 99))
])
```

### For Complex Math
Add comments explaining the mathematics:

```python
# Generate sphere using spherical coordinates
# φ ∈ [0, 2π], θ ∈ [0, π]
# x = r sin(θ) cos(φ)
# y = r sin(θ) sin(φ)
# z = r cos(θ)
phi = np.random.uniform(0, 2*np.pi, n)
theta = np.arccos(np.random.uniform(-1, 1, n))
...
```

## Running All Demos

```bash
# From project root (use hatch to ensure luxar is available).
# Each demo launches a viewer — press Ctrl+C to stop and move to the next.

# --- Scientific Visualization ---
hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py
hatch run python packages/luxar/src/luxar/demos/demo_rainbow_sphere.py
hatch run python packages/luxar/src/luxar/demos/demo_volumetric_cloud.py
hatch run python packages/luxar/src/luxar/demos/demo_cubic_array.py
hatch run python packages/luxar/src/luxar/demos/demo_mandelbulb.py
hatch run python packages/luxar/src/luxar/demos/demo_spiral_galaxy.py
hatch run python packages/luxar/src/luxar/demos/demo_quantum_orbitals.py
hatch run python packages/luxar/src/luxar/demos/demo_atp_synthase.py
hatch run python packages/luxar/src/luxar/demos/demo_nuclear_pore_complex.py
hatch run python packages/luxar/src/luxar/demos/demo_quasicrystal_3d.py
hatch run python packages/luxar/src/luxar/demos/demo_turing_patterns.py
hatch run python packages/luxar/src/luxar/demos/demo_bioluminescent_ocean.py
hatch run python packages/luxar/src/luxar/demos/demo_particle_collision.py
hatch run python packages/luxar/src/luxar/demos/demo_particle_collision_animated.py

# --- nD and Multi-Dimensional ---
hatch run python packages/luxar/src/luxar/demos/demo_4d_fractals.py
hatch run python packages/luxar/src/luxar/demos/demo_5d_spiral_galaxy.py
hatch run python packages/luxar/src/luxar/demos/demo_nd_transforms.py

# --- Feature Showcases ---
hatch run python packages/luxar/src/luxar/demos/demo_sharpness_showcase.py
hatch run python packages/luxar/src/luxar/demos/demo_lsystem_forest.py
hatch run python packages/luxar/src/luxar/demos/demo_hilbert_curve_3d.py
hatch run python packages/luxar/src/luxar/demos/demo_network_performance.py

# --- Embedding / UMAP ---
hatch run python packages/luxar/src/luxar/demos/demo_arxiv_paper_embeddings.py
hatch run python packages/luxar/src/luxar/demos/demo_arxiv_embeddings_kaggle.py
hatch run python packages/luxar/src/luxar/demos/demo_protein_embeddings_cafa5.py
hatch run python packages/luxar/src/luxar/demos/demo_esm3_protein_landscape.py
hatch run python packages/luxar/src/luxar/demos/demo_cytoself_protein_landscape.py
hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_multiome.py
hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_velocity_streamlines.py
hatch run python packages/luxar/src/luxar/demos/demo_human_multiome_peak_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_mouse_multiome_peak_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_multiome_peak_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap_sequence.py
hatch run python packages/luxar/src/luxar/demos/demo_tabula_sapiens.py
hatch run python packages/luxar/src/luxar/demos/demo_spotify_tracks.py

# --- Data-Driven (External Datasets) ---
hatch run python packages/luxar/src/luxar/demos/demo_gaia_milky_way_3m.py
hatch run python packages/luxar/src/luxar/demos/demo_desi_galaxies.py
hatch run python packages/luxar/src/luxar/demos/demo_earthquakes_3d.py
hatch run python packages/luxar/src/luxar/demos/demo_storm_3d_microtubules.py
hatch run python packages/luxar/src/luxar/demos/demo_huri_interactome.py
hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py
hatch run python packages/luxar/src/luxar/demos/demo_caida_as_topology.py
hatch run python packages/luxar/src/luxar/demos/demo_asteroids_solar_system.py
hatch run python packages/luxar/src/luxar/demos/demo_dipc_3d_genome.py
hatch run python packages/luxar/src/luxar/demos/demo_dmri_tractography.py

# --- GSplats: 2D ---
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_2d_codex_pancreas.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_2d_cmu1_pathology.py

# --- GSplats: 3D ---
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_milky_way_dust.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_visible_human_head.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_cryoem_virus.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_ct_totalsegmentator.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_organoid_dapi_nuclei.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_organoid_multichannel.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_cells3d_multichannel.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_kidney_multichannel_toggles.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_kidney_multichannel_layers.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_acto3d_heart.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_opencell_map4.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_tribolium_embryo.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_lod_tribolium.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_lod_embryo_line.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_recipes_tribolium.py

# --- GSplats: 4D ---
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_zebrafish_timelapse.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_neuromast_2ch.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py
```

## Troubleshooting

**"luxar command not found"** or **"No module named luxar"**:
```bash
# Always run demos through hatch to ensure luxar is available:
hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py

# Or activate the hatch shell first:
hatch shell
python packages/luxar/src/luxar/demos/demo_lorenz.py
```

**"Viewer not built"**:
```bash
cd packages/luxar-viewer
pnpm install
pnpm build
```

**Port already in use**:
The CLI automatically finds available ports, but if issues persist, try specifying a different port:
```bash
# Won't work with subprocess.run in demos currently
# For manual testing: luxar serve data.luxar.zarr --viewer --port 8001 --viewer-port 5174
```

## Philosophy

These demos are **teaching tools**. They should be:
- Simple enough to understand in 5 minutes
- Complete enough to show real capability
- Clean enough to use as templates
- Fun enough to inspire creativity!

Keep them self-contained so anyone can read ONE file and understand the complete workflow.
