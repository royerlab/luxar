# Luxar Demos

This folder contains self-contained demo scripts that showcase Luxar's capabilities. Each demo is a complete, runnable Python script that generates data, launches the viewer, and cleans up automatically.

## Purpose

These demos are:
- **Didactic**: Easy to understand and learn from
- **Self-contained**: All generation code in one file
- **Complete**: Generate → Serve → View → Cleanup workflow
- **Copy-pasteable**: Can be used as templates for your own visualizations

## Available Demos

### Scientific Visualization Demos

#### demo_lorenz.py - Lorenz Attractor
Beautiful chaotic attractor with rainbow color gradient.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py [--points=100000]`

**Demonstrates**: Differential equation integration, vectorized HSV-to-RGB color conversion, time-based color gradients, progressive writing, chaotic systems visualization.

---

#### demo_rainbow_sphere.py - Fibonacci Spiral Sphere
Dense sphere (200k points) with perfect distribution and rainbow colors.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_rainbow_sphere.py [--points=100000]`

**Demonstrates**: Fibonacci (golden angle) spiral for optimal sphere coverage, smooth rainbow gradient using phase-shifted sine waves, automatic point spacing calculation, high density visualization.

---

#### demo_volumetric_cloud.py - Fractal Cloud Structure
Realistic cloud using multi-octave fractal noise and varying point sizes.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_volumetric_cloud.py [--points=1000000]`

**Demonstrates**: Self-contained Perlin-like fractal noise, multi-octave detail at multiple scales, volumetric density filtering, varying point sizes based on local density, soft cloud-like appearance (low sharpness 0.5-2.0).

---

#### demo_cubic_array.py - 3D Cubic Grid with Star Background
Dense 100-cubed grid (1M points) with 500k background stars.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_cubic_array.py`

**Demonstrates**: Regular grids via meshgrid, multi-layer scenes (foreground + background), very high point density (1.5M total), different blending modes (additive vs normal), semi-transparent background layers.

---

#### demo_mandelbulb.py - 3D Mandelbulb Fractal
Stunning volumetric representation of the famous Mandelbulb 3D fractal.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_mandelbulb.py [--resolution=128] [--power=8]`

**Demonstrates**: 3D fractal mathematics (extension of Mandelbrot set), distance estimation for surface detection, iteration-based coloring, adaptive point sizing, spherical coordinate transformation, escape-time algorithm in 3D.

---

#### demo_spiral_galaxy.py - Realistic Multi-Armed Spiral Galaxy
Beautiful astronomical simulation of a barred spiral galaxy.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_spiral_galaxy.py [--stars=500000] [--arms=4]`

**Demonstrates**: Logarithmic spiral arm generation, realistic stellar population distributions, color variation (blue young stars in arms, red/yellow old stars in bulge), central galactic bulge and stellar halo modeling.

---

#### demo_quantum_orbitals.py - Quantum Atomic Orbitals
Hydrogen atom electron probability density visualization showing s, p, d, and f orbitals.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_quantum_orbitals.py [--grid=N]`

**Demonstrates**: Hydrogen wavefunction computation (radial functions and spherical harmonics), categorical navigation between quantum states (n, l, m), probability density coloring, 3D shapes (spheres, dumbbells, cloverleafs).

---

#### demo_atp_synthase.py - ATP Synthase Molecular Turbine
Visualizes the complete ATP Synthase rotary motor structure with F1 catalytic head and F0 membrane rotor, all subunits color-coded.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_atp_synthase.py`

**Demonstrates**: Molecular machine visualization, multi-subunit protein complex, color-coded structural components (alpha, beta, gamma, c-ring), biological energy production machinery (~600 kDa enzyme).

---

#### demo_nuclear_pore_complex.py - Nuclear Pore Complex
Downloads real Nup107-160 subcomplex structure from PDB and applies perfect 8-fold rotational symmetry to visualize the nuclear gateway.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_nuclear_pore_complex.py`

**Requires**: Internet access (downloads PDB structure).

**Demonstrates**: PDB structure download and parsing, C-alpha backbone trace, 8-fold rotational symmetry application, van der Waals radii for atomic sizes, color-coded spokes.

---

#### demo_quasicrystal_3d.py - 3D Aperiodic Quasicrystal
3D quasicrystal with icosahedral symmetry using the cut-and-project method from 6D to 3D.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_quasicrystal_3d.py`

**Demonstrates**: Cut-and-project method (6D lattice to 3D), icosahedral symmetry (5-fold), golden ratio projection matrices, aperiodic tiling (never repeats but ordered), color-coded by perpendicular space coordinates.

---

#### demo_turing_patterns.py - 2D Turing Reaction-Diffusion Patterns
Gray-Scott reaction-diffusion system creating organic patterns (spots, stripes, spirals, labyrinths) with temporal evolution.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_turing_patterns.py [--size=N] [--steps=N]`

**Demonstrates**: Gray-Scott model simulation, multiple pattern types via different (F, k) parameters, temporal evolution of pattern formation, categorical navigation between pattern types, emergent complexity from simple rules.

---

#### demo_bioluminescent_ocean.py - Bioluminescent Ocean
Ethereal underwater visualization with animated jellyfish, flowing tentacles, glowing plankton, and bioluminescent deep-sea atmosphere.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_bioluminescent_ocean.py`

**Demonstrates**: Jellyfish bell and tentacle geometry, bioluminescent color schemes (GFP-inspired), marine organism animation, atmospheric underwater effects.

---

#### demo_flywire_connectome.py - FlyWire Adult Drosophila Connectome
First complete wiring diagram of an adult animal brain: ~139k neurons as Points at their soma positions (in µm) and top-N neuron-to-neuron connections as tapered Lines inside the real brain envelope.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_flywire_connectome.py [--min-synapses=20] [--max-edges=300000]`

**Requires**: Internet access on first run (auto-downloads ~850 MB of connectivity data from Zenodo to `~/.cache/luxar/flywire/`; subsequent runs are offline).

**Demonstrates**: Biological **network visualization** (Points + Lines in the same scene), real EM-reconstructed 3D soma coordinates, dominant-presynaptic-neurotransmitter inference via argmax of per-NT probability columns, **one toggleable Lines layer per neurotransmitter** (acetylcholine / GABA / glutamate / dopamine / serotonin / octopamine) so users can isolate excitatory, inhibitory, or neuromodulatory circuits from the viewer's Layers panel, width-tapered directed connections, super-class legend with 10 neuron categories, large spatially-embedded graph (~139k nodes, up to 300k edges). Data: FlyWire 783 release ([Schlegel et al. 2024 Nature](https://github.com/flyconnectome/flywire_annotations), [Dorkenwald et al. Zenodo 10676866](https://zenodo.org/records/10676866)).

---

#### demo_particle_collision.py - Particle Collision Detector
Realistic visualization of particle physics collisions inspired by CERN's ATLAS and CMS detectors, showing helical particle tracks, jets, and energy deposits.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_particle_collision.py`

**Demonstrates**: Helical trajectories from Lorentz force (F = qv x B), transverse momentum and track curvature, detector geometry (barrel + endcap), particle jets and energy deposits, magnetic field effects on charged particles.

---

#### demo_particle_collision_animated.py - Animated Particle Collision Detector
Time-animated version of the particle collision demo. Watch particle tracks grow outward from the collision vertex as time advances.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_particle_collision_animated.py`

**Demonstrates**: Same physics as `demo_particle_collision.py` plus time dimension animation, event unfolding in real-time, 4D (XYZ + time) navigation.

---

### nD and Multi-Dimensional Demos

#### demo_4d_fractals.py - 4D Geometric Fractal Explorer
Interactive exploration of 6 different 4D geometric fractals with categorical dimension.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_4d_fractals.py [--grid=64]`

**Demonstrates**: 4D spatial navigation (XYZ + W dimension), categorical dimension (select between 6 fractal types), large dataset (~100M+ points, ~1M visible), XOR Fractal, Menger Sponge 4D, Sierpinski 4D, Cantor Dust 4D, Checkerboard and Diamond patterns.

---

#### demo_5d_spiral_galaxy.py - 5D Spiral Galaxy with Time Evolution
Large-scale 5D data with millions of points: multiple spiral arms evolving over time with channel-based coloring.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_5d_spiral_galaxy.py [--points=N]`

**Demonstrates**: 5D data (X, Y, Z, Time, Channel), continuous and discrete dimension navigation, time-based animation of spiral arm rotation, channel variation representing different wavelengths, logarithmic spiral arm mathematics.

---

#### demo_nd_transforms.py - Multi-Instrument Observatory with nD Transforms
Three instruments (optical telescope, radio telescope, X-ray satellite) observing the same galaxy cluster, aligned using per-dimension affine transforms.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_nd_transforms.py`

**Demonstrates**: `nd_transform` with affine scale + offset on Time dimension, categorical permutation on Channel dimension, multi-group scenes with per-group nD alignment, coherent multi-instrument view from a single time slider.

---

### Feature Showcase Demos

#### demo_sharpness_showcase.py - Point Sharpness Showcase
Comprehensive showcase of the point sharpness feature: gradient from soft (0.5) to sharp (15.0), fixed sharpness comparison rows, mixed sharpness cloud, and sinusoidal wave pattern.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_sharpness_showcase.py [--points N]`

**Demonstrates**: Sharpness parameter control (intensity = (1 - r^2)^sharpness), soft glowing points (0.5-1.0) vs sharp disc-like points (8.0-15.0), color-coded sharpness values, multiple visualization patterns.

---

#### demo_lsystem_forest.py - L-System Tree Forest (Lines Demo)
Beautiful procedural forest using L-system grammars to showcase the **Lines** node type.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_lsystem_forest.py [--iterations=6] [--trees=16]`

**Demonstrates**: **Lines node type** with thousands of line segments, L-system grammar expansion and interpretation, width tapering (thick trunk to thin twigs), color gradients (bark to foliage), 3D branching, multiple tree varieties (elegant, fractal, willow, bush, cherry), seasonal color schemes.

---

#### demo_hilbert_curve_3d.py - 3D Hilbert Space-Filling Curve (Lines Demo)
The 3D Hilbert curve — a continuous, self-similar polyline that visits every cell of a 2^n × 2^n × 2^n grid exactly once with consecutive cells always sharing a face. Vectorized Skilling algorithm produces orders 1 through max_order (default 6 = 262k vertices), each rendered as a single thin polyline with HSV hue swept along the traversal index. A slider on the `order` dim steps through the recursion so you can watch each level subdivide and rotate.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_hilbert_curve_3d.py [--max-order=6]`

**Demonstrates**: Single ultra-long `polyline` Lines node (262k+ vertices, 262k segments), thin constant width with color gradient along traversal, runtime-verified Hamiltonian path on the integer lattice, slider-driven recursion exploration (each order on its own slot of a non-displayed `order` dim), Skilling's vectorized 3D Hilbert algorithm.

---

#### demo_network_performance.py - Network Performance Testing
Large multi-cluster dataset (1M points) for testing viewer performance under network constraints.

**Run**:
```bash
hatch run python packages/luxar/src/luxar/demos/demo_network_performance.py
hatch run python packages/luxar/src/luxar/demos/demo_network_performance.py --profile 3g
hatch run python packages/luxar/src/luxar/demos/demo_network_performance.py --points=2000000 --profile satellite
```

**Demonstrates**: Network simulation (bandwidth throttling, latency, jitter, packet loss), progressive loading behavior with limited bandwidth, cache effectiveness under bandwidth constraints, multi-cluster particle systems (1M+ points).

---

### Embedding and UMAP Demos

#### demo_arxiv_paper_embeddings.py - ArXiv Paper Embeddings (Semantic Scholar)
Visualizes scientific papers in 3D embedding space. Papers cluster by topic, colored by research field, sized by citation count.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_arxiv_paper_embeddings.py`

**Requires**: Internet access, `sentence-transformers` and `umap-learn` packages. Embeds abstracts with Sentence-BERT (all-MiniLM-L6-v2) and reduces to 3D with UMAP.

**Demonstrates**: Semantic embedding of text, UMAP dimensionality reduction (768D to 3D), citation-based sizing, research field clustering, Semantic Scholar API usage.

---

#### demo_arxiv_embeddings_kaggle.py - ArXiv Paper Embeddings (Kaggle / OpenAI)
Visualizes arXiv papers using pre-computed OpenAI embeddings from the Kaggle "openai-arxiv-embeddings" dataset (2M+ papers, 3072D).

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_arxiv_embeddings_kaggle.py`

**Requires**: Internet access, Kaggle account (for first download, ~30 GB cached to `~/.cache/mlcroissant/`), `mlcroissant` and `umap-learn` packages. First run takes 8-15 minutes; subsequent runs use cache.

**Demonstrates**: Large-scale embedding visualization (2M+ papers), pre-computed OpenAI text-embedding-3-large, UMAP reduction, Kaggle dataset integration via mlcroissant.

---

#### demo_protein_embeddings_cafa5.py - Protein Function Landscape (ProtT5 Embeddings)
Visualizes 142k proteins from the CAFA5 challenge in 3D embedding space, showing how proteins with similar functions cluster together.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_protein_embeddings_cafa5.py`

**Requires**: Internet access, `umap-learn` package. Uses ProtT5 1024D embeddings reduced to 3D with UMAP.

**Demonstrates**: Protein language model embeddings (ProtT5), Gene Ontology (GO) functional annotations, UMAP dimensionality reduction, protein function clustering, CAFA5 challenge dataset.

---

#### demo_zebrahub_integrated_cells.py - Zebrahub Integrated Cells 3D UMAP
Visualizes 95k integrated single cells from zebrafish with categorical attribute navigation between Cell Type and Timepoint views.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_integrated_cells.py`

**Requires**: Internet access (downloads from CZ Biohub public store), `fsspec` and `zarr` packages.

**Demonstrates**: Remote zarr data loading, 3D UMAP embedding of single-cell data, categorical dimension navigation (cell type vs timepoint), real scientific dataset from Zebrahub.

---

#### demo_human_multiome_peak_umap.py - Human Multiome Peak 3D UMAP
3D UMAP embedding of ~1M single-cell ATAC-seq peaks, color-coded by cell type, lineage, timepoint, and other attributes.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_human_multiome_peak_umap.py`

**Requires**: Local parquet data file, `pandas` package.

**Demonstrates**: Large-scale single-cell visualization (~1M points), multiple categorical attributes (cell type, lineage, timepoint, peak type, chromosome), ATAC-seq chromatin accessibility data.

---

#### demo_mouse_multiome_peak_umap.py - Mouse Multiome Peak 3D UMAP
3D UMAP embedding of ~192k single-cell ATAC-seq peaks from mouse embryonic development (E7.5-E8.75), color-coded by cell type, lineage, and timepoint.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_mouse_multiome_peak_umap.py`

**Requires**: Local parquet data file, `pandas` package.

**Demonstrates**: Single-cell ATAC-seq visualization, embryonic developmental timepoints, multiple categorical attribute navigation, lineage-based coloring.

---

#### demo_zebrahub_multiome_peak_umap.py - Zebrahub Multiome Peak 3D UMAP
3D UMAP of 640k single-cell chromatin accessibility peaks from the Zebrahub project, with 30 cell types across 6 developmental stages.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_multiome_peak_umap.py`

**Requires**: Internet access (downloads from CZ Biohub public zarr store).

**Demonstrates**: Remote zarr data loading, 640k points with 30 cell types, 7 categorical attributes (cell type, chromosome, lineage, peak type, etc.), 6 developmental timepoints.

---

#### demo_chromatrace_choir_umap.py - Chromatrace CHOIR 3D UMAP (Cell-Type Atlas)
3D UMAP of ~60k single cells annotated with CHOIR bio-terms (88 fine-grained cell types) and bio-groups (8 broad lineages: Neuroectoderm, Neural Crest, Craniofacial Mesenchyme, Cardiac/Vascular, Mesoderm, Endoderm, Epidermis, Other). Two runtime-switchable color views with curated CHOIR palette and grouped HTML legend.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap.py [--data /path/to/zip-or-folder]`

**Requires**: User-provided `choir_umap_3d_viewer.zip` bundle (parquet + colormap JSON) — auto-detected in `~/Downloads/` or `~/.cache/luxar/chromatrace/`.

**Demonstrates**: Categorical attribute switching with two color views, curated upstream palette mapping, grouped two-column HTML legend, custom hover overlay (suppresses default top-right tooltip), large categorical dimension (88 unique cell types).

---

#### demo_chromatrace_choir_umap_sequence.py - Chromatrace CHOIR UMAP — Cell-Type Walkthrough
Variant of the Chromatrace demo with an 89-step slider stepping through each fine-grained CHOIR bio-term one at a time. Within each bio-group, cell types are reordered (nearest-neighbor + 2-opt TSP on UMAP centroids) so consecutive slots are spatially adjacent. Backdrop layer keeps the full UMAP outline visible across all slots; highlight layer shows only the active type in its CHOIR palette color.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap_sequence.py [--no-tsp] [--data /path/to/zip-or-folder]`

**Requires**: Same `choir_umap_3d_viewer.zip` bundle as `demo_chromatrace_choir_umap.py`.

**Demonstrates**: `extend_to_all` for shared backdrop across slider slots (60k points instead of 89×60k), per-slot `visible_range` overlays, two-layer toggleable scene (Backdrop + Highlight), explicit `CameraConfig` to bypass auto-fit on a high-cardinality non-spatial dim, TSP-optimized intra-group ordering for smooth cross-slot transitions.

---

### Data-Driven Demos (External Datasets)

#### demo_gaia_milky_way_3m.py - Milky Way Stars (Gaia DR3, 3M Stars)
Real Milky Way stars from Gaia DR3: top 3M brightest stars with real photometric colors (BP-RP index), galactocentric coordinates, and reference markers (Sun, Betelgeuse, Rigel).

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gaia_milky_way_3m.py`

**Requires**: Internet access (Gaia DR3 TAP query or cached data).

**Demonstrates**: Real astronomical data (Gaia space telescope), 3M star dataset, BP-RP photometric color-to-RGB conversion, galactocentric coordinate system, magnitude-dependent point radii.

---

#### demo_gaia_milky_way_8m.py - Milky Way Stars (8M Stars)
Pre-computed 8.1 million star dataset from a CSV source, visualized with magnitude-based coloring and sizing.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gaia_milky_way_8m.py`

**Requires**: Pre-computed `milky_way_gaia_8m.zarr.zip` data file (143 MB).

**Demonstrates**: Very large point cloud visualization (8M+ stars), magnitude-to-color conversion (blue bright to red faint), percentile-based normalization for outlier handling, pre-computed dataset loading.

---

#### demo_earthquakes_3d.py - Global Earthquake Visualization
Real-time earthquake data from USGS plotted on a 3D Earth sphere with vertical spikes showing magnitude and color-coded by time.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_earthquakes_3d.py`

**Requires**: Internet access (USGS earthquake API).

**Demonstrates**: Real-time data from USGS API, spherical Earth projection (lat/lon to XYZ), magnitude-based spike height (logarithmic Richter scale), time-based coloring, plate boundary visualization.

---

#### demo_cosmicflows_laniakea.py - Cosmicflows-4 Laniakea Flow Field
Recreates the Cosmicflows-4 / Laniakea visualization by Simone Conradi and Manlio De Domenico: 55,486 local-universe galaxies plus colored streamlines tracing matter flow through basins of attraction.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_cosmicflows_laniakea.py [--preset preview|full]`

**Requires**: Internet access on first run (downloads ~26 MB from the public `manlius/laniakea` GitHub data cache; reused from `~/.cache/luxar/laniakea/`). The full preset reproduces the reported 29,555 valid streamlines and writes a large line dataset; `--preset preview` is faster for iteration.

**Demonstrates**: Real astronomical catalogs, supergalactic coordinates, basin-of-attraction coloring, vectorized RK4 streamline integration, large indexed Lines layers, HDR additive rendering, layer toggles per basin. Data: Cosmicflows-4 / EDD and the open [`manlius/laniakea`](https://github.com/manlius/laniakea) pipeline.

---

#### demo_storm_3d_microtubules.py - 3D STORM Super-Resolution Microscopy
Microtubule cytoskeleton at nanometer resolution using real STORM super-resolution microscopy localizations as Gaussian splats.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_storm_3d_microtubules.py`

**Requires**: Internet access (downloads STORM localization data).

**Demonstrates**: STORM/PALM super-resolution data (~20 nm resolution), localization uncertainty as Gaussian splat size, 3D astigmatism-based z encoding, photon count-based coloring, microtubule cytoskeleton structure.

---

#### demo_huri_interactome.py - HuRI Human Reference Interactome (Network)
~8k human proteins as Points and ~50k direct Y2H protein-protein interactions as Lines, laid out in 3D by graph structure alone (spectral embedding + UMAP). Nodes colored by Louvain community or HGNC chromosome; intra-community edges colored by community, cross-community "bridge" edges in dim gray. Rich edge hovers show the pair, intra/cross label, and — when available — the shared CORUM complex.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_huri_interactome.py`

**Requires**: Internet access (downloads ~35 MB one-time: HuRI + HGNC + optional CORUM), `networkx>=3.0`, `umap-learn`, `scipy`, `pandas`, `requests`.

**Demonstrates**: First-class **network** visualization in Luxar (Points + Lines together), graph-derived layout (normalized Laplacian eigenvectors → UMAP), Louvain community detection, categorical attribute switching between two color views, edge-level hover labels with biological context (CORUM shared complexes), symmetric edge rendering (undirected, equal both endpoints).

---

#### demo_ppi_flow_field.py - HuRI PPI Flow Field (Signed Network Streamlines)
Turns the HuRI protein-protein interaction graph into a continuous 3D flow landscape. PageRank orients each interaction from lower-centrality to higher-centrality protein; UMAP embeds a signed sparse adjacency / flow-profile matrix so upstream and downstream roles separate; a cubic vector field blends nearby oriented edges with a regularized `1/d^3` weight; Gaussian-smoothed streamlines are seeded from proteins and advected forward through the volume.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py [--preset preview|full]`

**Requires**: Internet access on first run (reuses the HuRI/HGNC cache in `~/.cache/luxar/huri/`), `networkx>=3.0`, `umap-learn`, `scipy`, `pandas`, `requests`. The default `full` preset builds the requested 256³ vector field and caches a larger `.npz` file; use `--preset preview` for a faster 128³ authoring run.

**Demonstrates**: Directed biological **network flow** visualization, PageRank-oriented interactions, signed sparse adjacency UMAP, KD-tree-accelerated point-to-edge vector-field construction, regularized inverse-cubic edge weighting, Gaussian vector-field smoothing, vectorized RK4 streamline advection, optional tapered directed PPI edge layer, HDR additive Lines rendering.

---

#### demo_caida_as_topology.py - CAIDA AS Topology (The Internet as a Graph)
~80k Autonomous Systems as Points and ~350k BGP relationships as Lines, laid out in 3D from graph structure alone. Auto-fetches the latest CAIDA serial-2 snapshot. Edges are styled by RELATIONSHIP TYPE — warm amber tapered lines for provider-customer (thick at provider, thin at customer, encoding direction), cool cyan uniform lines for peers. Nodes colored by Louvain community or country (CAIDA as2org). Tier-1 ASes (no upstream providers) get a size bump so the backbone pops. Edge hovers narrate the relationship ("Tier-1 transit" / "Tier-1 peering" / "Provider → Customer" / etc.) with both endpoints' org names and countries.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_caida_as_topology.py`

**Requires**: Internet access (auto-downloads ~60 MB from CAIDA on first run), `networkx>=3.0`, `umap-learn`, `scipy`, `pandas`, `requests`.

**Demonstrates**: Large-scale **directed network** visualization (~80k nodes), aesthetic-tuned graph layout pipeline (50-dim spectral embedding → UMAP with n_neighbors=30, metric='cosine', spread=2.0 → PCA realignment for consistent orientation), edges styled by relationship kind (tapered-directional vs symmetric), tier-1 backbone detection from graph topology, per-edge narrative hover labels, auto-discovery of latest upstream dataset snapshots.

---

### GSplats Demos (Gaussian Splatting)

#### demo_gsplats_2d_codex_pancreas.py - 2D CODEX Pancreas (12-Channel Multiplexed Fluorescence)
12-channel multiplexed immunofluorescence image of human pancreas tissue as 2D Gaussian splats with per-channel colors, using tiled fitting.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_2d_codex_pancreas.py`

**Requires**: Internet access (downloads from Zenodo), GPU recommended. 476 megapixels per channel (25,816 x 18,440 px).

**Demonstrates**: Tiled 2D Gaussian splatting with Hann cosine apodization, 12-channel CODEX multiplexed fluorescence, per-channel biologically meaningful colors, tile merging for large images, Zenodo data download.

---

#### demo_gsplats_2d_cmu1_pathology.py - 2D Whole-Slide Pathology Image (CMU-1)
Gigapixel H&E-stained whole-slide pathology image as 2D Gaussian splats with per-channel (R/G/B) colors, using tiled fitting.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_2d_cmu1_pathology.py`

**Requires**: Internet access (downloads from OpenSlide test data, ~169 MB), GPU recommended. 1.5 gigapixels (46,000 x 32,914 px).

**Demonstrates**: Tiled 2D Gaussian splatting on gigapixel data, brightfield H&E histology (brightness inversion), R/G/B channel splitting, Aperio SVS whole-slide image format.

---

#### demo_gsplats_3d_organoid_dapi_nuclei.py - 3D Organoid DAPI Nuclei
Gaussian splatting compression of real 3D confocal microscopy data (DAPI-stained cell nuclei) from Image Data Resource (IDR).

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_organoid_dapi_nuclei.py`

**Requires**: Internet access (downloads OME-ZARR from IDR). Supports Metal (MPS) acceleration on Apple Silicon.

**Demonstrates**: 3D Gaussian splat fitting to real microscopy volumes, 20-50x compression vs raw voxels, oriented ellipsoids capturing elongated nuclear shapes, IDR/OME-ZARR data loading.

---

#### demo_gsplats_3d_organoid_multichannel.py - Multi-Channel 3D Organoid (IDR Pipeline)
Multi-channel 3D microscopy data as Gaussian splats, with full compute pipeline from Image Data Resource (IDR). Uses precomputed gsplats from Git LFS by default.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_organoid_multichannel.py [--recompute]`

**Requires**: Git LFS data (default) or internet access + GPU (with `--recompute`).

**Demonstrates**: Multi-channel Gaussian splatting with distinct colors per channel, full pipeline (fetch from IDR, fit per channel, merge), precomputed gsplats via Git LFS for fast demo.

---

#### demo_gsplats_3d_kidney_multichannel_toggles.py - 3D Multi-Channel Kidney (Boolean Toggles)
Three-channel confocal mouse kidney tissue (nuclei, WGA, actin) with independent boolean toggle dimensions for each channel.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_kidney_multichannel_toggles.py`

**Requires**: `scikit-image` package (for `kidney()` sample data, FluoCells Prepared Slide #3).

**Demonstrates**: Three-channel boolean toggles (8 visibility combinations), `extend_to_all` for cross-channel visibility, confocal fluorescence microscopy data, comparison with Layers panel approach.

---

#### demo_gsplats_3d_kidney_multichannel_layers.py - 3D Multi-Channel Kidney (Layers Panel)
Same kidney dataset as the toggles variant, but uses the **Layers panel** (`layer=True`) instead of boolean dimensions for per-channel visibility control.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_kidney_multichannel_layers.py`

**Requires**: `scikit-image` package.

**Demonstrates**: Layers panel (press **L**) with visibility toggle, display range, gamma, and blending mode per channel. Keeps the scene at 3D (no extra nD dimensions) while providing rich per-channel controls.

---

#### demo_gsplats_3d_acto3d_heart.py - 3D Mouse Embryo Heart (Acto3D / Zeiss Lightsheet 7)
Three-channel light-sheet microscopy volume of an E13.5 mouse embryo heart as Gaussian splats with per-channel colors.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_acto3d_heart.py`

**Requires**: Internet access (downloads ~1.65 GB TIFF from Google Drive), GPU recommended.

**Demonstrates**: Multi-channel light-sheet microscopy, 960 x 960 x 597 voxel volume, per-channel Gaussian splatting with distinct colors, Zeiss Lightsheet 7 data.

---

#### demo_gsplats_3d_opencell_map4.py - OpenCell MAP4 Cytoskeleton
Two-channel confocal z-stack of endogenously tagged MAP4 (microtubule-associated protein 4) in HEK293T cells from the OpenCell project, with Hoechst nuclear stain.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_opencell_map4.py`

**Requires**: Internet access (downloads from OpenCell, ~70 MB).

**Demonstrates**: Multi-channel Gaussian splatting (MAP4-GFP + Hoechst), spinning-disk confocal microscopy, OpenCell project data, cytoskeleton visualization, 51 x 600 x 600 voxel volume.

---

#### demo_gsplats_3d_tribolium_embryo.py - 3D Tribolium Embryo (Cell Tracking Challenge)
Large isotropic 3D light-sheet volume of a developing beetle (*Tribolium castaneum*) embryo using Gaussian splatting. Near-isotropic at 0.381 um per voxel.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_tribolium_embryo.py`

**Requires**: Internet access (downloads ~2.6 GB from Zenodo), GPU recommended. 965 x 1871 x 991 voxels.

**Demonstrates**: Large-volume Gaussian splatting, isotropic light-sheet microscopy, Zenodo/Cell Tracking Challenge data, Zeiss LightSheet Z.1 data.

---

#### demo_gsplats_lod_tribolium.py - Adaptive Level of Detail on the Tribolium Embryo
Takes the precomputed Tribolium embryo fit and builds an **adaptive Level of Detail (LOD)** pyramid on it — the embryo is stored at several resolutions, and the viewer shows the simplest one that still looks right at the current zoom. This demo uses *substitutive* LOD (each coarser level *replaces* the finer one with fewer, larger splats), and ships it with per-level debug colors (green→amber→red, finest→coarsest) so the viewer's pixel-size level-switching is visible as you zoom. The scaled-up companion to `examples/gsplats_lod_example.py`.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_lod_tribolium.py`

**Requires**: Precomputed Tribolium splats (Git LFS); no network/GPU needed for the default path. `--recompute` re-fits from Zenodo (network + GPU).

**Demonstrates**: Substitutive LOD via `make_substitutive_lod` (`kmeans_lloyd`), `add_gsplats_from_data(lod_group=True)`, auto level-count from base splat count (#levels scales as log_K(N)), per-level debug coloring, pixel-size LOD selection in the viewer. Options: `--levels=N`, `--factor=K`, `--method=NAME`, `--serve-only`.

---

#### demo_gsplats_lod_embryo_line.py - Near-Unlimited Scaling with Adaptive Level of Detail
Lays out `--count` (default 100) copies of the single adaptive-detail Tribolium embryo along a straight line, drops the camera near the middle of the line, and lets you fly down it. **Level of Detail (LOD)** is the idea that makes this scale: each embryo is kept at several resolutions, and the viewer picks — per object, every frame, by projected pixel size — the simplest version that still looks right. Near embryos render fine (green) while distant ones collapse to a few big splats (red), so the detail actually drawn stays roughly bounded by what the screen can resolve, no matter how long the line. All copies share **byte-identical splat arrays** (per-embryo orientation/jitter lives only in scene-graph transforms), so the encoder's `array_ref` deduplication stores the geometry once: 100 embryos cost ~55 MB on disk instead of ~1.3 GB.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_lod_embryo_line.py`

**Requires**: Precomputed Tribolium splats (Git LFS); no network/GPU needed for the default path. `--recompute` re-fits from Zenodo (network + GPU).

**Demonstrates**: Per-object pixel-size LOD selection at scale, scene-graph transforms (`add_group(transform=...)`, `transforms.compose`/`rotate`/`translate`) for placement so splat arrays stay identical, automatic `array_ref` array deduplication in the encoder, and initial-camera setup via `ViewerConfig(camera=CameraConfig(...))`. Options: `--count=N`, `--levels=N`, `--factor=K`, `--method=NAME`, `--serve-only`.

---

#### demo_gsplats_4d_zebrafish_timelapse.py - 4D Zebrafish Embryo Time-Lapse
4D (3D + time) confocal recording of a living zebrafish embryo during gastrulation, with per-timepoint Gaussian splatting and a time dimension slider.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_zebrafish_timelapse.py`

**Requires**: Internet access (downloads ~2.1 GB LSM from Zenodo), GPU recommended.

**Demonstrates**: 4D Gaussian splatting (3D + time), per-timepoint independent fitting, `dim_order` + `fill` for time coordinate assignment, zebrafish gastrulation imaging, Zeiss LSM format.

---

#### demo_gsplats_4d_celegans_tracking.py - 4D C. elegans Nuclei Tracking with Lines
4D confocal time-series of a developing *C. elegans* embryo: Gaussian splats for volume rendering combined with polylines for tracked cell nuclei trajectories.

**Run**: `hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py`

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
        output_path = Path(tmpdir) / "demo.zarr"

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
    output = Path(tmpdir) / "data.zarr"
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
hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_integrated_cells.py
hatch run python packages/luxar/src/luxar/demos/demo_human_multiome_peak_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_mouse_multiome_peak_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_zebrahub_multiome_peak_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap.py
hatch run python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap_sequence.py

# --- Data-Driven (External Datasets) ---
hatch run python packages/luxar/src/luxar/demos/demo_gaia_milky_way_3m.py
hatch run python packages/luxar/src/luxar/demos/demo_gaia_milky_way_8m.py
hatch run python packages/luxar/src/luxar/demos/demo_earthquakes_3d.py
hatch run python packages/luxar/src/luxar/demos/demo_storm_3d_microtubules.py
hatch run python packages/luxar/src/luxar/demos/demo_huri_interactome.py
hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py
hatch run python packages/luxar/src/luxar/demos/demo_caida_as_topology.py

# --- GSplats: 2D ---
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_2d_codex_pancreas.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_2d_cmu1_pathology.py

# --- GSplats: 3D ---
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_organoid_dapi_nuclei.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_organoid_multichannel.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_kidney_multichannel_toggles.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_kidney_multichannel_layers.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_acto3d_heart.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_opencell_map4.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_tribolium_embryo.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_lod_tribolium.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_lod_embryo_line.py

# --- GSplats: 4D ---
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_zebrafish_timelapse.py
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
# For manual testing: luxar serve data.zarr --viewer --port 8001 --viewer-port 5174
```

## Philosophy

These demos are **teaching tools**. They should be:
- Simple enough to understand in 5 minutes
- Complete enough to show real capability
- Clean enough to use as templates
- Fun enough to inspire creativity!

Keep them self-contained so anyone can read ONE file and understand the complete workflow.
