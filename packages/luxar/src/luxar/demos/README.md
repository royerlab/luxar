# Luxar Demos

This folder contains self-contained demo scripts that showcase Luxar's capabilities. Each demo is a complete, runnable Python script that generates data, launches the viewer, and cleans up automatically.

## Purpose

These demos are:
- **Didactic**: Easy to understand and learn from
- **Self-contained**: The code that makes the demo distinctive lives in one file
- **Complete**: Generate → Serve → View → Cleanup workflow
- **Copy-pasteable**: Can be used as templates for your own visualizations

## Quick Start

List the bundled demos, then run one by key or index — it generates the data
and opens the viewer for you:

```bash
luxar demo              # list all demos (key, index, requirements)
luxar demo run lorenz   # generate the Lorenz attractor demo and view it
```

Some demos need optional dependencies; run `luxar demo deps --install` to add
them. See the sections below for the full CLI and dependency details.

## Running Demos (the `luxar demo` CLI)

The easiest way to run any demo is the CLI (equivalent to the per-demo
`hatch run python ...` commands listed below):

```bash
luxar demo                  # list all demos (key, index, requirements)
luxar demo info lorenz      # metadata for one demo
luxar demo run lorenz       # run by key or index; forwards -- args
luxar demo run lorenz -- --no-serve --points=10000
luxar demo run-all          # generate demo datasets (--no-serve; skips GPU/large-download by default)
luxar demo stop             # stop demos left running (frees their ports); --dry-run to list
luxar demo cache list       # inventory demo caches under ~/.cache/luxar/
luxar demo deps                    # which optional dependencies are missing?
luxar demo deps --install          # install the extras that provide them
luxar demo deps --only scipy       # report one import module
luxar demo deps --only scipy --install  # install only scipy's constrained spec
```

A cache directory holding a **hand-placed input** — bytes with no download path,
listed in `luxar.demos.registry.PROTECTED_INPUT_DIRS` — is inventoried by `demo
cache list` (marked `🔒 hand-placed input`) but is never reported as an orphan and
never deleted by `demo cache clear`, by key, under `--all` or under `--orphans`.

A `local/` subdirectory inside a cache dir holds artifacts **this machine
computed for itself** — a demo's own refit, when its hosted data could not be
reached — written there by `luxar.demos.local_fit_path`. That is a
separate namespace from the manifest's own `~/.cache/luxar/<dataset>/<file>`,
which the fetch checksums and quarantines; a local fit stored under the hosted
name is destroyed and recomputed on every launch (#1618). `demo cache clear`
counts anything under `local/` as *computed*, so `--no-computed` spares it.
That classification keys on the `local/` path and nothing else: a demo that
still writes an expensive computed artifact straight into its cache dir — the
NEXRAD per-frame fits, the C. elegans preprocessed frames — is classified as a
download and IS deleted by `--no-computed`. Those names collide with no pinned
manifest entry, so they are not exposed to the #1618 quarantine and were left
where they are.

Eleven gsplat demos also accept `--show-roundtrip`, which renders the fitted
splats back and shows original / reconstruction / absolute-difference panels
with PSNR and MSE (needs `matplotlib`). The five whose figure is one row per
channel through a 3-D volume share the implementation in
`_roundtrip_common.py`; the other six keep their own — laid out over 2-D
images, over sampled timepoints, or with demo-specific titles for a
single-channel volume.

The two FlyLight MCFO demos share `_h5j.py`, which identifies reference and
signal channels from H5J metadata and decodes the stitched HEVC channel payloads.

The LOD topology a fitting demo writes its cached artifact with is chosen in
`_lod_policy.py`, not left to whichever fitter the demo happened to call
(`fit_gaussian_splats` returns one additive sub-LOD, the progressive fitter
several, which is how five shipped archives ended up with no ladder at all).
Demos call `save_with_lod(result, cache_file, recipe=...)` in place of
`result.save(...)`; the per-recipe parameters live in one table there so two
demos asking for `levels` cannot drift apart. `adaptive` is the one that changes
how a demo READS its cache back — it writes a `kind=partition` tree, which has
no flat `GSplatData` form, so those demos fetch paths with `ensure_dataset` and
graft each with `add_gsplats_from_file`. Anything costlier than `stream` is also
conditional on how the demo BUILDS its scene: `add_gsplats_from_data` and
`add_gsplats_from_file` carry a stored topology through, while plain
`add_gsplats(centers=…, amplitudes=…)` writes a flat leaf, so a demo that
rebuilds from arrays would pay `levels`' extra ~38% and then discard it.
`tests/test_lod_policy.py` holds both gates: a fitting demo either routes every
archive through the policy or is accounted for by name — it ships no artifact, it
names the topology itself with a literal `build_recipe(...)`, or it sits on the
shrinking pending list — and a demo choosing `levels`/`adaptive` must reach the
scene through an adder that preserves it, whichever of those doors it came
through. The choice applies from the next refit onwards — the
hosted archives keep whatever topology they were written with until they are
refitted and reuploaded, since the manifest pins their checksums.

Every demo scene opens in the cinematic look, and it costs one keyword:
`ViewerConfig(cinematic_mode=True)`, which the viewer's zarr bridge expands into
ACES, a subtle wide bloom, detector noise, a vignette and a 35 mm chromatic lens
for every field the scene did not set itself — so an explicit `tone_mapping`,
`exposure` or bloom value still wins. `tests/test_demos_cinematic_mode.py` is the
gate: no `create_scene` without a `viewer_config`, and no `ViewerConfig` without
a literal `cinematic_mode=True`.

The preset also expands the field of view from 47° to 63°. The viewer resolves
that FOV before automatic framing, so an auto-framed scene keeps the fitted
subject occupancy intended for the lens. A returning visitor's stored FOV takes
precedence only for auto-framed scenes; an authored position is restored with
the scene FOV it was composed for.

An authored camera position is a stronger contract, because its distance was
composed for one FOV. Every authored pose **composes for 63°** through
`_cinematic_camera.py`: read `CINEMATIC_FOV_DEG` when the distance comes from
the lens, or use `framing_scale()` / `pull_in()` with the pose's original
38°–50° FOV. Most poses preserve their authored framing exactly; the
biodiversity globe preserves its silhouette, while the forest and embryo-line
poses preserve camera clearance instead. `pull_in()` scales about the target,
so an off-origin camera keeps its aim. The guard rejects both a mixed-lens FOV
pin and a bare authored position whose FOV assumption nobody can read.

`_audio_synth.py` is the build-time synthesis + AAC encoding helper behind the sound layer's effects: numpy PCM → 16-bit WAV → `afconvert` (macOS) or `ffmpeg`, a loop-clean `synthesise_hum(frequency_hz, seconds, cache_dir)`, equal-power loop-point blending (`loop_crossfade`), first-order ambisonic encoding (`encode_foa`, `foa_from_stereo`, AmbiX ACN/SN3D) and `synthesise_foa_from_clip(src, cache_dir)` which decodes an existing MP3/AAC and re-encodes it as a 4-channel field. Clips are cached by a hash of the synthesis parameters; `LUXAR_AUDIO_ENCODER=afconvert|ffmpeg|none` overrides the detection (`none` keeps a CI build silent). Without an encoder the demo builds without the effect and warns once.

`_narration.py` is the build-time text-to-speech helper behind the sound layer's narration (`synthesise(text, voice, cache_dir)`): OpenAI TTS over plain HTTPS when `OPENAI_API_KEY` is set, else the macOS system voice (`say` → `afconvert` to AAC), else a `UserWarning` and no clip. Clips are cached by a hash of `(engine, model, voice, text)` under the calling demo's cache dir, so an unchanged text never re-synthesises; `LUXAR_NARRATION_ENGINE=openai|say|none` overrides the detection (`none` keeps a CI build offline).

Four demos pin scientific-fidelity exceptions. The two quantitative ortho demos
suppress bloom, vignette, lens distortion and detector noise so their
projection-derived scale bars and measured intensities remain meaningful. The
biodiversity globe and nD transform bench suppress lens distortion and detector
noise because their categorical or exact RGB hues carry data.

The three network demos (`caida_as_topology`, `huri_interactome`,
`ppi_flow_field`) share `_graph_common.py`, but not all of it. All three use the
cache-aware download and Louvain community detection; the sparse adjacency and
the community legend are shared by `caida_as_topology` and `huri_interactome`
(`ppi_flow_field` builds its own signed matrix). The HuRI/HGNC loaders and the
largest-connected-component filter belong to the two protein demos only —
`caida_as_topology` keeps its own LCC filter, which also detects tier-1 ASes,
sorts on a numeric ASN key and reindexes an org/country frame. `build_adjacency`
and `compute_communities` take the two endpoint column names explicitly, because
the AS graph keys on `asn_a`/`asn_b` and the protein networks on
`sym_a`/`sym_b`; the HuRI loaders hardcode `sym_a`/`sym_b`, being
protein-specific by construction.

Leaf implementation utilities shared by demos live under `_support/`; the flat
`_*.py` helpers remain at the package root.

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
| `luxar demo deps` | The installer/report. Surveys the table with `find_spec` (no imports), flags an installed-but-below-pin package `OUTDATED`, and in report-only mode exits 1 if anything is missing or out of date. `--only MODULE` narrows the row and installs its exact constrained requirement instead of a whole extra. |

Two rules govern the gate, both learned from real bugs:

1. **Gate at the point of use, never at the entry point.** A demo whose
   expensive artifact is already cached must run *without* the dependency that
   produced it. `tests/test_no_entrypoint_dependency_preflight.py` fails the
   build if an entry-point preflight reappears.
2. **Advertise the constrained requirement.** A bare `pip install metpy` resolves
   1.5.x, which declares only `numpy>=1.20` and then breaks at runtime against
   Luxar's `numpy>=2.0`.

`tests/test_demos_dependencies.py` enforces both: every spec must accept exactly
the versions its `pyproject.toml` pin accepts, every module passed to
`require_module` must exist in the table, and no scanned source — runtime message
or docstring alike — may spell out `pip install <pkg>` for a package the table
bounds without carrying that bound. Those guards — along with the substitutive-LOD
one in `tests/test_substitutive_lod_gated.py` and the fit-provenance one in
`tests/test_demo_fit_provenance.py`, which forbids saving a real fit with
`include_fitting_info=False` — read the set defined by
`tests/_scanned_modules.py`: every `*.py` directly under `demos/` **except**
`__init__.py` and `_dependencies.py`, plus every non-barrel helper in
`demos/_support/`. It is a denylist, not a `demo_*.py` glob or a `_*_common.py`
pattern, so a gate that moves out of a demo into a shared helper cannot escape
the guards — and any new module dropped in either location is covered without an
edit there. The flip side: a scratch `.py` file left in either directory is
scanned too, so keep throwaway scripts out of the package tree.

Installing everything:

```bash
make install-demo-deps             # hatch env: demos + gsplats + io extras
luxar demo deps --install          # missing extras, any environment
luxar demo deps --only scipy --install  # one exact constrained requirement
```

Generic `--install` manages Luxar extras. If the only unmet row belongs to no
extra (currently `gdown`), it reports a successful no-op and points at
`--only gdown --install`; that targeted form installs and verifies the exact
tabled requirement. Report-only mode remains the CI/setup gate and exits 1 for
any unmet row.

Some demos need something a package manager can't supply — a Kaggle credential,
a manual download, a `git lfs pull`, or a GPU. Those show up in the trailing
requirements column of `luxar demo` (as `kaggle`, `manual`, `git-lfs`, `GPU`,
`GPU?`) and in `luxar demo info <key>`, not here.

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

#### demo_volumetric_cloud.py - Evolving Cumulus (3D + time)
A convective cumulus lived through its whole life cycle on a hidden `time` axis: a low fragment at the condensation level, a cauliflower turret billowing upward, a mature top leaning downwind, then the whole body pulling in as the thermals feeding it die.

**Run**: `luxar demo run cloud [-- --parcels=1000000] [-- --frames=240]`

120 timepoints, ~2.7M points, about 140 s of CPU time to generate, 22.4 MB on disk, and about 1.1 GB peak RSS; scale `--parcels` down on a memory-tight machine. `--frames` is a pure **resolution** knob: speeds are expressed per unit phase and each frame advances by `dt = 1/(n_frames-1)`, so every frame count samples the same evolution, just more finely. Integrating a fixed displacement once per frame — which is what it did before — made the distance travelled proportional to the frame count, so asking for twice the temporal resolution would have silently given you a cloud that drifted twice as far.

Three things, kept separate. *Where the air goes* is an analytic velocity field built to be exactly divergence-free — an axisymmetric convection roll written through a Stokes stream function, plus a wind shear that leans the cloud downwind, plus a slow swirl — through which 600k parcels are pushed with midpoint steps, so a point is a parcel of air that keeps its identity frame to frame. (Solenoidality is load-bearing: the parcels are a Monte Carlo sample of a uniform density, and only a divergence-free field keeps that sample uniform as it deforms. Measured over the full 120-frame run at the shipped 600k parcels, the core parcel count holds to within about 2%.) *What shape the cloud is* is the union of rising thermal bubbles, rooted in a slab of cloud sitting on the condensation level — because a cumulus is not a shape but a process, a succession of buoyant bubbles punching up through the LCL, and rendering their union is what produces the cauliflower. *Where the water is* is 4D fractal noise in **material** coordinates — each parcel's fixed label — which welds the texture to the fluid so it stretches and folds instead of boiling in place.

The noise's time axis is a stack of static 3D fields at the fixed material coordinates, quintic-interpolated between — exactly 4D value noise, for the price of a lerp — with temporal frequency growing as `2^(2k/3)` rather than `2^k`, in the spirit of Kolmogorov eddy-turnover scaling, because at the naive rate the fine octaves read as shimmer.

**The renderer has to be in the same optical regime as the shading**, and this is where the demo went most wrong. A cumulus is optically *thick*: you see its surface. `additive` blending models the opposite regime — an optically thin emissive medium that ignores depth entirely. Shading each parcel by the water above it and then compositing with a mode that cannot occlude painted a dark interior that was fully visible *through* the lit shell: the cloud rendered as a glowing archway with a hole in the middle, and from overhead as a ring. Measured on that build, the core carried the **highest** point density (346 vs 51 per unit area at the rim) and the **lowest** brightness (0.46 vs 0.85) — a hole in the light, not in the geometry. The node is `volumetric`, with per-point RGBA alpha as optical depth.

Light is baked in three terms because points are emissive: direct sun attenuated along the sun's own ray, blue skylight attenuated along the vertical, and a weak ground bounce. The sun is deliberately **off to one side** — a light directly overhead illuminates every surface by depth alone, so two lobes at the same altitude are lit identically however they face and the relief vanishes.

Several details here were found by measurement, never by reading, and each is commented at the site because none of them announces itself:

- The old positional hash (`(xi*C1 + yi*C2 + zi*C3) % M`) left the value correlated with coordinate magnitude, so near the lattice origin every corner returned nearly the same number. Combined with sampling less than one lattice cell, the "7-octave fractal noise" was a smooth ramp, and the demo emitted **790 points out of 800,000 candidates**. A 32-bit avalanche finalizer fixes it.
- Blending two independent time keyframes with weights summing to one halves the variance mid-interval. Invisible spatially; along *time* every parcel shares one weight, so the whole cloud breathed in and out of focus.
- The threshold is standardized over the parcels *inside* the envelope, per frame. Measuring once is not enough (the coarsest octave's spatial mean walks with time); measuring globally is not enough either (the flow keeps replacing the air inside the cloud).
- Exposure has to leave headroom: under emission-absorption a thick medium's radiance tends to the parcel colour itself, and the preset's bloom threshold of 0.01 blooms the *whole* cloud onto itself. Three light terms that each looked reasonable summed to 1.4 and clipped every lit face to flat white.
- Once the medium is opaque, dissipation must show in the **silhouette**. Expressing decay by raising the noise threshold erodes the interior, which can no longer be seen at all.

**Demonstrates**: A 3D+time Points scene with `time` as a hidden discrete axis (integer frames, `step=1`, so the viewer's ±0.5 membership gate selects exactly one), `volumetric` emission-absorption blending with per-point RGBA optical depth, Lagrangian advection through an analytic solenoidal field, 4D fractal noise in material coordinates, a metaball-style union of thermals, baked three-term lighting for an emissive geometry type, a frozen per-parcel emission threshold so point density tracks condensate without flickering, an opening camera composed for the cinematic 63° lens from the data's own extent, an authored opening timepoint (`current_step`), a turntable (`auto_rotate`) at roughly one revolution every 26 seconds, and time playback running from the moment it loads (`animation`, indexed by dimension so only the hidden `time` axis runs). `K` pauses. Playback rate is a ceiling rather than a promise — the viewer throttles to what chunk streaming sustains, which for this scene measures about 4 fps.

---

#### demo_cubic_array.py - 3D Cubic Grid with Star Background
Dense 100-cubed grid (1M points) with 500k background stars.

**Run**: `luxar demo run cubic_array`

**Demonstrates**: Regular grids via meshgrid, multi-layer scenes (foreground + background), very high point density (1.5M total), different blending modes (additive vs normal), semi-transparent background layers.

---

#### demo_mandelbulb.py - 3D Mandelbulb Fractal
Stunning volumetric representation of the famous Mandelbulb 3D fractal.

**Run**: `luxar demo run mandelbulb [-- --resolution=128] [-- --power=8]`

**Demonstrates**: 3D fractal mathematics (extension of Mandelbrot set), distance estimation for surface detection, orbit-trap coloring, distance-field normals and ambient occlusion, baked key lighting, adaptive point sizing, spherical coordinate transformation, escape-time algorithm in 3D.

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

**Demonstrates**: Molecular machine visualization, multi-subunit protein complex, color-coded structural components (alpha, beta, gamma, c-ring), biological energy production machinery (~600 kDa enzyme), `volumetric` emission-absorption blending (kappa 2.5) so the packed interior reads as density instead of an opaque shell, with a raised display gain (`intensity=1.62`) as the matching exposure, `layer=True` for live blending / opacity / display-range control in the Layers panel (press **L**; the absorption slider is shown because the layer is `volumetric`).

---

#### demo_nuclear_pore_complex.py - Nuclear Pore Complex
The complete human NPC at atomic resolution: **4,937,064 atoms, 808 protein chains, 25 distinct nucleoporins**, assembled from the PDB deposition's *own* eight-fold symmetry operators and scrubbable between the constricted and dilated conformational states.

Uses PDB **7R5J** (dilated) / **7R5K** (constricted) — Mosalaganti et al., *Science* 2022, the reference whole-NPC model. Each entry deposits one C8 protomer (101 chains, 617,133 atoms) plus the eight operators that generate its declared `808-meric` biological assembly, so nothing about the radius, orientation or spacing is invented here. Measured on the assembled result: outer diameter 149.9 nm (constricted) / 159.7 nm (dilated), central channel 41.1 / 53.1 nm, axial height 72.2 / 76.5 nm. Atoms carry true van der Waals radii — no visibility fudge factor.

Six structural modules, every chain assigned by a curated table that the test suite refuses to let drift: cytoplasmic filaments (RanBP2/Nup358 x40, the Nup214-Nup88-p62 export platform), cytoplasmic ring and nuclear ring (32 Y-complexes = 16 + 16, two concentric rings per face, ELYS nuclear-only), inner ring (Nup205/188/93/155/35), membrane ring (gp210 x64, NDC1, ALADIN), and the central channel FG nucleoporins. Absent and documented as such: the nuclear basket (Tpr/Nup153/Nup50), most FG repeat regions, and the membrane itself.

**Run**: `luxar demo run nuclear_pore_complex`

**Options**: `--state=both|dilated|constricted`, `--color=module|nucleoporin|element|protomer`, `--representation=all|backbone|calpha`, `--split=none|nucleoporin`.

**Requires**: Internet access (~28 MB of mmCIF from RCSB).

**Demonstrates**: mmCIF parsing and biological-assembly expansion from deposited `_pdbx_struct_oper_list` operators; a 9.87M-element Points scene as **one** BSP-partitioned node — the NPC's subunits are concave and interpenetrate, so splitting by protein has no valid draw order while splitting by space does, and the recorded `bsp_tree` gives the viewer an exact Fuchs-Kedem-Naylor back-to-front traversal even with the camera inside the channel; a bounded streaming ladder inside every spatial part; a hidden **categorical** `state` axis placed **last** so `displayDims == [0, 1, 2]` and the BSP split columns stay displayed; symmetry-averaged baked ambient occlusion (`luxar.shading`) as a burial cue; depth-sorted `normal` blending for surface-like atomic structure; `layer=True` for live control in the Layers panel (press **L**).

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

**Run**: `luxar demo run fractals_4d [-- --grid=64]` (default: 147 MB,
~25 min, ~24 GB peak memory)

**Demonstrates**: 4D spatial navigation (XYZ + W dimension, every W slider stop shows structure), categorical dimension (select between 6 fractal types), large dataset (~44.5M points, up to ~7.5M per fractal), XOR Fractal, Menger Sponge 4D, Sierpinski 4D, Cantor Dust 4D, Checkerboard and Diamond patterns.

---

#### demo_galaxy_simulation.py - Galaxy Simulation (density-wave spiral)
A spiral galaxy integrated from its own mass model rather than drawn along logarithmic curves. Hernquist bulge + Miyamoto-Nagai disc + NFW halo give a flat rotation curve (260 km/s out past 30 kpc); `Omega` and the epicyclic frequency `kappa` are differentiated off it; each star rides a forced epicycle whose spiral response carries the resonant denominator `kappa^2 - m^2 (Omega - Omega_p)^2`. Because every star turns at its OWN `Omega(R)` while the pattern turns rigidly at `Omega_p`, stars visibly **overtake** the arms inside corotation (11.4 kpc) and lag outside it — and the arms do not wind up over the 480 Myr span, which is the point. Ages drive the rest: `sigma_R ~ age^0.33` heats old stars out of the arms, and colour is the blackbody colour of each population's effective temperature integrated through the CIE 1931 observer. Dust lanes on the upstream arm edge redden as well as dim; HII regions mark the shock.

Replaces the former `demo_5d_spiral_galaxy.py`, which scattered points along fixed logarithmic curves and rotated the whole picture rigidly — the material-arm model the winding problem rules out.

**Run**: `luxar demo run galaxy_simulation [-- --stars=N] [-- --frames=N]` — 100k disc stars over 241 frames of 2 Myr by default (29.7M points, 218 MB, ~1.3 min). Radii and per-point gain track the sample density, so `--stars` changes the weight of the scene rather than how bright it looks.

**Demonstrates**: 4D data (X, Y, Z, Time) with the stellar-age split exposed as five toggleable layers rather than a slicing dimension, a **discrete** time axis so the slider snaps to the 2 Myr frame grid instead of stranding the view between frames, `extend_to_all` for the time-independent halo, physically derived rotation curves and orbit frequencies, and blackbody colour synthesis.

---

#### demo_nd_transforms.py - nD Transform Test Bench
An instrument, not a picture: a calibrated bench that makes `nd_transform` readable tick by tick. Everything sits against a ruler along X where one tick = one frame index. A cyan cursor column (plain untransformed geometry tagged `Frame = k`) marks the WORLD index on the slider; each row is a group carrying ONE `nd_transform`, and its markers are 3D point-font digits that print their own LOCAL index — so whichever digit lights up IS the local frame the inverse-query resolved to, and the gap between digit and cursor, counted in ruler ticks, IS the transform. Faint always-on ghost digits (`extend_to_all`) mark every slot a row could light, so a dark row reads as "no local frame maps to this T" rather than "the row failed to load". A `visible_range`-gated readout prints the EXPECTED local index for every row at the current slice, computed from the same definitions that placed the data — render vs readout disagreement is the failure signal.

Rows: identity, `offset +5`, `offset -3`, `scale ×2`, `scale ×2 offset +2`, a nested `×2`-then-`+1` pair that must land on the same tick as the flat row above it (a real composition-**order** test — the wrong order lights the opposite parity), and a reversing `scale -1 offset +15`. A second section does categorical permutations with colour-coded letters that carry their own local channel identity.

**Run**: `luxar demo run nd_transforms`

**Demonstrates**: `nd_transform` affine on a discrete ordinal dimension (offset, scale, negative scale, scale+offset), categorical permutation, hierarchical composition through nested groups, a spatial 4x4 `transform` and an `nd_transform` on the same group, `extend_to_all` for static furniture and per-section pinning, `visible_range` overlays as a live expected-value readout, and composite group layers (`Frame_Section` / `Channel_Section` are the only `layer=True` nodes, so the Layers panel offers one row per half of the bench instead of one per marker). Doubles as the visual regression harness for the viewer's no-preimage rule (see `packages/luxar-viewer/src/data/transforms/README.md`).

---

### Feature Showcase Demos

#### demo_sharpness_showcase.py - Point Sharpness Showcase
Comprehensive showcase of the point sharpness feature: gradient from peaky (0.0) to hard-edged (1.0) on the normalized knob, fixed sharpness comparison rows, mixed sharpness cloud, and sinusoidal wave pattern.

**Run**: `luxar demo run sharpness_showcase [-- --points N]`

**Demonstrates**: Sharpness parameter control (normalized [0, 1] knob mapping to a super-Gaussian falloff exponent beta=2^(6s-2); 0.5 = Gaussian), soft/peaky points (low s) vs sharp disc-like points (high s), color-coded sharpness values, multiple visualization patterns.

---

#### demo_lsystem_forest.py - L-System Forest: a Year in a Growing Forest (All Four Geometry Types)
The flagship synthetic scene: a terrain-planted procedural forest scrubbable through time on two non-displayed dimensions — `growth` (six stages, each a genuine re-derivation of every tree at increasing iteration depth, staggered per tree so maturity rolls across the field in waves) and `season` (spring blossom, summer green, autumn fire, winter frost). All four geometry types share the frame: a shaded fBm-heightfield **Mesh** terrain (snow in winter), eight tree species as merged indexed **Lines** nodes (one Layers-panel row per species, per-vertex hover labels with species/instance/season/stage), volumetric **GSplat** foliage clouds oriented along their parent branches (blossom splats in spring, fire palette in autumn, evergreen conifers in winter), and **Points** accents (summer fireflies, winter frost sparkle, spring petals, each pinned to its season and extended over growth).

**Run**: `luxar demo run forest [-- --trees=800] [-- --iterations=5]`

**Demonstrates**: stochastic L-system productions, tropism (ABOP §1.7 — gravity droop for the willow and palm, upward phototropism for the columnar poplar, applied only at branch depth >= 1 so trunks stay straight), apical dominance via a leader symbol (Honda's monopodial conifer), indexed line networks with shared joint vertices and `normal` blending (trunks occlude), hand-packed 5D gsplat Cholesky factors for stacked categorical axes, `extend_to_all` + `fill` for season-pinned props, per-season mesh vertex colors, season/growth-conditional overlays (`visible_range`), a grammar card showing the actual production rules, an authored `ViewerConfig` (explicit ACES, forest-edge opening camera, autumn/ancient opening slice), and `cache_computed` for instant warm regeneration.

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

#### demo_arxiv_embeddings_kaggle.py - arXiv Papers (Kaggle / OpenAI)
Visualizes the **whole** `tomtum/openai-arxiv-embeddings` corpus — 3,286,365 preprints
(2,902,228 arXiv + 308,367 bioRxiv + 75,770 medRxiv, through 2025-12) with pre-computed
`text-embedding-3-large` vectors (3072D), projected to 3D by UMAP.

**Run**: `luxar demo run arxiv_papers_kaggle`

**Requires**: Internet access for the one-time ~30 GB embeddings download plus the ~1.8 GB
Cornell arXiv metadata snapshot (budget ~39 GB of disk under `~/.cache/luxar/`,
since the metadata ZIP is also extracted), and `scikit-learn` + `umap-learn`. No
Kaggle credentials are needed.

The 3072D vectors are never held in RAM: `vectors.dat` is streamed in blocks and projected
through a PCA basis down to 128D (fitted on a 300k-row uniform subsample), and only that
`(N, 128) float32` matrix is cached and handed to UMAP. Pass `--sample=N` for a *uniform
random* subset on a smaller machine, and `--pca-dim` / `--device` to trade quality for time.
The sample bounds UMAP, metadata, and label RAM/time; a cold run still downloads the full
archive, streams every vector twice, and writes the corpus-wide PCA cache. When launching
through `hatch run`, override its one-thread CPU defaults, for example with
`OMP_NUM_THREADS=16 MKL_NUM_THREADS=16`.

**Demonstrates**: Multi-million-element embedding visualization, streaming decode of a
40 GB binary attached to a ZIP, PCA pre-reduction before UMAP, optional GPU (cuML) UMAP,
self-calibrating point radii, and bounded additive Points streaming.

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

**Requires**: Internet access (downloads Swiss-Prot from UniProt), `esm>=3.0.0` (in the `demos` extra); a CUDA GPU to compute embeddings (first run computes ESM embeddings + UMAP, ~5h). Subsequent runs load cached results — and load them without `torch`, `esm` or `umap-learn` installed at all, because each is demanded only at the point where the corresponding uncached computation happens rather than as an entry-point preflight. If an earlier run left a quarantined `*.corrupt` artifact in `~/.cache/luxar/esm3_swissprot/`, the demo reports its path and size up front — re-download the complete file or delete the quarantined copy, otherwise the run starts over from scratch.

**Demonstrates**: Protein language model embeddings (ESM-3 / ESM C), large-scale embedding visualization (~572k proteins), UMAP dimensionality reduction, taxonomic-kingdom coloring, hover labels.

---

#### demo_esm3_protein_stories.py - ESM Protein Stories (ten guided tours)
The ESM C Swiss-Prot 3D UMAP with a hidden `story` dimension: Overview, then ten protein-family clusters — hemoglobin, photosystem II D1, Hsp70/DnaK, viral surface proteins, the prion protein, ATP synthase, RuBisCO, RecA and Rad51, insulin, cone-snail conotoxins — each with an authored camera `Waypoint`, a highlight layer and a right-hand panel of sourced facts closing on an open question. A story is a protein-name pattern plus a radius around the family's **densest** member (not its median: families the model splits into several blobs, like RuBisCO or the histones, have a median that lands between them), so the highlight is the visible blob, not the family's stragglers. Each cluster also gets a thin soap-bubble shell: a smooth icosphere with the physical material, full transmission, water-like IOR, iridescence and dispersion. A scene-captured environment lights its reflections, while `refract_data=True` bends the protein map behind the bubble and keeps data in front crisp. Auto-rotate is on by default, so a story step keeps the turntable spinning while it moves the point the camera spins around (the viewer's keep-orientation flight).

On the left, at panel height, each story shows its representative PDB structure as a **transparent, slowly turning structure** (hemoglobin 2HHB, the PSII complex 3WU2, DnaK 2KHO, the 1918 haemagglutinin 1RUZ, human PrP 1QLX, bovine F1-ATPase 1BMF, spinach RuBisCO 8RUC, RecA–ssDNA 3CMW, pig insulin 4INS, ω-conotoxin MVIIA 1OMG) — a minimalist matte "clay" molecular surface of the polymer alone, each chain a pastel shade of the story's own highlight colour, soft three-light studio lighting with shadows and ambient occlusion, no cartoon or ligands, standing on its longest principal axis and spinning about it in the same sense as the map's turntable. PyMOL computes the molecular surface once (one OBJ mesh per chain, cached) and a GPU offscreen renderer (`_clay_renderer.py`, moderngl: G-buffer, screen-space ambient occlusion, 2x supersampling) shades the frames in about a millisecond each — PyMOL's own ray tracer was the first implementation and took five and a half hours for the ten structures — streaming them into a VP9 WebM encoded as a stacked alpha matte — the colour over a grey matte of its alpha, one opaque frame twice as tall, which the viewer recombines in a shader so the clip is transparent in every browser including the exported app's WKWebView (a VP9 alpha plane was the first encoding; Safari decodes it and drops the alpha) — 900 frames, 0.4° per frame, one turn every 30 s at 30 fps, plus a transparent PNG poster, cached under `~/.cache/luxar/pdb_turntables/`, and placed with `Scene.add_video(alpha_matte="stacked")` so the clip pauses whenever its story is not showing.

**Run**: `luxar demo run esm3_protein_stories [-- --no-serve] [-- --no-auto-rotate] [-- --no-turntables] [-- --no-audio]`

**Requires**: the base demo's cache (`~/.cache/luxar/esm3_swissprot/`: UMAP positions + metadata). Run `luxar demo run esm3_protein_landscape` once to build it; this demo never recomputes embeddings. The turntables need **PyMOL open-source** and **ffmpeg** — a deliberate special case in the demo dependency policy, because PyMOL is not on PyPI: `brew install pymol ffmpeg` on macOS or `conda install -c conda-forge pymol-open-source ffmpeg` — plus `moderngl` from the `demos` extra and a machine that can open an OpenGL context (a headless Linux box without EGL cannot). Without any of them the demo prints the install hint and builds the scene with no turntables; the first render of all ten structures takes a few minutes (dominated by PyMOL's surface computation and the VP9 encode) and is cached thereafter.
It is also the reference scene for the **sound layer** (`docs/guides/specs/SOUND_SPEC.md`): a CC0 ambient pad (The Cynic Project's "Calm Ambient 1" from OpenGameArt, cached by checksum) plays under the whole tour on the `ambient` bus — re-encoded at build time as a first-order **ambisonic field** (`ambisonic="foa"`, left channel at +50°, right at −50°) that stays fixed to the world as the turntable spins, with its last 15 s blended into its first so the loop has no pop (`loop_crossfade`); kept as the original stereo MP3 only when the box has no decoder/encoder; each story slot carries an `on_arrive` narration on the `voice` bus — the story's authored spoken script (`Story.narration`, shorter and punchier than the panel; `OVERVIEW_NARRATION` for the overview) read when the flight lands, synthesised at BUILD time by `_narration.py` (OpenAI TTS when `OPENAI_API_KEY` is set, the macOS system voice otherwise, a warning and no narration on a box with neither) and cached under `~/.cache/luxar/esm3_protein_stories/narration/` by `(engine, voice, text)`; and each cluster hums — a spatial `effects` source `attach_to` the story's highlight node, pitched up a pentatonic ladder per story, with its falloff tuned to the story's waypoint camera, synthesised by `_audio_synth.py` (needs `afconvert` or `ffmpeg`) and cached under `.../hums/`. `--no-audio` builds the scene without the layer. For narration: `OPENAI_API_KEY` or macOS (`say` + `afconvert`); `LUXAR_NARRATION_ENGINE=none` forces a silent build; `LUXAR_AUDIO_ENCODER=none` skips the hums and the ambisonic bed.

**Demonstrates**: story waypoints (`ViewerConfig.waypoints`, the remote-control spec §4.1), dimension-aware overlays, physical mesh materials, scene-captured environments, data refraction with `refract_data`, `extend_to_all` backdrop + per-slot highlight layers, keep-orientation flights under auto-rotate, `scene.add_sound` (ambisonic bed, `on_arrive` narration, `attach_to` cluster hums, `ViewerConfig.audio`).

---

#### demo_cellxgene_census_umap.py - CZ CELLxGENE Census single-cell 3D UMAP (LOD stress test)
A very large 3D UMAP of human single cells from the CZ CELLxGENE Census, embedded from their **precomputed scVI latent** (50-d) with **cuML UMAP**, rendered with bounded additive Points streaming + a categorical `coloring` dimension (cell type / tissue / disease). The shipped default builds a 3M-element scene from a 1M-cell cache; larger caches are supported up to the portable 5,591,040 resident-Point node cap.

**Run**: `luxar demo run cellxgene_census_umap [-- --no-serve]`

**Requires**: nothing extra for the default; the checksum-verified dataset manifest fetches the 1M-cell coords cache (~12 MB) from the published record. Regenerating the coordinates at scale needs a CUDA GPU with `cellxgene-census` + `cuml` (RAPIDS) — see `scripts/gen_census_umap.py` (≈96.6M primary human cells available; ~140s scVI fetch + ~14min cuML UMAP for 10M). Point the demo at another cache with `CENSUS_UMAP_CACHE` and bound the resident cloud with `CENSUS_UMAP_MAX_CELLS` (maximum portable value 5,591,040).

**Demonstrates**: precomputed scVI single-cell embeddings, cuML UMAP at scale, categorical-dimension colour switching, and bounded additive Points streaming.

---

#### demo_cytoself_protein_landscape.py - CytoSelf Protein Localization 3D UMAP
~114k per-image CytoSelf embeddings from the OpenCell dataset as a 3D UMAP point cloud. Each point is a single fluorescence microscopy crop of an endogenously tagged protein, colored by subcellular localization or protein identity.

**Run**: `luxar demo run cytoself_protein_landscape [-- --no-serve] [-- --recompute] [-- --without-images]`

`--recompute` rebuilds both the 3D UMAP and the hover-thumbnail bundle; `--without-images` skips the image download entirely and the hover tooltip becomes text-only.

**Requires**: Internet access (downloads embeddings from Google Drive), `umap-learn`, `pandas`, `requests`. First run computes 3D UMAP (~10-30 min); subsequent runs load cached results. With hover thumbnails on (the default) the download is 185.8 GB — 4.23 GB of embeddings, 71 MB of label CSVs, and ten `Image_data` files of 11.3-23.6 GB each — so budget ~190 GB of free disk, since the downloads stay cached and the encoded thumbnails sit alongside them, and a 32 GB machine, since the thumbnail pass reads one `Image_data` archive whole (the largest is 23.6 GB) with the selected crops, the thumbnails encoded so far and the label tables live on top of it; `--without-images` keeps it to ~4.24 GB and ~16 GB of RAM. Downloads are staged and verified before they take their cache name and thumbnails are cached per source file, so an interrupted run resumes instead of starting over.

**Demonstrates**: Self-supervised image embeddings (CytoSelf VQ-VAE-2, 9,216-dim), subcellular localization landscape, ~1,311 OpenCell proteins, categorical attribute switching (localization vs protein), UMAP dimensionality reduction.

---

#### demo_zebrahub_multiome.py - Zebrahub Integrated Cells 3D UMAP
Visualizes integrated single cells from zebrafish with categorical attribute navigation between Cell Type and Timepoint views.

**Run**: `luxar demo run zebrahub_multiome`

**Requires**: Internet access (downloads the integrated 3D-UMAP parquet from the CZ Biohub public store). Substitutive Points LOD needs `luxar[gsplats]` (torch + scipy); without it the scene builds as a flat, fully viewable point cloud.

**Demonstrates**: Remote data loading, 3D UMAP embedding of single-cell data, categorical dimension navigation (cell type vs timepoint), real scientific dataset from Zebrahub.

---

#### demo_zebrahub_velocity_streamlines.py - Zebrahub 3D RNA-Velocity UMAP + Streamlines
Turns the Zebrahub VeloCyto AnnData (spliced/unspliced counts + precomputed 3D RNA-velocity UMAP embedding) into a luminous scene: cells as soft Points (colored by anatomy ontology) and RK4-integrated streamlines as Lines tracing the velocity field through UMAP space.

**Run**: `luxar demo run zebrahub_velocity_streamlines [-- --preset preview] [-- --no-serve] [-- --h5ad /path/to/zebrahub_velocity.h5ad]`

**Requires**: Internet access on first run (auto-downloads the `.h5ad` from the shared Zebrahub Google Drive into `~/.cache/luxar/zebrahub_velocity/`), `anndata>=0.10`, `h5py`, `gdown`, `scipy`. (anndata used to be capped at `<0.13`, because 0.13 requires `zarr>=3.1` and Luxar was pinned to zarr 2; that ceiling is gone now the project is on zarr 3.)

**Demonstrates**: RNA-velocity visualization (Points + Lines together), per-cell velocity binned into a regularized smoothed cubic vector field, vectorized RK4 streamline integration through UMAP space, categorical anatomy-ontology coloring, stratified streamline seeding.

---

#### demo_human_multiome_peak_umap.py - Human Multiome Peak 3D UMAP
3D UMAP embedding of ~1M single-cell ATAC-seq peaks, color-coded by cell type, lineage, timepoint, and other attributes.

**Run**: `luxar demo run human_multiome_peak_umap`

**Requires**: The ~34 MB coords parquet is resolved through the dataset manifest (cache → in-repo Git LFS → Zenodo); no manual data file is needed. Needs `pandas` (installed via `luxar[demos]`).

**Demonstrates**: Large-scale single-cell visualization (~1M points), multiple categorical attributes (cell type, lineage, timepoint, peak type, chromosome), ATAC-seq chromatin accessibility data.

---

#### demo_mouse_multiome_peak_umap.py - Mouse Multiome Peak 3D UMAP
3D UMAP embedding of ~192k single-cell ATAC-seq peaks from mouse embryonic development (E7.5-E8.75), color-coded by cell type, lineage, and timepoint.

**Run**: `luxar demo run mouse_multiome_peak_umap`

**Requires**: The ~7 MB coords parquet is resolved through the dataset manifest (cache → in-repo Git LFS → Zenodo); no manual data file is needed. Needs `pandas` (installed via `luxar[demos]`).

**Demonstrates**: Single-cell ATAC-seq visualization, embryonic developmental timepoints, multiple categorical attribute navigation, lineage-based coloring.

---

#### demo_zebrahub_multiome_peak_umap.py - Zebrahub Multiome Peak 3D UMAP
3D UMAP of 640k single-cell chromatin accessibility peaks from the Zebrahub project, with 30 cell types across 6 developmental stages.

**Run**: `luxar demo run zebrahub_multiome_peak_umap`

**Requires**: Internet access (downloads from CZ Biohub public zarr store).

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

#### demo_bird_plumage_colorspace.py - Bird Plumage Colour Space
Every reading in BirdColorBase (360,432 spectrophotometer measurements of plumage across 2,632 species, 300–700 nm) drawn as a spike in CIELAB: direction is hue, length is chroma, height is lightness, each painted its own colour. The non-displayed 4th dimension is **UV chroma** (R300–400 / R300–700) in deciles — the ultraviolet birds see and the CIE observer cannot, so two patches on the same spike can differ in UV as much as red differs from green. An `extend_to_all` ghost of the whole corpus stays behind every stop.

**Run**: `luxar demo run bird_plumage_colorspace [-- --no-serve] [-- --recompute]`

**Requires**: Internet access (~525 MB of `.xlsx` from the pinned BirdColorBase commit, cached). No optional dependency: the workbooks are streamed with the standard library and the CIE 1931 observer comes from the Wyman–Sloan–Shirley Gaussian fit.

**Demonstrates**: Reflectance-to-CIELAB colorimetry with no data file, a scientifically motivated non-displayed dimension (a measured axis human vision drops rather than time or channel), `extend_to_all` context layer against a sliced highlight layer, `line_type="segments"` spike geometry, categorical dimension labels, per-element Wikipedia links, block-wise reduction of a corpus too large to hold.

---

#### demo_chromatrace_choir_umap_sequence.py - Chromatrace CHOIR UMAP — Cell-Type Walkthrough
Variant of the Chromatrace demo with an 89-step slider stepping through each fine-grained CHOIR bio-term one at a time. Within each bio-group, cell types are reordered (nearest-neighbor + 2-opt TSP on UMAP centroids) so consecutive slots are spatially adjacent. Backdrop layer keeps the full UMAP outline visible across all slots; highlight layer shows only the active type in its CHOIR palette color.

**Run**: `luxar demo run chromatrace_choir_umap_sequence [-- --no-tsp] [-- --data /path/to/zip-or-folder]`

**Requires**: Same `choir_umap_3d_viewer.zip` bundle as `demo_chromatrace_choir_umap.py`.

**Demonstrates**: `extend_to_all` for shared backdrop across slider slots (60k points instead of 89×60k), per-slot `visible_range` overlays, two-layer toggleable scene (Backdrop + Highlight), explicit `CameraConfig` to bypass auto-fit on a high-cardinality non-spatial dim, TSP-optimized intra-group ordering for smooth cross-slot transitions.

---

### Data-Driven Demos (External Datasets)

#### demo_global_rivers_earth.py - Rivers of Earth
A relief-displaced textured mesh globe plus every HydroRIVERS reach (Lines) in geographic 3D.

**Run**: `luxar demo run global_rivers_earth`

**Demonstrates**: Mixed Mesh+Lines geometry, tiled high-resolution textures, geographic (lat/lon/elevation) coordinate mapping, and large real-world datasets with local caching (~1 GB download on first run).

---

#### demo_ocean_currents_earth.py - Ocean Currents of Earth
HYCOM surface-current streamlines (220k connected ribbons, coloured by speed) draped over a textured NASA Blue Marble mesh globe — a "Perpetual Ocean"-style visualization of the Gulf Stream, Kuroshio, and Antarctic Circumpolar Current.

**Run**: `luxar demo run ocean_currents_earth`

**Requires**: Internet access on first run (HYCOM GLBy0.08 surface u/v under `~/.cache/luxar/ocean_currents_earth/`, plus the shared NASA Blue Marble imagery under `~/.cache/luxar/blue_marble/` — ~28 MB, downloaded once and reused by all four Earth demos).

**Demonstrates**: Mixed Mesh+Lines geometry, fixed-arc-length RK4 streamline advection with along-segment land masking, per-vertex RGBA comet-tail fading, tiled globe textures, and a **partition-of-LOD current layer** — 16 per-tile `kind=lod` ladders keep the opening whole-globe view from retaining every ribbon. Coarse levels preserve whole streamlines and widen them linearly so the field's apparent ink stays stable across switches; `LOD_COMPRESSION` / `LOD_LEVELS` tune the residency-versus-disk trade.

---

#### demo_gaia_milky_way_3m.py - Milky Way Stars (Gaia DR3, 3M Stars)
Real Milky Way stars from Gaia DR3: top 3M brightest stars with real photometric colors (BP-RP index), galactocentric coordinates, and reference markers (Sun, Betelgeuse, Rigel).

**Run**: `luxar demo run gaia_milky_way`

**Requires**: The Gaia catalog is CC BY-NC, so it is not distributed with Luxar. Run `luxar demo deps --install`, then opt into the cached local build with `luxar demo run gaia_milky_way -- --build-catalog`; an interactive first run also offers the build. The ESA query and CPU transform take about 90 minutes for 3M stars, retain the completed raw table for resume, and write `~/.cache/luxar/milky_way_gaia_3m/milky_way_gaia_3m.zarr.zip`. Use `--recompute` to replace a cached copy. Substitutive Points LOD needs `luxar[gsplats]` (torch + scipy); without it the scene builds as a flat, fully viewable point cloud.

**Demonstrates**: Real astronomical data (Gaia space telescope), 3M star dataset, BP-RP photometric color-to-RGB conversion, galactocentric coordinate system, magnitude-dependent point radii, volumetric emission-absorption compositing on a mixed substitutive ladder, per-marker hover labels and a colour-swatch HTML legend built from the marker nodes.

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

**Requires**: Internet access on first run (downloads ~26 MB from the public `manlius/laniakea` GitHub data cache; reused from `~/.cache/luxar/laniakea/`). The full preset reproduces the reported 29,555 valid streamlines and writes a large line dataset; `--preset preview` is faster for iteration. The streaming ladder needs no optional dependencies — unlike substitutive coarsening, an additive prefix imports neither torch nor scipy, so there is no degraded no-LOD path.

**Demonstrates**: Real astronomical catalogs, supergalactic coordinates, basin-of-attraction coloring, vectorized RK4 streamline integration, a per-basin streaming ladder over large **indexed** Lines layers (the ladder verifies each connected component is an ascending simple path, so the ribbons survive intact rather than being coarsened into synthesised gsplats), HDR additive rendering, layer toggles per basin. Data: Cosmicflows-4 / EDD and the open [`manlius/laniakea`](https://github.com/manlius/laniakea) pipeline.

---

#### demo_desi_galaxies.py - DESI DR1: The Cosmic Web in 3D (~9.75M-object catalog)
The large-scale structure of the Universe from the full ~9.75M-object Dark Energy Spectroscopic Instrument first data release catalog. Each point is a real galaxy or quasar with a measured spectroscopic redshift; the redshift becomes a comoving distance so sky position + depth give true 3D Cartesian coordinates in megaparsecs. You sit at the observer's origin looking out at the two DESI footprint caps fanning into filaments, voids, and the baryon-acoustic shells. Two colorings toggle in the Layers panel: **by tracer** (BGS/LRG/ELG/QSO populations, naturally layered by distance) and **by redshift** (continuous depth colormap).

**Run**: `luxar demo run desi_galaxies [-- --recompute]`

**Requires**: Nothing extra by default — the checksum-verified dataset manifest fetches a compact precomputed point cloud (quantized XYZ + redshift + tracer id, ~73 MB) from the published record. With `--recompute` it auto-downloads the ~1 GB of DR1 LSS clustering catalogs to `~/.cache/luxar/desi_galaxies/` (resumable), reads them with `astropy`, and converts (RA, Dec, z) → comoving Mpc. Adds `astropy` to the `demos` extra. The built scene (with substitutive LOD) is cached in the demos output dir, so only the first launch pays the LOD-build cost. If the DESI data host is unavailable, rerun without `--recompute` to use the manifest-driven precomputed scene; a record-fetch failure is reported separately. Substitutive Points LOD needs `luxar[gsplats]` (torch + scipy); without it the scene builds as a flat, fully viewable point cloud.

**Demonstrates**: Real spectroscopic-survey catalogs → a 3D cosmic-web Points cloud, `(RA, Dec, redshift)` → comoving-Mpc conversion via `astropy.cosmology` (DESI fiducial ΛCDM), a spatially partitioned finest LOD with bounded additive streaming inside every part, dual coloring (categorical tracer vs. continuous redshift colormap) via layer toggles, HDR additive rendering, self-contained download → convert → cache-processed bootstrap. Data: [DESI DR1](https://data.desi.lbl.gov/doc/releases/dr1/) (DESI Collaboration 2026, arXiv:2503.14745; CC BY 4.0).

---

#### demo_asteroids_solar_system.py - The Solar System (~1.5M Real Asteroids, JPL SBDB)
Every catalogued minor planet placed in real 3D space by propagating its measured Keplerian orbit to a common epoch: ~1.5M asteroids as Points colored by semi-major axis, plus the eight planets, the Sun, and the planets' orbit ellipses (Lines). The main belt, Kirkwood gaps, Hilda triangle, and Jupiter Trojan clouds all emerge from the real orbital-element distribution. The asteroid `intensity` is reserved for the colormap's `[0, ~11 AU]` scalar window, while density-compensated opacity keeps aggregate additive weight stable across full, `--max-asteroids`, and animated builds. The opening camera orbits the Sun and frames the outer planets instead of fitting sparse distant-object outliers.

**Run**: `luxar demo run asteroids_solar_system [-- --animate] [-- --max-asteroids N]`

**Requires**: Internet access on first run (downloads the JPL SBDB catalog, ~140 MB, to `~/.cache/luxar/asteroids/`; the parsed catalog is then cached as `.npz`, so subsequent runs are offline).

**Demonstrates**: Real planetary-science data (NASA/JPL Small-Body Database), on-the-fly orbital mechanics (vectorized Kepler solve, element→Cartesian, mean-anomaly propagation — no astropy), ~1.5M-point cloud with `scalars`+colormap coloring and per-point hover labels, Points + Lines (planet orbits) in one scene, optional time-dimension animation (`--animate`) advancing every body along its orbit. Data: [JPL SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html) (public domain).

---

#### demo_dipc_3d_genome.py - Single-Cell 3D Genome (Dip-C)
The folded 3D structure of one human cell's genome from Dip-C (Tan et al. 2018, Science): the chromosomes coil through the nucleus as Lines, colored by chromosome. A non-displayed haplotype dimension isolates either maternal or paternal copy, while a faint always-visible context layer keeps the full diploid nucleus in view.

**Run**: `luxar demo run dipc_3d_genome [-- --recompute]`

**Requires**: Nothing extra by default — ships a small precomputed structure via Git LFS. With `--recompute` (or if the LFS asset isn't pulled) it auto-downloads the GEO archive (`GSE117876_RAW.tar`, ~4.7 GB) to `~/.cache/luxar/dipc_genome/`, extracts one cell's `.3dg`, and caches the processed `.npz`.

**Demonstrates**: 3D-genomics visualization (nothing else renders single-cell genome folding interactively), one indexed Lines node sliced by a non-displayed haplotype dimension, a second context Lines layer broadcast across every haplotype slice, per-vertex color and hover labels, self-contained download → extract → cache-processed bootstrap. Data: [Tan et al. 2018](https://doi.org/10.1126/science.aat5641), GEO GSE117876; [dip-c format](https://github.com/tanlongzhi/dip-c).

---

#### demo_dmri_tractography.py - Human White-Matter Tractography (HCP-1065)
The wiring of the human brain: all 87 named white-matter tracts of the HCP-1065 population atlas as 252,978 streamlines (7.08M vertices, 6.83M segments) in ICBM 2009a space, coloured by the standard diffusion-MRI direction convention (red = left-right, green = anterior-posterior, blue = inferior-superior). Each tract is its own Lines node exposed as a **Layers-panel** toggle (press **L**), so any bundle can be isolated — the corticospinal tract, the corpus callosum, the arcuate. Hover a zoomed-in tract for its full anatomical name and a one-line gloss of what it does, expanded from the terse atlas code.

**Run**: `luxar demo run dmri_tractography [-- --per-bundle 6000 --points 28]`

**Requires**: Internet access on first run — downloads `hcp1065_avg_tracts_trk.zip` (588 MB) to `~/.cache/luxar/dmri_tractography/` and caches the decoded bundles as an `.npz`. Needs `nibabel` (in the `demos` extra). No GPU. Building the scene takes ~25 min because every node lifts its segments to Gaussian beads for the substitutive LOD; the cached scene rebuilds on `--recompute` or when the demo builder fingerprint changes, unless `--keep-stale` is passed. Substitutive Lines LOD needs `luxar[gsplats]` (torch + scipy); without it the bundles are written flat (fully viewable, no coarse levels).

**Demonstrates**: Lines as the *native* geometry for data that is already made of curves — no conversion, unlike every volumetric demo. Arc-length resampling (the source is ~0.4 mm-sampled, ~10x finer than any rendered line width); `line_type="indexed"` with per-streamline contiguous vertices so thick tubes render seamless joints; 87 separate nodes, each sized to stay under both the un-laddered-leaf gate (200K vertices) and the per-node element-texture segment bound; per-node **substitutive LOD** at `compression_factor=256` — thin lines need a far larger K than the default, because the bead lift is driven by arc-length ÷ width rather than by segment count (at K=4 the "coarse" level comes out 5.6x heavier than the fine one, and the scene balloons to 2.1 GB); `blending_mode="additive"` at low opacity with a display window, which is order-independent and so cannot pop as the camera orbits — unlike `normal`, whose per-object transparent-pass sort flips between overlapping bundles; and per-tract **hover labels** broadcast across a node's vertices (Lines labels are per-vertex), which ride the ladder's finest level only, so the tooltip appears once a bundle is zoomed to roughly fill the view. Data: [Yeh 2022](https://doi.org/10.1038/s41467-022-32595-4), [HCP-1065 atlas](https://brain.labsolver.org/hcp_trk_atlas.html) (CC BY-SA 4.0; WU-Minn HCP data-use terms).

---

#### demo_storm_3d_microtubules.py - 3D STORM Super-Resolution Microscopy
Microtubule cytoskeleton at nanometer resolution using real STORM super-resolution microscopy localizations as Gaussian splats.

**Run**: `luxar demo run storm_3d_microtubules`

**Requires**: Internet access on first run (downloads the size-verified STORM localization CSV) and an NVIDIA CUDA GPU for an uncached widefield fit.

**Demonstrates**: Two Gaussian-splat provenance classes in one categorical view: a compact basis fitted to a photon-weighted, diffraction-blurred widefield volume, and measured STORM localization ellipsoids whose per-axis σ combines CRLB with 17 nm antibody-linkage uncertainty (about 18-19 nm and linkage-dominated in this dataset). The default 5M-row frame-ordered prefix is used for both views, so its widefield image is explicitly a partial acquisition.

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

**Run**: `luxar demo run caida_as_topology [-- --max-edges=80000 --keep-snapshots=4 --refresh-snapshots --recompute-pipeline --recompute-layout]`

**Requires**: Internet access on the first run (auto-downloads ~6 MB of compressed snapshots from CAIDA, tens of MB decompressed), `networkx>=3.0`, `umap-learn`, `scipy`, `pandas`, `requests`. Later runs make at most one discovery check a week and none at all while the memo is fresh, so a warm cache runs offline: `~/.cache/luxar/caida/` (or `--cache-dir`) holds which snapshot pair is current, the derived pipeline bundle (parse + LCC + tier-1 + Louvain) and the 3D layout. A new monthly CAIDA release triggers one download and recomputes both derived artifacts (the layout is the slow one, ~2-3 min); superseded snapshots are pruned to the newest `--keep-snapshots N` (default 2).

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

#### demo_gsplats_3d_blastocyst_dapi_nuclei.py - Mouse Blastocyst DAPI Nuclei
Gaussian splatting compression of real 3D confocal microscopy data (DAPI-stained cell nuclei) from Image Data Resource (IDR).

**Run**: `luxar demo run gsplats_3d_blastocyst_dapi_nuclei`

**Requires**: Internet access (downloads OME-ZARR from IDR). Supports Metal (MPS) acceleration on Apple Silicon.

**Demonstrates**: 3D Gaussian splat fitting to real microscopy volumes, 20-50x compression vs raw voxels, oriented ellipsoids capturing elongated nuclear shapes, IDR/OME-ZARR data loading, `plasma` colormap + ACES tone-mapping.

---

#### demo_gsplats_3d_blastocyst_multichannel.py - Multi-Channel Mouse Blastocyst (IDR Pipeline)
Multi-channel 3D microscopy data as Gaussian splats, with full compute pipeline from Image Data Resource (IDR). Uses checksum-verified precomputed gsplats from the published record by default.

**Run**: `luxar demo run gsplats_3d_blastocyst_multichannel [-- --recompute]`

**Requires**: Nothing extra by default; the dataset manifest fetches the precomputed gsplats (~0.4 MB). Internet access + a GPU are needed with `--recompute`.

**Demonstrates**: Multi-channel Gaussian splatting with distinct colors per channel, additive blending so the two superimposed channels mix instead of occluding each other, full pipeline (fetch from IDR, fit per channel, merge), precomputed gsplats from the published record for a fast default path.

---

#### demo_gsplats_3d_cells3d_multichannel.py - 3D Multi-Channel Cells (Layers + BOP LUTs)
Two-channel scikit-image `cells3d` fluorescence volume (membranes + nuclei) fitted per channel as 3D splats and shown as toggleable **layers**, each coloured by a BOP (Blue-Orange-Purple) microscopy LUT (`bop_orange` membranes, `bop_blue` nuclei). Fully self-contained (skimage downloads the sample) — the cheapest gsplat demo to run from scratch. Uses checksum-verified precomputed gsplats from the published record by default.

**Run**: `luxar demo run gsplats_3d_cells3d_multichannel [-- --recompute]`

**Requires**: Nothing extra by default; the dataset manifest fetches the precomputed gsplats (~1 MB). `scikit-image` + a GPU are needed with `--recompute`.

**Demonstrates**: Per-channel `layer=True` gsplats nodes with built-in BOP LUTs applied at display time (interactive colormap switching in the Layers panel, press L), shared amplitude-weighted centroid alignment, additive compositing for the two co-located channels, ACES tone-mapping. The lightweight, no-download sibling of `blastocyst_multichannel` and `kidney_multichannel_layers`. Its **mesh** counterpart on the same data is `mesh_isosurface_cells3d` — run both to compare the two representations side by side.

---

#### demo_mesh_isosurface_cells3d.py - Cells3D Isosurfaces (the reference Mesh demo)
Marching-cubes **isosurfaces** of the same two-channel scikit-image `cells3d` volume — membranes and nuclei — as two shaded, toggleable mesh layers. Isosurfaces and segmentation boundaries are the named target data for the Mesh geometry type (`docs/specs/MESH_NODE_SPEC.md` §1): routine outputs of the pipelines Luxar already serves, which before Mesh could only be approximated by a dense point cloud.

Deliberately the same dataset as the gsplat demo above, because the pairing is the lesson: splats approximate the whole intensity field and need no threshold, while an isosurface picks ONE level set and renders it as an opaque surface with real occlusion and silhouettes. Neither is the better answer — they answer different questions.

**Run**: `luxar demo run mesh_isosurface_cells3d`

**Requires**: `scikit-image` + `scipy` (both in the `demos` extra). **No GPU and no fitting step** — marching cubes is CPU-only and takes about two seconds, which makes this the cheapest end-to-end demo of any Luxar geometry type.

**Demonstrates**: Mesh as the only **shaded** geometry type — per-vertex marching-cubes gradient normals written with an explicit `normal_dims`, lit by the §6.2 view-anchored offset key, so nuclei inside membranes are genuinely occluded rather than summed. `opaque` blending by default (unlike the other three types' `additive`), the five mesh-only **Ambient** / **Shade falloff** / **Specular** / **Shininess** / **Alpha cutoff** Layers-panel sliders, per-channel `layer=True` toggling, physical units via marching_cubes' `spacing` (the dataset's voxels are mildly anisotropic — 0.29 µm in Z vs 0.26 µm in-plane, ~1.1x), and scale: ~537K vertices / 1.07M triangles across the two surfaces.

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

**Demonstrates**: Large-volume Gaussian splatting, isotropic light-sheet microscopy, Zenodo/Cell Tracking Challenge data, Zeiss LightSheet Z.1 data, `volumetric` emission-absorption blending with strong absorption (κ=3.13) and low opacity (0.06) so the embryo reads as dense tissue without the diffuse background overwhelming it, plus a 0–1.085 display window tuned for the direct-colour layer. Also demonstrates measuring a **specimen** background rather than trusting the default floor: this stack has two levels (a ~204-count detector offset outside the embryo and its own ~675-count autofluorescence inside), and subtracting only the first left ~84% of the emitted mass as haze.

---

#### demo_gsplats_3d_milky_way_dust.py - 3D Interstellar Dust of the Solar Neighborhood (Leike et al. 2020)
Gaussian-splats a real 3D reconstruction of the Milky Way's interstellar dust around the Sun — the splat pipeline applied to astrophysics rather than microscopy. The "volume" is a cube of *space*: a 740 × 740 × 540 pc reconstruction of dust extinction density at 1 pc resolution, fit into glowing 3D fog revealing the Local Bubble and the Orion / Taurus / Perseus molecular clouds.

**Run**: `luxar demo run gsplats_3d_milky_way_dust [-- --recompute]`

**Requires**: Nothing extra by default — the checksum-verified dataset manifest fetches a precomputed **full-resolution** fit from the published record (~10 MB: the native 740×740×540 cube fit to ~675k splats, PSNR ~35 dB). With `--recompute` it auto-downloads the 2.4 GB reconstruction (`mean_std.h5`) to `~/.cache/luxar/gsplats_milkyway_dust/` (resumable) and refits. The `--recompute` default reproduces the record's full-res fit and needs a large-VRAM GPU (~40 GB+); on a smaller card pass `--target-size 256 --max-splats 200000` for a lighter downscaled refit.

**Demonstrates**: Real *volumetric astronomy* → Gaussian splats (the same `cal → fit → convert` pipeline used for microscopy, on a dust-density cube), 20–50× compression, `inferno` colormap + ACES tone-mapping + light volumetric HDR rendering (absorption κ=0.3, so near dust softly occludes far dust), self-contained download → fit → cache-processed bootstrap. Data: [Leike, Glatzle & Enßlin 2020](https://doi.org/10.1051/0004-6361/202038169), A&A 639, A138 (Zenodo record 3993082, CC BY 4.0).

---

#### demo_gsplats_3d_visible_human_head.py - Visible Human Head (Real-Color Anatomy)
The human head Gaussian-splatted in **true photographic color** from the NLM Visible Human Project cryosections — actual photographs of a frozen cadaver sliced at 1 mm, so brain, skull, muscle and vasculature appear in natural anatomical color (not a false-color transfer function). The splat pipeline applied to real photographic volumetric anatomy.

**Run**: `luxar demo run gsplats_3d_visible_human_head [-- --recompute]`

**Requires**: nothing extra by default — the checksum-verified dataset manifest fetches a 1,908,888-splat fit plus its per-splat colors sidecar from the published record (~30 MB together). The fit carries colors natively; the sidecar remains a fallback for older colorless fits. With `--recompute` it auto-downloads the 377 color head slices (~1.1 GB) to `~/.cache/luxar/gsplats_visible_human_head/`, builds the masked RGB volume, fits luminance (GPU strongly preferred; CPU works but is slow), and samples per-splat colors from the stored splat order.

**Demonstrates**: True-color volumetric anatomy → Gaussian splats via a **single luminance fit + per-splat color sampling** (one fit, real photographic color — vs. the scalar-intensity-plus-colormap microscopy demos), warm-vs-blue tissue masking to drop the frozen-gel background, ACES tone-mapping, self-contained download → mask → fit → cache-processed bootstrap. Data: [NLM Visible Human Project](https://www.nlm.nih.gov/research/visible/visible_human.html) (Male color cryosections, head subset; public domain).

---

#### demo_gsplats_3d_cryoem_virus.py - Cryo-EM Giant Virus Capsid (Structural Biology)
Gaussian-splats a real cryo-electron-microscopy density map from the EMDB: the icosahedral capsid of *Paramecium bursaria* chlorella virus 1 (PBCV-1), a giant virus. The reconstructed electron-density volume — a hollow ~1650 Å shell tiled with capsomers — is exactly what the Luxar splat fitter eats, so this is the microscopy splat pipeline applied to structural biology (no isosurface threshold needed).

**Run**: `luxar demo run gsplats_3d_cryoem_virus [-- --recompute]`

**Requires**: Nothing extra by default — the checksum-verified dataset manifest fetches a precomputed fit from the published record (~11 MB: the 700³ EMDB map downsampled to 512³, fit to ~1.0M splats, PSNR ~28 dB). With `--recompute` it auto-downloads the 1.3 GB EMDB map (`emd_5384.map.gz`) to `~/.cache/luxar/gsplats_cryoem_virus/` (resumable), reads it with `mrcfile`, downsamples to 512³, and fits Gaussian splats on the GPU. Adds `mrcfile` to the `demos` extra.

**Demonstrates**: Real *structural-biology* electron density → Gaussian splats (the same `cal → fit → convert` pipeline used for microscopy, on an EMDB MRC/CCP4 map), solvent clipping + percentile normalization, `inferno` colormap (starts at black, so empty space stays black) + ACES tone-mapping + volumetric HDR rendering (absorption κ=5, so the near shell occludes the far one), self-contained download → read → fit → cache-processed bootstrap. Data: [EMDB EMD-5384](https://www.ebi.ac.uk/emdb/EMD-5384) (Zhang et al. 2011, PNAS 108(36):14837; public domain / CC0).

---

#### demo_gsplats_3d_ct_totalsegmentator.py - CT Anatomical Atlas (neck-to-pelvis, Organs in Color)
Gaussian-splats a real clinical CT scan — a neck-to-pelvis study (the fullest coverage routine CT offers; the head and distal limbs are outside the scan) — and colors every splat by the anatomical structure it belongs to, using the TotalSegmentator dataset's 117-organ segmentation. The result is a glowing, rotatable atlas — white skeleton, red great vessels, cyan lungs, and colored abdominal organs in their true 3D positions. The same **one-fit + per-splat label sampling** idea as the Visible Human head demo, but the sampled organ label drives the color (a tissue-grouped palette), a **hover tooltip** (the specific structure name — all 117 tissue types), and a split into **toggle-able Layers-panel groups** (Skeleton / Organs / Vessels & heart / Nervous system / Muscles, the last a faint boosted context) — the splat pipeline applied to clinical radiology.

**Run**: `luxar demo run gsplats_3d_ct_totalsegmentator [-- --recompute]`

**Requires**: Nothing extra by default — the checksum-verified dataset manifest fetches a precomputed fit plus its per-splat organ-label sidecar from the published record (~7 MB; the fit is a neck-to-pelvis subject at 1.5 mm, fitted to ~0.66M splats at PSNR ~43 dB). The fit carries labels natively; the sidecar remains a fallback for older label-less fits. With `--recompute` it auto-downloads the 3.2 GB TotalSegmentator subset to `~/.cache/luxar/gsplats_ct_totalsegmentator/` (resumable), extracts one subject, combines its 117 organ masks with `nibabel`, windows + fits on the GPU, and samples the per-splat organ label. Adds `nibabel` to the `demos` extra.

**Demonstrates**: Real *clinical CT* (neck-to-pelvis) + multi-organ segmentation → colored Gaussian splats, combining per-structure NIfTI masks into one label volume, Hounsfield windowing, per-splat organ-label sampling driving a tissue-grouped color palette + **per-splat hover tooltips** (specific structure names) + a split into **toggle-able tissue layers** (Skeleton/Organs/Vessels & heart/Nervous system/Muscles), cubic-voxel resampling, ACES tone-mapping + additive compositing for the co-located groups, self-contained download → combine → fit → cache-processed bootstrap. Data: [TotalSegmentator](https://zenodo.org/records/10047263) (Wasserthal et al. 2023, Radiology: AI; CC BY 4.0).

---

#### demo_gsplats_lod_tribolium.py - Adaptive Level of Detail on the Tribolium Embryo
Takes the precomputed Tribolium embryo fit and builds an **adaptive Level of Detail (LOD)** pyramid on it — the embryo is stored at several resolutions, and the viewer shows the simplest one that still looks right at the current zoom. This demo uses *substitutive* LOD (each coarser level *replaces* the finer one with fewer, larger splats), and ships it with per-level debug colors (green→amber→red, finest→coarsest) so the viewer's `coverage_fraction` level-switching is visible as you zoom. The scaled-up companion to `examples/gsplats_lod_example.py`.

**Run**: `luxar demo run gsplats_lod_tribolium`

**Requires**: A Tribolium fit in the local cache under `~/.cache/luxar/gsplats_tribolium`. A cold cache or `--recompute` re-fits from Zenodo (network + GPU).

**Demonstrates**: Substitutive LOD via `make_substitutive_lod` (`kmeans_lloyd`), `add_gsplats_from_data(lod_group=True)`, auto level-count from base splat count (#levels scales as log_K(N)), per-level debug coloring, `coverage_fraction` LOD selection in the viewer. Options: `--levels=N`, `--factor=K`, `--method=NAME`, `--serve-only`.

---

#### demo_gsplats_lod_embryo_line.py - Near-Unlimited Scaling with Adaptive Level of Detail
Lays out `--count` (default 100) copies of the single adaptive-detail Tribolium embryo along a straight line, drops the camera near the middle of the line, and lets you fly down it. **Level of Detail (LOD)** is the idea that makes this scale: each embryo is kept at several resolutions, and the viewer picks — per object, every frame, by viewport-relative `coverage_fraction` (how much of the screen it covers) — the simplest version that still looks right. Near embryos render fine (green) while distant ones collapse to a few big splats (red), so the detail actually drawn stays roughly bounded by what the screen can resolve, no matter how long the line. All copies share **byte-identical splat arrays** (per-embryo orientation/jitter lives only in scene-graph transforms), so the encoder's `array_ref` deduplication stores the geometry once: 100 embryos cost ~55 MB on disk instead of ~1.3 GB.

**Run**: `luxar demo run gsplats_lod_embryo_line`

**Requires**: A Tribolium fit in the local cache under `~/.cache/luxar/gsplats_tribolium`. A cold cache or `--recompute` re-fits from Zenodo (network + GPU).

**Demonstrates**: Per-object `coverage_fraction` LOD selection at scale, scene-graph transforms (`add_group(transform=...)`, `transforms.compose`/`rotate`/`translate`) for placement so splat arrays stay identical, automatic `array_ref` array deduplication in the encoder, a `layer=True` `embryo_line` container group (one Layers-panel row for the whole line rather than 100), and initial-camera setup via `ViewerConfig(camera=CameraConfig(...))`. Options: `--count=N`, `--levels=N`, `--factor=K`, `--method=NAME`, `--serve-only`.

---

#### demo_gsplats_recipes_tribolium.py - LOD `--recipe` gallery (flat / stream / levels / tiles / overview / adaptive)
Runs the unified `luxar gsplat lod --recipe` pipeline on the **one** precomputed Tribolium fit to build the six scale-ordered representation topologies and lays them out side by side for direct comparison: **flat** (one leaf) → **stream** (one leaf + a prefix-sum ladder) → **levels** (a `kind=lod` of *substitutive* levels — coarse↔fine replacement, one shown at a time) → **tiles** (a spatial BSP `kind=partition` where every part carries its own additive ladder) → **overview** (an *unbalanced-by-design* `kind=lod`: a cheap coarse substitutive cap for the far view, above a `tiles` fine branch for close-up — detail only where you look) → **adaptive** (a BSP `kind=partition` where every part is its own *substitutive* lod group — per-part coarse↔fine replacement, so each cell culls AND picks its own level by its own on-screen size). Structure is colour-coded so the topologies are legible: each BSP part gets a distinct colour, the overview coarse cap is red ("far") above cool-coloured fine parts ("near"), and the stream ladder runs blue→cyan coarse→fine. The reference demo for the `cal → fit → lod --recipe → convert → serve` workflow; each recipe build prints the equivalent CLI command.

**Run**: `luxar demo run gsplats_recipes_tribolium`

**Requires**: A Tribolium fit in the local cache under `~/.cache/luxar/gsplats_tribolium`. A cold cache or `--recompute` re-fits from Zenodo (network + GPU).

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

##### demo_gsplats_interop_inria_bonsai.py - Full-quality INRIA bonsai (HTTP-Range Zip64 extraction)
Extracts just the ~309 MB `bonsai/.../point_cloud.ply` member — 2% of the file — from INRIA's 14.7 GB `models.zip` over **HTTP Range requests**, then imports it with a tiled LOD. The archive is above 4 GiB, so locating any member at all goes through the **Zip64** end-of-central-directory path; this is the only demo that exercises it (the cluster-fly demo's ~1 GB archive takes the classic path). Opens framed on the bonsai tree itself, whose position is derived from the data — the blossoms are the only strongly pink splats in the room.
Previously pulled the *garden* member, which made it a near-duplicate of the Mip-NeRF demo: the antimatter15 `garden.splat` was produced from this very checkpoint, and the two stores came out at the same 5,834,784 splats with bit-identical amplitudes.
**Run**: `luxar demo run gsplats_interop_inria_bonsai`
**Requires**: Network (~309 MB range-extracted from a 14.7 GB archive; server must honor byte ranges — INRIA's does). No GPU.
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
Five hours of zebrafish gastrulation (Zenodo 1211599, confocal, 151 timepoints two minutes apart) as **one** 4D Gaussian-splat node: each timepoint is fitted separately, then the fits are stacked with `combine_as_new_dimension` so time is the fourth centre column rather than a per-timepoint sibling node. That single node carries a substitutive LOD ladder in which time is a **hard coarsening barrier**, so a coarse level never blends one frame's cells into the next. The labelled endodermal cells fill under 2% of the imaged voxels, so a second toggleable layer draws the acquisition volume as a **wireframe cage ruled every 100 µm** — without it the specimen floats in an unmarked void and its migration across the yolk cannot be read. The Time slider is in **minutes** on an exact 2-minute grid (the LSM records 120.01 s; the axis is rounded so its last stop is actually reachable). Each timepoint has its **connected components smaller than 4 voxels deleted before it is fitted**: a third of the early frames' energy is shot noise, and it is unusually literal — the median object in a frame is *one voxel*. A size filter is used rather than a smoothing one because it cannot damage what it keeps: measured against a stricter cell definition than the threshold itself, every threshold from 2 to 12 holds cell energy and mean cell peak at exactly 1.0000, whereas non-local means (tried first) dimmed cell peaks to 0.78 of raw. The record archive is 1,270,233 splats — a median of 7,730 per timepoint, ranging 1,312–18,783 as the specimen grows — in four substitutive levels, every one of which still carries all 151 timepoints. (No dB here on purpose: these fits are scored against their filtered input, so that figure is not fidelity to the microscope. The demo's module docstring records the method, including why the metric that originally chose NLM was measuring the wrong thing.)

**Run**: `luxar demo run gsplats_4d_zebrafish_timelapse`

**Requires**: Nothing extra by default — the checksum-verified dataset manifest fetches the fit from the published `cc-by-sa` record (~19 MB). `--recompute` downloads the ~2.1 GB LSM from Zenodo and refits all 151 timepoints (GPU strongly recommended).

**Demonstrates**: stacking per-timepoint 3D fits into one 4D node; barrier-aware substitutive LOD over a time axis; `extend_to_all` for a static reference layer that survives every scrub; a discrete viewer dimension carrying real physical units; and reading acquisition geometry out of a Zeiss LSM instead of assuming it.

---

#### demo_gsplats_4d_neuromast_2ch.py - 4D Two-Channel Zebrafish Neuromast Time-Lapse
100-timepoint iSIM recording of a developing zebrafish lateral-line **neuromast** (`she:GFP; cldnb:lyn-mScarlet`), shown as **two independently-toggleable Gaussian-splat layers**: membranes (mScarlet, `bop_blue` LUT) + nuclei (GFP, `bop_orange` LUT). Both channels are co-registered and share one 4D coordinate space, so they animate together — play the **Time** slider to scrub development, press **L** for the Layers panel to control each channel. The gsplats are pre-fit (per-channel calibration K*=64k → `batch-fit` → Z×2.5 anisotropy correction → redundancy cull), so no fitting/GPU is needed to view.

**Run**: `luxar demo run gsplats_4d_neuromast_2ch`

**Requires**: The two pre-fit channels are not bundled with the repo. The resolver fetches the 130 MB `.gsplats.zarr.zip` pair from the published `cc-by` record through `ensure_dataset` (SHA-256 verified, cached under `~/.cache/luxar/gsplats_4d_neuromast_2ch/`, expanded to a temporary directory on read), and falls back to an unzipped local pair under `$LUXAR_NEUROMAST_DATA_DIR` (default `~/luxar_demo_data/gsplats_neuromast_2ch/`) only when the manifest can build no download URL for the record. No GPU is needed.

**Demonstrates**: 4D + multi-channel gsplats, per-channel `layer=True` + named colormaps (`bop_blue`/`bop_orange`) for the Layers panel, `add_gsplats_from_file` grafting of pre-fit multi-LOD (`stream`, 8 LODs) nodes, Z-anisotropy correction baked via `transform --scale`, redundancy-based culling, and additive compositing because the two superimposed channels have no meaningful cross-layer order. Options: `--no-serve`, `--serve-only`.

---

#### demo_gsplats_4d_celegans_tracking.py - 4D C. elegans Nuclei Tracking with Lines
4D confocal time-series of a developing *C. elegans* embryo: Gaussian splats for volume rendering combined with polylines for tracked cell nuclei trajectories.

**Run**: `luxar demo run gsplats_4d_celegans_tracking`

**Requires**: Internet access (downloads ~26.1 GB from Zenodo), GPU recommended.

**Demonstrates**: Combined GSplats + Lines in 4D, cell lineage tracking as 4D polylines (X, Y, Z, Time), `extend_to_all` for persistent trajectory visibility, StarryNite tracking data, volume rendering + track overlay.

---

#### demo_gsplats_4d_nexrad_supercell.py - 4D NEXRAD El Reno Tornadic Supercell
82 WSR-88D weather-radar volume scans of the **2013-05-31 Oklahoma convective evening** — including the **El Reno supercell**, which produced the widest tornado ever recorded (4.2 km, EF3) — as a 4D Gaussian-splat timelapse spanning 21:00 UTC to 03:00 UTC. Scrubbing the **Time** slider carries you from a nearly empty sky through initiation, tornadogenesis (touchdown 23:03) and the tornado's end (23:43), into the overnight mesoscale system that flooded Oklahoma City. The opening frames already hold the separate convection to the north-east; the El Reno supercell itself appears around 22Z and then grows to dominate the domain. The 3D structure is the point: the **hook echo** curling around the low-level mesocyclone, the **bounded weak echo region** where the updraft is too violent for precipitation to form, and the **overshooting top** punching past 15 km. The domain spans 300 x 300 km, which retains 90% of the >=20 dBZ echo (a tighter storm-scale box kept only 75%). A **magenta vertical line** marks the surveyed tornado position on the frames when it was down — taken from the NWS ground damage survey (via the SPC tornado database), an independent source that lands right on the radar's hook echo. A faint **wireframe cube** outlines the analysis domain, which is what makes the 300 x 300 x 18 km slab legible as a volume rather than a floating cloud.

A radar does not sample a volume — it spins a 0.95° beam at 14 discrete elevations, so the raw data is a set of nested *cones* with gaps between them. The demo regrids them onto a Cartesian storm box with a Barnes-weighted kd-tree interpolation, masks every cell the beam could not actually reach, and fits each timepoint independently.

**Run**: `luxar demo run gsplats_4d_nexrad_supercell`

**Requires**: The default path downloads the fitted splats from the published record through the checksum-verified dataset manifest (~13 MB); it needs no GPU or radar decoder. `--recompute` downloads 800 MB of Level II volume scans from the NOAA archive and needs `metpy` plus a GPU.

**Demonstrates**: Weather/atmosphere as gsplats + lines, polar→Cartesian objective analysis (Barnes on a `cKDTree`, scipy only — no Py-ART), the 4/3-effective-earth beam-propagation model, geometric coverage masking instead of threshold-tuning, `combine_as_new_dimension` for a stacked 4D time axis (streaming ladder only — coarse substitutive levels average the merged amplitudes down and muddy the hail core, so they are deliberately not used), an *adaptive* splat budget (constant occupied-voxels-per-splat, since the system grows ~6x across the window), a *global* rather than per-frame intensity scale so the storm's intensification and decay survive, a baked `CameraConfig(up=(0,0,1))` because a geographic scene on the viewer's default up-vector renders altitude sideways, and a baked appearance (turbo over the full amplitude window at low opacity with moderate volumetric absorption) so colour still corresponds to conventional dBZ bands. Options: `--recompute`, `--no-serve`, `--serve-only`, `--max-timepoints=N`, `--grid-m=N`, `--grid-z-m=N`, `--splats=N`, `--dbz-floor=N`, `--vert-exag=N`, `--relist`.

---

#### demo_gsplats_4d_cell_tracking_challenge.py - 4D Cell Tracking Challenge (matrix)

Six crops of a developing zebrafish embryo from the public [**Biohub Cell Tracking During Development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) competition, laid out as a complete **3x2 matrix** where every tile is an independent 100-timepoint light-sheet timelapse. Each tile carries all three things at once: the image data as a 4D (ZYX + time) Gaussian-splat volume, the ground-truth position of every tracked cell as points that appear **only** at the timepoint they belong to, and every tracking link as one `indexed` Lines node. Colour is **lineage** throughout, so a founder cell and all of its descendants share a hue; scrub the Time slider and the markers walk along their own tracks while the volume animates beneath them. Cell **divisions** render as real forks, because consecutive links share a vertex row. The crops differ in a scientifically visible way — some show a coherent parallel migration, others a tangle — which is much of why the matrix reads well.

Each of the 199 training crops is an OME-Zarr **0.5** (zarr **v3**) store, `T=100, Z=64, Y=256, X=256` uint16 at 1.625 x 0.40625 x 0.40625 um (a 104 um cube), paired with a **GEFF** tracking graph. The demo ranks the crops by annotation density and shows the top six (~1,500-1,950 cells each; the median crop has only 659 and the sparsest 50, so the choice matters). `DATASETS` carries nine and `--datasets 9` uses them all; six is the default because it fills a rectangle exactly.

**Run**: `luxar demo run gsplats_4d_cell_tracking_challenge`

**Requires**: A Kaggle API token (`~/.kaggle/access_token` or `$KAGGLE_API_TOKEN`) plus the `kaggle` package — the competition endpoint is authenticated, so this demo cannot fetch its data unattended. ~4 GB of download and a CUDA GPU for the fitting pass (~24 s per timepoint, ~40 min per crop); a warm fit cache then rebuilds the scene with no GPU and no raw data at all.

**Demonstrates**: Reading **OME-Zarr 0.5** image stores and **GEFF** cell-lineage graphs (`luxar.gsplats.interop.geff`), all three geometry types in one scene per tile, `combine_as_new_dimension` for a stacked 4D time axis with `coarsen_dims` making time a hard LOD barrier, one `layer=True` group per crop so the Layers panel offers one row per embryo rather than three, `line_type="indexed"` for a lineage forest with shared joints, and two appearance lessons that are measured rather than guessed: dropping the top 5% of splats by characteristic size (a diffuse tail that otherwise renders as opaque discs burying the nuclei under volumetric blending), and an LOD ladder whose **depth is the frame budget** — the `screen-area` selector anchors its finest level at half the screen, so a matrix tile occupies a small fraction of it and draws a coarse level whatever the ladder's length; depth therefore sets both what the opening framing costs and how much detail it discards. A one-level ladder drew 20k splats per timepoint per crop (120k per frame across the matrix) and scrubbing was choppy; the shipped four-level ladder draws ~4,900 per timepoint per crop (29,774 per frame) and never fetches the finer levels, which rendering merged levels back to volume shows costs nothing visible above ~5k. Options: `--datasets=N`, `--timepoints=N`, `--seeds=K`, `--recompute`, `--no-serve`, `--serve-only`.

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
    """Generate the dataset - THE INTERESTING CODE IS IN THIS FUNCTION.

    Args:
        output_path: Where to write zarr
        **params: Generation parameters
    """
    with asection("Generating Data"):
        # 1. Generate positions, colors, radii, etc.
        # The generation logic that makes this demo what it is goes here;
        # for plumbing, reach for the shared helpers in §6.
        positions = ...
        colors = ...

        # 2. Write to zarr
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            # `layer=True` is not optional — see §8.
            scene.add_points('name', positions, colors=colors, layer=True, ...)

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
**The code that makes the demo what it is must be in the demo file itself.**

Whatever a reader opens the file to learn — the maths, the pipeline, the thing
being demonstrated — stays in the file. Plumbing does not: use the shared
helpers in §6 (downloading, caching, argv parsing, HSV→RGB, viewer launching,
optional-dependency gating, precomputed-data loading) rather than hand-rolling
them, and share heavier scaffolding through a `_*_common.py` module as the
interop demos do with `_interop_common.py`.

✅ **GOOD**:
```python
from luxar.demos import hsv_to_rgb  # shared plumbing — fine


def generate_my_data(output_path):
    # The interesting part is right here
    x = np.linspace(0, 10, 1000)
    positions = ...
    colors = hsv_to_rgb(positions[:, 0] / positions[:, 0].max())
    # Write to zarr here
    with LuxarZarrCompiler(output_path) as compiler:
        ...
```

❌ **BAD**:
```python
def generate_my_data(output_path):
    # The demo's whole point now lives somewhere else
    from my_utils import create_positions

    positions = create_positions()  # ← nothing left to read here
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
    add_demo_caption,  # standard bottom-right caption + DEMO_META credit
    launch_viewer,  # serve + open viewer (serve_args=[...] to pass e.g. --profile)
    cached_download,  # download once into ~/.cache/luxar/<name>/, skip-if-present
    robust_download,  # retry/resume a direct download to a chosen path
    download_with_checksum,  # direct download plus sha256 verification
    download_zip_member,  # fetch one member from a remote zip via HTTP ranges
    cache_computed,  # cache an expensive result (UMAP, field) — versioned, param-keyed
    require_local_data,  # gate LFS-tracked local data (clear "git lfs pull" message)
    require_module,  # gate an OPTIONAL dependency at its point of use (see #7)
    parse_demo_flags,  # --recompute / --keep-stale / --no-serve / --serve-only
    parse_int_arg,  # --points=N / --sample N integer flags
    parse_path_arg,  # --cache-dir PATH / --data=PATH path flags (expands ~)
    hsv_to_rgb,  # vectorized rainbow / hue-ramp colouring
    is_lfs_pointer,  # is this LFS-tracked file a pointer stub, not the data?
    print_data_provenance,  # source/licence notice before a third-party download
    detect_device,
    warn_if_no_cuda_gpu,  # GPU/MPS/CPU
    load_precomputed_gsplats,
    load_precomputed_bundle,  # LFS-shipped gsplat data
    voxel_sampled_payload_agreement,  # does a per-splat sidecar still match its fit?
)

# Download once, reused on every later run:
csv = cached_download("https://…/data.csv", "mydemo", "data.csv")

# Cache an expensive UMAP — the KEY must include every param that changes the
# output (sample size, feature set, …) so a stale cache is never reused:
positions = cache_computed(
    "mydemo", f"umap3d_n{n}_f{len(FEATURES)}", lambda: run_umap(features), version=1
)
```

Pass `cache_dir=` when the demo takes a `--cache-dir` override and the result
belongs beside the raw downloads it came from (`demo_caida_as_topology` does
this) — that directory is then used verbatim and `name` is unused.

`cache_computed` writes atomically and quarantines a corrupt cache to
`.corrupt` instead of crashing. A quarantined file is never reused, so
`warn_if_quarantined()` (called from `robust_download()`, the shared download
chokepoint) reports its path and size before a re-fetch starts — a demo that
pulls a multi-gigabyte artifact should not silently restart the download. Use
`find_quarantined_files()` / `format_quarantine_notice()` when a demo needs the
same information inside its own error message.

Sibling demos are importable normally
(`from luxar.demos.demo_x import helper`) — no `importlib` file-path tricks.

`luxar.demos` is the ONLY spelling for these helpers: never import guarded
`luxar.demos._support.*` concern modules (including dataset, download, and
remote-ZIP support), `luxar.utils.colors`, `luxar.utils.scenes`, or private demo
helpers such as `luxar.demos._support._fields` directly from a demo, even though
that is where they live. The authoritative guarded set is
`tests/test_demo_import_spelling.py`'s `DEEP_MODULES`. That test fails the build
on every deep spelling, including relative forms, in every demo module here and
in the three `gsplats/**/demos` trees, and also on a name the barrel does not
re-export. (Scope is the demo modules; `tests/` is out, since a test may
legitimately need the module a private lives in.) That second gap (two missing
symbols, which forced 12 of the 38 files that ended up deep) is what grew the
spelling, so a helper you cannot reach through `luxar.demos` is a bug in
`demos/__init__.py`, not a licence to reach past it.

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

### 8. Expose your geometry as a Layers-panel layer

Pass `layer=True` to every geometry adder (`add_points` / `add_lines` /
`add_gsplats` / `add_mesh` / `add_gsplats_from_*`). The Layers panel lists only
nodes whose zarr attrs carry `layer: true`, and it is the only way a viewer can
toggle a node, re-window its display range, change its gamma or switch its
blending mode — so a demo that omits the kwarg ships a panel that does nothing,
often for its one and only geometry node. `Node.layer` defaults to `False`, so
this is an omission, not a choice.

When a demo emits MANY sibling nodes — one per tile, per tree, per repeat — do
**not** mark each one; hundreds of rows is worse than none. Put them under a
container group added with `layer=True` and let the panel treat it as one
composite row that fans its controls down to every descendant:

```python
trees = scene.add_group("trees", layer=True)
for i, tree in enumerate(trees_to_write):
    trees.add_lines(f"tree_{i:04d}", ...)  # no per-node layer=
```

When the descendants render with a non-default mode, spell that same
`blending_mode` on the group: a group that authors none reads as the panel's
default (`additive`), which mislabels the row and hides mode-specific controls
such as the `volumetric` absorption slider. Emissive geometry (points, lines,
gsplats) already defaults to `additive`, so the `trees` group above needs
nothing; a `volumetric` gsplats wrapper does.

`tests/test_demo_layers.py` fails the build if a demo authors geometry that no
layer covers. Sibling nodes covered by a container group are listed in its
`EXEMPT` table together with the group that covers them, and that group is
checked too — so an exemption cannot outlive the layer it leans on.

## Creating New Demos

1. **Copy a template** (demo_lorenz.py or demo_cubic_array.py)
2. **Rename** to demo_yourname.py
3. **Update docstring** with what it demonstrates
4. **Implement generation** in the generate_* function (keep everything in that function!)
5. **Expose the geometry** with `layer=True` (or a `layer=True` container group) — see §8
6. **Add one caption** with `add_demo_caption(scene, ...)` so `DEMO_META` credit appears — see §6
7. **Test** by running: `hatch run python demo_yourname.py`
8. **Ctrl+C** to stop and verify cleanup works

## Tips

### For Large Datasets
Use progressive writing to avoid memory issues:

```python
# DON'T load all data at once if very large
# DO generate and write in chunks

dims = Dimensions.default_3d()
with LuxarZarrCompiler(output) as compiler:
    scene = compiler.create_scene(dimensions=dims)

    # Write in batches, under one composite group layer (see §8) rather than
    # one Layers-panel row per batch
    batches = scene.add_group("batches", layer=True)
    for i in range(num_batches):
        batch = generate_batch(i)
        batches.add_points(f"batch_{i}", batch)
```

### For nD Demos
Specify dimensions with proper display flags:

```python
dims = Dimensions(
    [
        Dimension("x", unit="um", display=True),
        Dimension("y", unit="um", display=True),
        Dimension("z", unit="um", display=True),
        Dimension("time", unit="s", display=False, discrete=True, range=(0, 99)),
    ]
)
```

### For Complex Math
Add comments explaining the mathematics:

```python
# Generate sphere using spherical coordinates
# φ ∈ [0, 2π], θ ∈ [0, π]
# x = r sin(θ) cos(φ)
# y = r sin(θ) sin(φ)
# z = r cos(θ)
phi = np.random.uniform(0, 2 * np.pi, n)
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
hatch run python packages/luxar/src/luxar/demos/demo_galaxy_simulation.py
hatch run python packages/luxar/src/luxar/demos/demo_nd_transforms.py

# --- Feature Showcases ---
hatch run python packages/luxar/src/luxar/demos/demo_sharpness_showcase.py
hatch run python packages/luxar/src/luxar/demos/demo_lsystem_forest.py
hatch run python packages/luxar/src/luxar/demos/demo_hilbert_curve_3d.py
hatch run python packages/luxar/src/luxar/demos/demo_network_performance.py

# --- Embedding / UMAP ---
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
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_blastocyst_dapi_nuclei.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_3d_blastocyst_multichannel.py
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
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_nexrad_supercell.py
hatch run python packages/luxar/src/luxar/demos/demo_gsplats_4d_cell_tracking_challenge.py
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
