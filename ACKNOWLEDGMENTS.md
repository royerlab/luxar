# Acknowledgments: datasets and scientific data

The Luxar gallery and demos visualize openly available scientific datasets, with
gratitude to the authors, labs, and consortia who produced and shared them. Luxar
only *renders* these data; all rights and credit remain with the original
providers, under their respective licenses. Each demo script's docstring carries
the full citation, and the [README](README.md#where-the-demo-data-lives) explains
where the derived products (Gaussian-splat fits and point catalogs) are archived.

## Microscopy & cell biology

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
- **STORM 3D Microtubules** — COS-7 α-tubulin (Alexa 647) 3D STORM localizations; Sieben (2019), [Zenodo 3547521](https://doi.org/10.5281/zenodo.3547521) (CC BY 4.0).
- **CODEX Pancreas** — 12-plex CODEX human pancreas section; Björklund et al. (2023), [Zenodo 7742474](https://doi.org/10.5281/zenodo.7742474) (CC BY 4.0).
- **Acto3D Heart** — E13.5 mouse heart, Zeiss Lightsheet 7, [Acto3D sample data](https://github.com/Acto3D/Acto3D); Takeshita et al. (2024), *Development*, [doi:10.1242/dev.202550](https://doi.org/10.1242/dev.202550). The sample data states no reuse terms, so it is fitted locally and never redistributed.
- **Zebrafish Gastrulation (endoderm, 4D)** — Pia Aanstad (2018), confocal timelapse, [Zenodo 1211599](https://doi.org/10.5281/zenodo.1211599) (CC BY-SA 4.0).
- **Cell Tracking (Kaggle)** — [Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development): zebrafish light-sheet crops and lineages, CZ Biohub SF (CC0 1.0).

## Medical & anatomy

- **CT Anatomy Atlas** — [TotalSegmentator](https://zenodo.org/records/10047263) (102-subject subset, v2.0.1, CC BY 4.0); Wasserthal et al. (2023), *Radiology: AI*, [doi:10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024).
- **Visible Human Head** — U.S. National Library of Medicine [Visible Human Project](https://www.nlm.nih.gov/research/visible/visible_human.html) (Male, axial color cryosections); Spitzer et al. (1996), *JAMIA*, [doi:10.1136/jamia.1996.96236280](https://doi.org/10.1136/jamia.1996.96236280) (public domain; no NLM endorsement implied).
- **CMU-1 Pathology Slide** — [OpenSlide test data](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/), Aperio `CMU-1.svs`; Goode et al. (2013), *J. Pathol. Inform.*, [doi:10.4103/2153-3539.119005](https://doi.org/10.4103/2153-3539.119005) (CC0 1.0).

## Astronomy & geoscience

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

## Connectomes, structures & networks

- **FlyWire Connectome** — FlyWire whole-brain connectome, public release 783; Dorkenwald et al. (2024) & Schlegel et al. (2024), *Nature*; [annotations](https://github.com/flyconnectome/flywire_annotations), [connectivity (Zenodo)](https://zenodo.org/records/10676866), [Codex](https://codex.flywire.ai/) (CC BY 4.0).
- **Single-Cell 3D Genome** — Dip-C; Tan et al. (2018), *Science* 361:924, [doi:10.1126/science.aat5641](https://doi.org/10.1126/science.aat5641); GEO GSE117876.
- **Human White-Matter Tractography** — [HCP-1065 population-averaged tractography atlas](https://brain.labsolver.org/hcp_trk_atlas.html); Yeh, F-C. (2022), *Nat. Commun.* 13:4933, [doi:10.1038/s41467-022-32595-4](https://doi.org/10.1038/s41467-022-32595-4) (CC BY-SA 4.0). Derived from the Human Connectome Project, WU-Minn Consortium (PIs David Van Essen & Kamil Ugurbil; 1U54MH091657), funded by the 16 NIH Institutes and Centers supporting the NIH Blueprint for Neuroscience Research, and by the McDonnell Center for Systems Neuroscience at Washington University; used under the [WU-Minn HCP open-access data-use terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
- **ATP Synthase** — PDB [5DN6](https://www.rcsb.org/structure/5DN6), Zhou et al. (2015), *eLife*, [doi:10.7554/eLife.10180](https://doi.org/10.7554/eLife.10180); RCSB [PDB-101](https://pdb101.rcsb.org/motm/72).
- **Nuclear Pore Complex** — human NPC scaffold, PDB [7R5K](https://www.rcsb.org/structure/7R5K) (constricted) and 7R5J (dilated); Mosalaganti et al. (2022), *Science* 376:eabm9506, [doi:10.1126/science.abm9506](https://doi.org/10.1126/science.abm9506); coordinates from the RCSB PDB.
- **PBCV-1 Capsid (cryo-EM)** — EMDB [EMD-5384](https://www.ebi.ac.uk/emdb/EMD-5384); Zhang et al. (2011), *PNAS* 108:14837, [doi:10.1073/pnas.1107847108](https://doi.org/10.1073/pnas.1107847108).
- **CAIDA AS topology** — [CAIDA](https://asrank.caida.org/) AS-relationships, AS-organizations & AS Rank (UC San Diego / CAIDA); Luckie et al. (2013), [doi:10.1145/2504730.2504735](https://doi.org/10.1145/2504730.2504735).
- **HuRI interactome** — Luck et al. (2020), *Nature* 580:402, [doi:10.1038/s41586-020-2188-x](https://doi.org/10.1038/s41586-020-2188-x); [Human Reference Interactome](https://interactome-atlas.org/).

## Single-cell atlases & embeddings

- **Tabula Sapiens** — Tabula Sapiens Consortium (2022), *Science*, [doi:10.1126/science.abl4896](https://doi.org/10.1126/science.abl4896); accessed via [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/) (CC BY 4.0).
- **Zebrahub** — [CZ Biohub Zebrahub](https://zebrahub.org); RNA-velocity embedding: Lange et al. (2024), *Cell*, [doi:10.1016/j.cell.2024.09.047](https://doi.org/10.1016/j.cell.2024.09.047); Zebrahub-Multiome: Kim et al. (2024), [bioRxiv:2024.10.18.618987](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v2); 3D embeddings by Yang-Joon Kim (CZ Biohub SF).
- **Protein Landscape** — CAFA5 protein embeddings via ProtT5 (Elnaggar et al. 2022, *IEEE TPAMI*, [doi:10.1109/TPAMI.2021.3095381](https://doi.org/10.1109/TPAMI.2021.3095381)); [CAFA5 challenge](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction).
- **Spotify Tracks** — [`maharshipandya/spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (Hugging Face), derived from the Spotify Web API audio features.
- **CELLxGENE Census UMAP** — 3D UMAP of the [CZ CELLxGENE Discover Census](https://cellxgene.cziscience.com/) scVI latent space; CZI Cell Science Program et al. (2025), *Nucleic Acids Res.* 53:D886, [doi:10.1093/nar/gkae1142](https://doi.org/10.1093/nar/gkae1142) (CC BY 4.0; the contributing studies are listed in the Census release).
- **ESM protein landscapes** — [UniProt/Swiss-Prot](https://www.uniprot.org/) sequences (CC BY 4.0) embedded with EvolutionaryScale ESM C / ESM3-open; **ESM Protein Universe** — the [ESM Atlas](https://biohub.ai/esm/protein/atlas) cluster map, Candido et al. (2026), bioRxiv, [doi:10.64898/2026.06.03.729735](https://doi.org/10.64898/2026.06.03.729735), shown with permission.
- **arXiv Papers** — 3.29M preprint embeddings from Kaggle [tomtum/openai-arxiv-embeddings](https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings); titles and categories from the Cornell University arXiv metadata snapshot.
- **Bird Plumage Colour Space** — [BirdColorBase](https://github.com/BirdColorBase/home), 360,432 reflectance spectra over 2,632 species (MIT).
- **Human Multiome** — peak-accessibility UMAP of human single-cell ATAC-seq; data from Domcke et al. (2020), *A human cell atlas of fetal chromatin accessibility*, *Science* 370:eaba7612, [doi:10.1126/science.aba7612](https://doi.org/10.1126/science.aba7612); peak-UMAP analysis from Zebrahub-Multiome, Kim et al. (2024), [bioRxiv:2024.10.18.618987](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v2).
- **Mouse Multiome** — peak-accessibility UMAP of a mouse single-cell multiome; data from Argelaguet et al. (2022), *Decoding gene regulation in the mouse embryo using single-cell multi-omics*, [bioRxiv:2022.06.15.496239](https://doi.org/10.1101/2022.06.15.496239); peak-UMAP analysis from Zebrahub-Multiome, Kim et al. (2024), [bioRxiv:2024.10.18.618987](https://www.biorxiv.org/content/10.1101/2024.10.18.618987v2).

## Photogrammetry & 3DGS captures

- **Cluster Fly** (macro photogrammetry) — Dany Bittel, [danybittel.ch/macro](https://danybittel.ch/macro) (CC BY 4.0).
- **Observatories** (Rubin, Gemini South) — Gaussian-splat captures by khyron, [khyron/Gaussian-Splatting](https://github.com/khyron/Gaussian-Splatting) (CC BY 4.0).
- **Scaniverse SPZ samples** — Niantic Labs [spz](https://github.com/nianticlabs/spz) reference captures (MIT).
- **INRIA "bonsai"** and **Mip-NeRF 360 "garden"** — Kerbl et al. (2023), *ACM TOG* 42(4), [doi:10.1145/3592433](https://doi.org/10.1145/3592433); captures from Mip-NeRF 360, Barron et al. (2022), *CVPR*, [doi:10.1109/CVPR52688.2022.00539](https://doi.org/10.1109/CVPR52688.2022.00539). Research-use terms, so both are fetched at run time and never redistributed.
- **MatrixCity aerial** — Li et al. (2023), *ICCV*, [doi:10.1109/iccv51070.2023.00297](https://doi.org/10.1109/iccv51070.2023.00297); 3DGS reconstruction hosted on [SuperSplat](https://superspl.at/scene/ace6e5b0).

Synthetic, procedurally generated demos — **Lorenz Attractor**, **Spiral Galaxy**, **Galaxy Simulation**, **Rainbow Sphere**, **Quantum Orbitals**, **Hilbert Curve**, **Bioluminescent Ocean**, **Particle Collision** (static and animated, inspired by CERN LHC events), **4D Fractals**, **Mandelbulb**, **Quasicrystal**, **Turing Patterns**, **Evolving Cloud**, **Cubic Array**, **L-System Forest**, the **nD Transform Test Bench**, and **Exotic Surfaces** (nodal surfaces after von Schnering & Nesper 1991, *Z. Phys. B* 83:407, [doi:10.1007/BF01313411](https://doi.org/10.1007/BF01313411)) — use no external data. The two **ChromaTrace** demos read a bundle you supply and are not hosted.
