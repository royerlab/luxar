#!/usr/bin/env python3
"""ESM Protein Stories — ten guided tours of the Swiss-Prot landscape.

A story-telling variant of ``demo_esm3_protein_landscape``: the same ~575K
Swiss-Prot proteins embedded with ESM C and laid out by 3D UMAP, plus a hidden
``story`` dimension that walks through ten protein families whose clusters in
the map each carry a genuinely interesting piece of biology. Stepping the
dimension flies the camera to the cluster, lights it up, and shows a panel of
facts and one open question.

This is the reference scene for the viewer's remote-control features
(``docs/guides/specs/REMOTE_CONTROL_SPEC.md``): the camera poses are authored
``Waypoint``\\s bound to the ``story`` dimension, the captions are dimension-aware
overlays, and auto-rotate is on so a story step keeps the turntable spinning
while it moves the point the camera spins around.

Scene structure (two toggleable layers plus one highlight per story):
    - **Backdrop** — every protein, dimmed, in the base demo's taxon colours,
      stored once with ``extend_to_all=["story"]`` so it is visible at every
      story without duplication.
    - **Story k** — the members of story k's cluster in the story's own
      colour, pinned to slot k of the ``story`` dimension.

Stories are defined by a protein-name pattern AND a radius around the family's
median position in the map, so a highlight is the visible blob, not the
family's stragglers; the counts in each panel say how many made it.

Navigation:
    Press '1' to select the story slider, then '[' / ']' to step. Or drive it
    from outside with ``LuxarApp.setDimensionValue(0, k)``.

Usage:
    python -m luxar.demos.demo_esm3_protein_stories
    python -m luxar.demos.demo_esm3_protein_stories --no-serve
    python -m luxar.demos.demo_esm3_protein_stories --no-auto-rotate

Requires the base demo's cache (``~/.cache/luxar/esm3_swissprot``: the UMAP
positions and the metadata). Run ``luxar demo run esm3_protein_landscape``
once to build it; this demo never recomputes embeddings itself.
"""

from __future__ import annotations

DEMO_META = {
    "key": "esm3_protein_stories",
    "title": "ESM Protein Stories — ten guided tours of the Swiss-Prot landscape",
    "description": (
        "The ESM C Swiss-Prot 3D UMAP with a hidden story dimension: step through "
        "ten protein-family clusters, each with a fly-to camera waypoint, a "
        "highlight and a panel of researched facts."
    ),
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    # Reads the base demo's cache; adds its own for the PDB turntables.
    "caches": ["esm3_swissprot", "pdb_turntables"],
    "outputs": ["esm3_protein_stories"],
    "citation": {
        "short": "UniProt/Swiss-Prot; embeddings by EvolutionaryScale ESM C, 2024",
        "ref": "UniProt / EvolutionaryScale 2024",
        "license": "CC BY 4.0",
    },
}

import html
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig, Waypoint
from luxar.demos import add_demo_caption, launch_viewer
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG, pull_in
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.demos._pdb_turntable import TurntableAssets, render_turntables
from luxar.demos.demo_esm3_protein_landscape import (
    TAXON_COLORS,
    _linkable_accessions,
    _uniprot_link_attrs,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# The stories
# =============================================================================
#
# Every number below was checked against a source when the demo was written;
# the source is named next to the fact. Keep it that way: a kiosk in front of
# an audience is the wrong place to discover a hallucinated statistic.


@dataclass(frozen=True)
class Story:
    """One stop of the tour: which proteins, where to look, what to say."""

    #: Slot label on the ``story`` dimension and the highlight layer's name.
    key: str
    title: str
    subtitle: str
    #: Regex over Swiss-Prot protein names selecting the family.
    pattern: str
    #: Highlight colour (linear RGB in [0, 1]).
    color: tuple[float, float, float]
    facts: tuple[str, ...]
    #: The open question that closes the panel.
    mystery: str
    #: Members must lie within this distance (UMAP units) of the family's
    #: median position — the visible blob, not the stragglers.
    radius: float = 0.8
    #: Optional taxon filter (a value of the base demo's kingdom column).
    kingdom: str | None = None
    #: Framing: camera distance = max(min_distance, distance_per_radius * r95).
    #: A blob of radius r95 then spans roughly a third of the frame height at
    #: the 63° cinematic lens, with the rest of the landscape for context.
    distance_per_radius: float = 9.0
    min_distance: float = 5.0
    flight_ms: int = 2500
    tags: tuple[str, ...] = field(default_factory=tuple)
    #: Representative PDB entry rendered as the left-hand turntable ("" = none).
    pdb_id: str = ""


STORIES: tuple[Story, ...] = (
    Story(
        key="Hemoglobin",
        title="Hemoglobin — the molecule of breath",
        subtitle="~570 hemoglobin chains from fish to humans, in one tight knot",
        pattern=r"^Hemoglobin subunit",
        color=(0.98, 0.22, 0.28),
        facts=(
            # ~280 million Hb molecules per red cell: standard physiology
            # figure (EBSCO Research Starters; LibreTexts).
            "Each red blood cell carries roughly 280 million hemoglobin "
            "molecules, and each of them can hold four oxygen molecules.",
            # Pauling, Itano, Singer & Wells, Science 110:543 (1949).
            "In 1949 Linus Pauling and Harvey Itano showed that sickle-cell "
            "anaemia is caused by an altered hemoglobin — the first "
            "“molecular disease”. The culprit is a single amino acid: "
            "glutamate swapped for valine at position 6 of the beta chain.",
            # Perutz's low-resolution model, 1959; Nobel 1962 with Kendrew
            # (MRC LMB; Britannica).
            "Max Perutz needed 22 years of X-ray work to see the molecule: "
            "his 1959 model, carved from balsa wood, showed four chains "
            "cradling four heme groups. Nobel Prize 1962, shared with John "
            "Kendrew for myoglobin.",
            "Look elsewhere in the map: myoglobin (muscle) and leghemoglobin "
            "(legume root nodules — and the “bleeding” in plant-based "
            "burgers) share hemoglobin's fold, yet the model places them in "
            "other neighbourhoods. It reads sequences, and theirs drifted "
            "apart long ago.",
        ),
        # Biagioli et al., PNAS 106:15454 (2009); reviews since.
        mystery=(
            "Hemoglobin also turns up inside dopamine neurons of the midbrain, "
            "nowhere near blood. What it does there — and whether it matters "
            "in Parkinson's disease — is still debated."
        ),
        tags=("blood", "medicine", "structure"),
        pdb_id="2HHB",
    ),
    Story(
        key="Photosystem II",
        title="Photosystem II D1 — the protein that made the sky breathable",
        subtitle="The reaction-centre protein of the only enzyme that splits water",
        pattern=r"Photosystem II protein D1",
        color=(0.4, 0.98, 0.4),
        radius=0.6,
        facts=(
            # Umena, Kawakami, Shen & Kamiya, Nature 473:55 (2011).
            "D1 sits at the heart of photosystem II, the only enzyme known "
            "that splits water. Its manganese–calcium cluster, "
            "Mn₄CaO₅, was finally seen atom by atom in 2011, at "
            "1.9 Å resolution.",
            # Great Oxidation Event ~2.3–2.4 Gya (Current Biology 2009; ScienceDirect).
            "Cyanobacteria running this machine drove the Great Oxidation "
            "Event about 2.4 billion years ago — the moment Earth's atmosphere "
            "began to fill with oxygen.",
            # D1 turns over fastest of all thylakoid proteins; half-life ~2 h in
            # growth light, faster in high light (Aro et al. 1993; Photosynth.
            # Res. reviews).
            "Splitting water has a price: D1 is damaged by its own chemistry "
            "and is the fastest-replaced protein in the photosynthetic "
            "membrane — a half-life of about two hours in ordinary light, "
            "shorter in full sun. A leaf rebuilds this protein all day long.",
            # Purple-bacteria reaction centre L/M chains are homologous to
            # D1/D2 (Deisenhofer, Huber & Michel; Nobel 1988).
            "Its neighbours in this map are the L and M chains of "
            "purple-bacteria reaction centres: D1's distant cousins, which "
            "harvest light but never learned to split water. Their kinship "
            "was a key clue when the first photosynthetic structure was "
            "solved (Nobel Prize in Chemistry 1988).",
        ),
        # Cardona et al., Geobiology / PMC6492235: early Archean origin.
        mystery=(
            "Molecular clocks place a water-splitting photosystem deep in the "
            "Archean, perhaps a billion years before oxygen rose. Why did the "
            "planet wait so long to change?"
        ),
        tags=("photosynthesis", "deep time"),
        pdb_id="3WU2",
    ),
    Story(
        key="Hsp70",
        title="Hsp70 — the oldest job in the cell",
        subtitle="Bacteria, archaea, plants and animals share one blob",
        pattern=r"Heat shock 70|Heat shock cognate 71|Chaperone protein [Dd]naK",
        color=(1.0, 0.82, 0.25),
        radius=0.7,
        facts=(
            "Hsp70 — DnaK in bacteria — is a chaperone: it holds unfolded "
            "proteins, refolds the damaged ones and hands the hopeless ones to "
            "the shredder. Every cell type in every domain of life has it.",
            # Human Hsp70 vs E. coli DnaK: ~47–48% identity (Brocchieri et al.
            # 2008; Frontiers Mol. Biosci. 2021).
            "After some three billion years of separate evolution, human Hsp70 "
            "and E. coli DnaK are still about 47% identical, letter for "
            "letter — one of the most conserved proteins known.",
            "That is why this blob mixes colours: bacteria, archaea, plants "
            "and animals interleave here. The map is showing a protein older "
            "than the deepest split in the tree of life.",
            # Ritossa, Experientia 18:571 (1962); Ritossa's own account in
            # Cell Stress & Chaperones (PMC4147064).
            "It was discovered by accident. In 1962 Ferruccio Ritossa saw new "
            "“puffs” on fruit-fly chromosomes after someone in the lab "
            "had nudged his incubator's temperature — the first observation "
            "of the heat-shock response.",
        ),
        mystery=(
            "Cancer cells over-produce Hsp70 to survive their own chaos, and "
            "drugs that turn this against them have been tried for decades. "
            "None has yet reached the clinic. Why is such a universal protein "
            "so hard to target?"
        ),
        tags=("chaperone", "evolution"),
        pdb_id="2KHO",
    ),
    Story(
        key="Viral surface proteins",
        title="The intruders' continent",
        subtitle="Haemagglutinin, spike and the coats of a thousand viruses",
        pattern=(
            r"^Hemagglutinin|^Spike glycoprotein|^Envelope glycoprotein|"
            r"^Fusion glycoprotein|^Outer capsid glycoprotein VP7|^Glycoprotein\b|"
            r"^Envelopment polyprotein|^Structural polyprotein|^Major capsid protein|"
            r"^Capsid protein|^Nucleoprotein$|^Matrix protein"
        ),
        color=(0.98, 0.35, 0.9),
        radius=1.2,
        kingdom="Viruses",
        distance_per_radius=4.5,
        flight_ms=3000,
        facts=(
            # Class I fusion proteins share the six-helix-bundle mechanism
            # (J. Virol. 77:8801 (2003); reviews PMC9166635, PMC8709411).
            "Influenza haemagglutinin, coronavirus spike, HIV's envelope and "
            "Ebola's glycoprotein belong to unrelated viruses, yet they pull "
            "the same trick: class I fusion proteins that snap into a "
            "six-helix bundle, dragging virus and cell membranes together.",
            "Haemagglutinin gives influenza its “H” (H1N1, H5N1...). "
            # 50–100 million: Johnson & Mueller 2002; CDC EID 2006.
            "The 1918 pandemic, an H1N1 virus, killed an estimated 50 million "
            "people — more than the war it followed.",
            "The coronavirus spike grips the human ACE2 receptor to open a "
            "cell; its receptor-binding domain is what most COVID-19 vaccines "
            "teach the immune system to recognise.",
            "Here they gather into one continent although they share no "
            "common ancestor. The language model groups them by how they are "
            "built and what they do — a hint of convergent design, visible "
            "as geography.",
        ),
        mystery=(
            "Most viral proteins have no known relatives at all — the "
            "“viral dark matter” of metagenomes. Where would they land "
            "on this map, and would they form continents of their own?"
        ),
        tags=("virology", "pandemics", "convergence"),
        pdb_id="1RUZ",
    ),
    Story(
        key="Prion protein",
        title="A protein that is its own pathogen",
        subtitle="The prion protein, PrP — infectious with no genes at all",
        pattern=r"prion protein",
        color=(0.35, 0.88, 1.0),
        radius=0.6,
        facts=(
            # Gajdusek, Nobel 1976; kuru among the Fore (Nobel press release 1997).
            "Kuru among the Fore people of New Guinea, scrapie in sheep, "
            "Creutzfeldt–Jakob disease in humans: brain diseases that "
            "spread like infections, yet no virus was ever found. Carleton "
            "Gajdusek traced kuru to funerary cannibalism (Nobel Prize 1976).",
            # Prusiner 1982 'prion'; Nobel 1997 (NobelPrize.org press release).
            "In 1982 Stanley Prusiner proposed the heresy that the agent is a "
            "protein alone — a misfolded shape that converts healthy copies "
            "into itself. He called it a prion. Nobel Prize 1997.",
            # BSE evident 1985–86 in the UK; vCJD first identified March 1996
            # (ECDC; Stanford prion timeline).
            "Mad cow disease surfaced in British cattle in the mid-1980s; its "
            "human form, variant CJD, was identified in 1996 — prions had "
            "crossed from one species to another through food.",
            "Every mammal carries the healthy form, PrPᶜ, on the surface "
            "of its neurons. In this map its neighbours are small "
            "neuropeptides, the brain's short messengers.",
        ),
        # Bremer et al., Nat. Neurosci. 2010 (myelin maintenance); reviews
        # BMC Biol. 2017, Front. Mol. Biosci. 2017.
        mystery=(
            "Forty years on, PrP's day job is still unsettled. The best evidence "
            "says it helps maintain the myelin insulation of nerves — but "
            "mice without it live almost normal lives. What is it really for?"
        ),
        tags=("neuroscience", "mystery"),
        pdb_id="1QLX",
    ),
    Story(
        key="ATP synthase",
        title="ATP synthase — the turbine in every cell",
        subtitle="The rotary motor that makes the currency of life, in bacteria and in us",
        pattern=r"ATP synthase subunit beta\b",
        color=(1.0, 0.5, 0.12),
        radius=0.6,
        facts=(
            # Boyer & Walker, Nobel Prize in Chemistry 1997 (NobelPrize.org).
            "ATP synthase is a machine with a rotating axle: a flow of protons "
            "turns it, and each turn presses out three ATP molecules. Paul "
            "Boyer proposed the mechanism, John Walker solved the structure — "
            "Nobel Prize 1997.",
            # Noji, Yasuda, Yoshida & Kinosita, Nature 386:299 (1997).
            "In 1997 the rotation was watched directly: a single F₁ motor, "
            "with a fluorescent actin filament glued to its axle, spun under "
            "the microscope at several revolutions per second.",
            # ~100–150 mol ATP/day ≈ 50–75 kg (BNID 105606; NIGMS Biobeat).
            "You make and spend roughly your own body weight in ATP every day "
            "— some 50 to 75 kilograms — recycling each molecule hundreds of "
            "times.",
            "This blob holds the beta subunit from bacteria, plant chloroplasts "
            "and animal mitochondria side by side: the same motor, inherited "
            "from the bacteria that became our mitochondria.",
        ),
        mystery=(
            "F₁ is one of the most efficient motors known: almost all of the "
            "energy that goes in comes out as rotation, with next to nothing "
            "lost as heat. How a protein manages that is still debated."
        ),
        tags=("energy", "structure"),
        pdb_id="1BMF",
    ),
    Story(
        key="RuBisCO",
        title="RuBisCO — the most abundant enzyme, and one of the slowest",
        subtitle="The protein that pulls carbon out of the air for almost all life",
        pattern=r"^Ribulose bisphosphate carboxylase large chain",
        color=(0.72, 1.0, 0.3),
        radius=0.6,
        facts=(
            # Bar-On & Milo, PNAS 116:4738 (2019): ~0.7 Gt; Raven 2013.
            "Nearly every carbon atom in every living thing passed through "
            "this enzyme. Earth carries about 0.7 billion tonnes of it — very "
            "likely the most abundant protein on the planet.",
            # Time-averaged ~0.03 s⁻¹ on land (Bar-On & Milo 2019); in vitro
            # only a few per second.
            "It is also remarkably slow: a few reactions per second at best, "
            "and averaged over a growing season a land plant's RuBisCO fixes "
            "about one CO₂ every thirty seconds. Plants compensate by making "
            "enormous amounts of it.",
            "It makes mistakes, too: it cannot tell O₂ from CO₂ well, and every "
            "time it grabs oxygen the plant pays in lost carbon and energy "
            "(photorespiration). Cyanobacteria and algae pack the enzyme into "
            "compartments — carboxysomes, pyrenoids — to feed it concentrated "
            "CO₂.",
            "Here the large chain forms one tight knot; its distant relatives "
            "in other microbes sit elsewhere in the map.",
        ),
        mystery=(
            "Three billion years of evolution have not produced a fast, "
            "accurate RuBisCO. Is speed against specificity a wall that cannot "
            "be climbed, or has nobody — nature or engineer — found the path?"
        ),
        tags=("photosynthesis", "enzyme"),
        pdb_id="8RUC",
    ),
    Story(
        # No '/' — the key doubles as a node name.
        key="RecA and Rad51",
        title="RecA and Rad51 — the machine that mends broken DNA",
        subtitle="One recombinase, from E. coli to the BRCA2 pathway in our cells",
        pattern=r"^Protein RecA|DNA repair protein RAD51",
        color=(0.62, 0.48, 1.0),
        radius=0.6,
        facts=(
            # Clark & Margulies 1965 (PNAS); reviewed Bell & Kowalczykowski,
            # Trends Biochem. Sci. 2016.
            "Found in 1965 by screening E. coli mutants that could no longer "
            "swap genes, RecA turned out to be the heart of homologous "
            "recombination: it coats a broken DNA strand into a filament that "
            "searches the whole genome for the matching sequence and pairs "
            "them up.",
            # SOS response: RecA–ssDNA filament triggers LexA self-cleavage.
            "In bacteria the same filament is an alarm: it triggers the SOS "
            "response, switching on dozens of repair genes when DNA is "
            "damaged.",
            # Human RAD51 is loaded onto resected ends by BRCA2's BRC repeats.
            "Our version is RAD51. It is loaded onto broken DNA by BRCA2 — the "
            "protein whose inherited mutations cause a large share of "
            "hereditary breast and ovarian cancer. Repair fails, and errors "
            "accumulate.",
            "This blob mixes bacteria and eukaryotes: the recombinase predates "
            "their split, and its shape has barely moved since.",
        ),
        mystery=(
            "A RecA filament finds one matching stretch among millions of base "
            "pairs in minutes. How the search is that fast — sliding, hopping, "
            "or testing many sites at once — is still argued over."
        ),
        tags=("DNA repair", "cancer"),
        pdb_id="3CMW",
    ),
    Story(
        key="Insulin",
        title="Insulin — a century of firsts",
        subtitle="The hormone that keeps being the first protein to do something",
        pattern=r"^Insulin$|^Insulin-\d|^Insulin A|^Insulin B",
        color=(1.0, 0.55, 0.65),
        radius=0.6,
        facts=(
            # Banting & Best isolate insulin 27 July 1921; Leonard Thompson,
            # 14, first injection 11 Jan 1922 (U. Toronto Fisher Library;
            # UMass Chan). Nobel 1923 to Banting and Macleod.
            "Isolated by Frederick Banting and Charles Best in the summer of "
            "1921; on 11 January 1922 a 14-year-old, Leonard Thompson, became "
            "the first patient injected. Diabetes stopped being a death "
            "sentence. Nobel Prize 1923.",
            # Sanger 1955; Nobel 1958.
            "In 1955 Frederick Sanger read its amino-acid sequence — the first "
            "protein ever sequenced, proof that proteins have a defined "
            "sequence at all. Nobel Prize 1958.",
            # Hodgkin 1969, 34 years after her first insulin crystals.
            "Dorothy Hodgkin solved its three-dimensional structure in 1969, "
            "thirty-four years after she first photographed its crystals.",
            # Humulin, FDA 1982: first recombinant-DNA drug.
            "In 1982 human insulin made by engineered bacteria (Humulin) "
            "became the first drug ever produced with recombinant DNA.",
        ),
        # daf-2 (the worm's insulin/IGF-1 receptor): loss doubles lifespan
        # (Kenyon et al. 1993; Kimura et al. 1997).
        mystery=(
            "In the worm C. elegans, weakening the insulin receptor (daf-2) "
            "doubles lifespan. Why does a hormone for blood sugar hold a dial "
            "for ageing — and does the dial exist in us?"
        ),
        tags=("medicine", "history"),
        pdb_id="4INS",
    ),
    Story(
        key="Cone-snail toxins",
        title="Conotoxins — venom that became medicine",
        subtitle="One superfamily of cone-snail peptides; the rest are scattered across the map",
        pattern=r"onotoxin",
        color=(0.3, 1.0, 0.75),
        radius=0.6,
        facts=(
            # ~800 Conus species, 100–1,000+ peptides each, >80,000 estimated
            # (Toxins 2019; Frontiers Mar. Sci. 2022).
            "Cone snails hunt fish, worms and other snails with a harpoon and a "
            "venom cocktail. There are around 800 species and each makes "
            "hundreds to a thousand different peptides — tens of thousands of "
            "toxins in all, each a precise key for one ion channel or "
            "receptor.",
            # Olivera (U. Utah), cone snails from the Philippines.
            "Much of what is known began with Baldomero Olivera collecting "
            "snails on Philippine reefs; his lab turned their toxins into "
            "tools that mapped the ion channels of the nervous system.",
            # Ziconotide = ω-conotoxin MVIIA (Conus magus), 25 aa; FDA 28 Dec
            # 2004 as Prialt — first marine natural product approved.
            "One of them, a 25-amino-acid peptide from Conus magus, is now a "
            "drug: ziconotide (Prialt), approved in 2004 for severe chronic "
            "pain. It blocks the calcium channels that carry pain signals in "
            "the spinal cord — the first medicine ever taken from the sea.",
            "Swiss-Prot holds over 1,200 conotoxins. This knot is one "
            "superfamily of them; the others are strewn across the whole map, "
            "because venom evolves faster than almost anything else.",
        ),
        mystery=(
            "Why do venom genes mutate and diversify so much faster than the "
            "rest of the genome? An arms race with prey is the usual answer, "
            "but the molecular engine of that speed is still being worked out."
        ),
        tags=("venom", "neuroscience", "medicine"),
        pdb_id="1OMG",
    ),
)

OVERVIEW_TITLE = "Ten stories in the protein universe"
OVERVIEW_HTML = (
    "Every point is one of {n:,} Swiss-Prot proteins, placed by a protein "
    "language model (ESM C) so that proteins with similar sequences sit close "
    "together, then projected to 3D with UMAP. Colours are taxonomic groups."
    "<br><br>Step the <b>story</b> dimension to fly to ten clusters that each "
    "tell a piece of biology: blood, sunlight, the oldest chaperone, the coats "
    "of viruses, a protein that infects without genes, the cell's turbine, "
    "the slowest important enzyme, the machine that mends DNA, a century of "
    "insulin, and venom that became medicine."
)

# Backdrop / highlight appearance. The backdrop keeps the base demo's taxon
# colours (so the overview still reads as the landscape) but dimmed, so a lit
# story cluster carries the frame.
BACKDROP_RADIUS = 0.012
BACKDROP_INTENSITY = 0.08
BACKDROP_OPACITY = 0.7
# The highlight sits under the cinematic bloom. Tuned in the browser at story
# distance: 0.032 / 0.55 saturated into one white blob, 0.02 / 0.22 vanished
# into the backdrop; this reads as a coloured cluster of individual proteins.
HIGHLIGHT_RADIUS = 0.028
HIGHLIGHT_INTENSITY = 0.38

STORY_DIM = "story"
PANEL_WIDTH = 0.32

# Marker sphere around each story's cluster: a subtle translucent shell so the
# cluster reads as a place, not just as brighter dots. Built from the existing
# mesh model — per-vertex RGBA alpha, additive blending, the light-free
# view-anchored shade term with the ambient floor removed so only the lit
# limb shows — and sat in its own depth band between backdrop and highlight.
SPHERE_RADIUS_SCALE = 1.35  # × the cluster's r95
SPHERE_MIN_RADIUS = 0.35
# Tuned in the browser: 0.10 read as a solid coloured disc over the cluster;
# this is a veil the points still shine through. Additive, double-sided, so
# the front and back shells sum to about twice this at the centre.
SPHERE_ALPHA = 0.035
SPHERE_SUBDIVISIONS = 3  # icosphere: 642 vertices, 1280 faces
SPHERE_LAYER_ORDER = 5  # backdrop 0 < sphere < highlight 10

# Left-hand turntable: a representative PDB structure per story, ray-traced by
# PyMOL into a transparent 60 fps WebM (see `_pdb_turntable`). Sits at panel
# height, clear of the activity rail; the caption goes just above it.
TURNTABLE_POSITION = (0.06, 0.5)
TURNTABLE_WIDTH = 0.26  # viewport-width fraction; height follows the square video
TURNTABLE_CAPTION_POSITION = (0.06, 0.26)
TURNTABLE_CACHE = "pdb_turntables"


# =============================================================================
# Pure helpers (unit-tested)
# =============================================================================


@dataclass(frozen=True)
class StoryCluster:
    """A story resolved against the data: who is in it and where it sits."""

    indices: np.ndarray
    centre: np.ndarray
    #: 95th-percentile distance of members to the centre (framing radius).
    r95: float
    #: How many proteins matched the name pattern before the radius cut.
    n_named: int


def _densest_member(family_pos: np.ndarray, radius: float) -> np.ndarray:
    """The member with the most family neighbours within ``radius``.

    Exact for families up to a few thousand members (one pairwise distance
    matrix); larger families are subsampled to 2,000 candidates, which is far
    denser than any blob the demo cares about.
    """
    n = len(family_pos)
    if n == 1:
        return family_pos[0]
    cand = family_pos if n <= 2000 else family_pos[:: max(1, n // 2000)]
    # (candidates × members) distances, chunked so memory stays modest.
    counts = np.empty(len(cand), dtype=np.int64)
    step = 512
    for start in range(0, len(cand), step):
        block = cand[start : start + step]
        d = np.linalg.norm(block[:, None, :] - family_pos[None, :, :], axis=2)
        counts[start : start + step] = (d <= radius).sum(axis=1)
    return cand[int(np.argmax(counts))]


def select_story_members(
    story: Story,
    names: np.ndarray,
    kingdoms: np.ndarray,
    positions: np.ndarray,
) -> StoryCluster:
    """Resolve a story to the proteins that form its visible blob.

    Members are the proteins whose name matches ``story.pattern`` (and whose
    taxon matches ``story.kingdom`` when set) AND that lie within
    ``story.radius`` of the family's densest member. The radius cut is what
    makes the highlight a cluster: a family's stragglers scattered across the
    map would otherwise light up in the wrong places.
    """
    rx = re.compile(story.pattern)
    named = np.fromiter((bool(rx.search(str(s))) for s in names), bool, len(names))
    if story.kingdom is not None:
        named &= np.asarray([str(k) == story.kingdom for k in kingdoms], dtype=bool)
    n_named = int(named.sum())
    if n_named == 0:
        raise ValueError(f"story {story.key!r}: pattern matched no protein names")
    # Centre on the family's DENSEST member, not its median: a family that the
    # model splits into several blobs (RuBisCO, the histones, ATP synthase all
    # do) has a median that lands between them, on nothing. The densest member
    # is the one with the most family neighbours within two thirds of the story
    # radius (tight enough to tell blobs apart, wide enough to weigh a whole
    # blob rather than its densest speck).
    family_pos = positions[named].astype(np.float64)
    seed = _densest_member(family_pos, story.radius * 2.0 / 3.0)
    dist = np.linalg.norm(positions - seed, axis=1)
    members = named & (dist <= story.radius)
    indices = np.flatnonzero(members)
    if len(indices) == 0:
        raise ValueError(f"story {story.key!r}: no member within radius {story.radius}")
    # Re-centre on the blob itself (the family median can be pulled by
    # stragglers) and measure its framing radius.
    centre = np.median(positions[indices], axis=0)
    r95 = float(np.percentile(np.linalg.norm(positions[indices] - centre, axis=1), 95))
    return StoryCluster(indices=indices, centre=centre, r95=r95, n_named=n_named)


def icosphere(subdivisions: int = SPHERE_SUBDIVISIONS) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere: ``(vertices (V, 3) float32, faces (F, 3) uint32)``.

    Loop-subdivides an icosahedron ``subdivisions`` times, re-projecting each
    midpoint onto the unit sphere, so the vertex positions double as outward
    unit normals (what the marker's smooth shading needs). Closed manifold:
    ``V = 10·4ⁿ + 2``, ``F = 20·4ⁿ``.
    """
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    for _ in range(subdivisions):
        vlist = [v for v in verts]
        midpoint: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            idx = midpoint.get(key)
            if idx is None:
                m = (vlist[a] + vlist[b]) / 2.0
                m /= np.linalg.norm(m)
                vlist.append(m)
                idx = len(vlist) - 1
                midpoint[key] = idx
            return idx

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(vlist, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)
    return verts.astype(np.float32), faces.astype(np.uint32)


def story_camera(
    cluster: StoryCluster,
    story: Story,
    global_centre: np.ndarray,
) -> CameraConfig:
    """Compose the waypoint pose for a cluster.

    The camera looks at the cluster centre from the OUTSIDE of the cloud
    (along the ray from the global centre through the cluster, lifted a little
    so the shot is not dead level), at a distance proportional to the blob's
    framing radius. Under auto-rotate only the target and distance are used —
    the turntable keeps its own direction — so the direction here matters
    exactly when the visitor has stopped the spin.
    """
    outward = cluster.centre - global_centre
    norm = float(np.linalg.norm(outward))
    if norm < 1e-6:
        outward = np.array([0.0, 0.35, 1.0])
    else:
        outward = outward / norm
        outward = outward + np.array([0.0, 0.35, 0.0])
        outward = outward / np.linalg.norm(outward)
    distance = max(story.min_distance, story.distance_per_radius * cluster.r95)
    position = cluster.centre + outward * distance
    return CameraConfig(
        position=tuple(float(v) for v in position),
        target=tuple(float(v) for v in cluster.centre),
        up=(0.0, 1.0, 0.0),
        fov=CINEMATIC_FOV_DEG,
    )


def story_panel_html(story: Story, n_members: int, index: int, total: int) -> str:
    """The right-hand story panel: title, subtitle, facts, open question."""
    r, g, b = (int(round(v * 255)) for v in story.color)
    colour = f"#{r:02x}{g:02x}{b:02x}"
    items = "".join(
        f'<li style="margin-bottom:0.7vh">{html.escape(f)}</li>' for f in story.facts
    )
    # Sized in vh so the panel scales with the display; at 1.3vh body text five
    # facts plus the question fit a 16:9 screen with room to spare and a square
    # window without spilling (checked in the browser).
    return (
        '<div style="font-size:1.3vh;line-height:1.35;color:#e8e8e8;'
        "background:rgba(0,0,0,0.62);padding:1.4vh 1.6vh;border-radius:6px;"
        f'border-left:0.5vh solid {colour}">'
        f'<div style="font-size:1.0vh;color:#aaa;letter-spacing:0.15em;'
        f'text-transform:uppercase">Story {index} of {total}</div>'
        f'<div style="font-size:2.2vh;font-weight:bold;color:{colour};'
        f'margin:0.3vh 0 0.2vh">{html.escape(story.title)}</div>'
        f'<div style="font-size:1.25vh;color:#bbb;margin-bottom:1.0vh">'
        f"{html.escape(story.subtitle)}</div>"
        f'<ul style="margin:0 0 0.8vh 1.4vh;padding:0">{items}</ul>'
        f'<div style="font-style:italic;color:#ffd48a;border-top:1px solid '
        f'rgba(255,255,255,0.15);padding-top:0.8vh">'
        f"Open question: {html.escape(story.mystery)}</div>"
        f'<div style="font-size:1.0vh;color:#888;margin-top:0.8vh">'
        f"{n_members:,} proteins highlighted</div>"
        "</div>"
    )


def overview_panel_html(n_proteins: int) -> str:
    """The overview panel shown at story 0."""
    return (
        '<div style="font-size:1.3vh;line-height:1.35;color:#e8e8e8;'
        "background:rgba(0,0,0,0.62);padding:1.4vh 1.6vh;border-radius:6px;"
        'border-left:0.5vh solid #ffffff">'
        f'<div style="font-size:2.2vh;font-weight:bold;margin-bottom:0.6vh">'
        f"{html.escape(OVERVIEW_TITLE)}</div>"
        f"{OVERVIEW_HTML.format(n=n_proteins)}"
        "</div>"
    )


# =============================================================================
# Scene construction
# =============================================================================


def load_landscape_cache(cache_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load the base demo's cached UMAP positions and metadata, or explain."""
    umap_cache = cache_dir / "umap3d_esmc_300m_all.npz"
    metadata_cache = cache_dir / "metadata_all.npz"
    missing = [p for p in (umap_cache, metadata_cache) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "This demo reads the ESM3 landscape demo's cache and never rebuilds "
            "it. Missing:\n  " + "\n  ".join(str(p) for p in missing) + "\n"
            "Run `luxar demo run esm3_protein_landscape -- --no-serve` once "
            "(needs the Swiss-Prot download and a CUDA GPU, or a precomputed "
            "embeddings cache) and try again."
        )
    positions = np.load(umap_cache)["positions"].astype(np.float32)
    meta = np.load(metadata_cache, allow_pickle=True)
    n = len(positions)
    for key in ("names", "organisms", "kingdoms"):
        if len(meta[key]) != n:
            raise ValueError(
                f"cache mismatch: {key} has {len(meta[key])} rows, positions {n}"
            )
    fields = {k: meta[k] for k in ("names", "organisms", "kingdoms")}
    fields["accessions"] = (
        meta["accessions"] if "accessions" in meta.files else np.array([], dtype=object)
    )
    return positions, fields


def build_stories_scene(
    output_path: Path,
    positions: np.ndarray,
    meta: dict[str, np.ndarray],
    *,
    stories: tuple[Story, ...] = STORIES,
    auto_rotate: bool = True,
    turntables: bool = True,
    turntable_cache: Path | None = None,
) -> int:
    """Write the stories scene. Returns the number of proteins in the backdrop."""
    n = len(positions)
    names = meta["names"]
    organisms = meta["organisms"]
    kingdoms = meta["kingdoms"]

    # Representative structures, ray-traced once and cached. PyMOL is a
    # special-case dependency (not pip-installable): when it or ffmpeg is
    # missing this prints the install hint and the scene simply has no
    # turntables. `--no-turntables` skips the step outright.
    assets: dict[str, TurntableAssets] = {}
    if turntables:
        with asection("Rendering PDB turntables"):
            cache = turntable_cache or (
                Path.home() / ".cache" / "luxar" / TURNTABLE_CACHE
            )
            assets = render_turntables(
                [s.pdb_id for s in stories if s.pdb_id],
                cache,
                colors={s.pdb_id: s.color for s in stories if s.pdb_id},
            )
            aprint(
                f"{len(assets)} of {sum(1 for s in stories if s.pdb_id)} turntables ready"
            )

    with asection("Resolving stories against the map"):
        clusters = [
            select_story_members(s, names, kingdoms, positions) for s in stories
        ]
        for s, c in zip(stories, clusters):
            aprint(
                f"{s.key}: {len(c.indices):,} of {c.n_named:,} named proteins "
                f"within {s.radius} of the blob; centre={c.centre.round(2)} "
                f"r95={c.r95:.2f}"
            )

    with asection("Building layers"):
        taxon_rgb = np.array(
            [TAXON_COLORS.get(str(k), TAXON_COLORS["Other"]) for k in kingdoms],
            dtype=np.float32,
        )
        labels = [f"{names[i]} — {organisms[i]} ({kingdoms[i]})" for i in range(n)]
        accessions = [str(a) for a in meta["accessions"]]
        linkable = _linkable_accessions(accessions, n) if accessions else None
        keys = linkable if linkable is not None else [str(s) for s in names]
        link_attrs = _uniprot_link_attrs(keys, exact_accessions=linkable is not None)

        # Backdrop: every protein once, story coordinate 0, extend_to_all.
        backdrop_positions = np.column_stack(
            [np.zeros(n, dtype=np.float32), positions]
        ).astype(np.float32)

    with asection("Composing waypoints"):
        global_centre = positions.mean(axis=0)
        spatial_max = float(np.abs(positions).max())
        overview_distance = spatial_max * 2.2
        overview_position = pull_in(
            (overview_distance, overview_distance * 0.55, overview_distance)
        )
        waypoints = [
            Waypoint(
                when={STORY_DIM: 0},
                camera=CameraConfig(
                    position=overview_position, target=(0.0, 0.0, 0.0), up=(0, 1, 0)
                ),
                duration_ms=3000,
            )
        ]
        for k, (s, c) in enumerate(zip(stories, clusters), start=1):
            waypoints.append(
                Waypoint(
                    when={STORY_DIM: k},
                    camera=story_camera(c, s, global_centre),
                    duration_ms=s.flight_ms,
                )
            )

        viewer_config = ViewerConfig(
            cinematic_mode=True,
            camera=CameraConfig(position=overview_position, target=(0.0, 0.0, 0.0)),
            # The turntable is the point: a story step keeps the spin and moves
            # what it spins around (keep-orientation flight). One turn every
            # two minutes reads as alive without smearing the highlight.
            auto_rotate=auto_rotate,
            auto_rotate_speed=0.5 if auto_rotate else None,
            # A fixed world axis: the camera-frame default precesses when the
            # opening shot looks down at the cloud (spin plus wobble), which
            # over minutes wanders the camera under the map.
            auto_rotate_axis="world-y" if auto_rotate else None,
            waypoints=waypoints,
        )

    dims = Dimensions(
        [
            Dimension(
                STORY_DIM,
                unit="",
                categories=["Overview", *(s.key for s in stories)],
                display=False,
                description="Guided tour: overview, then one protein-family story per step",
            ),
            Dimension("x", unit="UMAP", display=True),
            Dimension("y", unit="UMAP", display=True),
            Dimension("z", unit="UMAP", display=True),
        ]
    )

    with asection("Writing to Zarr"):
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                citation=DEMO_META["citation"],
                viewer_config=viewer_config,
            )

            scene.add_points(
                "Backdrop",
                backdrop_positions,
                colors=taxon_rgb,
                radii=np.full(n, BACKDROP_RADIUS, dtype=np.float32),
                sharpness=np.full(n, 0.6, dtype=np.float32),
                opacity=BACKDROP_OPACITY,
                intensity=BACKDROP_INTENSITY,
                labels=labels,
                **link_attrs,
                extend_to_all=[STORY_DIM],
                layer=True,
                additive_lod=stream_ladder(
                    n, slices=hidden_axis_stops(backdrop_positions, dims.non_displayed)
                ),
            )

            for k, (s, c) in enumerate(zip(stories, clusters), start=1):
                idx = c.indices
                m = len(idx)
                highlight_positions = np.column_stack(
                    [np.full(m, float(k), dtype=np.float32), positions[idx]]
                ).astype(np.float32)
                scene.add_points(
                    f"Story {k}: {s.key}",
                    highlight_positions,
                    colors=np.broadcast_to(
                        np.asarray(s.color, dtype=np.float32), (m, 3)
                    ).copy(),
                    radii=np.full(m, HIGHLIGHT_RADIUS, dtype=np.float32),
                    sharpness=np.full(m, 0.7, dtype=np.float32),
                    opacity=0.95,
                    intensity=HIGHLIGHT_INTENSITY,
                    labels=[labels[i] for i in idx],
                    keys=[keys[i] for i in idx],
                    link=link_attrs.get("link"),
                    layer=True,
                    # A highlight shares every position with its backdrop twin,
                    # so in one depth band the two z-fight and the backdrop
                    # (drawn later) hides it — checked in the browser: hiding
                    # the backdrop made the cluster appear. Bands are hard
                    # (LAYER_ORDER_SPEC §2): the stories always composite on top.
                    layer_order=10,
                )

            # Marker shells: one translucent sphere per cluster, pinned to its
            # story slot. The unit icosphere's vertices are its normals.
            unit_verts, unit_faces = icosphere()
            for k, (s, c) in enumerate(zip(stories, clusters), start=1):
                radius = max(SPHERE_MIN_RADIUS, SPHERE_RADIUS_SCALE * c.r95)
                nv = len(unit_verts)
                sphere_vertices = np.column_stack(
                    [
                        np.full(nv, float(k), dtype=np.float32),
                        (unit_verts * radius + c.centre.astype(np.float32)).astype(
                            np.float32
                        ),
                    ]
                ).astype(np.float32)
                rgba = np.empty((nv, 4), dtype=np.float32)
                rgba[:, :3] = np.asarray(s.color, dtype=np.float32)
                rgba[:, 3] = SPHERE_ALPHA
                scene.add_mesh(
                    f"Marker {k}: {s.key}",
                    sphere_vertices,
                    unit_faces,
                    normals=unit_verts,
                    normal_dims=[1, 2, 3],
                    colors=rgba,
                    shading="smooth",
                    double_sided=True,
                    blending_mode="additive",
                    # A low ambient floor keeps the shell continuous while the
                    # view-anchored key still gives it a soft 3D gradient; the
                    # highlight is the one curvature cue the veil needs.
                    ambient=0.3,
                    shade_exponent=1.5,
                    specular=0.3,
                    shininess=12,
                    layer=True,
                    layer_order=SPHERE_LAYER_ORDER,
                )

            # Title (constant)
            scene.add_text(
                "ESM Protein Stories — Swiss-Prot",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )

            # Hover label (centre-left), as in the base demo family.
            scene.add_text(
                "{hover_label}",
                position=(0.02, 0.5),
                anchor="center-left",
                font_size=0.02,
                color="white",
                background="rgba(0,0,0,0.72)",
                padding=0.01,
                text_align="left",
                transition="fade",
                transition_duration=0.15,
                hover=True,
            )

            # Story panels (right), one per slot.
            scene.add_html(
                overview_panel_html(n),
                position=(0.98, 0.5),
                anchor="center-right",
                width=PANEL_WIDTH,
                visible_range={STORY_DIM: 0},
                transition="fade",
                transition_duration=0.35,
            )
            total = len(stories)
            for k, (s, c) in enumerate(zip(stories, clusters), start=1):
                scene.add_html(
                    story_panel_html(s, len(c.indices), k, total),
                    position=(0.98, 0.5),
                    anchor="center-right",
                    width=PANEL_WIDTH,
                    visible_range={STORY_DIM: k},
                    transition="fade",
                    transition_duration=0.35,
                )

            # Left: the representative structure turning at 60 fps, transparent
            # over the map, with its PDB caption above. Hidden turntables are
            # paused by the viewer, so ten videos cost one decode at a time.
            for k, s in enumerate(stories, start=1):
                a = assets.get(s.pdb_id.upper()) if s.pdb_id else None
                if a is None:
                    continue
                scene.add_video(
                    a.webm,
                    position=TURNTABLE_POSITION,
                    anchor="center-left",
                    size=(TURNTABLE_WIDTH, None),
                    poster=a.poster,
                    visible_range={STORY_DIM: k},
                    transition="fade",
                    transition_duration=0.35,
                )
                scene.add_text(
                    f"PDB {a.pdb_id} · {a.title}",
                    position=TURNTABLE_CAPTION_POSITION,
                    anchor="bottom-left",
                    font_size=0.013,
                    width=TURNTABLE_WIDTH,
                    color="rgba(255,255,255,0.7)",
                    visible_range={STORY_DIM: k},
                    transition="fade",
                    transition_duration=0.35,
                )

            scene.add_text(
                "← [  •  ] →   step through the stories  (press 1 "
                "first to select the story slider)",
                position=(0.02, 0.97),
                font_size=0.014,
                anchor="bottom-left",
                color="#ffcc44",
            )

            add_demo_caption(
                scene,
                f"{n:,} proteins • ESM C embeddings • 3D UMAP • {len(stories)} stories",
                DEMO_META.get("citation"),
            )

    aprint(f"✓ Wrote {n:,} proteins and {len(stories)} stories to {output_path}")
    return n


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    """Load the landscape cache, build the stories scene, optionally serve it."""
    aprint("=" * 70)
    aprint("ESM PROTEIN STORIES — ten guided tours of the Swiss-Prot landscape")
    aprint("=" * 70)

    auto_rotate = "--no-auto-rotate" not in sys.argv
    turntables = "--no-turntables" not in sys.argv
    cache_dir = Path.home() / ".cache" / "luxar" / "esm3_swissprot"
    try:
        positions, meta = load_landscape_cache(cache_dir)
    except (FileNotFoundError, ValueError) as e:
        aprint(f"\nError: {e}")
        sys.exit(1)
    aprint(f"✓ Loaded {len(positions):,} proteins from {cache_dir}")

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "esm3_protein_stories.luxar.zarr"
        n = build_stories_scene(
            output_path, positions, meta, auto_rotate=auto_rotate, turntables=turntables
        )
        aprint(f"Dataset generated at {output_path} ({n:,} proteins)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_esm3_stories_") as tmpdir:
        output_path = Path(tmpdir) / "esm3_protein_stories.luxar.zarr"
        n = build_stories_scene(
            output_path, positions, meta, auto_rotate=auto_rotate, turntables=turntables
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("  Press '1' to select the STORY slider, then '[' / ']' to step.")
        aprint("  Each step flies the camera to a protein family and shows its story.")
        aprint(f"  Total proteins: {n:,}")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
