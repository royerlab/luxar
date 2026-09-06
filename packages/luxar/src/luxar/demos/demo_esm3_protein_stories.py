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
    python -m luxar.demos.demo_esm3_protein_stories --no-audio

Sound (``docs/guides/specs/SOUND_SPEC.md``): a CC0 ambient bed plays under the
whole tour and each story is narrated on arrival. Narration is synthesised when
the scene is BUILT — OpenAI TTS when ``OPENAI_API_KEY`` is set, the macOS
system voice otherwise, silence (with a warning) on a box with neither — and
cached under ``~/.cache/luxar/esm3_protein_stories/narration``. ``--no-audio``
skips the whole layer (no download, no synthesis).

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
        # The CC0 ambient bed (OpenGameArt "Calm Ambient 1", ~6 MiB); narration
        # is synthesised locally, never downloaded.
        "download_mb": 7,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    # Reads the base demo's cache; adds its own for the PDB turntables and
    # for the sound layer (the cached ambient bed + synthesised narration).
    "caches": ["esm3_swissprot", "pdb_turntables", "esm3_protein_stories"],
    "outputs": ["esm3_protein_stories"],
    "citation": {
        "short": "UniProt/Swiss-Prot; embeddings by EvolutionaryScale ESM C, 2024",
        "ref": "UniProt / EvolutionaryScale 2024",
        "license": "CC BY 4.0",
    },
}

import html
import math
import re
import sys
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import AudioConfig, CameraConfig, ViewerConfig, Waypoint
from luxar.demos import add_demo_caption, cached_download, launch_viewer
from luxar.demos._audio_synth import synthesise_foa_from_clip, synthesise_hum
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG, pull_in
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.demos._narration import resolve_engine, synthesise
from luxar.demos._pdb_turntable import TurntableAssets, render_turntables
from luxar.demos.demo_esm3_protein_landscape import (
    TAXON_COLORS,
    _linkable_accessions,
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
    #: Framing: the blob (diameter 2·r95) spans this fraction of the frame
    #: height under the 63° cinematic lens, the rest of the landscape giving
    #: context; the camera distance follows from the lens (see
    #: ``_story_camera``) but never drops below ``min_distance``.
    frame_fraction: float = 0.36
    min_distance: float = 5.0
    flight_ms: int = 2500
    tags: tuple[str, ...] = field(default_factory=tuple)
    #: Representative PDB entry rendered as the left-hand turntable ("" = none).
    pdb_id: str = ""
    #: What the narrator SAYS on arrival: a short spoken script, not the panel
    #: read aloud — the good facts, punchier, no "Story N of 10", ending on the
    #: open question. Consumed by the sound layer's narration builder.
    narration: str = ""


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
        narration=(
            "Hemoglobin, the molecule of breath. Each red blood cell carries "
            "some two hundred and eighty million of these, and each one holds "
            "four oxygens. In 1949 Linus Pauling showed that sickle-cell "
            "anaemia comes from a single swapped amino acid: the first "
            "molecular disease. Max Perutz needed twenty-two years to see its "
            "shape. And yet hemoglobin also turns up inside dopamine neurons, "
            "nowhere near blood. What it does there, nobody quite knows."
        ),
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
        narration=(
            "Photosystem II, the protein that made the sky breathable. Its D1 "
            "subunit sits at the heart of the only enzyme known that splits "
            "water. Cyanobacteria running this machine filled Earth's air with "
            "oxygen, two and a half billion years ago. The chemistry is so "
            "violent that D1 wrecks itself every couple of hours; a leaf "
            "rebuilds it all day long. Molecular clocks say water-splitting is "
            "far older than the rise of oxygen. So why did the planet wait so "
            "long to change?"
        ),
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
        narration=(
            "Hsp70, the oldest job in the cell. It holds unfolded proteins, "
            "refolds the damaged ones, and hands the hopeless ones to the "
            "shredder. After three billion years apart, the human and E. coli "
            "versions are still nearly half identical, letter for letter. That "
            "is why this cluster mixes bacteria, plants and animals. Cancer "
            "cells over-produce it to survive their own chaos, and drugs against "
            "it have been tried for decades. None has reached the clinic. Why "
            "is such a universal protein so hard to target?"
        ),
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
        frame_fraction=0.72,
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
        narration=(
            "The intruders' continent. Influenza's haemagglutinin, the "
            "coronavirus spike, HIV's envelope: unrelated viruses, one trick. "
            "Each snaps into a bundle that drags virus and cell together. The "
            "1918 flu killed fifty million people with a protein like this one. "
            "They gather here although they share no ancestor; the model groups "
            "them by how they are built and what they do. Most viral proteins "
            "have no known relatives at all. Where would that dark matter land "
            "on this map?"
        ),
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
        narration=(
            "A protein that is its own pathogen. Kuru, scrapie, "
            "Creutzfeldt-Jakob disease: brain diseases that spread like "
            "infections, yet no virus was ever found. In 1982 Stanley Prusiner "
            "proposed the heresy that the agent is a misfolded protein, one "
            "that converts healthy copies into itself. Mad cow disease later "
            "proved it could cross species through food. Every mammal carries "
            "the healthy form on its neurons. Forty years on, we still do not "
            "know what it is for."
        ),
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
        narration=(
            "ATP synthase, the turbine in every cell. Protons flowing through "
            "it turn an axle, and each turn presses out three molecules of "
            "ATP. In 1997 a single motor was filmed spinning under a "
            "microscope. You make and spend roughly your own body weight in "
            "ATP every day. It is one of the most efficient motors known, "
            "wasting almost nothing as heat. How a protein manages that is "
            "still debated."
        ),
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
        narration=(
            "RuBisCO, the most abundant enzyme on Earth, and one of the "
            "slowest. Nearly every carbon atom in every living thing has passed "
            "through it. It fixes about one CO2 every thirty seconds and keeps "
            "confusing oxygen with carbon dioxide, so plants make it by the "
            "tonne. Three billion years of evolution never produced a fast, "
            "accurate RuBisCO. Is that a wall that cannot be climbed, or has "
            "nobody found the path?"
        ),
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
        narration=(
            "RecA and Rad51, the machine that mends broken DNA. It coats a "
            "broken strand and searches the entire genome for the matching "
            "sequence. Our version, RAD51, is loaded by BRCA2, the protein "
            "whose mutations cause much of hereditary breast cancer. Bacteria "
            "and humans share this cluster; the shape has barely moved in "
            "billions of years. It finds one match among millions of base "
            "pairs in minutes. How it searches that fast is still argued over."
        ),
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
        narration=(
            "Insulin, a century of firsts. In January 1922 a fourteen-year-old "
            "boy received the first injection, and diabetes stopped being a "
            "death sentence. It was the first protein ever sequenced, by "
            "Frederick Sanger, and in 1982 the first drug ever made by "
            "engineered bacteria. In worms, weakening its receptor doubles "
            "lifespan. Why does a hormone for blood sugar hold a dial for "
            "ageing, and is the dial in us too?"
        ),
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
        narration=(
            "Conotoxins, venom that became medicine. Cone snails hunt with a "
            "harpoon and a cocktail of hundreds of peptides, each a precise key "
            "for one ion channel. One of them is now a drug for severe pain, "
            "the first medicine ever taken from the sea. This knot is a single "
            "superfamily; the rest are scattered across the whole map, because "
            "venom evolves faster than almost anything else. Why so fast is "
            "still being worked out."
        ),
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
# One link template for every highlight (the demo link guard wants a single
# module-level constant): a UniProt search by accession lands on the entry, a
# search by protein name on the family — whichever the cached metadata offers.
UNIPROT_LINK = "https://www.uniprot.org/uniprotkb?query={hover_key}"
PANEL_WIDTH = 0.32

# =============================================================================
# Sound layer (SOUND_SPEC.md §5): an ambient bed plus one narration per story
# =============================================================================

# A CC0 ambient pad — "Calm Ambient 1 (Synthwave 4k)" by The Cynic Project
# (cynicmusic.com), published on OpenGameArt: soft evolving pads and slow
# chords, no percussion, ~2.6 min. Chosen over a Freesound field-recording
# drone the owner found too industrial. The file is a stable direct download
# (no login), pinned by checksum so a silent swap upstream is caught.
AMBIENT_BED_URL = "https://opengameart.org/sites/default/files/001_Synthwave_4k_0.mp3"
AMBIENT_BED_FILENAME = "calm_ambient_1_cynicmusic.mp3"
AMBIENT_BED_SHA256 = "56b31f997020da1abd4092f5e82457592323717a23e4d560c5f8135d26c3b8be"
AMBIENT_BED_SOURCE_URL = "https://opengameart.org/content/calm-ambient-1-synthwave-4k"
AMBIENT_BED_ATTRIBUTION = (
    "The Cynic Project / cynicmusic.com — 'Calm Ambient 1 (Synthwave 4k)' "
    "(OpenGameArt, CC0)"
)
AMBIENT_BED_GAIN = 0.35
# Sound layer Phase 4: when a decoder + AAC encoder are on the build box, the
# stereo bed is re-encoded as a first-order ambisonic FIELD (left channel at
# +50°, right at -50°) the viewer rotates against the camera, so the music stays
# fixed to the world as the turntable spins. Otherwise the stereo MP3 plays as is.
AMBISONIC_BED_CACHE_DIR = (
    Path.home() / ".cache" / "luxar" / "esm3_protein_stories" / "ambisonic"
)
AMBISONIC_BED_SPREAD_DEG = 50.0

# Narration is synthesised at BUILD time (OpenAI TTS when a key is present, the
# macOS system voice otherwise; a Linux box without a key builds silently) and
# cached under this demo's own cache dir, keyed by (engine, voice, text).
NARRATION_CACHE_DIR = (
    Path.home() / ".cache" / "luxar" / "esm3_protein_stories" / "narration"
)
NARRATION_VOICES = {"openai": "alloy", "say": "Samantha"}
NARRATION_SOURCE_URL = "https://github.com/royerlab/luxar"
# Narration is `on_arrive`: it starts when the story's flight lands (the waypoint
# driver's arrival event), plus a beat so the picture settles first.
NARRATION_AFTER_FLIGHT_MS = 600

# Cluster hums (sound layer Phase 2): one spatial source per story, attached to
# the story's highlight node so it follows the cluster's centre, live only at
# that story, louder as the camera approaches. Synthesised at build time (needs
# afconvert or ffmpeg; otherwise the scene has no hums) and cached here.
HUM_CACHE_DIR = Path.home() / ".cache" / "luxar" / "esm3_protein_stories" / "hums"
HUM_SECONDS = 8.0
HUM_GAIN = 0.5
#: E2 as the ladder's root; each story a step up a major pentatonic scale, so
#: stepping through the tour also walks a melody.
HUM_BASE_HZ = 82.41
HUM_SEMITONES = (0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24, 26)
#: PannerNode distances relative to the cluster's framing radius: at the story
#: camera (about nine radii out, inverse model) the hum sits near -13 dB.
HUM_REF_DISTANCE_PER_R95 = 2.0
HUM_MAX_DISTANCE_PER_R95 = 40.0

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
# PyMOL into a transparent WebM turning once in 30 s (see `_pdb_turntable`). Sits at panel
# height, clear of the activity rail; the caption goes just above it.
TURNTABLE_POSITION = (0.06, 0.5)
TURNTABLE_WIDTH = 0.26  # viewport-width fraction; height follows the square video
# Just under the square clip: on a 16:9 display a 26 vw square is ~46 vh tall,
# so its lower edge sits near y = 0.73 when centred at 0.5.
# Centred under the clip: the clip is anchored centre-left at x = 0.06 and is
# TURNTABLE_WIDTH wide, so its centre (and the structure's, which the renderer
# fits to the frame's bounding sphere) sits at x = 0.06 + TURNTABLE_WIDTH / 2.
TURNTABLE_CAPTION_POSITION = (0.06 + 0.26 / 2, 0.75)
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


def _midpoint_vertex(
    a: int,
    b: int,
    vertices: list[np.ndarray],
    midpoint_indices: dict[tuple[int, int], int],
) -> int:
    key = (a, b) if a < b else (b, a)
    idx = midpoint_indices.get(key)
    if idx is None:
        midpoint = (vertices[a] + vertices[b]) / 2.0
        midpoint /= np.linalg.norm(midpoint)
        vertices.append(midpoint)
        idx = len(vertices) - 1
        midpoint_indices[key] = idx
    return idx


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
        midpoint_indices: dict[tuple[int, int], int] = {}

        new_faces = []
        for a, b, c in faces:
            ab = _midpoint_vertex(a, b, vlist, midpoint_indices)
            bc = _midpoint_vertex(b, c, vlist, midpoint_indices)
            ca = _midpoint_vertex(c, a, vlist, midpoint_indices)
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
    # Compose for the cinematic lens (cinematic mode sets it; the pose leaves
    # fov unset): the distance at which a blob of radius r95 spans
    # `frame_fraction` of the frame height under a 63° vertical field of view.
    half_height_per_unit = math.tan(math.radians(CINEMATIC_FOV_DEG) / 2)
    distance = max(
        story.min_distance,
        cluster.r95 / (0.5 * story.frame_fraction * half_height_per_unit),
    )
    position = cluster.centre + outward * distance
    return CameraConfig(
        position=tuple(float(v) for v in position),
        target=tuple(float(v) for v in cluster.centre),
        up=(0.0, 1.0, 0.0),
    )


#: Spoken introduction at the Overview slot (the sound layer's narration).
OVERVIEW_NARRATION = (
    "Every point here is a protein: five hundred and seventy-five thousand of "
    "them, placed by a language model that reads their sequences, so that "
    "similar proteins sit close together. Ten of these clusters hide a story. "
    "Step through them."
)


def story_narration(story: Story) -> str:
    """The spoken script for a story: its authored ``narration``, or — for a
    story without one — the title and the open question, never the whole panel.
    """
    if story.narration.strip():
        return story.narration.strip()
    return f"{story.title}. {story.mystery}"


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


def hum_frequency_hz(story_index: int) -> float:
    """Pitch of story ``story_index`` (1-based) on the pentatonic ladder."""
    semitone = HUM_SEMITONES[(story_index - 1) % len(HUM_SEMITONES)]
    return HUM_BASE_HZ * 2.0 ** (semitone / 12.0)


def add_story_sounds(
    scene: object,
    stories: tuple[Story, ...],
    *,
    narration_dir: Path = NARRATION_CACHE_DIR,
    engine: str | None = None,
    clusters: Sequence[StoryCluster] | None = None,
    hum_dir: Path = HUM_CACHE_DIR,
    ambisonic_dir: Path = AMBISONIC_BED_CACHE_DIR,
) -> int:
    """Add the ambient bed, one narration per story slot and one hum per cluster.

    Returns the node count. The bed is a cached CC0 download; a network failure
    skips it with a warning rather than failing the build. Narration is
    synthesised through :func:`luxar.demos._narration.synthesise`; when no
    engine is available the stories stay silent (the helper has already warned).
    Hums need ``clusters`` (for the centre and framing radius) and an AAC encoder
    (:mod:`luxar.demos._audio_synth`); without one there are simply no hums.
    """
    added = 0
    try:
        bed = cached_download(
            AMBIENT_BED_URL,
            "esm3_protein_stories",
            AMBIENT_BED_FILENAME,
            sha256=AMBIENT_BED_SHA256,
        )
    except Exception as e:  # noqa: BLE001 - offline build must still produce the scene
        aprint(f"⚠️ Ambient bed unavailable ({e}); building without it")
        bed = None
    if bed is not None:
        # The field version when the box can make one; the stereo clip otherwise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            field = synthesise_foa_from_clip(
                bed, ambisonic_dir, spread_deg=AMBISONIC_BED_SPREAD_DEG
            )
        if field is None:
            aprint("🔈 No audio decoder/encoder: the bed stays stereo")
        scene.add_sound(  # type: ignore[attr-defined]
            "bed_ambient",
            field if field is not None else bed,
            ambisonic="foa" if field is not None else None,
            trigger="continuous",
            gain=AMBIENT_BED_GAIN,
            fade_in_ms=1500,
            fade_out_ms=1500,
            bus="ambient",
            license="CC0",
            attribution=AMBIENT_BED_ATTRIBUTION,
            source_url=AMBIENT_BED_SOURCE_URL,
            # A Layers-panel row: eye = mute the bed, slider = its gain.
            layer=True,
        )
        added += 1

    chosen = resolve_engine(engine)
    voice = NARRATION_VOICES.get(chosen or "", "")
    # The spoken scripts are authored with the stories (`Story.narration`,
    # `OVERVIEW_NARRATION`): short and punchy, not the panel read aloud.
    slots: list[tuple[int, str, str]] = [(0, "Overview", OVERVIEW_NARRATION)]
    slots += [(k, s.key, story_narration(s)) for k, s in enumerate(stories, start=1)]
    for k, key, text in slots:
        clip = synthesise(text, voice, narration_dir, engine=chosen or "none")
        if clip is None:
            break
        scene.add_sound(  # type: ignore[attr-defined]
            f"narration_{key}",
            clip,
            hidden={STORY_DIM: k},
            # Starts when the story's flight lands (waypoint arrival), a beat later.
            trigger="on_arrive",
            delay_ms=NARRATION_AFTER_FLIGHT_MS,
            bus="voice",
            license="CC0",
            attribution=f"Narration synthesised at build time ({chosen}, voice {voice})",
            source_url=NARRATION_SOURCE_URL,
        )
        added += 1

    hums = 0
    if clusters is not None:
        for k, (s, c) in enumerate(zip(stories, clusters), start=1):
            clip = synthesise_hum(hum_frequency_hz(k), HUM_SECONDS, hum_dir)
            if clip is None:
                break
            scene.add_sound(  # type: ignore[attr-defined]
                f"hum_{s.key}",
                clip,
                hidden={STORY_DIM: k},
                attach_to=story_node_name(k, s),
                trigger="continuous",
                gain=HUM_GAIN,
                fade_in_ms=1200,
                fade_out_ms=1200,
                bus="effects",
                ref_distance=max(0.5, HUM_REF_DISTANCE_PER_R95 * c.r95),
                max_distance=max(20.0, HUM_MAX_DISTANCE_PER_R95 * c.r95),
                rolloff=1.0,
                license="CC0",
                attribution="Hum synthesised at build time (luxar demo)",
                source_url=NARRATION_SOURCE_URL,
                layer=True,
            )
            added += 1
            hums += 1
    bed_kind = "no" if bed is None else ("ambisonic" if field is not None else "stereo")
    aprint(
        f"🔈 {added} sound node(s): bed {bed_kind}, narration engine "
        f"{chosen}, {hums} cluster hum(s)"
    )
    return added


def story_node_name(story_index: int, story: Story) -> str:
    """Name of a story's highlight points node (what its hum attaches to)."""
    return f"Story {story_index}: {story.key}"


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
    audio: bool = True,
) -> int:
    """Write the stories scene. Returns the number of proteins in the backdrop.

    ``audio=False`` skips the sound layer (no bed download, no narration
    synthesis) — the ``--no-audio`` flag, for an offline or headless build.
    """
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
        for s, c in zip(stories, clusters, strict=True):
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
        for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
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
            # Supersampling: a kiosk-sized point cloud shimmers under the
            # turntable without it; the display is a single large screen with a
            # GPU to spare, so the cost is acceptable.
            ssaa_enabled=True,
            # The kiosk display is high-DPI and dedicated: render at its full
            # device resolution (the viewer's default caps DPR for laptops).
            allow_high_dpr=True,
            waypoints=waypoints,
            # Sound layer defaults: room speakers (equal-power), the bed under the
            # voice by 9 dB while a narration plays.
            audio=AudioConfig(
                enabled=True,
                master_gain=0.8,
                panning_model="equalpower",
                buses={"ambient": 0.6, "voice": 1.0, "effects": 0.8},
                duck_db=-9.0,
            )
            if audio
            else None,
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
                keys=keys,
                link=UNIPROT_LINK,
                copy="{hover_key}",
                extend_to_all=[STORY_DIM],
                layer=True,
                additive_lod=stream_ladder(
                    n, slices=hidden_axis_stops(backdrop_positions, dims.non_displayed)
                ),
            )

            for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
                idx = c.indices
                m = len(idx)
                highlight_positions = np.column_stack(
                    [np.full(m, float(k), dtype=np.float32), positions[idx]]
                ).astype(np.float32)
                scene.add_points(
                    story_node_name(k, s),
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
                    link=UNIPROT_LINK,
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
            for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
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
            for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
                scene.add_html(
                    story_panel_html(s, len(c.indices), k, total),
                    position=(0.98, 0.5),
                    anchor="center-right",
                    width=PANEL_WIDTH,
                    visible_range={STORY_DIM: k},
                    transition="fade",
                    transition_duration=0.35,
                )

            # Left: the representative structure turning slowly (30 s per turn), transparent
            # over the map, with its PDB caption below. Hidden turntables are
            # paused by the viewer, so ten videos cost one decode at a time.
            for k, s in enumerate(stories, start=1):
                a = assets.get(s.pdb_id.upper()) if s.pdb_id else None
                if a is None:
                    continue
                # Older RCSB entries shout their title in capitals; it stays as
                # deposited — sentence-casing mangles the acronyms (NMR, MVIIA).
                title = a.title
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
                    f"PDB {a.pdb_id} · {title}",
                    position=TURNTABLE_CAPTION_POSITION,
                    anchor="top-center",
                    text_align="center",
                    font_size=0.013,
                    width=TURNTABLE_WIDTH,
                    color="rgba(255,255,255,0.7)",
                    visible_range={STORY_DIM: k},
                    transition="fade",
                    transition_duration=0.35,
                )

            # A discreet signature in the lower-left corner: dim, letter-spaced,
            # light weight — meant to be found, not read. (The navigation hint
            # that used to sit here is redundant on a kiosk driven from outside.)
            scene.add_html(
                '<div style="font-size:1.05vh;letter-spacing:0.22em;'
                "font-weight:300;color:rgba(255,255,255,0.22);"
                'text-transform:uppercase;white-space:nowrap">'
                "Designed by Loic A. Royer</div>",
                position=(0.02, 0.97),
                anchor="bottom-left",
                interactive=False,
            )

            add_demo_caption(
                scene,
                f"{n:,} proteins • ESM C embeddings • 3D UMAP • {len(stories)} stories",
                DEMO_META.get("citation"),
            )

            if audio:
                with asection("Sound layer"):
                    add_story_sounds(scene, stories, clusters=clusters)

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
    audio = "--no-audio" not in sys.argv
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
            output_path,
            positions,
            meta,
            auto_rotate=auto_rotate,
            turntables=turntables,
            audio=audio,
        )
        aprint(f"Dataset generated at {output_path} ({n:,} proteins)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_esm3_stories_") as tmpdir:
        output_path = Path(tmpdir) / "esm3_protein_stories.luxar.zarr"
        n = build_stories_scene(
            output_path,
            positions,
            meta,
            auto_rotate=auto_rotate,
            turntables=turntables,
            audio=audio,
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
