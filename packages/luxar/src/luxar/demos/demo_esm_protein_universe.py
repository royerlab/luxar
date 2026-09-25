#!/usr/bin/env python3
"""ESM Protein Universe — twenty stories across 7.7 million protein clusters.

The big-map sibling of ``demo_esm3_protein_stories``. Every point is one of
7.7 million CLUSTERS of proteins from the ESM Atlas (Candido et al., bioRxiv
2026): 6.8 billion sequences from eight public databases — 5.6 billion of them
from metagenomes alone, read straight out of soil, seawater and guts rather than
from any organism grown in a lab — grouped by the features a protein language
model (ESM C) sees in them (Jaccard ≥ 0.6 to the cluster centre in
sparse-autoencoder feature space) and laid out by 3D UMAP. A point is a cluster
of at least fifty members; the 7.7 million together stand for 817 million
proteins. A hidden ``story`` dimension walks through twenty stops in five movements
(see ``TOUR_ORDER``, which is the narrative order and deliberately not the
order the stories are authored in below): the machinery every cell runs on
(hemoglobin, photosystem II, RuBisCO, ATP synthase, Hsp70, RecA); the map's
own geography (the ABC-transporter spur flung off the cloud, the dark
proteome, the phage universe); the arms race (the viral spike, CRISPR-Cas,
the jumping-gene nuclease TnpB that the Cas12 editors grew out of and its
eukaryotic cousin Fanzor,
beta-lactamases, the lanthipeptide antibiotics); life at the edges and the
senses (ice-binding proteins, the hyperthermophile's reverse gyrase, and the
three unrelated receptor families with which vertebrates, insects and
nematodes each invented smell); and a closer, the gut tyrosine decarboxylase
that destroys the Parkinson's drug levodopa.

Most stops fly the camera to the family's densest knot, light its members,
blow a soap bubble around them, show a panel of sourced facts and a spinning
representative structure, and are narrated on arrival. Seven are framed
otherwise: three constellations (below), three that light a whole region of
the map, and one that is not a knot at all.

Three stops are CONSTELLATIONS. A family rarely occupies one spot here, and
for the globins, the photosynthetic reaction centres and the lanthipeptide
machinery the split is the story — so instead of cutting one knot, those
stops light every place the family occupies, join the places with a spanning
tree of bright lines, and pull the camera back to hold the whole figure. A
constellation has no soap bubble; the two are mutually exclusive (see
:attr:`UniverseStory.constellation`).

One stop is deliberately not a knot either. Tyrosine decarboxylase does not
form a family in this map — its 52 clusters lie in about a dozen specks
strung across a fifth of the cloud, inside a much larger family — so it is authored as a ``scatter`` story
that lights every one of them and makes the absence of a knot the point (see
:attr:`UniverseStory.scatter`).

What changes against the Swiss-Prot tour, and why:

- **Scale.** Thirteen times the points. The backdrop carries no per-point hover
  labels (seven million strings would dominate the store); the knot and
  scatter story members do, with their product name, taxon and a UniRef link.
  The three region stories light too many points to label.
- **Grain.** This map is coherent at a far finer scale than the Swiss-Prot
  one: ten nearest neighbours share a Pfam family three times out of four, but
  they sit within a couple of hundredths of a unit, and a family forms several small
  pure knots rather than one blob. Stories therefore select by **Pfam family**
  (or a product-name pattern, or a region predicate) and frame the densest
  knot, scored by member count times purity, within a radius of ~0.3 rather
  than 0.8.
- **Colour.** The backdrop is coloured by the main branch of life of each
  cluster's dominant phylum, and clusters with no characterised member at all
  — one in four; no member carries a Pfam domain of known function, the same
  count the preprint reports as "over two million" — are dimmed, so the "dark
  proteome" reads as geography before its story is told.

Dataset facts are from the preprint (Candido et al., "Language Modeling
Materializes a World Model of Protein Biology", bioRxiv 2026, doi
10.64898/2026.06.03.729735 — the atlas section and Appendix A.5); every
map-specific line in a story (what a knot IS) was re-measured against the
annotation table.

Inputs are two parquet files handed over by the ESM Atlas team (not public, so
``local_data="manual-file"``): the coordinates
(``umap_coordinates_3d.parquet``: ``cluster_rep_protein_hash``, ``umap_1..3``)
and the annotations (``representative_proteins_min50_pfam_taxa_desc_named_v3
.parquet``: ``protein_hash``, ``cluster_pct_characterized``,
``cluster_top_pfam_domains``, ``top_phyla``, ``product_name``,
``uniref_match_accession``, ...), joined on the hash (MD5 of the representative
sequence). Drop them in ``~/.cache/luxar/esm_protein_universe/`` (or
``~/Downloads``, or pass ``--coords`` / ``--annotations``); the first run folds
them into a compact cache and every later run starts from that.

Usage:
    python -m luxar.demos.demo_esm_protein_universe
    python -m luxar.demos.demo_esm_protein_universe --no-serve
    python -m luxar.demos.demo_esm_protein_universe --no-audio --no-turntables
    python -m luxar.demos.demo_esm_protein_universe --high-quality   # kiosk: SSAA + full DPR + 95% dolly
    python -m luxar.demos.demo_esm_protein_universe --coords X.parquet --annotations Y.parquet

Touch panel (off by default):
    A kiosk can be driven from a tablet — a full-screen matrix of one tile per
    story, derived from the tour itself. ``--control`` exposes it; loopback
    alone is not reachable from a tablet, so a real kiosk also needs
    ``--host 0.0.0.0`` and, because that opens the display to the network, a
    ``--control-token``. The command prints both URLs.

    python -m luxar.demos.demo_esm_protein_universe --control
    python -m luxar.demos.demo_esm_protein_universe --control --host 0.0.0.0 --control-token SECRET
"""

from __future__ import annotations

DEMO_META = {
    "key": "esm_protein_universe",
    "title": (
        "ESM Protein Universe — twenty stories across 7.7 million protein clusters"
    ),
    "description": (
        "The ESM Atlas cluster map (Candido et al. 2026): a 3D UMAP of 7.7 "
        "million protein clusters drawn from 6.8 billion sequences, most of them "
        "environmental, with a hidden story dimension: "
        "twenty stops in a narrative order — six classic families, then the "
        "ABC-transporter spur, the dark proteome and the phage universe, then "
        "the arms race from viral surface proteins through CRISPR-Cas and the "
        "jumping-gene nucleases it grew out of to beta-lactamases and the "
        "lanthipeptide antibiotics, then life in ice and in boiling water and "
        "three separate inventions of smell, and last a gut enzyme that eats "
        "a Parkinson's drug — each with "
        "a fly-to waypoint, a highlight, a turning structure and a narrated "
        "panel of researched facts. Most are framed close on one knot inside "
        "a soap bubble; three are framed wide as constellations, where the "
        "places one family occupies are lit and joined by lines."
    ),
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        # The CC0 ambient bed (~6 MiB, shared with the Swiss-Prot tour) plus
        # ~144 MiB of RCSB assemblies when the turntable renderer is present
        # (115 MiB of it the P68 virion); the two parquet inputs (~900 MB)
        # are a manual hand-off, not a download.
        "download_mb": 150,
        "compute": "medium",
        "gpu": "none",
        "local_data": "manual-file",
    },
    # Its own cache (the folded parquet inputs, narration, ambisonic bed), the
    # shared turntable cache, and the Swiss-Prot tour's cache dir, where the
    # shared ambient-bed download lives (`add_story_sounds` fetches it there).
    "caches": ["esm_protein_universe", "pdb_turntables", "esm3_protein_stories"],
    "outputs": ["esm_protein_universe"],
    "citation": {
        "short": "ESM Atlas cluster map (Candido et al. 2026)",
        "ref": "ESM Atlas, Candido et al. 2026",
        "doi": "10.64898/2026.06.03.729735",
        "url": "https://biohub.ai/esm/protein/atlas",
        # What governs OUR data is the permission, not a public licence: the 3D
        # coordinates and the per-cluster annotation table are a hand-off from
        # the ESM Atlas team, not the published release (CC BY 4.0, with CC
        # BY-SA 4.0 on its sequence-bearing subsets), so the footer names only
        # the permission we actually have.
        "license": "3D map shown with permission",
    },
}

import html
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.group.compositing import position_bounds_from_array
from luxar.core.group.lod.group import partitioned_coverage_fractions
from luxar.core.group.partition import bsp_leaf_parts, spatial_bsp_tree
from luxar.core.viewer_config import (
    AudioConfig,
    CameraConfig,
    Chapter,
    ControlPanelConfig,
    EnvironmentConfig,
    ViewerConfig,
    Waypoint,
)
from luxar.demos import (
    bake_scene_environment,
    control_serve_args,
    launch_viewer,
    parse_path_arg,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG, pull_in
from luxar.demos._dependencies import require_module
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.demos._pdb_turntable import TurntableAssets, render_turntables
from luxar.demos.demo_esm3_protein_landscape import TAXON_COLORS
from luxar.demos.demo_esm3_protein_stories import (
    BIOHUB_LOGO,
    BIOHUB_LOGO_WIDTH,
    BUBBLE_DISPERSION,
    BUBBLE_IOR,
    BUBBLE_IRIDESCENCE,
    BUBBLE_REFRACT_DATA,
    BUBBLE_ROUGHNESS,
    BUBBLE_THICKNESS_FRAC,
    BUBBLE_TINT,
    PANEL_WIDTH,
    SPHERE_LAYER_ORDER,
    STORY_DIM,
    TURNTABLE_CACHE,
    TURNTABLE_CAPTION_POSITION,
    TURNTABLE_POSITION,
    TURNTABLE_WIDTH,
    Story,
    StoryCluster,
    _turntable_environment,
    add_story_sounds,
    bubble_radius,
    cluster_geometry,
    icosphere,
    story_camera,
    story_camera_distance,
    story_node_name,
    story_panel_html,
)
from luxar.demos.demo_esm3_protein_stories import (
    STORIES as SWISSPROT_STORIES,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Inputs and cache
# =============================================================================

CACHE_DIR = Path.home() / ".cache" / "luxar" / "esm_protein_universe"
COORDS_PARQUET = "umap_coordinates_3d.parquet"
ANNOTATIONS_PARQUET = "representative_proteins_min50_pfam_taxa_desc_named_v3.parquet"
#: The folded inputs: positions + the per-cluster columns every build needs.
#: Bump the suffix when the layout below changes.
UNIVERSE_CACHE = "universe_v3.npz"
#: Points farther than this from the cloud's median are UMAP outliers (0.12% of
#: the rows, some 60 units out) that would otherwise set the scene's extent.
CULL_RADIUS = 40.0
#: Beyond this radius from the centre lies the spur (see the ABC story) — but
#: not only the spur: a few hundred stragglers sit that far out in other
#: directions, and they would drag the story's bounding-box centre off the
#: streak (from [3, 5, -30] to [-9, -2, -15]). So the spur is the far points
#: within ``SPUR_HALF_ANGLE_DEG`` of the far set's mean direction, computed
#: from the data at build time.
SPUR_RADIUS = 22.0
SPUR_HALF_ANGLE_DEG = 15.0
#: The spur story is a GEOMETRY predicate whose panel makes a FAMILY claim, so
#: the build checks that at least ``SPUR_PFAM_MIN_FRACTION`` of the lit
#: clusters have an ABC-transporter Pfam family as their dominant domain; a
#: different Atlas release could move the spur. Measured on this release:
#: 8,277 clusters on the spur, 96% of them ABC parts — the ATP-binding cassette
#: itself (PF00005), AAA_21 (PF13304: named for AbiEii toxins, but on the
#: spur 90% of its clusters are named ABC transporter ATP-binding proteins) or the ABC
#: membrane domain (PF00664). Taking every point beyond r=22 instead gives
#: 9,196 at 86%: the difference is the stragglers.
SPUR_PFAMS = ("PF00005", "PF13304", "PF00664")
SPUR_PFAM_MIN_FRACTION = 0.9

# The annotation columns the cache folds in (the rest are read per story member
# at build time).
_ANNOTATION_COLUMNS = (
    "protein_hash",
    "cluster_pct_characterized",
    "cluster_top_pfam_domains",
    "top_phyla",
)
_MEMBER_COLUMNS = ("product_name", "uniref_match_accession", "lca_taxonomy")


def find_input(name: str, explicit: Path | None) -> Path:
    """Locate a hand-delivered parquet: ``explicit``, the cache dir, Downloads."""
    candidates = [explicit] if explicit is not None else []
    candidates += [CACHE_DIR / name, Path.home() / "Downloads" / name]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    raise FileNotFoundError(
        f"{name} not found. This demo needs the two parquet files from the ESM "
        f"Atlas team; put them in {CACHE_DIR} (or ~/Downloads), or pass "
        "--coords PATH / --annotations PATH."
    )


def _dominant_map_key(table: Any, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Per row of a ``map<string, int>`` column: the key with the largest value.

    Returns ``(codes, vocab)`` with ``codes[i] == -1`` for an empty/null map.
    """
    arr = table.column(column).combine_chunks()
    offsets = arr.offsets.to_numpy()
    rows = np.repeat(np.arange(len(arr)), np.diff(offsets))
    keys = np.asarray(arr.keys.to_numpy(zero_copy_only=False))
    values = np.asarray(arr.items.to_numpy(zero_copy_only=False), dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0)
    order = np.lexsort((-values, rows))
    first = np.r_[True, rows[order][1:] != rows[order][:-1]]
    dom_rows, dom_keys = rows[order][first], keys[order][first]
    vocab, inverse = np.unique(dom_keys, return_inverse=True)
    codes = np.full(len(arr), -1, dtype=np.int32)
    codes[dom_rows] = inverse.astype(np.int32)
    return codes, vocab


def _characterized_percentages(column: Any) -> np.ndarray:
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("cluster_pct_characterized contains null or non-finite values")
    if np.any((values < 0) | (values > 100)):
        raise ValueError("cluster_pct_characterized contains values outside [0, 100]")
    return values


def build_universe_cache(coords: Path, annotations: Path, out: Path) -> Path:
    """Fold the two parquet inputs into the compact per-cluster cache.

    Centres the coordinates on their median, culls the far outliers, joins the
    annotation table on the hash and keeps the per-cluster columns every build
    needs: characterised fraction, dominant Pfam family, dominant phylum, and
    the annotation row (so a build can pull member details later).
    """
    pq = require_module("pyarrow.parquet")
    t0 = time.time()
    with asection(f"Reading {coords.name}"):
        ct = pq.read_table(coords)
        hashes = np.array(
            ct.column("cluster_rep_protein_hash").to_pylist(), dtype="S32"
        )
        xyz = np.stack(
            [ct.column(c).to_numpy() for c in ("umap_1", "umap_2", "umap_3")], axis=1
        ).astype(np.float32)
        del ct
        centre = np.median(xyz, axis=0)
        xyz -= centre
        keep = np.linalg.norm(xyz, axis=1) <= CULL_RADIUS
        aprint(
            f"{len(xyz):,} clusters; culling {int((~keep).sum()):,} beyond "
            f"radius {CULL_RADIUS:g} from the median"
        )
        xyz, hashes = xyz[keep], hashes[keep]
    with asection(f"Reading {annotations.name}"):
        at = pq.read_table(annotations, columns=list(_ANNOTATION_COLUMNS))
        ahash = np.array(at.column("protein_hash").to_pylist(), dtype="S32")
        order = np.argsort(ahash)
        pos = np.searchsorted(ahash[order], hashes)
        pos = np.minimum(pos, len(order) - 1)
        hit = ahash[order][pos] == hashes
        row = np.where(hit, order[pos], -1).astype(np.int32)
        aprint(f"{int(hit.sum()):,} of {len(hashes):,} kept clusters have annotations")
        if hit.mean() < 0.99:
            raise ValueError(
                "fewer than 99% of the coordinate rows have an annotation row — "
                "are the two parquet files from the same release?"
            )
        pct_rows = _characterized_percentages(at.column("cluster_pct_characterized"))
        pfam_rows, pfam_vocab = _dominant_map_key(at, "cluster_top_pfam_domains")
        phylum_rows, phylum_vocab = _dominant_map_key(at, "top_phyla")
        del at

    def per_point(rows: np.ndarray, fill: float) -> np.ndarray:
        out_arr = np.full(len(row), fill, dtype=rows.dtype)
        out_arr[hit] = rows[row[hit]]
        return out_arr

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        positions=xyz,
        annotation_row=row,
        # A coordinate row without an annotation is unknown, not dark. Keep a
        # neutral fill here; `dark_mask` also checks `annotation_row` explicitly.
        pct_characterized=per_point(pct_rows, np.nan),
        pfam_code=per_point(pfam_rows, -1),
        pfam_vocab=pfam_vocab.astype("U16"),
        phylum_code=per_point(phylum_rows, -1),
        phylum_vocab=phylum_vocab.astype("U64"),
    )
    aprint(f"✓ Wrote {out} in {time.time() - t0:.0f}s")
    return out


@dataclass(frozen=True)
class Universe:
    """The per-cluster arrays a build works from (all aligned to ``positions``)."""

    positions: np.ndarray
    annotation_row: np.ndarray
    pct_characterized: np.ndarray
    pfam_code: np.ndarray
    pfam_vocab: np.ndarray
    phylum_code: np.ndarray
    phylum_vocab: np.ndarray

    def __len__(self) -> int:
        return len(self.positions)

    def pfam_mask(self, ids: tuple[str, ...]) -> np.ndarray:
        """Clusters whose DOMINANT Pfam family is one of ``ids``."""
        codes = np.flatnonzero(np.isin(self.pfam_vocab, list(ids)))
        return np.isin(self.pfam_code, codes)

    def phylum_mask(self, name: str) -> np.ndarray:
        codes = np.flatnonzero(self.phylum_vocab == name)
        return np.isin(self.phylum_code, codes)

    def dark_mask(self) -> np.ndarray:
        """Clusters with not one characterised member."""
        return (self.annotation_row >= 0) & (self.pct_characterized == 0)

    def phylum_names(self) -> np.ndarray:
        """Per-cluster dominant phylum name ('' when unknown)."""
        vocab = np.append(self.phylum_vocab, "")
        return vocab[np.where(self.phylum_code < 0, len(vocab) - 1, self.phylum_code)]


def load_universe(cache: Path) -> Universe:
    data = np.load(cache)
    return Universe(**{k: data[k] for k in data.files})


# =============================================================================
# Domains of life: phylum → the landscape demo's taxon palette
# =============================================================================

# The dominant phyla covering ~97% of the annotated clusters, mapped onto the
# groups the Swiss-Prot landscape colours (so the two maps read alike). Any
# phylum ending in "viricota" is a virus; anything else unlisted is "Other".
_PHYLUM_GROUPS: dict[str, str] = {
    "Pseudomonadota": "Proteobacteria",
    "Bacillota": "Firmicutes & Actino",
    "Actinomycetota": "Firmicutes & Actino",
    "Mycoplasmatota": "Firmicutes & Actino",
    "Chordata": "Other Vertebrates",
    "Arthropoda": "Insects & Worms",
    "Nematoda": "Insects & Worms",
    "Mollusca": "Insects & Worms",
    "Streptophyta": "Plants",
    "Chlorophyta": "Plants",
    "Bacillariophyta": "Plants",
    "Haptophyta": "Plants",
    "Ascomycota": "Fungi",
    "Basidiomycota": "Fungi",
}
_OTHER_BACTERIA = frozenset(
    {
        "Bacteroidota", "Acidobacteriota", "Chloroflexota", "Planctomycetota",
        "Verrucomicrobiota", "Cyanobacteriota", "Myxococcota",
        "Thermodesulfobacteriota", "Gemmatimonadota", "Campylobacterota",
        "Spirochaetota", "Nitrospirota", "Minisyncoccota", "Fidelibacterota",
        "Candidatus Saccharimonadota", "Bdellovibrionota", "Candidatus Binatota",
        "Candidatus Omnitrophota", "Candidatus Methylomirabilota", "Fusobacteriota",
        "Armatimonadota", "Candidatus Melainabacteria", "Ignavibacteriota",
        "Deinococcota", "Thermotogota", "Aquificota", "Chlorobiota",
        "Deferribacterota", "Synergistota", "Elusimicrobiota", "Balneolota",
        "Rhodothermota", "Calditrichota", "Candidatus Eisenbacteria",
    }
)  # fmt: skip
_ARCHAEA = frozenset(
    {
        "Methanobacteriota", "Thermoplasmatota", "Nitrososphaerota",
        "Candidatus Bathyarchaeota", "Thermoproteota", "Nanobdellota",
        "Candidatus Thermoplasmatota", "Candidatus Micrarchaeota",
        "Candidatus Aenigmarchaeota", "Candidatus Lokiarchaeota",
        "Candidatus Korarchaeota", "Candidatus Woesearchaeota",
        "Candidatus Nanohaloarchaeota", "Candidatus Diapherotrites",
    }
)  # fmt: skip
#: Brightness factor for clusters with no characterised member: dim enough that
#: the dark proteome reads as a shadow on the map, bright enough to still be
#: there when a story frames one of its knots.
DARK_DIM = 0.45


def phylum_group(phylum: str) -> str:
    """The landscape palette group a dominant phylum belongs to."""
    if not phylum:
        return "Other"
    if phylum in _PHYLUM_GROUPS:
        return _PHYLUM_GROUPS[phylum]
    if phylum.endswith("viricota"):
        return "Viruses"
    if phylum in _ARCHAEA:
        return "Archaea"
    if phylum in _OTHER_BACTERIA:
        return "Other Bacteria"
    return "Other"


def backdrop_colors(universe: Universe) -> np.ndarray:
    """Per-cluster linear RGB: the domain palette, dimmed for dark clusters."""
    groups = [phylum_group(str(p)) for p in universe.phylum_vocab] + ["Other"]
    palette = np.array(
        [TAXON_COLORS.get(g, TAXON_COLORS["Other"]) for g in groups], dtype=np.float32
    )
    code = np.where(universe.phylum_code < 0, len(groups) - 1, universe.phylum_code)
    rgb = palette[code]
    rgb[universe.dark_mask()] *= DARK_DIM
    return rgb


# =============================================================================
# The stories
# =============================================================================
#
# Every number below was checked against a source when the demo was written;
# the source is named next to the fact. Counts about THIS map come from the
# annotation table itself, and the dataset facts from the ESM Atlas preprint
# (Candido et al., bioRxiv 2026, doi 10.64898/2026.06.03.729735): 6,824,676,938
# sequences from eight databases (UniParc, IMG/M, IMG/VR, two JGI MAG sets,
# MGnify, UHGG, SPIRE), with 5.6 billion from the two metagenome catalogues
# SPIRE and MGnify alone; ESMC 6B embeddings → 16,384-feature SAE → Jaccard
# ≥ 0.6 to the cluster centre → 3.05 billion clusters, of which 7.7 million
# have at least fifty members (817 million proteins; this map); 1.1 billion
# ESMFold2 structures; "characterised" means the cluster has a member carrying
# a Pfam domain (release 38.1) whose FAMILY NAME does not mark it unknown
# (DUF/UPF/"uncharacterized" — Appendix A.4.5.2's string rule, which the paper
# itself calls a reasonable estimate), and "over two million" clusters have
# none. Note the paper's other dark number, 1.5 million, is clusters with no
# Pfam family AT ALL, a stricter cut than ours. Every
# map-specific line (what a knot IS) was re-measured on 2026-09-09; the knot
# compositions quoted in the comments come from that audit.


@dataclass(frozen=True)
class UniverseStory(Story):
    """A story resolved against the cluster map rather than Swiss-Prot names.

    Members are the union of: clusters whose dominant Pfam family is in
    ``pfam``; clusters whose product name matches ``pattern`` (when non-empty);
    and, for ``region`` stories, a predicate over the whole map — ``"spur"``
    (beyond :data:`SPUR_RADIUS`), ``"dark"`` (no characterised member) or
    ``"phage"`` (dominant phylum Uroviricota, the tailed phages). The camera
    frames the densest, purest knot of those members within ``radius``.
    """

    pfam: tuple[str, ...] = ()
    region: str | None = None
    #: Keep only clusters whose dominant phylum falls in these landscape
    #: palette groups (see :func:`phylum_group`), before the knot seed is
    #: picked. The lever for a family whose PUREST knot is not the one the
    #: story is about: hemoglobin was the case — the globin family forms two
    #: knots of equal size, one bacterial and one animal, and the
    #: purity-weighted seed lands on the bacterial one, which would have made
    #: a story titled "the molecule of breath" light flavohemoglobins. That
    #: story is now a constellation and lights both, so no shipped story sets
    #: this today; it stays because the failure it fixes is a property of the
    #: seed rule, not of that one family. Refused on a constellation, which
    #: cuts no knot.
    groups: tuple[str, ...] = ()
    #: Light and frame the WHOLE selection rather than its densest knot: a story
    #: about the map itself (a quarter of it is dark; half a million clusters
    #: are phage) must show every member, or the panel's numbers and the
    #: picture contradict each other. No knot cut, no bubble, camera pulled
    #: back to hold all of them.
    whole: bool = False
    #: Look at the members from the SIDE (perpendicular to the ray from the
    #: map's centre) instead of from the outside along it: the spur points
    #: straight away from the centre, so the default pose looks down its length
    #: and foreshortens an 18-unit streak into a dot.
    side_on: bool = False
    #: A ``whole`` story whose selection is SMALL and spread over the map rather
    #: than a region of it (tyrosine decarboxylase: 52 clusters, half a map
    #: apart). The point of such a story is how few and how scattered its
    #: members are, so every one must be individually visible — which the
    #: region stories' sub-pixel markers are not (see
    #: :data:`SCATTER_HIGHLIGHT_RADIUS`). Requires ``whole``.
    scatter: bool = False

    #: Tell this story as a CONSTELLATION: light every place the family
    #: occupies, join them with a spanning tree of bright lines, and frame the
    #: whole figure from far enough out that all of it is in shot (Alex Rives's
    #: suggestion, 2026-09-17, in the shape the owner asked for on
    #: 2026-09-16). A different mode of story, not a decoration on the knot
    #: one: there is no knot cut and no bubble, and the lit nodes ARE the line
    #: endpoints — see :func:`constellation_of`.
    #:
    #: Reserved for families whose split MEANS something, because the lines
    #: assert that it does. Three qualify on this map and are measured in
    #: their panels: the globins (animal oxygen carriers against bacterial
    #: nitric-oxide detoxifiers), the photosynthetic reaction centres (D1, D2
    #: and the purple-bacterial L/M chains) and the lanthipeptide machinery
    #: (dehydratase, cyclase, immunity, precursor). The three smell families
    #: were tried and dropped: their places are the same protein in several
    #: spots, so a line between them says nothing the bubble does not.
    constellation: bool = False

    def __post_init__(self) -> None:
        if self.constellation and not self.pfam:
            raise ValueError(
                f"story {self.key!r}: constellation needs a pfam selector — the "
                "line means 'same Pfam family', so there must be one"
            )
        if self.constellation and (self.whole or self.scatter or self.side_on):
            raise ValueError(
                f"story {self.key!r}: constellation is its own framing mode and "
                "cannot be combined with whole/scatter/side_on"
            )
        if self.constellation and self.groups:
            raise ValueError(
                f"story {self.key!r}: constellation cannot take a groups filter — "
                "the figure is the whole family in every place it sits, and "
                "`groups` exists to bias the KNOT seed, which a constellation "
                "does not cut"
            )
        if self.scatter and not self.whole:
            raise ValueError(
                f"story {self.key!r}: scatter=True needs whole=True — a scatter "
                "story is about every member, so it takes no knot cut"
            )


VALID_REGIONS = ("spur", "dark", "phage")

_SWISSPROT = {s.key: s for s in SWISSPROT_STORIES}


def _carry(key: str, **overrides: object) -> UniverseStory:
    """A Swiss-Prot story re-targeted at this map, its vetted facts kept.

    Only the map-specific lines change (subtitle, the fact that describes what
    the blob shows, the narration sentence that repeats it) plus the selector.
    """
    base = asdict(_SWISSPROT[key])
    base["kingdom"] = None  # Swiss-Prot taxon filter; this map selects by Pfam
    base.update(overrides)
    return UniverseStory(**base)  # type: ignore[arg-type]


def _swap_fact(key: str, index: int, replacement: str) -> tuple[str, ...]:
    facts = list(_SWISSPROT[key].facts)
    facts[index] = replacement
    return tuple(facts)


#: Knot radius for a family story in this map (the Swiss-Prot tour used 0.8).
FAMILY_RADIUS = 0.3
#: Camera floor for a family story: a 0.2 bubble framed at 46% sits ~0.7 away.
FAMILY_MIN_DISTANCE = 0.7
#: Every pose here is composed for the cinematic lens (the distance rule in
#: `story_camera_distance` divides by tan of half this angle).
STORY_LENS_FOV_DEG = CINEMATIC_FOV_DEG

# Appearance, retuned for thirteen times the point density AND a camera that
# parks ~0.7 units from a knot (the Swiss-Prot tour sat 2-5 units out). A point
# radius is a WORLD size: the tour's 0.028 highlight would span ~25 px of frame
# height here and its 0.012 backdrop ~10 px, so a story frame was a wall of
# overlapping soft discs (checked in the browser: the knot read as one blurred
# blob, the near backdrop as haze over the panel). Radii sized for ~1% of the
# frame at knot distance keep individual clusters visible and the map a texture
# behind them; at the overview they are sub-pixel and render as 1 px points as
# before. The bubble's floor drops from 0.35 to 0.2 so a knot fills its bubble.
# BRIGHTNESS IS A PAIR: a per-layer window and a global exposure. `intensity`
# is a plain linear colour multiplier (a 1/16 window here), `exposure` is in
# log2 stops on top of it, and the pair below is the one MEASURED in the
# browser: the overview reads as a cloud, the view from a knot through the
# map's centre stays short of a whiteout (it whited out at +4.4 stops, a value
# found persisted in the app's storage), and the story highlights (0.4, one
# node, composed once) sit ~6x above the backdrop so a knot still pops. The
# tour's 0.08 at 0 stops was the dim overview the owner reported. These values
# reach the material as written: the wrapper is the only node that carries
# them (see `_add_backdrop`).
BACKDROP_RADIUS = 0.004
BACKDROP_INTENSITY = 0.0625
#: Global exposure in log2 stops (see `_viewer_config`).
BACKDROP_EXPOSURE_STOPS = 2.5
BACKDROP_OPACITY = 0.5
HIGHLIGHT_RADIUS = 0.006
HIGHLIGHT_INTENSITY = 0.4
#: A knot of this many members gets the full highlight intensity; bigger
#: highlights (the region stories light thousands) are dimmed by the square
#: root of the ratio, so the additive glow of a knot stays roughly constant
#: instead of saturating into one white ball.
HIGHLIGHT_REFERENCE_MEMBERS = 300


def highlight_intensity(n_members: int, *, base: float = HIGHLIGHT_INTENSITY) -> float:
    """Per-node highlight intensity for a highlight of ``n_members`` clusters."""
    ratio = HIGHLIGHT_REFERENCE_MEMBERS / max(n_members, HIGHLIGHT_REFERENCE_MEMBERS)
    return base * ratio**0.5


BUBBLE_MIN_RADIUS = 0.2
#: A whole-map highlight (hundreds of thousands of points seen from the
#: overview distance) renders as 1 px points: a larger, brighter point than a
#: knot's, so the region reads as a coloured shadow laid over the dim backdrop.
WHOLE_HIGHLIGHT_RADIUS = 0.008
WHOLE_HIGHLIGHT_INTENSITY = 0.35
#: A whole-map highlight draws at most this many of its members (a fixed random
#: sample; the panel says so). The backdrop alone sits at the viewer's residency
#: ceiling (7.7M points ≈ 500 MiB), and a two-million-point highlight on top was
#: declined at its first rung — measured in the browser: 312K of the dark
#: proteome's 2.03M arrived, then nothing. At overview distance the points are
#: sub-pixel anyway, so one in seven draws the same shadow as all of them.
WHOLE_HIGHLIGHT_MAX_POINTS = 300_000
#: A SCATTER story (see :attr:`UniverseStory.scatter`) lights a few dozen
#: clusters spread over the map, and its point is that the specks are visible
#: at all — so each marker is sized well above the region stories' one pixel.
#: Sized from the framing, not by eye: a scatter has ``frame_fraction >= 1``
#: and so frames the whole cloud, 38.0 units out for the tyrosine
#: decarboxylase stop, where the 63° cinematic lens spans 2·38·tan(31.5°) ≈
#: 46.6 world units of frame height. A marker of this radius therefore spans
#: 2·0.11/46.6 ≈ 0.5% of the frame — about 5 px on a 1080p display, against
#: the backdrop's sub-pixel 0.004.
#:
#: DO NOT read "countable" into that. An earlier version of this comment
#: computed the same number against a 15-unit framing and claimed 13 px, and
#: also implied the visitor can count the members. Measured 2026-09-16 on the
#: shipped selector: the 52 clusters have a median NEAREST-neighbour distance
#: of 0.022 units, a fifth of this radius, so they draw as 13 blobs — two of
#: them merged clumps of 21 and 13 markers. No radius fixes that (resolving a
#: 0.022 gap needs a marker smaller than a knot's, invisible at 38 units), so
#: the panel describes the blobs instead of promising a count.
SCATTER_HIGHLIGHT_RADIUS = 0.11
#: Below the knot intensity: a scatter marker covers ~700x the pixels of a
#: backdrop point, and the cinematic bloom saturates fat bright highlights.
SCATTER_HIGHLIGHT_INTENSITY = 0.25
#: A scatter story carries hover labels (its members are few and worth
#: reading); a region story's hundreds of thousands do not. This cap is the
#: line between the two — see :func:`_member_details`.
LABELLED_MEMBER_CAP = 2_000


#: A family's members are grouped into PLACES by single-linkage at this
#: radius — the same scale a knot lives at (see FAMILY_RADIUS), so "a place"
#: means "what the tour would frame as one knot". A place is what the
#: constellation lights and what its lines join.
CONSTELLATION_RADIUS = 0.3
#: Only places with at least this many clusters become a node. Measured over
#: the three shipped constellations: at 10 a family shows 4-8 places, which
#: reads as a figure; every family has a long tail of singletons that would
#: turn the map into a hairball (the globins alone break into 40 components,
#: 4 of them above 10).
CONSTELLATION_MIN_MEMBERS = 10
#: Share of the frame HEIGHT the figure spans at its waypoint. Lower than the
#: tour's 0.46 on purpose: a constellation is framed to be read whole, and the
#: distance below is computed against `r_max` so that it still fits when
#: auto-rotate swings the camera to its least favourable angle — at 0.46 the
#: worst angle put the outermost node on the frame edge.
CONSTELLATION_FRAME_FRACTION = 0.40
#: Node marker radius, as a fraction of the figure's own `r_max`. Scaling with
#: the figure rather than fixing a world size is what makes one number work
#: for all three: the camera distance is also proportional to `r_max`, so a
#: proportional marker subtends a CONSTANT angle. At this value a marker spans
#: 2·0.013/(2/0.40) ≈ 0.5% of the frame height — about 6 px on a 1080p
#: display, against the backdrop's sub-pixel 0.004 — so a place of a hundred
#: clusters reads as one glowing bead rather than as haze. HALVED from 0.026
#: on the owner's review of the first build ("the nodes are too big by a
#: factor 2 perhaps"); halving the radius also quarters the number of markers
#: stacked on a pixel, which is what was blowing the bead cores out to white.
CONSTELLATION_MARKER_FRAC = 0.013
#: Base intensity of those beads, dimmed by the usual square-root rule. Well
#: below the knot's 0.4: the markers are still an order of magnitude bigger
#: than a knot's and overlap heavily inside a place, so the summed additive
#: light at a bead's centre is what sets the exposure, not this number alone.
#: Lowered from 0.35 with the marker halving above, on the owner's review of
#: the first build ("perhaps a tad bit too bright") — at 0.35 the bead cores
#: clipped to white and lost the story's hue, which the halo alone carried.
CONSTELLATION_HIGHLIGHT_INTENSITY = 0.22
#: Line half-width as a fraction of `r_max`, i.e. ~0.38 of the marker radius:
#: thin enough to read as a thread strung between beads rather than as a rod.
#: Tracks CONSTELLATION_MARKER_FRAC — halved with it, or the threads would
#: have come out nearly as thick as the beads they join.
CONSTELLATION_LINE_WIDTH_FRAC = 0.005
#: Floor on that width, sized like the backdrop point radius: below this a
#: thread disappears into the cloud whatever the figure's own scale says.
CONSTELLATION_LINE_MIN_WIDTH = 0.02
#: How far the line colour is pushed towards white: `c + (1 - c)·whiten`. The
#: owner's brief — "light up brightly nearly white but perhaps hold some of
#: the hue of the story". At 0.72 a saturated story colour keeps a clear tint
#: while every channel sits above 0.7, so the line reads as light.
CONSTELLATION_LINE_WHITEN = 0.72
#: Brighter than the beads it joins, unlike the knot stories' annotation
#: lines: in a constellation the figure IS the subject, and the lines are what
#: makes it one figure rather than four blobs.
CONSTELLATION_LINE_INTENSITY = 0.5


@dataclass(frozen=True)
class Constellation:
    """The places a family occupies, joined into one figure.

    ``places`` are index arrays into the map (one per place with at least
    :data:`CONSTELLATION_MIN_MEMBERS` clusters), ``centroids`` their means,
    ``edges`` an ``(E, 2)`` minimum spanning tree over those centroids, and
    ``members`` the sorted union — the clusters the story lights.

    The three geometries matter together. ``centre`` is the bounding-box
    centre of ``members`` and is what the camera targets; ``r_max`` is the
    furthest member from it, which bounds the figure's apparent radius from
    EVERY direction and so survives auto-rotate; ``view`` is the figure's
    thinnest principal axis, signed away from the map's centre, which is the
    one direction that sees the figure spread rather than end-on.

    **A line joins two places of the same Pfam family.** That is a homology
    statement by construction — a Pfam family is built from one seed
    alignment. It is NOT a path, a distance, or an evolutionary trajectory;
    UMAP positions are not metric, so the straight segment carries no
    information beyond joining its ends. A minimum spanning tree rather than
    every pair: it is the fewest and shortest lines that still make the places
    one figure, and E = places - 1 stays legible.

    Deliberately NOT the ESM Atlas preprint's own edge metric (Jaccard over
    max-pooled SAE features, "same family" at >= 0.6): measured on this tour's
    families, no two different places come close to that bar — the vertebrate
    and worm chemoreceptor knots reach 0.38, photosystem II's D1 and D2 only
    0.27, against a 0.18 random-pair baseline. A feature line would therefore
    either draw nothing at the calibrated threshold, or assert a relation the
    preprint's own calibration calls unremarkable.
    """

    places: tuple[np.ndarray, ...]
    centroids: np.ndarray
    edges: np.ndarray
    members: np.ndarray
    centre: np.ndarray
    r_max: float
    view: np.ndarray


def single_linkage_places(
    positions: np.ndarray,
    mask: np.ndarray,
    *,
    min_members: int = CONSTELLATION_MIN_MEMBERS,
    radius: float = CONSTELLATION_RADIUS,
) -> list[np.ndarray]:
    """Index arrays of the places a masked selection occupies, largest first."""
    spatial = require_module("scipy.spatial")
    csgraph = require_module("scipy.sparse.csgraph")
    sparse = require_module("scipy.sparse")
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return []
    tree = spatial.cKDTree(positions[idx])
    pairs = np.asarray(list(tree.query_pairs(radius)), dtype=np.int64)
    if len(pairs) == 0:
        labels = np.arange(len(idx))
    else:
        graph = sparse.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(idx), len(idx)),
        )
        _, labels = csgraph.connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    keep = [c for c in np.flatnonzero(sizes >= min_members)]
    keep.sort(key=lambda c: int(sizes[c]), reverse=True)
    return [np.sort(idx[labels == c]) for c in keep]


def constellation_of(
    positions: np.ndarray,
    mask: np.ndarray,
    global_centre: np.ndarray,
    *,
    min_members: int = CONSTELLATION_MIN_MEMBERS,
    radius: float = CONSTELLATION_RADIUS,
) -> Constellation | None:
    """Build the figure for a masked family, or ``None`` if it has one place.

    A single place is not a constellation — there would be nothing to join —
    so the caller must fall back to a knot story rather than draw a figure
    with no lines.
    """
    csgraph = require_module("scipy.sparse.csgraph")
    places = single_linkage_places(
        positions, mask, min_members=min_members, radius=radius
    )
    if len(places) < 2:
        return None
    centroids = np.stack([positions[s].mean(axis=0) for s in places]).astype(np.float32)
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    mst = csgraph.minimum_spanning_tree(dist).tocoo()
    edges = np.stack([mst.row, mst.col], axis=1).astype(np.int64)
    members = np.sort(np.concatenate(places))
    centre, radial = cluster_geometry(positions[members])
    # The thinnest principal axis of the lit members: looking ALONG it is the
    # one direction that sees the figure spread out instead of end-on. Signed
    # outward so the camera sits outside the cloud looking back into it.
    spread = positions[members].astype(np.float64)
    _, vectors = np.linalg.eigh(np.cov((spread - spread.mean(axis=0)).T))
    view = np.asarray(vectors[:, 0], dtype=np.float64)
    outward = centre - np.asarray(global_centre, dtype=np.float64)
    if float(view @ outward) < 0.0:
        view = -view
    return Constellation(
        places=tuple(places),
        centroids=centroids,
        edges=edges,
        members=members,
        centre=centre,
        r_max=float(radial.max()),
        view=view / float(np.linalg.norm(view)),
    )


def story_constellations(
    universe: Universe, stories: tuple[UniverseStory, ...]
) -> dict[str, Constellation]:
    """The figure for every constellation story, keyed by story key.

    Pure in ``(universe.positions, story.pfam)``, so every call site —
    framing, highlight, lines — recomputes the SAME figure rather than
    threading one object through, and the lines cannot drift off the beads.
    """
    centre = universe.positions.mean(axis=0)
    out: dict[str, Constellation] = {}
    for s in stories:
        if not s.constellation:
            continue
        figure = constellation_of(
            universe.positions, universe.pfam_mask(s.pfam), centre
        )
        if figure is None:
            raise ValueError(
                f"story {s.key!r} is a constellation, but its Pfam selection "
                f"{s.pfam} occupies fewer than two places of "
                f"{CONSTELLATION_MIN_MEMBERS}+ clusters — there is nothing to "
                "join. Tell it as a knot story instead."
            )
        out[s.key] = figure
    return out


def carries_labels(story: UniverseStory, n_members: int) -> bool:
    """Whether a story's highlight carries per-member hover labels.

    A knot always does. A ``whole`` story does only when its members are few
    enough to be worth reading and cheap enough to store: a scatter story's
    few dozen qualify, a region story's hundreds of thousands do not (see
    :data:`LABELLED_MEMBER_CAP`).
    """
    return not story.whole or n_members <= LABELLED_MEMBER_CAP


def highlight_appearance(
    story: UniverseStory, n_drawn: int, *, figure: Constellation | None = None
) -> tuple[float, float]:
    """Point radius and intensity for a story's highlight node.

    Four regimes: a constellation's beads, sized to the figure so they hold a
    constant angular size at its pulled-back framing; a scatter story's
    countable dots; a region story's sub-pixel shadow over the whole map; and
    a knot's points sized for the close framing (dimmed by
    :func:`highlight_intensity` when the knot is big).
    """
    if story.constellation:
        if figure is None:
            raise ValueError(
                f"story {story.key!r} is a constellation: pass its figure, whose "
                "extent sets the bead size"
            )
        return (
            max(HIGHLIGHT_RADIUS, CONSTELLATION_MARKER_FRAC * figure.r_max),
            highlight_intensity(n_drawn, base=CONSTELLATION_HIGHLIGHT_INTENSITY),
        )
    if story.scatter:
        return SCATTER_HIGHLIGHT_RADIUS, SCATTER_HIGHLIGHT_INTENSITY
    if story.whole:
        return WHOLE_HIGHLIGHT_RADIUS, WHOLE_HIGHLIGHT_INTENSITY
    return HIGHLIGHT_RADIUS, highlight_intensity(n_drawn)


#: The backdrop is a hand-built ``kind=partition`` of 64 BSP tiles of at most
#: this many points, each tile its own ``kind=lod`` ladder of two POINTS
#: levels: a fixed random 1-in-K subsample (colours scaled by K so the additive
#: light is conserved) and the full tile. Two ceilings at once: one points node
#: renders at most 5,591,040 on a 4096-class GPU and silently drops the tail
#: (one contiguous Hilbert-order region, so a clean-edged hole); and the owner
#: wants the drawn count near 2M for frame rate. ``coverage`` is evaluated PER
#: TILE against the tile's own projected box, so the tiles must be small: with
#: 8 tiles each was a 20-unit cube whose box filled the screen even from the
#: overview, and three of them sat at full detail there. At 64 tiles the
#: overview and the whole-map stories draw every tile coarse (~1.9M points)
#: and a knot swaps in only its nearest handful. Two dead ends, both measured
#: in the browser: a single lod group over a partition swapped ALL parts at
#: once (9.6M resident at every knot); and the library's substitutive coarse
#: levels — synthesised GAUSSIANS — rendered the view from a knot toward the
#: map's centre at 6 fps at DPR 1 (1 fps at DPR 2), against 144 fps for the
#: same 6.6M positions as plain points: merged Gaussians are fat where the
#: cloud is sparse, and from inside the cloud a million of them overlap on
#: every pixel. Points stay 1 px however many there are.
BACKDROP_TILE_POINTS = 125_000
#: Coarse level = one point in this many (light conserved by scaling colours).
BACKDROP_LOD_FACTOR = 4

_STORY_POOL: tuple[UniverseStory, ...] = (
    _carry(
        "Hemoglobin",
        # The carried narration never mentioned the map, and on this tour the
        # story is a CONSTELLATION — a visitor heard Perutz's balsa-wood model
        # while looking at four beads joined by threads. Overridden here and
        # not in the Swiss-Prot tour, whose map has no such figure.
        narration=(
            "Hemoglobin, the molecule of breath. Each red blood cell carries "
            "some two hundred and eighty million of these, and each one holds "
            "four oxygens. In 1949 Linus Pauling traced sickle-cell anaemia "
            "to an altered hemoglobin, the first molecular disease; the cause "
            "proved to be a single swapped amino acid. Max Perutz needed "
            "twenty-two years to see its shape. The lines here join the four places this fold is filed "
            "in: one mostly animal, three mostly bacterial, where many destroy nitric oxide instead of carrying oxygen. And yet "
            "hemoglobin also turns up inside dopamine neurons, nowhere near "
            "blood. What it does there, nobody quite knows."
        ),
        # A CONSTELLATION: the same fold in four places, one animal and three
        # bacterial, and the split is the story (see `UniverseStory.constellation`).
        constellation=True,
        subtitle="One fold in four places — animal oxygen carriers and bacterial nitric-oxide destroyers",
        pattern="",
        pfam=("PF00042",),  # Globin
        # Audit (2026-09-16): 423 globin clusters map-wide, 391 of them in four
        # places of 10+ at 0.3 linkage, joined by 3 lines totalling 20.2 units
        # (longest 14.1). Place 1, 159 clusters: animal (Chordata 53, Nematoda
        # 38, Arthropoda 9, plus Pseudomonadota 29), named "globin
        # domain-containing protein"; 14 named hemoglobin, 7 neuroglobin.
        # Place 2, 159: Pseudomonadota 133, mostly generic bacterial globins,
        # 13 named flavohemoprotein or NO dioxygenase; the 14.1 line joins
        # places 1 and 2. Place 3, 46: named "nitric
        # oxide dioxygenase", Actinomycetota 25 / Pseudomonadota 15. Place 4,
        # 27: FAD-binding oxidoreductase/globin fusions, Pseudomonadota 18.
        # This story used to cut a KNOT and needed a `groups` filter to stop
        # the purity-weighted seed landing on the bacterial half; a
        # constellation takes no knot cut, so the filter is gone (and is
        # refused outright — see `UniverseStory.__post_init__`).
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "Hemoglobin",
            3,
            # The lines are the claim: same Pfam family (PF00042), four
            # places. Vinogradov & Moens, JBC 283:8773 (2008), for the fold's
            # reach beyond animals; Gardner et al., PNAS 95:10378 (1998), for
            # flavohemoglobin as a nitric oxide dioxygenase.
            "The lines join the four places this one fold is filed in. One is "
            "mostly animal globins — hemoglobin beside the neuroglobin of "
            "nerves and the globins of worms and arthropods. The other three "
            "are mostly bacterial, and many of them are not carrying oxygen at "
            "all. Many are flavohemoglobins, which use oxygen to destroy toxic "
            "nitric oxide, including the bursts an immune system fires at "
            "them. Same fold, a different job, far apart on the map.",
        ),
    ),
    _carry(
        "Photosystem II",
        # As for hemoglobin: a constellation on this tour, and the carried
        # narration said nothing about the six places or the lines.
        narration=(
            "Photosystem II, the machine that made the sky breathable. Its D1 "
            "subunit sits at the heart of the only enzyme known that splits "
            "water. Cyanobacteria running this machine began filling Earth's "
            "air with oxygen, some 2.4 billion years ago. The chemistry is "
            "so violent that D1 wrecks itself within the hour in bright sun; "
            "a leaf rebuilds it all day long. The lines here join six places; "
            "the biggest are D1 itself, its partner D2, and the purple-bacterial chains that "
            "split no water. Some molecular clocks say water-splitting is far "
            "older than the rise of oxygen. So why did the planet wait so "
            "long to change?"
        ),
        # A CONSTELLATION: the reaction centre taken apart — D1, D2 and the
        # purple-bacterial L/M chains are filed in separate places.
        constellation=True,
        subtitle="The reaction centre taken apart — D1, D2 and the purple-bacterial chains, in six places",
        pattern="",
        pfam=("PF00124",),  # Photo_RC: D1/D2 and the L/M chains
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "Photosystem II",
            3,
            # Audit (2026-09-16): 265 clusters map-wide, 183 in six places of
            # 10+ at 0.3 linkage, 5 lines totalling 12.5 units (longest 7.0).
            # Largest place, 80: the L and M chains of purple bacteria
            # (Pseudomonadota 64). Next, 44: D1 — Uroviricota 18,
            # Cyanobacteriota 13, Rhodophyta 7, Streptophyta 5, and NOT ONE D2.
            # Third, 22: D2, eighteen of them named so outright. The remaining
            # three (16, 11, 10) are reaction-centre domains and D2 fragments.
            # D1/D2 homologous to L/M: Michel & Deisenhofer, Biochemistry
            # 27:1 (1988). Cyanophage psbA: Mann et al., Nature 424:741
            # (2003); Lindell et al., Nature 438:86 (2005).
            "The lines join six places, and they are the reaction centre taken "
            "apart. One is D1 itself, from cyanobacteria, red algae and "
            "plants — and two in five of its clusters belong to viruses, "
            "because cyanophages carry their own copy and switch it on during "
            "infection to keep the host photosynthesising while they "
            "replicate. Another is D1's partner D2. The largest of all is the "
            "L and M chains of purple bacteria, which run photosynthesis "
            "without ever splitting water. The model has pulled the machine "
            "apart and filed each piece on its own.",
        ),
    ),
    _carry(
        "Hsp70",
        subtitle="Nearly four thousand clusters of one chaperone across the map; this knot is one of dozens",
        pattern="",
        pfam=("PF00012",),  # HSP70
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "Hsp70",
            2,
            # Hsp70's "about 47% identical" (fact 1) is measured, not quoted:
            # a global alignment of E. coli DnaK (P0A6Y8) against human HSPA1A
            # (P0DMV8) gives 47.9% identity over 629 aligned columns, and 45.9%
            # against Hsc70 (P11142).
            # Audit: 3,799 HSP70 clusters in 73 knots (19 of 20+ members);
            # the big ones mix bacterial phyla (Pseudomonadota, Bacillota,
            # Actinomycetota, Bacteroidota...), and plants and vertebrates share
            # a few smaller ones with them — no knot is one branch of life.
            "The map holds it in dozens of knots — most of them "
            "bacterial DnaK in one variation or another, with the plant and "
            "animal versions filed among them — all recognisably the same "
            "protein. The map is showing a protein older than complex life, "
            "passed between the great branches of the tree.",
        ),
        narration=(
            "Hsp70, the oldest job in the cell. It holds unfolded proteins, "
            "refolds the damaged ones, and hands the hopeless ones to the "
            "shredder. Almost every bacterium and every eukaryote has one. "
            "After some two billion years apart, the human and E. coli "
            "versions are still nearly half identical, letter for letter. The "
            "map holds it in dozens of knots, most of them bacterial, all "
            "recognisably the same protein. "
            "Cancer cells over-produce it to survive their own chaos, and drugs "
            "against it have been tried for decades. None has been approved. "
            "Why is such a universal protein so hard to target?"
        ),
    ),
    _carry(
        "Viral surface proteins",
        title="The intruders' spike",
        subtitle="Spike, haemagglutinin and the coats of a thousand viruses",
        pattern=(
            r"(?i)hemagglutinin|spike glycoprotein|envelope glycoprotein|"
            r"fusion glycoprotein|major capsid|capsid protein|tail fiber|"
            r"tail fibre|coat protein"
        ),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "Viral surface proteins",
            3,
            # Audit: knot of 372 clusters, every named one a spike glycoprotein
            # (Pisuviricota, the phylum that holds the coronaviruses; most have
            # no phylum recorded). Hosts are the family's, not read from the
            # table: coronaviruses of bats, birds (IBV), pigs (PEDV, TGEV),
            # camels (MERS) and people.
            "This knot is the coronavirus spike in hundreds of versions — a "
            "family that infects bats, birds, pigs, camels and people — the "
            "protein the whole world learned to read in 2020.",
        ),
        # This map is built from metagenomes, viral ones included, and the
        # dark and phage stops come just before; the question is re-aimed.
        mystery=(
            "Most viral proteins have no known relatives at all, and much of "
            "this map is dark. How many more spikes hide in it, built like "
            "these but invisible to a sequence search?"
        ),
        pdb_id="6VXX",  # SARS-CoV-2 spike, closed state
        narration=(
            "The intruders' spike. Influenza's haemagglutinin, the coronavirus "
            "spike, HIV's envelope: unrelated viruses, one trick. Each snaps "
            "into a bundle that drags virus and cell together. The 1918 flu "
            "killed tens of millions of people with a protein like this one. "
            "This knot is the coronavirus spike, in hundreds of versions, from a "
            "family that infects bats, birds, camels and people. Most viral "
            "proteins have no known relatives at all. How many more spikes "
            "hide in this map's dark?"
        ),
    ),
    _carry(
        "ATP synthase",
        # The carried narration was all mechanism and no map.
        narration=(
            "ATP synthase, the turbine in almost every cell. Protons flowing "
            "through it turn an axle, and each turn presses out three "
            "molecules of ATP. In 1997 a single motor was filmed spinning "
            "under a microscope. You make and spend roughly your own body "
            "weight in ATP every day. The knot lit here is nearly three hundred "
            "clusters of the beta subunit as bacteria build it, with the "
            "chloroplast copies of plants filed among them. It is one of the "
            "most efficient motors known, wasting almost nothing as heat. "
            "How a protein manages that is still debated."
        ),
        subtitle="The rotary motor that makes the currency of life, in bacteria and in the cells that took them in",
        pattern=r"(?i)ATP synthase (subunit )?beta",
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "ATP synthase",
            3,
            # Audit: knot of 295, 93% pure — Bacillota 106, Pseudomonadota 62,
            # Bacteroidota 39, Actinomycetota 28, and 10 named "chloroplastic".
            "This knot is the beta subunit as bacteria build it, in hundreds of "
            "versions, with the chloroplast copies of plants filed among them. "
            "The copies in our own mitochondria descend from the same source: "
            "one motor, inherited from the bacteria that became part of our "
            "cells.",
        ),
    ),
    _carry(
        "RuBisCO",
        # The carried narration was all chemistry and no map.
        narration=(
            "RuBisCO, perhaps the most abundant enzyme on Earth, and one of "
            "the slowest. Nearly every carbon atom in every living thing has "
            "passed through it. Averaged over day and night, it fixes only "
            "about one CO2 every thirty seconds, and it keeps confusing oxygen "
            "with carbon dioxide, so plants make it by the tonne. The knot lit here is a hundred and seventeen "
            "clusters of the large chain: half the plant enzyme, the rest "
            "bacterial, with a few relatives that fix no carbon at all. "
            "Three billion years of evolution never produced a fast, accurate "
            "RuBisCO. Is that a wall that cannot be climbed, or has nobody "
            "found the path?"
        ),
        subtitle="The protein that pulls carbon out of the air for almost all life",
        pattern="",
        pfam=("PF00016", "PF02788"),  # RuBisCO_large, RuBisCO_large_N
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "RuBisCO",
            3,
            # RuBisCO's "one CO2 every thirty seconds" (fact 1) is Bar-On &
            # Milo, PNAS 116:4738 (2019): the effective time-averaged rate on
            # land is ~0.03/s, one per 33 s, about 1% of the ~3/s kcat. Their
            # 0.7 Gt global mass is the same paper.
            # Audit: knot of 117 — Streptophyta 58 (50%), Pseudomonadota 27,
            # Bacillota 10; 83 named for the large chain and seven "2,3-diketo-
            # 5-methylthiopentyl-1-phosphate enolase": the RuBisCO-like protein
            # of the methionine salvage pathway (Ashida et al., Science 302:286
            # (2003), Bacillus subtilis).
            "The knot here is the large chain: half of it the plant enzyme, the "
            "rest bacterial forms — and a few RuBisCO-like proteins that never "
            "fix carbon at all. In Bacillus, one of them recycles methionine "
            "instead.",
        ),
    ),
    _carry(
        "RecA and Rad51",
        subtitle="One recombinase, from phages and E. coli to the BRCA2 pathway in our cells",
        pattern="",
        pfam=("PF00154",),  # RecA (Rad51 is its own family, PF08423)
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "RecA and Rad51",
            3,
            # Audit: knot of 171, 169 of them Uroviricota (tailed phages); 669
            # of the 1,560 RecA clusters in the map are viral. Phage-encoded
            # RecA homologs (T4's UvsX among them) are well known. RAD51 is
            # its own Pfam family (PF08423, 535 clusters: archaea 222, fungi
            # 69, vertebrates 48, plants 47) in 21 components, the largest of
            # them archaeal RadA and centred one unit from this knot.
            "The densest RecA knot in this map belongs to viruses: many tailed "
            "phages carry a RecA of their own, to repair and reshuffle their "
            "genomes inside the host. The RAD51 family makes knots of its own, "
            "the largest of them the archaeal RadA, a short way off; "
            "the shape has barely moved in billions of years.",
        ),
        narration=(
            "RecA and Rad51, the machine that mends broken DNA. It coats a "
            "broken strand and searches the entire genome for the matching "
            "sequence. Our version, RAD51, is loaded by BRCA2, whose gene lies "
            "behind many inherited breast cancers. This knot belongs to phages, "
            "which carry a RecA of their own. The RAD51 family, led by the "
            "archaeal RadA, sits in knots nearby; the shape has barely moved in "
            "billions of years. It finds one match among "
            "millions of base pairs in minutes. How it searches that fast is "
            "still argued over."
        ),
    ),
    UniverseStory(
        key="ABC transporters",
        title="The spur — ABC transporters flung off the map",
        subtitle="Eight thousand clusters, nearly all of them parts of ABC transporters, in a streak of their own",
        pattern="",
        region="spur",
        whole=True,
        color=(0.3, 0.95, 0.95),
        # The streak is long and thin: seen side-on and framed at 1.2 it runs
        # across the frame like a river, not a dot in the distance.
        side_on=True,
        frame_fraction=1.2,
        flight_ms=3000,
        facts=(
            # Linton & Higgins, Mol. Microbiol. 28:5 (1998): ~5% of the E. coli
            # genome encodes ABC transporter components.
            "ATP-binding cassette transporters pump molecules across membranes, "
            "burning ATP to do it: nutrients in, toxins out. Every genome has "
            "them — about five percent of E. coli's genes are devoted to them — "
            "and they are among the largest protein families known.",
            # Dean, Rzhetsky & Allikmets, Genome Res. 11:1156 (2001): 48 human
            # ABC genes; CFTR = ABCC7; Juliano & Ling 1976 (P-glycoprotein).
            "Humans have 48. When one of them, CFTR, is broken, the result is "
            "cystic fibrosis; another, P-glycoprotein, pumps chemotherapy back "
            "out of cancer cells and is a well-known cause of drug resistance "
            "in tumours.",
            "The engine is the same everywhere: a pair of nucleotide-binding "
            "domains that clamp shut around two ATPs and spring open when they "
            "are spent. That engine — the cassette — is what these clusters "
            "share.",
            # 8,277 clusters on the spur; 96% carry an ABC-transporter Pfam
            # family as their dominant domain (see SPUR_PFAMS). ABC_tran
            # (PF00005) has long been reported as the largest Pfam-A family by
            # sequence count, and it is far and away the commonest dominant
            # family in this map: 48,106 clusters against 31,024 for the next,
            # the MFS transporters (PF07690). The measured number is ours; the
            # Pfam ranking is not restated as a bare fact in the panel.
            "The map put them on a spur of their own. A streak like this is "
            "what the layout algorithm does with a huge, tightly knit family "
            "that shares few features with anything else — partly an artefact, "
            "but an honest one: the ATP-binding cassette is among the largest "
            "protein families known, and by a wide margin the commonest on "
            "this map.",
        ),
        mystery=(
            "The same cassette powers importers and exporters, in bacteria and "
            "in us. How its ATP cycle is coupled to the movement of cargo — the "
            "actual mechanics of the pump — is still argued for most of the "
            "family."
        ),
        tags=("membranes", "medicine", "map artefact"),
        pdb_id="2HYD",  # Sav1866, a multidrug ABC exporter
        narration=(
            "The spur. These eight thousand clusters were flung off the map, and "
            "nearly all of them are one thing: the ATP-binding cassette, the "
            "engine of the ABC transporters. Every genome has them, pumping "
            "nutrients in and toxins out. Humans have forty-eight; a broken one "
            "causes cystic fibrosis, another pumps chemotherapy out of tumours. "
            "The streak is a habit of the layout algorithm, but an honest one: "
            "no design is repeated more often on this map. How the engine "
            "actually moves its cargo is still argued over."
        ),
    ),
    UniverseStory(
        key="Dark proteome",
        title="The dark proteome — two million clusters nobody has characterised",
        subtitle="A quarter of this map has no characterised member at all",
        pattern="",
        region="dark",
        whole=True,
        color=(0.5, 0.65, 1.0),
        frame_fraction=1.0,
        facts=(
            # Annotation table: 2,025,330 of 7,723,579 clusters have
            # cluster_pct_characterized == 0 (no member carries a Pfam domain of
            # known function — the preprint's own definition, which reports
            # "over two million" such clusters); 76% of those are named
            # "hypothetical protein".
            "Two million of the 7.7 million clusters here — one in four — "
            "contain no protein with a domain of known function: nobody has "
            "yet worked out what they do. Most wear the placeholder "
            "name “hypothetical protein”. They are the dim points of this map.",
            # Preprint, atlas section + Appendix A.5.1/A.5.2: 6.82 billion
            # sequences, 5.6 billion from metagenomic samples (SPIRE, MGnify:
            # gut, aquatic, soil, wastewater, agriculture, built environment);
            # 1.1 billion representative structures from ESMFold2.
            "Most of this atlas comes from metagenomics: DNA read straight out "
            "of soil, seawater, wastewater and guts, from organisms nobody has "
            "grown in a lab. Of the 6.8 billion sequences this map is drawn "
            "from, 5.6 billion came that way, and across the whole atlas 1.1 "
            "billion structures have been predicted.",
            # Annotation table (re-measured): the ten nearest neighbours of a
            # dark cluster are dark 80% of the time against a 26% base rate,
            # rising to 85% for dark clusters in the densest 1% of the map.
            # An earlier draft said "more than nine in ten in the densest
            # pockets" — that holds only for a handful of hand-picked voxels,
            # not for the densest regions generally, so it is gone.
            "Dark sits next to dark: eight of the ten nearest neighbours of an "
            "uncharacterised cluster are uncharacterised too, where one in "
            "four would be the rate if they were scattered. Whatever these proteins do, they do "
            "it in families of their own.",
            # CRISPR: Ishino et al. 1987 → the viral-spacer insight of 2005.
            # GFP: Shimomura 1962 → Chalfie et al. 1994.
            "Some of the most useful tools in biology were dark once. The "
            "CRISPR repeats were “unusual” DNA for nearly twenty years; green "
            "fluorescent protein was a jellyfish curiosity for thirty.",
        ),
        mystery=(
            "Are these families genuinely new chemistry, or old folds whose "
            "sequences drifted beyond recognition? The predicted structures say "
            "a little of both — and nobody yet knows the proportion."
        ),
        tags=("metagenomics", "unknown function"),
        pdb_id="2GA1",  # a DUF433 protein of unknown function (structural genomics)
        narration=(
            "The dark proteome. Two million of these clusters, one in four, "
            "contain no protein with a domain of known function. They are "
            "the dim points of this map, read straight out of soil, seawater "
            "and guts, from organisms nobody has grown. Dark sits next to dark: "
            "eight of the ten nearest neighbours of an uncharacterised cluster "
            "are uncharacterised too, where one in four would be the rate by chance. "
            "Some of biology's best tools were dark once; CRISPR was unusual "
            "DNA for almost twenty years. Are these new chemistry, or old folds "
            "drifted beyond recognition? Nobody knows the proportion."
        ),
    ),
    UniverseStory(
        key="Phage",
        title="The phage universe — the viruses that outnumber everything",
        subtitle="More than half a million clusters of tailed bacteriophages, half of them uncharacterised",
        pattern="",
        region="phage",
        whole=True,
        color=(1.0, 0.55, 0.75),
        frame_fraction=1.0,
        facts=(
            # Hendrix et al., PNAS 96:2192 (1999): ~10^31 tailed phages.
            # Annotation table: 580,733 clusters with Uroviricota (the tailed
            # phages) as dominant phylum — 93% of the 622,879 viral clusters.
            # Preprint: "Almost all viral sequences are metagenomic, largely
            # derived from bacteriophages" (atlas section).
            "Bacteriophages — the viruses of bacteria — are the most abundant "
            "biological entities on Earth: an estimated ten million trillion "
            "trillion of them, about ten for every bacterial and archaeal "
            "cell. Here "
            "581,000 clusters are theirs.",
            # Suttle, Nat. Rev. Microbiol. 5:801 (2007): viruses kill ~20% of
            # ocean microbial biomass per day (the "viral shunt").
            "In the oceans they kill around a fifth of all microbial biomass "
            "every day, spilling its carbon back into the water — the “viral "
            "shunt” that shapes the planet's carbon cycle.",
            # Twort, Lancet 1915; d'Hérelle, C. R. Acad. Sci. 1917.
            "Discovered twice, by Frederick Twort in 1915 and Félix d'Hérelle "
            "in 1917, they were medicine before penicillin — and, with "
            "antibiotic resistance rising, phage therapy is being tried again.",
            # Annotation table: 51% of Uroviricota clusters have
            # cluster_pct_characterized == 0.
            "Half of the phage clusters here are dark: no member has ever been "
            "characterised. Phage genomes are probably the largest reservoir "
            "of unexplored genes on the planet.",
        ),
        mystery=(
            "Every gram of soil and every millilitre of seawater holds millions "
            "of phages. What most of their genes do — and how many kinds of "
            "phage there really are — nobody knows."
        ),
        tags=("virology", "ecology"),
        # THE WHOLE PHAGE, not a part of one: the recognisable silhouette,
        # capsid through connector and tail to the fibres. Native
        # bacteriophage P68 has the whole virion in one
        # entry — 8 entities over 668 chains, 20 MDa, 3.8 A: major head
        # protein (235 chains) for the capsid, portal (12) and lower collar
        # (12) for the connector, minor structural (72) and tail fibre (72)
        # for the tail and legs, plus 15 head fibres.
        #
        # Three earlier choices were rejected. 6R21, the T7 "fiberless tail
        # complex", has the PORTAL as its first and largest entity — a head
        # connector, 12 of its 30 chains — and no fibres, yet a comment here
        # called it "the whole tail machine". 8IYK, the lambda tail tip, is
        # honestly a tail but only a tail. 9MJN, the near-complete PhiTE
        # virion, is the most complete phage in the PDB (1,996 chains,
        # 57.6 MDa) and is the right ANSWER but the wrong SIZE for this
        # pipeline: PyMOL's per-chain molecular surface passed 7.9 GB after
        # eighteen minutes without finishing, so it was abandoned. If the
        # turntable ever grows a coarse-surface path for huge assemblies,
        # 9MJN is the entry to come back to.
        pdb_id="6Q3G",
        narration=(
            "The phage universe. Bacteriophages, the viruses of bacteria, are "
            "the most abundant biological entities on Earth: ten million "
            "trillion trillion of them, outnumbering bacteria ten to one. "
            "More than half a million clusters here are theirs. In the oceans they kill a "
            "fifth of all microbial biomass every day. They were medicine before "
            "penicillin, and are being tried again. Half of the phage clusters "
            "here are dark. What most of their genes do, nobody knows."
        ),
    ),
    UniverseStory(
        key="Beta-lactamases",
        title="Beta-lactamase — the enzyme that fights back",
        subtitle="Nine thousand clusters of the beta-lactamase fold; this knot is the class A enzymes, TEM-1's own family",
        pattern="",
        # PF00144 (7,168 clusters) is the beta-lactamase FOLD — class C
        # enzymes, but also penicillin-binding proteins and esterases that never
        # touch an antibiotic; PF13354 (2,251) is the class A beta-lactamases
        # proper (TEM, SHV, CTX-M). The densest knot (398 clusters) is 97%
        # PF13354, so the turntable's TEM-1 is the right molecule for it.
        pfam=("PF00144", "PF13354"),  # Beta-lactamase, Beta-lactamase2
        color=(0.95, 0.25, 0.5),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "Penicillin works by jamming the enzymes that build the bacterial "
            "cell wall. A beta-lactamase cuts open the antibiotic's "
            "four-membered ring before it gets there, and the drug is dead.",
            # Abraham & Chain, Nature 146:837 (published 28 Dec 1940); the
            # first infection treated with purified penicillin was Albert
            # Alexander's, 12 Feb 1941.
            "Resistance was there before the cure. Edward Abraham and Ernst "
            "Chain described an E. coli enzyme that destroyed penicillin in "
            "December 1940 — weeks before purified penicillin was first used "
            "to treat a patient.",
            # Murray et al., Lancet 399:629 (2022): 4.95 million deaths
            # associated with, 1.27 million attributable to, bacterial AMR in 2019.
            "Antibiotic resistance is now associated with nearly five million "
            "deaths a year, 1.3 million of them directly attributable. "
            "Beta-lactamases — TEM-1, CTX-M, NDM-1 — are the core of the "
            "problem in Gram-negative bacteria.",
            # D'Costa et al., Nature 477:457 (2011): resistance genes in
            # 30,000-year-old Beringian permafrost.
            # Hall & Barlow, Drug Resist. Updat. 7:111 (2004): phylogenies put
            # the origin of the serine beta-lactamases more than two billion
            # years back, and some on plasmids for millions of years. The
            # beta-lactam producers are moulds (Penicillium) and soil bacteria
            # (Streptomyces).
            "Beta-lactamase genes have been recovered from 30,000-year-old "
            "permafrost, and phylogenies reckon the serine beta-lactamases far "
            "older still — a couple of billion years. Moulds and soil bacteria "
            "have long made penicillin-like antibiotics, and their neighbours "
            "have long destroyed them.",
        ),
        mystery=(
            "Thousands of beta-lactamase variants are known and new ones appear "
            "every year. Can inhibitors and new antibiotics keep pace with an "
            "enzyme family that evolves in real time, in every hospital on "
            "Earth?"
        ),
        tags=("medicine", "antibiotic resistance"),
        pdb_id="1BTL",  # TEM-1
        narration=(
            "Beta-lactamase, the enzyme that fights back. Penicillin jams the "
            "machinery that builds the bacterial cell wall; this enzyme cuts "
            "the antibiotic open first. Resistance came before the cure: it was "
            "described in 1940, weeks before purified penicillin first treated a patient. "
            "Today antibiotic resistance is linked to nearly five million deaths "
            "a year, and these genes turn up in thirty-thousand-year-old "
            "permafrost. The knot lit here is four hundred clusters, almost "
            "all class A, the family TEM-1 itself belongs to. Can new drugs "
            "keep pace with an enzyme that evolves in real time?"
        ),
    ),
    UniverseStory(
        key="CRISPR-Cas",
        title="CRISPR-Cas9 — a bacterial immune system turned into scissors",
        subtitle="Hundreds of clusters of Cas9 and its kin, the enzymes bacteria use to cut viral DNA",
        pattern="",
        # The preprint's own Cas9 class (Table S15 of Candido et al. 2026):
        # the RuvC, REC-lobe, bridge-helix, WED, PI and C-terminal families of
        # Cas9, plus HNH_4 (PF13395) — a GENERIC HNH endonuclease family, and
        # the one that supplies 585 of the 874 clusters, so the selection is
        # broader than "Cas9" at the family level. The knot it lands on is not:
        # 260 clusters, 88% pure, 132 of them named for Cas9 itself.
        pfam=(
            "PF22702",
            "PF13395",
            "PF16593",
            "PF16592",
            "PF16595",
            "PF18470",
            "PF21069",
            "PF18525",
            "PF18061",
            "PF17893",
            "PF17894",
            "PF18070",
        ),  # fmt: skip
        color=(0.4, 1.0, 0.8),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            # Barrangou et al., Science 315:1709 (2007).
            "CRISPR is an immune system. Bacteria keep snippets of the DNA of "
            "viruses that attacked them, and Cas proteins use those snippets "
            "as a guide to find and cut the same virus next time.",
            # Ishino et al., J. Bacteriol. 169:5429 (1987); Mojica et al.,
            # Pourcel et al., Bolotin et al. (2005).
            "The repeats were first noticed in E. coli in 1987 and stayed a "
            "curiosity for years; in 2005 three groups realised that the "
            "spacers between them matched viral DNA.",
            # Jinek et al., Science 337:816 (2012); Nobel Prize in Chemistry 2020.
            "In 2012 Jennifer Doudna and Emmanuelle Charpentier showed that "
            "Cas9 could be pointed at almost any DNA sequence by rewriting its guide — "
            "programmable scissors. Nobel Prize in Chemistry 2020.",
            # Casgevy (exagamglogene autotemcel): MHRA Nov 2023, FDA Dec 2023.
            "Eleven years later the first CRISPR medicine was approved: a "
            "therapy for sickle-cell disease — the illness of the first story "
            "on this tour.",
        ),
        # Makarova et al., Nat. Rev. Microbiol. 13:722 (2015): CRISPR-Cas in
        # 87% of archaeal and 50% of bacterial genomes.
        mystery=(
            "About half of bacteria and nearly nine in ten archaea carry CRISPR systems, "
            "yet many highly successful bacteria do without. Why would an "
            "organism give up an immune system?"
        ),
        tags=("genome editing", "immunity"),
        pdb_id="4OO8",  # S. pyogenes Cas9 with guide RNA and target DNA
        narration=(
            "CRISPR-Cas9, a bacterial immune system turned into scissors. "
            "Bacteria keep snippets of the viruses that attacked them, and Cas "
            "proteins use those snippets to find and cut the same virus next "
            "time. Noticed in 1987, linked to viruses in 2005, and in 2012 Doudna and "
            "Charpentier showed Cas9 could be pointed at almost any DNA. Eleven "
            "years later the first CRISPR medicine was approved, for sickle-cell "
            "disease, the illness of the first story on this tour. The knot "
            "lit here is two hundred and sixty clusters, most of them Cas9 and "
            "its close kin. Yet "
            "many successful bacteria do without CRISPR. Why give up an "
            "immune system?"
        ),
    ),
    # ---------------------------------------------------------------------
    # Added after the 2026-09-16 review by the ESM Atlas team (four themes
    # requested: extremophiles, antibiotic peptides, olfactory receptors,
    # bacterial tyrosine decarboxylase). Each selector below was measured
    # against this release before its panel was written — see the knot
    # composition quoted in each comment.
    # ---------------------------------------------------------------------
    UniverseStory(
        key="Lanthipeptides",
        # A CONSTELLATION: one gene cluster, filed by job — dehydratase,
        # cyclase, immunity, precursor, each in its own place.
        constellation=True,
        title="Lanthipeptides — antibiotics stitched into rings",
        subtitle=(
            "Lanthipeptide gene clusters in eight places — dehydratase, cyclase, "
            "immunity, and a ring-stitched peptide"
        ),
        pattern="",
        # PF05147 LanC-like cyclase, PF04738/PF14028 lantibiotic dehydratase
        # (LanB), PF04604 type-A lantibiotic, PF18218 Spa1 immunity, PF19402
        # SapB precursor. Audit (2026-09-16): 722 clusters map-wide, 650 in
        # eight places of 10+ at 0.3 linkage, 7 lines totalling 35.1 units
        # (longest 10.3). By size: 352 LanC-like cyclase (Bacillota 139,
        # Actinomycetota 83) — which is why the turntable is NisC, the nisin
        # cyclase; 108 lantibiotic dehydratase N-terminus; 90 dehydratase
        # C-terminus; 33 both dehydratase halves at once; 18 NisI/SpaI
        # immunity lipoprotein; 17 Spa1 immunity C-terminal; 17 more LanC-like;
        # 15 SapB/RamS precursor peptide.
        pfam=("PF05147", "PF04738", "PF14028", "PF04604", "PF18218", "PF19402"),
        color=(0.9, 1.0, 0.55),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "Nisin is a lanthipeptide: a short peptide whose amino acids are "
            "stapled together into five rings, each one closed by a single "
            "sulfur atom. An inhibitory substance from milk streptococci was "
            "reported in 1928; it has been a commercial preservative for more "
            "than half a century, starting with the processed cheese whose packs "
            "clostridia blow open, and it is approved as a food additive in the "
            "European Union, the United States and dozens of other countries.",
            "It kills by grabbing lipid II, the brick the bacterial cell wall "
            "is built from. That blocks construction of the wall, and the same "
            "captured brick then anchors the peptide while it punches pores in "
            "the membrane: two attacks on one target.",
            "The rings are not made by the peptide. A dehydratase strips water "
            "from its serines and threonines and a cyclase closes the sulfur "
            "bridges, and in many bacteria one enzyme does both jobs.",
            # Only six clusters in the whole map carry "nisin" in their name.
            "The lines join eight places, and they are that assembly line "
            "taken apart. Two hold the cyclase that closes the rings, one of "
            "them the largest place of all. Three more hold the dehydratase "
            "that prepares them, its two halves filed separately. Two hold "
            "the immunity proteins a producer needs so its own antibiotic "
            "does not kill it. The last holds a ring-stitched peptide itself, "
            "though not nisin: SapB, which Streptomyces uses to raise aerial "
            "threads rather than to kill. Mature lantibiotics seem too short "
            "and too variable to gather in one place, and the whole map holds just six "
            "clusters carrying nisin's own name. In any one producer these "
            "genes sit side by side and switch on together. The "
            "model, which sees only sequence, has scattered them across the "
            "map by what each one does.",
        ),
        # McKay & Baldwin, Appl. Environ. Microbiol. 47:68 (1984): nisin
        # resistance on the conjugative plasmid pNP40.
        mystery=(
            "Seventy years in the food supply and resistance to nisin has "
            "never become a real problem, though resistance genes are known, "
            "one even rides a transferable plasmid, and many wild strains are "
            "naturally hard to kill with it. Part of the answer may be that nisin has "
            "never been used the way we use clinical antibiotics. Part of it is "
            "not settled."
        ),
        tags=("antibiotics", "medicine", "peptides"),
        pdb_id="2G0D",  # Nisin cyclase NisC (Li et al., Science 2006)
        narration=(
            "Lanthipeptides, antibiotics stitched into rings. Nisin is the "
            "famous one, found in 1928 and used to preserve cheese since the 1950s. "
            "It grabs lipid II, the brick the bacterial wall is built from, "
            "blocks construction, then uses that same brick as an anchor while "
            "it punches holes in the membrane. Two attacks on one target. The "
            "lines join eight places: lanthipeptide gene clusters pulled apart — "
            "dehydratase, cyclase, immunity proteins, and a peptide, "
            "scattered across the map by what each one does. "
            "Seventy years of use, and resistance has never taken hold. Is that the molecule, or just the way we use it?"
        ),
    ),
    UniverseStory(
        key="Ice-binding proteins",
        title="Ice-binding proteins — surviving the cold",
        subtitle=(
            "Thirty-two clusters of ice-binding proteins, every one of them bacterial"
        ),
        pattern="",
        # PF11999 is the DUF3494 ice-binding domain, now named "Ice-binding-
        # like" (271 clusters); PF20597 is its putative adhesive relative
        # (134) and PF21300 the grass antifreeze beta roll (106). Audit: 511
        # clusters map-wide by dominant domain, 1,118 carrying one of the
        # three, 1,287 carrying any ice or antifreeze domain; the knot is 32 at
        # r95 0.086 — Actinomycetota 20, Chloroflexota 5, Candidatus
        # Saccharimonadota 4 — and 31 of the 32 are named for ice-binding.
        #
        # DO NOT ADD PF07589 BACK. The first draft of this story included it
        # and drew a 94-cluster knot of Verrucomicrobiota and Kiritimatiellota
        # that were 90/94 bare "hypothetical protein" — because **the Atlas's
        # own `cluster_top_pfam_names` column mislabels PF07589 as
        # "Ice-binding protein, C-terminal domain"**, while InterPro and
        # current Pfam call it "PEP-CTERM protein-sorting motif"
        # (PEP_exosort_dom). PEP-CTERM is characteristic of the PVC bacteria,
        # which is exactly what that knot was, and a handful of its members
        # still carried "PEP-CTERM sorting domain" in their product names —
        # the tell we missed first time round. PF07589 supplied 2,076 of the
        # 2,587 clusters that story used to claim, so the whole knot was the
        # wrong protein family.
        #
        # Every other Pfam accession the tour uses was checked
        # against InterPro after this: 44 accessions, and PF07589 is the only
        # one where the Atlas's name disagrees. Trust the accession, not the
        # Atlas's name for it.
        pfam=("PF11999", "PF20597", "PF21300"),
        color=(0.65, 0.92, 1.0),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "Antifreeze proteins were found in Antarctic fish in 1969: proteins "
            "that keep the blood liquid at the minus one point nine degrees of "
            "ice-laden seawater, far colder than the blood's own salts alone "
            "could manage.",
            "They do not work the way an ordinary antifreeze does, by sheer "
            "weight of dissolved material. They stick to the face of a growing "
            "ice crystal and stop it spreading, which opens a gap between the "
            "temperature at which ice melts and the lower one at which it will "
            "actually grow.",
            "The most widespread ice-binding domain of all is not the fish one. "
            "It is a domain shared by bacteria, archaea, algae, fungi and "
            "diatoms, in a scattered pattern across the tree of life that is "
            "best explained by the gene being passed sideways between species "
            "rather than inherited.",
            # Audit numbers (2026-09-16).
            "Some thirteen hundred clusters in this map carry an ice-binding or "
            "antifreeze domain. The 32 lit here are all bacteria — twenty of "
            "them actinobacteria — and all but one is named for the "
            "job: ice-binding.",
        ),
        mystery=(
            "How a protein recognises ice at all — a surface made of nothing "
            "but water, ordered — is still argued over."
        ),
        tags=("cold", "metagenomics", "bacteria"),
        pdb_id="3WP9",  # Ice-binding protein, Antarctic sea-ice Colwellia sp.
        narration=(
            "Ice-binding proteins. Antifreeze proteins were found in Antarctic "
            "fish in 1969, keeping their blood liquid in water cold enough to "
            "freeze it. They do not work by sheer weight of dissolved material: "
            "they stick to the face of a growing ice crystal and stop it "
            "spreading. The most widespread ice-binding domain is not the fish "
            "one but a microbial one, scattered across bacteria, archaea, algae "
            "and fungi as if the gene had been passed sideways. The thirty-two "
            "clusters lit here are all bacterial. How a protein recognises ice, "
            "a surface of nothing but ordered water, is still argued over."
        ),
    ),
    UniverseStory(
        key="Reverse gyrase",
        title="Reverse gyrase — the enzyme of life near boiling",
        subtitle=("Fourteen clusters, the tightest knot on this tour, mostly archaeal"),
        # Selected by NAME: reverse gyrase is a FUSION of a helicase-like
        # motor and a type IA topoisomerase, and has no Pfam family of its
        # own — its parts are shared with ordinary topoisomerases and
        # helicases, so a Pfam selector would light those instead. Audit: 29
        # clusters name it map-wide in 13 components; the knot is 14 at 93%
        # ball purity with r95 0.020, the tightest measured anywhere on this
        # tour — Thermoproteota 6, phylum unrecorded 5, Methanobacteriota 1,
        # Aquificota 1, Nitrososphaerota 1.
        pattern=r"(?i)reverse gyrase",
        color=(1.0, 0.86, 0.72),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "Every cell carries enzymes that take the twist out of its DNA. "
            "Reverse gyrase does the opposite: it is the only enzyme known "
            "built to wind extra positive twist in, and it spends ATP to do it.",
            "It was found in a hot-spring archaeon in 1984. Nine years later it turned out to be a chimera of two "
            "machines: a helicase-like motor fused to a topoisomerase in one "
            "protein chain.",
            "It turns up in every organism that grows best above eighty "
            "degrees, and in a scattering of merely hot-living bacteria that "
            "appear to have borrowed the gene from archaea. Cool-living "
            "relatives do without it almost without exception.",
            "The extra twist is thought to help hold the double helix shut at "
            "temperatures that would otherwise pull it apart. That is the "
            "standard explanation and it has never been shown directly: some "
            "of these cells keep their DNA only slightly overwound or even "
            "underwound, and the enzyme also protects DNA from breaking in a "
            "way that needs no twist at all.",
            # Audit numbers (2026-09-16).
            "Only 29 clusters in 7.7 million are named for it, and the 14 lit "
            "here sit within two hundredths of a unit of their centre: the "
            "smallest and tightest knot on this tour. Six are Thermoproteota, "
            "archaea of the boiling springs.",
        ),
        mystery=(
            "Delete the gene in an archaeon that likes eighty-five degrees and "
            "it does not die, it just grows badly, worse the hotter you push "
            "it. Delete it in one that likes a hundred and it will not grow at "
            "all above ninety. Something changes across those few degrees, and "
            "nobody knows what."
        ),
        tags=("extremophiles", "DNA", "archaea"),
        pdb_id="1GKU",  # Reverse gyrase, Archaeoglobus fulgidus
        narration=(
            "Reverse gyrase, the enzyme of life near boiling. Every cell "
            "has enzymes that take the twist out of its DNA. This one does the "
            "opposite: it is the only enzyme known built to wind extra positive "
            "twist in, spending ATP to do it, and it is two machines fused into "
            "one. It appears in everything that grows best above eighty "
            "degrees, and almost nowhere else. The extra twist is thought to "
            "hold the double helix shut, though that has never been shown "
            "directly. Fourteen clusters, the tightest knot on this tour. "
            "Delete the gene and one archaeon limps; a hotter one stops "
            "growing above ninety. Nobody knows what changes."
        ),
    ),
    UniverseStory(
        key="Olfactory receptors",
        title="Olfactory receptors — the largest family in our genome",
        subtitle="A knot of 136 clusters, nearly every one of them from a vertebrate",
        pattern="",
        # PF13853, the olfactory-receptor domain: 305 clusters map-wide, knot
        # 136 at radius 0.3, r95 0.123, EVERY member Chordata (lca
        # Euteleostomi 153 over the family). Purity is only 36% because the
        # ball also holds 90 clusters whose dominant family is the generic
        # 7tm_1 (PF00001) — also vertebrate — so the neighbourhood is
        # vertebrate class A receptors at 59%; step out to radius 0.6 and the
        # vertebrate share collapses to 17%. That is the fact the last panel
        # line states.
        pfam=("PF13853",),
        color=(0.85, 0.55, 1.0),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "Something like four hundred working olfactory receptor genes sit "
            "in the human genome, beside a slightly larger number of broken "
            "copies. It is the largest family of receptor genes we have, and "
            "mice carry close to three times as many working ones.",
            "Linda Buck and Richard Axel found the family in 1991, and shared "
            "the Nobel Prize for it in 2004.",
            "Each mature sensory neuron in the nose settles on a single "
            "receptor, and a smell is read as the pattern across many of them. "
            "That combinatorial code is how a few hundred receptors cover so "
            "enormous a range of odours — and the rule is not absolute: "
            "immature neurons carry several before committing, and mosquito "
            "neurons break it outright.",
            "The first structure of a human olfactory receptor arrived only in "
            "2023, thirty-two years after the genes were found: OR51E2, caught "
            "holding propionate, the sour, cheesy acid behind Swiss cheese.",
            # Audit numbers (2026-09-16).
            # Map audit: 115 of the 136 have a vertebrate lineage; the other
            # 21 are unresolved below Chordata.
            "The knot here is 136 clusters, nearly every one from a "
            "vertebrate, and the largest of the three smell knots on this "
            "tour. Pull back a little and it dissolves into a far larger, "
            "mixed neighbourhood.",
        ),
        mystery=(
            "We can now predict fairly well what a molecule will smell like. "
            "Going the other way — reading a receptor's sequence and saying "
            "what it detects — is still mostly beyond us, and most human "
            "receptors have no known odour at all."
        ),
        tags=("senses", "receptors", "genomics"),
        pdb_id="8F76",  # Human OR51E2 with propionate (Billesbolle et al. 2023)
        # The one turntable the completeness pass could not fix by rendering
        # the biological assembly, because the assembly IS five chains and
        # only one of them is the receptor this story is about: 330 of 1,177
        # residues, 28%. The other four are the machinery used to catch it in
        # its active state — the Gs heterotrimer (Gas-mini 261, Gb1 370,
        # Gg2 71) plus Nanobody 35, which is a structural-biology tool and not
        # anything in a nose. The deposited title names only miniGs399 of the
        # four, so the caption says the ratio instead. It deliberately claims
        # no position ("the receptor on top"): `principal_frame` picks the
        # orientation from the assembly's own inertia, not from biology.
        pdb_caption=(
            "Human olfactory receptor OR51E2 with propionate — one of five "
            "chains; the rest is the Gs protein and a nanobody that trap it"
        ),
        narration=(
            "Olfactory receptors, the largest family of receptor genes we have. "
            "About four hundred working copies in the human genome, and slightly "
            "more broken ones. Linda Buck and Richard Axel found them in 1991 "
            "and won the Nobel Prize in 2004. Each mature neuron in the nose "
            "settles on one receptor, and a smell is the pattern across many of "
            "them. Yet the first structure of a human one came only in 2023, "
            "thirty-two years after the genes. This knot is a hundred and "
            "thirty-six clusters, nearly all of them vertebrate. Reading a receptor's "
            "sequence and saying what it detects is still mostly beyond us."
        ),
    ),
    UniverseStory(
        key="Insect odorant receptors",
        title="Insect odorant receptors — smell invented a second time",
        subtitle="Seventy-five arthropod clusters, ten units from the vertebrate knot",
        pattern="",
        # PF02949, the insect 7tm odorant receptor: 197 clusters map-wide,
        # knot 75, EVERY member Arthropoda, r95 0.086, centre
        # [5.02, -1.19, -5.05] against the vertebrate knot's
        # [-1.70, -7.83, -1.90] — 9.96 units apart, on a map whose 95th
        # percentile radius is 13.3. Stable: radius 0.2 and 0.3 give the same
        # 75 members and the same centre.
        pfam=("PF02949",),
        color=(0.25, 0.85, 0.75),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "Insects do not smell with our receptors, or with anything related "
            "to them. A vertebrate olfactory receptor passes its signal to a "
            "G protein; an insect odorant receptor is itself an ion channel, "
            "opening to let current through when the odorant binds.",
            "Each one works inside a four-subunit channel built around Orco, a "
            "partner so conserved that it is recognisably the same protein in "
            "flies, moths, beetles and aphids, while the receptors beside it "
            # Re-checked 2026-09-17: the 1 OR : 3 Orco asymmetric tetramer is
            # now shown in SEVERAL complexes -- Aedes and Anopheles ORs on a
            # fig-wasp Orco scaffold (Zhao et al., Science 384:1460, 2024)
            # and the pea-aphid ApOR5-Orco of this story's own turntable
            # (8Z9Z). An earlier draft said "the one complex anyone has
            # solved", which was true when written and is not now.
            "vary enormously. In every complex solved so far — mosquito, aphid "
            "and more — three Orco subunits surround a single receptor, and "
            "only that one receptor binds the odour.",
            "A fruit fly manages with about sixty odorant receptors where we "
            "have four hundred. Mosquitoes use theirs to find people: knock out "
            "Orco and a malaria mosquito largely stops being drawn to human "
            "odour, which is why this family matters to malaria.",
            # Audit numbers (2026-09-16).
            "The knot here is 75 clusters, every one an arthropod, and it sits "
            # Map audit: 9.96 units; two random clusters sit a median 11.3
            # apart, so this is a different region, not an extreme.
            "about ten units from the vertebrate knot, in a different region "
            "of the map altogether: two unrelated answers to the same problem.",
        ),
        mystery=(
            "The near-constant partner turns out to be a scaffold: it builds "
            "the channel, and whichever receptor sits beside it opens the "
            "gate. What fixes each "
            "receptor's chemical taste — and how to read that taste off its "
            "sequence — is still being worked out."
        ),
        tags=("senses", "ion channels", "insects"),
        pdb_id="8Z9Z",  # Insect OR-Orco heterocomplex, Acyrthosiphon pisum
        narration=(
            "Smell, invented a second time. Insects do not use our receptors or "
            "anything related to them. A vertebrate receptor passes its signal "
            "to a G protein. An insect odorant receptor is an ion channel: it "
            "opens and lets current through. Each works inside a four-subunit "
            "channel built around Orco, a partner recognisable in nearly every insect "
            "while the receptors beside it vary enormously. A fruit fly gets by "
            "with sixty odorant receptors where we have four hundred. This knot "
            "sits ten units from the vertebrate one, in a different region of "
            "the map. Two solutions to the same problem, filed apart."
        ),
    ),
    UniverseStory(
        key="Worm chemoreceptors",
        title="Worm chemoreceptors — smell invented a third time",
        subtitle="Fifty-five nematode clusters, the tightest of the three smell knots",
        pattern="",
        # The nematode serpentine chemoreceptor families (Srh, Srw, Srt, Srx,
        # Srd, Srv, Srz, Str, Srsx and the class-ab chemoreceptors): 366
        # clusters map-wide. RADIUS 0.2 IS DELIBERATE and must not be raised
        # to FAMILY_RADIUS: the family has several comparable components, and
        # at 0.3 the purity-weighted seed leaves the tight one (55 members,
        # r95 0.051, 95% ball purity, all Nematoda) for a diffuse 40-member
        # component at 20% purity. Measured both ways on 2026-09-16.
        pfam=(
            "PF10324",
            "PF10318",
            "PF10326",
            "PF10292",
            "PF10320",
            "PF10321",
            "PF10323",
            "PF10328",
            "PF10317",
            "PF10325",
        ),  # fmt: skip
        color=(1.0, 0.62, 0.42),
        radius=0.2,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "A millimetre-long worm, Caenorhabditis elegans, spends something "
            "like thirteen hundred of its twenty thousand genes on "
            "chemoreceptors — about seven per cent of its genome, against the "
            "two per cent we spend on smell.",
            "It has only about thirty chemosensory neurons to put them in, so "
            "each neuron has to carry many receptors at once — one of them "
            "expresses close to a hundred. That is the opposite of the rule in "
            "our own nose, where a neuron picks one.",
            "For almost all of them nobody knows what they detect. The "
            "receptor for diacetyl, the smell of butter, is one of the few "
            "that has been pinned to its odour.",
            # Audit numbers (2026-09-16). No experimental structure of a
            # nematode chemoreceptor exists, hence the relative on the left.
            "The knot lit here is 55 clusters, every one a nematode, and the "
            "tightest of the three smell knots on this tour. Not one of these "
            "receptors has ever had its structure solved — nor has any other "
            "nematode chemoreceptor. So the turntable beside this panel is "
            "not one of them: it is FSHR-1, a worm hormone receptor, the "
            "nearest thing on the shelf.",
        ),
        mystery=(
            "What the rest of those receptors are for — and why an animal with "
            "about thirty chemosensory neurons needs thirteen hundred of them — is "
            "open."
        ),
        tags=("senses", "receptors", "nematodes"),
        pdb_id="8W1Z",  # A C. elegans family-1 GPCR: the nearest solved relative
        # The panel says no chemoreceptor has ever been solved and the
        # turntable then shows a structure, which reads as a contradiction
        # until you reach the panel's last clause — and a viewer reads the
        # CAPTION, which was the deposited title, "Structure of a LGR dimer
        # from Caenorhabditis elegans in apo state". Nothing there says it is
        # a stand-in. So the caption carries the disclaimer now, and names the
        # protein: 8W1Z is FSHR-1 (UniProt G5EG04), the worm's orthologue of
        # the follicle-stimulating-hormone receptor — a leucine-rich-repeat
        # GPCR, genuinely a hormone receptor.
        #
        # The absence is verified, not assumed: an RCSB search for
        # Caenorhabditis elegans against "chemoreceptor" and against
        # "serpentine receptor" returns zero entries, as does Nematoda-wide
        # "chemoreceptor" (2026-09-17). The one "odorant receptor" hit is a
        # false positive (3UA4, an arginine methyltransferase).
        pdb_caption=(
            "FSHR-1, a C. elegans hormone receptor — a stand-in, because no "
            "nematode chemoreceptor has ever been solved"
        ),
        narration=(
            "Smell, invented a third time. A millimetre-long worm spends "
            "something like thirteen hundred of its twenty thousand genes on "
            "chemoreceptors, a far bigger share of its genome than we spend on "
            "smell. It has only about thirty chemosensory neurons to put them in, so "
            "each neuron carries many receptors at once, the opposite of the "
            "rule in our nose. For almost all of them nobody knows what they "
            "detect. This is the tightest of the three smell knots, and not one "
            "of these receptors has ever had its structure solved. Why does a "
            "worm need thirteen hundred of them?"
        ),
    ),
    UniverseStory(
        key="TnpB and Fanzor",
        title="TnpB and Fanzor — the family Cas12 grew out of",
        subtitle=(
            "Six hundred clusters of the RNA-guided nucleases that jumping genes carry"
        ),
        pattern="",
        # PF07282 is the domain TnpB shares with the compact Cas12f nucleases.
        # NOTE WHAT IT IS: a zinc-ribbon target-nucleic-acid-binding (TNB)
        # module — the DNA grip, NOT the scissors. The nuclease is the RuvC
        # region, covered by PF01385. An earlier draft of this comment called
        # PF07282 a nuclease domain, which a structural biologist would catch.
        #
        # RADIUS 0.2, not FAMILY_RADIUS: at 0.3 the ball reaches 821 clusters
        # at 78% purity, at 0.2 it is 624 at 94%. Audit (2026-09-17): 1,575
        # clusters map-wide; the knot centres at [-14.22, 0.68, 1.24] with
        # Bacillota 228, Cyanobacteriota 103, Pseudomonadota 81,
        # Actinomycetota 60 and 42 viral clusters. Only four of the 624 are
        # eukaryotic — the Fanzors proper sit about 0.6 units away, which is
        # why the panel places them nearby rather than inside.
        #
        # Map audit: 435 of the map's 459
        # Cas12-named clusters and 830 of its 1,243 TnpB-named clusters lie
        # within one unit of this spot, so the neighbourhood is real in our
        # data and not just in theirs.
        pfam=("PF07282",),
        color=(0.7, 0.25, 0.75),
        radius=0.2,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            # ISDra2 TnpB is 408 aa (UniProt Q7DF80); SpCas9 is 1,368
            # (Q99ZW2). Karvelis et al., Nature 599:692 (2021) and
            # Altae-Tran et al., Science 374:57 (2021) established the
            # RNA-guided activity; TnpB genes ride in IS200/IS605 AND IS607
            # elements, so "jumping genes" rather than a named family.
            "TnpB is a small bacterial enzyme, about four hundred amino acids "
            "— roughly a third the size of the CRISPR protein Cas9 — that "
            "jumping genes carry around with them. It is handed a short piece "
            "of RNA as a search template and cuts DNA wherever that template "
            "matches.",
            # Altae-Tran, Shmakov, Makarova, Wolf, Kannan, Zhang & Koonin,
            # PNAS 120:e2308224120 (2023): "TnpB appears to be the
            # evolutionary ancestor of Cas12". Cas12 is polyphyletic, which is
            # why this says stock-and-recruitment rather than "TnpB evolved
            # into Cas12".
            #
            # The abstract: "the evolution of type V CRISPR-Cas effectors on
            # about 50 independent occasions".
            "Enzymes like it are the ancestral stock the CRISPR Cas12 editors "
            "arose from — and not once: this kind of jumping-gene protein was "
            "recruited into CRISPR systems about fifty separate times.",
            # Saito et al., Nature 620:660 (2023); Jiang et al., Science
            # Advances 9:eadk0171 (2023), which found Fanzor2 enriched in
            # Mimiviridae, Phycodnaviridae and Ascoviridae — all
            # Nucleocytoviricota. Viruses are not a domain of life, hence the
            # phrasing.
            "Their relatives in cells with nuclei, called Fanzors, turn up in "
            "chytrid fungi, in algae, in amoebae and in clams — and in the "
            "giant viruses that prey on single-celled hosts. One family, "
            "across all three domains of life and the viruses that infect "
            "them.",
            # Audit numbers (2026-09-17).
            "This knot is 624 clusters, nearly all bacterial, with forty-two "
            "viral ones among them. Only four are eukaryotic: the Fanzors "
            "proper sit a fraction of a unit away, in the same "
            "neighbourhood.",
            # The preprint, Appendix A.5.5, verified verbatim against the PDF:
            # "The cluster centroid has 0.87 Jaccard similarity in SAE feature
            # space and a TM-score of 0.59 to the canonical ISDra2 TnpB,
            # despite a sequence identity of only 13.9%." The Jaccard figure
            # is deliberately NOT quoted: it is an internal metric of this
            # model with no community threshold, and printing it beside a
            # TM-score would imply the two are equally calibrated. TM > 0.5 =
            # same fold (Xu & Zhang, Bioinformatics 26:889, 2010); 13.9%
            # identity is below Rost's 20-35% twilight zone and close to the
            # 8-9% expected of unrelated sequences (Rost 1997, 1999).
            "Sequence alone cannot see the kinship. The work behind this map "
            "reports a cluster whose centre folds like the best-studied TnpB "
            "— a structural match of 0.59, where anything above 0.5 means the "
            "same fold — while sharing under fourteen per cent of its "
            "letters, deep in the range where sequence alone cannot tell "
            "relatives from strangers.",
        ),
        # Preprint, Appendix A.5.5: the search against 1,927 Cas12/TnpB
        # cluster representatives "yielded 315 'dark' clusters with >= 0.6
        # similarity to any Cas12/TnpB". Verified verbatim.
        mystery=(
            "Three hundred and fifteen clusters with no annotation at all sit "
            "close enough in the model's feature space to belong to this "
            "neighbourhood. How many more RNA-guided systems are waiting in "
            "them, nobody knows."
        ),
        tags=("genome editing", "mobile elements", "evolution"),
        # ISDra2 TnpB with its reRNA — Sasnauskas et al., Nature 616:384
        # (2023), whose title is the argument for this choice: "TnpB structure
        # reveals minimal functional core of Cas12 nuclease family". Two
        # polymer entities (protein + guide RNA), 2.80 A, and the
        # most-studied TnpB, so it represents the shared core of a knot made
        # of TnpB and Cas12f-like proteins. It is the DNA-FREE state, which is
        # why the narration says "before it has found a target". The
        # Spizellomyces Fanzor structure (9CEU) was the alternative and was
        # rejected: the eukaryotic Fanzors are 4 of the 624 clusters lit, so
        # it would misrepresent the knot, and that entry is an MBP fusion.
        pdb_id="8BF8",
        narration=(
            "The family Cas12 grew out of. TnpB is a small bacterial "
            "enzyme, a third the size of Cas9, that jumping genes carry "
            "around with them: hand it a short piece of RNA and it cuts DNA "
            "wherever that template matches. Enzymes like it are the stock the "
            "Cas12 editors arose from, recruited about fifty separate times. "
            "Their relatives in cells with nuclei turn up in fungi, algae, "
            "amoebae, clams, and in giant viruses. Six hundred clusters are "
            "lit here, nearly all bacterial, the eukaryotic Fanzors just "
            "beside them, and the structure turning alongside is the "
            "best-studied one, caught holding its RNA guide before it has "
            "found a target."
        ),
    ),
    UniverseStory(
        key="Levodopa and the gut",
        title="Tyrosine decarboxylase — the gut enzyme that eats a Parkinson's drug",
        subtitle="Fifty-two clusters in about a dozen specks, never a family of its own",
        # Name match plus PF21391, the tyrosine decarboxylase C-terminal
        # domain (19 clusters). Audit: 52 clusters, and they are NOT a family
        # in this map — the densest ball of them holds 21 at 4% PURITY inside
        # the group II PLP decarboxylase family (PF00282, 1,444 clusters as
        # dominant family; it holds the glutamate, histidine and dopa
        # decarboxylases). Members sit a median 2.43 units
        # apart pairwise and the phyla are scattered (Pseudomonadota 11,
        # Bacillota 11, Methanobacteriota 5, Actinomycetota 4, Ascomycota 4,
        # Streptophyta 4; 8 archaeal in total). Eight are named MfnA, the
        # archaeal methanofuran-pathway enzyme, which decarboxylates tyrosine
        # for an unrelated purpose. So this is a SCATTER story: light every
        # one of the 52 and let the absence of a family be the point. A knot
        # story here would claim a family the map does not show.
        #
        # WHAT THE PICTURE ACTUALLY SHOWS, and why the panel says "about a
        # dozen specks" rather than fifty-two countable dots (measured
        # 2026-09-16, after the owner asked why this stop has neither a bubble
        # nor a constellation). The members' median NEAREST-neighbour distance
        # is 0.022 units — a fifth of SCATTER_HIGHLIGHT_RADIUS — so merging
        # anything closer than one marker diameter leaves 13 VISIBLE BLOBS of
        # sizes [21, 13, 4, 3, 3, 1x8]. No marker radius fixes that: resolving
        # a 0.022 gap needs a radius smaller than a knot's, which is invisible
        # at this story's 38-unit framing. Their bounding box is also 15.3
        # units against the map's 89.4-unit diagonal — 17% of it — so the
        # earlier "scattered across the whole map" was wrong twice over. The
        # text now describes the dozen specks and the fifth of the cloud.
        #
        # NOT a constellation, though two of its components clear the
        # 10-member floor (21 and 13): a single line joining them would assert
        # exactly the family relation the whole story is about NOT finding.
        # No bubble either, for the same reason — a bubble says "the story is
        # this blob", and there is no blob.
        pattern=r"(?i)tyrosine decarboxylase",
        pfam=("PF21391",),
        whole=True,
        scatter=True,
        color=(1.0, 1.0, 0.95),
        frame_fraction=1.0,
        flight_ms=3000,
        facts=(
            "Levodopa is the mainstay of Parkinson's treatment, and it only "
            "works if it reaches the brain. Gut bacteria carrying tyrosine "
            "decarboxylase convert it to dopamine on the way — in the gut, "
            "where it is no longer any use.",
            # Re-checked 2026-09-17. What the literature SHOWS is qualitative:
            # carbidopa "did not affect gut bacterial l-dopa decarboxylation"
            # in complex human gut communities (Maini Rekdal et al., Science
            # 364:eaau6323, 2019), because the human-AADC inhibitors are
            # substrate analogues that do not inhibit bacterial TyrDC (van
            # Kessel et al., Nat. Commun. 10:310, 2019). Potency: carbidopa is
            # 200x less active on E. faecalis TyrDC (IC50, Maini Rekdal) and
            # ~1.5x10^4 less potent on bacterial TDCs (van Kessel).
            "Patients are given a second drug, carbidopa, to block the human "
            "version of that reaction. It does not block the bacterial one — "
            "in human gut communities it leaves the bacterial conversion "
            "untouched — so the bacteria go on eating theirs.",
            # Audit numbers (2026-09-16): the scatter is the finding.
            "Now look at what this map does with the enzyme. Fifty-two "
            "clusters out of 7.7 million carry its name or its signature "
            "domain, and they never "
            "make a family of their own. They sit in about a dozen specks "
            "strung across a fifth of the cloud, and the two largest hold "
            "twenty-one and thirteen clusters packed so tightly that each "
            "draws as a single point. Eleven are in one bacterial phylum, "
            "eleven in another, and eight are an archaeal enzyme doing the "
            "same chemistry for an unrelated purpose.",
            "That scatter is the clinical problem in miniature. The enzyme sits "
            "in a family shared by some fourteen hundred other clusters of "
            "decarboxylases, among them the ones for glutamate and histidine "
            "and our own dopa decarboxylase, the enzyme carbidopa blocks — so "
            "finding it in a patient's "
            "gut means telling it apart from all of them.",
        ),
        mystery=(
            "How much of the difference between patients' responses to "
            "levodopa comes down to which bacteria they carry — and whether "
            "profiling this one enzyme could guide a dose — is open, and "
            "clinically live."
        ),
        tags=("medicine", "microbiome", "neurology"),
        pdb_id="5HSJ",  # Tyrosine decarboxylase with PLP, Lactobacillus brevis
        narration=(
            "Tyrosine decarboxylase, the gut enzyme that eats a Parkinson's "
            "drug. Levodopa only works if it reaches the brain, and gut "
            "bacteria carrying this enzyme convert it to dopamine on the way, "
            "in the gut, where it is wasted. Patients take a second drug to "
            "block the human version of that reaction, but it does not block "
            "the bacterial one. And look what the map does with it: fifty-two "
            "clusters out of seven point seven million, never a family of "
            "their own, sitting in about a dozen specks inside a family shared "
            "by some fourteen hundred other decarboxylases. That scatter is the "
            "clinical problem in miniature."
        ),
    ),
)

#: The order the tour walks, and the reason for it. The authoring blocks above
#: are grouped by where each story CAME FROM (the seven carried from the
#: Swiss-Prot tour, then the map-only ones, then the batch the ESM Atlas team
#: asked for); that is provenance, not a narrative, and the tour should be a
#: narrative. Five movements, each handing off to the next:
#:
#: 1-6   The machinery every cell runs on, opening on the one protein everyone
#:       already knows. Hemoglobin's oxygen leads to where oxygen came from,
#:       and photosystem II hands straight to RuBisCO — the two halves of
#:       photosynthesis, light and carbon — before energy, repair and rescue.
#: 7-9   The map's own geography, which only this dataset can show: a spur
#:       flung off the cloud, the quarter of it nobody has characterised, and
#:       the phage clusters, half of them inside that dark. (This trio must
#:       stay in the order spur, dark, phage — the dark story sets up how much
#:       of it is phage; a test pins it.)
#: 10-14 The arms race, chained: what a virus wears on its surface, the
#:       immune system bacteria evolved against viruses, the jumping-gene
#:       nuclease that immune system's scissors were recruited from — the
#:       owner asked for these two to be adjacent and they are, CRISPR first
#:       because Cas12 links the CRISPR system to its TnpB ancestors — then
#:       the chemical war, resistance before the weapon it defeats.
#: 15-19 Life at the edges, then the same trick invented three times: ice and
#:       boiling water, then smell in vertebrates, insects and nematodes.
#:       (The smell trio must stay in that order — each narration counts "a
#:       second time", "a third time".)
#: 20    The closer, and the only stop that is not a knot at all: a gut enzyme
#:       that eats a Parkinson's drug, so scattered that the absence of a
#:       family IS the story. The tour ends on the map admitting a limit.
TOUR_ORDER: tuple[str, ...] = (
    "Hemoglobin",
    "Photosystem II",
    "RuBisCO",
    "ATP synthase",
    "Hsp70",
    "RecA and Rad51",
    "ABC transporters",
    "Dark proteome",
    "Phage",
    "Viral surface proteins",
    "CRISPR-Cas",
    "TnpB and Fanzor",
    "Beta-lactamases",
    "Lanthipeptides",
    "Ice-binding proteins",
    "Reverse gyrase",
    "Olfactory receptors",
    "Insect odorant receptors",
    "Worm chemoreceptors",
    "Levodopa and the gut",
)


def _ordered(
    pool: tuple[UniverseStory, ...], order: tuple[str, ...]
) -> tuple[UniverseStory, ...]:
    """The pool walked in ``order``, which must name every story exactly once."""
    by_key = {s.key: s for s in pool}
    missing = sorted(set(by_key) - set(order))
    unknown = sorted(set(order) - set(by_key))
    if missing or unknown:
        raise ValueError(
            f"TOUR_ORDER must be a permutation of the story pool: "
            f"missing {missing}, unknown {unknown}"
        )
    if len(order) != len(set(order)):
        raise ValueError("TOUR_ORDER repeats a story key")
    return tuple(by_key[k] for k in order)


STORIES: tuple[UniverseStory, ...] = _ordered(_STORY_POOL, TOUR_ORDER)

OVERVIEW_TITLE = "Twenty stories in the protein universe"
ATTRIBUTION = f"{DEMO_META['citation']['ref']} · {DEMO_META['citation']['license']}"
OVERVIEW_HTML = (
    "Every point is one of {n:,} clusters of proteins from the ESM Atlas: 6.8 "
    "billion sequences, most of them read straight out of soil, seawater and "
    "guts rather than from any organism grown in a lab, grouped by the features "
    "a protein language model sees in them and laid out in 3D with UMAP so that "
    "similar clusters sit close together. Colours are the main branches of "
    "life; the dim points are clusters nobody has characterised."
    "<br><br>Step the <b>story</b> dimension to fly to twenty places that "
    "each tell a piece of biology: blood, sunlight, a famously slow "
    "enzyme, the cell's turbine, the oldest chaperone, the machine that mends "
    "DNA; then a spur flung off the map, the dark proteome and the phage "
    "universe; then the coronavirus spike, CRISPR, the jumping-gene scissors "
    "Cas12 grew out of, the enzyme that beats penicillin and the antibiotics "
    "bacteria stitch into rings; then life in ice and life in boiling water "
    "and three separate inventions of the sense of smell; and last, a gut "
    "enzyme that eats a Parkinson's drug."
)
#: Spoken introduction at the Overview slot.
OVERVIEW_NARRATION = (
    "Every point here is a cluster of related proteins: seven point seven million of "
    "them, drawn from nearly seven billion sequences, most read straight out "
    "of the environment, and grouped by a language model so that similar "
    "proteins sit close together. The dim points are clusters nobody has "
    "characterised. Twenty places on this map hide a story. Step through "
    "them."
)
UNIREF_LINK = "https://www.uniprot.org/uniref?query={hover_key}"
NARRATION_CACHE_DIR = CACHE_DIR / "narration"
AMBISONIC_BED_CACHE_DIR = CACHE_DIR / "ambisonic"


# =============================================================================
# Selection (unit-tested)
# =============================================================================

#: Candidate knot seeds are subsampled to this many members: exact for every
#: family story, and far denser than any knot for the million-member regions.
MAX_SEED_CANDIDATES = 20_000


def spur_mask(positions: np.ndarray) -> np.ndarray:
    """The spur: far points within :data:`SPUR_HALF_ANGLE_DEG` of their mean direction.

    The mean direction of everything beyond :data:`SPUR_RADIUS` is the spur's
    axis (the streak dominates that set), and the cone around it drops the
    stragglers that sit equally far out in other directions.
    """
    radius = np.linalg.norm(positions, axis=1)
    far = radius > SPUR_RADIUS
    if not far.any():
        return far
    axis = positions[far].mean(axis=0)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        raise ValueError("far clusters have no dominant direction for the spur")
    axis /= axis_norm
    cos = (positions @ axis) / np.maximum(radius, 1e-9)
    return far & (cos > np.cos(np.radians(SPUR_HALF_ANGLE_DEG)))


def densest_core(
    family_pos: np.ndarray,
    radius: float,
    all_tree: Any,
    *,
    seed: int = 0,
) -> np.ndarray:
    """The member whose ``radius``-ball holds the most members, weighted by purity.

    Score = members within the ball × (members / all clusters within the ball).
    Purity matters here because a region story's members (a quarter of the map
    for the dark proteome) are everywhere: the raw count would land on the
    densest place in the map, the purity weight lands on the darkest.
    """
    spatial = require_module("scipy.spatial")
    n = len(family_pos)
    if n == 1:
        return family_pos[0]
    if n > MAX_SEED_CANDIDATES:
        rng = np.random.default_rng(seed)
        cand = family_pos[rng.choice(n, MAX_SEED_CANDIDATES, replace=False)]
    else:
        cand = family_pos
    fam_tree = spatial.cKDTree(family_pos)
    fam_count = np.asarray(
        fam_tree.query_ball_point(cand, radius, return_length=True), dtype=np.float64
    )
    all_count = np.asarray(
        all_tree.query_ball_point(cand, radius, return_length=True), dtype=np.float64
    )
    score = fam_count * fam_count / np.maximum(all_count, 1.0)
    return cand[int(np.argmax(score))]


def family_mask(
    story: UniverseStory, universe: Universe, name_mask: np.ndarray | None
) -> np.ndarray:
    """Which clusters belong to the story before the knot cut (see the class doc)."""
    mask = np.zeros(len(universe), dtype=bool)
    if story.pfam:
        mask |= universe.pfam_mask(story.pfam)
    if story.pattern:
        if name_mask is None:
            raise ValueError(f"story {story.key!r} selects by name but no names given")
        mask |= name_mask
    if story.region == "spur":
        mask |= spur_mask(universe.positions)
    elif story.region == "dark":
        mask |= universe.dark_mask()
    elif story.region == "phage":
        mask |= universe.phylum_mask("Uroviricota")
    elif story.region is not None:
        raise ValueError(f"story {story.key!r}: unknown region {story.region!r}")
    if story.groups:
        names = universe.phylum_names()
        group_of = {p: phylum_group(str(p)) for p in np.unique(names)}
        in_group = np.array([group_of[p] in story.groups for p in names])
        mask &= in_group
    return mask


def select_universe_members(
    story: UniverseStory,
    universe: Universe,
    mask: np.ndarray,
    all_tree: Any,
    *,
    figure: Constellation | None = None,
) -> StoryCluster:
    """Resolve a story to the clusters of its densest knot — or, for a
    ``whole`` story, to every matching cluster, or, for a constellation, to
    every member of every place its figure joins.

    A knot story seeds on the member with the most (and purest) member
    neighbours within two thirds of ``story.radius``, keeps the members within
    ``story.radius`` of it, and re-centres on their bounding box (see
    :func:`~luxar.demos.demo_esm3_protein_stories.cluster_geometry`).

    A constellation takes no knot cut: its members are exactly the union of
    its places, so the lit clusters and the line endpoints are the same
    geometry and cannot drift apart.
    """
    n_named = int(mask.sum())
    if n_named == 0:
        raise ValueError(f"story {story.key!r}: no cluster matched its selector")
    if story.constellation:
        if figure is None:
            raise ValueError(f"story {story.key!r} is a constellation: pass its figure")
        centre, radial = cluster_geometry(universe.positions[figure.members])
        return StoryCluster(
            indices=figure.members,
            centre=centre,
            r95=float(np.percentile(radial, 95)),
            n_named=n_named,
            r50=float(np.percentile(radial, 50)),
            r_max=float(radial.max()),
        )
    if story.whole:
        indices = np.flatnonzero(mask)
        centre, radial = cluster_geometry(universe.positions[indices])
        return StoryCluster(
            indices=indices,
            centre=centre,
            r95=float(np.percentile(radial, 95)),
            n_named=n_named,
            r50=float(np.percentile(radial, 50)),
            r_max=float(radial.max()),
        )
    seed = densest_core(
        universe.positions[mask].astype(np.float64), story.radius * 2.0 / 3.0, all_tree
    )
    ball = np.asarray(all_tree.query_ball_point(seed, story.radius), dtype=np.int64)
    indices = ball[mask[ball]]
    if len(indices) == 0:
        raise ValueError(f"story {story.key!r}: no member within radius {story.radius}")
    indices = np.sort(indices)
    # Bounding-box centre + enclosing radius: the bubble holds every member
    # and sits on the knot, not on its dense half (see the stories tour).
    centre, radial = cluster_geometry(universe.positions[indices])
    return StoryCluster(
        indices=indices,
        centre=centre,
        r95=float(np.percentile(radial, 95)),
        n_named=n_named,
        r50=float(np.percentile(radial, 50)),
        r_max=float(radial.max()),
    )


def whole_highlight_sample(indices: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """The members a whole-map highlight draws: all of them, or a fixed sample."""
    if len(indices) <= WHOLE_HIGHLIGHT_MAX_POINTS:
        return indices
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(indices), WHOLE_HIGHLIGHT_MAX_POINTS, replace=False)
    return np.sort(indices[pick])


def highlight_unit(n_members: int, n_drawn: int) -> str:
    """The panel's count unit: ``clusters``, or the sampling ratio when drawn short."""
    if n_drawn >= n_members:
        return "clusters"
    return f"clusters (one in {round(n_members / n_drawn)} drawn)"


def scatter_camera(
    cluster: StoryCluster,
    story: UniverseStory,
    global_centre: np.ndarray,
    map_radius: float,
) -> CameraConfig:
    """The pose for a SCATTER story: the whole map, seen from the members' side.

    A scatter story claims its members are spread across the cloud, so the
    picture has to contain the cloud. Framing it like any other story does not:
    the bounding-box centre of 52 clusters is a biased point (measured 7.6
    units off the origin for tyrosine decarboxylase), and aiming there put the
    map in the right half of the frame with dead space beside it. So the camera
    targets the MAP's centre and pulls back to hold ``map_radius``, keeping
    only the direction from the members' own side of the cloud — which makes it
    a different shot from the Overview rather than a repeat of it.
    """
    outward = cluster.centre - global_centre
    norm = float(np.linalg.norm(outward))
    direction = (
        outward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    ) + np.array([0.0, 0.35, 0.0])
    direction /= np.linalg.norm(direction)
    half_height_per_unit = np.tan(np.radians(STORY_LENS_FOV_DEG) / 2)
    distance = max(
        story.min_distance,
        map_radius / (story.frame_fraction * float(half_height_per_unit)),
    )
    position = global_centre + direction * distance
    return CameraConfig(
        position=tuple(float(v) for v in position),
        target=tuple(float(v) for v in global_centre),
        up=(0.0, 1.0, 0.0),
    )


def constellation_camera(figure: Constellation, story: UniverseStory) -> CameraConfig:
    """The pose for a CONSTELLATION story: the whole figure, seen face-on.

    Two decisions, both forced by what a constellation is.

    *Distance* is set from ``r_max`` — the furthest member from the target —
    rather than from the projected extent, because ``r_max`` bounds the
    figure's apparent radius from EVERY direction. The tour runs with
    auto-rotate on, which keeps the target and the distance from this pose but
    supplies its own direction, so a distance computed for one viewing angle
    would let the outermost node swing out of frame at another. At
    :data:`CONSTELLATION_FRAME_FRACTION` the figure spans 80% of the frame
    height in its WORST orientation.

    *Direction* is the figure's thinnest principal axis (see
    :class:`Constellation`), which is the one view that sees the places spread
    out rather than stacked behind one another. It is what a visitor gets when
    they stop the spin, and what a still capture shows. No upward lift: the
    lift the knot stories use to avoid a dead-level shot would tilt the figure
    back off the face-on view that is the entire point.
    """
    half_height_per_unit = float(np.tan(np.radians(STORY_LENS_FOV_DEG) / 2))
    distance = max(
        story.min_distance,
        figure.r_max / (CONSTELLATION_FRAME_FRACTION * half_height_per_unit),
    )
    position = figure.centre + figure.view * distance
    # The view axis is data-driven and may come out near-vertical, which would
    # make the default up-vector degenerate; fall back to +z there.
    up = (0.0, 1.0, 0.0) if abs(float(figure.view[1])) < 0.95 else (0.0, 0.0, 1.0)
    return CameraConfig(
        position=tuple(float(v) for v in position),
        target=tuple(float(v) for v in figure.centre),
        up=up,
    )


def universe_story_camera(
    cluster: StoryCluster,
    story: UniverseStory,
    global_centre: np.ndarray,
    *,
    map_radius: float | None = None,
    figure: Constellation | None = None,
) -> CameraConfig:
    """The waypoint pose: outside-in, side-on, scatter, or constellation.

    Side-on keeps the tour's distance rule and swaps the direction for one
    perpendicular to the centre→cluster ray (the horizontal perpendicular,
    lifted a little), so an elongated radial feature is seen across, not along.
    A scatter story frames the whole map instead (see :func:`scatter_camera`),
    which needs ``map_radius``. A constellation frames its whole figure (see
    :func:`constellation_camera`), which needs ``figure``.
    """
    if story.constellation:
        if figure is None:
            raise ValueError(
                f"story {story.key!r} is a constellation, which frames its whole "
                "figure: pass figure"
            )
        return constellation_camera(figure, story)
    if story.scatter:
        if map_radius is None:
            raise ValueError(
                f"story {story.key!r} is a scatter, which frames the whole map: "
                "pass map_radius"
            )
        return scatter_camera(cluster, story, global_centre, map_radius)
    if not story.side_on:
        return story_camera(cluster, story, global_centre, min_radius=BUBBLE_MIN_RADIUS)
    outward = cluster.centre - global_centre
    side = np.cross(outward, np.array([0.0, 1.0, 0.0]))
    norm = float(np.linalg.norm(side))
    side = side / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    direction = side + np.array([0.0, 0.35, 0.0])
    direction /= np.linalg.norm(direction)
    distance = story_camera_distance(cluster, story, min_radius=BUBBLE_MIN_RADIUS)
    position = cluster.centre + direction * distance
    return CameraConfig(
        position=tuple(float(v) for v in position),
        target=tuple(float(v) for v in cluster.centre),
        up=(0.0, 1.0, 0.0),
    )


def spur_pfam_fraction(universe: Universe, cluster: StoryCluster) -> float:
    """Share of a cluster's members whose dominant Pfam family is an ABC part."""
    return float(universe.pfam_mask(SPUR_PFAMS)[cluster.indices].mean())


def check_spur_story(universe: Universe, cluster: StoryCluster) -> float:
    """Raise unless the spur's members are ABC-transporter-dominated; return the share."""
    share = spur_pfam_fraction(universe, cluster)
    if share < SPUR_PFAM_MIN_FRACTION:
        raise ValueError(
            f"the spur (r > {SPUR_RADIUS:g}) is only {share:.0%} ABC-transporter "
            f"Pfam families {SPUR_PFAMS}; the panel claims that family (needs >= "
            f"{SPUR_PFAM_MIN_FRACTION:.0%}). Re-measure this Atlas release."
        )
    return share


def overview_panel_html(n_clusters: int) -> str:
    """The overview panel shown at story 0."""
    return (
        '<div style="font-size:1.3vh;line-height:1.35;color:#e8e8e8;'
        "background:rgba(0,0,0,0.62);padding:1.4vh 1.6vh;border-radius:6px;"
        'border-left:0.5vh solid #ffffff">'
        f'<div style="font-size:2.2vh;font-weight:bold;margin-bottom:0.6vh">'
        f"{html.escape(OVERVIEW_TITLE)}</div>"
        f"{OVERVIEW_HTML.format(n=n_clusters)}"
        f'<div style="margin-top:1.1vh;font-size:1.05vh;color:rgba(232,232,232,0.55)">'
        f"{html.escape(ATTRIBUTION)}</div>"
        "</div>"
    )


def member_labels(
    names: list[str], phyla: list[str], lcas: list[str], unirefs: list[str]
) -> tuple[list[str], list[str]]:
    """Hover labels and link keys for a story's members.

    Label: ``product name — taxon`` (the dominant phylum, else the lowest common
    ancestor the table records, else nothing). Key: the UniRef match, which the
    link searches for; a cluster without one searches its product name instead.
    """
    labels, keys = [], []
    for name, phylum, lca, uniref in zip(names, phyla, lcas, unirefs, strict=True):
        taxon = phylum or (lca.split(":", 1)[-1] if lca else "")
        labels.append(f"{name} — {taxon}" if taxon else name)
        keys.append(uniref or name)
    return labels, keys


# =============================================================================
# Scene construction
# =============================================================================


def _member_details(
    annotations: Path, universe: Universe, clusters: dict[int, StoryCluster]
) -> dict[int, tuple[list[str], list[str]]]:
    """Read the label columns for every labelled story's members (one parquet pass).

    ``clusters`` maps story slot → cluster for the stories with few enough
    members to label: every knot story, plus a scatter story's few dozen. A
    whole-map highlight of two million points carries no hover labels — the
    strings would dominate the store — hence
    :data:`LABELLED_MEMBER_CAP`.
    """
    pq = require_module("pyarrow.parquet")
    if not clusters:
        return {}
    rows = np.unique(
        np.concatenate([universe.annotation_row[c.indices] for c in clusters.values()])
    )
    rows = rows[rows >= 0]
    table = pq.read_table(annotations, columns=list(_MEMBER_COLUMNS)).take(rows)
    cols = {c: table.column(c).to_pylist() for c in _MEMBER_COLUMNS}
    lookup = {int(r): i for i, r in enumerate(rows)}
    phyla = universe.phylum_names()
    out: dict[int, tuple[list[str], list[str]]] = {}
    for k, c in clusters.items():
        idx = [lookup.get(int(r), -1) for r in universe.annotation_row[c.indices]]
        names = [
            str(cols["product_name"][i] or "cluster") if i >= 0 else "cluster"
            for i in idx
        ]
        lcas = [str(cols["lca_taxonomy"][i] or "") if i >= 0 else "" for i in idx]
        unirefs = [
            str(cols["uniref_match_accession"][i] or "") if i >= 0 else "" for i in idx
        ]
        out[k] = member_labels(names, [str(p) for p in phyla[c.indices]], lcas, unirefs)
    return out


def _name_masks(
    annotations: Path, universe: Universe, stories: tuple[UniverseStory, ...]
) -> dict[str, np.ndarray]:
    """Per-point product-name matches for the stories that select by pattern."""
    pq = require_module("pyarrow.parquet")
    pc = require_module("pyarrow.compute")
    patterns = {s.key: s.pattern for s in stories if s.pattern}
    if not patterns:
        return {}
    names = pq.read_table(annotations, columns=["product_name"]).column("product_name")
    valid = universe.annotation_row >= 0
    out: dict[str, np.ndarray] = {}
    for key, pattern in patterns.items():
        row_hit = pc.match_substring_regex(names, pattern).to_numpy(
            zero_copy_only=False
        )
        row_hit = np.asarray(row_hit, dtype=bool)
        mask = np.zeros(len(universe), dtype=bool)
        mask[valid] = row_hit[universe.annotation_row[valid]]
        out[key] = mask
    return out


def _render_story_turntables(
    stories: tuple[UniverseStory, ...], output_path: Path, cache: Path | None
) -> dict[str, TurntableAssets]:
    with asection("Rendering PDB turntables"):
        cache_dir = cache or (Path.home() / ".cache" / "luxar" / TURNTABLE_CACHE)
        # Hints only for a durable store: on the serve path the output lives
        # in a TemporaryDirectory that is gone before anyone could bake it.
        environment = _turntable_environment(output_path)
        assets = render_turntables(
            [s.pdb_id for s in stories if s.pdb_id],
            cache_dir,
            colors={s.pdb_id: s.color for s in stories if s.pdb_id},
            environment=environment,
        )
        aprint(
            f"{len(assets)} of {sum(1 for s in stories if s.pdb_id)} turntables ready"
        )
        return assets


def _viewer_config(
    waypoints: list[Waypoint],
    overview: tuple[float, float, float],
    *,
    auto_rotate: bool,
    audio: bool,
    high_quality: bool = False,
    control_panel: ControlPanelConfig | None = None,
) -> ViewerConfig:
    # Mirrors the Swiss-Prot tour's kiosk settings (see its build for the why),
    # except for render quality: see `high_quality` below. `overview` is the raw
    # distance-tuned pose; `pull_in` carries it to the cinematic 63° lens.
    return ViewerConfig(
        # Names the browser tab AND the control panel's header — see the same
        # note in demo_esm3_protein_stories. A filename is not a title.
        title="The protein universe",
        control_panel=control_panel,
        cinematic_mode=True,
        camera=CameraConfig(position=pull_in(overview), target=(0.0, 0.0, 0.0)),
        # Authored, not left to the slider (see the BRIGHTNESS note by
        # `BACKDROP_INTENSITY`). The Rendering Controls remember per-source
        # edits in localStorage and those WIN over this value at load; "Reset
        # to Defaults" comes back here.
        exposure=BACKDROP_EXPOSURE_STOPS,
        auto_rotate=auto_rotate,
        auto_rotate_speed=0.5 if auto_rotate else None,
        auto_rotate_axis="world-y" if auto_rotate else None,
        # A slow breath in and out under the turntable, dialled in on the kiosk
        # display: the map reads as a volume rather than a flat cloud, and the
        # near extreme brings the knots close enough to read. Gated on
        # `auto_rotate` because that flag is what `--no-auto-rotate` uses to ask
        # for a still camera, and half a motion is worse than none.
        #
        # NOTE the cost, which is real (see `auto_dolly_amplitude_percent`):
        # screen area goes as 1/d^2, so the 95% kiosk swing makes the LOD ladder
        # load finer levels at the near extreme. The hosted/laptop build uses a
        # gentler breath; the kiosk serves its larger swing from a warm local
        # cache.
        auto_dolly=auto_rotate,
        auto_dolly_amplitude_percent=(95.0 if high_quality else 20.0)
        if auto_rotate
        else None,
        auto_dolly_period=58.5 if auto_rotate else None,
        # Render quality (2026-09-10 review): supersampling and rendering above
        # CSS resolution are what make the kiosk build crisp, and also what made
        # it crawl on an ordinary laptop — SSAA is a 4x fragment cost on top of
        # the 4x a 2x display already asks for. The shipped default is the
        # laptop build: SSAA off and the DPR capped at 1.0 (`allow_high_dpr`
        # False is that cap). `--high-quality` (the kiosk / big-GPU switch)
        # turns both back on.
        ssaa_enabled=high_quality,
        allow_high_dpr=high_quality,
        environment=EnvironmentConfig(source="scene", probe="auto"),
        waypoints=waypoints,
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


def _add_overlays(
    scene: Any,
    stories: tuple[UniverseStory, ...],
    clusters: list[StoryCluster],
    assets: dict[str, TurntableAssets],
    n: int,
    units: dict[int, str],
) -> None:
    scene.add_text(
        "ESM Protein Universe",
        position=(0.02, 0.02),
        font_size=0.05,
        anchor="top-left",
        color="rgba(255,255,255,0.65)",
        blend_mode="difference",
    )
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
            story_panel_html(s, len(c.indices), k, total, unit=units[k]),
            position=(0.98, 0.5),
            anchor="center-right",
            width=PANEL_WIDTH,
            visible_range={STORY_DIM: k},
            transition="fade",
            transition_duration=0.35,
        )
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
            alpha_matte="stacked",
            visible_range={STORY_DIM: k},
            transition="fade",
            transition_duration=0.35,
        )
        scene.add_text(
            f"PDB {a.pdb_id} · {s.pdb_caption or a.title}",
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
    scene.add_html(
        '<div style="font-size:1.05vh;letter-spacing:0.22em;'
        "font-weight:300;color:rgba(255,255,255,0.22);"
        'text-transform:uppercase;white-space:nowrap">'
        "Designed by Loic A. Royer</div>",
        position=(0.02, 0.97),
        anchor="bottom-left",
        interactive=False,
    )
    scene.add_image(
        BIOHUB_LOGO,
        position=(0.98, 0.97),
        anchor="bottom-right",
        size=(BIOHUB_LOGO_WIDTH, None),
        opacity=0.85,
        blend_mode="difference",
    )


def constellation_line_color(color: tuple[float, float, float]) -> np.ndarray:
    """The story colour pushed towards white: bright, still tinted."""
    base = np.asarray(color, dtype=np.float32)
    return (base + (1.0 - base) * CONSTELLATION_LINE_WHITEN).astype(np.float32)


def _add_constellations(
    scene: Any,
    stories: tuple[UniverseStory, ...],
    figures: dict[str, Constellation],
) -> None:
    """One lines node per constellation story, joining the places it lights.

    The vertices are the figure's own place centroids, and the story's
    highlight is the union of those same places (see
    :func:`select_universe_members`) — so an endpoint sits on a lit bead by
    construction. An earlier version drew the lines from the family's
    components while the highlight lit a KNOT ball around a purity-weighted
    seed, and the two disagreed by up to 0.45 world units: on the olfactory
    story the line visibly missed the blob it was supposed to leave.
    """
    for k, s in enumerate(stories, start=1):
        if not s.constellation:
            continue
        figure = figures[s.key]
        centroids = figure.centroids
        edges = figure.edges
        vertices = np.column_stack(
            [np.full(len(centroids), float(k), dtype=np.float32), centroids]
        ).astype(np.float32)
        spans = np.linalg.norm(centroids[edges[:, 0]] - centroids[edges[:, 1]], axis=1)
        aprint(
            f"{s.key}: {len(centroids)} places, {len(edges)} lines, "
            f"total {float(spans.sum()):.1f} units, longest {float(spans.max()):.1f}; "
            f"figure r_max {figure.r_max:.2f}"
        )
        width = max(
            CONSTELLATION_LINE_MIN_WIDTH, CONSTELLATION_LINE_WIDTH_FRAC * figure.r_max
        )
        scene.add_lines(
            f"Links {k}: {s.key}",
            vertices,
            widths=np.full(len(centroids), width, dtype=np.float32),
            colors=np.broadcast_to(
                constellation_line_color(s.color), (len(centroids), 3)
            ).copy(),
            indices=edges.astype(np.uint32),
            line_type="indexed",
            opacity=0.9,
            intensity=CONSTELLATION_LINE_INTENSITY,
            layer=True,
            blending_mode="additive",
            # Under the highlight, over the backdrop: the beads are the nodes
            # of the figure and must stay brighter than the threads.
            layer_order=8,
        )


def _add_bubbles(
    scene: Any, stories: tuple[UniverseStory, ...], clusters: list[StoryCluster]
) -> None:
    unit_verts, unit_faces = icosphere()
    nv = len(unit_verts)
    for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
        if s.whole:
            continue  # a story about the whole map has nothing to bubble
        if s.constellation:
            # Mutually exclusive by the owner's rule (2026-09-16): a bubble
            # says "the story is THIS blob", which is the opposite of what a
            # figure spanning a third of the map says, and a bubble sized to
            # hold every place would swallow the lines too.
            continue
        radius = bubble_radius(c, min_radius=BUBBLE_MIN_RADIUS)
        vertices = np.column_stack(
            [
                np.full(nv, float(k), dtype=np.float32),
                (unit_verts * radius + c.centre.astype(np.float32)).astype(np.float32),
            ]
        ).astype(np.float32)
        film = (1.0 - BUBBLE_TINT) * np.ones(3, dtype=np.float32) + (
            BUBBLE_TINT * np.asarray(s.color, dtype=np.float32)
        )
        scene.add_mesh(
            f"Marker {k}: {s.key}",
            vertices,
            unit_faces,
            normals=unit_verts,
            normal_dims=[1, 2, 3],
            colors=np.tile(film.astype(np.float32), (nv, 1)),
            shading="smooth",
            double_sided=True,
            material="physical",
            roughness=BUBBLE_ROUGHNESS,
            metalness=0.0,
            transmission=1.0,
            ior=BUBBLE_IOR,
            thickness=float(radius * BUBBLE_THICKNESS_FRAC),
            dispersion=BUBBLE_DISPERSION,
            iridescence=BUBBLE_IRIDESCENCE,
            refract_data=BUBBLE_REFRACT_DATA,
            layer=True,
            layer_order=SPHERE_LAYER_ORDER,
        )


def _add_backdrop(
    scene: Any, positions: np.ndarray, colors: np.ndarray, dims: Dimensions
) -> None:
    """The backdrop as BSP tiles of per-tile two-level POINTS ladders (see
    :data:`BACKDROP_TILE_POINTS`).

    Splits on the three spatial columns (the story column is constant), so the
    serialized split axes are the displayed dimensions and the viewer can use
    the tree for exact back-to-front ordering. Each tile is a hand-authored
    ``kind=lod`` group (the shape ``examples/partition_of_lod_example.py``
    builds): child 0 a seeded 1-in-K subsample with colours × K, child 1 the
    whole tile, thresholds from the partition-bound fills-screen rule. The
    compositing attrs ride on the WRAPPER: it is the node the Layers panel
    reads and pushes down to every descendant material.
    """
    tree = spatial_bsp_tree(
        positions, BACKDROP_TILE_POINTS, rule="median", split_axes=[1, 2, 3]
    )
    parts = bsp_leaf_parts(tree)
    sizes = [int(p.size) for p in parts]
    # The only cheap place to catch the per-node clamp: every leaf under budget.
    assert max(sizes) <= BACKDROP_TILE_POINTS, sizes
    aprint(
        f"Backdrop: {len(positions):,} clusters -> {len(parts)} BSP tiles "
        f"({min(sizes):,}..{max(sizes):,} points), coarse level 1 in "
        f"{BACKDROP_LOD_FACTOR} as points"
    )
    wrapper = scene.add_partition_group(
        "Backdrop",
        display_type="points",
        max_elements=BACKDROP_TILE_POINTS,
        layer=True,
        position_bounds=position_bounds_from_array(positions),
        bsp_tree=tree.to_serializable(),
        opacity=BACKDROP_OPACITY,
        intensity=BACKDROP_INTENSITY,
    )
    rng = np.random.default_rng(0)
    for i, idx in enumerate(parts):
        n_coarse = max(1, int(idx.size) // BACKDROP_LOD_FACTOR)
        coarse = np.sort(rng.choice(idx, n_coarse, replace=False))
        coverage = partitioned_coverage_fractions([n_coarse, int(idx.size)])
        lod = wrapper.add_lod_group(f"part_{i}", selector="screen-area")
        levels = ((coarse, float(BACKDROP_LOD_FACTOR)), (idx, 1.0))
        for level, (sel, gain) in enumerate(levels):
            level_positions = positions[sel]
            # No opacity/intensity here: compositing attrs compose
            # MULTIPLICATIVELY root→leaf, so a copy on the levels would square
            # the wrapper's (the closing review measured opacity 0.49 and a
            # 1/16 window reaching the material when both carried them).
            lod.add_points(
                f"child_{level}",
                level_positions,
                colors=(colors[sel] * np.float32(gain)).astype(np.float32),
                radii=np.full(sel.size, BACKDROP_RADIUS, dtype=np.float32),
                sharpness=np.full(sel.size, 0.6, dtype=np.float32),
                extend_to_all=[STORY_DIM],
                coverage_fraction=float(coverage[level]),
                # One hidden coordinate (story 0, extended to all): the slice
                # count is 1, but the policy is stated rather than assumed.
                additive_lod=stream_ladder(
                    sel.size,
                    slices=hidden_axis_stops(level_positions, dims.non_displayed),
                ),
            )


def resolve_stories(
    universe: Universe,
    annotations: Path,
    stories: tuple[UniverseStory, ...] = STORIES,
    figures: dict[str, Constellation] | None = None,
) -> list[StoryCluster]:
    """Resolve every story against the map (selection + knot framing).

    ``figures`` are the constellation figures from :func:`story_constellations`;
    they are recomputed when not supplied, since they are a pure function of
    the map and the story's Pfam selection.
    """
    spatial = require_module("scipy.spatial")
    if figures is None:
        figures = story_constellations(universe, stories)
    with asection("Resolving stories against the map"):
        all_tree = spatial.cKDTree(universe.positions)
        masks = _name_masks(annotations, universe, stories)
        clusters = []
        for s in stories:
            mask = family_mask(s, universe, masks.get(s.key))
            c = select_universe_members(
                s, universe, mask, all_tree, figure=figures.get(s.key)
            )
            clusters.append(c)
            if s.region == "spur":
                aprint(
                    f"{s.key}: {check_spur_story(universe, c):.0%} ABC-transporter Pfams"
                )
            if s.constellation:
                scope = f"{len(figures[s.key].places)} places joined"
            elif s.whole:
                scope = "whole map"
            else:
                scope = f"within {s.radius:g} of the knot"
            aprint(
                f"{s.key}: {len(c.indices):,} of {c.n_named:,} matching clusters, "
                f"{scope}; centre={c.centre.round(2)} r95={c.r95:.2f}"
            )
    return clusters


def build_universe_scene(
    output_path: Path,
    universe: Universe,
    annotations: Path,
    *,
    stories: tuple[UniverseStory, ...] = STORIES,
    auto_rotate: bool = True,
    turntables: bool = True,
    turntable_cache: Path | None = None,
    audio: bool = True,
    high_quality: bool = False,
) -> int:
    """Write the universe scene. Returns the number of clusters in the backdrop.

    ``high_quality`` re-enables the kiosk settings (SSAA, rendering at the
    display's full device pixel ratio, and the 95% dolly swing); the default is
    the laptop build.
    """
    n = len(universe)
    assets: dict[str, TurntableAssets] = {}
    if turntables:
        assets = _render_story_turntables(stories, output_path, turntable_cache)

    figures = story_constellations(universe, stories)
    clusters = resolve_stories(universe, annotations, stories, figures)
    labelled = {
        k: c
        for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1)
        if carries_labels(s, len(c.indices))
    }
    details = _member_details(annotations, universe, labelled)

    with asection("Composing waypoints"):
        positions = universe.positions
        global_centre = positions.mean(axis=0)
        # Frame the cloud, not the spur: the 99.9th-percentile radius is the
        # body of the map (the spur reaches almost twice as far).
        spatial_max = float(np.percentile(np.linalg.norm(positions, axis=1), 99.9))
        overview_distance = spatial_max * 2.2
        overview_raw = (overview_distance, overview_distance * 0.55, overview_distance)
        waypoints = [
            Waypoint(
                when={STORY_DIM: 0},
                camera=CameraConfig(
                    position=pull_in(overview_raw), target=(0.0, 0.0, 0.0), up=(0, 1, 0)
                ),
                duration_ms=3000,
                reveal="on_arrival",
            )
        ]
        for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
            waypoints.append(
                Waypoint(
                    when={STORY_DIM: k},
                    camera=universe_story_camera(
                        c,
                        s,
                        global_centre,
                        map_radius=spatial_max,
                        figure=figures.get(s.key),
                    ),
                    duration_ms=s.flight_ms,
                    reveal="on_arrival",
                )
            )
        viewer_config = _viewer_config(
            waypoints,
            overview_raw,
            # The touch panel, authored from the SAME `stories` the waypoints
            # above were built from, so the panel cannot drift from the tour.
            # Tile labels still come from the dimension's `categories` (each
            # story's short `key`); this adds the second line the tour wrote.
            control_panel=ControlPanelConfig(
                chapter_dimension=STORY_DIM,
                chapters={
                    # +1: slot 0 is the Overview, so story k sits at k+1.
                    index + 1: Chapter(sublabel=story.subtitle)
                    for index, story in enumerate(stories)
                },
            ),
            auto_rotate=auto_rotate,
            audio=audio,
            high_quality=high_quality,
        )

    dims = Dimensions(
        [
            Dimension(
                STORY_DIM,
                unit="",
                categories=["Overview", *(s.key for s in stories)],
                display=False,
                description="Guided tour: overview, then one story per step",
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
            backdrop_positions = np.column_stack(
                [np.zeros(n, dtype=np.float32), positions]
            ).astype(np.float32)
            # No hover labels on the backdrop: 7.7M strings would dominate the
            # store, and a kiosk visitor hovers the lit stories, which carry them.
            _add_backdrop(scene, backdrop_positions, backdrop_colors(universe), dims)
            units: dict[int, str] = {}
            for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
                idx = whole_highlight_sample(c.indices) if s.whole else c.indices
                m = len(idx)
                units[k] = highlight_unit(len(c.indices), m)
                labels, keys = details.get(k, (None, None))
                # Hover labels + UniRef link on the stories whose members are
                # few enough to carry them (see `_member_details`).
                # `labels`/`keys` accept None; `link` refuses it, so the
                # template rides in the link-metadata spread the element-cap
                # gate knows (`link_attrs`, never a budget).
                link_attrs = {"link": UNIREF_LINK} if keys is not None else {}
                radius, intensity = highlight_appearance(
                    s, m, figure=figures.get(s.key)
                )
                highlight_positions = np.column_stack(
                    [np.full(m, float(k), dtype=np.float32), positions[idx]]
                ).astype(np.float32)
                scene.add_points(
                    story_node_name(k, s),
                    highlight_positions,
                    colors=np.broadcast_to(
                        np.asarray(s.color, dtype=np.float32), (m, 3)
                    ).copy(),
                    radii=np.full(m, radius, dtype=np.float32),
                    sharpness=np.full(m, 0.85, dtype=np.float32),
                    opacity=0.95,
                    intensity=intensity,
                    labels=labels,
                    keys=keys,
                    **link_attrs,
                    # Every highlight streams like the backdrop: a whole-map
                    # one needs the ladder, a knot's few hundred points fit its
                    # first rung and cost nothing (and the slice-policy gate
                    # reads one call shape, not a conditional).
                    additive_lod=stream_ladder(
                        m,
                        slices=hidden_axis_stops(
                            highlight_positions, dims.non_displayed
                        ),
                    ),
                    layer=True,
                    # Additive and in its own band: the highlight shares every
                    # position with its backdrop twin (see the Swiss-Prot tour).
                    blending_mode="additive",
                    layer_order=10,
                )
            _add_bubbles(scene, stories, clusters)
            with asection("Constellations"):
                _add_constellations(scene, stories, figures)
            _add_overlays(scene, stories, clusters, assets, n, units)
            if audio:
                with asection("Sound layer"):
                    add_story_sounds(
                        scene,
                        stories,
                        narration_dir=NARRATION_CACHE_DIR,
                        ambisonic_dir=AMBISONIC_BED_CACHE_DIR,
                        overview_narration=OVERVIEW_NARRATION,
                    )

    aprint(f"✓ Wrote {n:,} clusters and {len(stories)} stories to {output_path}")
    # The bubbles are `material="physical"` and so read `scene.environment`.
    # Freeze it into the store instead of leaving every viewer to capture it
    # live; best-effort, and a no-op without a development checkout.
    bake_scene_environment(output_path)
    return n


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("ESM PROTEIN UNIVERSE — twenty stories across 7.7 million clusters")
    aprint("=" * 70)

    auto_rotate = "--no-auto-rotate" not in sys.argv
    turntables = "--no-turntables" not in sys.argv
    audio = "--no-audio" not in sys.argv
    # Kiosk / big-GPU build: SSAA, full device resolution, and the 95% dolly
    # swing. Off by default so the hosted demo runs on an ordinary laptop.
    high_quality = "--high-quality" in sys.argv
    try:
        annotations = find_input(ANNOTATIONS_PARQUET, parse_path_arg("annotations"))
        cache = CACHE_DIR / UNIVERSE_CACHE
        if not cache.exists():
            coords = find_input(COORDS_PARQUET, parse_path_arg("coords"))
            build_universe_cache(coords, annotations, cache)
    except (FileNotFoundError, ValueError) as e:
        aprint(f"\nError: {e}")
        sys.exit(1)
    universe = load_universe(cache)
    aprint(f"✓ Loaded {len(universe):,} clusters from {cache}")

    kwargs = dict(
        auto_rotate=auto_rotate,
        turntables=turntables,
        audio=audio,
        high_quality=high_quality,
    )
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "esm_protein_universe.luxar.zarr"
        n = build_universe_scene(output_path, universe, annotations, **kwargs)
        aprint(f"Dataset generated at {output_path} ({n:,} clusters)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_esm_universe_") as tmpdir:
        output_path = Path(tmpdir) / "esm_protein_universe.luxar.zarr"
        n = build_universe_scene(output_path, universe, annotations, **kwargs)
        aprint("")
        aprint("  Press '1' to select the STORY slider, then '[' / ']' to step.")
        aprint(f"  Total clusters: {n:,}")
        # Remote control is OFF unless asked for: `--control`, plus
        # `--control-token` and `--host 0.0.0.0` for a tablet on the LAN.
        launch_viewer(output_path, serve_args=control_serve_args())

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
