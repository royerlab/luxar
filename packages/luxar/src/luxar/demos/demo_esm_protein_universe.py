#!/usr/bin/env python3
"""ESM Protein Universe — twelve stories across 7.7 million protein clusters.

The big-map sibling of ``demo_esm3_protein_stories``. Every point is one of
7.7 million CLUSTERS of related proteins from the ESM Metagenomic Atlas — most
of them read straight out of environmental DNA rather than from any organism
grown in a lab — embedded by a protein language model and laid out by 3D UMAP.
A hidden ``story`` dimension walks through twelve stops: seven protein families
carried over from the Swiss-Prot tour (hemoglobin, photosystem II, Hsp70, the
viral spike, ATP synthase, RuBisCO, RecA), three stories only this map can tell
(the ABC-transporter spur flung off the cloud, the dark proteome, the phage
universe), and two more families whose knots are sharp here (beta-lactamases,
CRISPR-Cas). Each stop flies the camera to the family's densest knot, lights
its members, blows a soap bubble around them, shows a panel of sourced facts
and a spinning representative structure, and is narrated on arrival.

What changes against the Swiss-Prot tour, and why:

- **Scale.** Thirteen times the points. The backdrop carries no per-point hover
  labels (seven million strings would dominate the store); the story members
  do, with their product name, taxon and a UniRef link.
- **Grain.** This map is coherent at a far finer scale than the Swiss-Prot
  one: ten nearest neighbours share a Pfam family three times out of four, but
  they sit within a hundredth of a unit, and a family forms several small
  pure knots rather than one blob. Stories therefore select by **Pfam family**
  (or a product-name pattern, or a region predicate) and frame the densest
  knot, scored by member count times purity, within a radius of ~0.3 rather
  than 0.8.
- **Colour.** The backdrop is coloured by the domain of life of each cluster's
  dominant phylum, and clusters with no characterised member at all — one in
  four — are dimmed, so the "dark proteome" reads as geography before its
  story is told.

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
    python -m luxar.demos.demo_esm_protein_universe --coords X.parquet --annotations Y.parquet
"""

from __future__ import annotations

DEMO_META = {
    "key": "esm_protein_universe",
    "title": "ESM Protein Universe — twelve stories across 7.7 million protein clusters",
    "description": (
        "The ESM Metagenomic Atlas cluster-representative 3D UMAP (7.7 million "
        "protein clusters, mostly environmental) with a hidden story dimension: "
        "twelve stops — seven classic families, the ABC-transporter spur, the dark "
        "proteome, the phage universe, beta-lactamases and CRISPR-Cas — each with "
        "a fly-to waypoint, a highlight, a soap bubble, a turning structure and a "
        "narrated panel of researched facts."
    ),
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        # The CC0 ambient bed (~6 MiB, shared with the Swiss-Prot tour); the two
        # parquet inputs (~900 MB) are a manual hand-off, not a download.
        "download_mb": 7,
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
        "short": "ESM Metagenomic Atlas cluster map, ESM Atlas team 2026",
        "ref": "ESM Metagenomic Atlas / ESM Atlas team",
        "license": "unpublished, shown with permission",
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
from luxar.core.viewer_config import (
    AudioConfig,
    CameraConfig,
    EnvironmentConfig,
    ViewerConfig,
    Waypoint,
)
from luxar.demos import launch_viewer, parse_path_arg
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG, pull_in
from luxar.demos._dependencies import require_module
from luxar.demos._lod_policy import stream_ladder
from luxar.demos._pdb_turntable import (
    TurntableAssets,
    load_environment_faces,
    render_turntables,
)
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
    add_story_sounds,
    bubble_radius,
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
UNIVERSE_CACHE = "universe_v1.npz"
#: Points farther than this from the cloud's median are UMAP outliers (0.12% of
#: the rows, some 60 units out) that would otherwise set the scene's extent.
CULL_RADIUS = 40.0
#: Beyond this radius from the centre lies the spur (see the ABC story).
SPUR_RADIUS = 22.0

# The annotation columns the cache folds in (the rest are read per story member
# at build time).
_ANNOTATION_COLUMNS = (
    "protein_hash",
    "cluster_pct_characterized",
    "naming_tier",
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


def build_universe_cache(coords: Path, annotations: Path, out: Path) -> Path:
    """Fold the two parquet inputs into the compact per-cluster cache.

    Centres the coordinates on their median, culls the far outliers, joins the
    annotation table on the hash and keeps the per-cluster columns every build
    needs: characterised fraction, naming tier, dominant Pfam family, dominant
    phylum, and the annotation row (so a build can pull member details later).
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
        pct_rows = at.column("cluster_pct_characterized").to_numpy().astype(np.uint8)
        tier_rows = at.column("naming_tier").to_numpy().astype(np.int8)
        pfam_rows, pfam_vocab = _dominant_map_key(at, "cluster_top_pfam_domains")
        phylum_rows, phylum_vocab = _dominant_map_key(at, "top_phyla")
        del at

    def per_point(rows: np.ndarray, fill: int) -> np.ndarray:
        out_arr = np.full(len(row), fill, dtype=rows.dtype)
        out_arr[hit] = rows[row[hit]]
        return out_arr

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        positions=xyz,
        annotation_row=row,
        pct_characterized=per_point(pct_rows, 0),
        naming_tier=per_point(tier_rows, 5),
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
    naming_tier: np.ndarray
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
        return self.pct_characterized == 0

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
# annotation table itself (the audit is in the module docstring's numbers).


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
BACKDROP_RADIUS = 0.004
BACKDROP_INTENSITY = 0.08
BACKDROP_OPACITY = 0.7
HIGHLIGHT_RADIUS = 0.006
HIGHLIGHT_INTENSITY = 0.4
#: A knot of this many members gets the full highlight intensity; bigger
#: highlights (the region stories light thousands) are dimmed by the square
#: root of the ratio, so the additive glow of a knot stays roughly constant
#: instead of saturating into one white ball.
HIGHLIGHT_REFERENCE_MEMBERS = 300


def highlight_intensity(n_members: int) -> float:
    """Per-node highlight intensity for a knot of ``n_members`` clusters."""
    ratio = HIGHLIGHT_REFERENCE_MEMBERS / max(n_members, HIGHLIGHT_REFERENCE_MEMBERS)
    return HIGHLIGHT_INTENSITY * ratio**0.5


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
#: A whole-map highlight streams like the backdrop does.
WHOLE_LADDER_MIN_MEMBERS = 100_000

STORIES: tuple[UniverseStory, ...] = (
    _carry(
        "Hemoglobin",
        subtitle="Four hundred globin clusters; the knot is the oxygen carriers of animals",
        pattern="",
        pfam=("PF00042",),  # Globin
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "Hemoglobin",
            3,
            # Bacterial globins (flavohemoglobins, truncated globins) and plant
            # non-symbiotic hemoglobins: Vinogradov & Moens, JBC 283:8773 (2008).
            "The globin fold is far older than blood. Bacteria, fungi and plants "
            "carry globins that bind oxygen, sense it or detoxify nitric oxide, "
            "and the model files them here with ours — the same eight helices, "
            "reused for two billion years.",
        ),
    ),
    _carry(
        "Photosystem II",
        subtitle="The reaction-centre proteins: D1, D2 and their purple-bacteria cousins",
        pattern="",
        pfam=("PF00124",),  # Photo_RC: D1/D2 and the L/M chains
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "Photosystem II",
            3,
            # Purple-bacteria reaction centre L/M chains are homologous to
            # D1/D2 (Deisenhofer, Huber & Michel; Nobel 1988).
            "In this knot D1 and D2 sit with the L and M chains of "
            "purple-bacteria reaction centres: distant cousins that harvest "
            "light but never learned to split water. Their kinship was a key "
            "clue when the first photosynthetic structure was solved (Nobel "
            "Prize in Chemistry 1988).",
        ),
    ),
    _carry(
        "Hsp70",
        subtitle="Thousands of clusters of one chaperone, in every branch of life",
        pattern="",
        pfam=("PF00012",),  # HSP70
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "Hsp70",
            2,
            "That is why the map holds it in dozens of knots — each a version "
            "tuned to one branch of life, all of them recognisably the same "
            "protein. The map is showing a protein older than the deepest split "
            "in the tree of life.",
        ),
        narration=(
            "Hsp70, the oldest job in the cell. It holds unfolded proteins, "
            "refolds the damaged ones, and hands the hopeless ones to the "
            "shredder. After three billion years apart, the human and E. coli "
            "versions are still nearly half identical, letter for letter. The "
            "map holds it in dozens of knots, one for every branch of life. "
            "Cancer cells over-produce it to survive their own chaos, and drugs "
            "against it have been tried for decades. None has reached the "
            "clinic. Why is such a universal protein so hard to target?"
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
            # 562 spike clusters in the annotation table; the knot is 91% spike.
            "This knot is the coronavirus spike in hundreds of versions — from "
            "bats, birds, pigs, camels and people — a family the whole world "
            "learned to read in 2020.",
        ),
        pdb_id="6VXX",  # SARS-CoV-2 spike, closed state
        narration=(
            "The intruders' spike. Influenza's haemagglutinin, the coronavirus "
            "spike, HIV's envelope: unrelated viruses, one trick. Each snaps "
            "into a bundle that drags virus and cell together. The 1918 flu "
            "killed fifty million people with a protein like this one. This "
            "knot is the coronavirus spike, in hundreds of versions, from bats "
            "to people. Most viral proteins have no known relatives at all. "
            "Where does that dark matter land on this map?"
        ),
    ),
    _carry(
        "ATP synthase",
        subtitle="The rotary motor that makes the currency of life, in bacteria and in us",
        pattern=r"(?i)ATP synthase (subunit )?beta",
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "ATP synthase",
            3,
            "This knot is the beta subunit as bacteria build it, in hundreds of "
            "versions. The copies in our own mitochondria descend from exactly "
            "these: the same motor, inherited from the bacteria that became "
            "part of our cells.",
        ),
    ),
    _carry(
        "RuBisCO",
        subtitle="The protein that pulls carbon out of the air for almost all life",
        pattern="",
        pfam=("PF00016", "PF02788"),  # RuBisCO_large, RuBisCO_large_N
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "RuBisCO",
            3,
            "The knot here is the plant enzyme's large chain; the bacterial and "
            "archaeal forms make their own smaller knots elsewhere in the map.",
        ),
    ),
    _carry(
        "RecA and Rad51",
        subtitle="One recombinase, from phages and E. coli to the BRCA2 pathway in our cells",
        pattern="",
        pfam=("PF00154",),  # RecA (Rad51 has its own knot far away)
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=_swap_fact(
            "RecA and Rad51",
            3,
            # The densest RecA knot here is dominated by tailed-phage clusters;
            # phage-encoded RecA homologs (T4's UvsX among them) are well known.
            "The densest RecA knot in this map belongs to viruses: many tailed "
            "phages carry a RecA of their own, to repair and reshuffle their "
            "genomes inside the host. Our RAD51 forms a knot of its own "
            "elsewhere; the shape has barely moved in billions of years.",
        ),
        narration=(
            "RecA and Rad51, the machine that mends broken DNA. It coats a "
            "broken strand and searches the entire genome for the matching "
            "sequence. Our version, RAD51, is loaded by BRCA2, the protein "
            "whose mutations cause much of hereditary breast cancer. This knot "
            "belongs to phages, which carry a RecA of their own; the shape has "
            "barely moved in billions of years. It finds one match among "
            "millions of base pairs in minutes. How it searches that fast is "
            "still argued over."
        ),
    ),
    UniverseStory(
        key="ABC transporters",
        title="The spur — ABC transporters flung off the map",
        subtitle="Nine thousand clusters of the largest protein family on Earth, in a streak of their own",
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
            "out of cancer cells and is a major cause of drug resistance in "
            "tumours.",
            "The engine is the same everywhere: a pair of nucleotide-binding "
            "domains that clamp shut around two ATPs and spring open when they "
            "are spent. That engine — the cassette — is what these clusters "
            "share.",
            # 9,196 clusters beyond radius 22; 99% of the knot is PF00005.
            "The map put them on a spur of their own. A streak like this is a "
            "known habit of the layout algorithm when a huge family of very "
            "similar sequences shares few neighbours with anything else — an "
            "artefact, but an honest one: it marks the most repeated design in "
            "the protein world.",
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
            "The spur. These nine thousand clusters were flung off the map, and "
            "nearly all of them are one thing: the ATP-binding cassette, the "
            "engine of the ABC transporters. Every genome has them, pumping "
            "nutrients in and toxins out. Humans have forty-eight; a broken one "
            "causes cystic fibrosis, another pumps chemotherapy out of tumours. "
            "The streak is a habit of the layout algorithm, but an honest one: "
            "it marks the most repeated design in the protein world. How the "
            "engine actually moves its cargo is still argued over."
        ),
    ),
    UniverseStory(
        key="Dark proteome",
        title="The dark proteome — two million clusters nobody has named",
        subtitle="A quarter of this map has no characterised member at all",
        pattern="",
        region="dark",
        whole=True,
        color=(0.5, 0.65, 1.0),
        frame_fraction=1.0,
        facts=(
            # Annotation table: 2,025,330 of 7,723,579 clusters have
            # cluster_pct_characterized == 0; 76% of those are named
            # "hypothetical protein".
            "Two million of the 7.7 million clusters here — one in four — "
            "contain not a single protein anyone has characterised. Most carry "
            "the placeholder name “hypothetical protein”. They are the dim "
            "points of this map.",
            # Lin et al., Science 379:1123 (2023): ESM Metagenomic Atlas, 617
            # million predicted structures.
            "They come from metagenomics: DNA read straight out of soil, "
            "seawater, hot springs and guts, from organisms nobody has grown in "
            "a lab. The ESM Metagenomic Atlas predicted structures for more "
            "than 600 million such sequences.",
            # Annotation table: in the densest 0.5-unit voxels the dark
            # fraction exceeds 95%; ten nearest neighbours of a dark cluster
            # are dark 80% of the time against a 26% base rate.
            "Dark sits next to dark: in the densest pockets here, more than "
            "nine in ten neighbours are unnamed. Whatever these proteins do, "
            "they do it in families of their own.",
            # CRISPR: Ishino et al. 1987 → Jinek et al. 2012. GFP: Shimomura
            # 1962 → Chalfie et al. 1994.
            "Some of the most useful tools in biology were dark once. The "
            "CRISPR repeats were “unusual” DNA for twenty-five years; green "
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
            "contain not a single protein anyone has characterised. They are "
            "the dim points of this map, read straight out of soil, seawater "
            "and guts, from organisms nobody has grown. Dark sits next to dark: "
            "in the densest pockets here, nine in ten neighbours are unnamed. "
            "Some of biology's best tools were dark once; CRISPR was unusual "
            "DNA for twenty-five years. Are these new chemistry, or old folds "
            "drifted beyond recognition? Nobody knows the proportion."
        ),
    ),
    UniverseStory(
        key="Phage",
        title="The phage universe — the viruses that outnumber everything",
        subtitle="Half a million clusters of tailed bacteriophages, half of them unnamed",
        pattern="",
        region="phage",
        whole=True,
        color=(1.0, 0.55, 0.75),
        frame_fraction=1.0,
        facts=(
            # Hendrix et al., PNAS 96:2192 (1999): ~10^31 tailed phages.
            # Annotation table: 580,733 clusters with Uroviricota dominant.
            "Bacteriophages — the viruses of bacteria — are the most abundant "
            "biological entities on Earth: an estimated ten million trillion "
            "trillion of them, more than every other organism combined. Here "
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
            "characterised. Phage genomes are the largest reservoir of unknown "
            "genes on the planet.",
        ),
        mystery=(
            "Every gram of soil and every millilitre of seawater holds millions "
            "of phages. What most of their genes do — and how many kinds of "
            "phage there really are — nobody knows."
        ),
        tags=("virology", "ecology"),
        pdb_id="2XGF",  # T4 long tail fibre receptor-binding tip
        narration=(
            "The phage universe. Bacteriophages, the viruses of bacteria, are "
            "the most abundant biological entities on Earth: ten million "
            "trillion trillion of them, more than everything else combined. "
            "Half a million clusters here are theirs. In the oceans they kill a "
            "fifth of all microbial biomass every day. They were medicine before "
            "penicillin, and are being tried again. Half of the phage clusters "
            "here are dark. What most of their genes do, nobody knows."
        ),
    ),
    UniverseStory(
        key="Beta-lactamases",
        title="Beta-lactamase — the enzyme that fights back",
        subtitle="Seven thousand clusters of the proteins that destroy penicillin and its cousins",
        pattern="",
        pfam=("PF00144", "PF13354"),  # Beta-lactamase, Beta-lactamase2
        color=(0.95, 0.25, 0.5),
        radius=FAMILY_RADIUS,
        min_distance=FAMILY_MIN_DISTANCE,
        facts=(
            "Penicillin works by jamming the enzymes that build the bacterial "
            "cell wall. A beta-lactamase cuts open the antibiotic's "
            "four-membered ring before it gets there, and the drug is dead.",
            # Abraham & Chain, Nature 146:837 (1940); first patient treated
            # February 1941.
            "Resistance was there before the cure. Edward Abraham and Ernst "
            "Chain described an E. coli enzyme that destroyed penicillin in "
            "1940 — a year before the first patient was ever treated with it.",
            # Murray et al., Lancet 399:629 (2022): 4.95 million deaths
            # associated with, 1.27 million attributable to, bacterial AMR in 2019.
            "Antibiotic resistance is now associated with nearly five million "
            "deaths a year, 1.3 million of them directly attributable. "
            "Beta-lactamases — TEM-1, CTX-M, NDM-1 — are the core of the "
            "problem in Gram-negative bacteria.",
            # D'Costa et al., Nature 477:457 (2011): resistance genes in
            # 30,000-year-old Beringian permafrost.
            "Beta-lactamase genes have been recovered from 30,000-year-old "
            "permafrost. The enzymes are ancient: bacteria have been fighting "
            "moulds with penicillins, and each other with beta-lactamases, for "
            "millions of years.",
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
            "described in 1940, a year before the first patient was treated. "
            "Today antibiotic resistance is linked to nearly five million deaths "
            "a year, and these genes turn up in thirty-thousand-year-old "
            "permafrost. Can new drugs keep pace with an enzyme that evolves in "
            "real time?"
        ),
    ),
    UniverseStory(
        key="CRISPR-Cas",
        title="CRISPR-Cas9 — a bacterial immune system turned into scissors",
        subtitle="Hundreds of clusters of the enzyme bacteria use to cut viral DNA",
        pattern="",
        pfam=("PF16592", "PF16593", "PF13395"),  # Cas9 REC lobe, HNH, RuvC
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
            "Cas9 could be pointed at any DNA sequence by rewriting its guide — "
            "programmable scissors. Nobel Prize in Chemistry 2020.",
            # Casgevy (exagamglogene autotemcel): MHRA Nov 2023, FDA Dec 2023.
            "Eleven years later the first CRISPR medicine was approved: a "
            "therapy for sickle-cell disease — the illness of the first story "
            "on this tour.",
        ),
        # Makarova et al., Nat. Rev. Microbiol. 13:722 (2015): CRISPR-Cas in
        # ~45% of bacterial and ~85% of archaeal genomes.
        mystery=(
            "Roughly 40% of bacteria and nearly 90% of archaea carry CRISPR "
            "systems, yet many highly successful bacteria do without. Why would "
            "an organism give up an immune system?"
        ),
        tags=("genome editing", "immunity"),
        pdb_id="4OO8",  # S. pyogenes Cas9 with guide RNA and target DNA
        narration=(
            "CRISPR-Cas9, a bacterial immune system turned into scissors. "
            "Bacteria keep snippets of the viruses that attacked them, and Cas "
            "proteins use those snippets to find and cut the same virus next "
            "time. Noticed in 1987, understood in 2005, and in 2012 Doudna and "
            "Charpentier showed Cas9 could be pointed at any DNA at all. Eleven "
            "years later the first CRISPR medicine was approved, for sickle-cell "
            "disease, the illness of the first story on this tour. Yet many "
            "successful bacteria do without CRISPR. Why give up an immune "
            "system?"
        ),
    ),
)

OVERVIEW_TITLE = "Twelve stories in the protein universe"
ATTRIBUTION = f"{DEMO_META['citation']['ref']} · {DEMO_META['citation']['license']}"
OVERVIEW_HTML = (
    "Every point is one of {n:,} clusters of related proteins from the ESM "
    "Metagenomic Atlas — most of them read straight out of soil, seawater and "
    "guts rather than from any organism grown in a lab — placed by a protein "
    "language model so that similar proteins sit close together, then projected "
    "to 3D with UMAP. Colours are domains of life; the dim points are clusters "
    "nobody has characterised."
    "<br><br>Step the <b>story</b> dimension to fly to twelve knots that each "
    "tell a piece of biology: blood, sunlight, the oldest chaperone, the "
    "coronavirus spike, the cell's turbine, the slowest important enzyme, the "
    "machine that mends DNA, a spur flung off the map, the dark proteome, the "
    "phage universe, the enzyme that beats penicillin, and CRISPR."
)
#: Spoken introduction at the Overview slot.
OVERVIEW_NARRATION = (
    "Every point here is a family of proteins: seven point seven million of "
    "them, most read straight out of the environment, placed by a language "
    "model so that similar proteins sit close together. The dim points are "
    "families nobody has characterised. Twelve of these knots hide a story. "
    "Step through them."
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
        mask |= np.linalg.norm(universe.positions, axis=1) > SPUR_RADIUS
    elif story.region == "dark":
        mask |= universe.dark_mask()
    elif story.region == "phage":
        mask |= universe.phylum_mask("Uroviricota")
    elif story.region is not None:
        raise ValueError(f"story {story.key!r}: unknown region {story.region!r}")
    return mask


def select_universe_members(
    story: UniverseStory,
    universe: Universe,
    mask: np.ndarray,
    all_tree: Any,
) -> StoryCluster:
    """Resolve a story to the clusters of its densest knot — or, for a
    ``whole`` story, to every matching cluster.

    A knot story seeds on the member with the most (and purest) member
    neighbours within two thirds of ``story.radius``, keeps the members within
    ``story.radius`` of it, and re-centres on their median.
    """
    n_named = int(mask.sum())
    if n_named == 0:
        raise ValueError(f"story {story.key!r}: no cluster matched its selector")
    if story.whole:
        indices = np.flatnonzero(mask)
        centre = np.median(universe.positions[indices], axis=0)
        radial = np.linalg.norm(universe.positions[indices] - centre, axis=1)
        return StoryCluster(
            indices=indices,
            centre=centre,
            r95=float(np.percentile(radial, 95)),
            n_named=n_named,
            r50=float(np.percentile(radial, 50)),
        )
    seed = densest_core(
        universe.positions[mask].astype(np.float64), story.radius * 2.0 / 3.0, all_tree
    )
    ball = np.asarray(all_tree.query_ball_point(seed, story.radius), dtype=np.int64)
    indices = ball[mask[ball]]
    if len(indices) == 0:
        raise ValueError(f"story {story.key!r}: no member within radius {story.radius}")
    indices = np.sort(indices)
    centre = np.median(universe.positions[indices], axis=0)
    radial = np.linalg.norm(universe.positions[indices] - centre, axis=1)
    return StoryCluster(
        indices=indices,
        centre=centre,
        r95=float(np.percentile(radial, 95)),
        n_named=n_named,
        r50=float(np.percentile(radial, 50)),
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


def universe_story_camera(
    cluster: StoryCluster, story: UniverseStory, global_centre: np.ndarray
) -> CameraConfig:
    """The waypoint pose: the stories tour's outside-in shot, or side-on.

    Side-on keeps the tour's distance rule and swaps the direction for one
    perpendicular to the centre→cluster ray (the horizontal perpendicular,
    lifted a little), so an elongated radial feature is seen across, not along.
    """
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
    """Read the label columns for every knot-story member (one parquet pass).

    ``clusters`` maps story slot → cluster for the KNOT stories only: a
    whole-map highlight of two million points carries no hover labels (the
    strings would dominate the store).
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
        names = [str(cols["product_name"][i]) if i >= 0 else "cluster" for i in idx]
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
        environment = load_environment_faces(output_path)
        if environment is None:
            aprint(
                "ℹ️  No baked environment in the output store yet: turntables use "
                f"the studio lights only. Run `luxar env bake {output_path}` and "
                "rebuild to light them with the map."
            )
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
) -> ViewerConfig:
    # Mirrors the Swiss-Prot tour's kiosk settings (see its build for the why).
    # `overview` is the raw distance-tuned pose; `pull_in` carries it to the
    # cinematic 63° lens.
    return ViewerConfig(
        cinematic_mode=True,
        camera=CameraConfig(position=pull_in(overview), target=(0.0, 0.0, 0.0)),
        auto_rotate=auto_rotate,
        auto_rotate_speed=0.5 if auto_rotate else None,
        auto_rotate_axis="world-y" if auto_rotate else None,
        ssaa_enabled=True,
        allow_high_dpr=True,
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
            f"PDB {a.pdb_id} · {a.title}",
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


def _add_bubbles(
    scene: Any, stories: tuple[UniverseStory, ...], clusters: list[StoryCluster]
) -> None:
    unit_verts, unit_faces = icosphere()
    nv = len(unit_verts)
    for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
        if s.whole:
            continue  # a story about the whole map has nothing to bubble
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


def resolve_stories(
    universe: Universe,
    annotations: Path,
    stories: tuple[UniverseStory, ...] = STORIES,
) -> list[StoryCluster]:
    """Resolve every story against the map (selection + knot framing)."""
    spatial = require_module("scipy.spatial")
    with asection("Resolving stories against the map"):
        all_tree = spatial.cKDTree(universe.positions)
        masks = _name_masks(annotations, universe, stories)
        clusters = []
        for s in stories:
            mask = family_mask(s, universe, masks.get(s.key))
            c = select_universe_members(s, universe, mask, all_tree)
            clusters.append(c)
            scope = "whole map" if s.whole else f"within {s.radius:g} of the knot"
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
) -> int:
    """Write the universe scene. Returns the number of clusters in the backdrop."""
    n = len(universe)
    assets: dict[str, TurntableAssets] = {}
    if turntables:
        assets = _render_story_turntables(stories, output_path, turntable_cache)

    clusters = resolve_stories(universe, annotations, stories)
    knots = {
        k: c
        for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1)
        if not s.whole
    }
    details = _member_details(annotations, universe, knots)

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
                    camera=universe_story_camera(c, s, global_centre),
                    duration_ms=s.flight_ms,
                    reveal="on_arrival",
                )
            )
        viewer_config = _viewer_config(
            waypoints, overview_raw, auto_rotate=auto_rotate, audio=audio
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
            scene.add_points(
                "Backdrop",
                backdrop_positions,
                colors=backdrop_colors(universe),
                radii=np.full(n, BACKDROP_RADIUS, dtype=np.float32),
                sharpness=np.full(n, 0.6, dtype=np.float32),
                opacity=BACKDROP_OPACITY,
                intensity=BACKDROP_INTENSITY,
                extend_to_all=[STORY_DIM],
                layer=True,
                additive_lod=stream_ladder(n),
            )
            units: dict[int, str] = {}
            for k, (s, c) in enumerate(zip(stories, clusters, strict=True), start=1):
                idx = whole_highlight_sample(c.indices) if s.whole else c.indices
                m = len(idx)
                units[k] = highlight_unit(len(c.indices), m)
                labels, keys = details.get(k, (None, None))
                # Hover labels + UniRef link on knot stories only (see
                # `_member_details`). `labels`/`keys` accept None; `link`
                # refuses it, so the template rides in the link-metadata spread
                # the element-cap gate knows (`link_attrs`, never a budget).
                link_attrs = {"link": UNIREF_LINK} if keys is not None else {}
                radius = WHOLE_HIGHLIGHT_RADIUS if s.whole else HIGHLIGHT_RADIUS
                intensity = (
                    WHOLE_HIGHLIGHT_INTENSITY if s.whole else highlight_intensity(m)
                )
                scene.add_points(
                    story_node_name(k, s),
                    np.column_stack(
                        [np.full(m, float(k), dtype=np.float32), positions[idx]]
                    ).astype(np.float32),
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
                    additive_lod=(
                        stream_ladder(m) if m >= WHOLE_LADDER_MIN_MEMBERS else None
                    ),
                    layer=True,
                    # Additive and in its own band: the highlight shares every
                    # position with its backdrop twin (see the Swiss-Prot tour).
                    blending_mode="additive",
                    layer_order=10,
                )
            _add_bubbles(scene, stories, clusters)
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
    return n


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("ESM PROTEIN UNIVERSE — twelve stories across 7.7 million clusters")
    aprint("=" * 70)

    auto_rotate = "--no-auto-rotate" not in sys.argv
    turntables = "--no-turntables" not in sys.argv
    audio = "--no-audio" not in sys.argv
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

    kwargs = dict(auto_rotate=auto_rotate, turntables=turntables, audio=audio)
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
        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
