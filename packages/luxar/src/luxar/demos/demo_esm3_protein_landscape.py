"""
ESM-3 Protein Landscape — Swiss-Prot 3D Embedding
===================================================

Visualizes ~575K Swiss-Prot protein embeddings from ESM C (or ESM-3) as a
3D UMAP point cloud. Each point is a protein, colored by taxonomic group (from
its UniProt lineage), with hover labels showing protein name, organism and group.

Run behavior:
  First run: Download Swiss-Prot, compute ESM embeddings, run UMAP (~5h)
  Subsequent runs: Load cached results instantly

  Computing the ESM embeddings requires a CUDA GPU. On a machine without CUDA,
  supply a complete precomputed embeddings cache (the demo validates it on load
  and fails fast with guidance if it is missing/truncated).

Data source: UniProt/Swiss-Prot (CC BY 4.0)
Model: ESM-3 open (Hayes et al. 2025) or ESM C 300M (EvolutionaryScale)

Usage:
    python -m luxar.demos.demo_esm3_protein_landscape
    python -m luxar.demos.demo_esm3_protein_landscape --no-serve
    python -m luxar.demos.demo_esm3_protein_landscape --sample=100000
    python -m luxar.demos.demo_esm3_protein_landscape --model=esmc-300m

Dependencies:
    pip install 'luxar[demos]'   # includes esm>=3.0.0, umap-learn, h5py
    pip install 'torch>=2.2,<3.0'  # CUDA build, to compute embeddings
    pip install 'luxar[gsplats]'   # torch + scipy, pinned; missing either one
                                   # makes the cached path write flat Points
                                   # instead of the Points LOD ladder

Cache hygiene:
    A cached artifact that fails validation is quarantined to ``<name>.corrupt``
    and never reused. The demo reports any quarantined file (path + size) before
    doing anything expensive — re-download the complete file or delete the
    quarantined copy, otherwise the run starts over from scratch.
"""

DEMO_META = {
    "key": "esm3_protein_landscape",
    "title": "ESM3 Protein Landscape",
    "description": "~575K Swiss-Prot protein embeddings from ESM C (or ESM-3) as a 3D UMAP, colored by taxonomic group.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 90,  # approx (Swiss-Prot FASTA)
        "compute": "heavy",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["esm3_swissprot"],
    "outputs": ["esm3_protein_landscape"],
    # The proteins are UniProt's and the coordinates are the model's, so the
    # credit names both -- crediting only the model would attribute someone
    # else's dataset to it. The model is chosen at RUNTIME (`--model=`), so this
    # static entry names the default (`esmc-300m`); the scene itself is stamped
    # with whichever model actually ran. Keep the two in step if the default
    # changes.
    "citation": {
        "short": "UniProt/Swiss-Prot; embeddings by EvolutionaryScale ESM C, 2024",
        "ref": "UniProt / EvolutionaryScale 2024",
        "license": "CC BY 4.0",
    },
}

import gzip
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    QUARANTINE_SUFFIX,
    MissingDependencyError,
    add_demo_caption,
    find_quarantined_files,
    format_quarantine_notice,
    launch_viewer,
    quarantine_file,
    require_module,
    stack_colorings,
    warn_if_quarantined,
)
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 0  # 0 = all (~572K)


def _linkable_accessions(accessions: list[str], n_proteins: int) -> list[str] | None:
    """Return aligned accessions, or preserve an old cache with search fallback."""
    if len(accessions) == n_proteins:
        return accessions
    aprint("  ⓘ Cached metadata has no accessions — using protein-name search")
    return None


def _uniprot_link_attrs(
    keys: list[str], *, exact_accessions: bool
) -> dict[str, object]:
    """Link exact accessions, or search clean protein-name keys.

    The visible label also includes organism and taxonomic category, whose
    punctuation makes UniProt's query parser return no results. Cached metadata
    predating accessions therefore uses the bare protein name as ``hover_key``.
    """
    if not exact_accessions:
        return {
            "keys": keys,
            "link": "https://www.uniprot.org/uniprotkb?query={hover_key}",
            "copy": "{hover_key}",
        }
    return {
        "keys": keys,
        "link": "https://www.uniprot.org/uniprotkb/{hover_key}/entry",
        "copy": "{hover_key}",
    }


SWISSPROT_FASTA_URLS = [
    # ExPASy mirror (faster, more reliable)
    "https://ftp.expasy.org/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
    # Primary UniProt FTP (can be slow)
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz",
]

# Checkpoint frequency for embedding computation
CHECKPOINT_INTERVAL = 10000

# 12 taxonomic categories with distinct colors (max requested by user)
TAXON_COLORS: dict[str, tuple[float, float, float]] = {
    "Human": (0.25, 0.55, 1.0),  # bright blue
    "Mouse & Rat": (0.5, 0.75, 1.0),  # light blue
    "Other Vertebrates": (0.3, 0.8, 0.85),  # teal
    "Insects & Worms": (0.7, 0.4, 0.9),  # purple
    "Plants": (0.2, 0.75, 0.3),  # green
    "Fungi": (0.9, 0.8, 0.2),  # yellow
    "Other Eukaryotes": (0.6, 0.6, 0.9),  # lavender
    "Proteobacteria": (0.4, 0.9, 0.5),  # lime
    "Firmicutes & Actino": (0.6, 0.8, 0.3),  # olive
    "Other Bacteria": (0.3, 0.65, 0.4),  # dark green
    "Archaea": (1.0, 0.5, 0.1),  # orange
    "Viruses": (0.9, 0.2, 0.2),  # red
    "Other": (0.5, 0.5, 0.5),  # gray fallback
}

# Coarse domain view: the 12 fine taxa collapse to the major domains of life,
# giving a second (switchable) coloring alongside the fine taxon view.
_DOMAIN_OF: dict[str, str] = {
    "Human": "Eukaryota",
    "Mouse & Rat": "Eukaryota",
    "Other Vertebrates": "Eukaryota",
    "Insects & Worms": "Eukaryota",
    "Plants": "Eukaryota",
    "Fungi": "Eukaryota",
    "Other Eukaryotes": "Eukaryota",
    "Proteobacteria": "Bacteria",
    "Firmicutes & Actino": "Bacteria",
    "Other Bacteria": "Bacteria",
    "Archaea": "Archaea",
    "Viruses": "Viruses",
    "Other": "Other",
}
DOMAIN_COLORS: dict[str, tuple[float, float, float]] = {
    "Eukaryota": (0.25, 0.55, 1.0),  # blue
    "Bacteria": (0.4, 0.9, 0.5),  # green
    "Archaea": (1.0, 0.5, 0.1),  # orange
    "Viruses": (0.9, 0.2, 0.2),  # red
    "Other": (0.5, 0.5, 0.5),  # gray
}

# Organism classification rules — checked in order, first match wins.
# Tuples of (pattern, category). Pattern matched against lowercase organism name.
_CLASSIFICATION_RULES: list[tuple[str, str]] = [
    # --- Viruses (check first: "virus" is unambiguous) ---
    ("virus", "Viruses"),
    ("phage", "Viruses"),
    ("viridae", "Viruses"),
    ("sars-cov", "Viruses"),
    ("vaccinia", "Viruses"),
    ("influenza", "Viruses"),
    # --- Archaea ---
    ("methan", "Archaea"),
    ("halobacterium", "Archaea"),
    ("sulfolobus", "Archaea"),
    ("thermococcus", "Archaea"),
    ("pyrococcus", "Archaea"),
    ("archaeoglobus", "Archaea"),
    ("haloferax", "Archaea"),
    ("thermoplasma", "Archaea"),
    ("aeropyrum", "Archaea"),
    # --- Human ---
    ("homo sapiens", "Human"),
    # --- Mouse & Rat ---
    ("mus musculus", "Mouse & Rat"),
    ("rattus", "Mouse & Rat"),
    # --- Plants ---
    ("arabidopsis", "Plants"),
    ("oryza", "Plants"),
    ("zea mays", "Plants"),
    ("nicotiana", "Plants"),
    ("solanum", "Plants"),
    ("glycine max", "Plants"),
    ("triticum", "Plants"),
    ("hordeum", "Plants"),
    ("medicago", "Plants"),
    ("populus", "Plants"),
    ("vitis", "Plants"),
    ("physcomitrella", "Plants"),
    ("marchantia", "Plants"),
    ("chlamydomonas", "Plants"),
    # --- Fungi ---
    ("saccharomyces", "Fungi"),
    ("schizosaccharomyces", "Fungi"),
    ("candida", "Fungi"),
    ("aspergillus", "Fungi"),
    ("neurospora", "Fungi"),
    ("cryptococcus", "Fungi"),
    ("ustilago", "Fungi"),
    ("yarrowia", "Fungi"),
    ("kluyveromyces", "Fungi"),
    ("emericella", "Fungi"),
    # --- Insects & Worms ---
    ("drosophila", "Insects & Worms"),
    ("caenorhabditis", "Insects & Worms"),
    ("anopheles", "Insects & Worms"),
    ("aedes", "Insects & Worms"),
    ("bombyx", "Insects & Worms"),
    ("tribolium", "Insects & Worms"),
    ("apis mellifera", "Insects & Worms"),
    ("schistosoma", "Insects & Worms"),
    ("brugia", "Insects & Worms"),
    # --- Proteobacteria (E. coli, Salmonella, Pseudomonas, etc.) ---
    ("escherichia", "Proteobacteria"),
    ("salmonella", "Proteobacteria"),
    ("shigella", "Proteobacteria"),
    ("pseudomonas", "Proteobacteria"),
    ("vibrio", "Proteobacteria"),
    ("helicobacter", "Proteobacteria"),
    ("campylobacter", "Proteobacteria"),
    ("neisseria", "Proteobacteria"),
    ("bordetella", "Proteobacteria"),
    ("rickettsia", "Proteobacteria"),
    ("rhizobium", "Proteobacteria"),
    ("agrobacterium", "Proteobacteria"),
    ("caulobacter", "Proteobacteria"),
    ("legionella", "Proteobacteria"),
    ("acinetobacter", "Proteobacteria"),
    ("klebsiella", "Proteobacteria"),
    ("haemophilus", "Proteobacteria"),
    ("brucella", "Proteobacteria"),
    ("burkholderia", "Proteobacteria"),
    ("xanthomonas", "Proteobacteria"),
    ("yersinia", "Proteobacteria"),
    # --- Firmicutes & Actinobacteria ---
    ("bacillus", "Firmicutes & Actino"),
    ("staphylococcus", "Firmicutes & Actino"),
    ("streptococcus", "Firmicutes & Actino"),
    ("clostridium", "Firmicutes & Actino"),
    ("lactobacillus", "Firmicutes & Actino"),
    ("enterococcus", "Firmicutes & Actino"),
    ("listeria", "Firmicutes & Actino"),
    ("mycobacterium", "Firmicutes & Actino"),
    ("corynebacterium", "Firmicutes & Actino"),
    ("streptomyces", "Firmicutes & Actino"),
    ("bifidobacterium", "Firmicutes & Actino"),
    # --- Other Vertebrates ---
    ("bos taurus", "Other Vertebrates"),
    ("sus scrofa", "Other Vertebrates"),
    ("gallus", "Other Vertebrates"),
    ("xenopus", "Other Vertebrates"),
    ("danio", "Other Vertebrates"),
    ("pongo", "Other Vertebrates"),
    ("pan troglodytes", "Other Vertebrates"),
    ("oryctolagus", "Other Vertebrates"),
    ("canis", "Other Vertebrates"),
    ("equus", "Other Vertebrates"),
    ("ovis", "Other Vertebrates"),
    ("macaca", "Other Vertebrates"),
    ("takifugu", "Other Vertebrates"),
    ("torpedo", "Other Vertebrates"),
]


def _classify_organism(organism: str) -> str:
    """Classify an organism into one of 12 taxonomic categories."""
    org_lower = organism.lower()

    # Check explicit patterns first
    for pattern, category in _CLASSIFICATION_RULES:
        if pattern in org_lower:
            return category

    # Bacterial genus suffix heuristics
    first_word = org_lower.split()[0] if org_lower else ""
    bacterial_suffixes = (
        "bacillus",
        "coccus",
        "monas",
        "bacter",
        "bacterium",
        "spirillum",
        "vibrio",
        "plasma",
        "phila",
        "oides",
        "ella",
        "inia",
        "eria",
    )
    if any(first_word.endswith(s) for s in bacterial_suffixes):
        return "Other Bacteria"

    # Archaeal suffix heuristics
    archaeal_suffixes = ("archaeum", "archaeon", "pyrus")
    if any(first_word.endswith(s) for s in archaeal_suffixes):
        return "Archaea"

    # Default: other eukaryotes (amoeba, algae, protists, etc.)
    return "Other Eukaryotes"


# =============================================================================
# Taxon categories from the UniProt lineage
# =============================================================================

#: Every reviewed entry's taxonomic lineage (~6 MB gzipped TSV), the source of
#: truth for the taxon categories. The organism-name rules above are only the
#: fallback for an entry the table does not cover.
SWISSPROT_LINEAGE_URL = (
    "https://rest.uniprot.org/uniprotkb/stream?compressed=true"
    "&fields=accession%2Clineage&format=tsv&query=%28reviewed%3Atrue%29"
)
LINEAGE_CACHE_NAME = "uniprot_sprot_lineage.tsv.gz"
#: Stamped into the metadata cache once its categories come from the lineage.
KINGDOM_SOURCE = "uniprot-lineage"

#: Ordered (markers, category): the first rule whose markers meet the lineage
#: wins, so the bacterial phyla precede "Bacteria" and the named animals
#: precede "Eukaryota".
_LINEAGE_RULES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"Viruses"}), "Viruses"),
    (frozenset({"Archaea"}), "Archaea"),
    (frozenset({"Pseudomonadota", "Proteobacteria"}), "Proteobacteria"),
    (
        frozenset({"Bacillota", "Firmicutes", "Actinomycetota", "Actinobacteria"}),
        "Firmicutes & Actino",
    ),
    (frozenset({"Bacteria"}), "Other Bacteria"),
    (frozenset({"Homo"}), "Human"),
    (frozenset({"Mus", "Rattus"}), "Mouse & Rat"),
    (frozenset({"Vertebrata"}), "Other Vertebrates"),
    (frozenset({"Insecta", "Nematoda", "Platyhelminthes"}), "Insects & Worms"),
    (frozenset({"Viridiplantae"}), "Plants"),
    (frozenset({"Fungi"}), "Fungi"),
    (frozenset({"Eukaryota"}), "Other Eukaryotes"),
)


def _classify_lineage(lineage: str) -> str | None:
    """Map a UniProt lineage ("Bacteria (domain), Pseudomonadota (phylum), ...")
    onto the twelve categories, or ``None`` when it names no domain."""
    taxa = {part.rsplit(" (", 1)[0].strip() for part in lineage.split(",")}
    for markers, category in _LINEAGE_RULES:
        if taxa & markers:
            return category
    return None


def _lineage_table(cache_dir: Path) -> dict[str, str] | None:
    """Accession → lineage, downloaded once into ``cache_dir``; ``None`` offline."""
    path = cache_dir / LINEAGE_CACHE_NAME
    if not path.exists():
        import urllib.request

        part = path.with_name(path.name + ".part")
        with asection("Downloading the Swiss-Prot taxonomic lineages (~6 MB)"):
            try:
                with (
                    urllib.request.urlopen(  # nosec B310 - fixed https URL
                        SWISSPROT_LINEAGE_URL, timeout=900
                    ) as response,
                    open(part, "wb") as out,
                ):
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                part.replace(path)
            except OSError as e:
                part.unlink(missing_ok=True)
                aprint(f"  ⚠ lineage download failed ({e}); keeping name-based taxa")
                return None
    table: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            acc, _, lineage = line.rstrip("\n").partition("\t")
            table[acc] = lineage
    return table


def _recover_accessions(
    cache_dir: Path, meta: dict[str, np.ndarray]
) -> np.ndarray | None:
    """Accessions for a cache written before they were stored, re-read from
    the cached FASTA — only when its entries match the cache row for row."""
    fasta = cache_dir / "uniprot_sprot.fasta.gz"
    names = meta.get("names")
    if names is None or not fasta.exists():
        return None
    accessions, fasta_names, _, _ = _parse_swissprot_fasta(fasta)
    if len(fasta_names) != len(names) or any(
        str(a) != str(b)
        for a, b in zip(fasta_names, names)  # noqa: B905 - lengths checked
    ):
        aprint("  ⓘ Cached FASTA does not match the cache; keeping name-based taxa")
        return None
    return np.array(accessions, dtype=object)


def refresh_kingdoms(
    cache_dir: Path,
    metadata_cache: Path,
    meta: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the taxon categories, re-deriving them from the UniProt lineage
    (and rewriting ``metadata_cache``) when the cache predates that.

    Idempotent: a cache already stamped :data:`KINGDOM_SOURCE` is returned as
    is. A cache without accessions, or a machine that cannot reach UniProt,
    keeps its existing categories.
    """
    kingdoms = np.asarray(meta["kingdoms"], dtype=object)
    if str(meta.get("kingdom_source", "")) == KINGDOM_SOURCE:
        return kingdoms
    accessions = meta.get("accessions")
    if accessions is None or len(accessions) != len(kingdoms):
        accessions = _recover_accessions(cache_dir, meta)
        if accessions is None:
            return kingdoms
        meta = {**meta, "accessions": accessions}
    table = _lineage_table(cache_dir)
    if table is None:
        return kingdoms
    organisms = meta["organisms"]
    fresh = np.array(
        [
            _classify_lineage(table.get(str(acc), "")) or _classify_organism(str(org))
            for acc, org in zip(accessions, organisms)  # noqa: B905 - same length
        ],
        dtype=object,
    )
    changed = int((fresh != kingdoms).sum())
    aprint(f"✓ Taxon categories from the UniProt lineage ({changed:,} relabelled)")
    np.savez(
        metadata_cache,
        **{k: v for k, v in meta.items() if k != "kingdoms"},
        kingdoms=fresh,
        kingdom_source=np.array(KINGDOM_SOURCE),
    )
    return fresh


# =============================================================================
# Swiss-Prot Parsing
# =============================================================================


def _parse_swissprot_fasta(
    fasta_path: Path,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Parse Swiss-Prot FASTA: extract accessions, names, organisms, sequences.

    Header format: >sp|ACCESSION|ENTRY_NAME Description OS=Organism OX=TaxID ...
    """
    accessions = []
    names = []
    organisms = []
    sequences = []

    with asection(f"Parsing Swiss-Prot FASTA ({fasta_path.name})"):
        opener = gzip.open if str(fasta_path).endswith(".gz") else open
        current_seq_parts: list[str] = []
        current_acc = ""
        current_name = ""
        current_org = ""

        with opener(fasta_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence
                    if current_acc:
                        accessions.append(current_acc)
                        names.append(current_name)
                        organisms.append(current_org)
                        sequences.append("".join(current_seq_parts))
                        current_seq_parts = []

                    # Parse header: >sp|P12345|INS_HUMAN Insulin OS=Homo sapiens OX=9606 ...
                    parts = line[1:].split("|")
                    if len(parts) >= 3:
                        current_acc = parts[1]
                        # Entry name + description
                        rest = parts[2]
                        # Name is everything before " OS="
                        os_match = re.search(r"\s+OS=", rest)
                        if os_match:
                            current_name = rest[
                                rest.index(" ") + 1 : os_match.start()
                            ].strip()
                        else:
                            current_name = rest.split()[0] if rest else ""

                        # Organism: between OS= and OX= (or end)
                        os_match2 = re.search(
                            r"OS=(.+?)(?:\s+OX=|\s+GN=|\s+PE=|$)", rest
                        )
                        current_org = os_match2.group(1).strip() if os_match2 else ""
                    else:
                        current_acc = line[1:].split()[0]
                        current_name = ""
                        current_org = ""
                else:
                    current_seq_parts.append(line)

            # Don't forget the last sequence
            if current_acc:
                accessions.append(current_acc)
                names.append(current_name)
                organisms.append(current_org)
                sequences.append("".join(current_seq_parts))

        aprint(f"✓ Parsed {len(accessions):,} proteins")
        # Show some stats
        avg_len = sum(len(s) for s in sequences) / max(len(sequences), 1)
        aprint(f"  Average sequence length: {avg_len:.0f} residues")
        aprint(f"  Sample: {accessions[0]} — {names[0]} ({organisms[0]})")

    return accessions, names, organisms, sequences


# =============================================================================
# ESM Embedding Computation
# =============================================================================

# Remedies embedded in a quarantine notice, per failure site. The no-CUDA branch
# is fixable only by supplying a cache or moving to a CUDA box, and its message
# names the destination `.npy` in a bullet right above the notice, so "that path"
# resolves there.
_CUDA_ACTION = (
    "re-download the complete file to that path, or delete the "
    "quarantined copy and rerun on a CUDA machine"
)


def _dep_action(embeddings_cache: Path) -> str:
    """Remedy for a torch/esm gate failure, for the quarantine notice.

    Deliberately not `_CUDA_ACTION`: installing the package is a second way
    forward here (and "rerun on a CUDA machine" is wrong for the esm gate, which
    is only reached when CUDA IS available). It also has to spell the
    destination out — a dependency error carries no surrounding text naming it,
    so a bare "that path" would point at the `.corrupt` file the notice lists.
    """
    return (
        "install the missing dependency above and rerun, or supply a complete "
        f"file at {embeddings_cache} — a complete cache skips the dependency "
        "entirely, and the quarantined copy is never reused"
    )


def _compute_esm3_embeddings(
    sequences: list[str],
    cache_dir: Path,
    model_name: str = "esm3-open",
    max_length: int = 1024,
    *,
    already_reported_quarantine: frozenset[Path] = frozenset(),
) -> np.ndarray:
    """Compute ESM-3 mean-pooled embeddings with checkpointing.

    Returns (N, d_model) float32 array.
    """
    embeddings_cache = cache_dir / f"embeddings_{model_name.replace('-', '_')}.npy"
    checkpoint_path = (
        cache_dir / f"embeddings_{model_name.replace('-', '_')}_checkpoint.npz"
    )

    # Per-model embedding dimension (mirrors the model-load block below), used
    # to validate a precomputed/cached embeddings file before trusting it.
    expected_dim = {"esmc-300m": 960, "esmc-600m": 1152}.get(model_name, 1536)
    expected_shape = (len(sequences), expected_dim)

    # Load and VALIDATE a completed cache. A truncated / incomplete / wrong-
    # shaped or unreadable file (e.g. a partial download) is quarantined to
    # `<name>.corrupt` and treated as absent, so the demo never silently
    # proceeds with malformed embeddings.
    if embeddings_cache.exists():
        with asection("Loading cached ESM embeddings"):
            try:
                embeddings = np.load(embeddings_cache)
            except Exception as exc:  # noqa: BLE001 - any read failure ⇒ quarantine
                embeddings = None
                aprint(f"⚠ Cached embeddings could not be read: {exc}")
            if embeddings is not None and embeddings.shape == expected_shape:
                aprint(
                    f"✓ Loaded {embeddings.shape[0]:,} × {embeddings.shape[1]}D embeddings"
                )
                return embeddings
            actual = "unreadable" if embeddings is None else f"shape {embeddings.shape}"
            quarantine_file(
                embeddings_cache,
                reason=f"invalid embeddings ({actual}, expected {expected_shape})",
            )

    # Report ANY quarantined copy — including one left behind by an EARLIER run
    # (the common case: the `.npy` is already gone, so the block above never
    # fires). Without this the demo silently restarts a multi-GB fetch/compute
    # with no hint that a rejected copy is sitting in the cache.
    # verbose=False: `main()` already prints a dir-wide notice for this cache
    # before anything expensive starts, so printing again here would show the
    # user two near-identical warnings about the SAME file three lines apart —
    # which reads like two separate corrupt artifacts. Collect the paths silently
    # and let them enrich the errors below — the no-CUDA RuntimeError AND the
    # torch/esm dependency errors — so a caller that bypassed `main()` still sees
    # them.
    quarantined = warn_if_quarantined(embeddings_cache, verbose=False)

    # `main()` may already have printed a notice for some of these paths up
    # front; embed ONLY the ones it did not, so this notice is emitted once per
    # file. The check is per-PATH (not a dir-wide flag): `main()`'s scan is
    # dir-wide, so it may well have reported some OTHER model's leftover, but a
    # file quarantined during THIS run (by the block above) is not in
    # `already_reported_quarantine` and so is still named here. An empty
    # `unreported` is the single guard for both "nothing quarantined" and "main
    # already reported everything".
    unreported = [p for p in quarantined if p not in already_reported_quarantine]

    def _quarantine_note(action: str) -> str:
        """Format a quarantine notice for the not-yet-reported rejected copies."""
        return format_quarantine_notice(unreported, indent="  ", action=action)

    # Past the cache check, so the compute path is genuinely being taken: this
    # is where torch becomes mandatory (the CUDA probe below needs it). `esm` is
    # demanded later, just before the model load — on a machine without CUDA the
    # "supply a complete cache" message below is the actionable one, and it
    # carries the quarantine notice, so it must not be pre-empted by a
    # missing-esm error the user cannot act on anyway. When a rejected copy is
    # sitting in the cache and `main()` did not already report it, append the
    # quarantine notice so a direct/programmatic caller learns about the
    # multi-GB artifact instead of it being silently invisible.
    try:
        torch = require_module("torch")
    except MissingDependencyError as exc:
        note = _quarantine_note(_dep_action(embeddings_cache))
        if note:
            raise MissingDependencyError(f"{exc}\n{note}") from exc
        raise

    # Computing ESM embeddings for ~572K proteins is only practical on a CUDA
    # GPU. If no usable cache is present and no CUDA device is available, fail
    # fast with an actionable message rather than downloading the model and then
    # crashing on `.to("cuda")` (or grinding for many hours on CPU/MPS).
    if not torch.cuda.is_available():
        note = _quarantine_note(_CUDA_ACTION)
        note_line = f"{note}\n" if note else ""
        raise RuntimeError(
            "No usable cached embeddings were found and CUDA is not available, "
            "so ESM embeddings cannot be (re)computed on this machine.\n"
            "  • This demo computes embeddings only on an NVIDIA/CUDA GPU "
            "(computing ~572K proteins on CPU/MPS is impractical).\n"
            "  • Supply a complete precomputed embeddings file at:\n"
            f"      {embeddings_cache}\n"
            f"    (expected shape: {expected_shape}, float32).\n"
            f"{note_line}"
            "  • Or run this demo on a CUDA GPU machine to compute it from "
            "scratch."
        )

    # Load model — the one place `esm` itself is genuinely needed. As with the
    # torch gate, enrich a missing-esm error with the quarantine notice for a
    # direct caller that has not already been told about the rejected copy.
    try:
        require_module("esm")
    except MissingDependencyError as exc:
        note = _quarantine_note(_dep_action(embeddings_cache))
        if note:
            raise MissingDependencyError(f"{exc}\n{note}") from exc
        raise
    with asection(f"Loading ESM model: {model_name}"):
        if model_name == "esmc-300m":
            from esm.models.esmc import ESMC

            model = ESMC.from_pretrained("esmc_300m").to("cuda").eval()
            embed_dim = 960
        elif model_name == "esmc-600m":
            from esm.models.esmc import ESMC

            model = ESMC.from_pretrained("esmc_600m").to("cuda").eval()
            embed_dim = 1152
        else:
            from esm.models.esm3 import ESM3

            model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda").eval()
            embed_dim = 1536

        aprint(f"✓ Model loaded on GPU (embedding dim: {embed_dim})")

    n = len(sequences)
    embeddings = np.zeros((n, embed_dim), dtype=np.float32)

    # Resume from checkpoint if available
    start_idx = 0
    if checkpoint_path.exists():
        with asection("Resuming from checkpoint"):
            ckpt = np.load(checkpoint_path)
            start_idx = int(ckpt["completed"])
            embeddings[:start_idx] = ckpt["embeddings"][:start_idx]
            aprint(f"✓ Resumed from protein {start_idx:,}/{n:,}")

    # Sort by length for efficient batching (process in order, store in original order)
    indices_by_length = sorted(range(start_idx, n), key=lambda i: len(sequences[i]))

    with asection(f"Computing embeddings ({n - start_idx:,} remaining)"):
        from esm.sdk.api import ESMProtein, LogitsConfig

        completed = start_idx
        for idx in indices_by_length:
            seq = sequences[idx][:max_length]  # Truncate long sequences

            try:
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)

                with torch.no_grad():
                    output = model.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_embeddings=True),
                    )

                # Mean-pool over sequence length
                emb = output.embeddings[0].mean(dim=0).cpu().numpy()
                embeddings[idx] = emb

            except Exception as e:
                if completed % 1000 == 0:
                    aprint(f"  ⚠ Protein {idx} ({len(seq)} residues) failed: {e}")
                # Leave as zeros — will appear as gray outlier

            completed += 1

            # Progress reporting
            if completed % 1000 == 0:
                pct = completed / n * 100
                aprint(f"  {completed:,}/{n:,} ({pct:.1f}%)")

            # Checkpoint
            if completed % CHECKPOINT_INTERVAL == 0:
                np.savez(
                    checkpoint_path,
                    embeddings=embeddings,
                    completed=completed,
                )
                aprint(f"  💾 Checkpoint saved at {completed:,}")

        aprint(f"✓ All {n:,} embeddings computed")

    # Save final result and remove checkpoint
    np.save(embeddings_cache, embeddings)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    aprint(
        f"✓ Saved to {embeddings_cache} ({embeddings_cache.stat().st_size / 1e9:.1f} GB)"
    )

    return embeddings


# =============================================================================
# UMAP Reduction
# =============================================================================


def _reduce_to_3d(
    embeddings: np.ndarray,
    cache_path: Path | None = None,
) -> np.ndarray:
    """UMAP 3D reduction with caching."""
    if cache_path and cache_path.exists():
        positions = np.load(cache_path)["positions"]
        aprint(f"✓ Loaded 3D UMAP from cache ({len(positions):,} points)")
        return positions

    UMAP = require_module("umap").UMAP

    with asection(
        f"UMAP reduction ({embeddings.shape[0]:,} × {embeddings.shape[1]}D → 3D)"
    ):
        aprint("Parameters: n_neighbors=15, min_dist=0.1, metric=cosine")
        aprint("This may take 30-60 minutes for ~500K proteins...")

        reducer = UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            n_jobs=-1,
            low_memory=True,
            verbose=True,
        )
        positions = reducer.fit_transform(embeddings).astype(np.float32)
        positions -= positions.mean(axis=0)
        aprint(f"✓ UMAP complete: {positions.shape}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, positions=positions)
        aprint(f"✓ Cached to {cache_path}")

    return positions


# =============================================================================
# Scene Generation
# =============================================================================


def generate_esm3_landscape(
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    # Same default as `main()`, so a caller that omits it gets the model
    # DEMO_META credits rather than one the CLI never runs.
    model_name: str = "esmc-300m",
    cache_dir: Path | None = None,
    already_reported_quarantine: frozenset[Path] = frozenset(),
) -> int:
    """Generate 3D ESM-3 protein embedding landscape."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "luxar" / "esm3_swissprot"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Check for final UMAP cache (instant path) ---
    sample_tag = f"_{sample_size}" if sample_size > 0 else "_all"
    umap_cache = cache_dir / f"umap3d_{model_name.replace('-', '_')}{sample_tag}.npz"
    metadata_cache = cache_dir / f"metadata{sample_tag}.npz"

    if umap_cache.exists() and metadata_cache.exists():
        with asection("Loading cached results (instant)"):
            positions = np.load(umap_cache)["positions"]
            meta = np.load(metadata_cache, allow_pickle=True)
            protein_names = list(meta["names"])
            organism_names = list(meta["organisms"])
            kingdoms = list(meta["kingdoms"])
            # Absent from caches written before accessions were persisted. The
            # cache is expensive to rebuild, so the protein name becomes a
            # UniProt search key rather than forcing regeneration.
            accessions = list(meta["accessions"]) if "accessions" in meta.files else []
            kingdoms = list(
                refresh_kingdoms(
                    cache_dir, metadata_cache, {k: meta[k] for k in meta.files}
                )
            )
            n = len(positions)
            aprint(f"✓ Loaded {n:,} proteins from cache")
    else:
        # --- Step 1: Download Swiss-Prot ---
        fasta_path = cache_dir / "uniprot_sprot.fasta.gz"
        if not fasta_path.exists():
            from luxar.demos import robust_download

            with asection("Downloading Swiss-Prot FASTA (~90 MB)"):
                for url in SWISSPROT_FASTA_URLS:
                    try:
                        robust_download(url, fasta_path)
                        break
                    except Exception as e:
                        aprint(f"  ⚠ {url.split('/')[2]} failed: {e}")
                        continue
                else:
                    raise RuntimeError("All Swiss-Prot mirrors failed")

        # --- Step 2: Parse FASTA ---
        accessions, protein_names, organism_names, sequences = _parse_swissprot_fasta(
            fasta_path
        )
        lineages = _lineage_table(cache_dir)
        kingdoms = [
            (_classify_lineage(lineages.get(acc, "")) if lineages else None)
            or _classify_organism(org)
            for acc, org in zip(accessions, organism_names)  # noqa: B905 - same length
        ]

        # Subsample
        n = len(sequences)
        if sample_size > 0 and sample_size < n:
            rng = np.random.default_rng(42)
            indices = rng.choice(n, sample_size, replace=False)
            indices.sort()
            accessions = [accessions[i] for i in indices]
            protein_names = [protein_names[i] for i in indices]
            organism_names = [organism_names[i] for i in indices]
            sequences = [sequences[i] for i in indices]
            kingdoms = [kingdoms[i] for i in indices]
            n = sample_size
            aprint(f"Subsampled to {n:,} proteins")

        # Save metadata
        np.savez(
            metadata_cache,
            names=np.array(protein_names, dtype=object),
            organisms=np.array(organism_names, dtype=object),
            kingdoms=np.array(kingdoms, dtype=object),
            # The Swiss-Prot accession, for the UniProt link (#1917). Parsed
            # all along and then thrown away; the visible label is
            # "<protein name> — <organism> (<kingdom>)", which no URL can be
            # built from.
            accessions=np.array(accessions, dtype=object),
            **({"kingdom_source": np.array(KINGDOM_SOURCE)} if lineages else {}),
        )

        # --- Step 3: Compute ESM embeddings ---
        embeddings = _compute_esm3_embeddings(
            sequences,
            cache_dir,
            model_name,
            already_reported_quarantine=already_reported_quarantine,
        )

        # Sequences are already subsampled (and the cache is shape-validated
        # against len(sequences)), so embeddings line up 1:1 with the rng.choice
        # selected names/organisms/kingdoms. Assert it loudly rather than
        # positionally truncating, which would silently misalign the metadata.
        assert len(embeddings) == n, (
            f"embeddings ({len(embeddings)}) misaligned with selected metadata ({n})"
        )

        # --- Step 4: UMAP ---
        positions = _reduce_to_3d(embeddings, cache_path=umap_cache)

    # --- Step 5: Build Luxar scene ---
    n = len(positions)
    with asection("Generating visualization"):
        # Two switchable coloring views: fine taxon (12 categories) and coarse
        # domain of life (Eukaryota / Bacteria / Archaea / Viruses).
        domains = [_DOMAIN_OF.get(k, "Other") for k in kingdoms]
        taxon_colors = np.array(
            [TAXON_COLORS.get(k, TAXON_COLORS["Other"]) for k in kingdoms],
            dtype=np.float32,
        )
        domain_colors = np.array(
            [DOMAIN_COLORS.get(d, DOMAIN_COLORS["Other"]) for d in domains],
            dtype=np.float32,
        )

        kingdom_counts: dict[str, int] = {}
        for k in kingdoms:
            kingdom_counts[k] = kingdom_counts.get(k, 0) + 1
        aprint("✓ Proteins by taxon:")
        for k, count in sorted(kingdom_counts.items(), key=lambda x: -x[1]):
            aprint(f"  {k}: {count:,}")

        # Hover shows the category active in the current view.
        taxon_labels = [
            f"{protein_names[i]} — {organism_names[i]} ({kingdoms[i]})"
            for i in range(n)
        ]
        domain_labels = [
            f"{protein_names[i]} — {organism_names[i]} ({domains[i]})" for i in range(n)
        ]
        # Click a protein to open its Swiss-Prot entry, right-click to copy the
        # accession (#1917). Neither hover label contains it — both are
        # "<protein name> — <organism> (<category>)" — so the URL comes from
        # `keys=`, passed once for the whole cloud because a protein's accession
        # does not change with the colour scheme, only its label does.
        #
        # When cached metadata predates accessions, retain the expensive cache
        # and use the clean protein name as a UniProt search key instead.
        linkable_accessions = _linkable_accessions(accessions, n)
        stacked = stack_colorings(
            positions,
            [
                {"label": "Taxon", "colors": taxon_colors, "labels": taxon_labels},
                {"label": "Domain", "colors": domain_colors, "labels": domain_labels},
            ],
            keys=linkable_accessions
            if linkable_accessions is not None
            else protein_names,
        )
        assert stacked.keys is not None
        link_attrs = _uniprot_link_attrs(
            stacked.keys, exact_accessions=linkable_accessions is not None
        )
        radii = np.full(len(stacked.positions), 0.012, dtype=np.float32)

    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension(
                    "coloring",
                    unit="",
                    categories=stacked.categories,
                    display=False,
                    description="Color scheme: fine taxon vs coarse domain of life",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        model_label = "ESM-3" if "esm3" in model_name else f"ESM C ({model_name})"
        # Cite the model that actually produced the embeddings: ESM3 → Hayes
        # et al. 2025; ESM C (esmc-*) → the EvolutionaryScale ESM C release.
        model_citation = (
            "Hayes et al. 2025"
            if "esm3" in model_name
            else "EvolutionaryScale ESM C, 2024"
        )
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                # Stamp the model that actually produced these embeddings, not
                # the one DEMO_META names for the default run -- otherwise a
                # `--model=` override writes a credit contradicting this
                # scene's own footer two calls below.
                citation={
                    **DEMO_META["citation"],
                    "short": f"UniProt/Swiss-Prot; embeddings by {model_citation}",
                },
                viewer_config=ViewerConfig(cinematic_mode=True),
            )

            # Additive ladder only — no substitutive levels. ~572k proteins per
            # coloring is not a large cloud by this viewer's standards: stacked
            # on the hidden `coloring` axis, the resident slice is ~575,503
            # points against a 5,591,040 Points cap, ~10x under it. The coarse
            # levels served a framing the screen-area selector never picks (the
            # finest is anchored at half-screen occupancy and this demo opens
            # auto-fitted), and cost four levels of nodes: 13 groups -> 5.
            #
            # It also retires the substitutive_lod_or_flat gate here. That gate
            # existed because the coarsening write path imports torch+scipy,
            # which broke the "complete cache runs anywhere" contract; an
            # additive ladder imports neither, so the contract now holds without
            # a degraded no-LOD fallback.
            scene.add_points(
                "proteins",
                positions=stacked.positions,
                colors=stacked.colors,
                radii=radii,
                sharpness=np.full(len(stacked.positions), 0.6, dtype=np.float32),
                opacity=0.9,
                intensity=0.12,
                labels=stacked.labels,
                **link_attrs,
                layer=True,
                additive_lod=stream_ladder(
                    len(stacked.positions),
                    slices=hidden_axis_stops(stacked.positions, dims.non_displayed),
                ),
            )

            scene.add_text(
                f"{model_label} Protein Landscape — Swiss-Prot",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            add_demo_caption(
                scene,
                f"{n:,} proteins • {model_label} embeddings • 3D UMAP • {model_citation}",
                DEMO_META.get("citation"),
            )

            # Per-view color legends (each visible only on its coloring slot).
            def _legend_html(heading: str, items: list) -> str:
                """Build an HTML color-swatch legend from (name, rgb) items."""
                html = (
                    '<div style="font-size:1.2vh;line-height:1.5;'
                    'background:rgba(0,0,0,0.5);padding:0.5vh;border-radius:3px">'
                    f'<div style="font-weight:bold;color:#ccc;'
                    f'margin-bottom:0.3vh">{heading}</div>'
                )
                for name, rgb in items:
                    r, g, b = (int(round(v * 255)) for v in rgb)
                    html += (
                        f'<div><span style="color:#{r:02x}{g:02x}{b:02x}">█</span> '
                        f"{name}</div>"
                    )
                return html + "</div>"

            taxa_present = [
                k for k, _ in sorted(kingdom_counts.items(), key=lambda x: -x[1])
            ]
            scene.add_html(
                _legend_html(
                    "Taxon",
                    [
                        (k, TAXON_COLORS.get(k, TAXON_COLORS["Other"]))
                        for k in taxa_present
                    ],
                ),
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"coloring": 0},
                transition="fade",
                transition_duration=0.3,
            )
            domain_counts: dict[str, int] = {}
            for d in domains:
                domain_counts[d] = domain_counts.get(d, 0) + 1
            domains_present = [
                d for d, _ in sorted(domain_counts.items(), key=lambda x: -x[1])
            ]
            scene.add_html(
                _legend_html(
                    "Domain",
                    [
                        (d, DOMAIN_COLORS.get(d, DOMAIN_COLORS["Other"]))
                        for d in domains_present
                    ],
                ),
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"coloring": 1},
                transition="fade",
                transition_duration=0.3,
            )

    aprint(f"✓ Wrote {n:,} proteins to {output_path}")
    return n


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Parse args, report the cache state, and build/serve the landscape scene."""
    aprint("=" * 70)
    aprint("ESM-3 PROTEIN LANDSCAPE — Swiss-Prot Embeddings")
    aprint("=" * 70)
    aprint("")
    aprint("~572K proteins embedded with ESM-3, visualized as 3D UMAP.")
    aprint("Hover over any point to see protein name, organism, and kingdom.")
    aprint("")

    # Parse args
    sample_size = DEFAULT_SAMPLE_SIZE
    model_name = "esmc-300m"
    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])
            aprint(f"Sample size: {sample_size:,}")
        elif arg.startswith("--model="):
            model_name = arg.split("=")[1]
            aprint(f"Model: {model_name}")

    if sample_size > 0:
        aprint(f"Using {sample_size:,} proteins (subsample)")
    else:
        aprint("Using ALL Swiss-Prot proteins (~572K)")
    aprint(f"Model: {model_name}")
    aprint("")

    cache_dir = Path.home() / ".cache" / "luxar" / "esm3_swissprot"

    # Report the cache's real state FIRST. A quarantined `.corrupt` artifact is
    # the difference between "instant run" and "multi-GB re-download", so the
    # user learns about it even when a dependency gate below also trips.
    quarantined = find_quarantined_files(cache_dir)
    # The SET of paths reported here, threaded into the compute path so it embeds
    # the notice only for paths we did NOT report — so this leftover notice is
    # emitted once per file. It covers only what was on disk BEFORE the run: a
    # cache rejected DURING the run announces itself through `quarantine_file`,
    # and the compute path then names it in the error it raises.
    if quarantined:
        aprint(
            format_quarantine_notice(
                quarantined,
                indent="",
                action=(
                    "re-download the complete file under the same name (minus "
                    f"'{QUARANTINE_SUFFIX}') into {cache_dir}, or delete the "
                    "quarantined copy to reclaim the disk space — otherwise this "
                    "run recomputes/re-downloads it from scratch"
                ),
            )
        )
        aprint("")

    # NO dependency preflight here on purpose. torch / esm / umap-learn are
    # demanded by `require_module` at the exact points that need them, so a
    # machine holding complete caches runs the demo without any of them
    # installed — scene generation falls back to flat Points when torch or
    # scipy is absent (the Points LOD path needs both). Gating up front would
    # refuse the cache-only path that the quarantine notice above tells the user
    # to aim for.

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "esm3_protein_landscape.luxar.zarr"
        try:
            n = generate_esm3_landscape(
                output_path,
                sample_size=sample_size,
                model_name=model_name,
                cache_dir=cache_dir,
                already_reported_quarantine=frozenset(quarantined),
            )
            if n == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total proteins: {n:,}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_esm3_") as tmpdir:
        output_path = Path(tmpdir) / "esm3_protein_landscape.luxar.zarr"

        try:
            n = generate_esm3_landscape(
                output_path,
                sample_size=sample_size,
                model_name=model_name,
                cache_dir=cache_dir,
                already_reported_quarantine=frozenset(quarantined),
            )
            if n == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("Explore the protein universe:")
        aprint(
            "  - Colored by taxonomic group (12 categories: vertebrates in blues, "
            "bacteria in greens, Archaea orange, Viruses red, ...)"
        )
        aprint(
            "  - Clusters = proteins with similar ESM-3 embeddings (shared function/fold)"
        )
        aprint("  - Hover over any point to see protein name and organism")
        aprint("")
        aprint(f"Total proteins: {n:,}")
        aprint("")
        aprint("Press Ctrl+C when done.")

        launch_viewer(output_path)

    aprint("✓ Cleanup complete")


if __name__ == "__main__":
    main()
