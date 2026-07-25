"""
ESM-3 Protein Landscape — Swiss-Prot 3D Embedding
===================================================

Visualizes ~572K Swiss-Prot protein embeddings from ESM-3 (or ESM C) as a
3D UMAP point cloud. Each point is a protein, colored by taxonomic kingdom,
with hover labels showing protein name, organism, and kingdom.

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

Cache hygiene:
    A cached artifact that fails validation is quarantined to ``<name>.corrupt``
    and never reused. The demo reports any quarantined file (path + size) before
    doing anything expensive — re-download the complete file or delete the
    quarantined copy, otherwise the run starts over from scratch.
"""

DEMO_META = {
    "key": "esm3_protein_landscape",
    "title": "ESM3 Protein Landscape",
    "description": "~572K Swiss-Prot protein embeddings from ESM-3 as a 3D UMAP, colored by taxonomic kingdom.",
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
}

import gzip
import importlib
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer, stack_colorings
from luxar.utils.download import (
    QUARANTINE_SUFFIX,
    find_quarantined_files,
    format_quarantine_notice,
    warn_if_quarantined,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 0  # 0 = all (~572K)

# Optional heavyweight dependencies, each demanded ONLY at the point where the
# corresponding uncached computation happens — never as an entry-point preflight.
# A complete embeddings cache returns before torch/esm are touched, and a
# complete UMAP cache returns before umap-learn is, so gating up front would
# refuse to run a machine that has every artifact it needs. Values are
# (pip spec, luxar extra, extra note).
_INSTALL_HINTS: dict[str, tuple[str, str, str]] = {
    "torch": (
        "torch>=2.2,<3.0",
        "gsplats",
        "Needed only to COMPUTE embeddings (a CUDA GPU is required for that).",
    ),
    "esm": (
        "esm>=3.0.0",
        "demos",
        "Needed only to COMPUTE embeddings; a complete cached embeddings file "
        "skips the model entirely.",
    ),
    "umap": (
        "umap-learn>=0.5.0",
        "demos",
        "Needed only to COMPUTE the 3D projection; a cached UMAP skips it.",
    ),
}


def _require_module(name: str) -> Any:
    """Import an optional dependency, or raise with an actionable install hint.

    Raises ``ImportError`` rather than exiting, so callers keep control and the
    demo's own error reporting stays in one place.
    """
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        spec, extra, note = _INSTALL_HINTS.get(name, (name, "demos", ""))
        raise ImportError(
            f"Missing dependency: {name}. Install with `pip install '{spec}'` "
            f"(or the whole extra: `pip install 'luxar[{extra}]'`). {note}".rstrip()
        ) from exc


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


def _compute_esm3_embeddings(
    sequences: list[str],
    cache_dir: Path,
    model_name: str = "esm3-open",
    max_length: int = 1024,
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
            corrupt_path = embeddings_cache.with_name(
                embeddings_cache.name + ".corrupt"
            )
            embeddings_cache.rename(corrupt_path)
            aprint(
                f"⚠ Cached embeddings are invalid ({actual}, expected "
                f"{expected_shape}); quarantined to {corrupt_path.name} — recomputing."
            )

    # Report ANY quarantined copy — including one left behind by an EARLIER run
    # (the common case: the `.npy` is already gone, so the block above never
    # fires). Without this the demo silently restarts a multi-GB fetch/compute
    # with no hint that a rejected copy is sitting in the cache.
    # verbose=False: `main()` already prints a dir-wide notice for this cache
    # before anything expensive starts, so printing again here would show the
    # user two near-identical warnings about the SAME file three lines apart —
    # which reads like two separate corrupt artifacts. Collect the paths silently
    # and let them enrich the RuntimeError below instead, which is where a
    # caller that bypassed `main()` still needs them.
    quarantined = warn_if_quarantined(embeddings_cache, verbose=False)

    # Past the cache check, so the compute path is genuinely being taken: this
    # is where torch becomes mandatory (the CUDA probe below needs it). `esm` is
    # demanded later, just before the model load — on a machine without CUDA the
    # "supply a complete cache" message below is the actionable one, and it
    # carries the quarantine notice, so it must not be pre-empted by a
    # missing-esm error the user cannot act on anyway.
    torch = _require_module("torch")

    # Computing ESM embeddings for ~572K proteins is only practical on a CUDA
    # GPU. If no usable cache is present and no CUDA device is available, fail
    # fast with an actionable message rather than downloading the model and then
    # crashing on `.to("cuda")` (or grinding for many hours on CPU/MPS).
    if not torch.cuda.is_available():
        notice = format_quarantine_notice(
            quarantined,
            indent="  ",
            action=(
                "re-download the complete file to that path, or delete the "
                "quarantined copy and rerun on a CUDA machine"
            ),
        )
        quarantine_note = f"{notice}\n" if notice else ""
        raise RuntimeError(
            "No usable cached embeddings were found and CUDA is not available, "
            "so ESM embeddings cannot be (re)computed on this machine.\n"
            "  • This demo computes embeddings only on an NVIDIA/CUDA GPU "
            "(computing ~572K proteins on CPU/MPS is impractical).\n"
            "  • Supply a complete precomputed embeddings file at:\n"
            f"      {embeddings_cache}\n"
            f"    (expected shape: {expected_shape}, float32).\n"
            f"{quarantine_note}"
            "  • Or run this demo on a CUDA GPU machine to compute it from "
            "scratch."
        )

    # Load model — the one place `esm` itself is genuinely needed.
    _require_module("esm")
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

    UMAP = _require_module("umap").UMAP

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
    model_name: str = "esm3-open",
    cache_dir: Path | None = None,
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
            n = len(positions)
            aprint(f"✓ Loaded {n:,} proteins from cache")
    else:
        # --- Step 1: Download Swiss-Prot ---
        fasta_path = cache_dir / "uniprot_sprot.fasta.gz"
        if not fasta_path.exists():
            from luxar.utils.download import robust_download

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
        kingdoms = [_classify_organism(org) for org in organism_names]

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
        )

        # --- Step 3: Compute ESM embeddings ---
        embeddings = _compute_esm3_embeddings(sequences, cache_dir, model_name)

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
        stacked = stack_colorings(
            positions,
            [
                {"label": "Taxon", "colors": taxon_colors, "labels": taxon_labels},
                {"label": "Domain", "colors": domain_colors, "labels": domain_labels},
            ],
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
            scene = compiler.create_scene(dimensions=dims)

            # Substitutive Points LOD: ~572k proteins is a large cloud, so coarse
            # levels replace it with fewer, larger merged splats when zoomed out
            # (census-style wiring; coarse splats stay pure per coloring via the
            # `coloring` barrier).
            scene.add_points(
                "proteins",
                positions=stacked.positions,
                colors=stacked.colors,
                radii=radii,
                sharpness=np.full(len(stacked.positions), 0.6, dtype=np.float32),
                opacity=0.9,
                intensity=0.12,
                labels=stacked.labels,
                substitutive_lod=dict(compression_factor=8, levels=3, device="auto"),
            )

            scene.add_text(
                f"{model_label} Protein Landscape — Swiss-Prot",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            scene.add_text(
                f"{n:,} proteins • {model_label} embeddings • 3D UMAP • {model_citation}",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

            # Per-view color legends (each visible only on its coloring slot).
            def _legend_html(heading: str, items: list) -> str:
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
    # demanded by `_require_module` at the exact points that need them, so a
    # machine holding complete caches runs the demo without any of them
    # installed. Gating up front would refuse the cache-only path that the
    # quarantine notice above tells the user to aim for.

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "esm3_protein_landscape.luxar.zarr"
        try:
            n = generate_esm3_landscape(
                output_path,
                sample_size=sample_size,
                model_name=model_name,
                cache_dir=cache_dir,
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
