"""
ESM-3 Protein Landscape — Swiss-Prot 3D Embedding
===================================================

Visualizes ~572K Swiss-Prot protein embeddings from ESM-3 (or ESM C) as a
3D UMAP point cloud. Each point is a protein, colored by taxonomic kingdom,
with hover labels showing protein name, organism, and kingdom.

Run behavior:
  First run: Download Swiss-Prot, compute ESM embeddings, run UMAP (~5h)
  Subsequent runs: Load cached results instantly

Data source: UniProt/Swiss-Prot (CC BY 4.0)
Model: ESM-3 open (Hayes et al. 2025) or ESM C 300M (EvolutionaryScale)

Usage:
    python -m luxar.demos.demo_esm3_protein_landscape
    python -m luxar.demos.demo_esm3_protein_landscape --no-serve
    python -m luxar.demos.demo_esm3_protein_landscape --sample=100000
    python -m luxar.demos.demo_esm3_protein_landscape --model=esmc-300m

Dependencies:
    pip install luxar[demos] esm
"""

import gzip
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 0  # 0 = all (~572K)

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

    # Check for completed cache
    if embeddings_cache.exists():
        with asection("Loading cached ESM embeddings"):
            embeddings = np.load(embeddings_cache)
            aprint(
                f"✓ Loaded {embeddings.shape[0]:,} × {embeddings.shape[1]}D embeddings"
            )
            return embeddings

    import torch

    # Load model
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

    from umap import UMAP

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

        # Subsample embeddings if needed (in case full embeddings were cached but we want a subset)
        if sample_size > 0 and len(embeddings) > n:
            embeddings = embeddings[:n]

        # --- Step 4: UMAP ---
        positions = _reduce_to_3d(embeddings, cache_path=umap_cache)

    # --- Step 5: Build Luxar scene ---
    n = len(positions)
    with asection("Generating visualization"):
        # Colors by kingdom
        colors = np.zeros((n, 3), dtype=np.float32)
        kingdom_counts: dict[str, int] = {}
        for i, k in enumerate(kingdoms):
            colors[i] = TAXON_COLORS.get(k, TAXON_COLORS["Other"])
            kingdom_counts[k] = kingdom_counts.get(k, 0) + 1

        aprint("✓ Proteins by taxon:")
        for k, count in sorted(kingdom_counts.items(), key=lambda x: -x[1]):
            aprint(f"  {k}: {count:,}")

        radii = np.full(n, 0.012, dtype=np.float32)

        # Hover labels: "Insulin — Homo sapiens (Eukaryota)"
        labels = [
            f"{protein_names[i]} — {organism_names[i]} ({kingdoms[i]})"
            for i in range(n)
        ]

    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        model_label = "ESM-3" if "esm3" in model_name else f"ESM C ({model_name})"
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "proteins",
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=np.full(n, 0.6, dtype=np.float32),
                opacity=0.9,
                intensity=0.12,
                labels=labels,
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
                f"{n:,} proteins • {model_label} embeddings • 3D UMAP • Hayes et al. 2025",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
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

    # Check dependencies
    try:
        import torch  # noqa: F401
    except ImportError:
        aprint("Missing dependency: torch")
        aprint("Install with: pip install torch")
        sys.exit(1)

    try:
        import esm  # noqa: F401
    except ImportError:
        aprint("Missing dependency: esm")
        aprint("Install with: pip install esm")
        sys.exit(1)

    try:
        import umap  # noqa: F401
    except ImportError:
        aprint("Missing dependency: umap-learn")
        aprint("Install with: pip install umap-learn")
        sys.exit(1)

    cache_dir = Path.home() / ".cache" / "luxar" / "esm3_swissprot"

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "esm3_protein_landscape.zarr"
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
        output_path = Path(tmpdir) / "esm3_protein_landscape.zarr"

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
            "  - Blue = Eukaryota, Green = Bacteria, Orange = Archaea, Red = Viruses"
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
