#!/usr/bin/env python3
"""Self-Contained Demo: Protein Function Landscape (ProtT5 Embeddings)

Visualize 142k proteins in 3D embedding space, showing how proteins with
similar functions cluster together.

================================================================================
PROTEIN FUNCTION PREDICTION & EMBEDDINGS
================================================================================

CAFA (Critical Assessment of Functional Annotation) is a community challenge
for predicting protein function from sequence. The CAFA5 dataset contains
142,246 proteins with known functional annotations.

WHAT ARE PROTEIN EMBEDDINGS?
-----------------------------
Protein embeddings are learned vector representations that capture a protein's
sequence, structure, and function in a high-dimensional space. Similar to how
word embeddings (Word2Vec) place semantically similar words close together,
protein embeddings place functionally similar proteins nearby.

PROTT5 MODEL:
-------------
ProtT5 is a protein language model based on T5 (Text-To-Text Transfer Transformer)
architecture, trained on billions of protein sequences. It learns to:
- Understand amino acid patterns
- Recognize functional motifs
- Capture evolutionary relationships
- Predict structural features

The model generates 1,024-dimensional embeddings that encode a protein's
characteristics.

Cite: Elnaggar et al. (2021), "ProtTrans: Toward Understanding the Language of
Life Through Self-Supervised Learning", IEEE TPAMI. DOI: 10.1109/TPAMI.2021.3095381
CAFA5 challenge: https://www.kaggle.com/competitions/cafa-5-protein-function-prediction

NAMING THE CLUSTERS (UniProt keyword enrichment):
-------------------------------------------------
The Kaggle bundle ships no usable annotation for these accessions — its
``CAFA1_train_terms.tsv`` covers 1,387 PDB-style entries (``3PHF-2``) and has
**zero** overlap with the 142,246 UniProt accessions in ``train_ids.npy``. So
the demo names its clusters from the annotation database itself: a random
sample of each cluster's members is looked up in the UniProt REST API, and the
cluster is named after the UniProt keyword most **over-represented** in it
relative to the pooled sample (lift = P(keyword | cluster) / P(keyword | all)).

That turns "Cluster 3" into "Transit peptide" or "G protein-coupled receptor" —
a name derived from the data, recomputed every run, never hardcoded (UMAP is
not seeded, so cluster identity is not stable across runs). Clusters with no
keyword clearly above background are honestly labelled "Mixed".

VISUALIZATION STRATEGY:
-----------------------
This demo:
1. Loads 142k pre-computed ProtT5 embeddings (1,024D)
2. Reduces to 3D using UMAP (preserves functional relationships)
3. Clusters the 3D landscape with k-means
4. Names each cluster by UniProt keyword enrichment, and colors by cluster

WHAT YOU'LL SEE:
- Named functional territories rather than anonymous colored blobs
- Secreted / signal-peptide proteins separated from cytoplasmic ones
- Organelle-targeted proteins (transit peptide, mitochondrion) as their own region
- Receptors / GPCRs as a distinct island
- Hover a point for its UniProt accession and cluster name

DATASET: CAFA5 (Kaggle)
-----------------------
Source: https://www.kaggle.com/datasets/horikitasaku/prott5-embedding-for-cafa5
- 142,246 proteins with ProtT5-XL embeddings
- 1,024 dimensions per protein
- 540 MB download (compressed)
- NumPy format (easy loading!)

Usage:
    python demo_protein_embeddings_cafa5.py [--sample=N]

    Options:
    --sample=N       Number of proteins to visualize (default: all 142k)
    --no-serve       Generate dataset without launching viewer

Requirements:
    - pip install umap-learn
    - Internet access for the UniProt lookup used to name the clusters
      (~22k accessions, a few minutes on the first run, then cached in
      ~/.cache/luxar/protein_embeddings/uniprot_keywords.json)

Controls:
    - Explore clusters of functionally similar proteins
    - Color = landscape cluster, named by its enriched UniProt keyword
    - Ctrl+C to stop
"""

DEMO_META = {
    "key": "protein_landscape",
    "title": "Protein Landscape",
    "description": "142k CAFA5 proteins as a 3D UMAP of ProtT5 embeddings — clusters named by UniProt keyword enrichment.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 540,
        "compute": "heavy",
        "gpu": "none",
        "local_data": "kaggle-auth",
    },
    "caches": ["protein_embeddings"],
    "outputs": ["protein_landscape"],
}

import gzip
import json
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer, require_module
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = None  # Use all 142k proteins by default

#: k-means regions carved out of the 3D landscape. 14 rather than 10 because it
#: is what the naming actually resolves: measured over 5 resamplings of the real
#: dataset, 14 regions yield 12 named / 13 stable, and they separate specific
#: biology (Chloroplast, Mitochondrion, Cell wall, Transducer) that 10 regions
#: blur into generic "Nucleus" / "Cytoplasm" territories (8 named / 7 stable).
N_CLUSTERS = 14

#: Colors for the landscape clusters — a full hue sweep, one per cluster, with
#: alternating lightness so neighbouring hues stay separable in the legend.
CLUSTER_COLORS = np.array(
    [
        [1.00, 0.30, 0.30],  # Red
        [1.00, 0.55, 0.20],  # Orange
        [0.95, 0.80, 0.25],  # Amber
        [0.75, 0.95, 0.30],  # Yellow-green
        [0.40, 0.90, 0.35],  # Green
        [0.25, 0.85, 0.60],  # Emerald
        [0.30, 0.90, 0.90],  # Cyan
        [0.35, 0.65, 1.00],  # Azure
        [0.35, 0.45, 0.95],  # Blue
        [0.60, 0.40, 1.00],  # Violet
        [0.80, 0.35, 0.95],  # Purple
        [0.95, 0.35, 0.75],  # Magenta
        [1.00, 0.45, 0.55],  # Rose
        [0.85, 0.65, 0.50],  # Tan
    ],
    dtype=np.float32,
)

# --- Cluster naming (UniProt keyword enrichment) -----------------------------

UNIPROT_REST = "https://rest.uniprot.org"
#: UniProt asks API clients to identify themselves.
UNIPROT_USER_AGENT = "luxar-demo/1.0 (https://github.com/royerlab/luxar)"

#: Bulk mapping jobs are occasionally refused when several are submitted
#: back to back; retry the batch rather than losing the whole lookup.
UNIPROT_MAX_ATTEMPTS = 4
UNIPROT_BACKOFF_SECONDS = 5.0

#: Proteins sampled per cluster and looked up in UniProt to name the cluster —
#: far cheaper than annotating all 142k, and enough to rank keywords reliably.
#: Not smaller: correlated keywords ("Nucleus" / "Transcription" / "DNA-binding"
#: all describe the same territory) sit close enough in score that a few hundred
#: samples let resampling noise decide between them. Measured over 5 resamplings
#: of the real dataset, 400/cluster keeps only 6 of 14 names identical, while
#: 1600/cluster keeps 13 of 14.
NAMING_SAMPLE_PER_CLUSTER = 1600

#: UniProt keyword categories that say what a protein *is* or *does*. The other
#: categories ("Technical term", "PTM", "Disease", "Coding sequence diversity",
#: ...) describe experimental provenance or modifications and make poor names —
#: "3D-structure" is a statement about the PDB, not about the cluster.
INFORMATIVE_KEYWORD_CATEGORIES = frozenset(
    {
        "Molecular function",
        "Biological process",
        "Cellular component",
        "Ligand",
        "Domain",
    }
)

#: A keyword must appear in at least this fraction of a cluster's sample to be
#: eligible as its name (guards against naming a cluster after a handful of
#: members), AND in at least ``MIN_KEYWORD_COUNT`` of them.
MIN_KEYWORD_FRACTION = 0.05
MIN_KEYWORD_COUNT = 10

#: ...and it must be at least this over-represented versus the pooled sample.
#: Some k-means cells are genuinely unstructured mixtures; naming those after
#: their top keyword at 1.4x lift would be as meaningless as "Cluster 0", so
#: they get ``UNNAMED_CLUSTER_LABEL`` instead.
MIN_KEYWORD_LIFT = 2.0
UNNAMED_CLUSTER_LABEL = "Mixed"


# =============================================================================
# Dataset Download
# =============================================================================


def download_cafa5_dataset(output_dir: Path) -> Path:
    """Download CAFA5 ProtT5 embeddings from Kaggle with robust retry and resume.

    Uses the robust_download utility which provides:
    - Automatic retry on network errors (up to 3 attempts)
    - Resume capability for partial downloads
    - Progress tracking with ETA
    - File size verification

    Args:
        output_dir: Where to save the dataset

    Returns:
        Path to downloaded/extracted dataset directory
    """
    import zipfile

    from luxar.utils.download import robust_download

    dataset_zip = output_dir / "cafa5_prott5.zip"

    with asection("Downloading CAFA5 ProtT5 Embeddings"):
        aprint("Dataset: CAFA5 Protein Function Prediction")
        aprint(
            "URL: https://www.kaggle.com/datasets/horikitasaku/prott5-embedding-for-cafa5"
        )
        aprint("")
        aprint("Download: 540 MB (pre-computed embeddings)")
        aprint("Features: Auto-retry on errors, resume on interruption")
        aprint("")

        if dataset_zip.exists():
            aprint(f"✓ Dataset already downloaded: {dataset_zip}")
        else:
            url = "https://www.kaggle.com/api/v1/datasets/download/horikitasaku/prott5-embedding-for-cafa5"

            try:
                dataset_zip = robust_download(
                    url=url,
                    output_path=dataset_zip,
                    max_retries=3,  # Retry up to 3 times
                    timeout=300,  # 5 minute initial connection timeout
                    chunk_size=1024 * 1024,  # 1MB chunks
                    verify_size=True,  # Verify final size
                )

                aprint("✓ Download complete and verified!")

            except Exception as e:
                aprint(f"❌ Download failed after all retries: {e}")
                if dataset_zip.exists():
                    partial_size = dataset_zip.stat().st_size
                    aprint(
                        f"   Partial download saved: {partial_size / (1024**2):.0f} MB"
                    )
                    aprint("   Run again to resume from this point")
                raise

        # Extract if needed
        extracted_dir = output_dir / "cafa5_data"
        if not extracted_dir.exists():
            aprint("Extracting...")
            with zipfile.ZipFile(dataset_zip, "r") as zf:
                zf.extractall(extracted_dir)
            aprint(f"✓ Extracted to {extracted_dir}")

    return extracted_dir


# =============================================================================
# Data Loading
# =============================================================================


def load_protein_embeddings(
    data_dir: Path,
    sample_size: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load ProtT5 embeddings and their UniProt accessions from the CAFA5 bundle.

    Note the bundle carries no usable functional annotation for these proteins:
    its ``*_train_terms.tsv`` lists PDB-style entries (``3PHF-2``) that do not
    intersect the UniProt accessions in ``train_ids.npy`` at all. Annotation is
    fetched from UniProt at naming time instead (see :func:`name_clusters`).

    Args:
        data_dir: Directory containing extracted CAFA5 data
        sample_size: Optional number of proteins to sample

    Returns:
        Tuple of (embeddings, protein_ids)
    """
    with asection("Loading CAFA5 protein embeddings"):
        # Find the embedding files
        npy_files = list(data_dir.rglob("*.npy"))

        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

        # Find the LARGEST train_embeddings.npy (main dataset, not subsets)
        embedding_candidates = [
            f for f in npy_files if "embeddings" in f.name and "train" in f.name
        ]
        ids_candidates = [f for f in npy_files if "ids" in f.name and "train" in f.name]

        if not embedding_candidates:
            raise FileNotFoundError("Could not find train_embeddings.npy")

        # Use the largest embedding file (full dataset)
        embedding_file = max(embedding_candidates, key=lambda f: f.stat().st_size)
        ids_file = (
            max(ids_candidates, key=lambda f: f.stat().st_size)
            if ids_candidates
            else None
        )

        aprint(
            f"Selected embeddings file: {embedding_file.name} "
            f"({embedding_file.stat().st_size / (1024**2):.1f} MB)"
        )

        aprint(f"Loading embeddings: {embedding_file.name}")
        embeddings = np.load(embedding_file)
        aprint(f"✓ Loaded {len(embeddings):,} embeddings (shape: {embeddings.shape})")

        # Load protein IDs (UniProt accessions — what the cluster naming needs)
        if ids_file:
            aprint(f"Loading protein IDs: {ids_file.name}")
            protein_ids = [str(x) for x in np.load(ids_file, allow_pickle=True)]
            aprint(f"✓ Loaded {len(protein_ids):,} protein IDs")
        else:
            protein_ids = [f"Protein_{i}" for i in range(len(embeddings))]

        # Sample if requested (seeded for reproducibility)
        if sample_size and sample_size < len(embeddings):
            aprint(f"Sampling {sample_size:,} proteins...")
            rng = np.random.default_rng(0)
            indices = rng.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[indices]
            protein_ids = [protein_ids[i] for i in indices]

        aprint(f"✓ Final dataset: {len(embeddings):,} proteins")

    return embeddings, protein_ids


# =============================================================================
# Cluster Naming (UniProt keyword enrichment)
# =============================================================================


def _http_get(url: str, timeout: float = 120.0) -> bytes:
    """GET a URL with a descriptive User-Agent (UniProt asks for one)."""
    request = urllib.request.Request(url, headers={"User-Agent": UNIPROT_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def fetch_keyword_categories(cache_dir: Path) -> dict[str, str]:
    """Fetch the UniProt controlled vocabulary of keywords -> category.

    Used to drop keywords whose category describes provenance rather than
    biology (``3D-structure``, ``Reference proteome``, ...). ~1,200 entries,
    one paginated request, cached on disk forever after.

    Args:
        cache_dir: Directory holding the demo's cached artifacts.

    Returns:
        Mapping of keyword name -> category name.
    """
    cache_path = cache_dir / "uniprot_keyword_categories.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    categories: dict[str, str] = {}
    url: str | None = (
        f"{UNIPROT_REST}/keywords/search"
        "?query=*&format=tsv&fields=id,name,category&size=500"
    )
    while url:
        request = urllib.request.Request(
            url, headers={"User-Agent": UNIPROT_USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            body = response.read().decode()
            link = response.headers.get("Link", "")
        for line in body.splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) >= 3:
                categories[fields[1]] = fields[2]
        url = link.split(">")[0][1:] if 'rel="next"' in link else None

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(categories))
    return categories


def _uniprot_keyword_batch(batch: Sequence[str]) -> dict[str, list[str]]:
    """Resolve one batch of accessions through the UniProt ID-mapping service.

    Bulk mapping means a few thousand proteins cost one round trip rather than
    one request each: submit the accessions, poll until the job leaves the
    queue, then stream a gzipped TSV of results.

    Args:
        batch: UniProt accessions to resolve.

    Returns:
        Mapping of accession -> keyword names, with an empty list for the
        accessions UniProt did not resolve (obsolete, demerged, ...).
    """
    payload = urllib.parse.urlencode(
        {"from": "UniProtKB_AC-ID", "to": "UniProtKB", "ids": ",".join(batch)}
    ).encode()
    request = urllib.request.Request(
        f"{UNIPROT_REST}/idmapping/run",
        data=payload,
        headers={"User-Agent": UNIPROT_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        job_id = json.load(response)["jobId"]

    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        status = json.loads(_http_get(f"{UNIPROT_REST}/idmapping/status/{job_id}"))
        if status.get("jobStatus") not in ("RUNNING", "NEW"):
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"UniProt ID-mapping job {job_id} did not finish")

    raw = _http_get(
        f"{UNIPROT_REST}/idmapping/uniprotkb/results/stream/{job_id}"
        "?format=tsv&fields=accession,keyword&compressed=true",
        timeout=600,
    )
    lines = gzip.decompress(raw).decode().splitlines()
    header = lines[0].split("\t")
    keyword_column = header.index("Keywords")

    # Default every requested accession to "no keywords" so unmapped ones are
    # remembered rather than retried on every run.
    resolved: dict[str, list[str]] = {accession: [] for accession in batch}
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) > keyword_column:
            resolved[fields[0]] = [k for k in fields[keyword_column].split(";") if k]
    return resolved


def fetch_uniprot_keywords(
    accessions: Sequence[str],
    cache_dir: Path,
    batch_size: int = 5000,
) -> dict[str, list[str]]:
    """Look up UniProt keywords for a list of accessions, with an on-disk cache.

    Batches are retried with exponential backoff and each success is persisted
    before the next batch starts, so a rate-limited or dropped request costs
    only that batch. A batch that fails every attempt is *skipped* rather than
    raised: naming reads whatever resolved, and 18k of 22k annotated proteins
    name the landscape just as well as all of them (see :func:`name_clusters`,
    which drops unresolved accessions from the sample rather than counting them
    as unannotated).

    Args:
        accessions: UniProt accessions to look up.
        cache_dir: Directory holding the demo's cached artifacts.
        batch_size: Accessions per ID-mapping job.

    Returns:
        Mapping of accession -> list of keyword names (possibly empty). Keys are
        the accessions that resolved, which may be a subset of ``accessions``.
    """
    cache_path = cache_dir / "uniprot_keywords.json"
    keywords: dict[str, list[str]] = (
        json.loads(cache_path.read_text()) if cache_path.exists() else {}
    )

    missing = sorted({a for a in accessions if a not in keywords})
    if not missing:
        return keywords

    aprint(f"Looking up {len(missing):,} accessions in UniProt...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed = 0

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        for attempt in range(UNIPROT_MAX_ATTEMPTS):
            try:
                keywords.update(_uniprot_keyword_batch(batch))
                cache_path.write_text(json.dumps(keywords))
                aprint(
                    f"  {min(start + len(batch), len(missing)):,}/{len(missing):,} "
                    "resolved"
                )
                break
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                if attempt == UNIPROT_MAX_ATTEMPTS - 1:
                    failed += len(batch)
                    aprint(f"  ⚠️  batch of {len(batch):,} failed ({exc}) — skipping")
                    break
                backoff = UNIPROT_BACKOFF_SECONDS * 2**attempt
                aprint(f"  ⚠️  {exc} — retrying in {backoff:.0f}s")
                time.sleep(backoff)

    if failed:
        aprint(f"⚠️  {failed:,} accessions unresolved; naming uses the rest")

    return keywords


def enriched_cluster_names(
    cluster_samples: Mapping[int, Sequence[str]],
    keywords: Mapping[str, Sequence[str]],
    categories: Mapping[str, str],
) -> dict[int, tuple[str, float]]:
    """Name each cluster after the UniProt keyword most enriched within it.

    The score is ``p_cluster * log2(p_cluster / p_pooled)`` — lift weighted by
    support, so a keyword that is 10x enriched but present in 2% of the cluster
    loses to one that is 5x enriched and present in half of it. Names are unique:
    each keyword is claimed by the single cluster that scores highest for it, and
    a cluster left without an eligible keyword is reported as
    ``UNNAMED_CLUSTER_LABEL`` rather than given a meaningless one.

    Args:
        cluster_samples: cluster id -> sampled accessions from that cluster.
        keywords: accession -> UniProt keyword names.
        categories: keyword name -> UniProt keyword category.

    Returns:
        cluster id -> (name, lift). Lift is 0.0 for unnamed clusters.
    """
    counts: dict[int, Counter] = {}
    sizes: dict[int, int] = {}
    pooled: Counter = Counter()
    pooled_size = 0

    for cluster, accessions in cluster_samples.items():
        counter: Counter = Counter()
        for accession in accessions:
            informative = {
                k
                for k in keywords.get(accession, ())
                if categories.get(k) in INFORMATIVE_KEYWORD_CATEGORIES
            }
            counter.update(informative)
        counts[cluster] = counter
        sizes[cluster] = len(accessions)
        pooled.update(counter)
        pooled_size += len(accessions)

    if pooled_size == 0:
        return {c: (UNNAMED_CLUSTER_LABEL, 0.0) for c in cluster_samples}

    candidates: list[tuple[float, float, int, str]] = []
    for cluster, counter in counts.items():
        size = sizes[cluster]
        if size == 0:
            continue
        floor = max(MIN_KEYWORD_COUNT, MIN_KEYWORD_FRACTION * size)
        for keyword, count in counter.items():
            if count < floor:
                continue
            p_cluster = count / size
            p_pooled = pooled[keyword] / pooled_size
            lift = p_cluster / p_pooled
            if lift < MIN_KEYWORD_LIFT:
                continue
            score = p_cluster * float(np.log2(lift))
            candidates.append((score, lift, cluster, keyword))

    # Greedy highest-score-first assignment gives each cluster its best keyword
    # while keeping the legend free of duplicate names.
    candidates.sort(key=lambda row: (-row[0], row[3], row[2]))
    named: dict[int, tuple[str, float]] = {}
    claimed: set[str] = set()
    for _score, lift, cluster, keyword in candidates:
        if cluster in named or keyword in claimed:
            continue
        named[cluster] = (keyword, lift)
        claimed.add(keyword)

    for cluster in cluster_samples:
        named.setdefault(cluster, (UNNAMED_CLUSTER_LABEL, 0.0))
    return named


def name_clusters(
    cluster_ids: np.ndarray,
    protein_ids: Sequence[str],
    cache_dir: Path,
) -> list[str]:
    """Derive a human-readable name for every landscape cluster.

    Samples members of each cluster, looks their keywords up in UniProt, and
    picks the most over-represented one (see :func:`enriched_cluster_names`).
    Falls back to plain ``Cluster N`` labels only when *nothing* resolves — the
    demo stays runnable offline, it just cannot name anything. A partial lookup
    is used as-is on the accessions that did resolve.

    Args:
        cluster_ids: Per-protein cluster assignment.
        protein_ids: Per-protein UniProt accession, aligned with ``cluster_ids``.
        cache_dir: Directory holding the demo's cached artifacts.

    Returns:
        One name per cluster, indexed by cluster id.
    """
    n_clusters = int(cluster_ids.max()) + 1 if len(cluster_ids) else 0
    fallback = [f"Cluster {c}" for c in range(n_clusters)]

    with asection("Naming clusters from UniProt keyword enrichment"):
        rng = np.random.default_rng(0)
        cluster_samples: dict[int, list[str]] = {}
        for cluster in range(n_clusters):
            members = np.flatnonzero(cluster_ids == cluster)
            if len(members) == 0:
                cluster_samples[cluster] = []
                continue
            picked = rng.choice(
                members,
                min(NAMING_SAMPLE_PER_CLUSTER, len(members)),
                replace=False,
            )
            cluster_samples[cluster] = [protein_ids[i] for i in picked]

        wanted = [a for sample in cluster_samples.values() for a in sample]
        try:
            categories = fetch_keyword_categories(cache_dir)
            keywords = fetch_uniprot_keywords(wanted, cache_dir)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            aprint(f"⚠️  UniProt lookup failed ({exc}) — using generic cluster labels")
            return fallback

        # Keep only accessions UniProt actually answered for. An accession that
        # resolved with no keywords is real evidence and stays; one that never
        # resolved is absence of evidence and would otherwise dilute the
        # frequencies of every cluster it landed in.
        resolved_samples = {
            cluster: [a for a in sample if a in keywords]
            for cluster, sample in cluster_samples.items()
        }
        resolved = sum(len(sample) for sample in resolved_samples.values())
        if resolved == 0:
            aprint("⚠️  No accessions resolved — using generic cluster labels")
            return fallback

        annotated = sum(1 for a in wanted if keywords.get(a))
        aprint(
            f"✓ {resolved:,}/{len(wanted):,} sampled proteins resolved, "
            f"{annotated:,} of them carrying UniProt keywords"
        )

        named = enriched_cluster_names(resolved_samples, keywords, categories)
        names = []
        for cluster in range(n_clusters):
            name, lift = named.get(cluster, (UNNAMED_CLUSTER_LABEL, 0.0))
            names.append(name)
            suffix = f" ({lift:.1f}x enriched)" if lift else " (no dominant keyword)"
            aprint(f"  Cluster {cluster}: {name}{suffix}")

    return names


# =============================================================================
# UMAP Dimensionality Reduction
# =============================================================================


def load_cached_umap(
    cache_path: Path,
    legacy_ids_path: Path | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    """Load cached UMAP coordinates and their protein accessions, if usable.

    Older caches stored per-point *function* labels instead of accessions.
    Cluster naming needs accessions, so such a cache is upgraded in place when
    the source ``train_ids.npy`` still lines up with it (which it does for an
    unsampled run, where the point order is the file order) and discarded
    otherwise — recomputing a 142k UMAP is expensive enough to be worth the
    one-off repair.

    Args:
        cache_path: ``.npz`` written by :func:`reduce_embeddings_umap`.
        legacy_ids_path: ``train_ids.npy`` to recover accessions from, for a
            cache written before they were stored. ``None`` disables recovery.

    Returns:
        ``(positions, protein_ids)``, or ``None`` if the cache is unusable.
    """
    if not cache_path.exists():
        return None

    with asection("Loading cached UMAP coordinates"):
        aprint(f"Cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        positions = cached["positions"]

        if "protein_ids" in cached.files:
            protein_ids = [str(x) for x in cached["protein_ids"]]
            aprint(f"✓ Loaded {len(positions):,} proteins from cache (INSTANT!)")
            return positions, protein_ids

        aprint("Cache predates stored accessions — attempting in-place upgrade")
        if legacy_ids_path is None or not legacy_ids_path.exists():
            aprint("⚠️  No source accessions available — recomputing UMAP")
            return None
        recovered = [str(x) for x in np.load(legacy_ids_path, allow_pickle=True)]
        if len(recovered) != len(positions):
            aprint(
                f"⚠️  {len(recovered):,} accessions vs {len(positions):,} cached "
                "points — recomputing UMAP"
            )
            return None

        np.savez(
            cache_path,
            positions=positions,
            protein_ids=np.array(recovered, dtype=object),
        )
        aprint(f"✓ Upgraded cache with {len(recovered):,} accessions")
        return positions, recovered


def reduce_embeddings_umap(
    embeddings: np.ndarray,
    protein_ids: list[str],
    cache_path: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Reduce protein embeddings to 3D using UMAP.

    Args:
        embeddings: (n_proteins, n_features) array
        protein_ids: UniProt accession per protein, aligned with ``embeddings``
        cache_path: Optional path to cache UMAP results

    Returns:
        Tuple of (positions, protein_ids) - both cached together, so a cached
        run can still name its clusters without re-reading the 540 MB bundle.
    """

    # Gated here, not in main(): the cache hit above returns without UMAP.
    UMAP = require_module("umap").UMAP

    with asection(f"Reducing {embeddings.shape[1]}D → 3D with UMAP"):
        aprint(
            f"Input: {embeddings.shape[0]:,} proteins × {embeddings.shape[1]} dimensions"
        )
        aprint("Parameters: n_neighbors=15, metric=cosine, all CPU cores")

        reducer = UMAP(
            n_components=3,
            n_neighbors=15,
            metric="cosine",
            n_jobs=-1,  # All cores
            low_memory=False,  # Speed optimization
            verbose=True,
        )

        reduced = reducer.fit_transform(embeddings)

        aprint("✓ UMAP complete")
        aprint(f"  Output: {reduced.shape}")
        aprint(f"  Range: [{reduced.min():.2f}, {reduced.max():.2f}]")

        # Center at barycenter
        centroid = reduced.mean(axis=0)
        reduced = reduced - centroid
        aprint("✓ Centered at barycenter")
        aprint(f"  New range: [{reduced.min():.2f}, {reduced.max():.2f}]")

        # Cache results (positions AND accessions together!)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                positions=reduced,
                protein_ids=np.array(protein_ids, dtype=object),
            )
            aprint(f"✓ Cached UMAP + accessions to {cache_path}")
            aprint("  Future runs will load instantly!")

    return reduced.astype(np.float32), protein_ids


# =============================================================================
# Visualization Generation
# =============================================================================


def generate_protein_landscape(
    output_path: Path,
    sample_size: int | None = None,
) -> int:
    """Generate 3D protein function landscape.

    Args:
        output_path: Where to write zarr
        sample_size: Optional number of proteins to sample

    Returns:
        Number of proteins visualized
    """
    # Setup caching
    cache_dir = Path.home() / ".cache" / "luxar" / "protein_embeddings"
    dataset_cache = cache_dir / "cafa5_data"
    umap_cache = cache_dir / f"umap_{sample_size or 'all'}.npz"

    # Download dataset if needed
    if not dataset_cache.exists():
        dataset_cache = download_cafa5_dataset(cache_dir)

    # Check UMAP cache first (complete early exit if cached!). An unsampled run
    # keeps the source file order, so a pre-accession cache can be repaired from
    # train_ids.npy instead of recomputing.
    legacy_ids = dataset_cache / "train_ids.npy" if sample_size is None else None
    cached = load_cached_umap(umap_cache, legacy_ids_path=legacy_ids)
    if cached is not None:
        positions, protein_ids = cached
    else:
        # Load embeddings (only if UMAP not cached)
        embeddings, protein_ids = load_protein_embeddings(
            dataset_cache,
            sample_size=sample_size,
        )

        if len(embeddings) == 0:
            aprint("❌ No proteins loaded")
            return 0

        # Reduce to 3D with UMAP and cache
        positions, protein_ids = reduce_embeddings_umap(
            embeddings, protein_ids, cache_path=umap_cache
        )

    # Generate visualization
    with asection("Generating visualization"):
        n_proteins = len(positions)

        # Carve the 3D landscape into regions. k-means on the UMAP coordinates
        # (not the raw embeddings) means the regions match the blobs you can
        # actually see, and it works on the cached path where the 1,024D
        # embeddings are no longer in hand.
        n_clusters = min(N_CLUSTERS, n_proteins)
        if n_clusters >= 2:
            from sklearn.cluster import KMeans

            cluster_ids = KMeans(
                n_clusters=n_clusters, random_state=0, n_init=10
            ).fit_predict(positions)
        else:
            cluster_ids = np.zeros(n_proteins, dtype=int)

        cluster_colors = CLUSTER_COLORS[cluster_ids % len(CLUSTER_COLORS)]

        # Name each region after the UniProt keyword most enriched inside it.
        cluster_names = name_clusters(cluster_ids, protein_ids, cache_dir)
        cluster_sizes = Counter(int(c) for c in cluster_ids)

        # Disambiguate: two clusters can both end up "Mixed", and the legend
        # needs one row per cluster.
        display_names: list[str] = []
        name_uses = Counter(cluster_names)
        seen: Counter = Counter()
        for cluster, name in enumerate(cluster_names):
            if name_uses[name] > 1:
                seen[name] += 1
                display_names.append(f"{name} {seen[name]}")
            else:
                display_names.append(name)

        aprint("✓ Proteins by landscape cluster:")
        for cluster in range(n_clusters):
            aprint(f"  {display_names[cluster]}: {cluster_sizes[cluster]:,}")

        radii = np.full(n_proteins, 0.02, dtype=np.float32)

        # Hover shows the actual protein, plus the region it landed in.
        labels = [
            f"{protein_ids[i]} · {display_names[int(cluster_ids[i])]}"
            for i in range(n_proteins)
        ]

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "proteins",
                positions=np.asarray(positions, dtype=np.float32),
                colors=cluster_colors,
                radii=radii,
                sharpness=np.full(n_proteins, 0.55, dtype=np.float32),
                opacity=0.9,
                intensity=0.124,
                labels=labels,
            )

            # --- Overlays ---
            scene.add_text(
                "Protein Function Landscape (CAFA5)",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Legend: one row per cluster, using the name derived from UniProt
            # keyword enrichment and the cluster's real colour.
            def _rgb_to_hex(rgb: np.ndarray) -> str:
                r, g, b = (int(round(float(c) * 255)) for c in rgb[:3])
                return f"#{r:02x}{g:02x}{b:02x}"

            legend_html = (
                '<div style="font-size:1.25vh;line-height:1.45;'
                'background:rgba(0,0,0,0.5);padding:0.5vh 0.7vh;border-radius:3px">'
                '<div style="font-weight:bold;color:#ccc;margin-bottom:0.3vh">'
                "Enriched UniProt keyword</div>"
            )
            for cluster in sorted(range(n_clusters), key=lambda c: -cluster_sizes[c]):
                color = _rgb_to_hex(CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)])
                legend_html += (
                    f'<div><span style="color:{color}">\u2588</span> '
                    f"{display_names[cluster]} "
                    f'<span style="color:#888">({cluster_sizes[cluster]:,})</span></div>'
                )
            legend_html += "</div>"

            scene.add_html(
                legend_html,
                # Clear of the viewer's left icon rail: 14 rows reach far enough
                # up the screen to collide with it at x=0.02, which the previous
                # 10-row legend did not.
                position=(0.045, 0.97),
                anchor="bottom-left",
            )

            scene.add_text(
                f"{n_proteins:,} proteins • ProtT5 embeddings • 3D UMAP • "
                "clusters named by UniProt keyword enrichment • Elnaggar et al. 2021",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Visualization created with {n_proteins:,} proteins")

    return n_proteins


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    sample_size = DEFAULT_SAMPLE_SIZE

    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("PROTEIN FUNCTION LANDSCAPE - PROTT5 EMBEDDINGS")
    aprint("=" * 70)
    aprint("")
    aprint("Dataset: CAFA5 Protein Function Prediction")
    aprint("https://www.kaggle.com/datasets/horikitasaku/prott5-embedding-for-cafa5")
    aprint("")
    aprint("What this shows:")
    aprint("  • 142k proteins with ProtT5-XL embeddings (1,024D)")
    aprint("  • 3D UMAP projection showing functional relationships")
    aprint("  • Proteins with similar functions cluster together")
    aprint("  • Color = landscape cluster")
    aprint("  • Cluster names = most over-represented UniProt keyword")
    aprint("")
    aprint("Parameters:")
    aprint(
        f"  Proteins: {sample_size:,}" if sample_size else "  Proteins: All (142,246)"
    )
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "protein_landscape.luxar.zarr"
        try:
            n_proteins = generate_protein_landscape(
                output_path,
                sample_size=sample_size,
            )
            if n_proteins == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total proteins: {n_proteins:,}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_proteins_") as tmpdir:
        output_path = Path(tmpdir) / "protein_landscape.luxar.zarr"

        try:
            n_proteins = generate_protein_landscape(
                output_path,
                sample_size=sample_size,
            )

            if n_proteins == 0:
                return

        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS - EXPLORE THE PROTEOME")
        aprint("=" * 70)
        aprint("")
        aprint("What to look for:")
        aprint("  - Named territories in the legend (bottom-left)")
        aprint("  - Secreted / signal-peptide proteins vs cytoplasmic ones")
        aprint("  - Organelle-targeted proteins as their own region")
        aprint("  - Receptors / GPCRs as a distinct island")
        aprint("  - Functional boundaries and overlaps between regions")
        aprint("")
        aprint("Try this:")
        aprint("  1. Zoom out: See overall functional organization")
        aprint("  2. Zoom in: Explore specific protein families")
        aprint("  3. Hover a point: UniProt accession + its cluster")
        aprint("")
        aprint(f"Total proteins: {n_proteins:,}")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Explore the landscape of protein function!")
        aprint("Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
