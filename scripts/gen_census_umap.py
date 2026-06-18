"""Generate a large CZ CELLxGENE Census scVI-UMAP coordinate cache (GPU).

Fetches ``--n`` primary human cells' precomputed **scVI latent** (50-d) plus
categorical obs metadata from the public Census, runs **cuML UMAP** -> 3D, and
writes a compact NPZ that ``demos/demo_cellxgene_census_umap.py`` loads:
``coords`` (N,3) float32 + per-cell int codes for cell_type / tissue_general /
disease / assay + a ``labels_json`` category-label map.

Requires ``cellxgene-census`` + ``cuml`` (RAPIDS) on a CUDA GPU, e.g.::

    conda create -n umap10m python=3.11
    pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12 cellxgene-census tiledbsoma
    python scripts/gen_census_umap.py --n 10000000 --out census_umap_10m.npz

Sampling uses stratified-contiguous blocks (a random scatter of soma_joinids is
~20x slower to read from TileDB): ``--n-blocks`` contiguous runs evenly spread
across the sorted primary-cell id range — representative *and* local.
~96.6M primary human cells are available in the 2025-11-08 Census.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

CATS = ["cell_type", "tissue_general", "disease", "assay"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--version", default="2025-11-08")
    ap.add_argument("--out", default="census_umap.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--n-blocks", type=int, default=2000)
    a = ap.parse_args()
    v = a.version

    import cellxgene_census as cc
    import cellxgene_census.experimental as exp
    from cellxgene_census._get_anndata import (
        CENSUS_EMBEDDINGS_LOCATION_BASE_URI as BASE,
    )
    from cellxgene_census._open import _uri_join

    meta = exp.get_embedding_metadata_by_name("scvi", "homo_sapiens", v)
    uri = _uri_join(BASE, f"{v}/{meta['id']}")
    census = cc.open_soma(census_version=v)
    ctx = census.context
    hs = census["census_data"]["homo_sapiens"]

    t = time.time()
    allids = np.sort(
        hs.obs.read(
            value_filter="is_primary_data == True", column_names=["soma_joinid"]
        )
        .concat()
        .to_pandas()["soma_joinid"]
        .to_numpy()
    )
    print(f"primary cells: {len(allids):,} ({time.time() - t:.1f}s)", flush=True)

    n = min(a.n, len(allids))
    nb = min(a.n_blocks, max(1, n // 1000))
    bs = n // nb
    step = len(allids) // nb
    sel = np.concatenate([allids[i * step : i * step + bs] for i in range(nb)])
    if len(sel) < n:  # top up exactly to n from the (contiguous) tail
        sel = np.concatenate([sel, allids[-(n - len(sel)) :]])
    ids = np.sort(np.unique(sel)).astype(np.int64)[:n]
    n = len(ids)
    print(f"sampled {n:,} cells in {nb} blocks of ~{bs}", flush=True)

    t = time.time()
    obs = (
        hs.obs.read(coords=(ids,), column_names=["soma_joinid"] + CATS)
        .concat()
        .to_pandas()
        .set_index("soma_joinid")
        .loc[ids]
    )
    print(f"obs cats {obs.shape} ({time.time() - t:.1f}s)", flush=True)

    t = time.time()
    emb = exp.get_embedding(v, uri, ids, context=ctx).astype(np.float32)
    census.close()
    print(f"scvi {emb.shape} ({time.time() - t:.1f}s)", flush=True)

    from cuml.manifold import UMAP

    t = time.time()
    coords = np.asarray(
        UMAP(
            n_components=3,
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            random_state=a.seed,
        ).fit_transform(emb)
    )
    print(
        f"UMAP -> {coords.shape} ({time.time() - t:.1f}s) "
        f"finite={bool(np.isfinite(coords).all())}",
        flush=True,
    )

    save = {"coords": coords.astype(np.float32), "soma_joinid": ids}
    labels = {}
    for c in CATS:
        cat = obs[c].astype("category")
        save[f"{c}_code"] = cat.cat.codes.to_numpy().astype(np.int32)
        labels[c] = list(map(str, cat.cat.categories))
    save["labels_json"] = np.array(json.dumps(labels))
    np.savez_compressed(a.out, **save)
    print("WROTE", a.out, "ncats:", {c: len(labels[c]) for c in CATS}, flush=True)


if __name__ == "__main__":
    main()
