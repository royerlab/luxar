"""
luxar.cli – Command-line interface for building, serving, and inspecting Luxar Zarr scenes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Generator, Tuple

import typer
import uvicorn
import zarr
from arbol import aprint
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from luxar.scene import Scene

app = typer.Typer(help="luxar – build and serve Zarr-backed 3-D scenes")


# ────────────────────────────── random ───────────────────────────────────────
@app.command()
def random(
    out: Path = typer.Option(..., "--out", "-o", help="Directory to write Zarr"),
    n: int = typer.Option(100_000, "--n", "-n", help="Number of points"),
    seed: int | None = typer.Option(None, help="Random seed"),
) -> None:
    """
    Generate a random point cloud scene and write it to a Zarr store.

    Args:
        out (Path): Output directory for Zarr store.
        n (int, optional): Number of points. Defaults to 100,000.
        seed (int, optional): Random seed. Defaults to None.
    """
    try:
        aprint(f"Generating random scene with {n} points at {out}.")
        Scene.random_demo(out, n=n, seed=seed)
        aprint(f"✔ wrote {n:,} points → {out}")
        typer.secho(f"✔ wrote {n:,} points → {out}", fg=typer.colors.GREEN, bold=True)
    except Exception as e:
        aprint(f"Error generating random scene: {e}")
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ────────────────────────────── serve ────────────────────────────────────────
@app.command()
def serve(
    store: Path = typer.Argument(..., exists=True, readable=True),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host address to bind to (use 0.0.0.0 for all interfaces)",
    ),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """
    Serve a Zarr scene using FastAPI and Uvicorn.

    Args:
        store (Path): Path to the Zarr store to serve.
        host (str, optional): Host address. Defaults to "127.0.0.1".
        port (int, optional): Port number. Defaults to 8000.
    """
    try:
        if store.suffix == ".zip":
            aprint("ZipStore not yet supported.")
            typer.secho("ZipStore not yet supported.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        aprint(f"Serving {store} at http://{host}:{port}/data/{store.name}/")
        api = FastAPI(title="Luxar static server", docs_url=None, redoc_url=None)

        # Add CORS middleware to allow requests from the viewer
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        api.mount("/data", StaticFiles(directory=store.parent, html=False))
        typer.secho(
            f"🛰️  Serving {store} at http://{host}:{port}/data/{store.name}/",
            fg=typer.colors.CYAN,
            bold=True,
        )
        uvicorn.run(api, host=host, port=port, reload=False, log_level="info")
    except Exception as e:
        aprint(f"Error serving Zarr store: {e}")
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ────────────────────────────── info ─────────────────────────────────────────
@app.command()
def info(path: Path) -> None:
    """
    Show root attributes, group hierarchy, and point-cloud stats for a Zarr scene.

    Args:
        path (Path): Path to the Zarr store.
    """
    try:
        if not path.exists():
            aprint("Path does not exist.")
            typer.secho("Path does not exist.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        root = zarr.open_group(path, mode="r")
        hdr: Callable[[str], None] = lambda s: typer.secho(
            s, bold=True, fg=typer.colors.BLUE
        )
        hdr("Root attributes")
        for k, v in root.attrs.items():
            typer.echo(f"  {k}: {v}")
        hdr("\nHierarchy")
        point_datasets = []
        for depth, grp in _dfs(root):
            indent = "   " * depth
            typer.echo(f"{indent}└─ {grp.basename or '/'}  attrs={dict(grp.attrs)}")
            if "positions" in grp:
                point_datasets.append(grp["positions"])
        n_clouds = len(point_datasets)
        n_points = sum(d.shape[0] for d in point_datasets)
        hdr("\nSummary")
        typer.echo(f"  point clouds : {n_clouds}")
        typer.echo(f"  total points : {n_points:,}")
        aprint(f"Info for {path}: {n_clouds} clouds, {n_points} points.")
    except Exception as e:
        aprint(f"Error reading info for {path}: {e}")
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


def _dfs(
    group: zarr.Group, depth: int = 0
) -> Generator[Tuple[int, zarr.Group], None, None]:
    """
    Depth-first walk that yields (depth, group) for every subgroup.

    Args:
        group (zarr.Group): Zarr group to traverse.
        depth (int, optional): Current depth. Defaults to 0.
    Yields:
        Tuple[int, zarr.Group]: (depth, group) for each subgroup.
    """
    try:
        yield depth, group
        for name in group.group_keys():
            yield from _dfs(group[name], depth + 1)
    except Exception as e:
        aprint(f"Error traversing Zarr group hierarchy: {e}")
        raise


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
