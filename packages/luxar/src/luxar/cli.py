"""luxar.cli – Command-line interface for building, serving, and inspecting Luxar Zarr scenes."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Tuple

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
    """Generate a random point cloud scene and write it to a Zarr store.

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
    path: Path = typer.Argument(..., exists=True, readable=True),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host address to bind to (use 0.0.0.0 for all interfaces)",
    ),
    port: int = typer.Option(8000, "--port", "-p"),
) -> None:
    """Serve a directory or Zarr dataset via HTTP with directory listing support.

    Args:
        path (Path): Path to directory or Zarr dataset to serve.
        host (str, optional): Host address. Defaults to "127.0.0.1".
        port (int, optional): Port number. Defaults to 8000.
    """
    try:
        # Determine what we're serving
        if path.is_dir():
            serve_path = path
            if path.name.endswith('.zarr'):
                aprint(f"Serving Zarr dataset: {path}")
            else:
                aprint(f"Serving directory: {path}")
        else:
            aprint(f"Error: {path} is not a directory")
            raise typer.Exit(1)


        from starlette.responses import (
            FileResponse,
            HTMLResponse,
            JSONResponse,
        )

        # Create custom static files handler with directory listing
        class DirectoryListingStaticFiles(StaticFiles):
            """Static files handler with JSON directory listing support."""

            async def get_response(self, path: str, scope):
                """Override to provide directory listing."""
                # Handle OPTIONS requests for CORS
                if scope.get("method") == "OPTIONS":
                    from starlette.responses import Response
                    return Response(
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                            "Access-Control-Allow-Headers": "*",
                        }
                    )

                full_path = self.directory / path if path else self.directory

                # Handle .zgroup files for zarr directories
                if path.endswith(".zgroup") and full_path.exists() and full_path.is_file():
                    return FileResponse(full_path)

                # If it's a directory, provide listing
                if full_path.exists() and full_path.is_dir():
                    # Check Accept header
                    headers = dict(scope.get("headers", []))
                    accept = headers.get(b"accept", b"").decode("utf-8")

                    # Generate directory listing
                    entries = []
                    try:
                        for item in sorted(full_path.iterdir()):
                            # Skip hidden files except .zgroup
                            if item.name.startswith('.') and item.name != '.zgroup':
                                continue

                            item_type = "directory" if item.is_dir() else "file"
                            # Check if it's a zarr directory
                            if item.is_dir() and item.name.endswith('.zarr'):
                                item_type = "zarr"
                            elif item.is_dir() and (item / ".zgroup").exists():
                                item_type = "zarr"

                            entries.append({
                                "name": item.name,
                                "type": item_type,
                                "size": item.stat().st_size if item.is_file() else None
                            })
                    except PermissionError:
                        return Response("Permission denied", status_code=403)

                    # Return JSON for API requests
                    if "application/json" in accept:
                        return JSONResponse({"entries": entries})

                    # Return HTML for browser requests
                    html_content = "<html><body><h1>Directory Listing</h1><ul>"
                    if path:
                        html_content += '<li><a href="../">../</a></li>'
                    for entry in entries:
                        name = entry["name"]
                        if entry["type"] in ("directory", "zarr"):
                            name += "/"
                        html_content += f'<li><a href="{name}">{name}</a></li>'
                    html_content += "</ul></body></html>"
                    return HTMLResponse(content=html_content)

                # Fall back to default static file serving
                return await super().get_response(path, scope)

        api = FastAPI(title="Luxar static server", docs_url=None, redoc_url=None)

        # Add CORS middleware to allow requests from the viewer
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Mount the static files handler with directory listing
        api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

        typer.secho(
            f"🛰️  Serving {serve_path} at http://{host}:{port}/",
            fg=typer.colors.CYAN,
            bold=True,
        )
        typer.secho(
            f"📊 Viewer URL: http://localhost:5173/?src=http://{host}:{port}/",
            fg=typer.colors.GREEN,
        )

        uvicorn.run(api, host=host, port=port, reload=False, log_level="info")
    except Exception as e:
        aprint(f"Error serving path: {e}")
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# ────────────────────────────── info ─────────────────────────────────────────
@app.command()
def info(path: Path) -> None:
    """Show root attributes, group hierarchy, and point-cloud stats for a Zarr scene.

    Args:
        path (Path): Path to the Zarr store.
    """
    try:
        if not path.exists():
            aprint("Path does not exist.")
            typer.secho("Path does not exist.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1)
        root = zarr.open_group(path, mode="r")

        def hdr(s: str) -> None:
            typer.secho(s, bold=True, fg=typer.colors.BLUE)

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
    """Depth-first walk that yields (depth, group) for every subgroup.

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
