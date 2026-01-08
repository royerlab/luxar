"""Allow running luxar as a module: python -m luxar

This enables running the CLI via `python -m luxar` in addition to the
installed `luxar` command. This is particularly useful for demos that
need to launch the viewer using sys.executable.
"""

from luxar.cli import app

if __name__ == "__main__":
    app()
