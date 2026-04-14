#!/usr/bin/env python3
"""Build the merging error analysis supplementary document.

Compiles the standalone LaTeX document (pdflatex + bibtex).

Usage::

    hatch run python manuscript/supp_doc/merging_error_analysis/build.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).parent
TEX_FILE = HERE / "merging_error_analysis.tex"


def main():
    print(f"Building: {TEX_FILE.stem}")

    if not TEX_FILE.exists():
        print(f"  ERROR: {TEX_FILE} not found")
        return

    print(f"  Compiling {TEX_FILE.name}...")
    for i, cmd in enumerate([
        ["pdflatex", "-interaction=nonstopmode", TEX_FILE.name],
        ["bibtex", TEX_FILE.stem],
        ["pdflatex", "-interaction=nonstopmode", TEX_FILE.name],
        ["pdflatex", "-interaction=nonstopmode", TEX_FILE.name],
    ]):
        result = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if result.returncode != 0 and i != 1:
            print(f"  WARNING: {cmd[0]} returned {result.returncode}")

    out_pdf = HERE / f"{TEX_FILE.stem}.pdf"
    if out_pdf.exists():
        print(f"  Built: {out_pdf.name} ({out_pdf.stat().st_size // 1024} KB)")
    else:
        print(f"  ERROR: {out_pdf.name} not produced")


if __name__ == "__main__":
    main()
