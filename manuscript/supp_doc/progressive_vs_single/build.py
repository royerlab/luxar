#!/usr/bin/env python3
"""Build the progressive vs single-pass supplementary document.

Concatenates per-dataset comparison and slice montage PDFs into a single document.

Usage::

    hatch run python manuscript/supp_doc/progressive_vs_single/build.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ANALYSIS_RESULTS = HERE.parent.parent / "analysis" / "progressive_vs_single" / "results"
ANALYSIS_DIR = HERE.parent.parent / "analysis" / "progressive_vs_single"

DATASETS = [
    "opencell_map4_ch0",
    "opencell_map4_ch1",
    "opencell_lmnb1_ch0",
    "opencell_lmnb1_ch1",
    "kidney_dapi",
    "kidney_actin",
    "cells3d_nuclei",
    "cells3d_membrane",
    "organoid_ch0",
    "celegans_t100",
    "tribolium",
    "acto3d_heart_nuclei",
]


def merge_pdfs_gs(pdf_paths: list[Path], output: Path) -> bool:
    """Merge PDFs using ghostscript."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
        f"-sOutputFile={output}",
    ] + [str(p) for p in pdf_paths]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ERROR: ghostscript (gs) not available.")
        return False


def main():
    print("Building: progressive_vs_single_all.pdf")

    pdfs = []
    for ds in DATASETS:
        for name in ["fig_progressive_comparison.pdf", "fig_progressive_slices.pdf"]:
            p = ANALYSIS_RESULTS / ds / name
            if p.exists():
                pdfs.append(p)

    if not pdfs:
        print("  No PDFs found. Run the progressive analysis first.")
        return

    out = HERE / "progressive_vs_single_all.pdf"
    print(f"  Merging {len(pdfs)} PDFs...")
    if merge_pdfs_gs(pdfs, out):
        print(f"  Built: {out.name} ({out.stat().st_size // 1024} KB)")

    # Copy interpretation
    interp = ANALYSIS_DIR / "INTERPRETATION.md"
    if interp.exists():
        shutil.copy2(interp, HERE / "INTERPRETATION.md")


if __name__ == "__main__":
    main()
