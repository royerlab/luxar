#!/usr/bin/env python3
"""Build the splat_count_vs_quality supplementary document.

Copies per-dataset figures from the analysis results into figures/,
then compiles the LaTeX document.

Usage::

    hatch run python manuscript/supp_doc/splat_count_vs_quality/build.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
ANALYSIS_RESULTS = HERE.parent.parent / "analysis" / "splat_count_vs_quality" / "results"
FIG_DIR = HERE / "figures"
TEX_FILE = HERE / "splat_count_vs_quality_all.tex"

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


def main():
    print(f"Building: {TEX_FILE.stem}")

    # 1. Copy per-dataset figures into figures/
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for ds in DATASETS:
        for src_dir in [ANALYSIS_RESULTS / ds, ANALYSIS_RESULTS / f"{ds}_n2s"]:
            if not src_dir.exists():
                continue
            for pdf in src_dir.glob("fig_*.pdf"):
                dst = FIG_DIR / f"{ds}_{pdf.name}"
                shutil.copy2(pdf, dst)
                n_copied += 1
    print(f"  Copied {n_copied} figures")

    # 2. Compile LaTeX (3 passes for references + TOC)
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
        print(f"  Built: {out_pdf.name} ({out_pdf.stat().st_size // 1024 // 1024} MB)")
    else:
        print(f"  ERROR: {out_pdf.name} not produced")


if __name__ == "__main__":
    main()
