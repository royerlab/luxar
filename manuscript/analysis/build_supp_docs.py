#!/usr/bin/env python3
"""Build supplementary documents by concatenating per-dataset PDFs.

Creates one merged PDF per analysis, then copies them to manuscript/supp_doc/.

Usage::

    hatch run python manuscript/analysis/build_supp_docs.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Try pypdf first (modern), fall back to PyPDF2
try:
    from pypdf import PdfMerger
except ImportError:
    try:
        from PyPDF2 import PdfMerger
    except ImportError:
        PdfMerger = None


ANALYSIS_DIR = Path(__file__).parent
SUPP_DOC_DIR = ANALYSIS_DIR.parent / "supp_doc"

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


def merge_pdfs(pdf_paths: list[Path], output: Path) -> bool:
    """Merge multiple PDFs into one. Returns True on success."""
    if PdfMerger is None:
        print("  WARNING: pypdf/PyPDF2 not installed. Using fallback (gs).")
        return merge_pdfs_gs(pdf_paths, output)

    merger = PdfMerger()
    for p in pdf_paths:
        merger.append(str(p))
    output.parent.mkdir(parents=True, exist_ok=True)
    merger.write(str(output))
    merger.close()
    return True


def merge_pdfs_gs(pdf_paths: list[Path], output: Path) -> bool:
    """Fallback: merge PDFs using ghostscript."""
    import subprocess

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gs", "-dBATCH", "-dNOPAUSE", "-q", "-sDEVICE=pdfwrite",
        f"-sOutputFile={output}",
    ] + [str(p) for p in pdf_paths]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ERROR: Neither pypdf nor ghostscript available.")
        return False


def build_splat_count_vs_quality():
    """Build the splat_count_vs_quality supplementary document via LaTeX.

    First copies per-dataset figures into the LaTeX figures/ directory,
    then compiles the .tex file to produce the final PDF.
    """
    import subprocess

    base = ANALYSIS_DIR / "splat_count_vs_quality" / "results"
    tex_dir = SUPP_DOC_DIR / "splat_count_vs_quality"
    fig_dir = tex_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Copy per-dataset figures into figures/ for LaTeX
    n_copied = 0
    for ds in DATASETS:
        for src_dir, prefix in [
            (base / ds, ds),
            (base / f"{ds}_n2s", ds),
        ]:
            if not src_dir.exists():
                continue
            for pdf in src_dir.glob("fig_*.pdf"):
                dst = fig_dir / f"{prefix}_{pdf.name}"
                shutil.copy2(pdf, dst)
                n_copied += 1

    print(f"  Copied {n_copied} figures to {fig_dir}")

    # Compile LaTeX (3 passes for references + TOC)
    tex_file = tex_dir / "splat_count_vs_quality_all.tex"
    if not tex_file.exists():
        print(f"  WARNING: {tex_file} not found, skipping LaTeX compilation")
        return

    print(f"  Compiling {tex_file.name}...")
    for i, cmd in enumerate([
        ["pdflatex", "-interaction=nonstopmode", tex_file.name],
        ["bibtex", tex_file.stem],
        ["pdflatex", "-interaction=nonstopmode", tex_file.name],
        ["pdflatex", "-interaction=nonstopmode", tex_file.name],
    ]):
        result = subprocess.run(cmd, cwd=tex_dir, capture_output=True, text=True)
        if result.returncode != 0 and i != 1:  # bibtex warnings are OK
            print(f"  WARNING: {cmd[0]} returned {result.returncode}")

    out_pdf = tex_dir / f"{tex_file.stem}.pdf"
    if out_pdf.exists():
        print(f"  Built: {out_pdf} ({out_pdf.stat().st_size // 1024 // 1024} MB)")
    else:
        print(f"  ERROR: {out_pdf} not produced")


def build_convergence():
    """Merge all convergence PDFs."""
    base = ANALYSIS_DIR / "convergence" / "results"
    pdfs = []

    for ds in DATASETS:
        p = base / ds / "fig_convergence.pdf"
        if p.exists():
            pdfs.append(p)

    if pdfs:
        out = SUPP_DOC_DIR / "convergence" / "convergence_all.pdf"
        print(f"Merging {len(pdfs)} PDFs -> {out}")
        merge_pdfs(pdfs, out)

        interp = ANALYSIS_DIR / "convergence" / "INTERPRETATION.md"
        if interp.exists():
            shutil.copy2(interp, out.parent / "INTERPRETATION.md")
    else:
        print("  No PDFs found for convergence")


def build_progressive():
    """Merge all progressive vs single-pass PDFs."""
    base = ANALYSIS_DIR / "progressive_vs_single" / "results"
    pdfs = []

    for ds in DATASETS:
        for name in ["fig_progressive_comparison.pdf", "fig_progressive_slices.pdf"]:
            p = base / ds / name
            if p.exists():
                pdfs.append(p)

    if pdfs:
        out = SUPP_DOC_DIR / "progressive_vs_single" / "progressive_vs_single_all.pdf"
        print(f"Merging {len(pdfs)} PDFs -> {out}")
        merge_pdfs(pdfs, out)

        interp = ANALYSIS_DIR / "progressive_vs_single" / "INTERPRETATION.md"
        if interp.exists():
            shutil.copy2(interp, out.parent / "INTERPRETATION.md")
    else:
        print("  No PDFs found for progressive_vs_single")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build supplementary PDF documents")
    parser.add_argument(
        "--analysis",
        choices=["all", "splat_count_vs_quality", "convergence", "progressive"],
        default="all",
        help="Which analysis to build (default: all)",
    )
    args = parser.parse_args()

    print("Building supplementary documents...\n")

    if args.analysis in ("all", "splat_count_vs_quality"):
        build_splat_count_vs_quality()
    if args.analysis in ("all", "progressive"):
        build_progressive()
    if args.analysis in ("all", "convergence"):
        build_convergence()

    print("\nDone. Supplementary documents in:", SUPP_DOC_DIR)


if __name__ == "__main__":
    main()
