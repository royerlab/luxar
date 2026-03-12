#!/usr/bin/env python3
"""
Analyze CUDA performance bisection results and generate comparison report.

Reads JSON results from docs/benchmarks/bisection/ and produces a markdown
report showing performance trends across commits.

Usage:
    python scripts/benchmarks/benchmark_bisect_analyze.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "benchmarks" / "bisection"
OUTPUT_MD = REPO_ROOT / "docs" / "benchmarks" / "cuda_bisection_report.md"

# Commit order (oldest to newest)
COMMIT_ORDER = [
    "f8a1ebb",
    "922493e",
    "cc0882f",
    "5ec1367",
    "aba16bb",
    "18e4d09",
    "b985c6f",
    "e47c55d",
    "daa7d5f",
    "adcdb2d",
    "ee57fea",
    "101fd4d",
    "0629c4f",
    "136b5d7",
    "cee7441",
]

COMMIT_MESSAGES = {
    "f8a1ebb": "feat: add CUDA backend for gsplats",
    "922493e": "fix(cuda): fix build system after make clean-cuda",
    "cc0882f": "feat(cuda): add global splat handling for large splats",
    "5ec1367": "feat(cuda): add FP16 precision support",
    "aba16bb": "feat(cuda): add AMP support and expand FP16/test coverage",
    "18e4d09": "feat(cuda): add grad_output caching and refactor kernels",
    "b985c6f": "feat(cuda): add dynamic GPU capability detection",
    "e47c55d": "feat(cuda): add dynamic SPLAT_BATCH_SIZE scaling",
    "daa7d5f": "perf(cuda): architecture-aware batch size selection",
    "adcdb2d": "fix(cuda): use architecture defaults for shared_memory",
    "ee57fea": "feat: add gsplat constraint params, dynamic range quantization",
    "101fd4d": "feat: add comprehensive gsplat CLI tools, improve dynamic ops",
    "0629c4f": "feat: comprehensive codebase review improvements",
    "136b5d7": "perf(cuda): revert register-spilling, use fast math intrinsics",
    "cee7441": "chore: misc fixes from concurrent agents",
}

CONFIG_LABELS = [
    "3D_128_1K",
    "3D_256_10K",
    "3D_512_50K",
    "2D_1024_5K",
    "2D_4096_50K",
]


def load_results():
    """Load all benchmark results."""
    results = {}
    for sha in COMMIT_ORDER:
        json_path = RESULTS_DIR / f"bench_{sha}.json"
        if json_path.exists():
            with open(json_path) as f:
                results[sha] = json.load(f)
        else:
            results[sha] = {"commit": sha, "error": "No benchmark data"}
    return results


def fmt_ms(val, prev_val=None):
    """Format a millisecond value with optional delta indicator."""
    if val is None:
        return "-"
    s = f"{val:.2f}"
    if prev_val is not None and prev_val > 0:
        pct = ((val - prev_val) / prev_val) * 100
        if abs(pct) > 5:
            arrow = "↑" if pct > 0 else "↓"
            s += f" ({arrow}{abs(pct):.0f}%)"
    return s


def generate_report(results):
    """Generate markdown comparison report."""
    lines = []
    lines.append("# CUDA Performance Bisection Report\n")
    lines.append(
        f"**Commits analyzed**: {len([r for r in results.values() if 'error' not in r])}"
    )

    # Get GPU info from first successful result
    for r in results.values():
        if "gpu" in r:
            lines.append(f"**GPU**: {r['gpu']}")
            lines.append(f"**CUDA**: {r['cuda_version']}")
            lines.append(f"**PyTorch**: {r['pytorch_version']}")
            break
    lines.append("")

    # Commit summary table
    lines.append("## Commits (oldest → newest)\n")
    lines.append("| # | Commit | Message | Status |")
    lines.append("|---|--------|---------|--------|")
    for i, sha in enumerate(COMMIT_ORDER, 1):
        r = results.get(sha, {})
        msg = COMMIT_MESSAGES.get(sha, "?")
        status = "Error: " + r.get("error", "?") if "error" in r else "OK"
        lines.append(f"| {i} | `{sha}` | {msg} | {status} |")
    lines.append("")

    # Per-config performance tables
    for config_label in CONFIG_LABELS:
        lines.append(f"## {config_label.replace('_', ' ')}\n")

        # Inference table
        lines.append("### Inference (ms, lower is better)\n")
        lines.append("| # | Commit | FP32 | AMP | FP16 | FP32 GV/s |")
        lines.append("|---|--------|------|-----|------|-----------|")

        prev_fp32 = None
        prev_amp = None
        for i, sha in enumerate(COMMIT_ORDER, 1):
            r = results.get(sha, {})
            if "error" in r:
                lines.append(f"| {i} | `{sha}` | - | - | - | - |")
                continue

            cfg = r.get("configs", {}).get(config_label, {})
            fp32 = cfg.get("fp32_inference_ms")
            amp = cfg.get("amp_inference_ms")
            fp16 = cfg.get("fp16_inference_ms")
            gvs = cfg.get("throughput_gvs_fp32")

            fp32_s = fmt_ms(fp32, prev_fp32)
            amp_s = fmt_ms(amp, prev_amp)
            fp16_s = fmt_ms(fp16)
            gvs_s = f"{gvs:.1f}" if gvs else "-"

            lines.append(f"| {i} | `{sha}` | {fp32_s} | {amp_s} | {fp16_s} | {gvs_s} |")

            if fp32 is not None:
                prev_fp32 = fp32
            if amp is not None:
                prev_amp = amp
        lines.append("")

        # Training table
        lines.append("### Training (ms, lower is better)\n")
        lines.append("| # | Commit | FP32 Fwd | FP32 Bwd | FP32 Tot | AMP Tot |")
        lines.append("|---|--------|----------|----------|----------|---------|")

        prev_fp32_tot = None
        for i, sha in enumerate(COMMIT_ORDER, 1):
            r = results.get(sha, {})
            if "error" in r:
                lines.append(f"| {i} | `{sha}` | - | - | - | - |")
                continue

            cfg = r.get("configs", {}).get(config_label, {})
            fp32_fwd = cfg.get("fp32_train_fwd_ms")
            fp32_bwd = cfg.get("fp32_train_bwd_ms")
            fp32_tot = cfg.get("fp32_train_total_ms")
            amp_tot = cfg.get("amp_train_total_ms")

            fp32_fwd_s = fmt_ms(fp32_fwd)
            fp32_bwd_s = fmt_ms(fp32_bwd)
            fp32_tot_s = fmt_ms(fp32_tot, prev_fp32_tot)
            amp_tot_s = fmt_ms(amp_tot)

            lines.append(
                f"| {i} | `{sha}` | {fp32_fwd_s} | {fp32_bwd_s} | {fp32_tot_s} | {amp_tot_s} |"
            )

            if fp32_tot is not None:
                prev_fp32_tot = fp32_tot
        lines.append("")

    # Summary: identify regressions and improvements
    lines.append("## Performance Change Summary\n")
    lines.append("Comparing each commit to its predecessor. ")
    lines.append("Changes > 10% are highlighted.\n")

    lines.append(
        "| Commit | Message | 3D_128_1K | 3D_256_10K | 3D_512_50K | 2D_1024_5K | 2D_4096_50K |"
    )
    lines.append(
        "|--------|---------|-----------|------------|------------|------------|-------------|"
    )

    successful_shas = [
        sha for sha in COMMIT_ORDER if "error" not in results.get(sha, {"error": True})
    ]
    for idx, sha in enumerate(successful_shas):
        msg = COMMIT_MESSAGES.get(sha, "?")[:50]
        r = results[sha]
        cells = []
        for config_label in CONFIG_LABELS:
            cfg = r.get("configs", {}).get(config_label, {})
            fp32 = cfg.get("fp32_inference_ms")
            if fp32 is None:
                cells.append("-")
                continue

            # Find previous successful commit's value
            prev_fp32 = None
            if idx > 0:
                prev_sha = successful_shas[idx - 1]
                prev_cfg = results[prev_sha].get("configs", {}).get(config_label, {})
                prev_fp32 = prev_cfg.get("fp32_inference_ms")

            if prev_fp32 and prev_fp32 > 0:
                pct = ((fp32 - prev_fp32) / prev_fp32) * 100
                if pct > 10:
                    cells.append(f"**+{pct:.0f}%** SLOWER")
                elif pct < -10:
                    cells.append(f"**{pct:.0f}%** FASTER")
                else:
                    cells.append(f"{pct:+.0f}%")
            else:
                cells.append(f"{fp32:.2f}ms (base)")

        lines.append(f"| `{sha}` | {msg} | {' | '.join(cells)} |")
    lines.append("")

    # Errors section
    errors = {sha: r for sha, r in results.items() if "error" in r}
    if errors:
        lines.append("## Errors\n")
        for sha, r in errors.items():
            lines.append(f"- `{sha}` ({COMMIT_MESSAGES.get(sha, '?')}): {r['error']}")
        lines.append("")

    return "\n".join(lines)


def main():
    print(f"Loading results from {RESULTS_DIR}...")
    results = load_results()

    ok = sum(1 for r in results.values() if "error" not in r)
    err = sum(1 for r in results.values() if "error" in r)
    print(f"  Loaded: {ok} successful, {err} failed/missing")

    report = generate_report(results)

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)

    print(f"\nReport written to {OUTPUT_MD}")
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
