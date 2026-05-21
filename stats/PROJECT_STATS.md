# Luxar Project Statistics

_Generated 2026-05-21 12:21 &middot; For the styled report with progress bars and per-language detail, open [`project_stats.html`](./project_stats.html) locally._

_To refresh both this file and the HTML report, run `make stats` from the project root._

## Summary

| Metric | Value |
| --- | ---: |
| Total files | 1,793 |
| Lines of code (executable) | 388,910 |
| Total lines | 554,047 |
| Active languages | 14 |
| Test files | 532 |
| Tests collected | 0 |
| Coverage (Py/TS weighted) | 0.0% |
| Total commits | 1,647 |
| Commits in last 30 days | 742 |
| Project size | 334.8 MB |
| CI workflows | 4 |

## Language Breakdown

| Language | Files | Code | Total | Comments | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 452 | 110,064 | 178,580 | 37,951 | 28.3% |
| TypeScript | 912 | 140,765 | 210,648 | 44,229 | 36.2% |
| Rust | 11 | 2,139 | 3,407 | 835 | 0.5% |
| CUDA | 9 | 1,364 | 2,366 | 719 | 0.4% |
| Go | 1 | 147 | 220 | 51 | 0.0% |
| CSS | 24 | 4,156 | 5,742 | 668 | 1.1% |
| JavaScript | 5 | 827 | 1,714 | 767 | 0.2% |
| Shell | 4 | 244 | 347 | 57 | 0.1% |
| JSON | 26 | 52,069 | 52,069 | 0 | 13.4% |
| TOML | 3 | 392 | 506 | 65 | 0.1% |
| YAML | 6 | 3,226 | 4,216 | 53 | 0.8% |
| Markdown | 334 | 70,401 | 90,872 | 0 | 18.1% |
| HTML | 5 | 902 | 984 | 12 | 0.2% |
| Makefile | 1 | 2,214 | 2,376 | 64 | 0.6% |

## Primary Languages

- **Python** &mdash; 452 files, 110,064 LOC, 764 classes, 1,758 functions, 3,483 methods (coverage 0.0%, tests 0).
- **TypeScript** &mdash; 912 files, 140,765 LOC, 148 classes, 1,339 functions, 532 interfaces, 154 types (coverage 0.0%, tests 0).
- **Rust (WASM)** &mdash; 11 files, 2,139 LOC, 100 functions (tests 0).
- **CUDA** &mdash; 9 files, 1,364 LOC, 2 kernels.
- **Go (launchers)** &mdash; 1 files, 147 LOC.

## Python subpackages (`packages/luxar/src/luxar/`)

| Subpackage | Files | Code Lines | Comments |
| --- | ---: | ---: | ---: |
| `gsplats` | 203 | 43,915 | 15,437 |
| `demos` | 58 | 22,558 | 8,959 |
| `cli` | 17 | 10,316 | 2,425 |
| `core` | 32 | 6,940 | 2,552 |
| `io` | 20 | 6,273 | 2,143 |
| `validation` | 15 | 2,884 | 1,018 |
| `encoding` | 14 | 2,768 | 1,058 |
| `utils` | 18 | 2,588 | 1,152 |
| `typing_utils` | 9 | 861 | 382 |
| `colormaps` | 5 | 343 | 102 |
| `<root>` | 6 | 132 | 59 |
| `tests` | 2 | 16 | 5 |

## TypeScript subpackages (`packages/luxar-viewer/src/`)

| Subpackage | Files | Code Lines | Comments |
| --- | ---: | ---: | ---: |
| `tests` | 399 | 84,763 | 12,758 |
| `ui` | 84 | 13,843 | 5,416 |
| `rendering` | 94 | 12,144 | 6,622 |
| `data` | 88 | 11,493 | 6,188 |
| `scene` | 18 | 2,444 | 2,225 |
| `controls` | 22 | 1,832 | 805 |
| `workers` | 30 | 1,812 | 995 |
| `cache` | 14 | 1,773 | 1,300 |
| `core` | 29 | 1,754 | 1,100 |
| `input` | 22 | 1,749 | 1,771 |
| `config` | 52 | 1,644 | 714 |
| `wasm` | 12 | 1,228 | 809 |
| `themes` | 8 | 1,104 | 499 |
| `types` | 10 | 1,079 | 1,197 |
| `utils` | 14 | 790 | 679 |
| `profiling` | 1 | 303 | 234 |
| `<root>` | 2 | 15 | 58 |

## Tests & Coverage

| Metric | Python | TypeScript | Rust | E2E | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test files | 143 | 322 | 9 | 58 | 532 |
| Tests collected | 0 | 0 | 0 | &mdash; | 0 |
| Tests passed | 0 | 0 | 0 | &mdash; | 0 |
| Coverage | 0.0% | 0.0% | &mdash; | &mdash; | 0.0% |

## Git Activity

| Metric | Value |
| --- | ---: |
| Current branch | `main` |
| Total commits | 1,647 |
| Contributors (all-time) | 5 |
| Local branches | 1 |
| Remote branches | 7 |
| Tags | 2 |
| Commits (last 30 days) | 742 |
| Files changed (last 30 days) | 1,721 |
| First commit | 2025-08-04 |
| Last commit | 2026-05-21 |

### Top contributors

| Name | Commits |
| --- | ---: |
| royer | 2,834 |
| Loic A. Royer (obsidian) | 388 |
| Loic A. Royer | 196 |
| Loic Royer | 84 |
| Jordao Bragantini | 3 |

## Dependencies

| Ecosystem | Production | Development | Total | Notes |
| --- | ---: | ---: | ---: | --- |
| Python | 13 | 33 | 46 | groups: demos=13, dev=9, docs=3, gsplats=2, io=2, test=4 |
| Node.js | 4 | 23 | 27 | `packages/luxar-viewer/package.json` |
| Rust | 1 | 1 | 2 | `packages/luxar-viewer/src/wasm/rust/Cargo.toml` |

## Largest Source Files

### Python

| Path | Code Lines |
| --- | ---: |
| `packages/luxar/src/luxar/cli/gsplat_commands.py` | 3,802 |
| `packages/luxar/src/luxar/cli/tests/test_gsplat_cli_extended.py` | 1,913 |
| `packages/luxar/src/luxar/io/compiler.py` | 1,778 |
| `packages/luxar/src/luxar/gsplats/tests/test_gsplat_data.py` | 1,753 |
| `stats/generate_stats.py` | 1,744 |
| `packages/luxar/src/luxar/demos/demo_ppi_flow_field.py` | 1,369 |
| `packages/luxar/src/luxar/gsplats/gsplat_data.py` | 1,295 |
| `packages/luxar-viewer/tests/fixtures/generate_test_data.py` | 1,190 |
| `packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/test_cuda_nd.py` | 1,100 |
| `packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py` | 1,017 |

### TypeScript

| Path | Code Lines |
| --- | ---: |
| `packages/luxar-viewer/src/tests/benchmarks/wasm-benchmark.ts` | 1,501 |
| `packages/luxar-viewer/src/tests/mocks/three.mock.ts` | 1,379 |
| `packages/luxar-viewer/src/tests/unit/wasm/wasm-comparison.test.ts` | 1,249 |
| `packages/luxar-viewer/src/tests/unit/cache/multi-level-caching-store.test.ts` | 1,236 |
| `packages/luxar-viewer/src/ui/data-loading-monitor.ts` | 1,230 |
| `packages/luxar-viewer/src/data/points/points-spatial-index-loader.ts` | 1,042 |
| `packages/luxar-viewer/src/ui/data-loading-monitor/templates.ts` | 991 |
| `packages/luxar-viewer/src/tests/unit/data/array-decoder/decoder.test.ts` | 960 |
| `packages/luxar-viewer/src/tests/unit/data/scene-loader.test.ts` | 946 |
| `packages/luxar-viewer/src/tests/unit/data/data-loading-monitor.test.ts` | 920 |

---

_Excludes: `node_modules`, `__pycache__`, `coverage`, `dist`, `build`, `target`, `_build`, `playwright-report`, `datasets`, `delme`, `.venv`. Source: `stats/generate_stats.py`._
