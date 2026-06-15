# Luxar Project Statistics

_Generated 2026-06-10 23:01 &middot; For the styled report with progress bars and per-language detail, open [`project_stats.html`](./project_stats.html) locally._

_To refresh both this file and the HTML report, run `make stats` from the project root._

## Summary

| Metric | Value |
| --- | ---: |
| Total files | 2,133 |
| Lines of code (executable) | 448,483 |
| Total lines | 639,965 |
| Active languages | 14 |
| Test files | 675 |
| Tests collected | 11,566 |
| Coverage (Py/TS weighted) | 84.1% |
| Total commits | 1,969 |
| Commits in last 30 days | 818 |
| Project size | 299.3 MB |
| CI workflows | 2 |

## Language Breakdown

| Language | Files | Code | Total | Comments | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 575 | 124,208 | 201,894 | 43,964 | 27.7% |
| TypeScript | 1,058 | 168,431 | 254,434 | 56,786 | 37.6% |
| Rust | 11 | 2,452 | 3,853 | 940 | 0.5% |
| CUDA | 9 | 1,364 | 2,366 | 719 | 0.3% |
| Go | 1 | 147 | 220 | 51 | 0.0% |
| CSS | 24 | 4,211 | 5,835 | 697 | 0.9% |
| JavaScript | 5 | 857 | 1,820 | 842 | 0.2% |
| Shell | 5 | 304 | 435 | 77 | 0.1% |
| JSON | 73 | 65,928 | 65,928 | 0 | 14.7% |
| TOML | 4 | 383 | 517 | 85 | 0.1% |
| YAML | 6 | 3,277 | 4,300 | 90 | 0.7% |
| Markdown | 356 | 73,795 | 94,978 | 0 | 16.5% |
| HTML | 5 | 902 | 984 | 12 | 0.2% |
| Makefile | 1 | 2,224 | 2,401 | 76 | 0.5% |

## Primary Languages

- **Python** &mdash; 575 files, 124,208 LOC, 834 classes, 2,194 functions, 3,951 methods (coverage 86.0%, tests 4,036).
- **TypeScript** &mdash; 1,058 files, 168,431 LOC, 154 classes, 1,567 functions, 571 interfaces, 171 types (coverage 82.6%, tests 7,468).
- **Rust (WASM)** &mdash; 11 files, 2,452 LOC, 102 functions (tests 62).
- **CUDA** &mdash; 9 files, 1,364 LOC, 2 kernels.
- **Go (launchers)** &mdash; 1 files, 147 LOC.

## Python subpackages (`packages/luxar/src/luxar/`)

| Subpackage | Files | Code Lines | Comments |
| --- | ---: | ---: | ---: |
| `gsplats` | 221 | 45,715 | 16,369 |
| `demos` | 60 | 23,053 | 9,252 |
| `core` | 78 | 12,566 | 4,891 |
| `cli` | 18 | 10,759 | 2,664 |
| `io` | 50 | 8,171 | 2,741 |
| `validation` | 17 | 3,310 | 1,155 |
| `encoding` | 14 | 2,845 | 1,125 |
| `utils` | 18 | 2,732 | 1,241 |
| `typing_utils` | 9 | 891 | 407 |
| `colormaps` | 5 | 390 | 138 |
| `<root>` | 6 | 135 | 77 |
| `tests` | 2 | 16 | 5 |

## TypeScript subpackages (`packages/luxar-viewer/src/`)

| Subpackage | Files | Code Lines | Comments |
| --- | ---: | ---: | ---: |
| `tests` | 512 | 109,743 | 22,016 |
| `ui` | 84 | 14,415 | 5,872 |
| `data` | 112 | 12,553 | 6,968 |
| `rendering` | 96 | 12,233 | 6,992 |
| `scene` | 19 | 2,836 | 2,667 |
| `core` | 29 | 1,886 | 1,291 |
| `cache` | 15 | 1,849 | 1,460 |
| `controls` | 22 | 1,835 | 857 |
| `workers` | 32 | 1,774 | 1,183 |
| `input` | 22 | 1,753 | 1,815 |
| `config` | 52 | 1,667 | 758 |
| `wasm` | 12 | 1,355 | 934 |
| `types` | 12 | 1,216 | 1,395 |
| `themes` | 8 | 1,101 | 510 |
| `utils` | 15 | 861 | 802 |
| `profiling` | 1 | 322 | 280 |
| `<root>` | 2 | 15 | 58 |

## Tests & Coverage

| Metric | Python | TypeScript | Rust | E2E | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test files | 177 | 429 | 9 | 60 | 675 |
| Tests collected | 4,036 | 7,468 | 62 | &mdash; | 11,566 |
| Tests passed | 3,818 | 7,457 | 62 | &mdash; | 11,337 |
| Coverage | 86.0% | 82.6% | &mdash; | &mdash; | 84.1% |

## Git Activity

| Metric | Value |
| --- | ---: |
| Current branch | `viewer-embedder-api` |
| Total commits | 1,969 |
| Contributors (all-time) | 6 |
| Local branches | 3 |
| Remote branches | 7 |
| Tags | 2 |
| Commits (last 30 days) | 818 |
| Files changed (last 30 days) | 2,100 |
| First commit | 2025-08-04 |
| Last commit | 2026-06-10 |

### Top contributors

| Name | Commits |
| --- | ---: |
| royer | 2,457 |
| Loic A. Royer | 420 |
| Loic A. Royer (obsidian) | 394 |
| Loic Royer | 84 |
| dependabot[bot] | 11 |

## Dependencies

| Ecosystem | Production | Development | Total | Notes |
| --- | ---: | ---: | ---: | --- |
| Python | 13 | 29 | 42 | groups: demos=13, dev=6, docs=3, gsplats=2, io=2, test=3 |
| Node.js | 4 | 23 | 27 | `packages/luxar-viewer/package.json` |
| Rust | 1 | 1 | 2 | `packages/luxar-viewer/src/wasm/rust/Cargo.toml` |

## Largest Source Files

### Python

| Path | Code Lines |
| --- | ---: |
| `packages/luxar/src/luxar/cli/gsplat_commands.py` | 3,914 |
| `packages/luxar/src/luxar/cli/tests/test_gsplat_cli_extended.py` | 1,984 |
| `stats/generate_stats.py` | 1,744 |
| `packages/luxar/src/luxar/gsplats/gsplat_data.py` | 1,415 |
| `packages/luxar/src/luxar/demos/demo_ppi_flow_field.py` | 1,369 |
| `packages/luxar-viewer/tests/fixtures/generate_test_data.py` | 1,291 |
| `packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/test_cuda_nd.py` | 1,100 |
| `packages/luxar/src/luxar/io/compiler.py` | 1,077 |
| `packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py` | 1,015 |
| `packages/luxar/src/luxar/cli/main.py` | 972 |

### TypeScript

| Path | Code Lines |
| --- | ---: |
| `packages/luxar-viewer/src/tests/benchmarks/wasm-benchmark.ts` | 1,501 |
| `packages/luxar-viewer/src/tests/mocks/three.mock.ts` | 1,379 |
| `packages/luxar-viewer/src/ui/data-loading-monitor.ts` | 1,357 |
| `packages/luxar-viewer/src/tests/unit/cache/multi-level-caching-store.test.ts` | 1,314 |
| `packages/luxar-viewer/src/tests/unit/ui/data-loading-monitor.test.ts` | 1,163 |
| `packages/luxar-viewer/src/tests/unit/data/array-decoder/decoder.test.ts` | 1,148 |
| `packages/luxar-viewer/src/ui/data-loading-monitor/templates.ts` | 1,115 |
| `packages/luxar-viewer/src/tests/unit/data/scene-loader.test.ts` | 1,111 |
| `packages/luxar-viewer/src/tests/unit/wasm/wasm-vs-typescript.test.ts` | 1,075 |
| `packages/luxar-viewer/src/tests/unit/wasm/visibility-fallbacks.test.ts` | 1,027 |

---

_Excludes: `node_modules`, `__pycache__`, `coverage`, `dist`, `build`, `target`, `_build`, `playwright-report`, `datasets`, `delme`, `.venv`. Source: `stats/generate_stats.py`._
