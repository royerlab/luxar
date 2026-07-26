# Luxar Project Statistics

_Generated 2026-07-26 09:40 &middot; For the styled report with progress bars and per-language detail, open [`project_stats.html`](./project_stats.html) locally._

_To refresh both this file and the HTML report, run `make stats` from the project root._

## Summary

| Metric | Value |
| --- | ---: |
| Total files | 2,501 |
| Lines of code (executable) | 481,561 |
| Total lines | 719,025 |
| Active languages | 14 |
| Test files | 805 |
| Tests collected | 15,255 |
| Coverage (Py/TS weighted) | 86.2% |
| Total commits | 2,266 |
| Commits in last 30 days | 262 |
| Project size | 666.3 MB |
| CI workflows | 4 |

## Language Breakdown

| Language | Files | Code | Total | Comments | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 793 | 165,521 | 264,838 | 57,383 | 34.4% |
| TypeScript | 1,190 | 206,604 | 315,866 | 75,290 | 42.9% |
| Rust | 8 | 2,598 | 3,982 | 966 | 0.5% |
| CUDA | 9 | 1,364 | 2,366 | 719 | 0.3% |
| Go | 2 | 259 | 362 | 70 | 0.1% |
| CSS | 27 | 5,026 | 7,026 | 938 | 1.0% |
| JavaScript | 6 | 755 | 1,002 | 167 | 0.2% |
| Shell | 6 | 450 | 649 | 122 | 0.1% |
| JSON | 71 | 10,973 | 10,973 | 0 | 2.3% |
| TOML | 4 | 440 | 643 | 150 | 0.1% |
| YAML | 9 | 3,348 | 4,498 | 238 | 0.7% |
| Markdown | 370 | 81,040 | 103,363 | 0 | 16.8% |
| HTML | 5 | 910 | 994 | 13 | 0.2% |
| Makefile | 1 | 2,273 | 2,463 | 82 | 0.5% |

## Primary Languages

- **Python** &mdash; 793 files, 165,521 LOC, 1,122 classes, 3,373 functions, 5,147 methods (coverage 88.0%, tests 5,917).
- **TypeScript** &mdash; 1,190 files, 206,604 LOC, 168 classes, 2,054 functions, 671 interfaces, 192 types (coverage 84.8%, tests 9,267).
- **Rust (WASM)** &mdash; 8 files, 2,598 LOC, 123 functions (tests 71).
- **CUDA** &mdash; 9 files, 1,364 LOC, 2 kernels.
- **Go (launchers)** &mdash; 2 files, 259 LOC.

## Python subpackages (`packages/luxar/src/luxar/`)

| Subpackage | Files | Code Lines | Comments |
| --- | ---: | ---: | ---: |
| `gsplats` | 288 | 61,689 | 22,243 |
| `demos` | 98 | 30,085 | 11,739 |
| `cli` | 66 | 19,318 | 3,978 |
| `core` | 84 | 14,515 | 5,592 |
| `io` | 70 | 9,741 | 3,663 |
| `encoding` | 31 | 5,311 | 1,959 |
| `utils` | 24 | 3,935 | 1,639 |
| `validation` | 17 | 3,717 | 1,360 |
| `typing_utils` | 11 | 1,092 | 455 |
| `colormaps` | 7 | 473 | 189 |
| `<root>` | 6 | 134 | 86 |
| `tests` | 3 | 51 | 14 |

## TypeScript subpackages (`packages/luxar-viewer/src/`)

| Subpackage | Files | Code Lines | Comments |
| --- | ---: | ---: | ---: |
| `tests` | 586 | 137,207 | 28,760 |
| `ui` | 97 | 16,283 | 6,850 |
| `data` | 121 | 15,082 | 9,776 |
| `rendering` | 107 | 15,001 | 11,697 |
| `scene` | 26 | 3,815 | 4,074 |
| `core` | 33 | 2,683 | 1,873 |
| `cache` | 19 | 2,393 | 2,032 |
| `config` | 57 | 2,080 | 1,022 |
| `workers` | 34 | 1,940 | 1,337 |
| `controls` | 22 | 1,870 | 897 |
| `input` | 22 | 1,849 | 1,900 |
| `wasm` | 9 | 1,465 | 908 |
| `types` | 16 | 1,449 | 1,840 |
| `themes` | 8 | 1,130 | 534 |
| `utils` | 18 | 937 | 882 |
| `profiling` | 1 | 425 | 398 |
| `<root>` | 2 | 25 | 61 |

## Tests & Coverage

| Metric | Python | TypeScript | Rust | E2E | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test files | 246 | 488 | 6 | 65 | 805 |
| Tests collected | 5,917 | 9,267 | 71 | &mdash; | 15,255 |
| Tests passed | 5,695 | 9,261 | 71 | &mdash; | 15,027 |
| Coverage | 88.0% | 84.8% | &mdash; | &mdash; | 86.2% |

## Git Activity

| Metric | Value |
| --- | ---: |
| Current branch | `stats-fix` |
| Total commits | 2,266 |
| Contributors (all-time) | 5 |
| Local branches | 18 |
| Remote branches | 17 |
| Tags | 2 |
| Commits (last 30 days) | 262 |
| Files changed (last 30 days) | 1,549 |
| First commit | 2025-08-04 |
| Last commit | 2026-07-25 |

### Top contributors

| Name | Commits |
| --- | ---: |
| royer | 2,500 |
| Loic A. Royer | 689 |
| Loic A. Royer (obsidian) | 394 |
| Loic Royer | 84 |
| dependabot[bot] | 44 |

## Dependencies

| Ecosystem | Production | Development | Total | Notes |
| --- | ---: | ---: | ---: | --- |
| Python | 13 | 38 | 51 | groups: demos=18, dev=7, docs=3, gsplats=2, io=2, test=4, tracksdata=2 |
| Node.js | 4 | 21 | 25 | `packages/luxar-viewer/package.json` |
| Rust | 1 | 1 | 2 | `packages/luxar-viewer/src/wasm/rust/Cargo.toml` |

## Largest Source Files

### Python

| Path | Code Lines |
| --- | ---: |
| `packages/luxar/src/luxar/cli/tests/test_gsplat_cli_extended.py` | 4,124 |
| `packages/luxar-viewer/tests/fixtures/generate_test_data.py` | 2,112 |
| `packages/luxar/src/luxar/gsplats/tests/test_batch.py` | 1,787 |
| `stats/generate_stats.py` | 1,766 |
| `packages/luxar/src/luxar/demos/demo_ppi_flow_field.py` | 1,386 |
| `packages/luxar/src/luxar/gsplats/lod/tests/test_substitutive.py` | 1,328 |
| `packages/luxar/src/luxar/gsplats/io/tests/test_save_load.py` | 1,251 |
| `packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/test_cuda_nd.py` | 1,100 |
| `packages/luxar/src/luxar/demos/demo_gsplats_4d_celegans_tracking.py` | 1,019 |
| `packages/luxar/src/luxar/gsplats/gsplat_data.py` | 952 |

### TypeScript

| Path | Code Lines |
| --- | ---: |
| `packages/luxar-viewer/src/tests/unit/scene/lod-group-registry.test.ts` | 1,722 |
| `packages/luxar-viewer/src/tests/unit/cache/multi-level-caching-store.test.ts` | 1,536 |
| `packages/luxar-viewer/src/ui/data-loading-monitor.ts` | 1,527 |
| `packages/luxar-viewer/src/tests/unit/data/scene-loader.test.ts` | 1,481 |
| `packages/luxar-viewer/src/ui/data-loading-monitor/templates.ts` | 1,472 |
| `packages/luxar-viewer/src/tests/mocks/three.mock.ts` | 1,379 |
| `packages/luxar-viewer/src/tests/benchmarks/wasm-benchmark.ts` | 1,339 |
| `packages/luxar-viewer/src/tests/unit/rendering/depth-sort-coordinator.test.ts` | 1,294 |
| `packages/luxar-viewer/src/tests/unit/ui/data-loading-monitor.test.ts` | 1,272 |
| `packages/luxar-viewer/src/tests/unit/data/array-decoder/decoder.test.ts` | 1,211 |

---

_Excludes: `node_modules`, `__pycache__`, `coverage`, `dist`, `build`, `target`, `_build`, `playwright-report`, `datasets`, `delme`, `.venv`. Source: `stats/generate_stats.py`._
