# Project Statistics

This directory contains tools and reports for analyzing the Luxar codebase.

## Quick Start

Generate the statistics report:

```bash
make stats         # full report (runs the test suites with coverage)
make stats-fast    # file counts only, skips test runs (--no-tests)
```

Or run directly (with options):

```bash
hatch run python stats/generate_stats.py                 # full report with tests
hatch run python stats/generate_stats.py --no-tests      # fast, file counts only
hatch run python stats/generate_stats.py --no-coverage   # run tests, skip coverage
hatch run python stats/generate_stats.py --json          # also emit project_stats.json
```

Then view the report:

```bash
open stats/project_stats.html      # styled, browser-rendered
cat stats/PROJECT_STATS.md         # GitHub-friendly markdown summary
```

## Files

- **`generate_stats.py`** — Python script that analyzes the codebase and renders the reports
- **`project_stats.html`** — Generated HTML report with comprehensive project statistics
- **`PROJECT_STATS.md`** — GitHub-friendly markdown summary linked from the main README

## What's Analyzed

The statistics analyzer examines the entire Luxar codebase and provides:

### Supported Languages

The analyzer supports **14 programming and configuration languages**:

**Primary Languages (code):**
- **Python** (.py) - Backend, data processing, API
- **TypeScript** (.ts, .tsx) - Viewer frontend, WebGL rendering
- **Rust** (.rs) - WASM module for high-performance computations
- **CUDA** (.cu, .cuh) - GPU acceleration kernels
- **Go** (.go) - Native launcher binaries (`luxar export --native`)
- **JavaScript** (.js, .jsx, .mjs) - Scripts and configurations
- **CSS** (.css, .scss) - Styling

**Configuration:**
- **JSON** (.json) - Package manifests, configs
- **TOML** (.toml) - Rust and Python configs (pyproject.toml, Cargo.toml)
- **YAML** (.yaml, .yml) - CI/CD workflows, configs
- **Shell** (.sh, .bash) - Build and utility scripts
- **Makefile** (Makefile, .mk) - Build automation

**Documentation:**
- **Markdown** (.md) - READMEs, specifications, guides
- **HTML** (.html, .htm) - Generated reports

### Code Metrics
- **Lines of code** (executable code only)
- **Total lines** (including comments and blanks)
- **File counts** (all supported languages)
- **Code composition** (percentages of code, comments, blank lines)

### Structure Analysis
- **Classes** - Python classes and TypeScript classes
- **Functions** - Top-level functions
- **Methods** - Class methods (Python)
- **Interfaces** - TypeScript interfaces
- **Type Aliases** - TypeScript type definitions

### Module Breakdown
- Per-module statistics for Python packages
- Per-package statistics for TypeScript code
- Files and code lines for each module

## Report Features

The generated HTML report includes:

- 🎨 **Beautiful gradient design** with professional styling
- 📊 **Visual summary cards** with key metrics
- 📈 **Interactive progress bars** showing code composition
- 📂 **Detailed tables** for Python and TypeScript breakdowns
- 💡 **Key insights** about code density, language distribution, and architecture
- 🎯 **Hover effects** for better interactivity

## What's Excluded

The analyzer skips a directory whenever any path component matches a name in
`SKIP_DIR_NAMES`, so generated/build artifacts and vendored dependencies don't
inflate the counts. The skipped names include:

- Dependencies & caches: `node_modules/`, `__pycache__/`, `.pytest_cache/`,
  `.mypy_cache/`, `.ruff_cache/`, `.hatch/`, `.venv/`, `venv/`, `.tox/`, `.eggs/`
- Build & coverage output: `dist/`, `build/`, `_build/`, `target/`, `coverage/`,
  `htmlcov/`, `playwright-report/`, `test-results/`, `.playwright-mcp/`
- Data & scratch: `datasets/`, `delme/`, `build-cuda-logs/`, `cuda-build-logs/`
- VCS & tooling: `.git/`, `.idea/`, `.vscode/`, `.claude/`

## Technical Details

### Analysis Method

The analyzer:
1. **Recursively scans** files for all 14 supported languages
2. **Counts lines** by category (code, comments, blank) with language-specific comment detection
3. **Parses definitions** using regex patterns:
   - **Python**: classes, functions, methods
   - **TypeScript**: classes, functions, interfaces, type aliases
   - **Rust**: structs, functions, traits, impls, enums
   - **CUDA**: kernels, device functions, host functions
   - **Go**: functions, structs, interfaces
   - **CSS**: rules, variables, media queries
4. **Runs tests** with coverage for Python, TypeScript, and Rust
5. **Gathers Git statistics** (commits, contributors, activity)
6. **Analyzes dependencies** from package manifests (pyproject.toml, package.json, Cargo.toml)
7. **Generates HTML** with embedded CSS for portable viewing

### Line Counting

- **Code lines**: Non-blank, non-comment lines with executable code
- **Comment lines**: Lines starting with `#` (Python) or `//` (TypeScript), plus block comments
- **Blank lines**: Lines containing only whitespace

### Definition Counting

**Python:**
- Classes: `class ClassName`
- Functions: Top-level `def function_name()` (not indented)
- Methods: Indented `def method_name()` inside classes

**TypeScript:**
- Classes: `class ClassName` or `export class ClassName`
- Functions: `function name()`, `const name = () =>`, etc.
- Interfaces: `interface IName` or `export interface IName`
- Type Aliases: `type TypeName =` or `export type TypeName =`

## Updating the Report

The report is not automatically updated. Regenerate it:
- After significant code changes
- Before releases
- When you want current statistics

Simply run:
```bash
make stats
```

## Integration with Development Workflow

The stats command is designed to be:
- **Fast** - Completes in seconds
- **Non-invasive** - Doesn't modify any code
- **Portable** - HTML report can be shared/archived
- **Comprehensive** - Covers all relevant metrics

Use it to:
- Track project growth over time
- Understand codebase composition
- Identify large modules that might need refactoring
- Document project scope for contributors
- Generate metrics for project reports

## Example Output

Typical statistics for Luxar (run `make stats` for current values):
- **~2,500 files** across all supported languages (the exact count varies by revision)
- **~65,000+ lines of code** (executable)
- **Primary code mix**: Python (~60%), TypeScript (~30%), Rust (~5%), CUDA (~2%)
- **~300+ classes/structs** total
- **~600+ functions** total
- **Comprehensive test coverage** with Python/TypeScript/Rust tests

See `project_stats.html` for the full interactive report with charts and detailed breakdowns!
