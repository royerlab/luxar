# Project Statistics

This directory contains tools and reports for analyzing the Luxar codebase.

## Quick Start

Generate the statistics report:

```bash
make stats
```

Or run directly:

```bash
python3 stats/generate_stats.py
```

Then view the report:

```bash
open stats/project_stats.html
```

## Files

- **`generate_stats.py`** - Python script that analyzes the codebase and generates statistics
- **`project_stats.html`** - Generated HTML report with comprehensive project statistics

## What's Analyzed

The statistics analyzer examines the entire Luxar codebase and provides:

### Code Metrics
- **Lines of code** (executable code only)
- **Total lines** (including comments and blanks)
- **File counts** (Python and TypeScript)
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

The analyzer excludes:
- `node_modules/` - Node.js dependencies
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `coverage/` - Coverage reports
- `dist/` - Distribution builds
- `build/` - Build artifacts
- `.git/` - Git repository data

## Technical Details

### Analysis Method

The analyzer:
1. **Recursively scans** Python (`.py`) and TypeScript (`.ts`, `.tsx`) files
2. **Counts lines** by category (code, comments, blank)
3. **Parses definitions** using regex patterns for classes, functions, interfaces
4. **Categorizes by module** for detailed breakdowns
5. **Generates HTML** with embedded CSS for portable viewing

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

Typical statistics for Luxar:
- **~290 files** (Python + TypeScript)
- **~58,000 lines of code**
- **~73% Python**, ~27% TypeScript
- **~288 classes** total
- **~535 functions** total

See `project_stats.html` for the full interactive report!
