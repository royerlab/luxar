# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Table of Contents
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Luxar-Specific Conventions](#luxar-specific-conventions)
- [Quick Commands Reference](#quick-commands-reference)
- [Quality Assurance](#quality-assurance)
- [Important Reminders](#important-reminders)
- [Technical Documentation](#technical-documentation)

## Development Environment

### Python Development with Hatch
- **ALWAYS use Hatch** for Python tasks when possible or the corresponding `make` command:
  - Running tests: `hatch run test` or `hatch run pytest`
  - Running tests with coverage: `hatch run test-cov`
  - Building: `hatch build`
  - Installing dependencies: Dependencies are managed in `pyproject.toml`
  - Python version management: Hatch handles this automatically
  - Running scripts: `hatch run python script.py`
  - Linting: `hatch run python -m ruff check .`
  - Type checking: `hatch run mypy packages/luxar/src/luxar/`

### TypeScript/JavaScript Development
- Use pnpm for the luxar-viewer package (NOT npm)
- Development server: `pnpm dev`
- Build: `pnpm build`
- Unit tests: `pnpm test --run` (use --run for non-interactive mode)
- E2E tests: `pnpm test:e2e` (Playwright tests)
- Coverage: `pnpm run test:coverage`
- Type checking: `pnpm run typecheck`
- Linting: `pnpm run lint`
- Formatting: `pnpm run format`

#### Playwright Testing & AI-Assisted Debugging
The luxar-viewer has comprehensive Playwright integration for E2E testing and AI-assisted debugging:

**AI Debugging (for Claude Code)**:
```bash
cd packages/luxar-viewer
pnpm agent:debug              # Headless mode - shows browser console logs
pnpm agent:debug:visible      # Visible browser - watch it run
```

This allows Claude Code to:
- See all browser console logs in terminal
- Inspect Three.js scene state via JSON output
- Take screenshots for visual verification
- Debug issues autonomously without asking user to check browser

**E2E Testing**:
```bash
pnpm test:e2e                 # Run all E2E tests
pnpm test:e2e:ui              # Interactive test UI
pnpm test:e2e:debug           # Debug mode
pnpm test:e2e:report          # View HTML report (after running tests)
```

**E2E Test Artifacts** (Transient, Not Committed):

**Playwright automatically generates**:
- `test-results/` - Per-test artifacts (screenshots, videos, traces, error context)
- `playwright-report/` - Interactive HTML report with all results

**Viewing Test Results**:
```bash
# After running tests, view the HTML report:
pnpm test:e2e:report

# This opens an interactive report showing:
# - All test results (pass/fail)
# - Screenshots for every test
# - Videos of failures
# - Trace viewer for debugging
# - Timings and performance
```

**For Claude (AI Debugging)**:
After running E2E tests, Claude can:
- Read screenshots from `test-results/*/test-failed-*.png`
- Inspect error-context.md files
- Review visual state of failed tests
- Verify rendering correctness

**For Users**:
- View `playwright-report/` HTML (recommended)
- Or browse `test-results/` folders directly
- Each test gets its own folder with complete artifacts

**Important**: Always use `?debug` URL parameter to enable the debug interface:
- `http://localhost:5173/?debug` - Exposes `window.__luxarDebug`
- Contains: scene, camera, renderer, controls, getState(), renderOnce(), etc.

See [PLAYWRIGHT_GUIDE.md](packages/luxar-viewer/docs/PLAYWRIGHT_GUIDE.md) for complete documentation.

Note: When possible, use `make` commands for convenience (see below).

## Project Structure

### Main Directories
- `/packages/luxar/` - Main Python package
  - `/packages/luxar/src/luxar/` - Source code (organized into subpackages)
    - `core/` - Core data structures (Node, Scene, Points, Dimensions, Transforms)
    - `io/` - Input/output operations (Compiler, Streaming, Writer)
    - `typing_utils/` - Type definitions (Protocols, Aliases, Enums, Constants)
    - `validation/` - Validation functions
    - `utils/` - Utility functions (Array helpers, Demo generators)
  - `/packages/luxar/src/luxar/tests/` - Python tests
- `/packages/luxar-viewer/` - TypeScript/WebGL viewer
  - `/packages/luxar-viewer/src/` - TypeScript source with per-package documentation
  - Each TypeScript package has its own README.md that MUST be kept in sync with code:
    - `/src/controls/README.md` - Control system (orbit, fly, input management)
    - `/src/rendering/README.md` - HDR rendering pipeline and post-processing
    - `/src/scene/README.md` - Scene management and animation control
    - `/src/data/README.md` - Zarr loading and nD slicing
    - `/src/ui/README.md` - UI components and layouts
    - `/src/input/README.md` - Input handling and context management
    - `/src/config/README.md` - Unified configuration system
    - `/src/utils/README.md` - Utility functions and helpers
    - `/src/types/README.md` - TypeScript type definitions
    - `/src/core/README.md` - Core initialization and app lifecycle
- `/packages/luxar/examples/` - Example scripts (use `*_example.py` naming convention)
- `/docs/` - **Documentation directory (IMPORTANT: Keep this up-to-date!)**
  - Contains various documentation files that were moved from root
  - Must be maintained in sync with code changes
  - Includes technical guides, format specs, and development docs

### Key Documentation Files
- `README.md` - Main project documentation
- `CLAUDE.md` - This file - guidance for Claude Code
- `pyproject.toml` - Python project configuration (dependencies, tools)
- `Makefile` - Convenient development commands
- `/docs/` folder containing:
  - `LUXAR_ZARR_FORMAT.md` - Data format specification
  - `CONTRIBUTING.md` - Contributing guidelines
  - `DEVELOPMENT_TOOLS.md` - Development and build tools documentation
  - `UI_DESIGN.md` - **UI design system and guidelines (MUST READ for UI work)**
  - `CONSOLE_OUTPUT_STYLE.md` - **Console logging style guide and standards**
  - `luxar-fly-controls-guide.md` - Detailed fly controls implementation guide
  - Various technical guides and specifications
- **TypeScript package READMEs**: Each package in `/packages/luxar-viewer/src/` has its own comprehensive README.md
- **Python package READMEs**: **MANDATORY** - Each subpackage in `/packages/luxar/src/luxar/` MUST have its own comprehensive README.md documenting:
  - Purpose and responsibilities of the package
  - Key classes and functions
  - Usage examples
  - Internal architecture notes
  - Dependencies and requirements
  - Testing information
- **Python package SPECIFICATIONS.md**: **MANDATORY** - Each major subpackage in `/packages/luxar/src/luxar/` MUST have a SPECIFICATIONS.md file that:
  - Defines the essential logic, algorithms, and data structures
  - Specifies behavior independent of implementation details
  - Documents mathematical formulas and key algorithms
  - Provides enough detail to re-implement if code was lost
  - Focuses on WHAT and WHY, not HOW
  - Serves as the authoritative specification for the package
  - Should be implementation-agnostic (could re-implement in another language)
  - **MUST include version and changelog** (see format below)

**SPECIFICATIONS.md Format Requirements**:
```markdown
# luxar.{package} - Technical Specification

**Version**: X.Y.Z
**Last Updated**: YYYY-MM-DD

## Purpose
{Brief description of the package's purpose}

---

{... specification content ...}

---

## Changelog

- **vX.Y.Z** (YYYY-MM-DD): {Summary of changes}
  - {Detail 1}
  - {Detail 2}
  - **BREAKING**: {Breaking change if any}

- **vX.Y.Z-1** (YYYY-MM-DD): {Previous version changes}
  ...
```

**Version numbering**: Use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes to the specification
- MINOR: New features or sections added
- PATCH: Clarifications, typo fixes, minor updates

**Cross-Reference Format**:
When referencing other specifications, use this standard format:
```markdown
**Related Specifications**:
- `luxar.package` - Brief description (see `relative/path/SPECIFICATIONS.md`)
```

Examples from different package locations:
- From root package (e.g., `core/`): `(see `encoding/SPECIFICATIONS.md`)`
- From sub-package (e.g., `gsplats/io/`): `(see `../../encoding/SPECIFICATIONS.md`)`
- From sibling: `(see `../other/SPECIFICATIONS.md`)`
- To parent: `(see `../SPECIFICATIONS.md`)`

Use **relative paths** from the specification file's location.

**CRITICAL**:
1. When making changes to TypeScript code, ALWAYS update the corresponding package README.md
2. When making changes to Python code:
   - **MANDATORY**: Update the subpackage README.md if functionality changes
   - **MANDATORY**: Update the subpackage SPECIFICATIONS.md if algorithms or core logic changes
   - **MANDATORY**: Ensure all major Python subpackages (core, io, utils, cli, gsplats, typing_utils, validation) have comprehensive README.md AND SPECIFICATIONS.md files
   - Check if `/docs/` folder documentation needs updating
3. Keep all documentation synchronized with the implementation!
4. SPECIFICATIONS.md should capture the essence that would let someone re-implement from scratch
5. Python package structure follows best practices:
   - Flat is better than nested (except for logical groupings)
   - Each package has clear separation of concerns
   - Backward compatibility maintained via main `__init__.py`
   - Every major subpackage MUST have a README.md file

## Development Workflow

### Testing Strategy
- Always run Python tests with: `hatch run test`
- For Python test coverage reports: `hatch run test-cov`
- View Python coverage HTML report: `open coverage/python/htmlcov/index.html`
- View TypeScript coverage HTML report: `open coverage/typescript/index.html`
- Minimum acceptable coverage: 80%
- Run all tests (Python + TypeScript): `make test-all` (Note: TypeScript dependencies will be auto-installed if missing)
- Single test file: `hatch run pytest packages/luxar/src/luxar/tests/test_specific.py`
- **CRITICAL**: NEVER skip tests just because they're difficult to fix. If a test is failing:
  1. First, try to fix the underlying issue
  2. If mocking is needed (e.g., WebGL), create proper mocks
  3. If absolutely impossible to test (rare), document WHY in detail
  4. Skipping tests without fixing them is unacceptable and defeats the purpose of testing

### Cross-Language End-to-End (E2E) Testing

**IMPORTANT**: End-to-end testing that exercises both Python and TypeScript code together is crucial for this project. The Python encoder and TypeScript decoder must stay in sync - bugs in cross-language compatibility (like the array_ref resolution issue) can only be caught through E2E tests.

**Why E2E Testing Matters**:
- Python writes zarr data with specific encoding metadata
- TypeScript reads and decodes that data
- Unit tests in isolation can't catch mismatches between the two
- Changes to Python encoding can silently break TypeScript decoding

**Two Approaches to Cross-Language E2E Testing**:

1. **Python → TypeScript Unit Tests (No Browser)**:
   - Python generates test fixtures (zarr datasets with specific encodings)
   - TypeScript unit tests load and verify these fixtures
   - Example: `packages/luxar-viewer/tests/fixtures/` contains zarr datasets generated by `generate_test_data.py`
   - Tests run with `pnpm test --run` in Node.js (no browser needed)
   - Fast feedback loop, easy to debug
   - **Use this for**: Data format compatibility, encoding/decoding correctness

2. **Playwright-Based E2E Testing (Full Browser)**:
   - Tests the complete pipeline: Python data → TypeScript loading → WebGL rendering
   - **CRITICAL**: This is the only way to test browser-specific behavior (WebGL, Three.js, canvas rendering)
   - Many bugs only manifest in the browser and cannot be caught by Node.js tests
   - Run with: `cd packages/luxar-viewer && pnpm test:e2e`
   - **Use this for**: Visual rendering, WebGL shaders, UI interactions, browser APIs

**When to Write E2E Tests**:
- After changing Python encoding format → verify TypeScript can still decode
- After changing TypeScript decoder → verify it handles all Python formats
- After adding new encoding modes → test full round-trip
- After visual/rendering changes → Playwright visual regression tests

### Git Workflow
- Never commit `.zarr` directories (they're in .gitignore)
- Always run all tests (Python & TypeScript), run all checks (typing, linting), and format code before committing
- Use detailed descriptive commit messages
- Pre-commit hooks are configured - install with: `hatch run pre-commit install`

### Pre-commit Checklist
Before committing, ensure overall consistency:
- Run all tests (`make test-all`) and ensure they pass
- Run linting and type checking (`make check`)
- Update README.md files if functionality changed
- Update examples if APIs have changed
- Check that documentation reflects the current state
- Update LUXAR_ZARR_FORMAT.md if the data format changes
- Update DEVELOPMENT_TOOLS.md if dev and build tools change
- Verify that new features have appropriate tests

## Code Standards

### General Code Style
- Follow modern Python and TypeScript conventions
- Follow existing code patterns in the codebase when all else is equal
- Use type hints for all function parameters and return values
- **ALWAYS use Arbol for Python console output** - see [Arbol Usage](#arbol-usage) section below
- Keep docstrings concise but informative
- TypeScript code comments should be in JSDoc format
- Python code is formatted with ruff (88 char line length)
- TypeScript code is formatted with prettier
- **Console logging**: Follow the style guide in `/docs/CONSOLE_OUTPUT_STYLE.md`
  - Use the logging utility in `/src/utils/log.ts`
  - Format: `[emoji] [Module] message`
  - Import: `import { log, Modules, LogEmoji } from '../utils/log';`

### Arbol Usage
**Arbol** is a Python library for organizing print statements in hierarchical, tree-like structures that makes console output readable and well-structured. It is MANDATORY for all Python console output in Luxar.

#### What is Arbol?
- A lightweight library that replaces `print()` with structured, hierarchical output
- Provides automatic elapsed time measurement for code sections
- Offers tree-like visualization of code execution flow
- Designed to make complex scripts with many print statements comprehensible

#### Key Components:
- **`aprint()`**: Direct replacement for `print()` that integrates with the tree structure
- **`asection(context)`**: Context manager that creates hierarchical sections in the output
- **Automatic timing**: Each section shows elapsed time
- **Optional colors**: Enhanced visual clarity with color packages

#### When to Use:
- **aprint()**: Replace ALL `print()` statements with `aprint()` in:
  - Examples and demo scripts
  - CLI tools and commands
  - Test files (where console output is needed)
  - Debug and development scripts
- **asection()**: Use for logical code sections where multiple operations occur:
  - Complex functions with multiple steps
  - Processing loops with substantial work
  - File I/O operations
  - Model training or data processing phases
  - Any code block where flat logging would be hard to follow

#### Usage Examples:
```python
from arbol import aprint, asection

# Simple replacement for print
aprint("Loading dataset...")

# Hierarchical sections for complex operations
with asection("Data preprocessing"):
    aprint("Reading input files...")
    with asection("Validation"):
        aprint("Checking data integrity")
        aprint("Validating dimensions")
    aprint("Preprocessing complete")

with asection("Model training"):
    aprint("Initializing model...")
    # ... training code ...
```

#### Configuration:
- Set `Arbol.max_depth = 4` to limit tree depth
- Use `Arbol.elapsed_time = True` for timing (default)
- Install `ansicolors` or `colorama` for colored output
- Configure globally in main entry points

### TypeScript Configuration
- **Configuration**: Unified configuration system in `packages/luxar-viewer/src/config/`
- All config in `config/index.ts` with types in `config/types.ts`
- Use camelCase consistently (not UPPER_SNAKE_CASE)
- Prefix unused variables with underscore to avoid warnings

## Luxar-Specific Conventions

### Physical Units and Data
- Physical units: Be inclusive (support nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au)
- Point attributes: positions (required), colors, radii, sharpness (all optional)
- Zarr chunks: Use appropriate chunk sizes for data patterns (default 32KB elements)
- Data types: Float32 for positions/radii/sharpness, Uint8 or Float32 for colors (HDR support)

### Transform System
- All transforms are 4x4 matrices (float32)
- Transforms are automatically validated in Node.__init__
- Use `luxar.transforms` module for creating transforms (translate, rotate, scale, compose, etc.)
- Node class has a `transform` property for easy access/modification
- Transforms are stored as 16-element lists in zarr attributes
- **CRITICAL**: Transpose matrices when storing for THREE.js compatibility (see Technical Documentation)

### Scene-Level Dimensions
- Use `Dimensions` and `Dimension` classes to define coordinate systems
- Dimensions include: name, unit, range, step, display status
- Scene validates all objects against defined dimensions
- Step sizes are used for keyboard navigation in viewer

### nD Point Layer Support
- Points can have arbitrary dimensions (not just 3D)
- Non-displayed dimensions are "sliced" for visualization
- Radius-based slicing: points visible based on nD hypersphere intersection
- Keyboard navigation: Press 1-9 to select dimension, [/] to navigate

## Quick Commands Reference

### Python/Hatch Commands
```bash
hatch run test                    # Run tests
hatch run test-cov               # Run tests with coverage
hatch run python script.py       # Run a Python script in the Hatch environment
hatch run python -m ruff check . # Run linting
hatch run mypy packages/luxar/src/luxar/ # Type checking
hatch build                      # Build distribution packages
```

### Luxar CLI Commands
```bash
luxar demo                       # Quick demo with viewer (auto-opens browser)
luxar demo --no-serve --output demo.zarr --points 100000  # Generate demo dataset without serving
luxar serve <data.zarr>          # Serve zarr data (default port 8000)
luxar serve <data.zarr> --viewer # Serve data with viewer
luxar viewer --data <data.zarr>  # Serve viewer with data
luxar info <data.zarr>           # Display dataset information
luxar info <data.zarr> --stats   # Display with detailed statistics
```

### Development Commands
```bash
cd packages/luxar-viewer && pnpm dev         # Start viewer dev server (port 5173)
make viewer-test                             # Run TypeScript tests
make test-all                                # Run all tests (Python + TypeScript)
make clean                                   # Clean all artifacts (including TypeScript dist/, node_modules/)
make check                                   # Run all quality checks
make format-all                              # Format all code (Python + TypeScript)
make run-examples                            # Generate all example datasets
make serve-examples                          # Serve examples directory
make demo-and-serve                          # Create demo and start servers
make stats                                   # Generate project statistics report (HTML)
```

**Important Note**: Always ensure that you are at the root of the project directory when running `make` commands.

## Quality Assurance

### Code Quality Checklist
When making significant changes:
1. Run Python tests: `hatch run test-cov` (coverage must be >80%)
2. Run TypeScript unit tests: `cd packages/luxar-viewer && pnpm test`
3. **Run TypeScript E2E tests**: `cd packages/luxar-viewer && pnpm test:e2e` (Playwright)
4. Run TypeScript build: `cd packages/luxar-viewer && pnpm build` (check current folder first!)
5. Check Python linting: `hatch run python -m ruff check .`
6. Check TypeScript: `pnpm run typecheck` and `pnpm run lint`
7. Fix TypeScript unused warnings by prefixing with underscore
8. **Update TypeScript package READMEs**: Each package in `/packages/luxar-viewer/src/` has its own README.md that MUST be updated when code changes
9. **Update documentation in `/docs/` folder** - check ALL relevant docs for Python changes
10. Update root README.md if features or usage changes
11. Update LUXAR_ZARR_FORMAT.md if data structures change
12. Add/update examples if introducing new features
13. Run integration tests on all examples
14. Update this CLAUDE.md file with important learnings

### TypeScript Quality Checks
After making changes in luxar-viewer, run:
- `pnpm run lint` - Check code style
- `pnpm run typecheck` - Check TypeScript types
- `pnpm run format` - Auto-fix formatting
- `pnpm test` - Run unit tests (Vitest)
- `pnpm test:e2e` - Run E2E tests (Playwright)
- `pnpm run check` - Run all checks (typecheck, lint, test)

### AI-Assisted Debugging (IMPORTANT for Claude Code)
When debugging TypeScript/viewer issues, use the Playwright agent driver:

```bash
cd packages/luxar-viewer
pnpm agent:debug
```

This shows:
- `[BROWSER-CONSOLE-*]` - All browser console logs (errors, warnings, info)
- JSON state dump - Three.js scene state, point counts, camera position
- `debug-view.png` - Screenshot of current state

**How to use**:
1. User reports a bug in the viewer
2. Run `pnpm agent:debug` to see browser console output
3. Inspect JSON state to understand what's loaded
4. Add debug logging (`console.log()`) if needed
5. Run again to verify fix
6. Remove debug logging when done

**Available debug properties** (when `?debug` is in URL):
- `window.__luxarDebug.scene` - THREE.Scene object
- `window.__luxarDebug.camera` - Camera object
- `window.__luxarDebug.renderer` - WebGL renderer
- `window.__luxarDebug.getState()` - Current state snapshot
- `window.__luxarDebug.renderOnce()` - Trigger single frame
- `window.__luxarDebug.app` - LuxarApp instance
- `window.__luxarDebug.consoleInterceptor` - Console message history

**Example debugging workflow**:
```bash
# 1. Run agent driver
pnpm agent:debug

# 2. See output:
[BROWSER-CONSOLE-LOG] Query result: 0 cells → 0 ranges → 0 points
# Identifies the issue: no points loaded

# 3. Add debug logging to code
console.log('[DEBUG] Spatial index query:', queryTolerance);

# 4. Run again
pnpm agent:debug
[BROWSER-CONSOLE-LOG] [DEBUG] Spatial index query: [0, 0, 0, 0]
# Found the bug: tolerance is all zeros!

# 5. Fix and verify
# ... make fix ...
pnpm agent:debug
[BROWSER-CONSOLE-LOG] Query result: 50 cells → 10 ranges → 12000 points ✅
```

See [packages/luxar-viewer/PLAYWRIGHT_GUIDE.md](packages/luxar-viewer/docs/PLAYWRIGHT_GUIDE.md) for complete guide.

## Important Reminders

### Development Best Practices
1. Check for existing implementations before writing new code
2. Validate all inputs according to the type system in `types.py`
3. Keep examples simple, well-documented, and following the same standard of file naming, style and operation as existing examples
4. Test edge cases, especially for validation functions
5. Do not use unittest, instead use PyTest. Using mocking only as a last resort
6. Use `arbol` for console output in examples and CLI tools
7. When implementing new features, avoid over-engineering - "Complete before you perfect"
8. This is still an early-stage project, DO NOT BOTHER about backwards compatibility, deprecation or migration guides. If something needs to be changed, just change it and update all relevant documentation and examples. Do not keep old code around just for backwards compatibility.
9. When running test 'by-hand', or doing experiments that generate files, put these files in a 'delme' directory, so that they can be easily cleaned up later
10. Example/test datasets should always be named: 'something_something_example(.py|.zarr)' (e.g., 'test_4d_rainbow_sphere_example.zarr')
11. Resulting zarr datasets from examples can be left in the packages/luxar/examples folder - no need to copy them elsewhere
12. **You CAN now run and debug the viewer autonomously** using `pnpm agent:debug` (Playwright). Use this to verify fixes, inspect state, and debug issues without asking the user to open a browser. See the "AI-Assisted Debugging" section above.

## Technical Documentation

### Critical Compatibility Issues

#### Matrix Storage Order: Python/NumPy vs TypeScript/THREE.js
**IMPORTANT**: Python/NumPy and TypeScript/THREE.js use different matrix storage conventions:

- **Python/NumPy**: Row-major order (C-style)
  - 4x4 matrix flattened as: `[m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33]`
  - Translation components at indices: `[3, 7, 11]` (when flattened)
  
- **TypeScript/THREE.js**: Column-major order (OpenGL-style)
  - 4x4 matrix flattened as: `[m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33]`
  - Translation components at indices: `[12, 13, 14]` (when flattened)

**Solution**: When storing transforms in zarr for THREE.js consumption:
1. Transpose the matrix before flattening: `matrix.T.ravel().tolist()`
2. When reading back in Python, transpose again: `np.array(flat_list).reshape(4, 4).T`

This issue was discovered when hierarchical transforms weren't working - all objects were at origin because THREE.js was reading translation values from the wrong array indices.

### Recent Updates and Learnings

#### Critical Bug Fixes from Code Review (January 2025)
- **Transform Composition Bug**: Fixed critical math error in `compose()` function
  - **Issue**: Matrix multiplication order was reversed (left-multiply instead of right-multiply)
  - **Impact**: `compose(T1, T2, T3)` was applying T3 first instead of T1 first
  - **Fix**: Changed `result = transform @ result` to `result = result @ transform`
  - **Location**: `core/transforms.py:271`
  - **Test Added**: `test_compose_application_order()` with 4 rigorous test cases
  - **Lesson**: Order matters for non-commutative transforms (rotate+translate). Always test with order-sensitive operations.

- **Points Metadata Loss Bug**: Fixed critical constructor initialization order bug
  - **Issue**: `Points.__init__()` set `self._metadata` before calling `super().__init__()`, then `Node.__init__()` overwrote it with empty dict
  - **Impact**: ALL Points objects lost their metadata (has_colors, has_radii, max_radius, etc. all lost)
  - **Fix**: Call `super().__init__()` BEFORE setting `self._metadata` in Points class
  - **Location**: `core/points.py:50-58`
  - **Test Added**: `test_points_metadata_preservation()` with comprehensive checks
  - **Lesson**: When subclass and parent both initialize the same attribute, parent must initialize first

#### Legacy Code Removal (January 2025)
- **Legacy Mode Removed**: Eliminated unused "legacy mode" from Node class (~40 lines dead code)
  - Removed `group` parameter and all `if self._group is not None:` branches
  - Node now only supports progressive writing mode (simpler, clearer)
  - No production code ever used legacy mode

- **Deprecated Parameters Removed**: Cleaned up deprecated API surface
  - Removed `units` parameter from `LuxarZarrCompiler` (use Dimensions instead)
  - Removed `DimensionMetadata` class (use full-featured `Dimension` instead)
  - Removed unused version constants (LEGACY, PREVIOUS, FUTURE)
  - Total: ~160 lines of dead/deprecated code removed

- **Result**: Clean API with one clear way to do everything, zero backward-compatibility baggage

#### Code Quality Improvements (January 2025)
- **Compiler Refactoring**: Reduced `write_points()` from 432 to 117 lines (73% reduction)
  - Extracted 8 focused helper methods with single responsibilities
  - Much easier to test, understand, and maintain

- **Validation Consolidation**: Created `validation/types.py` centralizing all validation
  - Eliminated ~350 lines of duplication between `protocols.py` and `validation/base.py`
  - Clear organization: types.py (basic), base.py (detailed for writing), nd.py (dimensional)

- **Transform Handling Centralized**: Added `read_transform_from_zarr()` companion function
  - Single source of truth for NumPy ↔ THREE.js transform conversion
  - Eliminated ~50 lines of duplicate transpose logic

- **Magic Numbers Extracted**: Created 18 named constants for spatial index tuning
  - All grid sizing heuristics now configurable via constants
  - Self-documenting code with clear intent

- **Performance**: Vectorized HSV→RGB conversion in demos (30-100x faster)

- **Documentation**: Added 80+ inline comments explaining complex algorithms (cell ID calculation, grid shape heuristics, transform composition math)

#### Fullscreen Resize Bug Fix (January 2025)
- **Issue**: Point sizes changed incorrectly on first fullscreen toggle or window resize
- **Root Cause**: Scene initialization didn't call `updateSize()`, causing different behavior on first resize
- **Solution**: Make initialization call `this.updateSize()` in `scene-manager.ts` init() method
- **Lesson**: Ensure initialization and resize paths are identical to avoid first-time-only bugs
- **Testing**: Always test both initial state AND state after first resize/fullscreen

#### Debug Console & Console Logging (January 2025)
- **In-App Debug Console**: Press Ctrl+L to toggle debug console that captures all browser console output
- **Ring Buffer Implementation**: Console interceptor uses 10,000 message ring buffer to prevent memory overflow  
- **Early Message Capture**: Console messages captured from app initialization via early import of interceptor
- **Configuration Constants**: Debug console dimensions and styling moved to `config/debug-console.ts`
- **Standardized Logging**: All console logs use format: `[emoji] [Luxar] message` for consistency
- **Debug Interface**: Debug tools available at `window.__luxarDebug` when `?debug` URL param is present

#### HDR Color Pipeline Changes (January 2025)
- **Float32 Colors**: Changed from Uint8Array to Float32Array for HDR color support
- **nD Slicing Fix**: Updated slicing algorithms to use `sliceColorsFloat32()` for proper HDR colors
- **WebGL Limitation**: Discovered WebGL canvas doesn't support true HDR output (limited to 8-bit)
- **HDR Detection**: Added comprehensive HDR capability detection in `utils/hdr-detection.ts`

#### World-Space Point Sizing (January 2025)
- **Physical Accuracy**: Points now use world-space sizing instead of screen-space
- **Key Property**: Two points with radius r at distance 2r will just touch
- **FOV Independence**: Points maintain physical size regardless of field of view changes
- **Implementation**: Uses angular size calculation in vertex shaders
- **Formula**: `angularSize = 2 * atan(radius/distance)`, then converted to pixels
- **Resolution Handling**: Uses actual framebuffer size (includes devicePixelRatio)

#### nD Visualization Implementation
- **Slicing Tolerance**: Use point radius for visibility, not fixed tolerance
- **Scene Dimensions**: Always define at scene level for consistency
- **Keyboard Navigation**: Simple 2-step: select dimension (1-9), navigate ([/])
- **TypeScript Integration**: Scene dimensions loaded from zarr attrs, used for step sizes
- **Examples**: Keep nD examples simple with clear shapes/patterns

#### Data Loading Architecture Refactor (January 2025)
- **Removed Lazy Loading**: Eliminated LazyDataManager in favor of spatial index-based loading
- **Spatial Index Required**: All datasets now require spatial indices for efficient loading
- **Range-Based Caching**: New RangeCache system for intelligent memory management
- **Improved Monitoring**: Enhanced DataLoadingMonitor with better error handling and disposal
- **Cleaner Architecture**: Removed intermediate abstractions for simpler, more maintainable code

### Architecture Overview

#### Data Flow
```
Python Data → Luxar Core → Zarr Archive → Luxar Player → WebGL → Display
```

#### Scene Graph Structure
- Scene (root) contains Groups and Points
- Groups can contain other Groups and Points (hierarchical)
- Each node has optional transform (4x4 matrix)
- Transforms compose hierarchically (parent → child)
- Points have positions (nD), colors, radii, sharpness

#### Zarr Storage Format
- Chunked, compressed storage for streaming
- Consolidated metadata for fast loading (`.zmetadata`)
- Scene attributes: version, dimensions, units
- Node attributes: type, transform, rendering properties
- Array data: positions, colors, radii, sharpness

### Performance Considerations
- Target: 100K-10M points for smooth interaction
- Chunk size: 32KB-1MB per chunk optimal
- Compression: Blosc with zstd level 3
- Use Float32 for positions/radii, Uint8 or Float32 for colors
- Progressive loading for large datasets

### Broader Vision and Known Issues

#### Future Extensions
- **Multiple blending modes**: Support different blending modes (additive, normal) per layer/object
- **Beyond points**: Support for meshes, lines, volumes, and other geometry types
- **Material system**: More sophisticated materials with different shading models
- **Level of Detail (LOD)**: Automatic LOD for massive datasets
- **Streaming**: Progressive loading and culling for TB-scale data