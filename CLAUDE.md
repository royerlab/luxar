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
- Use pnpm for the luxar-player package (NOT npm)
- Development server: `pnpm dev`
- Build: `pnpm build`
- Tests: `pnpm test --run` (use --run for non-interactive mode)
- Coverage: `pnpm run test:coverage`
- Type checking: `pnpm run typecheck`
- Linting: `pnpm run lint`
- Formatting: `pnpm run format`

Note: When possible, use `make` commands for convenience (see below).

## Project Structure

### Main Directories
- `/packages/luxar/` - Main Python package
  - `/packages/luxar/src/luxar/` - Source code
  - `/packages/luxar/src/luxar/tests/` - Python tests
- `/packages/luxar-player/` - TypeScript/WebGL viewer
  - `/packages/luxar-player/src/` - TypeScript source
  - `/packages/luxar-player/src/config/` - Unified configuration
- `/examples/` - Example scripts (use `*_example.py` naming convention)

### Key Documentation Files
- `README.md` - Main project documentation
- `LUXAR_ZARR_FORMAT.md` - Data format specification
- `CONTRIBUTING.md` - Contributing guidelines
- `pyproject.toml` - Python project configuration (dependencies, tools)
- `Makefile` - Convenient development commands
Important Note: Keep these documentation files synced as you make changes to the code!

## Development Workflow

### Testing Strategy
- Always run Python tests with: `hatch run test`
- For Python test coverage reports: `hatch run test-cov`
- View coverage HTML report: `open htmlcov/index.html`
- Minimum acceptable coverage: 80%
- Run all tests (Python + TypeScript): `make test-all` (Note: TypeScript dependencies will be auto-installed if missing)
- Single test file: `hatch run pytest packages/luxar/src/luxar/tests/test_specific.py`

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
- Use arbol's `aprint` instead of `print` for console output in examples and CLI tools
- Keep docstrings concise but informative
- TypeScript code comments should be in JSDoc format
- Python code is formatted with ruff (88 char line length)
- TypeScript code is formatted with prettier

### TypeScript Configuration
- **Configuration**: Unified configuration system in `packages/luxar-player/src/config/`
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
luxar serve <data.zarr>          # Serve zarr data (default port 8000)
luxar build <scene.py>           # Build a scene from Python script
luxar random --out demo.zarr --n 100000  # Generate random demo dataset
luxar info <data.zarr>           # Display dataset information
```

### Development Commands
```bash
cd packages/luxar-player && pnpm dev         # Start viewer dev server (port 5173)
make viewer-test                             # Run TypeScript tests
make test-all                                # Run all tests (Python + TypeScript)
make clean                                   # Clean all artifacts (including TypeScript dist/, node_modules/)
make check                                   # Run all quality checks
make format-all                              # Format all code (Python + TypeScript)
make run-examples                            # Generate all example datasets
make serve-examples                          # Serve examples directory
make demo-and-serve                          # Create demo and start servers
```

**Important Note**: Always ensure that you are at the root of the project directory when running `make` commands.

## Quality Assurance

### Code Quality Checklist
When making significant changes:
1. Run Python tests: `hatch run test-cov` (coverage must be >80%)
2. Run TypeScript build: `cd packages/luxar-player && pnpm build` (check current folder first!)
3. Check Python linting: `hatch run python -m ruff check .`
4. Check TypeScript: `pnpm run typecheck` and `pnpm run lint`
5. Fix TypeScript unused warnings by prefixing with underscore
6. Update relevant documentation (README.md, API docs, docstrings)
7. Update LUXAR_ZARR_FORMAT.md if adding new data fields or changing the format
8. Add/update examples if introducing new features
9. Run integration tests on all examples
10. Update this CLAUDE.md file with important learnings

### TypeScript Quality Checks
After making changes in luxar-player, run:
- `pnpm run lint` - Check code style
- `pnpm run typecheck` - Check TypeScript types
- `pnpm run format` - Auto-fix formatting
- `pnpm run check` - Run all checks (typecheck, lint, test)

## Important Reminders

### Development Best Practices
1. Check for existing implementations before writing new code
2. Validate all inputs according to the type system in `types.py`
3. Keep examples simple, well-documented, and following the same standard of file naming, style and operation as existing examples
4. Test edge cases, especially for validation functions
5. Do not use unittest, instead use PyTest. Using mocking only as a last resort
6. Use `arbol` for console output in examples and CLI tools
7. When implementing new features, avoid over-engineering - "Complete before you perfect"
8. This is still an early-stage project, don't bother about backwards compatibility, deprecation or migration guides
9. When running test 'by-hand', or doing experiments that generate files, put these files in a 'delme' directory, so that they can be easily cleaned up later
10. Example/test datasets should always be named: 'something_something_example(.py|.zarr)' (e.g., 'test_4d_rainbow_sphere_example.zarr')
11. Resulting zarr datasets from examples can be left in the examples folder - no need to copy them elsewhere
12. Do not try to run the viewer yourself - ask the user to run it and request console output if needed

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
- **nD Slicing Fix**: Updated `dims-navigation.ts` to use `sliceColorsFloat32()` for proper HDR colors
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
- **Multiple blending modes**: Support different blending modes (additive, normal, multiply) per layer/object
- **Beyond points**: Support for meshes, lines, volumes, and other geometry types
- **Material system**: More sophisticated materials with different shading models
- **Level of Detail (LOD)**: Automatic LOD for massive datasets
- **Streaming**: Progressive loading and culling for TB-scale data