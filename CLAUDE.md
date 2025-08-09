markdown
# Claude Code Instructions for the Luxar Project

> **Important**: This file contains essential information for Claude Code instances working on the Luxar project. 
> Keep this file updated with new learnings, conventions, and important project details while maintaining clarity and conciseness.
> If you discover something that would be useful for future Claude instances, add it here.

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
- **ALWAYS use Hatch** for Python tasks when possible:
  - Running tests: `hatch run test` or `hatch run pytest`
  - Running tests with coverage: `hatch run test-cov`
  - Building: `hatch build`
  - Installing dependencies: Dependencies are managed in `pyproject.toml`
  - Python version management: Hatch handles this automatically

### TypeScript/JavaScript Development
- Use pnpm for the luxar-player package (NOT npm)
- Development server: `pnpm dev`
- Build: `pnpm build`
- Tests: `pnpm test --run` (use --run for non-interactive mode)
- Coverage: `pnpm run test:coverage`

## Project Structure
- `/packages/luxar/` - Main Python package
- `/packages/luxar-player/` - TypeScript/WebGL viewer
- `/examples/` - Example scripts (use `*_example.py` naming convention)
- Python tests go in `/packages/luxar/src/luxar/tests/`

## Development Workflow

### Testing Strategy
- Always run tests with: `hatch run test`
- For coverage reports: `hatch run test-cov`
- View coverage HTML report: `open htmlcov/index.html`
- Minimum acceptable coverage: 80% (currently at 95.69%)
- Run all tests (Python + TypeScript): `make test-all`
- Note: TypeScript dependencies will be auto-installed if missing

### Git Workflow
- Never commit `.zarr` directories (they're now in .gitignore)
- Always run tests before committing
- Use descriptive commit messages
- Do not include the robot emoji and Claude Code attribution in commits

### Pre-commit Checklist
Before committing, ensure overall consistency:
- Run all tests (`make test-all`) and ensure they pass
- Update README.md files if functionality changed
- Update examples if APIs changed
- Check that documentation reflects the current state
- Update LUXAR_ZARR_FORMAT.md if the data format changes
- Verify that new features have appropriate tests
- Run linting and type checking (`make check`)

## Code Standards

### General Code Style
- Follow existing code patterns in the codebase
- Use type hints for all function parameters and return values
- Use arbol's `aprint` instead of `print` for console output in examples and CLI tools
- Keep docstrings concise but informative

### TypeScript Configuration
- **Configuration**: Unified configuration system in `packages/luxar-player/src/config/`
- All config in `config/index.ts` with types in `config/types.ts`
- Use camelCase consistently (not UPPER_SNAKE_CASE)
- Advanced rendering controls panel should be on the left side
- Trigger animation when rendering parameters change

## Luxar-Specific Conventions

### Physical Units and Data
- Physical units: Be inclusive (support nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au)
- Point attributes: positions (required), colors, radii, sharpness (all optional)
- Default values: radius=0.1, sharpness=2.0
- Zarr chunks: Use appropriate chunk sizes for data patterns

### Transform System
- All transforms are 4x4 matrices (float32)
- Transforms are automatically validated in Node.__init__
- Use `luxar.transforms` module for creating transforms (translate, rotate, scale, compose, etc.)
- Node class has a `transform` property for easy access/modification
- Transforms are stored as 16-element lists in zarr attributes

### Scene-Level Dimensions
- Use `Dimensions` and `Dimension` classes to define coordinate systems
- Dimensions include: name, unit, range, step, display status
- Scene validates all objects against defined dimensions
- Step sizes are used for keyboard navigation in viewer

### nD Point Cloud Support
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
```

### Luxar CLI Commands
```bash
luxar serve <data.zarr>          # Serve zarr data
luxar build <scene.py>           # Build a scene
```

### Development Commands
```bash
cd packages/luxar-player && pnpm dev         # Start viewer dev server
make viewer-test                             # Run TypeScript tests
make test-all                                # Run all tests (Python + TypeScript)
make clean                                   # Clean all artifacts (including TypeScript dist/, node_modules/)
```

**Important Note**: Always ensure that you are at the root of the project directory when running these commands, especially for `make` commands.

## Quality Assurance

### Code Quality Checklist
When making significant changes:
1. Run Python tests: `hatch run test-cov` (coverage must be >80%)
2. Run TypeScript build: `cd packages/luxar-player && pnpm build`
3. Check Python linting: `hatch run python -m ruff check .`
4. Fix TypeScript unused warnings by prefixing with underscore
5. Update relevant documentation (README.md, API docs, docstrings)
6. Update LUXAR_ZARR_FORMAT.md if adding new data fields or changing the format
7. Add/update examples if introducing new features
8. Run integration tests on all examples
9. Update this CLAUDE.md file with important learnings

### TypeScript Quality Checks
After making changes, run:
- `pnpm run lint` - Check code style
- `pnpm run typecheck` - Check TypeScript types

## Important Reminders

### Development Best Practices
1. Check for existing implementations before writing new code
2. Validate all inputs according to the type system in `types.py`
3. Keep examples simple and well-documented
4. Test edge cases, especially for validation functions
5. Do not use unittest or mocking
6. Use `arbol` for console output in examples and CLI tools
7. When implementing new features, avoid over-engineering - "Complete before you perfect"
8. This is still an early-stage project, don't bother about backwards compatibility, deprecation or migration guides
9. When running test 'by-hand', or doing experiments that generate files, put these files in a 'delme' directory, so that they can be easily cleaned up later
10. Example/test datasets should always be named: 'something_something_example' (e.g., 'test_4d_rainbow_sphere_example.zarr')
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

#### nD Visualization Implementation
- **Slicing Tolerance**: Use point radius for visibility, not fixed tolerance
- **Scene Dimensions**: Always define at scene level for consistency
- **Keyboard Navigation**: Simple 2-step: select dimension (1-9), navigate ([/])
- **TypeScript Integration**: Scene dimensions loaded from zarr attrs, used for step sizes
- **Examples**: Keep nD examples simple with clear shapes/patterns

### Broader Vision and Known Issues

#### Future Extensions
- **Multiple blending modes**: Support different blending modes (additive, normal, multiply) per layer/object
- **Beyond points**: Support for meshes, lines, volumes, and other geometry types
- **Material system**: More sophisticated materials with different shading models

#### Anti-Aliasing Brightness Issues

##### SSAA Brightness Issue
**Problem**: When using SSAA (Supersampling Anti-Aliasing), the scene gets dimmer with higher multipliers. This is because:
- Points currently use additive blending
- Higher resolution = more pixels per point
- Downsampling averages the contributions, reducing brightness

##### MSAA Brightness Issue
**Problem**: When using MSAA (Multisample Anti-Aliasing), the scene gets BRIGHTER with more samples. This is because:
- Points use additive blending (THREE.AdditiveBlending)
- Each MSAA sample accumulates the additive contribution
- More samples = more accumulation = brighter result
- This is a fundamental incompatibility between MSAA and additive blending

**Important**: Do NOT compensate for this in individual shaders! Any solution must:
1. Work for all blending modes (not just additive)
2. Work for all geometry types (not just points)
3. Not interfere with post-processing effects like bloom

**Potential solutions to explore**:
- Use normal alpha blending instead of additive (but loses HDR glow effect)
- Custom resolve shader for MSAA that accounts for blend mode
- Render additive objects to separate buffer without MSAA
- Post-process brightness normalization based on MSAA sample count
```