# Claude Code Instructions for the Luxar Project

> **Important**: This file contains essential information for Claude Code instances working on the Luxar project. 
> Keep this file updated with new learnings, conventions, and important project details while maintaining clarity and conciseness.
> If you discover something that would be useful for future Claude instances, add it here.

## Project-Specific Tools and Preferences

### Python Development
- **ALWAYS use Hatch** for Python tasks when possible:
  - Running tests: `hatch run test` or `hatch run pytest`
  - Running tests with coverage: `hatch run test-cov`
  - Building: `hatch build`
  - Installing dependencies: Dependencies are managed in `pyproject.toml`
  - Python version management: Hatch handles this automatically

### Testing
- Always run tests with: `hatch run test`
- For coverage reports: `hatch run test-cov`
- View coverage HTML report: `open htmlcov/index.html`
- Minimum acceptable coverage: 80% (currently at 95.69%)
- Run all tests (Python + TypeScript): `make test-all`
- Note: TypeScript dependencies will be auto-installed if missing

### Code Style
- Follow existing code patterns in the codebase
- Use type hints for all function parameters and return values
- Use arbol's `aprint` instead of `print` for console output in examples and CLI tools
- Keep docstrings concise but informative

### Git Workflow
- Never commit `.zarr` directories (they're now in .gitignore)
- Always run tests before committing
- Use descriptive commit messages
- Do not include the robot emoji and Claude Code attribution in commits
- **Pre-commit Checklist**: Before committing, ensure overall consistency:
  - Run all tests (`make test-all`) and ensure they pass
  - Update README.md files if functionality changed
  - Update examples if APIs changed
  - Check that documentation reflects the current state
  - Verify that new features have appropriate tests
  - Run linting and type checking (`make check`)

### Project Structure
- `/packages/luxar/` - Main Python package
- `/packages/luxar-player/` - TypeScript/WebGL viewer
- `/examples/` - Example scripts (use `*_example.py` naming convention)
- Python tests go in `/packages/luxar/src/luxar/tests/`

### Luxar-Specific Conventions
- Physical units: Be inclusive (support nm, um, mm, cm, m, meter, metre, km, inch, foot, px, au)
- Point attributes: positions (required), colors, radii, sharpness (all optional)
- Default values: radius=0.1, sharpness=2.0
- Zarr chunks: Use appropriate chunk sizes for data patterns
- **Transform System**: 
  - All transforms are 4x4 matrices (float32)
  - Transforms are automatically validated in Node.__init__
  - Use `luxar.transforms` module for creating transforms (translate, rotate, scale, compose, etc.)
  - Node class has a `transform` property for easy access/modification
  - Transforms are stored as 16-element lists in zarr attributes
- **Scene-Level Dimensions**:
  - Use `Dimensions` and `Dimension` classes to define coordinate systems
  - Dimensions include: name, unit, range, step, display status
  - Scene validates all objects against defined dimensions
  - Step sizes are used for keyboard navigation in viewer
- **nD Point Cloud Support**:
  - Points can have arbitrary dimensions (not just 3D)
  - Non-displayed dimensions are "sliced" for visualization
  - Radius-based slicing: points visible based on nD hypersphere intersection
  - Keyboard navigation: Press 1-9 to select dimension, [/] to navigate

### TypeScript/JavaScript
- Use pnpm for the luxar-player package (NOT npm)
- Development server: `pnpm dev`
- Build: `pnpm build`
- Tests: `pnpm test --run` (use --run for non-interactive mode)
- Coverage: `pnpm run test:coverage`
- **Configuration**: Unified configuration system in `packages/luxar-player/src/config/`
  - All config in `config/index.ts` with types in `config/types.ts`
  - Use camelCase consistently (not UPPER_SNAKE_CASE)
  - Advanced rendering controls panel should be on the left side
  - Trigger animation when rendering parameters change
- **Quality Checks**: After making changes, run:
  - `pnpm run lint` - Check code style
  - `pnpm run typecheck` - Check TypeScript types

## Quick Commands Reference
```bash
# Python/Hatch
hatch run test                    # Run tests
hatch run test-cov               # Run tests with coverage
hatch run python script.py       # Run a Python script in the Hatch environment

# Luxar CLI
luxar serve <data.zarr>          # Serve zarr data
luxar build <scene.py>           # Build a scene

# Development
cd packages/luxar-player && pnpm dev         # Start viewer dev server
make viewer-test                             # Run TypeScript tests
make test-all                                # Run all tests (Python + TypeScript)
make clean                                   # Clean all artifacts (including TypeScript dist/, node_modules/)
```

## Recent Updates and Learnings

### nD Visualization Implementation (Latest)
- **Slicing Tolerance**: Use point radius for visibility, not fixed tolerance
- **Scene Dimensions**: Always define at scene level for consistency
- **Keyboard Navigation**: Simple 2-step: select dimension (1-9), navigate ([/])
- **TypeScript Integration**: Scene dimensions loaded from zarr attrs, used for step sizes
- **Examples**: Keep nD examples simple with clear shapes/patterns

### Code Quality Checklist
When making significant changes:
1. Run Python tests: `hatch run test-cov` (coverage must be >80%)
2. Run TypeScript build: `cd packages/luxar-player && pnpm build`
3. Check Python linting: `hatch run python -m ruff check .`
4. Fix TypeScript unused warnings by prefixing with underscore
5. Update relevant documentation (README.md, API docs, docstrings)
6. Add/update examples if introducing new features
7. Run integration tests on all examples
8. Update this CLAUDE.md file with important learnings

## Important Reminders
1. Check for existing implementations before writing new code
2. Validate all inputs according to the type system in `types.py`
3. Keep examples simple and well-documented
4. Test edge cases, especially for validation functions
5. Do not use unittest or mocking
6. Use `arbol` for console output in examples and CLI tools
7. When implementing new features, avoid over-engineering - "Complete before you perfect"
8. This is still an early-stage project, don't bother about backwards compatibility, deprecation or migration guides.
9. When running test 'by-hand', or doing  experiments that generate files, put these files in a 'delme' directory, so that they can be easily cleaned up later.