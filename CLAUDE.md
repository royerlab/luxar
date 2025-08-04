# Claude Code Instructions for the Luxar Project

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
- Minimum acceptable coverage: 90%

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

### TypeScript/JavaScript
- Use pnpm for the luxar-player package (NOT npm)
- Development server: `pnpm dev`
- Build: `pnpm build`
- Tests: `pnpm test --run` (use --run for non-interactive mode)
- Coverage: `pnpm run test:coverage`

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
```

## Important Reminders
1. Check for existing implementations before writing new code
2. Validate all inputs according to the type system in `types.py`
3. Keep examples simple and well-documented
4. Test edge cases, especially for validation functions
5. Do not use unittest or mocking.
5. Use `arbol` for console output in examples and CLI tools