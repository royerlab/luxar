# 🤝 Contributing to Luxar

We welcome contributions! The project uses automated tooling to maintain high code quality.
By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚀 Quick Start for Contributors

```bash
# 1. Fork and clone the repository
git clone <your-fork-url>
cd luxar

# 2. Set up development environment
make setup-dev  # Installs dependencies and pre-commit hooks

# 3. Create feature branch
git checkout -b feature/amazing-feature origin/dev

# 4. Make your changes
# ... edit code ...

# 5. Run quality checks (before committing)
make check-all  # Runs all quality checks
make test-cov-python   # Ensure tests pass with coverage

# 6. Commit (pre-commit hooks run automatically)
git commit -m "Add amazing feature"

# 7. Push and create Pull Request
git push origin feature/amazing-feature
gh pr create --base dev
```

## 🛠️ Development Environment

### Prerequisites

The build system works on **fresh Linux/macOS machines** with minimal pre-installed tools:
- Python 3.12+ (usually pre-installed)
- Git and curl

**Ubuntu/Debian only** (due to PEP 668):
```bash
sudo apt-get install -y pipx && pipx ensurepath && source ~/.bashrc
```

### First-time Setup
```bash
make setup-dev     # Complete development environment (auto-installs Node.js, pnpm, Hatch)
make check-deps    # Verify what's installed
make check-all     # Run all quality checks
```

`make setup-dev` automatically installs (no sudo needed):
- Node.js 22.22+ (the setup installs Node.js 22 LTS by default) via nvm (Linux) or Homebrew (macOS)
- pnpm for TypeScript package management
- Hatch for Python environment management
- All project dependencies

### Daily Development Workflow
```bash
make format-all    # Format code before editing
make check-all     # Run all quality checks
make test-cov-python   # Run tests with coverage
```

### Troubleshooting Setup Issues
```bash
make check-deps              # See what's installed/missing
make clean-setup         # Reset everything and start fresh
```

For detailed build system documentation, see [BUILD_SYSTEM_SPEC.md](docs/guides/developer/BUILD_SYSTEM_SPEC.md).

### Available Commands

| Command | Purpose |
|---------|---------|
| **Setup** | |
| `make setup-dev` | Complete development environment setup (auto-installs dependencies) |
| `make check-deps` | Check what dependencies are installed/missing |
| `make install-rust` | Install Rust + wasm-pack for viewer builds |
| `make clean-setup` | Remove ALL dev tools (for testing fresh setup) |
| **Quality** | |
| `make format-python` | Format Python code with ruff |
| `make format-typescript` | Format TypeScript code with prettier |
| `make format-all` | Format all code (Python + TypeScript) |
| `make lint-python` | Run ruff linting on Python |
| `make lint-typescript` | Run ESLint on TypeScript |
| `make type-check-python` | Run mypy type checking |
| `make type-check-typescript` | Run TypeScript type checking |
| `make security` | Run bandit security scan |
| `make check-all` | Run all quality checks |
| `make check-typescript` | Run all TypeScript checks |
| `make check-rust` | Run Rust type/lint checks |
| **Testing** | |
| `make test-all` | Run all tests (Python + TypeScript + Rust) |
| `make test-python` | Run Python tests only |
| `make test-cov-python` | Run Python tests with coverage report |
| `make test-cov-typescript` | Run TypeScript tests with coverage |
| `make test-viewer` | Run TypeScript unit tests |
| `make test-e2e` | Run Playwright E2E tests |
| `make test-wasm` | Run Rust unit tests |
| **Viewer** | |
| `make viewer` | Start viewer development server |
| `make build-viewer` | Build viewer for production (requires Rust) |
| `make build-wasm` | Build WASM module |
| **Utilities** | |
| `make clean-all` | Clean all artifacts |
| `make clean-viewer` | Clean viewer artifacts only |
| `make help` | Show all available commands |

## 📋 Development Standards

The project enforces quality standards automatically through pre-commit hooks and CI/CD:

### 🐍 Python Code Quality
- **Formatting**: Ruff formatter (automatic via pre-commit)
- **Linting**: Ruff with comprehensive rules
- **Type Safety**: MyPy strict mode with full type annotations
- **Security**: Bandit security vulnerability scanning
- **Testing**: Coverage thresholds are enforced by `pyproject.toml` and the viewer's `coverage-thresholds.mjs`
- **Documentation**: Google-style docstrings for all public APIs

### 🌐 TypeScript Code Quality
- **Formatting**: ESLint + Prettier (automatic)
- **Lint suppressions**: A reduced baselined count fails lint until you run `pnpm lint --prune-suppressions` from `packages/luxar-viewer/` and commit `eslint-suppressions.json`; inspect the removed entries with `git diff -- eslint-suppressions.json`. Deleting a baselined file is not detected, so prune manually after deletions. After moving a baselined file, re-key it with `pnpm exec eslint . --suppress-rule <rule>`, then prune. Resolve suppression-file conflicts by pruning after the merge, never by hand-merging counts or adding `eslint-disable` comments.
- **Type Safety**: Strict TypeScript configuration
- **Documentation**: JSDoc comments for complex functions
- **Memory Management**: Proper WebGL resource cleanup

## 🧪 Writing Tests

### Test Structure
```python
def test_new_feature(tmp_path):
    """Test description following Google style."""
    # Arrange
    dims = Dimensions.default_3d()
    with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
        scene = compiler.create_scene(dimensions=dims)

        # Act
        scene.add_points("test_points", positions, colors)

    # Assert
    loaded = LuxarScene.load(tmp_path / "test.zarr")
    assert "test_points" in loaded.list_points()
```

### Test Requirements
- All new public methods must have tests
- Edge cases and error conditions must be covered
- Use pytest fixtures for common setup
- Maintain the enforced coverage floors; after viewer test changes, run `pnpm test:coverage && pnpm check:coverage-slack` and refresh `coverage-thresholds.mjs` when required
- Mark slow tests with `@pytest.mark.slow`
- Mark integration tests with `@pytest.mark.integration`

### Running Tests
```bash
hatch run test              # Run all tests
hatch run test-cov          # Run with coverage report
hatch run pytest -m "not slow"  # Skip slow tests
hatch run pytest -k test_scene  # Run specific tests
```

## 🔍 Code Review Process

### Automated Checks
Every commit automatically runs:
- Code formatting (ruff, prettier)
- Linting (ruff, eslint)
- Type checking (mypy, tsc)
- Security scanning (bandit)
- Test suite with coverage

### Manual Review
Maintainers review for:
- Algorithm correctness and efficiency
- API design and usability
- Documentation completeness
- Test coverage of new features
- Breaking changes impact

## 📝 Pull Request Guidelines

### Target `dev`, not `main`

Open pull requests against **`dev`**, the integration branch:

```bash
gh pr create --base dev
```

`main` is the promoted branch. It is protected, it is what release tags are cut
from, and changes reach it by promotion from `dev` rather than directly. Name
the base explicitly rather than relying on the repository default, so the
command keeps meaning the same thing if that default ever changes.

### Coordinate Issue Work

Before creating a branch for a numbered issue, check whether an open pull
request already declares that it closes the issue:

```bash
hatch run python scripts/check_open_issue_pr.py <issue-number>
```

Exit status 1 means the issue already has an open PR. Continue that work or
coordinate on the existing PR instead of opening another one. Exit status 3
means the GitHub query failed, so do not treat it as either claimed or
unclaimed. This check uses GitHub's closing-issue references, not a local branch
list, so it also sees work created from another checkout or host. It is a manual
coordination convention, not a CI gate.
The check requires a GitHub CLI version that supports the
`closingIssuesReferences` PR field and `gh api --slurp`.

Re-run the check immediately before `gh pr create` to close the window where
another PR can open while work is in progress. When checking work already
attached to a PR, pass `--exclude-pr <your-pr>` so it does not match itself.
Use `--loose` for an advisory list of open PR titles or bodies that mention the
issue without declaring that they close it; those mentions do not change the
exit status.

If duplicate PRs still exist, do not select or close one by age alone. Choose
the survivor by completeness and discussion quality, then inventory both
branches before closing the other:

```bash
hatch run python scripts/check_open_issue_pr.py <issue-number> \
    --exclude-pr <duplicate-pr> --compare-pr <duplicate-pr>
```

Transfer or explicitly account for every duplicate-only path. Shared paths are
not proof that the patches are equivalent; inspect both diffs for unique hunks.
Also copy any materially different review conclusion onto the surviving PR so
the disagreement remains visible where the work continues.

### Before Submitting

GitHub pre-fills the canonical [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
Use its verification and documentation checklist when preparing your PR.
If the on-disk format changes, update
[the Luxar Zarr format specification](docs/guides/user/LUXAR_ZARR_FORMAT.md).

## 🎯 Development Focus Areas

### Core Python Library
- Scene graph management and optimization
- Zarr I/O performance improvements
- CLI functionality enhancements
- Memory efficiency for large datasets

### WebGL Viewer
- Rendering performance optimization
- New visual effects and shaders
- User experience improvements
- Browser compatibility

### Testing & Quality
- Expand test coverage
- Performance benchmarking
- Integration tests
- Documentation improvements

## 🔧 IDE Configuration

### VS Code Setup
Create `.vscode/settings.json`:
```json
{
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true
    },
    "ruff.lint.enable": true,
    "mypy-type-checker.reportingScope": "file",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "packages/luxar/"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.mypy_cache": true,
        "**/.pytest_cache": true
    }
}
```

### PyCharm Setup
1. Enable Ruff formatter in Settings → Tools → External Tools
2. Enable MyPy in Settings → Editor → Inspections → Python
3. Configure pytest as test runner in Settings → Tools → Python Integrated Tools

## 🐛 Debugging & Troubleshooting

### Common Development Issues

**Pre-commit hooks failing:**
```bash
make run-pre-commit  # Run manually to see specific errors
```

**Type checking errors:**
```bash
make type-check-python      # Run mypy to see detailed Python type issues
make type-check-typescript  # Run tsc to see TypeScript type issues
```

**Test failures:**
```bash
make test-cov-python   # Run tests with detailed output and coverage
pytest -v -s       # Verbose output with print statements
```

**Code formatting issues:**
```bash
make format-all    # Auto-fix most formatting problems
```

**Clean development environment:**
```bash
make clean-all     # Remove all temporary files and caches
```

## 🎓 Learning Resources

### Python Development
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Type Checking](https://mypy.readthedocs.io/)
- [Pytest Testing](https://docs.pytest.org/)
- [Zarr Documentation](https://zarr.readthedocs.io/)

### WebGL/Three.js Development
- [Three.js Documentation](https://threejs.org/docs/)
- [WebGL Fundamentals](https://webglfundamentals.org/)
- [GLSL Shaders](https://thebookofshaders.com/)

### Scientific Visualization
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [Scientific Visualization Principles](https://www.data-to-viz.com/)

## 📞 Getting Help

### Communication Channels
- **Questions**: Open a GitHub Discussion
- **Bugs**: Create a GitHub Issue with reproduction steps
- **Features**: Discuss in GitHub Issues before implementing
- **Documentation**: All public APIs need docstrings and examples

### Issue Templates
When reporting bugs, please include:
- Python/Node.js version
- Operating system
- Minimal reproduction steps
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests
When proposing features:
- Describe the use case and motivation
- Provide examples of how it would be used
- Consider backwards compatibility
- Discuss implementation approach

## 🏆 Recognition

Contributors are recognized in:
- README.md acknowledgments
- Release notes for significant contributions
- GitHub contributors page
- Special thanks for major features or bug fixes

Thank you for contributing to Luxar! 🌌
