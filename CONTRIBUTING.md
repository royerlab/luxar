# 🤝 Contributing to Luxar

We welcome contributions! The project uses automated tooling to maintain high code quality.

## 🚀 Quick Start for Contributors

```bash
# 1. Fork and clone the repository
git clone <your-fork-url>
cd luxar

# 2. Set up development environment
make dev-setup  # Installs dependencies and pre-commit hooks

# 3. Create feature branch
git checkout -b feature/amazing-feature

# 4. Make your changes
# ... edit code ...

# 5. Run quality checks (before committing)
make check      # Runs all quality checks
make test-cov   # Ensure tests pass with coverage

# 6. Commit (pre-commit hooks run automatically)
git commit -m "Add amazing feature"

# 7. Push and create Pull Request
git push origin feature/amazing-feature
```

## 🛠️ Development Environment

### First-time Setup
```bash
make dev-setup     # Complete development environment
make check         # Verify everything works
```

### Daily Development Workflow
```bash
make format        # Format code before editing
make check         # Run all quality checks
make test-cov      # Run tests with coverage
```

### Available Commands

| Command | Purpose |
|---------|---------|
| `make dev-setup` | Complete development environment setup |
| `make format` | Format code with ruff |
| `make lint` | Run ruff linting |
| `make type-check` | Run mypy type checking |
| `make security` | Run bandit security scan |
| `make test` | Run test suite |
| `make test-cov` | Run tests with coverage report |
| `make check` | Run all quality checks |
| `make clean` | Clean temporary files |
| `make help` | Show all available commands |

## 📋 Development Standards

The project enforces quality standards automatically through pre-commit hooks and CI/CD:

### 🐍 Python Code Quality
- **Formatting**: Ruff formatter (automatic via pre-commit)
- **Linting**: Ruff with comprehensive rules
- **Type Safety**: MyPy strict mode with full type annotations
- **Security**: Bandit security vulnerability scanning
- **Testing**: 80%+ code coverage requirement
- **Documentation**: Google-style docstrings for all public APIs

### 🌐 TypeScript Code Quality
- **Formatting**: ESLint + Prettier (automatic)
- **Type Safety**: Strict TypeScript configuration
- **Documentation**: JSDoc comments for complex functions
- **Memory Management**: Proper WebGL resource cleanup

## 🧪 Writing Tests

### Test Structure
```python
def test_new_feature(tmp_path):
    """Test description following Google style."""
    # Arrange
    scene = Scene(tmp_path / "test.zarr")
    
    # Act
    result = scene.your_new_method()
    
    # Assert
    assert result.is_valid()
    assert result.count == expected_count
```

### Test Requirements
- All new public methods must have tests
- Edge cases and error conditions must be covered
- Use pytest fixtures for common setup
- Maintain 80%+ coverage (enforced automatically)
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

### Before Submitting
- ✅ `make check` passes without errors
- ✅ `make test-cov` shows adequate coverage
- ✅ All pre-commit hooks pass
- ✅ Documentation updated for new features
- ✅ Examples added for new functionality
- ✅ docs/guides/user/LUXAR_ZARR_FORMAT.md updated (if data format changes)
- ✅ CLAUDE.md updated (if significant learnings)
- ✅ CHANGELOG.md updated (if applicable)

### PR Description Template
```markdown
## Description
Brief description of changes

## Motivation and Context
Why is this change needed? What problem does it solve?

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally with `make test-cov`
- [ ] New tests added for new functionality
- [ ] Existing tests updated if needed

## Screenshots (if applicable)
Add screenshots for UI changes

## Breaking Changes
List any breaking changes and migration steps

## Related Issues
Closes #(issue_number)
```

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
    "python.formatting.provider": "ruff",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "packages/luxar/src/luxar/tests"
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
make pre-commit-run  # Run manually to see specific errors
```

**Type checking errors:**
```bash
make type-check     # Run mypy to see detailed type issues
```

**Test failures:**
```bash
make test-cov      # Run tests with detailed output and coverage
pytest -v -s       # Verbose output with print statements
```

**Code formatting issues:**
```bash
make format        # Auto-fix most formatting problems
```

**Clean development environment:**
```bash
make clean         # Remove all temporary files and caches
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