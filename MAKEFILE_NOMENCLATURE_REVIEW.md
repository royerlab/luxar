# Makefile Command Nomenclature Review

**STATUS: ✅ APPLIED**
All recommended changes have been implemented following the **action-first** pattern.

## Critical Issues Found (RESOLVED)

### 1. **install-*** vs **setup-*** Semantic Confusion

The current naming creates ambiguity about what each command does:

| Command | What It Actually Does | Problem |
|---------|----------------------|---------|
| `install-node` | Installs Node.js tool via nvm/homebrew | ✅ Clear |
| `install-pnpm` | Installs pnpm tool globally | ✅ Clear |
| `install-hatch` | Installs Hatch tool via pipx | ✅ Clear |
| `install-python` | Installs **Luxar package** in editable mode | ❌ Misleading - not installing Python! |
| `install-pre-commit` | **Activates** pre-commit hooks | ❌ Misleading - not installing pre-commit tool! |
| `install-viewer` | Installs **viewer dependencies** | ❌ Ambiguous - not installing a "viewer" tool |
| `setup-dev` | **Orchestrates** complete dev environment | ✅ Clear orchestrator |
| `setup-rust` | Installs Rust + wasm-pack | ❓ Why setup- instead of install-? |
| `setup-cuda` | **Orchestrates** CUDA install + build | ✅ Clear orchestrator |

### 2. Inconsistent Patterns

**Tool Installation:**
- `install-node` but `setup-rust` - Both install external tools, different prefixes
- `install-pnpm` but `setup-cuda` - Inconsistent for similar operations

**Misleading Names:**
- `install-python` sounds like "install Python interpreter" but actually means "install Luxar Python package in editable mode"
- `install-pre-commit` sounds like "install pre-commit tool" but actually means "activate/enable hooks"
- `install-viewer` sounds like "install viewer application" but actually means "install node_modules for viewer"

## Proposed Nomenclature Rules

**IMPORTANT: Action-First Pattern**
All commands follow the **action-first** pattern (e.g., `install-dev`, not `dev-install`).

### Rule 1: **install-<tool>** - Install a Single External Tool
External tools that get installed into the system (not part of the project).

**Examples:**
- `install-node` - Install Node.js via nvm/homebrew
- `install-pnpm` - Install pnpm via npm
- `install-hatch` - Install Hatch via pipx
- `install-rust` - Install Rust + wasm-pack (renamed from setup-rust)

**Pattern:** `make install-<tool>` installs an external tool to your system.

---

### Rule 2: **setup-<component>** - Orchestrated Multi-Step Setup
Complex setup involving multiple installs, checks, and configuration.

**Examples:**
- `setup-dev` - Complete dev environment (Node, pnpm, Hatch, Python env, TypeScript deps)
- `setup-cuda` - CUDA environment (system packages, PyTorch, build extension)

**Pattern:** `make setup-<component>` orchestrates multiple steps for a subsystem.

---

### Rule 3: **install-<component>-deps** - Install Dependencies
Install dependencies for a specific component (package.json, requirements.txt, etc.).

**Examples:**
- `install-viewer-deps` (renamed from install-viewer)
- `install-python-deps` (potential new command for pip install -r requirements.txt)

**Pattern:** `make install-<component>-deps` installs dependencies listed in manifest files.

---

### Rule 4: **install-dev** - Editable Package Install
Install the project package itself in development mode. Follows action-first pattern.

**Examples:**
- `install-dev` (renamed from install-python)

**Pattern:** `make install-dev` installs the project package in editable/development mode.

---

### Rule 5: **enable-<feature>** - Enable/Activate a Feature
Turn on a feature or hook that's already installed.

**Examples:**
- `enable-pre-commit` (renamed from install-pre-commit)

**Pattern:** `make enable-<feature>` activates an already-installed feature.

---

### Rule 6: **build-<component>** - Build Artifacts
Compile or build artifacts for a component.

**Examples:**
- `build-viewer` ✅
- `build-wasm` ✅
- `build-cuda` ✅
- `build-docs` ✅

**Pattern:** Already consistent!

---

### Rule 7: Other Prefixes (Already Good)
- `test-*` - Run tests ✅
- `check-*` - Check/verify something ✅
- `clean-*` - Clean artifacts ✅
- `format-*` - Format code ✅
- `run-*` - Run scripts/examples ✅
- `serve-*` - Start a server ✅

---

## Recommended Changes

### High Priority (Misleading Names) - APPLIED ✅

```makefile
# OLD NAME                    # NEW NAME                       # REASON
install-python          →     install-dev                      # Actually installs Luxar package, not Python (action-first)
install-pre-commit      →     enable-pre-commit                # Activates hooks, doesn't install tool
install-viewer          →     install-viewer-deps              # Installs dependencies, not a viewer
setup-rust              →     install-rust                     # Matches install-node pattern
```

### Implementation Example - APPLIED ✅

```makefile
# ❌ OLD (misleading)
install-python:  ## Install the Python package in editable mode
	pip install -e .

install-pre-commit:  ## Install pre-commit hooks
	hatch run pre-commit install

install-viewer:  ## Install viewer dependencies
	cd packages/luxar-viewer && pnpm install

setup-rust:  ## Install/update Rust and wasm-pack for WASM development
	# ... rust installation ...

# ✅ NEW (clear) - Action-first pattern
install-dev:  ## Install Luxar Python package in editable mode for development
	pip install -e .

enable-pre-commit:  ## Enable and activate pre-commit hooks
	hatch run pre-commit install

install-viewer-deps:  ## Install viewer dependencies (node_modules)
	cd packages/luxar-viewer && pnpm install

install-rust:  ## Install Rust and wasm-pack for WASM development
	# ... rust installation ...
```

---

## Benefits of Applied Changes ✅

1. **Action-first pattern**: All commands follow consistent `<action>-<target>` format
   - install-dev (not dev-install)
   - install-rust (not setup-rust)
   - enable-pre-commit (not pre-commit-enable)

2. **install-*** prefix is reserved for:
   - External tools (install-node, install-rust, install-pnpm)
   - Dependencies (install-viewer-deps)
   - Development packages (install-dev)

3. **setup-*** prefix is reserved for:
   - Orchestrated multi-step setups (setup-dev, setup-cuda)

4. **enable-*** prefix for activation:
   - enable-pre-commit (activate hooks)

5. **No ambiguity:** Each command name clearly indicates what it does
   - "install-python" could mean Python interpreter → "install-dev" means Luxar package
   - "install-pre-commit" could mean the tool → "enable-pre-commit" means activate hooks
   - "install-viewer" could mean app → "install-viewer-deps" means dependencies

---

## Migration Strategy - APPLIED ✅

**Selected: Option A (Breaking Change)**
Old commands have been completely removed. Clean break with clear new names.

Changes applied:
- `install-python` → `install-dev` (removed old name)
- `install-pre-commit` → `enable-pre-commit` (removed old name)
- `install-viewer` → `install-viewer-deps` (removed old name)
- `setup-rust` → `install-rust` (removed old name)

No deprecation period or aliases - this is an early-stage project where breaking changes are acceptable.

---

## Summary of Nomenclature Rules - APPLIED ✅

**All commands follow the action-first pattern: `<action>-<target>`**

| Pattern | Purpose | Examples |
|--------|---------|----------|
| `install-<tool>` | Install external tool | install-node, install-rust, install-hatch |
| `install-<component>-deps` | Install dependencies | install-viewer-deps |
| `install-dev` | Install project package editable | install-dev |
| `setup-<component>` | Orchestrated multi-step setup | setup-dev, setup-cuda |
| `enable-<feature>` | Activate a feature | enable-pre-commit |
| `build-<component>` | Build artifacts | build-viewer, build-wasm, build-cuda |
| `test-<scope>` | Run tests | test-all, test-python, test-e2e |
| `check-<aspect>` | Verify/check | check-deps, check-all, check-cuda-deps |
| `clean-<scope>` | Clean artifacts | clean-all, clean-viewer, clean-cuda |
| `format-<language>` | Format code | format-python, format-typescript |
| `run-<script>` | Run scripts | run-examples, run-demos |
| `serve-<target>` | Start server | serve-docs, serve-dataset |

---

## Decisions Made ✅

1. **Breaking changes with no deprecation period**
   - Clean break - old command names completely removed
   - Justification: Early-stage project where breaking changes are acceptable

2. **Action-first pattern for all commands**
   - `install-dev` (not `dev-install`)
   - `install-rust` (not `setup-rust`)
   - Consistency with existing patterns: `test-all`, `check-deps`, `build-viewer`

3. **Clear semantic separation:**
   - `install-<tool>` for external tools (node, rust, hatch)
   - `install-<component>-deps` for dependencies (node_modules)
   - `install-dev` for Luxar package itself
   - `setup-<component>` for orchestrated multi-step setups (setup-dev, setup-cuda)
   - `enable-<feature>` for activating features (enable-pre-commit)

4. **Documentation updated:**
   - ✅ CLAUDE.md: Added nomenclature section + updated all references
   - ✅ Makefile: All command names and references updated
   - ✅ This review document: Updated with applied changes
