# Documentation Quality Gate

Luxar uses baseline-driven **ratchets** for documentation completeness and
TypeDoc warnings. Pre-existing debt is tolerated through checked-in baselines,
but a new finding fails the pull-request check unless its exact baseline entry
is added explicitly and reviewed. Sphinx starts from a clean baseline: every
warning is fatal.

The `docs-quality` CI job runs the full gate for pull requests that change:

- `docs/**`, Markdown, or reStructuredText;
- Python source under `packages/luxar/src/luxar/**`;
- TypeScript source under `packages/luxar-viewer/src/**`; or
- documentation checkers, baselines, build configuration, or workflows.

The job reports success explicitly for unrelated changes, so it can remain a
stable required check. If change detection fails, the job fails safe by running
the documentation checks.

## Required checks

`make check-docs` mirrors the required CI gate:

1. `scripts/check_documentation.py` checks README, Python docstring, TypeScript
   JSDoc, and tracked repository-path completeness against
   `scripts/docs_baseline.json`.
2. TypeDoc converts and validates the viewer API without emitting output, then
   compares normalized warning messages with
   `packages/luxar-viewer/typedoc-warnings-baseline.json`.
3. Sphinx builds the HTML documentation with `-W --keep-going`, so every
   warning and broken internal reference is reported and the build fails.

The required gate does **not** probe external HTTP links. Remote availability,
rate limits, and anti-bot responses are not deterministic merge dependencies.
Run `hatch run docs:linkcheck` when auditing external links. Any unavoidable
exception must be a narrow, commented `linkcheck_ignore` entry in
`docs/conf.py`; broad domain-wide or catch-all exceptions are not acceptable.

The one remote dependency left inside the gate is intersphinx, which fetches
three third-party inventories. Sphinx reports an unreachable inventory as an
untyped warning that `suppress_warnings` cannot name, so `docs/conf.py` demotes
that single record to informational — an outage at `docs.python.org` must not
fail an unrelated pull request. Unresolved references inside our own
documentation are untouched and remain fatal.

## Completeness ratchet

`scripts/check_documentation.py` has the following scope.

### Python

For `packages/luxar/src/luxar/`, one pass per top-level package:

- `README.md` exists for each package.
- README quality: it has a `## Quick Start` / `## Getting Started` section, is
  at least 500 characters, and contains a code example.
- Every non-underscore `*.py` file, plus `__init__.py`, has a module docstring.
- Docstring coverage is at least 70%, counting every `def`, `async def`, and
  `class` the file defines, including methods and nested definitions.
- Python is measured from the parsed AST, so shebangs, UTF-8 BOMs, PEP 263
  encoding cookies, and multiline signatures are handled as Python handles
  them. An unparseable file becomes a `Python syntax` finding rather than
  crashing the run.

### TypeScript

For `packages/luxar-viewer/src/`, one pass per package directory:

- `README.md` exists for each package.
- JSDoc coverage across exported symbols
  (`export function|class|interface|type|const`) is at least 70%.

### Repository paths

For tracked package `README.md` files:

- Backticked path-like references must resolve to tracked repository files.
- Distinct broken references in one README receive distinct baseline keys.

Each finding has a stable, portable key:

```text
<check_name>::<repo-relative-posix-path>[::<detail>]
```

The key excludes the human message, so volatile coverage percentages do not
change baseline identity. The optional detail distinguishes checks that can
fail more than once in one file, such as broken path references.

> **Known limitation:** coverage keys are file-granular. Once a low-coverage
> file is baselined, further coverage decay inside that file is not a new key.
> The ratchet prevents newly under-documented files and symbols but cannot yet
> detect every intra-file regression.

### Completeness baseline

The baseline lives at `scripts/docs_baseline.json`:

```json
{
  "_comment": "Documentation-debt baseline ...",
  "failures": ["<key>", "<key>", "..."]
}
```

The checker compares current failing keys with that list:

- **New** = current − baseline: regressions; the check fails.
- **Still baselined** = current ∩ baseline: tolerated existing debt.
- **Fixed** = baseline − current: entries that should be removed.

Exit code `1` means there are new findings. Existing baseline debt alone does
not fail the build.

Regenerate the baseline only after inspecting every addition and removal:

```bash
hatch run docs:python scripts/check_documentation.py --update-baseline
```

When debt is fixed, commit the smaller baseline in the same pull request so the
finding cannot silently return.

### Machine-readable report

`--json` prints a deterministic report and retains the ratchet exit status:

```json
{
  "summary": { "passed": 0, "failed": 0, "total": 0 },
  "findings": [
    {
      "key": "<check>::<file>",
      "check_name": "...",
      "file": "<repo-relative-posix-path>",
      "passed": false,
      "message": "...",
      "line_number": null
    }
  ],
  "ratchet": {
    "new": ["<key>", "..."],
    "fixed": ["<key>", "..."],
    "still_present": ["<key>", "..."]
  }
}
```

Findings and ratchet lists are sorted. The `ratchet` field is `null` only when
no baseline is consulted.

## TypeDoc warning ratchet

`pnpm run typedoc:check-warnings` uses TypeDoc's API to convert and validate the
viewer without generating HTML. It records the warning **multiset**, normalizes
checkout-specific paths and terminal escapes, and compares it with
`packages/luxar-viewer/typedoc-warnings-baseline.json`.

- A new warning or an increased duplicate count fails.
- Removed warnings are reported and should be deleted from the baseline.
- TypeDoc conversion or compiler errors always fail, regardless of the warning
  baseline.
- `--json` provides a machine-readable summary.

Update the baseline only after reviewing the warning-set diff:

```bash
cd packages/luxar-viewer
pnpm run typedoc:check-warnings -- --update-baseline
```

## Internal and external links

The warning-fatal Sphinx HTML build is the deterministic internal-link gate. It
validates the published RST/MyST document graph, references, and local assets.
The completeness checker separately validates path-like references in package
READMEs. Reference warnings are intentionally not suppressed in `docs/conf.py`.
The one linkcheck-only local exception, `viewer/index.html`, is generated by
TypeDoc after Sphinx; the deployment workflow separately fails if that API
directory is empty.

External URL checking is opt-in:

```bash
hatch run docs:linkcheck
```

A transient external failure does not block unrelated pull requests. Fix a
genuine broken URL. If a site cannot be checked reliably, add the narrowest
possible documented `linkcheck_ignore` entry in `docs/conf.py` so the exception
is visible in review.

## Phased debt reduction

The completeness ratchet has already eliminated its structural README and
Python docstring categories. On August 7, 2026, the remaining checked-in debt is
**11 completeness findings**, all TypeScript JSDoc coverage, and **86 TypeDoc
warnings**. Sphinx allows zero warnings.

The counts are snapshots. The two baseline files and the commands below are the
authoritative measurements. Every completed phase must commit a smaller
baseline; a baseline must never grow merely to make CI green.

| Phase | Completeness target | TypeDoc warning target |
| --- | --- | --- |
| 0 — enforce | No growth beyond 11 findings | No growth beyond 86 warnings |
| 1 — finish completeness | Empty completeness baseline | At most 60 warnings |
| 2 — TypeDoc burn-down | Empty completeness baseline | At most 30 warnings |
| 3 — warning-free API | Empty completeness baseline | Empty warning baseline |

### Completed completeness phases

1. **Structural README and module gaps:** the 4 Quick Start, 2 code-example,
   and 3 module-docstring findings were eliminated.
2. **Python docstring coverage:** all 24 files reached the 70% floor.
3. **TypeScript JSDoc coverage:** this phase started with 37 files and is being
   reduced package by package; the remaining count is read from
   `scripts/docs_baseline.json` and targets zero.

### TypeDoc warning phases

Reduce the warning multiset in review-sized package or warning-class batches.
The current baseline consists primarily of unresolved comment links and
referenced symbols that TypeDoc cannot include. The intermediate count targets
bound each batch while allowing ownership to follow the affected packages.

## Local commands

```bash
make check-docs
make check-docs-verbose

# Completeness ratchet
hatch run docs:python scripts/check_documentation.py
hatch run docs:python scripts/check_documentation.py --json
hatch run docs:python scripts/check_documentation.py --update-baseline
hatch run docs:python scripts/check_documentation.py --no-baseline

# TypeDoc warning ratchet
cd packages/luxar-viewer
pnpm run typedoc:check-warnings
pnpm --silent run typedoc:check-warnings -- --json
pnpm run typedoc:check-warnings -- --update-baseline

# Sphinx internal-link/warning gate and optional external-link audit
cd ../../
hatch run docs:build
hatch run docs:linkcheck
```

`--no-baseline` ignores the completeness baseline and fails on any finding. It
is useful for measuring total debt, not as the merge gate.
