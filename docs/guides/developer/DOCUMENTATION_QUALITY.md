# Documentation Quality Gate

`scripts/check_documentation.py` is a baseline-driven **ratchet** for
documentation completeness. Pre-existing documentation debt is tolerated via a
checked-in baseline, but any **new** missing README / docstring / JSDoc finding
fails the check. This lets us stop the bleeding today and pay down debt over
time without a flag-day rewrite.

## What it checks

**Python** (`packages/luxar/src/luxar/`, one pass per top-level package):

- `README.md` exists for each package.
- README quality: has a `## Quick Start` / `## Getting Started` section, is at
  least 500 characters, and contains a code example.
- Every non-underscore `*.py` file (plus `__init__.py`) has a module docstring.
- Docstring coverage is at least 70%, counting every `def`/`async def`/`class`
  the file defines — methods and nested definitions included.
- Both are measured from the parsed AST rather than a text heuristic, so a
  shebang, a UTF-8 BOM, a PEP 263 encoding cookie and a multi-line signature
  are all handled the way Python itself handles them.
- A `*.py` file that cannot be parsed is reported as a `Python syntax` finding
  (the run continues rather than crashing).

**TypeScript** (`packages/luxar-viewer/src/`, one pass per package directory):

- `README.md` exists for each package.
- JSDoc coverage across exported symbols
  (`export function|class|interface|type|const`) is at least 70%.

Each finding is identified by a stable, portable key:

```
<check_name>::<repo-relative-posix-path>
```

The key deliberately **excludes** the human message, so volatile coverage
percentages (e.g. "72%") never destabilize the baseline.

> **Known limitation:** keys are *file-granular* — they omit the coverage
> percentage, so once a file is baselined, further coverage decay *within* that
> same file (e.g. adding more undocumented functions to an already-listed file)
> is not caught. The ratchet prevents NEW under-documented files/symbols, not
> intra-file regressions of already-baselined files.

## The ratchet model

The baseline lives at `scripts/docs_baseline.json`:

```json
{
  "_comment": "Documentation-debt baseline ...",
  "failures": ["<key>", "<key>", ...]
}
```

The `failures` list is sorted and human-diffable. On each run the checker
computes the current failing keys and compares them to the baseline:

- **New** = current − baseline → regressions. **These fail the check.**
- **Still baselined** = current ∩ baseline → tolerated pre-existing debt.
- **Fixed** = baseline − current → debt you paid down.

Exit code is `1` if and only if there are new findings; pre-existing debt does
**not** fail the build.

### Regenerating and tightening the baseline

Create or overwrite the baseline from the current state:

```bash
hatch run python scripts/check_documentation.py --update-baseline
```

When you fix documentation debt, the checker reports the entries as *Fixed* and
reminds you to tighten the baseline. Re-run `--update-baseline` and commit the
smaller `docs_baseline.json` so the fixed items can never regress silently.

## Machine-readable output

`--json` prints a deterministic report to stdout instead of the human summary:

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
    "new": ["<key>", ...],
    "fixed": ["<key>", ...],
    "still_present": ["<key>", ...]
  }
}
```

Findings (both passed and failed) are sorted by `key`. The `ratchet` block
classifies the current failures against the baseline (each list sorted); it is
`null` only when no baseline is consulted. The exit code still honors the
ratchet: `--json` exits `1` when `ratchet.new` is non-empty.

## Local commands

```bash
make check-docs                                          # run the gate
hatch run python scripts/check_documentation.py          # ratchet mode (default)
hatch run python scripts/check_documentation.py --json   # machine-readable
hatch run python scripts/check_documentation.py --update-baseline  # (re)write baseline
hatch run python scripts/check_documentation.py --no-baseline      # legacy strict mode
```

`--no-baseline` ignores the baseline entirely and fails on *any* finding — the
original strict behavior, useful for measuring total debt.

## Phased debt reduction

The ratchet (see [The ratchet model](#the-ratchet-model)) holds the line at
today's debt: it stops new findings but does not, on its own, remove the
pre-existing ones. This plan pays that debt down in bounded phases,
cheapest-first. Each phase ends by re-running the gate with `--update-baseline`
and committing the smaller baseline (see [Regenerating and tightening the
baseline](#regenerating-and-tightening-the-baseline)), so the reclaimed ground
can never regress.

### Current debt snapshot

With Phases 1 and 2 complete, the baseline holds 37 findings, all in one check:

| Count | Check | What it means |
|------:|-------|---------------|
| 37 | JSDoc coverage | Exported-symbol JSDoc below the 70% floor, in `packages/luxar-viewer/src/`. |

The Phase 1 categories (*Quick Start section*, *Module docstring*, *Code
examples*) and the Phase 2 *Docstring coverage* category are now at zero and are
held there by the ratchet.

These counts are a **snapshot** and will drift as the tree changes; do not
trust the prose. The authoritative live breakdown comes from re-measuring:

```bash
hatch run python scripts/check_documentation.py --no-baseline --json | \
  python3 -c "import json,sys; from collections import Counter; \
  c=Counter(f['check_name'] for f in json.load(sys.stdin)['findings'] if not f['passed']); \
  [print(f'{v:4d}  {k}') for k,v in sorted(c.items(), key=lambda x:-x[1])]"
```

### Phase 1 — Structural README + module docstrings ✅ done

The 4 *Quick Start section*, 2 *Code examples* and 3 *Module docstring*
findings (9 items). These were quick, mechanical, and high-signal: added the
missing `## Quick Start` section and a `python`/`typescript` fenced example to
each README, and a module docstring to each flagged `*.py` file. All three of
these check categories are now at zero, and the baseline has been tightened so
they cannot regress.

### Phase 2 — Python docstring coverage ✅ done

The *Docstring coverage* findings. **Target:** every listed Python file reaches
the 70% coverage floor. Driven down in review-sized batches by package rather
than one flag-day sweep, so each change stayed readable: the core-library
packages first, then the `demos/` and `tests/` batch. This category is now at
zero and the baseline has been tightened so it cannot regress.

### Phase 3 — TypeScript JSDoc coverage

The 37 *JSDoc coverage* findings, the largest bucket. **Target:** every listed
viewer file reaches the 70% JSDoc floor. As with Phase 2, work package by
package in review-sized batches.
