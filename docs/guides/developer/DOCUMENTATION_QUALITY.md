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
- Docstring coverage across `def`/`class` definitions is at least 70%.
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
