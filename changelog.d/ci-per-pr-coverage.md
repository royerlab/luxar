#### Faster PR CI: coverage moves to a dedicated push-to-dev workflow

`python-tests` was gated on a coverage-instrumented run of the full `not slow`
suite — the dominant CPU cost of that job (~60–90 min under load) and the
bottleneck capping how fast PRs reach a mergeable state. Pull-request runs now
execute the same suite without coverage instrumentation (new `test-nocov` hatch
script). The coverage collection plus the 89% threshold move to a dedicated
`coverage.yml` workflow that runs on every push to `dev` (and on
`workflow_dispatch`). Because it is a `push` to `dev`, its check-run attaches to
the dev commit itself, so promotion — which reads per-commit check-runs for the
commits ahead of `main` — halts on a coverage regression; its per-commit
`concurrency` group (`cancel-in-progress: false`) lets every dev commit's
coverage run to completion. The full-matrix `ci.yml` dispatch still runs
`test-cov` as well. The `python-tests` context name is unchanged, so no required
status is orphaned.
