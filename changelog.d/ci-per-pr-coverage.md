#### Faster PR CI: coverage moves to a dedicated push-to-dev workflow

`python-tests` was gated on a coverage-instrumented run of the full `not slow`
suite — the dominant CPU cost of that job (~60–90 min under load) and the
bottleneck capping how fast PRs reach a mergeable state. Pull-request runs now
execute the same suite without coverage instrumentation (new `test-nocov` hatch
script). Pushes to `dev` still run `test-cov` on all three `python-tests` legs;
the protected `python-tests (3.12)` context remains the promotion-visible 89%
gate. A dedicated `coverage.yml` workflow also runs one 3.12 coverage leg on
every push to `dev` (and on `workflow_dispatch`) with per-commit, non-cancelling
concurrency. Its check attaches to the dev commit and always completes, but is
advisory until the `coverage` context is added to repository protection. Thus a
dev push currently computes coverage four times (three ci.yml legs plus the
dedicated observation), while PRs avoid the instrumentation cost. ci.yml
dispatches also run `test-cov`. The `python-tests` context name is unchanged, so
no required status is orphaned.
