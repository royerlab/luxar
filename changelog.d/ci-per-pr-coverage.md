#### Faster PR CI: coverage runs on push-to-dev, not on every PR

`python-tests` was gated on a coverage-instrumented run of the full `not slow`
suite — the dominant CPU cost of that job (~60–90 min under load) and the
bottleneck capping how fast PRs reach a mergeable state. Pull-request runs now
execute the same suite without coverage instrumentation (new `test-nocov` hatch
script); coverage collection and the 89% threshold still run on every push to
`dev` and on the full-matrix `workflow_dispatch`. A coverage regression
therefore surfaces on `dev` (halting promotion to `main`) rather than blocking
each PR.
