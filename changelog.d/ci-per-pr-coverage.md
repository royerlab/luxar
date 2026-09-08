#### Faster PR CI: coverage moves off the per-PR critical path

`python-tests` was gated on a coverage-instrumented run of the full `not slow`
suite — the dominant CPU cost of that job (~60–90 min under load) and the
bottleneck capping how fast PRs reach a mergeable state. Pull-request runs now
execute the same suite without coverage instrumentation (new `test-nocov` hatch
script), and the coverage collection plus the 89% threshold run on a new
**scheduled** dev cron (every 4h) and on every `workflow_dispatch` — best-effort
on push to `dev`, where the workflow's `cancel-in-progress` concurrency usually
cancels the long coverage leg before it finishes. The scheduled run has its own
concurrency group, so pushes cannot cancel it; it is the reliable home of the
gate. A coverage regression therefore surfaces on the scheduled/dispatch dev run
(halting promotion to `main`) rather than blocking each PR. The `python-tests`
context name is unchanged, so no required status is orphaned.
