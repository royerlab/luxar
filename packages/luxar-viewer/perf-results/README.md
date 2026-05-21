# perf-bench output

This directory holds per-commit JSON results from the line perf bench.

Layout:

    perf-results/
      <commit-sha>/
        results.json

Compare two:

    pnpm perf:diff perf-results/<base>/results.json perf-results/<new>/results.json

