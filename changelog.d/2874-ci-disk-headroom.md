#### CI fails fast before disk exhaustion

The long Python and coverage jobs now reclaim unused tooling on both hosted and
self-hosted runners, then require 25 GiB of free root-disk space before checkout
or environment creation. A saturated CI host therefore stops immediately with
an explicit disk-headroom error instead of failing an unrelated test mid-suite.
