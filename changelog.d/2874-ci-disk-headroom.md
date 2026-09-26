#### CI fails fast before disk exhaustion

The long Python and coverage jobs reclaim unused tooling on hosted runners,
then require 25 GiB of free root-disk space before checkout or environment
creation. A saturated CI host therefore stops early with an explicit disk
headroom error instead of starting a job with too little space.
