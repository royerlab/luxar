#### Rebuild only stale example fixtures

`make run-examples` now fingerprints each example producer independently from
its own source and the local Luxar modules and example helpers reachable from
its imports. The fixture marker records those dependencies and outputs per
producer, so changing one builder or an unrelated production module no longer
regenerates the whole example corpus. Missing outputs and removed producers are
handled independently, successful producers remain stamped across a partial
failure, and `--check` names the producers that need rebuilding.
