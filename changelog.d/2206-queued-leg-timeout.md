#### Give the required TypeScript CI leg enough starvation headroom

Increased the required TypeScript CI leg's timeout from 60 to 120 minutes because
its measured 15.5-minute runtime can stretch to about 51 minutes under the
documented `SCHED_IDLE` load, leaving the old budget with the thinnest margin of
any obsidian-routed job. Also corrected the runner-watchdog documentation to state
that a dispatched leg can time out before any step starts or runner is recorded.
