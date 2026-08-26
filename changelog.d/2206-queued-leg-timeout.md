#### Give the required TypeScript CI leg enough starvation headroom

Increased the required TypeScript CI leg's timeout from 60 to 120 minutes because
a self-hosted timeout can begin at dispatch: the observed roughly 20-minute
pre-step delay plus a conservative 51-minute `SCHED_IDLE` bound does not fit the
old budget. Also documented the distinct outcomes for a job left queued versus
one lost after dispatch.
