#### Give the required TypeScript CI leg enough starvation headroom

Increased the required TypeScript CI leg's timeout from 60 to 120 minutes. A
successful attempt ran 17m53s of real steps; at the documented 3.3x `SCHED_IDLE`
extreme, a healthy starved run projects to roughly 59 minutes, leaving the old
budget no headroom for pre-step dispatch latency. Also documented the distinct
outcomes for a job left queued versus one lost after dispatch.
