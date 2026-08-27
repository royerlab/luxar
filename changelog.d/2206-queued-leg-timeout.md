#### Give the required TypeScript CI leg enough starvation headroom

Increased the required TypeScript CI leg's timeout from 60 to 120 minutes. A
successful TypeScript attempt ran 17m53s of real steps; at the documented 3.3x
`SCHED_IDLE` extreme, a healthy starved run projects to roughly 59 minutes,
leaving the former 60-minute budget no headroom for any pre-step dispatch
latency. A dispatch-lost leg can spend the same budget without starting a step.
Also documented the distinct outcome for a job left queued.
