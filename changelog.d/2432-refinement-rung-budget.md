#### Warm-cache refinement now respects the residency ceiling within a pass

Progressive Points, Lines, GSplats, and Mesh refinement now receive the shared
budget's fair share of remaining byte headroom, with enough allowance for one
estimated rung, and stop after the first rung that spends it. Cached ladders can
no longer climb through several resident rungs under one admission before the
ceiling is measured again. The budget reconciles each loader's measured
post-pass footprint before considering the next node.
