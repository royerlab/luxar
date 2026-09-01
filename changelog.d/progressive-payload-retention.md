#### Progressive ladders retain one cumulative payload

Progressive Points, Gaussian Splat, Lines, and Mesh ladders now fold decoded
rungs into one cumulative payload and release the redundant child buffers,
avoiding roughly doubled terminal viewer residency while preserving logical
ladder depth for refinement, rollback, monitoring, caching, and picking.

Demo ladder guidance now describes additive ladders as bounding first-paint
latency and individual commit work rather than reducing terminal geometry.
