### Fixed

- Progressive Points, Gaussian Splat, and Mesh ladders now release decoded rung buffers after folding them into one cumulative payload, matching Lines and avoiding roughly doubled terminal viewer residency.
- Demo ladder guidance now describes additive ladders as bounding first-paint and individual commit work rather than reducing the terminal geometry itself.
