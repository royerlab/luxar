#### Quantized radii and widths stay inside their chunk bounds in the viewer

The positive-scalar round-trip pad budgeted one float32 ULP for decoding, but the
viewer reconstructs linear quantization through separately rounded bounds, scale,
product, and sum. For a narrow class of float64 arrays on the uint16 tier, those
roundings could move a decoded radius or line width just beyond the pad used to
build its chunk bound, allowing an edge query to miss that chunk.

The pad now covers the full float32 affine reconstruction while remaining tighter
than two measured upward displacements in the deterministic regression.
