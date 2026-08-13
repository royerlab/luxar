#### An end-to-end guard for the mesh reveal ladder

Every defect the mesh reveal ladder shipped with was caught by adversarial review
or by loading a store in a browser by hand — never by a unit test, and the unit
suites were green through all of them. `test_mesh_reveal_ladder.luxar.zarr` (a
4-level reveal of a subdivided icosphere beside an UNLADDERED copy of the same
surface) and `mesh-reveal-ladder.spec.ts` close that gap.

The control node is the design, not padding: each of those bugs was a difference
between the laddered node and what an ordinary mesh does with identical data, and
none is legible without something correct in the same scene under the same camera.

Every test is mutation-verified against the defect it names — disabling the
in-place attribute refresh fails the colour-tail test, returning a numeric energy
fraction fails the stamp test, and dropping the parent's normal frame fails the
shading test. One candidate test was **removed** for failing that bar: buffer
capacity sized from the committed prefix and from the node total converge at
completion, so no end-state observable distinguishes them. Its guard is the
unit-level assertion that attribute objects survive three growing commits; the
spec now says so rather than implying coverage it does not have.
