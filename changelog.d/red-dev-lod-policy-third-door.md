#### The LOD-policy gate recognises a demo that names its own topology

Two changes that were each green on their own branch were red together: the gate
that requires every fitting demo to choose an LOD topology, and the cell-tracking
challenge demo. The demo does choose — `build_lod()` asks `build_recipe` for
`levels` with time as a hard coarsening barrier, and the precomputed crop it ships
is that result — but it makes the call itself rather than going through
`save_with_lod`, which was the gate's only definition of "chose".

Neither existing exemption described it: it does ship an artifact, and it does not
keep whatever topology its fitter happened to produce. So the gate gained a third,
narrow door for a demo that names the topology of its shipped artifact in its own
source with a literal `build_recipe(...)` recipe. Narrow is the point: an
AST-parsed detector demands a constant in the recipe slot, since a variable there
(the recipes gallery loops over all of them) names nothing a reviewer can read off
the source, and a new staleness test re-measures the claim every run — the member
must still fit, must still not use the helper, and must still carry that literal
call, so the exemption cannot outlive its reason.
