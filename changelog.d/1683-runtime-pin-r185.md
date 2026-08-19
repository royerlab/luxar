#### Three.js runtime moves to r185, closing the one-minor type/runtime skew

`three` was held at `~0.184.0` while `@types/three` sat a minor ahead at `~0.185.1` —
a deliberate skew, because the r184 definitions reject a correct componentwise
`pow(vec3, vec3)` in the capsule-line shader and the r185 ones do not. The runtime
could not follow because r185 broke the TSL path in four measurable ways, most
starkly a gsplat pick surface that rendered zero pixels. #1697 root-caused all four
to one bug of ours — pick shaders sharing fragment values across `colorNode` and
`depthNode`, so they depended on which entry point three built first, and r185 flipped
that order — and fixed it by assigning every shared value in an unconditional prologue.

With the blocker gone, the runtime is now `~0.185.1` and both pins sit on the same
minor. The shaders still measure clean at r184 *and* r185: the fix removed the
dependency on build order rather than adapting to r185's, so `tsl-shader-parity` +
`tsl-codegen-snapshot` were 101 passed at both revisions. What the bump changes is
which revision the checked-in snapshots pin.

All 74 generated-shader snapshots were regenerated, and the diff is presentational
throughout: the `r184` header line, a new empty `structs` section, the `object` and
`render` std140 blocks swapping order with the `nodeUniformN` / `nodeVarN` renumbering
that follows, `modelViewMatrix` emitted lazily at its first use instead of at the top
of `main()`, and `(!x)` printed `( ! x )`. The one structural move is the flip itself:
in the six pick fragment shaders the whole `depthNode` flow, `gl_FragDepth` write
included, now sits above the discards. That is inert — a discarded fragment updates no
buffer, depth included — but it is the change an order-insensitive check cannot see, so
it is named rather than folded into "presentational". The rest was checked rather than
assumed: with
the generated identifiers normalized away, 58 of the 74 `main()` bodies are an
identical multiset of statements and the remaining 16 differ only by that `!` spacing.
The lazy `modelViewMatrix` emission is the same hazard class #1697 fixed, so it was
re-checked the same way: a scan for values first assigned inside a branch and later
read at top level returns the same set under both revisions.

The bump also trades around which half of the codegen guard is live. The brace-depth
half was inert while the runtime sat at r184 (colour-first order builds even a
free-standing `.toVar()` at top level) and is the live one now — r185's depth-first
order is exactly what buried the pick chain in an `else` arm. The read-order half went
the other way: with the prologue emitted at the top of `main()` by the depth flow,
nothing `colorNode` does can trip it. One consequence is called out at the site rather
than left to be rediscovered — `colorNode`'s own prologue call is verified by nothing
at r185 (removing it leaves the generated GLSL byte-identical) and is kept deliberately,
so both entry points stay self-sufficient whichever one a future three builds first.

The revision floor in `assertThreeRevision` moves to 185 to track the peer range's
minor. So does the pinned importmap URL in the `examples/embed/` sample, which reaches
for `three` over a CDN and so is not covered by the peer range at all — left at r184 it
would have thrown the guard's own error on load. And `THREE_VERSION_NOTES.md` now describes the pin that exists rather
than the one it used to block. Comments elsewhere that state what the *current* pin
does — the WebGPU compat-mode adapter workaround, which three still defaults to at
r185 exactly as it did at r184 — were updated; dated r184 measurements in the mesh and
depth-sorting specs were left as the record of what was checked and when.
