#### The line primitive sizes itself to the scene

A new `Settings → Advanced → Line primitive` policy (`Auto` / `Capsule` /
`Quad`, default `Auto`) decides which primitive a lines node builds. Auto
keeps the capsule default but builds the cheaper screen-space quad for
nodes whose effective segment load — authored count scaled by an
extent-normalized width factor — reaches two million, where the capsule's
measured extra GPU cost starts eating a meaningful slice of the frame
budget on discrete GPUs (~1.5× the quad on thin lines at every count,
over 3× on wide lines). The visual and picking materials resolve through
one seam so they always agree, the decision is made once per node at
material build and never re-runs, and `?linePrimitive=` remains the
session's strongest word.

The line perf bench gains the matching measurement axis:
`LUXAR_PERF_SYNTHETIC_COUNTS` sweeps the synthetic scenarios across
segment counts, and synthetic scenes now carry the authored-count
metadata production scenes always had.
