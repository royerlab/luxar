#### Capsule lines: packed joint state and one less varying

The capsule line primitive's per-segment joint state now rides packed
half-precision varyings — 12 flat floats down to 9 — with the bisector-cut
normals deliberately left at full precision, since halves there would
re-introduce the #1502 hairpin banding (each leg quantizes in its own
frame, so the error does not cancel between two planes that must be exact
complements). The WebGL shaders also derive perspective correction from
`gl_FragCoord.w` rather than carrying a dedicated `1/w` varying and
dividing per fragment: the two are the same number by spec identity.

Together these cut the worst-case (10 M hairline segments) overhead over
the legacy quad from ~1.32× to ~1.23×; fill-bound wide-line scenes are
unchanged. No joint, cap or profile behaviour changes at any width: the
two backends stay in value parity across the capsule parity fixtures, and
the only numeric difference is the endpoint radii rounding to half
precision (≤0.2% of profile, invisible).
