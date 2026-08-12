#### Capsule lines: packed joint state, hairline cut gate, and one less varying

The capsule line primitive's per-segment joint state now rides packed
half-precision varyings (the bisector-cut normals stay full precision —
halves there would re-introduce the #1502 hairpin banding), lines whose
apparent radius sits below the AA floor skip the joint partition entirely
(the artifact it prevents is sub-pixel at that size, and zooming in
re-enables it automatically), and the WebGL shaders derive perspective
correction from `gl_FragCoord.w` instead of a dedicated varying. Together
these cut the worst-case (10 M hairline segments) overhead over the
legacy quad from ~1.32× to ~1.23× with pixel-identical output on every
parity fixture; wide-line scenes are unchanged. Slice-clipped ends keep
their exact perpendicular butt at any width — nothing may draw past a
slice plane.
