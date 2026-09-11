### Changed

- GSplat substitutive levels now stamp their median scene-unit footprint, and
  the viewer uses it to choose the coarsest level whose typical splat projects
  to at most 1.5 logical CSS pixels. Unstamped ladders retain the existing
  occupancy selector.
