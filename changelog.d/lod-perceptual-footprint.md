#### GSplat LOD selection follows projected footprint

GSplat substitutive levels now stamp their median scene-unit footprint, and the
viewer uses it to choose the coarsest level whose typical splat projects to at
most 1.5 logical CSS pixels. Unstamped ladders retain the existing occupancy
selector. Footprint-selected ladders currently switch with hard swaps rather
than the occupancy selector's cross-fade band.

This deliberately holds fine geometry much longer than occupancy selection: in
a representative 300k-splat ladder it committed up to 16.7 times as many splats
across the multi-object viewing range. The #2685 scene sweep retained the
current 1.5 px viewer policy and the explicit `lodBias` override.
