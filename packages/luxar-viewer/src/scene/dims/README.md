# Scene Dimension Helpers

Pure nD navigation helpers owned by the scene layer.

- `step-math.ts` calculates step sizes and cyclic next positions from dimension
  ranges.
- `selection.ts` identifies non-displayed dimensions and maps digit shortcuts to
  them.

The modules are side-effect free and shared by scene coordination, input
bindings, and focused unit tests.
