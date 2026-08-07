# luxar.core.tests

Unit tests for the `luxar.core` package covering scene graph classes, transforms, dimensions, and data nodes.

## What's Tested

- **Scene structure** - Scene creation, hierarchy, group nesting, validation, and persisted dimension reassignment
- **Data nodes** - Points, Lines, GSplats, Mesh creation and metadata
- **Transforms** - Identity, translate, rotate, scale, compose, inverse, look_at, serialization
- **Dimensions** - Dimension/Dimensions creation, validation, categorical dimensions, spatial flags
- **dim_order** - Mapping lower-dimensional data into higher-dimensional scenes
- **extend_to_all** - Visibility extension across non-displayed dimensions (Points, GSplats)
- **Node properties** - Opacity, gamma, intensity, offset, blending mode, method chaining
- **Rendering attributes** - Validation of rendering properties on Node
- **HDR colors** - Edge cases for HDR color support
- **Physical units** - Unit support through Dimensions system
- **ViewerConfig** - ViewerConfig, CameraConfig, UIConfig, DimensionsConfig, AnimationConfig

## How to Run

```bash
hatch run pytest packages/luxar/src/luxar/core/tests/
hatch run pytest packages/luxar/src/luxar/core/tests/test_transforms.py  # Single file
```
