# luxar.typing_utils.tests

Tests for the Luxar type utilities package.

## Test Files

- `test_constants.py` - Guards that constants mirrored in the TypeScript viewer stay literally equal on both sides
- `test_enums.py` - Tests for enum types (BlendingMode, NodeType, PhysicalUnit, RenderingLimits, Defaults)
- `test_format_contract.py` - Tests that the Python projection of `format-contract/contract.yaml` stays in sync
- `test_geometry_capabilities.py` - Tests for the per-geometry-type capability table

## Running

```bash
hatch run pytest packages/luxar/src/luxar/typing_utils/tests/
```
