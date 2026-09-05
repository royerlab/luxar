# luxar.typing_utils.tests

Tests for the Luxar type utilities package.

## Test Files

- `test_constants.py` - Cross-language constant guards: viewer unions are parsed and compared here, numeric mirrors are asserted per side, and the Python-side single-sourcing of the default radii (plus `lift`'s documented divergence) is pinned
- `test_enums.py` - Tests for enum types (BlendingMode, NodeType, PhysicalUnit, RenderingLimits, Defaults)
- `test_format_contract.py` - Tests that the committed Python and TypeScript projections of `format-contract/contract.yaml` — and the published version claims quoting them — stay in sync
- `test_geometry_capabilities.py` - Tests for the per-geometry-type capability table

## Running

```bash
hatch run pytest packages/luxar/src/luxar/typing_utils/tests/
```
