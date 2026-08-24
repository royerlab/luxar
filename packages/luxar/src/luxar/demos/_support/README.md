# Demo Support Helpers

This private subpackage contains leaf utilities owned exclusively by the demo
package. Modules here support several demos but are not demos themselves and do
not belong in the core runtime or shared `luxar.utils` package.

The existing flat `demos/_*.py` helpers remain at the package root. Small
implementation helpers such as UMAP colour/legend utilities and flow-field
math belong here.

## Quick Start

Demo modules import shared helpers through the public demo barrel rather than
reaching into this private package:

```python
from luxar.demos import FlowField, cubic_bounds, rk4_step
```

`fields.py` provides the frozen `FlowField` dataclass, symmetric cubic bounds,
trilinear sampling, normalized flow directions, vectorized RK4 advection, and a
reference-cube scene helper. The PPI flow-field and zebrahub velocity demos keep
their own binning, smoothing, caching, and streamline-seeding policies.

Every non-`__init__.py` module in this directory is included in the demo guard
set defined by `demos/tests/_scanned_modules.py`.
