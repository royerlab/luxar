# Demo Runtime Support

This private package owns process-time behavior shared by executable demos:
device selection, common command-line flags, scene provenance, stale-output
detection, deterministic demo ports, and viewer launch orchestration. These
helpers are specific to the demo application lifecycle and do not belong in the
general-purpose `luxar.utils` namespace. Their supported entry point is the
`luxar.demos` barrel, which keeps every demo on one import spelling.

## Quick Start

Select a device and parse the standard demo flags through the public barrel:

```python
from luxar.demos import detect_device, parse_demo_flags

flags = parse_demo_flags()
device = detect_device()
print(device, flags["recompute"])
```

`device.py` probes CUDA, MPS, and CPU availability. `flags.py` implements the
shared demo CLI contract. `provenance.py` fingerprints builders and decides when
an existing scene is safe to reuse. `viewer.py` allocates stable ports and starts
the server/viewer pair with package-root process helpers. Dataset acquisition and
download integrity are intentionally separated into sibling support packages.
