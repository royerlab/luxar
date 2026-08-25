# Demo Runtime Support

This private package owns process-time behavior shared by executable demos:
device selection, common command-line flags, scene provenance, stale-output
detection, deterministic demo ports, and viewer launch orchestration. These
helpers are specific to the demo application lifecycle and do not belong in the
general-purpose `luxar.utils` namespace. Demo-facing helpers use the
`luxar.demos` barrel, which keeps every demo on one import spelling;
`demo_ports()` remains internal runtime plumbing.

## Quick Start

Select a device and parse the standard demo flags through the public barrel:

```python
from luxar.demos import detect_device, parse_demo_flags

flags = parse_demo_flags()
device = detect_device()
print(device, flags["recompute"])
```

## Key Functions

- `demo_ports()` derives a stable per-dataset data/viewer port pair so demos do
  not contend for 8000/5173. No two datasets share a full pair, so a stale
  browser tab cannot silently front another demo's server. Explicit `--port`
  and `--viewer-port` values in `serve_args` override the pair.
- `detect_device()` selects CUDA, then MPS, then CPU.
- `warn_if_no_cuda_gpu()` reports when CUDA is unavailable.
- `BUILDER_FINGERPRINT_ATTR`, `demo_source_fingerprint()`, and
  `scene_is_current()` identify the producing demo, hash its sources plus the
  writer environment, and reuse only completed scenes written by the current
  producer.
- `parse_demo_flags()` parses the shared `--recompute`, `--keep-stale`,
  `--no-serve`, and `--serve-only` flags.
- `parse_int_arg()` parses integer `--name=VALUE` or `--name VALUE` forms and
  falls back to its default after warning on malformed values.
- `parse_path_arg()` parses and expands path flags, returning `None` when the
  flag is absent or has no value.

`device.py` probes CUDA, MPS, and CPU availability. `flags.py` implements the
shared demo CLI contract. `provenance.py` fingerprints builders and decides when
an existing scene is safe to reuse. `viewer.py` allocates stable ports and starts
the server/viewer pair with package-root process helpers. Dataset acquisition and
download integrity are intentionally separated into sibling support packages.
