# Demo Download Support

This private package owns the network and archive mechanics used to obtain demo
assets. It contains resumable HTTP downloads, checksum verification, quarantine
reporting, HTTP-range ZIP extraction, and ZIP-member path validation. These are
demo distribution concerns rather than general Luxar runtime utilities: callers
should use names exported from `luxar.demos`, not import this package directly.
Keeping the implementation private lets the demos evolve retry and hosting
policy without creating another supported top-level API.

## Quick Start

Demo code requests a hosted file through the public demo barrel:

```python
from luxar.demos import download_with_checksum

download_with_checksum(
    "https://example.org/demo.npy",
    "demo.npy",
    expected_sha256="0123456789abcdef",
)
```

`download.py` stages bytes beside the destination, validates resumptions, and
only promotes complete files. `remote_zip.py` extracts one member without
fetching an entire archive. `zip_safety.py` centralizes member-name validation
for both local and remote archives. Dataset policy, cache layout, and manifest
resolution belong in the sibling `datasets` package; this package only moves
and validates bytes.
