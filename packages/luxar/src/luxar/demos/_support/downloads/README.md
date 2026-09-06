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

## Key Functions

- `robust_download()` retries with exponential backoff and resumes through HTTP
  Range requests. Bytes stay in `<dest>.part` until complete and size-verified;
  `If-Range` uses the ETag or Last-Modified value in
  `<dest>.part.validator`, so changed remote content is restarted rather than
  spliced onto stale bytes. Caller-supplied headers are scoped to the original
  URL across redirects.
- `verify_file_checksum()` and `download_with_checksum()` enforce MD5/SHA256
  integrity, deleting a completed download when verification fails.
- `find_quarantined_files()`, `format_quarantine_notice()`, and
  `warn_if_quarantined()` report rejected `.corrupt` cache artifacts before a
  potentially large replacement download starts.
- `download_zip_member()` validates the member path, caps the central directory
  at 64 MiB and uncompressed output at 256 GiB by default, then verifies size
  and CRC before atomically promoting the selected member. Caller-supplied
  headers are scoped to the original URL across redirects.

`zip_safety.py` centralizes member-name validation for local and remote
archives. Dataset policy, cache layout, and manifest resolution belong in the
sibling `datasets` package; this package only moves and validates bytes.
