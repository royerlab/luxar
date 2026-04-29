# Native launcher binaries

Built artifacts that back `luxar export --native ...`. Source lives in
`packages/luxar-launcher/`. Build with:

```bash
make build-launchers
```

The launcher uses CGO + a system WebView library, so `make build-launchers`
builds for the **host platform only** — pure Go cross-compilation does not
work. Linux + Windows binaries must be built on hosts of the matching
OS (typically via CI matrix runners).

Outputs land here as:

- `darwin-universal` — macOS universal (arm64 + amd64), built on macOS
- `linux-amd64`, `linux-arm64` — built on Linux of the matching arch

The binaries are gitignored. Without them, `luxar export --native` raises
a clear error pointing the user at the build command.
