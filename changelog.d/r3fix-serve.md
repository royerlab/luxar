#### Serving and first-run friction from the third round of new-user testing

`luxar demo run` accepts `--port` and `--viewer-port` (1-65535, directly or after
`--`), forwarded to `luxar serve` like `--host`, so a demo behind a port forward
or a firewall rule no longer has to be re-served by hand. Serving flags never
send a run to a custom-variant build, after `--` or not. A re-run still rebuilds
the scene (so a code change always shows), but when the default build is present
it first prints the `luxar serve … --viewer` command, with the same host and
ports, that serves the existing build without rebuilding. A server bound to a
wildcard host (`--host 0.0.0.0`) now also prints a `127.0.0.1` viewer URL: the
advertised LAN address inside a container or VM is an internal one, while a port
forward reaches the servers on loopback. `luxar demo run <key> --help` now
describes the demo, as `-- --help` already did; a wrapper such as `hatch run`
swallows the `--`, which used to print the generic `demo run` usage instead.

An exported folder's `serve.py` redirects `/` to the viewer with the scene loaded
instead of showing a directory listing. The redirect is built from the request's
own host, so it is right through a port forward, and never carries the control
flag or token; with the relay on, `serve.py` says at startup that the bare
address opens a view-only viewer and the printed Display URL is the one the
panel drives.

The viewer no longer logs two `HEAD … 404` lines on every load of a format-3 store
at a URL without a `.zarr` suffix (`luxar serve` mounts a store at `/`, an export
at `/data`): the store probe now asks for `zarr.json` first and stops at the first
hit, each probe with its own timeout, instead of firing all three markers in
parallel. Over plain http from another host, where the browser exposes no OPFS,
the missing disk cache is one `console.debug` line instead of a warning with a
`TypeError` stack; a genuine OPFS failure still warns.

`gsplat convert --blending-mode volumetric` prints the predicted density on the
opening view's brightest rays and the `--absorption` that gives about one optical
depth there. Auto exposure does not run for volumetric, so that density is set by
the fit (on a 10-timepoint zebrafish nuclei fit it was 3.95, so κ ≈ 0.25), and the
README's example, which used κ = 4, rendered such fits as an opaque wall. The
README now uses κ = 0.25 with a black-footed colormap and explains the choice. The
ACES/colormap-LUT notice is printed once per convert (a LOD store printed it
twice) and names `--tone-mapping`.

`make test-fast` no longer always runs both halves (superseding the #3110 entry):
setting only `PYTEST_ARGS` runs only the scoped Python half, setting only
`TS_ARGS` runs only the scoped vitest half, both runs both scoped, and neither
runs everything as before. Its documented timing now quotes the measurement
precisely: ~22.5 minutes for the Python half alone (~20k tests) on a fresh
20-core machine, before ~18k vitest tests. The `gsplat view` SIGHUP test no
longer fails when it inherits an ignored SIGHUP (`nohup`, agent shells), which
also made it touch the real 8000/5173 ports.

Install docs: the README's capability table says a native app export needs a
source checkout (`make build-launchers`), the Ubuntu `apt` lines run `apt update`
first, and the install skill's "Hatch owns the environment" advice moved under
the NVIDIA profile's dev-checkout heading, where it applies.

#### `gsplat convert` points dense volumes at volumetric blending

When the predicted opening view stacks more than about 20 splat peaks along its brightest rays (dense nuclear stains: a cells3d nuclei fit predicts 31, a full Allen FOV 28; sparse cells sit at 0.6–10), the exposure line is followed by one hint: additive blending shows an integrated glow there, and `--blending-mode volumetric` (which then prints a starting `--absorption`) shows interior structure. The lod substitutive reducer also stops warning about the MPS float64 fallback when the device was chosen automatically.
