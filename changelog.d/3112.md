#### Demos and servers behave on remote, headless and scripted machines

`luxar demo run` now forwards `--host` and `--no-open` to the server of every
demo that serves through `launch_viewer` (`luxar demo run lorenz -- --host
0.0.0.0` serves it to another machine), and `luxar demo run <key> -- --help`
describes the demo — its details, its own notes and the shared demo flags —
instead of starting it. Demos that served out of a temporary directory now
build into the demos output directory exactly as `--no-serve` does, so
`luxar demo` marks them built afterwards. A store written into the demos
directory is now built in a hidden sibling and renamed into place only when
complete (`LuxarZarrCompiler(atomic=...)`, on by default there), so a reader
never sees a half-written scene and a failed rebuild keeps the previous one.
`luxar demo run` refuses to start a demo that is already running, `run-all`
skips running demos, and `demo cache clear --outputs` keeps their scenes.
Arguments that change what a demo builds (`-- --points 100`) build into
`demos/variants/custom-<hash>/`, leaving the default build — the one the
catalogue marks and `run-all` skips — untouched.

The browser `luxar viewer`, `luxar gsplat view` and demos open by default is
skipped where nobody can see it — over SSH, on Linux without
`DISPLAY`/`WAYLAND_DISPLAY`, or with no terminal attached — and the URL is
printed instead, unless `$BROWSER` routes it somewhere (VS Code Remote-SSH,
Codespaces). A typed `--open` always launches one; `LUXAR_OPEN_BROWSER=1` or
`=0` overrides everything. `luxar gsplat view` gains `--host`, binding both servers
and advertising a dialable address like `luxar serve` does.

The new top-level `luxar stop` stops servers started by `luxar serve`,
`luxar viewer`, `luxar gsplat view` and `luxar export --open` — which now
register themselves in the run registry while they serve — plus every running
demo. A server is stopped by PID, never by process group, so the script that
started it survives, and only after checking that the PID still runs the same
luxar subcommand and started before it registered — a stale entry whose PID
was recycled is pruned, not signalled, and without a process table nothing is
signalled at all. `luxar stop`, `luxar demo stop` and `luxar demo cache
clear` no longer wait at a `[y/N]` prompt when stdin is not a terminal: they
exit 1 and ask for `--yes`.

`luxar export --open` runs the exported `serve.py` as an owned child: stopping
or killing the command (a closed session, a harness timeout, `luxar stop`) now
takes the server down instead of orphaning it on its port, a server that fails
is reported with its exit status instead of ending the command with status 0,
and the command's own output is flushed before the server starts so a
redirected log keeps it. From an installed wheel, `luxar export --native` and
a missing viewer build no longer point at `make build-launchers` or
`pnpm build`, which only exist in a source checkout; they say what is needed.
