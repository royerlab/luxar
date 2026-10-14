#### A fresh Linux machine gets through `make setup-dev` and `make test-fast`

On a fresh Ubuntu box the first `make setup-dev` installed nvm, never installed
Node, stopped with a bare `Error 3` — and printed a ✅ first. Sourcing `nvm.sh`
runs an implicit `nvm use` that honours the repository's `.nvmrc`, and it returns
3 while Node 22 is not installed yet; under the Makefile's `-e` shell the `&&`
chain silently skipped `nvm install`. Every nvm site now sources with `--no-use`
and selects Node explicitly. Installing runs as separate checked steps, and
`node -v` is checked against the floor before anything reports success.

The first Hatch environment build (torch, napari/Qt, about 10-25 minutes) now
says up front what it is downloading and streams the installer's output. A
failure stops setup with a pointer instead of being swallowed by `|| true`.
Every run then checks the env with a verbose `hatch -v run`. On an env left
half-built by an interrupted first run, that installs what is missing in plain
sight or fails with the `hatch env remove default` hint. Before, `env create`
reported "already exists" and the sync happened silently inside the tolerated
pre-commit step.
`installer = "uv"` was evaluated and left out, because this env must keep `pip`.

The fresh-machine test loop works too. napari's auto-loaded pytest plugin is
disabled (`-p no:napari`), so pytest no longer crashes at startup on a box
without libGL; no test used its fixtures. The `jq` requirement in
`test_ci_diff_classifier.py` moved from a module-wide autouse assert (200 errors
without jq) into the two runners that execute jq scripts. Locally those tests
skip with an install hint; under `CI=true` they fail, so CI can never drop them
silently. `make test-fast` now always runs both the Python and TypeScript halves
and combines their exit codes, except that a Ctrl-C during the Python half stops
there. Its xdist run, and the local `hatch run test` / `test-cov-all` scripts,
pass `--max-worker-restart=0`: the run aborts at the first worker crash instead
of respawning workers and hanging at 99%.

CUDA advice no longer recommends Ubuntu's `nvidia-cuda-toolkit` (CUDA 12.0, too
old for cu13x torch and for Blackwell). `make check-cuda-deps` and
`make setup-cuda` print NVIDIA's `cuda-keyring` repository commands for x86_64
or sbsa (aarch64). Ubuntu derivatives such as Mint and Pop!_OS map through
`UBUNTU_CODENAME`, and other distros get NVIDIA's download page. The toolkit major
is picked to match `torch.version.cuda` (11, 12 or 13; a CPU-only torch gets the
current default with a note). The cu130 hints note that they need driver >= 580,
and point to cu128 for older drivers.
`setup-cuda` no longer runs `sudo apt install python3-dev` without asking; it
now sits behind the same prompt as the other packages. The torch index map
gains cu126 and cu130. `make install-rust` checks for a C compiler before
`cargo install wasm-pack` needs one, and `make check-deps` reports `cc` and `jq`.
README, CONTRIBUTING and the build-system guide share one prerequisite list for
Ubuntu/Debian and macOS. CONTRIBUTING's test loop now matches the rest of the
docs: `make test-fast` while iterating, `make test-all` before pushing.
