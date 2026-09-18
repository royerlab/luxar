#### wasm-pack 0.15, a dependency refresh, and an honest `rebuild-viewer`

`wasm-pack` moves 0.14.0 → 0.15.0 across all six pin sites — the Makefile plus
five workflow jobs — which silences the update warning printed on every viewer
build. The generated WASM is byte-identical: same 50,858-byte binary, same
`.d.ts`, same JS shim. The only difference is a trailing newline in the generated
`package.json`, which is gitignored. The 0.15.0 release also moved the upstream
repository from `drager/wasm-pack` to `wasm-bindgen/wasm-pack`; the release URLs
CI downloads still resolve through the rename redirect, which was checked against
the real assets rather than assumed.

`make install-rust` now actually enforces that pin. It used to accept whatever
`wasm-pack` it found — so a developer sitting on 0.13.0 was told "already
installed" and stayed there, and bumping the pin changed nothing locally. It now
compares against the pinned version and reinstalls on a mismatch — then re-probes
`PATH` and fails outright if `PATH` still answers with something else, because
`cargo install --force` only replaces the copy in cargo's own install root and a
Homebrew or distro copy earlier in `PATH` keeps winning. A pin nobody can observe
is not a pin. The version itself moves to a single `WASM_PACK_VERSION` variable at
the top of the Makefile.

Routine dependency refreshes ride along: `zarrita` 0.7.4, `mediabunny` 1.54.0,
Vitest 4.1.10, ESLint 10.8.1, Knip 6.32.2, dependency-cruiser 18.2.0, tsx, and
`globals`. `@typescript-eslint`'s plugin and parser had drifted onto different
versions and now move together; Dependabot groups the family so they stay
aligned. The Vitest runner, coverage provider, and UI are grouped for the same
reason. `@types/three` keeps its `~0.185.1` range and picks up 0.185.4 within it.

`THREE_VERSION_NOTES.md` gains the explanation for something that has looked like
an oversight for a while: `three` is pinned at `~0.184.0` while `@types/three`
sits at `~0.185.1`, a deliberate one-minor lead. `three` ships no `.d.ts`, so
`@types/three` is the sole type description of the runtime — and `0.184.1`, the
last r184 definitions release, declares TSL `pow` as
`(x: Node<"vec3">, y: Node<"vec3">)` where a `VarNode<"vec3">` is not assignable,
so a correct componentwise `pow(vec3, vec3)` call fails to typecheck. The r185
definitions widen those overloads and describe the r184 runtime *more* accurately
than the r184 ones do. Closing the skew by moving the runtime to r185 was
attempted and reverted: it breaks the TSL/WebGPU path in four places, most
starkly a gsplat pick surface that renders zero pixels. That is now tracked in
its own issue, and the notes record both the reason for the skew and the reason
the runtime stays put for now. (The blocker was root-caused and fixed shortly after,
and the runtime has since moved to r185 — the notes describe the closed skew.)

Finally, `make rebuild-viewer` stops advertising a "complete clean rebuild" it
never performed. It clears the JS/TS artifacts but deliberately leaves
`public/wasm/` and the cargo target directory alone, so the Rust step is a cache
hit whenever its sources are unchanged — which is the right default, since the
stale-artifact bugs the target exists to clear are vite/TS ones and the release
profile costs ~2 minutes to rebuild from scratch. The help text now says what it
does and points at `make clean-wasm rebuild-viewer` for the full thing.
