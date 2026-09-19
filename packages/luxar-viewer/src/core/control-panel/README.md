# `core/control-panel` — the kiosk touch panel

The second page in this bundle. `control.html` is a tablet on a plinth: a
full-screen grid of chapter tiles, no 3D, no data. Tapping a tile moves one
dimension on a viewer somewhere else, and the scene's authored waypoints do
everything visible — the camera flies, the overlays swap, the narration cues.

```
luxar serve scene.luxar.zarr --viewer --control --host 0.0.0.0
#   display: http://<host>:5173/?src=...&control
#   panel:   http://<host>:5173/control.html?control
```

## Files

| File                   | What it owns                                                                                       |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| `main.ts`              | Page bootstrap. Wiring only — read the URL, connect, derive, render, keep the active tile in step. |
| `controller-socket.ts` | The controller half of the wire: sends requests, tracks replies, receives events, reconnects.      |

The pieces it wires live elsewhere on purpose, so each sits behind the coverage
floor and the layer rules that suit it:

- `config/control-panel/derive-chapters.ts` — what the chapters _are_ (pure).
- `ui/control-panel/render-panel.ts` — the DOM (jsdom-testable, port-injected).
- `utils/json-rpc.ts` — framing, shared with the viewer's own client.
- `styles/control-panel.css` — the whole look, self-contained.

## Zero authoring required

`Dimension(categories=[...])` already reaches the viewer as
`DimensionMetadata.categories`, so the panel asks `getDimensions()` and reads
the stop names that are already in the scene. A tour built before this page
existed gets a working menu with no changes at all. An authored block can
enrich the result later; it is not needed to get started.

If a scene has no chapter dimension — most scenes are not tours — the page says
so rather than drawing an empty grid.

The two-minute idle reset is armed by a tile interaction, not by opening the
panel. If a visitor walks away after making a selection, the display returns to
the first chapter; merely connecting a panel never moves a display an operator
has deliberately parked elsewhere.

## Things worth knowing before you change this

**It must not pull the renderer in.** The panel imports the viewer's `config`
and `ui` modules, so a shared chunk could drag `three` or a codec in without a
single offending import here. `scripts/check-eager-chunks.mjs` asserts that per
HTML entry, and that assertion is verified to fire. Current budget: 2 eager
chunks, against the viewer's 5.

**It is a second entry in one build**, declared in `vite.config.ts`'s
`rolldownOptions.input`. Setting `input` at all removes Vite's implicit
`index.html` default, so both pages must be named or one silently disappears.
Sharing the build is what makes the page ride into `dist`, `luxar export` and
the native bundles for free — all three copy the directory whole.

**Taps are notifications, not calls.** A tile sends `setDimensionValue` and does
not wait: the display should move immediately, and the authoritative position
comes back as a `dimensions-changed` event, which is also what marks the active
tile. `call()` is for the things that genuinely need an answer.

**It ships without `ThemeManager`.** That singleton persists its choice to
`localStorage`, and the panel shares an origin with the display — so setting a
theme here would re-theme the big screen on its next reload. Every colour in
`control-panel.css` therefore reads a `--luxar-*` token _with a fallback_, and a
lock test keeps it that way.

**The class and custom-property names are a public API.** A scene may ship its
own stylesheet, which puts those names in other people's files.
`src/tests/unit/styles/control-panel-contract.test.ts` pins them from both
sides — DOM and stylesheet — so a rename fails the build instead of an exhibit.

**`?panel=<module>` is the escape hatch.** Same-origin only, with no opt-out:
the module is `import()`ed, so it is executable code. It receives an
already-connected socket and the page body (`CustomPanelContext`). That exists
so the first exhibit to outgrow CSS has a supported path instead of forking this
directory.

## See also

- `docs/guides/specs/REMOTE_CONTROL_SPEC.md` — the three-party contract.
- `../app/control/README.md` — the viewer's side of the same channel.
