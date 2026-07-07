# rail-panels

Rich control popovers hosted by the left **control rail** (`ui/control-rail/`).

The rail's chip flyout (View options) only holds on/off toggles. When a rail
button needs sliders, dropdowns, or a mode selector, it opens a **panel popover**
instead — a glass-surface container (`.luxar-control-rail__popover`) that hosts a
flat, headerless `GUI`. These builders produce that content.

## Modules

| Module                   | Rail button         | Trigger     | Contents                                                                                                                                                      |
| ------------------------ | ------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `settings-popover.ts`    | Settings (gear)     | left-click  | Theme picker (reuses `setup/theme-setup`); home for future viewer-wide prefs                                                                                  |
| `performance-popover.ts` | Performance (gauge) | right-click | Adaptive Resolution + Manual DPR + live DPR/FPS (reuses `setup/performance-setup`)                                                                            |
| `navigation-popover.ts`  | Navigation          | right-click | Segmented mode selector (orbit/fly/ortho, current highlighted, click to switch) + the current mode's parameters                                               |
| `home-popover.ts`        | Home (house)        | right-click | Reset actions: fit scene (the F shortcut), center on origin, reset dimension sliders, reset rendering settings                                                |
| `popover-gui.ts`         | —                   | —           | `makePopoverGui(host, title)`: builds a `GUI` mounted in the popover host, styled as flat content (drops the nested glass marker; CSS neutralizes its frame). |

## Conventions

- **Builders return a teardown** `() => void` run when the popover closes (dispose
  the GUI, clear intervals). Popovers are rebuilt on each open, so they always
  reflect live state — never cache across opens.
- **The popover is the glass surface.** Don't add a second `.luxar-glass-surface`
  inside; `makePopoverGui` strips it from the nested GUI.
- **Home is the one builder with no nested GUI** — it renders plain action rows
  (`.luxar-control-rail__action`, label + hint) instead of sliders, so it skips
  `makePopoverGui` entirely and its teardown is a no-op. The popover stays open
  after an action so several resets can be fired in a row.
- **Navigation** left-click cycles modes via the shared `toggleControlMode`
  command; the popover mode selector switches directly via `setControlMode`.
  Both reuse the same context/sync wiring in
  `input/input-handler/commands/control-mode.ts` so behaviour never drifts.
- **Persistence** for navigation/performance flows through
  `RenderingControls.saveSettings()` — the shared `RenderingSettings` object is
  the single source of truth, persisted under the panel's per-scene key.
