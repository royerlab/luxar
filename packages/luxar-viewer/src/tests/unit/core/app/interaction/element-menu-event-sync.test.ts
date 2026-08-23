/**
 * Guard: the Shift+F10 keybinding's dispatched event name stays equal to
 * `OPEN_ELEMENT_MENU_EVENT` (issue #1917).
 *
 * The name is necessarily written twice. The listener lives in
 * `core/app/interaction/canvas-actions.ts`, the dispatcher in
 * `input/input-handler/key-bindings/navigation-bindings.ts`, and the layer
 * contract in `.dependency-cruiser.cjs` forbids `input/` from importing
 * `core/` — so the binding cannot import the constant and hard-codes the
 * string instead. (`open-dataset-browser` is duplicated the same way, for the
 * same reason.)
 *
 * Divergence fails SILENTLY and in the least visible way possible: the
 * keyboard path is the one route with no pointer feedback, so a rename on
 * either side just means Shift+F10 quietly stops working, with the mouse path
 * still fine and every unit test still green.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { OPEN_ELEMENT_MENU_EVENT } from '../../../../../core/app/interaction/canvas-actions';

describe('element-menu keybinding ↔ canvas-actions listener', () => {
  const bindingsPath = resolve(
    dirname(fileURLToPath(import.meta.url)),
    '../../../../../input/input-handler/key-bindings/navigation-bindings.ts'
  );

  it('the keybinding dispatches exactly OPEN_ELEMENT_MENU_EVENT', () => {
    const source = readFileSync(bindingsPath, 'utf-8');
    const match = source.match(/new CustomEvent\(\s*'([^']*element-menu[^']*)'\s*\)/);
    expect(
      match,
      'no element-menu CustomEvent dispatch found in navigation-bindings.ts'
    ).toBeTruthy();
    expect(match![1]).toBe(OPEN_ELEMENT_MENU_EVENT);
  });

  it('the dispatch is gated on focus being on the scene, not a panel', () => {
    // The layers panel binds these same two keys on its own row listener and
    // calls preventDefault() but NOT stopPropagation(), so the event reaches
    // the window handler too. `openContextMenu` is module-global, so an
    // ungated dispatch would close the layer menu and open the canvas one.
    const source = readFileSync(bindingsPath, 'utf-8');
    const fn = source.match(/const openElementMenu[\s\S]{0,600}?\n {2}\};/);
    expect(fn, 'openElementMenu not found').toBeTruthy();
    expect(fn![0]).toContain('isFocusOnSceneCanvas');
  });

  it('both keys are bound on the NAVIGATION context', () => {
    // FLY_CONTROLS filters keys through an `allowedKeys` whitelist that
    // contains neither F10 nor ContextMenu, and only reaches a NAVIGATION
    // binding via passthrough — so registering elsewhere would silently
    // disable the shortcut in fly mode.
    const source = readFileSync(bindingsPath, 'utf-8');
    const f10 = source.match(
      /registerBinding\(\s*InputContext\.NAVIGATION\s*,\s*\{[^}]*key:\s*'F10'[^}]*\}/s
    );
    const ctxMenu = source.match(
      /registerBinding\(\s*InputContext\.NAVIGATION\s*,\s*\{[^}]*key:\s*'ContextMenu'/s
    );
    expect(f10, 'Shift+F10 not registered on InputContext.NAVIGATION').toBeTruthy();
    expect(ctxMenu, 'ContextMenu key not registered on InputContext.NAVIGATION').toBeTruthy();
    expect(f10![0]).toContain('shift: true');
  });
});
