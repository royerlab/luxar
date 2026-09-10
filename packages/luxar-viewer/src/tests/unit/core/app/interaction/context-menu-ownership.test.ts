// @vitest-environment jsdom
/**
 * Native context menu over the viewer's own surfaces (WKWebView regression).
 *
 * The bug these cover: in a WKWebView the `contextmenu` target is the overlay
 * sitting above the canvas, not the canvas, so the canvas-only suppression the
 * controls install never fired and the native "Copy Image" sheet opened over
 * the scene mid right-drag. Each case below is an element measured leaking in
 * the shipped ESM kiosk bundle.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { installContextMenuOwnership } from '../../../../../core/app/interaction/context-menu-ownership';
import { EventGroup } from '../../../../../utils/cross-layer/event-group';
import { resetViewerContainer, setViewerContainer } from '../../../../../utils/viewer-container';

/** Dispatch a cancellable `contextmenu` and report whether anything ate it. */
function rightClick(target: Element): boolean {
  const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
  target.dispatchEvent(event);
  return event.defaultPrevented;
}

function appendTo(parent: Element, tag: string, className = ''): HTMLElement {
  const el = document.createElement(tag);
  if (className) el.className = className;
  parent.appendChild(el);
  return el;
}

describe('installContextMenuOwnership', () => {
  let events: EventGroup;
  let canvas: HTMLCanvasElement;

  beforeEach(() => {
    document.body.innerHTML = '';
    events = new EventGroup();
    canvas = document.createElement('canvas');
    canvas.id = 'app';
    document.body.appendChild(canvas);
    setViewerContainer(document.body);
    installContextMenuOwnership(canvas, events);
  });

  afterEach(() => {
    events.dispose();
    resetViewerContainer();
    window.getSelection()?.removeAllRanges();
    document.body.className = '';
    document.body.innerHTML = '';
  });

  it('suppresses the menu on the 3D canvas', () => {
    expect(rightClick(canvas)).toBe(true);
  });

  it.each([
    ['scene overlay', 'div', 'luxar-overlay luxar-overlay--text'],
    ['turntable matte canvas', 'canvas', 'luxar-overlay__matte'],
    ['turntable video', 'video', 'luxar-overlay__matte-source'],
    ['data monitor', 'div', 'luxar-data-monitor'],
    ['dimension sliders', 'div', 'luxar-dimension-sliders'],
    ['gui panel', 'div', 'luxar-gui'],
  ])('suppresses the menu on the %s', (_label, tag, className) => {
    expect(rightClick(appendTo(document.body, tag, className))).toBe(true);
  });

  it.each([
    ['dimension slider range', 'luxar-dimension-sliders', 'range'],
    ['gui checkbox', 'luxar-gui', 'checkbox'],
  ])('suppresses the menu on the %s input', (_label, panelClass, inputType) => {
    const panel = appendTo(document.body, 'div', panelClass);
    const input = appendTo(panel, 'input') as HTMLInputElement;
    input.type = inputType;
    expect(rightClick(input)).toBe(true);
  });

  it('suppresses the menu on an unclassed child of an overlay (the logo img)', () => {
    // The image overlay's <img> carries no class of its own — only `closest`
    // on the ancestor overlay identifies it as ours.
    const overlay = appendTo(document.body, 'div', 'luxar-overlay luxar-overlay--image');
    expect(rightClick(appendTo(overlay, 'img'))).toBe(true);
  });

  it('leaves a text field its native menu, which is the only way to paste', () => {
    const panel = appendTo(document.body, 'div', 'luxar-layers-panel');
    const input = appendTo(panel, 'input') as HTMLInputElement;
    input.type = 'text';
    expect(rightClick(input)).toBe(false);
    expect(rightClick(appendTo(panel, 'textarea'))).toBe(false);
    expect(rightClick(appendTo(panel, 'select'))).toBe(false);
    const editable = appendTo(panel, 'div');
    editable.setAttribute('contenteditable', 'true');
    expect(rightClick(editable)).toBe(false);
  });

  it('leaves DOM the viewer does not own alone', () => {
    // An embedder sharing document.body keeps their own menus: the default
    // container is body, whose control-rail marker must not claim every child.
    document.body.classList.add('luxar-has-control-rail');
    expect(rightClick(appendTo(document.body, 'div', 'host-app-sidebar'))).toBe(false);
    expect(rightClick(appendTo(document.body, 'div', 'my-luxar-panel'))).toBe(false);
  });

  it('keeps Copy available for selected text in an interactive overlay', () => {
    const overlay = appendTo(document.body, 'div', 'luxar-overlay luxar-overlay--interactive');
    const caption = appendTo(overlay, 'span');
    caption.textContent = 'Selectable accession';
    const range = document.createRange();
    range.selectNodeContents(caption);
    const selection = window.getSelection();
    selection?.removeAllRanges();
    selection?.addRange(range);

    expect(selection?.isCollapsed).toBe(false);
    expect(rightClick(caption)).toBe(false);
  });

  it('still fires when an inner handler stops propagation', () => {
    // The dimension sliders' play button calls stopPropagation on its own
    // menu; a bubble-phase listener would never see it.
    const panel = appendTo(document.body, 'div', 'luxar-dimension-sliders');
    const button = appendTo(panel, 'button');
    button.addEventListener('contextmenu', (e) => e.stopPropagation());
    expect(rightClick(button)).toBe(true);
  });

  it('removes its listener on dispose', () => {
    events.dispose();
    expect(rightClick(canvas)).toBe(false);
  });
});
