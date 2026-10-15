// @vitest-environment jsdom
/**
 * Scene/demo overlays anchored at the LEFT edge clear the control rail on
 * desktop too (the coarse-pointer variant already did), using the same
 * `--luxar-rail-gutter` as the docked panels; right- and centre-anchored
 * overlays are never shifted.
 *
 * Asserted on the CSS text (jsdom does no layout), plus the anchor classes the
 * overlay manager puts on each element, which the selector keys on.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ruleBody, stripComments } from './_helpers/css-text';
import { OverlayManager } from '../../../ui/overlay-manager';

const HERE = dirname(fileURLToPath(import.meta.url));
const overlayCss = stripComments(
  readFileSync(resolve(HERE, '../../../styles/components/overlay-layer.css'), 'utf8')
);

const LEFT_ANCHORED =
  '.luxar-has-control-rail .luxar-overlay:not(.luxar-overlay--right-anchored, .luxar-overlay--center-anchored)';

describe('overlay rail gutter (desktop)', () => {
  it('clamps left-anchored overlays to the rail gutter while the rail is mounted', () => {
    expect(ruleBody(overlayCss, LEFT_ANCHORED)).toMatch(
      /left:\s*max\(\s*var\(--luxar-overlay-x\),\s*var\(--luxar-rail-gutter\)\s*\)/
    );
  });

  it('caps an explicit-width left-anchored text overlay to the room right of its clamped edge', () => {
    const rule = overlayCss.match(
      /\.luxar-has-control-rail\s+\.luxar-overlay--text\.luxar-overlay--explicit-width:not\(\s*\.luxar-overlay--right-anchored,\s*\.luxar-overlay--center-anchored\s*\)\s*\{([^}]*)\}/
    );
    expect(rule).not.toBeNull();
    expect(rule![1]).toMatch(
      /max-width:\s*calc\(\s*100%\s*-\s*max\(\s*var\(--luxar-overlay-x\),\s*var\(--luxar-rail-gutter\)\s*\)\s*-\s*12px\s*\)/
    );
  });

  it('is not inside a media query (applies to fine pointers)', () => {
    const at = overlayCss.indexOf('.luxar-has-control-rail');
    expect(at).toBeGreaterThan(-1);
    expect(overlayCss.slice(0, at)).not.toMatch(/@media/);
  });

  it('keeps the unclamped base position for every overlay', () => {
    expect(ruleBody(overlayCss, '.luxar-overlay')).toMatch(/left:\s*var\(--luxar-overlay-x\)/);
  });
});

describe('overlay anchor classes the gutter selector keys on', () => {
  async function classesFor(anchor: string): Promise<DOMTokenList> {
    const manager = new OverlayManager();
    await manager.loadOverlays(
      [
        {
          name: `o-${anchor}`,
          type: 'overlay_text',
          text: 'Title',
          position: [0.02, 0.05],
          anchor,
          opacity: 1,
          transition: 'none',
          transition_duration: 0,
          interactive: false,
          z_index: 0,
        },
      ],
      'http://localhost/'
    );
    const el = document.querySelector<HTMLElement>('.luxar-overlay');
    expect(el).not.toBeNull();
    const classes = el!.classList;
    manager.dispose();
    return classes;
  }

  it('marks left anchors with neither exclusion class', async () => {
    for (const anchor of ['top-left', 'center-left', 'bottom-left']) {
      const classes = await classesFor(anchor);
      expect(classes.contains('luxar-overlay--right-anchored'), anchor).toBe(false);
      expect(classes.contains('luxar-overlay--center-anchored'), anchor).toBe(false);
    }
  });

  it('marks right and centre anchors so the clamp skips them', async () => {
    for (const anchor of ['top-right', 'center-right', 'bottom-right']) {
      const classes = await classesFor(anchor);
      expect(classes.contains('luxar-overlay--right-anchored'), anchor).toBe(true);
    }
    for (const anchor of ['top-center', 'center', 'bottom-center']) {
      const classes = await classesFor(anchor);
      expect(classes.contains('luxar-overlay--center-anchored'), anchor).toBe(true);
    }
  });
});
