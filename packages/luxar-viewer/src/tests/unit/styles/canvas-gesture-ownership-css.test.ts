/**
 * The standalone page's canvas must declare `touch-action: none` in CSS (the
 * fast path, before any script runs) in `styles/base/layout.css`; the JS stamp
 * in the app covers embedder-supplied canvases. Without it a two-finger pinch is page zoom and
 * the orbit controls' touch handlers never run (see
 * `core/app/interaction/canvas-gesture-ownership.ts`).
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ruleBody, stripComments } from './_helpers/css-text';

const HERE = dirname(fileURLToPath(import.meta.url));
const STYLES_ROOT = resolve(HERE, '../../../styles');

describe('standalone canvas gesture ownership (CSS)', () => {
  it('layout.css #app owns touch gestures', () => {
    const layout = stripComments(readFileSync(resolve(STYLES_ROOT, 'base/layout.css'), 'utf8'));
    const app = ruleBody(layout, '#app');
    expect(app).toMatch(/touch-action:\s*none/);
    expect(app).toMatch(/-webkit-touch-callout:\s*none/);
    expect(app).toMatch(/user-select:\s*none/);
  });
});
