/**
 * Unit tests for the viewer-container registry — the single mount root the
 * viewer appends its overlays/panels into, and the containing-block promotion
 * it applies to a host-provided element.
 *
 * @see src/utils/viewer-container.ts
 */
import { afterEach, describe, expect, it } from 'vitest';
import {
  getViewerContainer,
  setViewerContainer,
  resetViewerContainer,
} from '../../../utils/viewer-container';

describe('viewer-container registry', () => {
  afterEach(() => {
    // Always return the global back to the default so one test can't leak
    // a custom container (or its mutated styles) into the next.
    resetViewerContainer();
  });

  it('defaults to document.body', () => {
    expect(getViewerContainer()).toBe(document.body);
  });

  it('points at a host-provided element after setViewerContainer()', () => {
    const host = document.createElement('div');
    setViewerContainer(host);
    expect(getViewerContainer()).toBe(host);
  });

  it('promotes a non-body container to a containing block', () => {
    const host = document.createElement('div');
    setViewerContainer(host);
    // contain: layout makes it the containing block for fixed/absolute
    // descendants without containing size.
    expect(host.style.contain).toBe('layout');
  });

  it('sets position:relative only when the container is statically positioned', () => {
    const staticHost = document.createElement('div');
    staticHost.style.position = 'static';
    setViewerContainer(staticHost);
    expect(staticHost.style.position).toBe('relative');
    resetViewerContainer();

    const absoluteHost = document.createElement('div');
    absoluteHost.style.position = 'absolute';
    setViewerContainer(absoluteHost);
    // Already a containing block — must NOT be downgraded to relative.
    expect(absoluteHost.style.position).toBe('absolute');
  });

  it('does NOT mutate styles when the container is document.body', () => {
    setViewerContainer(document.body);
    expect(document.body.style.contain).toBe('');
  });

  it('restores the prior inline styles on reset', () => {
    const host = document.createElement('div');
    host.style.position = 'static';
    host.style.contain = 'paint';
    setViewerContainer(host);
    // We mutated both.
    expect(host.style.position).toBe('relative');
    expect(host.style.contain).toBe('layout');

    resetViewerContainer();
    // Exactly the embedder's authored values are back.
    expect(host.style.position).toBe('static');
    expect(host.style.contain).toBe('paint');
    expect(getViewerContainer()).toBe(document.body);
  });

  it('removes our inline props entirely when none were authored', () => {
    const host = document.createElement('div');
    setViewerContainer(host);
    expect(host.style.contain).toBe('layout');

    resetViewerContainer();
    // No inline declaration left behind — not an empty `contain:;`.
    expect(host.getAttribute('style') ?? '').not.toContain('contain');
  });

  it('restores the previous container styles when swapped directly', () => {
    const first = document.createElement('div');
    setViewerContainer(first);
    expect(first.style.contain).toBe('layout');

    const second = document.createElement('div');
    setViewerContainer(second);
    // Swapping to a new container must clean up the previous one.
    expect(first.style.contain).toBe('');
    expect(second.style.contain).toBe('layout');
    expect(getViewerContainer()).toBe(second);
  });
});
