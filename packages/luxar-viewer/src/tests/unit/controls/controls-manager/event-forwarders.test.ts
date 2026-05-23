/**
 * Unit tests for controls-manager/event-forwarders.ts.
 *
 * Targets audit finding G4 (attachControlEventForwarders is only
 * exercised transitively via the orchestrator). Direct tests verify
 * the wiring + the disposer-registers-cleanup contract.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { attachControlEventForwarders } from '../../../../controls/controls-manager/event-forwarders';
import { EventGroup } from '../../../../utils/cross-layer/event-group';

type ChangeStartEndMap = { change: {}; start: {}; end: {} };

function makeFakeDispatcher(): THREE.EventDispatcher<ChangeStartEndMap> {
  return new THREE.EventDispatcher<ChangeStartEndMap>();
}

describe('attachControlEventForwarders', () => {
  it('forwards change events from controls to dispatch callback', () => {
    const controls = makeFakeDispatcher();
    const dispatch = vi.fn();
    const group = new EventGroup();

    attachControlEventForwarders(controls, dispatch, group);
    controls.dispatchEvent({ type: 'change' });

    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith('change');
  });

  it('forwards start events', () => {
    const controls = makeFakeDispatcher();
    const dispatch = vi.fn();
    const group = new EventGroup();

    attachControlEventForwarders(controls, dispatch, group);
    controls.dispatchEvent({ type: 'start' });

    expect(dispatch).toHaveBeenCalledWith('start');
  });

  it('forwards end events', () => {
    const controls = makeFakeDispatcher();
    const dispatch = vi.fn();
    const group = new EventGroup();

    attachControlEventForwarders(controls, dispatch, group);
    controls.dispatchEvent({ type: 'end' });

    expect(dispatch).toHaveBeenCalledWith('end');
  });

  it('group.dispose() detaches all three forwarders (no leak after dispose)', () => {
    const controls = makeFakeDispatcher();
    const dispatch = vi.fn();
    const group = new EventGroup();

    attachControlEventForwarders(controls, dispatch, group);
    // Pre-dispose: events flow through.
    controls.dispatchEvent({ type: 'change' });
    expect(dispatch).toHaveBeenCalledTimes(1);

    group.dispose();
    dispatch.mockClear();

    // Post-dispose: events MUST NOT flow through.
    controls.dispatchEvent({ type: 'change' });
    controls.dispatchEvent({ type: 'start' });
    controls.dispatchEvent({ type: 'end' });
    expect(dispatch).not.toHaveBeenCalled();
  });

  it('multiple events from the same source produce multiple dispatch calls', () => {
    const controls = makeFakeDispatcher();
    const dispatch = vi.fn();
    const group = new EventGroup();

    attachControlEventForwarders(controls, dispatch, group);
    for (let i = 0; i < 5; i++) controls.dispatchEvent({ type: 'change' });
    expect(dispatch).toHaveBeenCalledTimes(5);
  });

  it('does not fire dispatch for unknown event types', () => {
    const controls = makeFakeDispatcher();
    const dispatch = vi.fn();
    const group = new EventGroup();

    attachControlEventForwarders(controls, dispatch, group);
    // Cast to bypass typing; we want to ensure the helper only wires the
    // three named events.
    (controls as unknown as THREE.EventDispatcher<Record<string, {}>>).dispatchEvent({
      type: 'pointerdown',
    });
    expect(dispatch).not.toHaveBeenCalled();
  });
});
