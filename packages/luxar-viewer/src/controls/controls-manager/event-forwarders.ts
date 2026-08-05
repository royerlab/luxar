/**
 * Wire the change / start / end events of an active control instance up
 * to a manager-level dispatcher. Extracted from `controls-manager.ts`
 * so the orchestrator stays focused on lifecycle.
 *
 * The dispatch site stays on the orchestrator: the caller passes a
 * `dispatch` callback that invokes `this.dispatchEvent({ type })` on the
 * manager. The helper only owns the listener-attach/detach plumbing.
 */

import * as THREE from 'three';
import { EventGroup } from '../../utils/cross-layer/event-group';

/** Events a control instance emits and the manager re-dispatches. */
export type ControlEventName = 'change' | 'start' | 'end';

type ControlEventMap = {
  change: {};
  start: {};
  end: {};
};

/** A control instance viewed as a dispatcher of change/start/end events. */
export type ControlEventDispatcher = THREE.EventDispatcher<ControlEventMap>;

/**
 * Attach change / start / end listeners to `controls` that re-dispatch
 * via `dispatch`, registering matching detach callbacks with `group`.
 *
 * THREE.EventDispatcher isn't a DOM EventTarget so EventGroup.on()
 * doesn't apply; register manual cleanup callbacks instead.
 */
export function attachControlEventForwarders(
  controls: ControlEventDispatcher,
  dispatch: (type: ControlEventName) => void,
  group: EventGroup
): void {
  const change = (): void => dispatch('change');
  const start = (): void => dispatch('start');
  const end = (): void => dispatch('end');

  controls.addEventListener('change', change);
  controls.addEventListener('start', start);
  controls.addEventListener('end', end);

  group.add(() => controls.removeEventListener('change', change));
  group.add(() => controls.removeEventListener('start', start));
  group.add(() => controls.removeEventListener('end', end));
}
