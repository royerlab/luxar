/**
 * Cycle the data-loading monitor through its visibility states
 * (hidden → mini → expanded → hidden). Pulled out of input-handler.ts
 * so the orchestrator only owns the binding wiring.
 *
 * @module input/input-handler/commands/data-monitor-cycle
 */

import { eventBus } from '../../../utils/cross-layer/event-bus';
import { log, Modules } from '../../../utils/log';

export function cycleDataMonitor(): void {
  eventBus.emit('panel-cycle', { panelId: 'data-monitor' });
  log.info(Modules.DATA_MONITOR, 'Data loading monitor cycled');
}
