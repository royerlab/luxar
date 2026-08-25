/**
 * Shared binding fixtures for InputContextManager unit tests.
 *
 * @module tests/unit/input/input-handler/context-manager-test-utils
 */

import {
  InputContext,
  InputContextManager,
  type KeyBinding,
} from '../../../../input/input-handler/context-manager';

/** Register a binding with stable test defaults for required registry metadata. */
export function registerTestBinding(
  manager: InputContextManager,
  context: InputContext | string,
  binding: Omit<KeyBinding, 'actionId' | 'description' | 'help'> &
    Partial<Pick<KeyBinding, 'actionId' | 'description' | 'help'>>
): void {
  manager.registerBinding(context, {
    ...binding,
    actionId: binding.actionId ?? `test.${binding.key}.${JSON.stringify(binding.modifiers ?? {})}`,
    description: binding.description ?? 'Test binding',
    help: binding.help ?? false,
  });
}
