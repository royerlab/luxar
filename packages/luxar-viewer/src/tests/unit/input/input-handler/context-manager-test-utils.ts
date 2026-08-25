import {
  InputContext,
  InputContextManager,
  type KeyBinding,
} from '../../../../input/input-handler/context-manager';

export function registerTestBinding(
  manager: InputContextManager,
  context: InputContext,
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
