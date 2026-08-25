/** Stable identifier for a keyboard-shortcut help section. */
export type ShortcutHelpSectionId = 'basics' | 'fly' | 'dimensions' | 'panels';

/** Presentation metadata for one grouped row in the shortcut overlay. */
export interface ShortcutHelpMetadata {
  section: ShortcutHelpSectionId;
  group: string;
  keys?: readonly string[];
  order: number;
}

/** Registry metadata needed to render and inspect one keyboard binding. */
export interface RegisteredShortcutBinding {
  actionId: string;
  actionParameter?: string | number;
  key: string;
  /** User-facing label derived from the registered chord. */
  shortcutLabel?: string;
  description: string;
  help: ShortcutHelpMetadata | false;
}

/** Registered keyboard bindings grouped by input-context identifier. */
export type RegisteredShortcutBindings = ReadonlyMap<string, readonly RegisteredShortcutBinding[]>;
