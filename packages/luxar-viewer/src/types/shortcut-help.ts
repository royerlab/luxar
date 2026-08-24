export type ShortcutHelpSectionId = 'basics' | 'fly' | 'dimensions' | 'panels';

export interface ShortcutHelpMetadata {
  section: ShortcutHelpSectionId;
  group: string;
  keys: readonly string[];
  order: number;
}

export interface RegisteredShortcutBinding {
  actionId: string;
  actionParameter?: string | number;
  key: string;
  description: string;
  help: ShortcutHelpMetadata | false;
}

/** Registered keyboard bindings grouped by input-context identifier. */
export type RegisteredShortcutBindings = ReadonlyMap<
  string,
  readonly RegisteredShortcutBinding[]
>;
