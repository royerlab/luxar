# `input-handler/context-manager/`

Pure helpers used only by `../context-manager.ts`.

- `routing-rules.ts` — two pure functions plus the minimal structural
  interfaces (`KeyFilterConfig`, `PriorityConfig`) they consume. The
  full `ContextConfig` in the parent satisfies both shapes.
  - `isKeyAllowedInContext(key, config, bindingKey?)` — allow/block
    filter evaluation. `bindingKey` is the canonical modifier-aware
    chord and defaults to the normalized base key. Allowlist entries
    match either the base key or exact chord; blocklist entries match
    only the exact chord, so blocking `w` does not block `Ctrl+W`.
    Chord spelling and modifier order are normalized. An absent
    `allowedKeys` means "allow all"; an empty array means "allow none".
    `blockedKeys` always wins over `allowedKeys` (defense in depth).
  - `sortContextsByPriority(configs, excludeContext?)` — returns the
    context map's entries sorted by descending `priority` (missing
    priority treated as 0), optionally skipping the currently-active
    context. Stable for equal priorities. Drives the passthrough
    fallback loop: when the active context doesn't claim a key, the
    remaining contexts are tried in priority order until one accepts
    it.

Extracted so they can be unit-tested without setting up a full
`InputContextManager` instance with bindings, listeners, and an event
source. The matching suite lives at
`tests/unit/input/input-handler/context-manager/routing-rules.test.ts`.

The parent dispatcher treats handlers as accepting an event unless they return
`false` synchronously. A declined event continues through passthrough without
calling `preventDefault`; async handlers are always accepted. Keyup routing
likewise skips bindings without a `keyupHandler` and keeps searching.
