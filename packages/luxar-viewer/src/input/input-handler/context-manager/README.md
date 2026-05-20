# `input-handler/context-manager/`

Pure helpers used only by `../context-manager.ts`.

- `routing-rules.ts` — `isKeyAllowedInContext` (allow/block filter
  evaluation) + `sortContextsByPriority` (descending-priority ordering
  for passthrough fallback). Extracted so they can be unit-tested
  without setting up a full `InputContextManager` instance with
  bindings, listeners, and an event source.
