# debug-console (internals)

Private helpers for the sibling public-API file `../debug-console.ts`
(the in-app developer console toggled with `Ctrl+L`). Extracted out of
the panel class so the pure logic is unit-testable without instantiating
the DOM.

## File Structure

```
debug-console/
└── formatters.ts   # arg stringifier, filter matcher, timestamp formatter
```

## `formatters.ts`

Three pure functions, all imported from `../debug-console.ts`:

| Export                               | Purpose                                                                                                                                                                                              |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `formatArgs(args)`                   | Stringify an arbitrary `unknown[]` console-arg list to a single space-separated display string. Objects pretty-print via `JSON.stringify(_, null, 2)`; falls back to `String(arg)` on circular refs. |
| `messageMatchesFilter(text, filter)` | Case-insensitive substring matcher for the panel's filter input. Returns `true` when `filter` is empty/whitespace-only.                                                                              |
| `formatConsoleTimestamp(date)`       | 24-hour `HH:mm:ss.SSS` timestamp via `toLocaleTimeString('en-US', …)`. Takes a `Date` so tests are deterministic.                                                                                    |

The same `formatArgs` output is used both as the row's visible text and
as the filter haystack, so the "filter by visible text" promise holds
exactly.

## See Also

- [`../README.md`](../README.md) — UI package overview (Debug Console section)
- `../debug-console.ts` — Sole consumer; class `DebugConsole` (toggled by `Ctrl+L`)
- [`../../core/app/debug/README.md`](../../core/app/debug/README.md) — `window.__luxarDebug.consoleInterceptor`, the buffered console history this panel renders
