# Data Loading Monitor Templates

HTML-string renderers are split by the UI concern that owns them. Consumers
import the owning module directly; this directory intentionally has no barrel.

| Module           | Responsibility                                                                                                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `primitives.ts`  | Shared icons, color classes, metric cards, progress bars, and stat grids.                                                                                                              |
| `format.ts`      | Number, byte, and memory-pressure formatting built on the shared color primitives.                                                                                                     |
| `overview.ts`    | Overview-tab loader, summary, secondary-metric, and failed-load markup.                                                                                                                |
| `cache.ts`       | Cache-tab markup, section metadata, status badges, and hit-rate color helpers.                                                                                                         |
| `memory.ts`      | Memory-tab GPU-pool and accumulator markup plus reuse-rate helpers.                                                                                                                    |
| `insights.ts`    | Insights-tab recommendation and all-clear markup.                                                                                                                                      |
| `scene-graph.ts` | Scene-tree markup, LOD summaries, draw-order and density chips (a `MONITOR_ICONS` glyph carries the bucket / thinning meaning, the tooltip the full explanation), and node statistics. |

The dependency direction is `primitives.ts` ← `format.ts` ← concern renderers;
concern modules do not import one another. Cache hit-rate helpers live in
`cache.ts`, rather than `memory.ts`, because the Cache tab owns their only
production consumer.

All static styling belongs in `styles/components/data-loading-monitor.css`;
templates use CSS classes rather than inline styles, with `luxar-color--*`
modifier classes for dynamic colors.

The Cache tab's markup is an incremental-update contract. `tabs/cache.ts`
patches elements by stable `data-field` selectors and replaces
`luxar-color--*` classes in place, so changing those attributes or class names
requires corresponding updater and test changes.
