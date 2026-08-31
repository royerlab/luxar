#### Keep same-repo CI on obsidian

Same-repo Python and TypeScript CI now stays on the self-hosted `obsidian`
runner without automatic capacity or backlog overflow to paid runners. Forks and
the `LUXAR_CI_FORCE_HOSTED=1` operator break-glass remain hosted; the obsolete
queue redispatcher was removed because a fresh attempt would rejoin the same queue.
