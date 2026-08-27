#### Bound stale obsidian queue backlogs (#2217)

CI now sends new long-running jobs to GitHub-hosted runners when the self-hosted
capacity heartbeat is stale and five obsidian jobs have already waited at least five
minutes. Fresh capacity stays on the zero-API-call path, the backlog scan is bounded,
and the queue cap can be tuned with `LUXAR_CI_MAX_QUEUED_OBSIDIAN`.
