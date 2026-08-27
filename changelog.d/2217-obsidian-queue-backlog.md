#### Bound stale obsidian queue backlogs (#2217)

CI now sends new long-running jobs to GitHub-hosted runners when the self-hosted
capacity heartbeat is stale and five obsidian jobs have already waited at least five
minutes, or when the ten-run scan bound is reached after finding any aged obsidian
backlog. Fresh capacity stays on the zero-API-call path, and the five-job cap can be
tuned with `LUXAR_CI_MAX_QUEUED_OBSIDIAN`.
