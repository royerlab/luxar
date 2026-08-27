#### Drain cancelled promotion checks across the dev backlog

Scheduled CI now repairs cancelled required checks for every commit still between
`main` and the scheduled `dev` SHA, not only the commit that happened to receive the
cron window. Rerun concurrency is isolated by original workflow run so those backlog
repairs cannot cancel each other.
