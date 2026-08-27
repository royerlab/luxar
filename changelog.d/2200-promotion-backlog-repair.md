#### Drain cancelled promotion checks across the dev backlog

Scheduled CI now repairs cancelled required checks for up to the two newest repairable
commits between `main` and the scheduled `dev` SHA, not only the commit that happened
to receive the cron window. Rerun concurrency is isolated by original workflow run so
those repairs cannot cancel each other once the original push run contains the new
policy.
