### Fixed

- Increased the required TypeScript CI leg's timeout from 60 to 120 minutes so a
  saturated self-hosted queue has roughly the same tolerance as the Python leg,
  and corrected the runner-watchdog documentation to reflect that GitHub can
  charge a job timeout while the job is still queued.
