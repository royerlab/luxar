#### Re-dispatch CI runs stranded behind saturated obsidian

A bounded scheduled watchdog now cancels and fully reruns one first-attempt workflow whose required obsidian work has remained wholly queued for 30 minutes while other obsidian jobs are active. The durable hosted handoff makes the rerun re-evaluate runner routing without risking a cancelled required leg or an automatic retry loop.
