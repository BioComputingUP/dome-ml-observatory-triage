# Workflows for the private runs repository

`enrichment-alerts.yml` is the GitHub side of the enrichment alerts (the laptop side is
`scripts/run_watchdog.py`). It lives in the PRIVATE runs repository as
`.github/workflows/enrichment-alerts.yml`; this folder is its source of truth.

The workflow opens issues as github-actions[bot], assigned to and mentioning the owner, so GitHub emails
them. An issue opened by your own token would not email you.

**Install or update.** Writing a workflow file needs a token with the `workflow` scope. Run
`gh auth refresh -h github.com -s workflow` once, then:

```bash
scripts/start_watchdog.sh --install
```

That uploads the file through the GitHub API. It never touches the local clone that the checkpoint
pusher commits from.

**Cost.** The dead-man schedule runs every 20 minutes, about 72 Actions minutes a day on a private
repository. When a run finishes, the workflow disables itself, and `run_watchdog.py` re-enables it at
the start of the next run.
