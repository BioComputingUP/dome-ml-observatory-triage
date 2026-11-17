#!/usr/bin/env bash
# Starts the enrichment alert watchdog (scripts/run_watchdog.py) in its own tmux session, `enrich-watch`,
# separate from the run's session so starting or stopping it never touches the run.
#
#   scripts/start_watchdog.sh batch2_high_impact     # start watching a cohort
#   scripts/start_watchdog.sh --test batch2_high_impact   # send one test alert (an issue -> email) and exit
#   scripts/start_watchdog.sh --install              # upload the workflow to the private runs repository
#                                                    # (needs: gh auth refresh -h github.com -s workflow)
# Watch it: tmux attach -t enrich-watch   (Ctrl-b d detaches)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
set -a; . ./.env; set +a
RUNS="${ENRICHMENT_RUNS_REPO:-../dome-observatory-enrichment-runs}"
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"

slug() { git -C "$RUNS" remote get-url origin | sed -E 's#.*github\.com[:/]([^/]+/[^/.]+)(\.git)?$#\1#'; }

if [[ "${1:-}" == "--install" ]]; then
  S="$(slug)"
  [[ "$(gh api "repos/$S" --jq .private)" == "true" ]] || { echo "refusing: $S is not private"; exit 1; }
  P=.github/workflows/enrichment-alerts.yml
  SHA="$(gh api "repos/$S/contents/$P" --jq .sha 2>/dev/null || true)"
  args=(-X PUT "repos/$S/contents/$P" -f message="enrichment-alerts workflow (from dome-ml-observatory-triage)"
        -f content="$(base64 -w0 scripts/runs_repo_workflows/enrichment-alerts.yml)")
  [[ -n "$SHA" ]] && args+=(-f sha="$SHA")
  gh api "${args[@]}" --jq '.commit.sha' && echo "installed $P in $S"
  exit 0
fi

TEST=""
[[ "${1:-}" == "--test" ]] && { TEST="--test"; shift; }
COHORT="${1:?usage: scripts/start_watchdog.sh [--test] <cohort> | --install}"
if [[ -n "$TEST" ]]; then
  GH_TOKEN="$(gh auth token)" docker compose run --rm -T -e GH_TOKEN tools \
    python scripts/run_watchdog.py --cohort "$COHORT" --test && echo "test alert sent: check your email"
  exit 0
fi
tmux has-session -t enrich-watch 2>/dev/null && { echo "already running: tmux attach -t enrich-watch"; exit 1; }
tmux new-session -d -s enrich-watch -c "$REPO" -e "HOST_UID=$HOST_UID" -e "HOST_GID=$HOST_GID" \
  -e "GH_TOKEN=$(gh auth token)" \
  "docker compose run --rm -e GH_TOKEN tools python scripts/run_watchdog.py --cohort $COHORT; echo; echo '[watchdog stopped: press Enter]'; read"
echo "watchdog started: tmux attach -t enrich-watch"
