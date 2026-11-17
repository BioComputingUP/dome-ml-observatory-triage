#!/usr/bin/env bash
# Runs an enrichment cohort (prompt e3, the latest, on the EBI gateway) in a tmux session you can watch.
#
#   scripts/run_enrichment.sh core_bioinformatics                 # start (or resume) the whole cohort
#   scripts/run_enrichment.sh core_bioinformatics --concurrency 64
#   scripts/run_enrichment.sh core_bioinformatics --limit 500     # at most 500 new records per chunk
#   scripts/run_enrichment.sh core_bioinformatics --chunks c01,c02
#
# It opens tmux session `enrich-e3` with three windows:
#   0 run         the chunks in order, each a foreground `docker compose run` (never detached: AGENTS rule 5)
#   1 checkpoint  pushes events + provenance to the private runs repository every 15 minutes
#   2 status      live progress, speed, ETA, quality flags (scripts/enrich_status.py)
# Watch:   tmux attach -t enrich-e3     (Ctrl-b 0/1/2 switches window, Ctrl-b d detaches; the run continues)
# Or:      docker compose run --rm tools python scripts/enrich_status.py --cohort <cohort>
#
# Re-running the same command resumes: finished records are skipped (the events file is the state).
# The gateway must answer before each chunk and each retry: while it does not (VPN or Wi-Fi down, gateway
# restarting) the run waits, polling every 60 s, instead of spending its attempts. A chunk whose run
# crashes is retried up to 3 times, only after `docker ps` confirms no pipeline container is still
# working on it. The run holds a systemd sleep/idle inhibitor (where available) so the laptop stays awake. A finished chunk with parse errors or truncations above 2% stops
# the run for review.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
SESSION=enrich-e3

# Waits until the gateway lists its models (HTTP 200). The key goes to curl on stdin, never on its command line.
wait_for_gateway() {
  local code waited=0
  while :; do
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -H @- "$EBI_LLM_BASE_URL/models" \
              <<<"Authorization: Bearer $EBI_LLM_API_KEY" || true)"
    [[ "$code" == 200 ]] && { (( waited )) && echo "=== $(date -u +%FT%TZ) gateway reachable again" | tee -a "$LOG"; return 0; }
    (( waited % 600 == 0 )) && echo "=== $(date -u +%FT%TZ) gateway not reachable (HTTP $code; is the EBI VPN up?); waiting" | tee -a "$LOG"
    sleep 60; waited=$(( waited + 60 ))
  done
}

if [[ "${1:-}" == "--run-chunks" ]]; then
  shift; COHORT="$1"; shift
  set -a; . ./.env; set +a
  CONC=""; LIMIT=""; ONLY=""
  while [[ $# -gt 0 ]]; do case "$1" in
    --concurrency) CONC="$2"; shift 2;; --limit) LIMIT="$2"; shift 2;; --chunks) ONLY="$2"; shift 2;;
    *) echo "unknown option $1"; exit 2;; esac; done
  WORK="moros_pipeline/output/enrichment_e3/$COHORT"
  LOG="$WORK/run.log"
  for input in $(ls "$WORK"/*.csv | grep -v '_events\.csv$' | sort); do
    stem="$(basename "$input" .csv)"
    if [[ -n "$ONLY" ]] && ! grep -qE "_($(echo "$ONLY" | tr ',' '|'))$" <<<"$stem"; then continue; fi
    args=(dome-triage llm-classify enrich --input "$input" --events-out "$WORK/${stem}_events.csv")
    [[ -n "$CONC" ]] && args+=(--concurrency "$CONC")
    [[ -n "$LIMIT" ]] && args+=(--limit "$LIMIT")
    for attempt in 1 2 3 4; do
      wait_for_gateway
      echo "=== $(date -u +%FT%TZ) $stem attempt $attempt: ${args[*]}" | tee -a "$LOG"
      if docker compose run --rm pipeline "${args[@]}" 2>&1 | tee -a "$LOG"; then break; fi
      if [[ $attempt == 4 ]]; then echo "=== $stem failed 4 times; stopping for review" | tee -a "$LOG"; exit 1; fi
      while [[ -n "$(docker ps -q --filter label=com.docker.compose.service=pipeline)" ]]; do
        echo "=== a pipeline container is still running; waiting before retrying" | tee -a "$LOG"; sleep 30
      done
      sleep 20
    done
    if ! docker compose run --rm -T tools python scripts/enrich_status.py --cohort "$COHORT" --once --check >/dev/null; then
      echo "=== a finished chunk has parse errors or truncations above 2%; stopping for review" | tee -a "$LOG"
      exit 1
    fi
  done
  echo "=== $(date -u +%FT%TZ) cohort $COHORT: all selected chunks done" | tee -a "$LOG"
  docker compose run --rm -T -e GH_TOKEN tools python scripts/checkpoint_events.py \
    --cohort "$COHORT" | tee -a "$LOG" || true
  exit 0
fi

COHORT="${1:?usage: scripts/run_enrichment.sh <cohort> [--concurrency N] [--limit N] [--chunks c01,c02]}"; shift
set -a; . ./.env; set +a
RUNS="${ENRICHMENT_RUNS_REPO:-../dome-observatory-enrichment-runs}"
SRC="$RUNS/inputs/$COHORT"
WORK="moros_pipeline/output/enrichment_e3/$COHORT"
[[ -d "$SRC" ]] || { echo "no inputs at $SRC: build them with scripts/inputs_from_zenodo.py --cohort $COHORT"; exit 1; }
command -v tmux >/dev/null || { echo "tmux is required"; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session $SESSION already exists: tmux attach -t $SESSION"; exit 1; }
[[ -z "$(docker ps -q --filter label=com.docker.compose.service=pipeline)" ]] \
  || { echo "a pipeline container is already running (docker ps); not starting a second run"; exit 1; }

# The inputs must be exactly what the runs repository recorded.
(cd "$RUNS" && grep " inputs/$COHORT/" SHA256SUMS | sha256sum -c --quiet) \
  || { echo "inputs do not match $RUNS/SHA256SUMS; refusing"; exit 1; }
mkdir -p "$WORK"
for f in "$SRC"/*.csv; do
  dst="$WORK/$(basename "$f")"
  if [[ -e "$dst" ]]; then cmp -s "$f" "$dst" || { echo "$dst differs from $f; refusing"; exit 1; }
  else cp "$f" "$dst"; fi
done

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
# The token goes in through tmux's environment (-e), never on a command line that `ps` would show.
TMUXENV=(-e "HOST_UID=$HOST_UID" -e "HOST_GID=$HOST_GID" -e "GH_TOKEN=$(gh auth token)")
INHIBIT=""
command -v systemd-inhibit >/dev/null && INHIBIT="systemd-inhibit --what=sleep:idle --who=dome-enrichment --why='enrichment run' "
tmux new-session -d -s "$SESSION" -n run -c "$REPO" "${TMUXENV[@]}" \
  "${INHIBIT}bash scripts/run_enrichment.sh --run-chunks $COHORT $*; echo; echo '[run finished: press Enter]'; read"
tmux new-window -t "$SESSION" -n checkpoint -c "$REPO" "${TMUXENV[@]}" \
  "docker compose run --rm -e GH_TOKEN tools python scripts/checkpoint_events.py --cohort $COHORT --every 900"
tmux new-window -t "$SESSION" -n status -c "$REPO" "${TMUXENV[@]}" \
  "docker compose run --rm tools python scripts/enrich_status.py --cohort $COHORT"
tmux select-window -t "$SESSION:status"
echo "started: tmux attach -t $SESSION   (Ctrl-b 0 run, 1 checkpoint, 2 status; Ctrl-b d to detach)"
