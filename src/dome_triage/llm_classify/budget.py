"""USD spend cap for Step 20's DeepSeek second-curator work -- a separate sub-cap layered under
(not instead of) AGENTS.md/pipeline.yaml's existing project-wide GBP `budget.total_cap_gbp` check,
in its own currency-unambiguous log file rather than mixed into the shared GBP
`compute_spend_log.csv`. Every real-money `llm-classify` command (`validate-criteria`, `calibrate`,
`calibrate-fallback`, `classify`, `run-fallback`) calls `check_cap()` before spending and
`log_spend()` after, so the cumulative check is honest from the very first live call, not just the
"big" classify runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

SPEND_LOG_COLUMNS = [
    "timestamp",
    "step_name",
    "tier",
    "mode",
    "n_records",
    "estimated_usd",
    "actual_usd",
    "confirmed_by",
]


class BudgetExceededError(Exception):
    pass


def _cumulative_spend_usd(spend_log_path: Path) -> float:
    path = Path(spend_log_path)
    if not path.exists():
        return 0.0
    log = pd.read_csv(path)
    if log.empty:
        return 0.0
    # actual_usd, once known, is a better record of real spend than the estimate that preceded it
    # -- fall back to estimated_usd only for a row where the actual figure was never filled in.
    actual = pd.to_numeric(log["actual_usd"], errors="coerce")
    estimated = pd.to_numeric(log["estimated_usd"], errors="coerce")
    spent = actual.fillna(estimated).fillna(0.0)
    return float(spent.sum())


def check_cap(
    spend_log_path: Path, total_cap_usd: float, projected_additional_usd: float, confirmed: bool
) -> None:
    """Raises `BudgetExceededError` unless `confirmed=True` -- the CLI layer is responsible for
    printing the cost projection (via `cost_estimator.project_cost`) before it ever passes
    `confirmed=True` through here; this function itself has no way to verify the projection was
    actually shown, so callers must not skip that step. Does not touch pipeline.yaml's existing
    GBP project-wide cap -- both checks apply independently, in different currencies."""
    cumulative = _cumulative_spend_usd(spend_log_path)
    projected_total = cumulative + projected_additional_usd
    if projected_total > total_cap_usd and not confirmed:
        raise BudgetExceededError(
            f"Projected spend ${projected_total:.2f} (${cumulative:.2f} already spent + "
            f"${projected_additional_usd:.2f} projected) would exceed the ${total_cap_usd:.2f} "
            "DeepSeek sub-cap. Review the printed cost projection, then re-run with --confirm to "
            "proceed anyway."
        )
    if projected_total > total_cap_usd and confirmed:
        print(
            f"WARNING: proceeding past the ${total_cap_usd:.2f} DeepSeek sub-cap (projected total "
            f"${projected_total:.2f}) -- --confirm was passed explicitly."
        )


def log_spend(
    spend_log_path: Path,
    step_name: str,
    tier: str,
    mode: str,
    n_records: int,
    estimated_usd: Optional[float],
    actual_usd: Optional[float],
    confirmed_by: str,
) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step_name": step_name,
        "tier": tier,
        "mode": mode,
        "n_records": n_records,
        "estimated_usd": estimated_usd,
        "actual_usd": actual_usd,
        "confirmed_by": confirmed_by,
    }
    spend_log_path = Path(spend_log_path)
    spend_log_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not spend_log_path.exists()
    pd.DataFrame([row], columns=SPEND_LOG_COLUMNS).to_csv(
        spend_log_path, mode="a", header=header_needed, index=False
    )
