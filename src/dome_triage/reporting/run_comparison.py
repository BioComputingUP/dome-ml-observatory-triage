"""A/B comparison of two classification event logs over the records they share.

Purpose: prove that an infrastructure change (concurrency, connection pooling, a different output
path) did not change what the model actually decides. Any residual disagreement is DeepSeek's own
run-to-run nondeterminism at temperature > 0, not a pipeline difference -- which is exactly why
this reports the real disagreement rate rather than asserting bit-identity.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _latest_per_record(df: pd.DataFrame) -> pd.DataFrame:
    """Last event wins, matching the convention every other event log in this project uses."""
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df.drop_duplicates(subset=["record_id"], keep="last")


def compare_runs(path_a: Path, path_b: Path) -> dict:
    a = _latest_per_record(pd.read_csv(path_a, dtype=str))
    b = _latest_per_record(pd.read_csv(path_b, dtype=str))

    merged = a.merge(b, on="record_id", suffixes=("_a", "_b"), how="inner")
    n = len(merged)
    if n == 0:
        return {
            "n_a": len(a), "n_b": len(b), "n_shared": 0, "n_agree": 0,
            "agreement_rate": None, "disagreements": pd.DataFrame(),
        }

    agree = merged["classification_a"] == merged["classification_b"]
    disagreements = merged.loc[~agree, ["record_id", "classification_a", "classification_b"]]

    # Same prompt+criteria on both sides is the precondition for the comparison to mean anything.
    same_criteria = set(merged["criteria_sha256_a"]) == set(merged["criteria_sha256_b"])
    same_prompt = set(merged["prompt_version_a"]) == set(merged["prompt_version_b"])

    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_shared": n,
        "n_agree": int(agree.sum()),
        "agreement_rate": float(agree.mean()),
        "same_criteria_sha256": same_criteria,
        "same_prompt_version": same_prompt,
        "crosstab": pd.crosstab(merged["classification_a"], merged["classification_b"]),
        "disagreements": disagreements,
    }


def print_comparison(result: dict, label_a: str, label_b: str) -> None:
    print(f"\n=== run comparison: {label_a}  vs  {label_b} ===")
    print(f"  records in A: {result['n_a']:,}   in B: {result['n_b']:,}   shared: {result['n_shared']:,}")
    if not result["n_shared"]:
        print("  NO SHARED RECORDS -- nothing to compare.")
        return
    print(f"  identical classifications: {result['n_agree']:,}/{result['n_shared']:,} "
          f"({result['agreement_rate']:.2%})")
    print(f"  same criteria_sha256: {result['same_criteria_sha256']}   "
          f"same prompt_version: {result['same_prompt_version']}")
    print("\n  crosstab (rows = A, cols = B):")
    print(result["crosstab"].to_string())
    d = result["disagreements"]
    if len(d):
        print(f"\n  {len(d):,} disagreement(s) -- first 20:")
        print(d.head(20).to_string(index=False))
    else:
        print("\n  Zero disagreements -- byte-identical decisions on every shared record.")
