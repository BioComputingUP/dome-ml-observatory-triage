"""Standalone charts for `fetch_licensing.py`'s output -- same "not part of dome_triage, not run
through Docker" scope as that script. Only pandas/matplotlib, run directly with `python3`.

Produces, in `output/`:
- `license_breakdown.png` -- count of each real license string EPMC returned (e.g. "cc by",
  "cc by-nc"), plus an explicit "no license (non-OA / not disclosed)" bucket -- the real
  distribution needed before Phase 8's download/redistribution feature can be scoped.
- `open_access_breakdown.png` -- open-access vs. not, the coarser cut of the same data.
- `licensing_summary.json` -- the real counts behind both charts, plus fetch-coverage (how many of
  the target PMIDs actually got a result back from EPMC at all).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_LICENSING_PATH = THIS_DIR / "output" / "epmc_pmid_licensing.csv"
DEFAULT_TARGET_PATH = THIS_DIR.parent / "data" / "interim" / "bulk_candidates.csv"
DEFAULT_OUTPUT_DIR = THIS_DIR / "output"

_NO_LICENSE_LABEL = "no license (non-OA / not disclosed)"


def _add_caption(fig, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=8, wrap=True, color="#444444")
    fig.subplots_adjust(bottom=0.22)


def load_and_normalize(licensing_path: Path) -> pd.DataFrame:
    df = pd.read_csv(licensing_path, dtype=str)
    df["license_display"] = df["license"].fillna("").str.strip()
    df.loc[df["license_display"] == "", "license_display"] = _NO_LICENSE_LABEL
    df["is_open_access"] = df["is_open_access"].fillna("N")
    return df


def plot_license_breakdown(df: pd.DataFrame, output_path: Path, top_n: int = 12) -> pd.Series:
    counts = df["license_display"].value_counts()
    top = counts.head(top_n)
    if len(counts) > top_n:
        other_count = counts.iloc[top_n:].sum()
        top = pd.concat([top, pd.Series({"(other license types)": other_count})])

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(top) + 2)))
    order = top.sort_values(ascending=True)
    bars = ax.barh(order.index, order.values, color="#4C72B0")
    total = int(counts.sum())
    for bar, value in zip(bars, order.values):
        pct = value / total * 100 if total else 0.0
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {int(value):,} ({pct:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("Records")
    ax.set_title(f"EPMC AI/ML Search-Space License Breakdown (n={total:,} fetched)")
    _add_caption(
        fig,
        "Real license strings as returned by the Europe PMC API (resultType=core). "
        f'"{_NO_LICENSE_LABEL}" covers every non-open-access record, where EPMC discloses no '
        "reuse license at all -- these are not redistributable under any of these terms.",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


def plot_open_access_breakdown(df: pd.DataFrame, output_path: Path) -> pd.Series:
    counts = df["is_open_access"].value_counts()
    labels = {"Y": "Open Access", "N": "Not Open Access"}
    display_counts = counts.rename(index=lambda k: labels.get(k, k))

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#55A868" if idx == "Open Access" else "#C44E52" for idx in display_counts.index]
    total = int(display_counts.sum())
    wedges, _texts, autotexts = ax.pie(
        display_counts.values,
        labels=display_counts.index,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total)):,})",
        colors=colors,
        startangle=90,
    )
    ax.set_title(f"EPMC AI/ML Search-Space: Open Access Vs. Not (n={total:,})")
    _add_caption(
        fig,
        "Open Access alone does not mean redistributable under any license -- see "
        "license_breakdown.png for the real per-license split within the Open Access slice.",
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return counts


def compute_coverage(licensing_path: Path, target_path: Path) -> dict:
    target = pd.read_csv(target_path, dtype=str, usecols=["pmid"])
    target_pmids = set(target["pmid"].dropna())
    fetched = pd.read_csv(licensing_path, dtype=str, usecols=["pmid"])
    fetched_pmids = set(fetched["pmid"].dropna())
    return {
        "n_target_pmids": len(target_pmids),
        "n_fetched_pmids": len(fetched_pmids),
        "n_missing_pmids": len(target_pmids - fetched_pmids),
        "coverage_rate": len(fetched_pmids) / len(target_pmids) if target_pmids else 0.0,
    }


def run(licensing_path: Path, target_path: Path, output_dir: Path) -> None:
    if not licensing_path.exists():
        raise FileNotFoundError(f"{licensing_path} does not exist -- run fetch_licensing.py first.")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_normalize(licensing_path)

    license_counts = plot_license_breakdown(df, output_dir / "license_breakdown.png")
    oa_counts = plot_open_access_breakdown(df, output_dir / "open_access_breakdown.png")
    coverage = compute_coverage(licensing_path, target_path) if target_path.exists() else {}

    summary = {
        "n_fetched": int(len(df)),
        "license_counts": license_counts.to_dict(),
        "open_access_counts": oa_counts.to_dict(),
        "coverage": coverage,
    }
    summary_path = output_dir / "licensing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"visualize_licensing: {len(df):,} records summarized.")
    if coverage:
        print(
            f"visualize_licensing: coverage {coverage['n_fetched_pmids']:,}/{coverage['n_target_pmids']:,} "
            f"({coverage['coverage_rate'] * 100:.1f}%) of the target PMID pool -- "
            f"{coverage['n_missing_pmids']:,} not yet fetched or not found in EPMC."
        )
    print(f"visualize_licensing: wrote license_breakdown.png, open_access_breakdown.png, "
          f"licensing_summary.json to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--licensing", type=Path, default=DEFAULT_LICENSING_PATH)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH, help="For coverage stats only.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.licensing, args.target, args.output_dir)
