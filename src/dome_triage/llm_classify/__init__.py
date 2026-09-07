"""Step 20: independent, blind DeepSeek second-curator classification + agreement metrics.

Never imported by pipeline/steps.py at module scope in a way that pulls torch into the CLI's
common import path (see AGENTS.md/bulk_scores.py's precedent) -- this package has no torch
dependency of its own, just requests + pandas.
"""
