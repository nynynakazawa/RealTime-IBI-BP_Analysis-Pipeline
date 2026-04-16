from __future__ import annotations

"""
Baseline-family evaluation series are preserved for future research.

Default behavior:
- disabled
- realtime session evaluation runs only RTBP / SinBP_D / SinBP_M
"""

ENABLE_BASELINE_EXPERIMENTAL_SERIES_DEFAULT = False

BASELINE_EXPERIMENTAL_SERIES: tuple[str, ...] = (
    "INITIAL_BASELINE",
    "RICH_BASELINE",
    "SHARED_D_BASELINE",
    "RICH_DYNAMIC",
)

