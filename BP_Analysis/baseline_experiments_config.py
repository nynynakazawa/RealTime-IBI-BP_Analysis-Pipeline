from __future__ import annotations

"""
Baseline adaptation experiments are kept for future research.

Default behavior:
- disabled in the main realtime coefficient pipeline
- core evaluation runs only RTBP / SinBP_D / SinBP_M on
  current_app_smoothed / refit_map_pp_smoothed / leave_one_session_out
"""

# Keep disabled by default in production-like reruns.
ENABLE_BASELINE_EXPERIMENTS_DEFAULT = False

# Baseline-family series are preserved for optional offline research.
BASELINE_SERIES_ORDER: tuple[str, ...] = (
    "smartphone_initial_baseline",
    "smartphone_rich_baseline",
    "smartphone_rich_dynamic_blend",
    "smartphone_shared_sinbpd_baseline",
    "smartphone_shared_sinbpd_baseline_loso",
    "smartphone_rich_dynamic_blend_loso",
    "smartphone_rich_baseline_loso",
    "smartphone_initial_baseline_loso",
)

