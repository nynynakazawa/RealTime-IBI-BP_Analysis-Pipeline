from __future__ import annotations

from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_ROOT.parent
DATA_ROOT = ANALYSIS_ROOT / "Data"

REALTIME_SESSIONS_ROOT = DATA_ROOT / "realtime_sessions"
CNAP_AUXILIARY_ROOT = DATA_ROOT / "pdp" / "realtime_aux"
AROB_TRACKING_ROOT = DATA_ROOT / "arob_tracking"
REALTIME_COEFFICIENT_ROOT = DATA_ROOT / "realtime_coefficient"

CURRENT_AROB_TRACKING_RUN = AROB_TRACKING_ROOT / "tracking_eval_20260415_112219"
CURRENT_REALTIME_COEFFICIENT_RUN = REALTIME_COEFFICIENT_ROOT / "realtime_map_pp_fit_20260414_160022"

PRIMARY_TASK_DEFINITION = "within-session blood pressure tracking"
PRIMARY_METRICS: tuple[str, ...] = (
    "centered MAE",
    "delta correlation",
    "detrended correlation",
)

LIVE_APP_METHOD_NAMES: tuple[str, ...] = (
    "RTBP",
    "SinBP_D",
    "SinBP_M",
)

PAPER_CORE_METHOD_NAMES: tuple[str, ...] = (
    "RTBP",
    "SinBP_M",
    "SinBP_D",
)

PAPER_SUPPLEMENTAL_METHOD_NAMES: tuple[str, ...] = (
    "SinBP_D_PPShapeC",
)

PAPER_METHOD_NAMES: tuple[str, ...] = PAPER_CORE_METHOD_NAMES + PAPER_SUPPLEMENTAL_METHOD_NAMES

DIAGNOSTIC_METHOD_NAMES: tuple[str, ...] = (
    "SinBP_D_EOnly",
    "SinBP_D_E2",
    "SinBP_D_LocalA",
    "SinBP_D_PPShapeA",
    "SinBP_D_PPShapeB",
)

REPRESENTATIVE_SESSION_ID = "eitaro_20260410_155023"
