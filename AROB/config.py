from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodSpec:
    name: str
    label: str
    prefix: str
    sbp_col: str
    dbp_col: str
    map_col: str
    pp_col: str
    output_valid_col: str | None = None
    reject_reason_col: str | None = None


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "Analysis"
REALTIME_SESSIONS_ROOT = ANALYSIS_ROOT / "Data" / "realtime_sessions"
OUTPUT_ROOT = ANALYSIS_ROOT / "Data" / "arob_tracking"

TIME_COL = "経過時間_秒"
SESSION_COL = "session_id"
BEAT_COL = "beat_index"
REF_SBP_COL = "ref_SBP"
REF_DBP_COL = "ref_DBP"
REF_MAP_COL = "ref_MAP"
REF_PP_COL = "ref_PP"
ARTIFACT_COL = "artifact_flag"

TRACKING_TARGET_SPECS: tuple[tuple[str, str, str], ...] = (
    ("SBP", REF_SBP_COL, "pred_SBP"),
    ("DBP", REF_DBP_COL, "pred_DBP"),
    ("MAP", REF_MAP_COL, "pred_MAP"),
    ("PP", REF_PP_COL, "pred_PP"),
)

WINDOW_SECONDS = (5, 10, 20)
PRIMARY_WINDOW_SECONDS = 20

METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="RTBP",
        label="RTBP",
        prefix="M1",
        sbp_col="M1_SBP_calibrated",
        dbp_col="M1_DBP_calibrated",
        map_col="M1_MAP_calibrated",
        pp_col="M1_PP_calibrated",
        output_valid_col="M1_output_valid",
        reject_reason_col="M1_reject_reason",
    ),
    MethodSpec(
        name="SinBP_M",
        label="sinBP(M)",
        prefix="M3",
        sbp_col="M3_SBP_calibrated",
        dbp_col="M3_DBP_calibrated",
        map_col="M3_MAP_calibrated",
        pp_col="M3_PP_calibrated",
        output_valid_col="M3_output_valid",
        reject_reason_col="M3_reject_reason",
    ),
    MethodSpec(
        name="SinBP_D",
        label="sinBP(D)",
        prefix="M2",
        sbp_col="M2_SBP_calibrated",
        dbp_col="M2_DBP_calibrated",
        map_col="M2_MAP_calibrated",
        pp_col="M2_PP_calibrated",
        output_valid_col="M2_output_valid",
        reject_reason_col="M2_reject_reason",
    ),
    MethodSpec(
        name="SinBP_D_EOnly",
        label="sinBP(D-EOnly)",
        prefix="SinBP_D_EOnly",
        sbp_col="SinBP_D_EOnly_SBP_calibrated",
        dbp_col="SinBP_D_EOnly_DBP_calibrated",
        map_col="SinBP_D_EOnly_MAP_calibrated",
        pp_col="SinBP_D_EOnly_PP_calibrated",
        output_valid_col="SinBP_D_EOnly_output_valid",
        reject_reason_col="SinBP_D_EOnly_reject_reason",
    ),
)
