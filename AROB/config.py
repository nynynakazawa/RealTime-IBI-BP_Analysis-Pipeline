from __future__ import annotations

from dataclasses import dataclass

from current_direction import (
    ANALYSIS_ROOT,
    PAPER_CORE_METHOD_NAMES,
    PAPER_METHOD_NAMES,
    PAPER_SUPPLEMENTAL_METHOD_NAMES,
    REALTIME_SESSIONS_ROOT,
)


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

CORE_METHOD_SPECS: tuple[MethodSpec, ...] = (
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
)

DIAGNOSTIC_METHOD_SPECS: tuple[MethodSpec, ...] = (
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
    MethodSpec(
        name="SinBP_D_E2",
        label="sinBP(D-E2)",
        prefix="SinBP_D_E2",
        sbp_col="SinBP_D_E2_SBP_calibrated",
        dbp_col="SinBP_D_E2_DBP_calibrated",
        map_col="SinBP_D_E2_MAP_calibrated",
        pp_col="SinBP_D_E2_PP_calibrated",
        output_valid_col="SinBP_D_E2_output_valid",
        reject_reason_col="SinBP_D_E2_reject_reason",
    ),
    MethodSpec(
        name="SinBP_D_LocalA",
        label="sinBP(D-LocalA)",
        prefix="SinBP_D_LocalA",
        sbp_col="SinBP_D_LocalA_SBP_calibrated",
        dbp_col="SinBP_D_LocalA_DBP_calibrated",
        map_col="SinBP_D_LocalA_MAP_calibrated",
        pp_col="SinBP_D_LocalA_PP_calibrated",
        output_valid_col="SinBP_D_LocalA_output_valid",
        reject_reason_col="SinBP_D_LocalA_reject_reason",
    ),
    MethodSpec(
        name="SinBP_D_PPShapeA",
        label="sinBP(D-PP-A)",
        prefix="M2PPA",
        sbp_col="M2PPA_SBP_calibrated",
        dbp_col="M2PPA_DBP_calibrated",
        map_col="M2PPA_MAP_calibrated",
        pp_col="M2PPA_PP_calibrated",
        output_valid_col="M2PPA_output_valid",
        reject_reason_col="M2PPA_reject_reason",
    ),
    MethodSpec(
        name="SinBP_D_PPShapeB",
        label="sinBP(D-PP-B)",
        prefix="M2PPB",
        sbp_col="M2PPB_SBP_calibrated",
        dbp_col="M2PPB_DBP_calibrated",
        map_col="M2PPB_MAP_calibrated",
        pp_col="M2PPB_PP_calibrated",
        output_valid_col="M2PPB_output_valid",
        reject_reason_col="M2PPB_reject_reason",
    ),
)

SUPPLEMENTAL_METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        name="SinBP_D_PPShapeC",
        label="sinBP(D-PP-C)",
        prefix="M2PPC",
        sbp_col="M2PPC_SBP_calibrated",
        dbp_col="M2PPC_DBP_calibrated",
        map_col="M2PPC_MAP_calibrated",
        pp_col="M2PPC_PP_calibrated",
        output_valid_col="M2PPC_output_valid",
        reject_reason_col="M2PPC_reject_reason",
    ),
)

# NOTE:
# Keep diagnostic specs defined for optional offline experiments, but
# exclude them from the default AROB tracking run.
# Active comparison series are intentionally limited to:
# RTBP / SinBP_M / SinBP_D / SinBP_D_PPShapeC.
METHOD_SPECS: tuple[MethodSpec, ...] = CORE_METHOD_SPECS + SUPPLEMENTAL_METHOD_SPECS
METHOD_SPEC_BY_NAME: dict[str, MethodSpec] = {spec.name: spec for spec in METHOD_SPECS}
PAPER_METHOD_SPECS: tuple[MethodSpec, ...] = tuple(METHOD_SPEC_BY_NAME[name] for name in PAPER_METHOD_NAMES)
PAPER_CORE_METHOD_SPECS: tuple[MethodSpec, ...] = tuple(METHOD_SPEC_BY_NAME[name] for name in PAPER_CORE_METHOD_NAMES)
PAPER_SUPPLEMENTAL_METHOD_SPECS: tuple[MethodSpec, ...] = tuple(
    METHOD_SPEC_BY_NAME[name] for name in PAPER_SUPPLEMENTAL_METHOD_NAMES
)
