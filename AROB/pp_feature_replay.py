from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PP_MIN = 15.0
PP_MAX = 100.0
RIDGE_ALPHA = 4.0
EWMA_ALPHA = 0.42
BASE_FEATURES = (
    "M2_A_used",
    "M2_HR_used",
    "M2_V2P_relTTP_used",
    "M2_P2V_relTTP_used",
    "M2_Stiffness_used",
    "M2_E_used",
)
ENGINEERED_FEATURES = (
    "M2_ppf_asymmetry_gap",
    "M2_ppf_asymmetry_ratio",
    "M2_ppf_up_down_width_ratio",
    "M2_ppf_local_width_samples",
    "M2_ppf_peak_sharpness",
    "M2_ppf_peak_sharpness_width",
    "M2_ppf_v2p_minus_p2v",
    "M2_ppf_range_per_width",
    "M2_ppf_area_ratio",
    "M2_ppf_local_width_ms",
    "M2_ppf_e_norm",
    "M2_ppf_stiffness_e_product",
)
CANDIDATE_METHODS = (
    ("SinBP_D_PPShapeA", "sinBP(D-PP-A)", "M2PPA"),
    ("SinBP_D_PPShapeB", "sinBP(D-PP-B)", "M2PPB"),
)


@dataclass(frozen=True)
class PpReplayOutputs:
    session_frames: dict[str, pd.DataFrame]
    screening_df: pd.DataFrame
    culprit_df: pd.DataFrame
    model_coefficients_df: pd.DataFrame


def _safe_divide(lhs: pd.Series, rhs: pd.Series) -> pd.Series:
    rhs_numeric = pd.to_numeric(rhs, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(lhs, errors="coerce") / rhs_numeric


def _safe_corr(lhs: pd.Series, rhs: pd.Series) -> float:
    local = pd.DataFrame({"lhs": lhs, "rhs": rhs}).dropna()
    if len(local) < 3:
        return float("nan")
    if float(local["lhs"].std(ddof=0)) == 0.0 or float(local["rhs"].std(ddof=0)) == 0.0:
        return float("nan")
    return float(local["lhs"].corr(local["rhs"]))


def _centered(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric - float(numeric.median())


def _ewma(series: pd.Series, alpha: float = EWMA_ALPHA) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    state = math.nan
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        state = value if not math.isfinite(state) else alpha * value + (1.0 - alpha) * state
        result[index] = state
    return pd.Series(result, index=series.index)


def _clamp_pp(pp: pd.Series) -> pd.Series:
    return pd.to_numeric(pp, errors="coerce").clip(lower=PP_MIN, upper=PP_MAX)


def engineer_pp_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    ibi = pd.to_numeric(enriched.get("M2_IBI_current_ms"), errors="coerce")
    enriched["M2_ppf_asymmetry_gap"] = (
        pd.to_numeric(enriched.get("M2_diastole_ratio"), errors="coerce")
        - pd.to_numeric(enriched.get("M2_systole_ratio"), errors="coerce")
    )
    enriched["M2_ppf_asymmetry_ratio"] = _safe_divide(
        enriched.get("M2_systole_ratio"),
        enriched.get("M2_diastole_ratio"),
    )
    enriched["M2_ppf_up_down_width_ratio"] = _safe_divide(
        enriched.get("M2_V2P_relTTP_used"),
        enriched.get("M2_P2V_relTTP_used"),
    )
    enriched["M2_ppf_local_width_samples"] = pd.to_numeric(enriched.get("M2_beat_sample_count"), errors="coerce")
    enriched["M2_ppf_peak_sharpness"] = _safe_divide(
        enriched.get("M2_beat_range"),
        enriched.get("M2_beat_std"),
    )
    enriched["M2_ppf_peak_sharpness_width"] = _safe_divide(
        enriched.get("M2_beat_range"),
        enriched.get("M2_beat_sample_count"),
    )
    enriched["M2_ppf_v2p_minus_p2v"] = (
        pd.to_numeric(enriched.get("M2_V2P_relTTP_used"), errors="coerce")
        - pd.to_numeric(enriched.get("M2_P2V_relTTP_used"), errors="coerce")
    )
    enriched["M2_ppf_range_per_width"] = _safe_divide(
        enriched.get("M2_beat_range"),
        enriched.get("M2_beat_sample_count"),
    )
    enriched["M2_ppf_area_ratio"] = _safe_divide(
        pd.to_numeric(enriched.get("M2_beat_range"), errors="coerce")
        * pd.to_numeric(enriched.get("M2_systole_ratio"), errors="coerce"),
        pd.to_numeric(enriched.get("M2_beat_std"), errors="coerce")
        * pd.to_numeric(enriched.get("M2_diastole_ratio"), errors="coerce"),
    )
    enriched["M2_ppf_local_width_ms"] = ibi * pd.to_numeric(enriched.get("M2_systole_ratio"), errors="coerce")
    enriched["M2_ppf_e_norm"] = _safe_divide(
        enriched.get("M2_E_used"),
        enriched.get("M2_beat_range"),
    )
    enriched["M2_ppf_stiffness_e_product"] = (
        pd.to_numeric(enriched.get("M2_Stiffness_used"), errors="coerce")
        * pd.to_numeric(enriched.get("M2_E_used"), errors="coerce")
    )
    return enriched


def _feature_screening_rows(session_id: str, df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in ENGINEERED_FEATURES + BASE_FEATURES:
        if feature not in df.columns:
            continue
        local = df[[feature, "ref_PP"]].dropna().copy()
        if len(local) < 3:
            continue
        raw_corr = _safe_corr(local[feature], local["ref_PP"])
        centered_corr = _safe_corr(_centered(local[feature]), _centered(local["ref_PP"]))
        rows.append(
            {
                "session_id": session_id,
                "feature": feature,
                "raw_corr_to_ref_pp": raw_corr,
                "centered_corr_to_ref_pp": centered_corr,
                "abs_centered_corr_to_ref_pp": abs(centered_corr) if math.isfinite(centered_corr) else math.nan,
                "feature_std": float(pd.to_numeric(local[feature], errors="coerce").std(ddof=0)),
            }
        )
    return rows


def _build_culprit_rows(screening_df: pd.DataFrame) -> pd.DataFrame:
    if screening_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for session_id, group in screening_df.groupby("session_id", dropna=False):
        inverse = group[group["centered_corr_to_ref_pp"] < 0.0].copy()
        if inverse.empty:
            continue
        top = inverse.sort_values("abs_centered_corr_to_ref_pp", ascending=False).iloc[0]
        rows.append(
            {
                "session_id": session_id,
                "top_inverse_feature": top["feature"],
                "top_inverse_centered_corr_to_ref_pp": float(top["centered_corr_to_ref_pp"]),
                "top_inverse_abs_centered_corr_to_ref_pp": float(top["abs_centered_corr_to_ref_pp"]),
            }
        )
    return pd.DataFrame(rows)


def _ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    predictions = np.column_stack([np.ones(len(x_test)), x_test]) @ coefficients
    return predictions, coefficients


def _valid_feature_frame(df: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    required = ["session_id", "ref_PP", "M2_MAP_calibrated", "M2_PP_calibrated", "M2_output_valid", "M2_reject_reason", *feature_names]
    local = df[required].copy()
    local = local[local["M2_output_valid"].fillna(0).astype(int) == 1].copy()
    local = local[local["M2_reject_reason"].fillna("").astype(str).str.strip().isin(["ok", ""])].copy()
    local = local.dropna(subset=["ref_PP", "M2_MAP_calibrated", "M2_PP_calibrated", *feature_names]).copy()
    if "artifact_flag" in df.columns:
        local = local[df.loc[local.index, "artifact_flag"].fillna(0).astype(int) == 0].copy()
    return local


def _select_candidate_b_features(train_df: pd.DataFrame) -> tuple[str, ...]:
    ranking: list[tuple[float, str]] = []
    for feature in ENGINEERED_FEATURES:
        local = train_df[[feature, "ref_PP"]].dropna()
        if len(local) < 3:
            continue
        centered_corr = _safe_corr(_centered(local[feature]), _centered(local["ref_PP"]))
        if math.isfinite(centered_corr):
            ranking.append((abs(centered_corr), feature))
    ranking.sort(reverse=True)
    top_engineered = [feature for _, feature in ranking[:6]]
    default = [
        "M2_ppf_asymmetry_gap",
        "M2_ppf_up_down_width_ratio",
        "M2_ppf_local_width_samples",
        "M2_ppf_peak_sharpness",
        "M2_ppf_v2p_minus_p2v",
        "M2_ppf_e_norm",
    ]
    ordered = top_engineered or default
    return tuple(dict.fromkeys([*BASE_FEATURES, *ordered]))


def _predict_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: tuple[str, ...],
    correction_blend: float = 1.0,
) -> tuple[pd.Series, np.ndarray]:
    train_local = train_df.copy()
    test_local = test_df.copy()
    for feature in feature_names:
        train_local[f"{feature}__centered"] = train_local.groupby("session_id")[feature].transform(_centered)
        test_local[f"{feature}__centered"] = test_local.groupby("session_id")[feature].transform(_centered)

    train_local["ref_pp_centered"] = train_local.groupby("session_id")["ref_PP"].transform(_centered)
    train_local["base_pp_centered"] = train_local.groupby("session_id")["M2_PP_calibrated"].transform(_centered)
    test_local["base_pp_centered"] = test_local.groupby("session_id")["M2_PP_calibrated"].transform(_centered)

    train_x = train_local.loc[:, [f"{feature}__centered" for feature in feature_names]].to_numpy(dtype=float)
    test_x = test_local.loc[:, [f"{feature}__centered" for feature in feature_names]].to_numpy(dtype=float)
    train_y = (train_local["ref_pp_centered"] - train_local["base_pp_centered"]).to_numpy(dtype=float)
    correction_prediction, coefficients = _ridge_fit_predict(train_x, train_y, test_x)
    correction_prediction = correction_blend * correction_prediction
    base_centered = test_local["base_pp_centered"].to_numpy(dtype=float)
    smartphone_anchor = float(pd.to_numeric(test_local["M2_PP_calibrated"], errors="coerce").median())
    prediction = _clamp_pp(pd.Series(smartphone_anchor + base_centered + correction_prediction, index=test_df.index))
    prediction = _ewma(prediction)
    prediction = _clamp_pp(prediction)
    return prediction, coefficients


def add_pp_replay_candidates(session_frames: dict[str, pd.DataFrame]) -> PpReplayOutputs:
    engineered_frames = {session_id: engineer_pp_features(df) for session_id, df in session_frames.items()}

    screening_rows: list[dict[str, object]] = []
    for session_id, df in engineered_frames.items():
        screening_rows.extend(_feature_screening_rows(session_id, df))
    screening_df = pd.DataFrame(screening_rows)
    culprit_df = _build_culprit_rows(screening_df)

    session_prediction_frames = {session_id: df.copy() for session_id, df in engineered_frames.items()}
    model_rows: list[dict[str, object]] = []
    session_ids = sorted(engineered_frames)
    candidate_a_features = (*BASE_FEATURES, *ENGINEERED_FEATURES)

    for held_out_session in session_ids:
        train_df = pd.concat(
            [engineered_frames[session_id] for session_id in session_ids if session_id != held_out_session],
            ignore_index=True,
        )
        test_df = engineered_frames[held_out_session]

        train_a = _valid_feature_frame(train_df, candidate_a_features)
        test_a = _valid_feature_frame(test_df, candidate_a_features)
        if not train_a.empty and not test_a.empty:
            pred_a, coef_a = _predict_candidate(train_a, test_a, candidate_a_features)
            target_df = session_prediction_frames[held_out_session]
            target_df.loc[pred_a.index, "M2PPA_PP_calibrated"] = pred_a
            target_df.loc[pred_a.index, "M2PPA_MAP_calibrated"] = target_df.loc[pred_a.index, "M2_MAP_calibrated"]
            target_df.loc[pred_a.index, "M2PPA_DBP_calibrated"] = (
                target_df.loc[pred_a.index, "M2PPA_MAP_calibrated"] - target_df.loc[pred_a.index, "M2PPA_PP_calibrated"] / 3.0
            )
            target_df.loc[pred_a.index, "M2PPA_SBP_calibrated"] = (
                target_df.loc[pred_a.index, "M2PPA_DBP_calibrated"] + target_df.loc[pred_a.index, "M2PPA_PP_calibrated"]
            )
            target_df.loc[pred_a.index, "M2PPA_output_valid"] = 1
            target_df.loc[pred_a.index, "M2PPA_reject_reason"] = "ok"
            for feature_name, coefficient in zip(("intercept", *candidate_a_features), coef_a):
                model_rows.append(
                    {
                        "held_out_session": held_out_session,
                        "method": "SinBP_D_PPShapeA",
                        "feature": feature_name,
                        "coefficient": float(coefficient),
                    }
                )

        candidate_b_features = _select_candidate_b_features(train_a if not train_a.empty else train_df)
        train_b = _valid_feature_frame(train_df, candidate_b_features)
        test_b = _valid_feature_frame(test_df, candidate_b_features)
        if not train_b.empty and not test_b.empty:
            pred_b, coef_b = _predict_candidate(train_b, test_b, candidate_b_features)
            target_df = session_prediction_frames[held_out_session]
            target_df.loc[pred_b.index, "M2PPB_PP_calibrated"] = pred_b
            target_df.loc[pred_b.index, "M2PPB_MAP_calibrated"] = target_df.loc[pred_b.index, "M2_MAP_calibrated"]
            target_df.loc[pred_b.index, "M2PPB_DBP_calibrated"] = (
                target_df.loc[pred_b.index, "M2PPB_MAP_calibrated"] - target_df.loc[pred_b.index, "M2PPB_PP_calibrated"] / 3.0
            )
            target_df.loc[pred_b.index, "M2PPB_SBP_calibrated"] = (
                target_df.loc[pred_b.index, "M2PPB_DBP_calibrated"] + target_df.loc[pred_b.index, "M2PPB_PP_calibrated"]
            )
            target_df.loc[pred_b.index, "M2PPB_output_valid"] = 1
            target_df.loc[pred_b.index, "M2PPB_reject_reason"] = "ok"
            for feature_name, coefficient in zip(("intercept", *candidate_b_features), coef_b):
                model_rows.append(
                    {
                        "held_out_session": held_out_session,
                        "method": "SinBP_D_PPShapeB",
                        "feature": feature_name,
                        "coefficient": float(coefficient),
                    }
                )

    return PpReplayOutputs(
        session_frames=session_prediction_frames,
        screening_df=screening_df.sort_values(["feature", "session_id"]).reset_index(drop=True),
        culprit_df=culprit_df.sort_values("session_id").reset_index(drop=True),
        model_coefficients_df=pd.DataFrame(model_rows),
    )


def write_pp_replay_artifacts(output_dir: Path, outputs: PpReplayOutputs) -> dict[str, Path]:
    screening_path = output_dir / "pp_feature_screening.csv"
    culprit_path = output_dir / "pp_feature_culprits.csv"
    coefficients_path = output_dir / "pp_feature_candidate_coefficients.csv"
    outputs.screening_df.to_csv(screening_path, index=False)
    outputs.culprit_df.to_csv(culprit_path, index=False)
    outputs.model_coefficients_df.to_csv(coefficients_path, index=False)
    return {
        "pp_feature_screening": screening_path,
        "pp_feature_culprits": culprit_path,
        "pp_feature_candidate_coefficients": coefficients_path,
    }
