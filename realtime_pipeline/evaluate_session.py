from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .map_pp_runtime import append_runtime_map_pp_columns


CORE_METHOD_SPECS = (
    {
        "name": "RTBP",
        "prefix": "M1",
        "sbp_col": "M1_SBP",
        "dbp_col": "M1_DBP",
        "valid_col": "M1_output_valid",
        "reject_col": "M1_reject_reason",
        "calibration_key": "RTBP",
    },
    {
        "name": "SinBP_D",
        "prefix": "M2",
        "sbp_col": "M2_SBP",
        "dbp_col": "M2_DBP",
        "valid_col": "M2_output_valid",
        "reject_col": "M2_reject_reason",
        "calibration_key": "SinBP_D",
    },
    {
        "name": "SinBP_D_EOnly",
        "prefix": "SinBP_D_EOnly",
        "sbp_col": "SinBP_D_EOnly_SBP",
        "dbp_col": "SinBP_D_EOnly_DBP",
        "valid_col": "SinBP_D_EOnly_output_valid",
        "reject_col": "SinBP_D_EOnly_reject_reason",
        "calibration_key": "SinBP_D",
    },
    {
        "name": "SinBP_D_E2",
        "prefix": "SinBP_D_E2",
        "sbp_col": "SinBP_D_E2_SBP",
        "dbp_col": "SinBP_D_E2_DBP",
        "valid_col": "SinBP_D_E2_output_valid",
        "reject_col": "SinBP_D_E2_reject_reason",
        "calibration_key": "SinBP_D",
    },
    {
        "name": "SinBP_D_LocalA",
        "prefix": "SinBP_D_LocalA",
        "sbp_col": "SinBP_D_LocalA_SBP",
        "dbp_col": "SinBP_D_LocalA_DBP",
        "valid_col": "SinBP_D_LocalA_output_valid",
        "reject_col": "SinBP_D_LocalA_reject_reason",
        "calibration_key": "SinBP_D",
    },
    {
        "name": "SinBP_M",
        "prefix": "M3",
        "sbp_col": "M3_SBP",
        "dbp_col": "M3_DBP",
        "valid_col": "M3_output_valid",
        "reject_col": "M3_reject_reason",
        "calibration_key": "SinBP_M",
    },
)

EXPERIMENTAL_METHOD_SPECS = tuple(
    {
        "name": f"{method_name}_{series}",
        "prefix": f"{prefix}_{series}",
        "sbp_col": f"{prefix}_{series}_SBP",
        "dbp_col": f"{prefix}_{series}_DBP",
        "valid_col": f"{prefix}_{series}_output_valid",
        "reject_col": f"{prefix}_{series}_reject_reason",
        "calibration_key": calibration_key,
        "already_smoothed": True,
    }
    for method_name, prefix, calibration_key in (
        ("RTBP", "M1", "RTBP"),
        ("SinBP_D", "M2", "SinBP_D"),
        ("SinBP_M", "M3", "SinBP_M"),
    )
    for series in (
        "INITIAL_BASELINE",
        "RICH_BASELINE",
        "SHARED_D_BASELINE",
        "RICH_DYNAMIC",
    )
)

ALL_METHOD_SPECS = CORE_METHOD_SPECS + EXPERIMENTAL_METHOD_SPECS

PLOT_SERIES = "smoothed"

MAX_ABS_TIME_DELTA_MS = 350.0
ALPHA_MAP = 0.30
ALPHA_PP = 0.50
MIN_PP = 20.0
MAX_PP = 100.0
MIN_REF_PP = 15.0
MAX_REF_PP = 100.0
REF_PP_SIGMA_MULTIPLIER = 3.0
REF_PP_MIN_BAND = 8.0

POSTPROCESS_COEFFICIENTS = {
    "RTBP": {"map_a": 0.0, "map_b": 1.0, "pp_a": 0.0, "pp_b": 1.0},
    "SinBP_D": {"map_a": 0.0, "map_b": 1.0, "pp_a": 0.0, "pp_b": 1.0},
    "SinBP_M": {"map_a": 0.0, "map_b": 1.0, "pp_a": 0.0, "pp_b": 1.0},
}

TARGET_SPECS = (
    ("SBP", "ref_SBP", "pred_SBP"),
    ("DBP", "ref_DBP", "pred_DBP"),
    ("MAP", "ref_MAP", "pred_MAP"),
    ("PP", "ref_PP", "pred_PP"),
)


def _corr(lhs: pd.Series, rhs: pd.Series) -> float:
    if len(lhs) < 2 or len(rhs) < 2:
        return float("nan")
    local = pd.concat([lhs, rhs], axis=1).dropna()
    if len(local) < 2:
        return float("nan")
    left = local.iloc[:, 0].to_numpy(dtype=float)
    right = local.iloc[:, 1].to_numpy(dtype=float)
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denom = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denom)


def _centered(series: pd.Series) -> pd.Series:
    return series - float(series.median())


def _tracking_gain(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    denom = float((ref_centered**2).sum())
    if denom == 0.0:
        return float("nan")
    return float((ref_centered * pred_centered).sum() / denom)


def _direction_agreement(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    ref_delta = np.sign(np.diff(ref_centered.to_numpy(dtype=float)))
    pred_delta = np.sign(np.diff(pred_centered.to_numpy(dtype=float)))
    valid = (ref_delta != 0) & (pred_delta != 0)
    if valid.sum() == 0:
        return float("nan")
    return float((ref_delta[valid] == pred_delta[valid]).mean())


def _amplitude_ratio(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    ref_std = float(ref_centered.std(ddof=0))
    if ref_std == 0.0:
        return float("nan")
    return float(pred_centered.std(ddof=0) / ref_std)


def _markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_csv(index=False)


def _clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


def _derive_postprocessed_columns(
    df: pd.DataFrame,
    calibration_key: str,
    prefix: str,
    sbp_col: str,
    dbp_col: str,
    valid_col: str,
    reject_col: str,
) -> pd.DataFrame:
    coeffs = POSTPROCESS_COEFFICIENTS[calibration_key]
    map_smoothed_prev = float("nan")
    pp_smoothed_prev = float("nan")

    map_raw_values: list[float] = []
    pp_raw_values: list[float] = []
    map_smoothed_values: list[float] = []
    pp_smoothed_values: list[float] = []
    map_calibrated_values: list[float] = []
    pp_calibrated_values: list[float] = []
    sbp_smoothed_values: list[float] = []
    dbp_smoothed_values: list[float] = []
    sbp_calibrated_values: list[float] = []
    dbp_calibrated_values: list[float] = []
    applied_values: list[int] = []

    for row in df.itertuples(index=False):
        output_valid = getattr(row, valid_col, 0)
        reject_reason = str(getattr(row, reject_col, "missing")).strip()
        sbp = getattr(row, sbp_col, np.nan)
        dbp = getattr(row, dbp_col, np.nan)

        if output_valid != 1 or reject_reason != "ok" or pd.isna(sbp) or pd.isna(dbp):
            map_raw_values.append(np.nan)
            pp_raw_values.append(np.nan)
            map_smoothed_values.append(np.nan)
            pp_smoothed_values.append(np.nan)
            map_calibrated_values.append(np.nan)
            pp_calibrated_values.append(np.nan)
            sbp_smoothed_values.append(np.nan)
            dbp_smoothed_values.append(np.nan)
            sbp_calibrated_values.append(np.nan)
            dbp_calibrated_values.append(np.nan)
            applied_values.append(0)
            continue

        sbp = float(sbp)
        dbp = float(dbp)
        map_raw = (sbp + 2.0 * dbp) / 3.0
        pp_raw = sbp - dbp
        if np.isnan(map_smoothed_prev):
            map_smoothed = map_raw
            pp_smoothed = pp_raw
        else:
            map_smoothed = ALPHA_MAP * map_raw + (1.0 - ALPHA_MAP) * map_smoothed_prev
            pp_smoothed = ALPHA_PP * pp_raw + (1.0 - ALPHA_PP) * pp_smoothed_prev
        map_smoothed_prev = map_smoothed
        pp_smoothed_prev = pp_smoothed

        map_calibrated = coeffs["map_a"] + coeffs["map_b"] * map_smoothed
        pp_calibrated = coeffs["pp_a"] + coeffs["pp_b"] * pp_smoothed

        dbp_smoothed = map_smoothed - pp_smoothed / 3.0
        sbp_smoothed = dbp_smoothed + pp_smoothed

        dbp_calibrated = map_calibrated - pp_calibrated / 3.0
        sbp_calibrated = dbp_calibrated + pp_calibrated

        map_raw_values.append(map_raw)
        pp_raw_values.append(pp_raw)
        map_smoothed_values.append(map_smoothed)
        pp_smoothed_values.append(pp_smoothed)
        map_calibrated_values.append(map_calibrated)
        pp_calibrated_values.append(pp_calibrated)
        sbp_smoothed_values.append(sbp_smoothed)
        dbp_smoothed_values.append(dbp_smoothed)
        sbp_calibrated_values.append(sbp_calibrated)
        dbp_calibrated_values.append(dbp_calibrated)
        applied_values.append(1)

    enriched = df.copy()
    enriched[f"{prefix}_MAP_raw"] = map_raw_values
    enriched[f"{prefix}_PP_raw"] = pp_raw_values
    enriched[f"{prefix}_MAP_smoothed"] = map_smoothed_values
    enriched[f"{prefix}_PP_smoothed"] = pp_smoothed_values
    enriched[f"{prefix}_MAP_calibrated"] = map_calibrated_values
    enriched[f"{prefix}_PP_calibrated"] = pp_calibrated_values
    enriched[f"{prefix}_SBP_smoothed"] = sbp_smoothed_values
    enriched[f"{prefix}_DBP_smoothed"] = dbp_smoothed_values
    enriched[f"{prefix}_SBP_calibrated"] = sbp_calibrated_values
    enriched[f"{prefix}_DBP_calibrated"] = dbp_calibrated_values
    enriched[f"{prefix}_postprocess_applied"] = applied_values
    return enriched


def _ensure_identity_postprocessed_columns(
    df: pd.DataFrame,
    prefix: str,
    sbp_col: str,
    dbp_col: str,
    valid_col: str,
    reject_col: str,
) -> pd.DataFrame:
    enriched = df.copy()
    output_valid = enriched[valid_col].fillna(0).astype(float) == 1.0
    reject_ok = enriched[reject_col].fillna("missing").astype(str).str.strip() == "ok"
    valid = output_valid & reject_ok
    sbp = pd.to_numeric(enriched[sbp_col], errors="coerce")
    dbp = pd.to_numeric(enriched[dbp_col], errors="coerce")
    map_value = (sbp + 2.0 * dbp) / 3.0
    pp_value = sbp - dbp
    sbp_valid = sbp.where(valid)
    dbp_valid = dbp.where(valid)
    map_valid = map_value.where(valid)
    pp_valid = pp_value.where(valid)
    enriched[f"{prefix}_MAP_raw"] = map_valid
    enriched[f"{prefix}_PP_raw"] = pp_valid
    enriched[f"{prefix}_MAP_smoothed"] = map_valid
    enriched[f"{prefix}_PP_smoothed"] = pp_valid
    enriched[f"{prefix}_MAP_calibrated"] = map_valid
    enriched[f"{prefix}_PP_calibrated"] = pp_valid
    enriched[f"{prefix}_SBP_smoothed"] = sbp_valid
    enriched[f"{prefix}_DBP_smoothed"] = dbp_valid
    enriched[f"{prefix}_SBP_calibrated"] = sbp_valid
    enriched[f"{prefix}_DBP_calibrated"] = dbp_valid
    enriched[f"{prefix}_postprocess_applied"] = valid.astype(int)
    return enriched


def ensure_postprocessed_columns(merged_df: pd.DataFrame) -> pd.DataFrame:
    enriched = append_runtime_map_pp_columns(merged_df)
    for spec in ALL_METHOD_SPECS:
        required = {spec["sbp_col"], spec["dbp_col"], spec["valid_col"], spec["reject_col"]}
        if required.issubset(set(enriched.columns)):
            if spec.get("already_smoothed"):
                enriched = _ensure_identity_postprocessed_columns(
                    enriched,
                    spec["prefix"],
                    spec["sbp_col"],
                    spec["dbp_col"],
                    spec["valid_col"],
                    spec["reject_col"],
                )
            else:
                enriched = _derive_postprocessed_columns(
                    enriched,
                    spec["calibration_key"],
                    spec["prefix"],
                    spec["sbp_col"],
                    spec["dbp_col"],
                    spec["valid_col"],
                    spec["reject_col"],
                )
    return enriched


def build_filtered_view(merged_df: pd.DataFrame) -> pd.DataFrame:
    filtered = ensure_postprocessed_columns(merged_df).copy()
    if "abs_time_delta_ms" not in filtered.columns:
        filtered["abs_time_delta_ms"] = np.nan
    if "artifact_flag" not in filtered.columns:
        filtered["artifact_flag"] = 0
    filtered["ref_filter_reason"] = "ok"
    mask = filtered["artifact_flag"].fillna(0).astype(float) == 0.0
    if filtered["abs_time_delta_ms"].notna().any():
        mask &= filtered["abs_time_delta_ms"].fillna(np.inf) <= MAX_ABS_TIME_DELTA_MS
    if "ref_SBP" in filtered.columns and "ref_DBP" in filtered.columns:
        filtered["ref_PP"] = filtered["ref_SBP"] - filtered["ref_DBP"]
        filtered["ref_MAP"] = (filtered["ref_SBP"] + 2.0 * filtered["ref_DBP"]) / 3.0
        valid_ref_pp = filtered.loc[filtered["ref_PP"].notna(), "ref_PP"].astype(float)
        if not valid_ref_pp.empty:
            pp_median = float(valid_ref_pp.median())
            pp_mad = float((valid_ref_pp - pp_median).abs().median())
            pp_sigma = 1.4826 * pp_mad
            pp_band = max(REF_PP_MIN_BAND, REF_PP_SIGMA_MULTIPLIER * pp_sigma)
            ref_pp_lower = max(MIN_REF_PP, pp_median - pp_band)
            ref_pp_upper = min(MAX_REF_PP, pp_median + pp_band)
            filtered["ref_pp_lower_bound"] = ref_pp_lower
            filtered["ref_pp_upper_bound"] = ref_pp_upper
            ref_pp_inlier = filtered["ref_PP"].between(ref_pp_lower, ref_pp_upper, inclusive="both")
            filtered["ref_pp_inlier"] = ref_pp_inlier.astype(int)
            filtered.loc[~ref_pp_inlier & filtered["ref_PP"].notna(), "ref_filter_reason"] = "ref_pp_outlier"
            mask &= ref_pp_inlier.fillna(False)
        else:
            filtered["ref_pp_lower_bound"] = np.nan
            filtered["ref_pp_upper_bound"] = np.nan
            filtered["ref_pp_inlier"] = 0
    filtered = filtered.loc[mask].copy()
    return filtered


def get_available_methods(
    merged_df: pd.DataFrame,
    include_experimental: bool = False,
) -> list[dict[str, str]]:
    methods: list[dict[str, str]] = []
    specs = ALL_METHOD_SPECS if include_experimental else CORE_METHOD_SPECS
    for spec in specs:
        required = {
            spec["sbp_col"],
            spec["dbp_col"],
            spec["valid_col"],
            spec["reject_col"],
            f"{spec['prefix']}_MAP_raw",
            f"{spec['prefix']}_PP_raw",
            f"{spec['prefix']}_SBP_smoothed",
            f"{spec['prefix']}_DBP_smoothed",
            f"{spec['prefix']}_MAP_smoothed",
            f"{spec['prefix']}_PP_smoothed",
            f"{spec['prefix']}_SBP_calibrated",
            f"{spec['prefix']}_DBP_calibrated",
            f"{spec['prefix']}_MAP_calibrated",
            f"{spec['prefix']}_PP_calibrated",
        }
        if required.issubset(set(merged_df.columns)):
            methods.append(spec)
    return methods


def _method_subset(df: pd.DataFrame, spec: dict[str, str], series: str) -> pd.DataFrame:
    subset = df.copy()
    subset = subset[
        (subset[spec["valid_col"]].fillna(0).astype(float) == 1.0)
        & (subset[spec["reject_col"]].fillna("missing").astype(str).str.strip() == "ok")
    ].copy()
    if series == "calibrated":
        pred_sbp_col = f"{spec['prefix']}_SBP_calibrated"
        pred_dbp_col = f"{spec['prefix']}_DBP_calibrated"
        pred_map_col = f"{spec['prefix']}_MAP_calibrated"
        pred_pp_col = f"{spec['prefix']}_PP_calibrated"
    elif series == "smoothed":
        pred_sbp_col = f"{spec['prefix']}_SBP_smoothed"
        pred_dbp_col = f"{spec['prefix']}_DBP_smoothed"
        pred_map_col = f"{spec['prefix']}_MAP_smoothed"
        pred_pp_col = f"{spec['prefix']}_PP_smoothed"
    else:
        pred_sbp_col = spec["sbp_col"]
        pred_dbp_col = spec["dbp_col"]
        pred_map_col = f"{spec['prefix']}_MAP_raw"
        pred_pp_col = f"{spec['prefix']}_PP_raw"
    subset["pred_SBP"] = subset[pred_sbp_col]
    subset["pred_DBP"] = subset[pred_dbp_col]
    subset["pred_MAP"] = subset[pred_map_col]
    subset["pred_PP"] = subset[pred_pp_col]
    return subset


def _evaluate_method_specs(
    filtered: pd.DataFrame,
    methods: list[dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in methods:
        for series in ("raw", "smoothed", "calibrated"):
            subset = _method_subset(filtered, spec, series)
            for target_label, ref_col, pred_col in TARGET_SPECS:
                target_subset = subset.dropna(subset=[ref_col, pred_col]).copy()
                if target_subset.empty:
                    rows.append(
                        {
                            "method": spec["name"],
                            "series": series,
                            "target": target_label,
                            "filters": f"abs_time_delta_ms<={MAX_ABS_TIME_DELTA_MS}, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",
                            "n": 0,
                            "mae": np.nan,
                            "rmse": np.nan,
                            "corr": np.nan,
                            "signed_bias": np.nan,
                            "centered_mae": np.nan,
                            "centered_rmse": np.nan,
                            "centered_corr": np.nan,
                            "tracking_gain": np.nan,
                            "amplitude_ratio": np.nan,
                            "direction_agreement": np.nan,
                            "inversion_like": np.nan,
                            "pp_mae": np.nan,
                            "pp_signed_bias": np.nan,
                        }
                    )
                    continue

                error = target_subset[pred_col] - target_subset[ref_col]
                pp_error = target_subset["pred_PP"] - target_subset["ref_PP"]
                ref_centered = _centered(target_subset[ref_col].astype(float))
                pred_centered = _centered(target_subset[pred_col].astype(float))
                centered_error = pred_centered - ref_centered
                gain = _tracking_gain(ref_centered, pred_centered)
                rows.append(
                    {
                        "method": spec["name"],
                        "series": series,
                        "target": target_label,
                        "filters": f"abs_time_delta_ms<={MAX_ABS_TIME_DELTA_MS}, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier",
                        "n": int(len(target_subset)),
                        "mae": float(np.abs(error).mean()),
                        "rmse": float(np.sqrt((error ** 2).mean())),
                        "corr": _corr(target_subset[ref_col], target_subset[pred_col]),
                        "signed_bias": float(error.mean()),
                        "centered_mae": float(np.abs(centered_error).mean()),
                        "centered_rmse": float(np.sqrt((centered_error ** 2).mean())),
                        "centered_corr": _corr(ref_centered, pred_centered),
                        "tracking_gain": gain,
                        "amplitude_ratio": _amplitude_ratio(ref_centered, pred_centered),
                        "direction_agreement": _direction_agreement(ref_centered, pred_centered),
                        "inversion_like": int(target_label == "PP" and np.isfinite(gain) and gain < 0.0),
                        "pp_mae": float(np.abs(pp_error).mean()),
                        "pp_signed_bias": float(pp_error.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _term_pairs(columns: list[str], prefix: str) -> dict[str, tuple[str, str]]:
    sbp_terms = {
        column[len(f"{prefix}_SBP_term_") :]: column
        for column in columns
        if column.startswith(f"{prefix}_SBP_term_")
    }
    dbp_terms = {
        column[len(f"{prefix}_DBP_term_") :]: column
        for column in columns
        if column.startswith(f"{prefix}_DBP_term_")
    }
    return {
        term: (sbp_terms[term], dbp_terms[term])
        for term in sorted(set(sbp_terms).intersection(dbp_terms))
    }


def build_pp_term_diagnostics(
    filtered: pd.DataFrame,
    methods: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_rows: list[dict[str, object]] = []
    culprit_rows: list[dict[str, object]] = []
    for spec in methods:
        pairs = _term_pairs(list(filtered.columns), spec["prefix"])
        if not pairs or "ref_PP" not in filtered.columns:
            continue
        raw_subset = _method_subset(filtered, spec, "raw")
        if raw_subset.empty:
            continue
        for term, (sbp_col, dbp_col) in pairs.items():
            local = raw_subset[[sbp_col, dbp_col, "pred_PP", "ref_PP"]].dropna().copy()
            if len(local) < 3:
                continue
            term_pp = local[sbp_col].astype(float) - local[dbp_col].astype(float)
            ref_pp = local["ref_PP"].astype(float)
            pred_pp = local["pred_PP"].astype(float)
            term_centered = _centered(term_pp)
            ref_centered = _centered(ref_pp)
            pred_centered = _centered(pred_pp)
            gain_to_ref = _tracking_gain(ref_centered, term_centered)
            amplitude_ratio = _amplitude_ratio(ref_centered, term_centered)
            inverse_strength = 0.0
            if np.isfinite(gain_to_ref) and gain_to_ref < 0.0 and np.isfinite(amplitude_ratio):
                inverse_strength = abs(gain_to_ref) * amplitude_ratio
            term_rows.append(
                {
                    "method": spec["name"],
                    "prefix": spec["prefix"],
                    "term": term,
                    "term_kind": "baseline" if term == "intercept" else "dynamic",
                    "n": int(len(local)),
                    "pp_term_mean": float(term_pp.mean()),
                    "pp_term_median": float(term_pp.median()),
                    "pp_term_std": float(term_pp.std(ddof=0)),
                    "corr_to_ref_pp": _corr(term_pp, ref_pp),
                    "centered_corr_to_ref_pp": _corr(term_centered, ref_centered),
                    "centered_corr_to_pred_pp": _corr(term_centered, pred_centered),
                    "gain_to_ref_pp": gain_to_ref,
                    "amplitude_ratio_to_ref_pp": amplitude_ratio,
                    "direction_agreement_to_ref_pp": _direction_agreement(ref_centered, term_centered),
                    "inverse_strength": inverse_strength,
                }
            )
        method_terms = pd.DataFrame([row for row in term_rows if row["method"] == spec["name"]])
        if method_terms.empty:
            continue
        dynamic_terms = method_terms[method_terms["term_kind"] == "dynamic"].copy()
        if dynamic_terms.empty:
            continue
        dynamic_terms = dynamic_terms.sort_values(["inverse_strength", "gain_to_ref_pp"], ascending=[False, True])
        top = dynamic_terms.iloc[0]
        culprit_rows.append(
            {
                "method": spec["name"],
                "top_inverse_term": top["term"],
                "top_inverse_strength": float(top["inverse_strength"]),
                "top_gain_to_ref_pp": float(top["gain_to_ref_pp"]),
                "top_centered_corr_to_ref_pp": float(top["centered_corr_to_ref_pp"]),
                "top_amplitude_ratio_to_ref_pp": float(top["amplitude_ratio_to_ref_pp"]),
            }
        )
    return pd.DataFrame(term_rows), pd.DataFrame(culprit_rows)


def _write_summary_files(summary_df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    summary_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def evaluate_merged_session(
    merged_df: pd.DataFrame,
    output_dir: Path,
    include_experimental: bool = False,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = build_filtered_view(merged_df)
    methods = get_available_methods(filtered, include_experimental=include_experimental)
    filtered_csv = output_dir / "session_evaluation_input_filtered.csv"
    filtered.to_csv(filtered_csv, index=False)
    summary_df = _evaluate_method_specs(filtered, methods)
    summary_csv = output_dir / "session_evaluation_summary.csv"
    summary_json = output_dir / "session_evaluation_summary.json"
    _write_summary_files(summary_df, summary_csv, summary_json)
    pp_term_df, pp_culprit_df = build_pp_term_diagnostics(filtered, methods)
    _write_summary_files(
        pp_term_df,
        output_dir / "session_pp_term_diagnostics.csv",
        output_dir / "session_pp_term_diagnostics.json",
    )
    _write_summary_files(
        pp_culprit_df,
        output_dir / "session_pp_term_culprits.csv",
        output_dir / "session_pp_term_culprits.json",
    )

    if not include_experimental:
        experimental_methods = get_available_methods(filtered, include_experimental=True)
        experimental_methods = [
            spec for spec in experimental_methods if spec not in CORE_METHOD_SPECS
        ]
        if experimental_methods:
            experimental_df = _evaluate_method_specs(filtered, experimental_methods)
            _write_summary_files(
                experimental_df,
                output_dir / "session_evaluation_summary_experimental.csv",
                output_dir / "session_evaluation_summary_experimental.json",
            )
    return summary_csv, summary_json


def generate_session_plots(merged_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = build_filtered_view(merged_df).dropna(subset=["ref_SBP", "ref_DBP"]).copy()
    methods = get_available_methods(valid, include_experimental=False)
    if valid.empty or not methods:
        return []
    if "経過時間_秒" not in valid.columns:
        valid["経過時間_秒"] = np.arange(len(valid), dtype=float)

    outputs: list[Path] = []
    for target_label, ref_col, suffix in (
        ("SBP", "ref_SBP", "sbp"),
        ("DBP", "ref_DBP", "dbp"),
        ("MAP", "ref_MAP", "map"),
        ("PP", "ref_PP", "pp"),
    ):
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        ax.plot(valid["経過時間_秒"], valid[ref_col], label=f"CNAP {target_label}", linewidth=2.4, color="#111111")
        for spec in methods:
            method_subset = _method_subset(valid, spec, PLOT_SERIES)
            if method_subset.empty:
                continue
            ax.plot(
                method_subset["経過時間_秒"],
                method_subset[f"pred_{target_label}"],
                label=f"{spec['name']} {PLOT_SERIES}",
                linewidth=1.4,
            )
        ax.set_xlabel("Elapsed Time (s)")
        ax.set_ylabel(f"{target_label} (mmHg)")
        ax.set_title(f"{target_label} Time Series Comparison")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = output_dir / f"{suffix}_timeseries.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_session_report(
    merged_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    plot_paths: list[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "session_report.md"
    valid = merged_df.dropna(subset=["ref_SBP", "ref_DBP"]).copy()
    raw_summary = summary_df[summary_df["series"] == "raw"].copy()
    smoothed_summary = summary_df[summary_df["series"] == "smoothed"].copy()
    calibrated_summary = summary_df[summary_df["series"] == "calibrated"].copy()
    pp_term_path = output_dir / "session_pp_term_diagnostics.csv"
    pp_culprit_path = output_dir / "session_pp_term_culprits.csv"
    lines = [
        "# Session Report",
        "",
        f"- samples_with_reference: {len(valid)}",
        f"- total_samples: {len(merged_df)}",
        f"- evaluation_filters: abs_time_delta_ms<={MAX_ABS_TIME_DELTA_MS}, output_valid=true, reject_reason=ok, artifact_flag=0, ref_pp_inlier",
        f"- plot_series: {PLOT_SERIES}",
        "",
        "## Smoothed Metrics",
        "",
        _markdown_table(smoothed_summary),
        "",
        "## Calibrated Metrics",
        "",
        _markdown_table(calibrated_summary),
        "",
        "## Raw Metrics",
        "",
        _markdown_table(raw_summary),
        "",
    ]
    if pp_culprit_path.exists():
        pp_culprit_df = pd.read_csv(pp_culprit_path)
        lines.extend(
            [
                "## PP Term Culprits",
                "",
                _markdown_table(pp_culprit_df),
                "",
            ]
        )
    if pp_term_path.exists():
        pp_term_df = pd.read_csv(pp_term_path)
        lines.extend(
            [
                "## PP Term Diagnostics",
                "",
                _markdown_table(pp_term_df),
                "",
            ]
        )
    if plot_paths:
        lines.extend(["## Plots", ""])
        for path in plot_paths:
            lines.append(f"- {path.name}")
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
