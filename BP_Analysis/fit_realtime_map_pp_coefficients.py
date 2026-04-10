from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


MAX_ABS_TIME_DELTA_MS = 350.0
ALPHA_MAP = 0.30
ALPHA_PP = 0.50
MIN_REF_PP = 15.0
MAX_REF_PP = 100.0
REF_PP_SIGMA_MULTIPLIER = 3.0
REF_PP_MIN_BAND = 8.0
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_BASELINE_RIDGE_ALPHA = 10.0
DEFAULT_RICH_BASELINE_RIDGE_ALPHA = 10.0
DEFAULT_BASELINE_SHRINKAGE = 1.0
DEFAULT_DYNAMIC_BLEND_GAIN = 0.25
INITIAL_BASELINE_BEATS = 30
EVALUATION_FILTERS = (
    f"abs_time_delta_ms<={MAX_ABS_TIME_DELTA_MS}, output_valid, "
    "reject_reason=ok, artifact_flag=0, ref_pp_inlier"
)
SERIES_ORDER = (
    "current_app_smoothed",
    "refit_map_pp_smoothed",
    "smartphone_initial_baseline",
    "smartphone_rich_baseline",
    "smartphone_rich_dynamic_blend",
    "smartphone_shared_sinbpd_baseline",
    "smartphone_shared_sinbpd_baseline_loso",
    "smartphone_rich_dynamic_blend_loso",
    "smartphone_rich_baseline_loso",
    "smartphone_initial_baseline_loso",
    "leave_one_session_out",
)
METHOD_ORDER = ("RTBP", "SinBP_D", "SinBP_D_EOnly", "SinBP_D_E2", "SinBP_D_LocalA", "SinBP_M")
REFIT_METHOD_ORDER = ("RTBP", "SinBP_D", "SinBP_M")
PLOT_METHOD_ORDER = ("RTBP", "SinBP_D", "SinBP_M")

BASELINE_METHOD_SPECS = {
    "RTBP": ("M1_SBP_smoothed", "M1_DBP_smoothed", "M1_output_valid", "M1_reject_reason"),
    "SinBP_D": ("M2_SBP_smoothed", "M2_DBP_smoothed", "M2_output_valid", "M2_reject_reason"),
    "SinBP_D_EOnly": (
        "SinBP_D_EOnly_SBP_smoothed",
        "SinBP_D_EOnly_DBP_smoothed",
        "SinBP_D_EOnly_output_valid",
        "SinBP_D_EOnly_reject_reason",
    ),
    "SinBP_D_E2": (
        "SinBP_D_E2_SBP_smoothed",
        "SinBP_D_E2_DBP_smoothed",
        "SinBP_D_E2_output_valid",
        "SinBP_D_E2_reject_reason",
    ),
    "SinBP_D_LocalA": (
        "SinBP_D_LocalA_SBP_smoothed",
        "SinBP_D_LocalA_DBP_smoothed",
        "SinBP_D_LocalA_output_valid",
        "SinBP_D_LocalA_reject_reason",
    ),
    "SinBP_M": ("M3_SBP_smoothed", "M3_DBP_smoothed", "M3_output_valid", "M3_reject_reason"),
}

RICH_BASELINE_PREFIXES = {
    "RTBP": ("M1_",),
    "SinBP_D": ("M2_", "SinBP_D_EOnly_", "SinBP_D_E2_", "SinBP_D_LocalA_"),
    "SinBP_M": ("M3_",),
}

RICH_BASELINE_EXCLUDED_TOKENS = (
    "_coef_",
    "_POST_",
    "_ALPHA",
    "_BETA",
    "_BASE_C",
    "_BASE_D",
    "_CORR_G",
    "_CORR_H",
    "_C0",
    "_C1",
    "_C2",
    "_C3",
    "_C4",
    "_D0",
    "_D1",
    "_D2",
    "_D3",
    "_D4",
    "postprocess_applied",
    "output_valid",
    "reject_reason",
    "feature_clamp_reason",
    "clamp_applied",
    "constraint_applied",
    "used_smoothed_ibi",
)

RICH_BASELINE_ALLOWED_TOKENS = (
    "_A",
    "_A_used",
    "_HR",
    "_HR_used",
    "_V2P_relTTP",
    "_V2P_relTTP_used",
    "_P2V_relTTP",
    "_P2V_relTTP_used",
    "_E",
    "_E_used",
    "_Stiffness",
    "_Stiffness_used",
    "_Mean",
    "_Mean_used",
    "_Phi",
    "_sinPhi",
    "_sinPhi_used",
    "_cosPhi",
    "_cosPhi_used",
    "_MAP_raw",
    "_MAP_smoothed",
    "_PP_raw",
    "_PP_smoothed",
    "_SBP_raw",
    "_SBP_smoothed",
    "_DBP_raw",
    "_DBP_smoothed",
    "_SBP_term_",
    "_DBP_term_",
    "_beat_range",
    "_beat_std",
    "_fit_rmse",
    "_systole_ratio",
    "_diastole_ratio",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    valid_col: str
    reject_col: str
    feature_cols: tuple[str, ...]


@dataclass
class Sample:
    session: str
    row_index: int
    row: dict[str, str]
    x: np.ndarray
    ref_map: float
    ref_pp: float


@dataclass
class FittedModels:
    rtbp_map: np.ndarray
    rtbp_pp: np.ndarray
    sinbpd_residual_map: np.ndarray
    sinbpd_residual_pp: np.ndarray
    sinbpm_map: np.ndarray
    sinbpm_pp: np.ndarray


@dataclass
class SessionInitialSummary:
    summary_features: np.ndarray
    anchor_features: np.ndarray
    ref_map_anchor: float
    ref_pp_anchor: float
    n: int


@dataclass
class AdaptiveMethodModel:
    method: str
    feature_names: tuple[str, ...]
    summary_feature_names: tuple[str, ...]
    summary_source_columns: tuple[str, ...]
    baseline_map: np.ndarray
    baseline_pp: np.ndarray
    delta_map: np.ndarray
    delta_pp: np.ndarray
    population_map_anchor: float
    population_pp_anchor: float
    baseline_shrinkage: float


@dataclass
class AdaptiveModels:
    by_method: dict[str, AdaptiveMethodModel]


RTBP = MethodSpec(
    name="RTBP",
    valid_col="M1_output_valid",
    reject_col="M1_reject_reason",
    feature_cols=("M1_A_used", "M1_HR_used", "M1_V2P_relTTP_used", "M1_P2V_relTTP_used"),
)
SINBP_D = MethodSpec(
    name="SinBP_D",
    valid_col="M2_output_valid",
    reject_col="M2_reject_reason",
    feature_cols=(
        "M2_A_used",
        "M2_HR_used",
        "M2_V2P_relTTP_used",
        "M2_P2V_relTTP_used",
        "M2_Stiffness_used",
        "M2_E_used",
    ),
)
SINBP_M = MethodSpec(
    name="SinBP_M",
    valid_col="M3_output_valid",
    reject_col="M3_reject_reason",
    feature_cols=("M3_A_used", "M3_HR_used", "M3_Mean_used", "M3_sinPhi_used", "M3_cosPhi_used"),
)


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _session_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / f"{path.name}_merged.csv").exists())


def _ref_pp_bounds(rows: list[dict[str, str]]) -> tuple[float, float]:
    ref_pp_values: list[float] = []
    for row in rows:
        ref_sbp = _to_float(row.get("ref_SBP"))
        ref_dbp = _to_float(row.get("ref_DBP"))
        if not math.isfinite(ref_sbp) or not math.isfinite(ref_dbp):
            continue
        abs_dt = _to_float(row.get("abs_time_delta_ms"))
        if math.isfinite(abs_dt) and abs_dt > MAX_ABS_TIME_DELTA_MS:
            continue
        if _to_int(row.get("artifact_flag")) != 0:
            continue
        ref_pp_values.append(ref_sbp - ref_dbp)
    if not ref_pp_values:
        return MIN_REF_PP, MAX_REF_PP

    values = np.array(ref_pp_values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    band = max(REF_PP_MIN_BAND, REF_PP_SIGMA_MULTIPLIER * 1.4826 * mad)
    return max(MIN_REF_PP, median - band), min(MAX_REF_PP, median + band)


def _row_passes_reference_filters(row: dict[str, str], ref_pp_lower: float, ref_pp_upper: float) -> bool:
    ref_sbp = _to_float(row.get("ref_SBP"))
    ref_dbp = _to_float(row.get("ref_DBP"))
    if not math.isfinite(ref_sbp) or not math.isfinite(ref_dbp):
        return False
    abs_dt = _to_float(row.get("abs_time_delta_ms"))
    if math.isfinite(abs_dt) and abs_dt > MAX_ABS_TIME_DELTA_MS:
        return False
    if _to_int(row.get("artifact_flag")) != 0:
        return False
    return ref_pp_lower <= ref_sbp - ref_dbp <= ref_pp_upper


def load_reference_rows(sessions_root: Path) -> list[dict[str, str]]:
    reference_rows: list[dict[str, str]] = []
    for session_dir in _session_dirs(sessions_root):
        rows = _load_csv(session_dir / f"{session_dir.name}_merged.csv")
        ref_pp_lower, ref_pp_upper = _ref_pp_bounds(rows)
        for row_index, row in enumerate(rows):
            if not _row_passes_reference_filters(row, ref_pp_lower, ref_pp_upper):
                continue
            enriched = dict(row)
            ref_sbp = _to_float(enriched.get("ref_SBP"))
            ref_dbp = _to_float(enriched.get("ref_DBP"))
            enriched["_session"] = session_dir.name
            enriched["_row_index"] = str(row_index)
            enriched["_ref_MAP"] = str((ref_sbp + 2.0 * ref_dbp) / 3.0)
            enriched["_ref_PP"] = str(ref_sbp - ref_dbp)
            reference_rows.append(enriched)
    return reference_rows


def build_samples(rows: list[dict[str, str]], spec: MethodSpec) -> list[Sample]:
    samples: list[Sample] = []
    for row in rows:
        if _to_int(row.get(spec.valid_col)) != 1:
            continue
        if str(row.get(spec.reject_col, "missing")).strip() != "ok":
            continue
        features = [_to_float(row.get(col)) for col in spec.feature_cols]
        if not all(math.isfinite(value) for value in features):
            continue
        samples.append(
            Sample(
                session=str(row["_session"]),
                row_index=int(row["_row_index"]),
                row=row,
                x=np.array(features, dtype=float),
                ref_map=float(row["_ref_MAP"]),
                ref_pp=float(row["_ref_PP"]),
            )
        )
    return samples


def fit_standardized_ridge_arrays(x: np.ndarray, y: np.ndarray, ridge_alpha: float) -> np.ndarray:
    if len(x) == 0:
        raise ValueError("no samples to fit")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), z])
    penalty = np.eye(design.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    slopes = beta[1:] / scale
    intercept = beta[0] - float(np.sum(beta[1:] * mean / scale))
    return np.concatenate([[intercept], slopes])


def fit_standardized_ridge(samples: list[Sample], target: str, ridge_alpha: float) -> np.ndarray:
    if not samples:
        raise ValueError("no samples to fit")
    x = np.vstack([sample.x for sample in samples])
    y = np.array([getattr(sample, target) for sample in samples], dtype=float)
    return fit_standardized_ridge_arrays(x, y, ridge_alpha)


def predict(coefficients: np.ndarray, features: np.ndarray) -> float:
    return float(coefficients[0] + np.dot(coefficients[1:], features))


def train_models(samples_by_method: dict[str, list[Sample]], train_sessions: set[str], ridge_alpha: float) -> FittedModels:
    rtbp_samples = [sample for sample in samples_by_method["RTBP"] if sample.session in train_sessions]
    sinbpd_samples = [sample for sample in samples_by_method["SinBP_D"] if sample.session in train_sessions]
    sinbpm_samples = [sample for sample in samples_by_method["SinBP_M"] if sample.session in train_sessions]

    rtbp_map = fit_standardized_ridge(rtbp_samples, "ref_map", ridge_alpha)
    rtbp_pp = fit_standardized_ridge(rtbp_samples, "ref_pp", ridge_alpha)

    residual_samples: list[Sample] = []
    for sample in sinbpd_samples:
        base_features = sample.x[:4]
        residual_features = sample.x[4:]
        residual_samples.append(
            Sample(
                session=sample.session,
                row_index=sample.row_index,
                row=sample.row,
                x=residual_features,
                ref_map=sample.ref_map - predict(rtbp_map, base_features),
                ref_pp=sample.ref_pp - predict(rtbp_pp, base_features),
            )
        )
    sinbpd_residual_map = fit_standardized_ridge(residual_samples, "ref_map", ridge_alpha)
    sinbpd_residual_pp = fit_standardized_ridge(residual_samples, "ref_pp", ridge_alpha)

    sinbpm_map = fit_standardized_ridge(sinbpm_samples, "ref_map", ridge_alpha)
    sinbpm_pp = fit_standardized_ridge(sinbpm_samples, "ref_pp", ridge_alpha)

    return FittedModels(
        rtbp_map=rtbp_map,
        rtbp_pp=rtbp_pp,
        sinbpd_residual_map=sinbpd_residual_map,
        sinbpd_residual_pp=sinbpd_residual_pp,
        sinbpm_map=sinbpm_map,
        sinbpm_pp=sinbpm_pp,
    )


def _summary_feature_names(spec: MethodSpec) -> tuple[str, ...]:
    return tuple(
        f"{feature}_{stat}"
        for stat in ("median", "std", "p10", "p90")
        for feature in spec.feature_cols
    )


def build_initial_summary(samples: list[Sample]) -> SessionInitialSummary | None:
    if not samples:
        return None
    initial_samples = sorted(samples, key=lambda sample: sample.row_index)[:INITIAL_BASELINE_BEATS]
    if not initial_samples:
        return None
    x = np.vstack([sample.x for sample in initial_samples])
    median = np.median(x, axis=0)
    std = np.std(x, axis=0)
    p10 = np.percentile(x, 10, axis=0)
    p90 = np.percentile(x, 90, axis=0)
    ref_map_anchor = float(np.median([sample.ref_map for sample in initial_samples]))
    ref_pp_anchor = float(np.median([sample.ref_pp for sample in initial_samples]))
    return SessionInitialSummary(
        summary_features=np.concatenate([median, std, p10, p90]),
        anchor_features=median,
        ref_map_anchor=ref_map_anchor,
        ref_pp_anchor=ref_pp_anchor,
        n=len(initial_samples),
    )


def _rich_baseline_columns(method: str, samples: list[Sample]) -> tuple[str, ...]:
    prefixes = RICH_BASELINE_PREFIXES[method]
    candidates: set[str] = set()
    for sample in samples:
        for column, value in sample.row.items():
            if not column.startswith(prefixes):
                continue
            if any(token in column for token in RICH_BASELINE_EXCLUDED_TOKENS):
                continue
            if "_term_intercept" in column:
                continue
            if not any(token in column for token in RICH_BASELINE_ALLOWED_TOKENS):
                continue
            if math.isfinite(_to_float(value)):
                candidates.add(column)
    return tuple(sorted(candidates))


def _rich_summary_feature_names(columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"{column}_{stat}"
        for stat in ("median", "std", "p10", "p90", "delta")
        for column in columns
    )


def build_rich_initial_summary(samples: list[Sample], columns: tuple[str, ...]) -> SessionInitialSummary | None:
    if not samples:
        return None
    initial_samples = sorted(samples, key=lambda sample: sample.row_index)[:INITIAL_BASELINE_BEATS]
    if not initial_samples:
        return None
    anchor_features = np.median(np.vstack([sample.x for sample in initial_samples]), axis=0)

    summary_values: list[float] = []
    for column in columns:
        values = np.array([_to_float(sample.row.get(column)) for sample in initial_samples], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            summary_values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        summary_values.extend(
            [
                float(np.median(finite)),
                float(np.std(finite)),
                float(np.percentile(finite, 10)),
                float(np.percentile(finite, 90)),
                float(finite[-1] - finite[0]) if len(finite) >= 2 else 0.0,
            ]
        )

    ref_map_anchor = float(np.median([sample.ref_map for sample in initial_samples]))
    ref_pp_anchor = float(np.median([sample.ref_pp for sample in initial_samples]))
    return SessionInitialSummary(
        summary_features=np.array(summary_values, dtype=float),
        anchor_features=anchor_features,
        ref_map_anchor=ref_map_anchor,
        ref_pp_anchor=ref_pp_anchor,
        n=len(initial_samples),
    )


def train_adaptive_models(
    samples_by_method: dict[str, list[Sample]],
    train_sessions: set[str],
    baseline_ridge_alpha: float,
    delta_ridge_alpha: float,
    baseline_shrinkage: float,
) -> AdaptiveModels:
    specs = {"RTBP": RTBP, "SinBP_D": SINBP_D, "SinBP_M": SINBP_M}
    models: dict[str, AdaptiveMethodModel] = {}
    for method, spec in specs.items():
        method_samples = [sample for sample in samples_by_method[method] if sample.session in train_sessions]
        summaries: dict[str, SessionInitialSummary] = {}
        for session in sorted(train_sessions):
            session_samples = [sample for sample in method_samples if sample.session == session]
            summary = build_initial_summary(session_samples)
            if summary is not None:
                summaries[session] = summary
        if len(summaries) < 2:
            raise ValueError(f"need at least two session summaries for adaptive model: {method}")

        ordered_sessions = sorted(summaries)
        summary_x = np.vstack([summaries[session].summary_features for session in ordered_sessions])
        baseline_map_y = np.array([summaries[session].ref_map_anchor for session in ordered_sessions], dtype=float)
        baseline_pp_y = np.array([summaries[session].ref_pp_anchor for session in ordered_sessions], dtype=float)
        baseline_map = fit_standardized_ridge_arrays(summary_x, baseline_map_y, baseline_ridge_alpha)
        baseline_pp = fit_standardized_ridge_arrays(summary_x, baseline_pp_y, baseline_ridge_alpha)

        delta_x: list[np.ndarray] = []
        delta_map_y: list[float] = []
        delta_pp_y: list[float] = []
        for sample in method_samples:
            summary = summaries.get(sample.session)
            if summary is None:
                continue
            delta_x.append(sample.x - summary.anchor_features)
            delta_map_y.append(sample.ref_map - summary.ref_map_anchor)
            delta_pp_y.append(sample.ref_pp - summary.ref_pp_anchor)
        if not delta_x:
            raise ValueError(f"no adaptive delta samples: {method}")

        models[method] = AdaptiveMethodModel(
            method=method,
            feature_names=spec.feature_cols,
            summary_feature_names=_summary_feature_names(spec),
            summary_source_columns=spec.feature_cols,
            baseline_map=baseline_map,
            baseline_pp=baseline_pp,
            delta_map=fit_standardized_ridge_arrays(np.vstack(delta_x), np.array(delta_map_y, dtype=float), delta_ridge_alpha),
            delta_pp=fit_standardized_ridge_arrays(np.vstack(delta_x), np.array(delta_pp_y, dtype=float), delta_ridge_alpha),
            population_map_anchor=float(np.mean(baseline_map_y)),
            population_pp_anchor=float(np.mean(baseline_pp_y)),
            baseline_shrinkage=baseline_shrinkage,
        )
    return AdaptiveModels(by_method=models)


def train_rich_baseline_models(
    samples_by_method: dict[str, list[Sample]],
    train_sessions: set[str],
    baseline_ridge_alpha: float,
    delta_ridge_alpha: float,
    baseline_shrinkage: float,
) -> AdaptiveModels:
    specs = {"RTBP": RTBP, "SinBP_D": SINBP_D, "SinBP_M": SINBP_M}
    models: dict[str, AdaptiveMethodModel] = {}
    for method, spec in specs.items():
        method_samples = [sample for sample in samples_by_method[method] if sample.session in train_sessions]
        rich_columns = _rich_baseline_columns(method, method_samples)
        if not rich_columns:
            raise ValueError(f"no rich baseline columns: {method}")

        summaries: dict[str, SessionInitialSummary] = {}
        for session in sorted(train_sessions):
            session_samples = [sample for sample in method_samples if sample.session == session]
            summary = build_rich_initial_summary(session_samples, rich_columns)
            if summary is not None:
                summaries[session] = summary
        if len(summaries) < 2:
            raise ValueError(f"need at least two session summaries for rich baseline model: {method}")

        ordered_sessions = sorted(summaries)
        summary_x = np.vstack([summaries[session].summary_features for session in ordered_sessions])
        baseline_map_y = np.array([summaries[session].ref_map_anchor for session in ordered_sessions], dtype=float)
        baseline_pp_y = np.array([summaries[session].ref_pp_anchor for session in ordered_sessions], dtype=float)

        delta_x: list[np.ndarray] = []
        delta_map_y: list[float] = []
        delta_pp_y: list[float] = []
        for sample in method_samples:
            summary = summaries.get(sample.session)
            if summary is None:
                continue
            delta_x.append(sample.x - summary.anchor_features)
            delta_map_y.append(sample.ref_map - summary.ref_map_anchor)
            delta_pp_y.append(sample.ref_pp - summary.ref_pp_anchor)
        if not delta_x:
            raise ValueError(f"no rich baseline delta samples: {method}")

        models[method] = AdaptiveMethodModel(
            method=method,
            feature_names=spec.feature_cols,
            summary_feature_names=_rich_summary_feature_names(rich_columns),
            summary_source_columns=rich_columns,
            baseline_map=fit_standardized_ridge_arrays(summary_x, baseline_map_y, baseline_ridge_alpha),
            baseline_pp=fit_standardized_ridge_arrays(summary_x, baseline_pp_y, baseline_ridge_alpha),
            delta_map=fit_standardized_ridge_arrays(np.vstack(delta_x), np.array(delta_map_y, dtype=float), delta_ridge_alpha),
            delta_pp=fit_standardized_ridge_arrays(np.vstack(delta_x), np.array(delta_pp_y, dtype=float), delta_ridge_alpha),
            population_map_anchor=float(np.mean(baseline_map_y)),
            population_pp_anchor=float(np.mean(baseline_pp_y)),
            baseline_shrinkage=baseline_shrinkage,
        )
    return AdaptiveModels(by_method=models)


def train_shared_sinbpd_baseline_models(
    samples_by_method: dict[str, list[Sample]],
    train_sessions: set[str],
    baseline_ridge_alpha: float,
    delta_ridge_alpha: float,
    baseline_shrinkage: float,
) -> AdaptiveModels:
    specs = {"RTBP": RTBP, "SinBP_D": SINBP_D, "SinBP_M": SINBP_M}
    sinbpd_samples = [sample for sample in samples_by_method["SinBP_D"] if sample.session in train_sessions]
    rich_columns = _rich_baseline_columns("SinBP_D", sinbpd_samples)
    if not rich_columns:
        raise ValueError("no SinBP_D rich baseline columns")

    baseline_summaries: dict[str, SessionInitialSummary] = {}
    for session in sorted(train_sessions):
        session_samples = [sample for sample in sinbpd_samples if sample.session == session]
        summary = build_rich_initial_summary(session_samples, rich_columns)
        if summary is not None:
            baseline_summaries[session] = summary
    if len(baseline_summaries) < 2:
        raise ValueError("need at least two session summaries for shared SinBP_D baseline")

    ordered_sessions = sorted(baseline_summaries)
    summary_x = np.vstack([baseline_summaries[session].summary_features for session in ordered_sessions])
    baseline_map_y = np.array([baseline_summaries[session].ref_map_anchor for session in ordered_sessions], dtype=float)
    baseline_pp_y = np.array([baseline_summaries[session].ref_pp_anchor for session in ordered_sessions], dtype=float)
    baseline_map = fit_standardized_ridge_arrays(summary_x, baseline_map_y, baseline_ridge_alpha)
    baseline_pp = fit_standardized_ridge_arrays(summary_x, baseline_pp_y, baseline_ridge_alpha)

    models: dict[str, AdaptiveMethodModel] = {}
    for method, spec in specs.items():
        method_samples = [sample for sample in samples_by_method[method] if sample.session in train_sessions]
        method_summaries: dict[str, SessionInitialSummary] = {}
        for session in ordered_sessions:
            session_samples = [sample for sample in method_samples if sample.session == session]
            summary = build_initial_summary(session_samples)
            if summary is not None:
                method_summaries[session] = summary

        delta_x: list[np.ndarray] = []
        delta_map_y: list[float] = []
        delta_pp_y: list[float] = []
        for sample in method_samples:
            method_summary = method_summaries.get(sample.session)
            baseline_summary = baseline_summaries.get(sample.session)
            if method_summary is None or baseline_summary is None:
                continue
            delta_x.append(sample.x - method_summary.anchor_features)
            delta_map_y.append(sample.ref_map - baseline_summary.ref_map_anchor)
            delta_pp_y.append(sample.ref_pp - baseline_summary.ref_pp_anchor)
        if not delta_x:
            raise ValueError(f"no shared baseline delta samples: {method}")

        models[method] = AdaptiveMethodModel(
            method=method,
            feature_names=spec.feature_cols,
            summary_feature_names=_rich_summary_feature_names(rich_columns),
            summary_source_columns=rich_columns,
            baseline_map=baseline_map,
            baseline_pp=baseline_pp,
            delta_map=fit_standardized_ridge_arrays(np.vstack(delta_x), np.array(delta_map_y, dtype=float), delta_ridge_alpha),
            delta_pp=fit_standardized_ridge_arrays(np.vstack(delta_x), np.array(delta_pp_y, dtype=float), delta_ridge_alpha),
            population_map_anchor=float(np.mean(baseline_map_y)),
            population_pp_anchor=float(np.mean(baseline_pp_y)),
            baseline_shrinkage=baseline_shrinkage,
        )
    return AdaptiveModels(by_method=models)


def map_pp_to_sbp_dbp(map_coefficients: np.ndarray, pp_coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return map_coefficients + (2.0 / 3.0) * pp_coefficients, map_coefficients - (1.0 / 3.0) * pp_coefficients


def smooth_map_pp(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    smoothed: list[tuple[float, float]] = []
    last_map = float("nan")
    last_pp = float("nan")
    for map_value, pp_value in values:
        if not math.isfinite(last_map):
            map_smoothed = map_value
            pp_smoothed = pp_value
        else:
            map_smoothed = ALPHA_MAP * map_value + (1.0 - ALPHA_MAP) * last_map
            pp_smoothed = ALPHA_PP * pp_value + (1.0 - ALPHA_PP) * last_pp
        last_map = map_smoothed
        last_pp = pp_smoothed
        smoothed.append((map_smoothed, pp_smoothed))
    return smoothed


def predict_method_map_pp(models: FittedModels, method: str, features: np.ndarray) -> tuple[float, float]:
    if method == "RTBP":
        return predict(models.rtbp_map, features), predict(models.rtbp_pp, features)
    if method == "SinBP_D":
        base_features = features[:4]
        residual_features = features[4:]
        map_value = predict(models.rtbp_map, base_features) + predict(models.sinbpd_residual_map, residual_features)
        pp_value = predict(models.rtbp_pp, base_features) + predict(models.sinbpd_residual_pp, residual_features)
        return map_value, pp_value
    if method == "SinBP_M":
        return predict(models.sinbpm_map, features), predict(models.sinbpm_pp, features)
    raise ValueError(f"unknown method: {method}")


def evaluate_predictions(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {
            "n": 0,
            "SBP_mae": float("nan"),
            "SBP_rmse": float("nan"),
            "SBP_bias": float("nan"),
            "DBP_mae": float("nan"),
            "DBP_rmse": float("nan"),
            "DBP_bias": float("nan"),
            "PP_mae": float("nan"),
            "PP_bias": float("nan"),
            "SBP_corr": float("nan"),
            "DBP_corr": float("nan"),
        }
    pred_sbp = np.array([float(row["pred_SBP"]) for row in rows], dtype=float)
    pred_dbp = np.array([float(row["pred_DBP"]) for row in rows], dtype=float)
    ref_sbp = np.array([float(row["ref_SBP"]) for row in rows], dtype=float)
    ref_dbp = np.array([float(row["ref_DBP"]) for row in rows], dtype=float)
    sbp_error = pred_sbp - ref_sbp
    dbp_error = pred_dbp - ref_dbp
    pp_error = (pred_sbp - pred_dbp) - (ref_sbp - ref_dbp)

    def corr(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "n": int(len(rows)),
        "SBP_mae": float(np.mean(np.abs(sbp_error))),
        "SBP_rmse": float(np.sqrt(np.mean(sbp_error**2))),
        "SBP_bias": float(np.mean(sbp_error)),
        "DBP_mae": float(np.mean(np.abs(dbp_error))),
        "DBP_rmse": float(np.sqrt(np.mean(dbp_error**2))),
        "DBP_bias": float(np.mean(dbp_error)),
        "PP_mae": float(np.mean(np.abs(pp_error))),
        "PP_bias": float(np.mean(pp_error)),
        "SBP_corr": corr(pred_sbp, ref_sbp),
        "DBP_corr": corr(pred_dbp, ref_dbp),
    }


def replay_model(models: FittedModels, samples_by_method: dict[str, list[Sample]], eval_sessions: set[str]) -> list[dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []
    for method in ("RTBP", "SinBP_D", "SinBP_M"):
        method_samples = [sample for sample in samples_by_method[method] if sample.session in eval_sessions]
        for session in sorted({sample.session for sample in method_samples}):
            session_samples = sorted(
                [sample for sample in method_samples if sample.session == session],
                key=lambda sample: sample.row_index,
            )
            raw_map_pp = [predict_method_map_pp(models, method, sample.x) for sample in session_samples]
            smoothed_map_pp = smooth_map_pp(raw_map_pp)
            for sample, (map_raw, pp_raw), (map_smoothed, pp_smoothed) in zip(session_samples, raw_map_pp, smoothed_map_pp):
                pred_dbp = map_smoothed - pp_smoothed / 3.0
                pred_sbp = pred_dbp + pp_smoothed
                prediction_rows.append(
                    {
                        "session": sample.session,
                        "row_index": sample.row_index,
                        "method": method,
                        "series": "refit_map_pp_smoothed",
                        "pred_MAP_raw": map_raw,
                        "pred_PP_raw": pp_raw,
                        "pred_MAP": map_smoothed,
                        "pred_PP": pp_smoothed,
                        "pred_SBP": pred_sbp,
                        "pred_DBP": pred_dbp,
                        "ref_SBP": _to_float(sample.row.get("ref_SBP")),
                        "ref_DBP": _to_float(sample.row.get("ref_DBP")),
                        "ref_MAP": sample.ref_map,
                        "ref_PP": sample.ref_pp,
                        "elapsed_s": _to_float(sample.row.get("経過時間_秒")),
                    }
                )
    return prediction_rows


def replay_adaptive_model(
    models: AdaptiveModels,
    samples_by_method: dict[str, list[Sample]],
    eval_sessions: set[str],
    series: str,
    rich_summary: bool = False,
) -> list[dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []
    for method in ("RTBP", "SinBP_D", "SinBP_M"):
        model = models.by_method[method]
        method_samples = [sample for sample in samples_by_method[method] if sample.session in eval_sessions]
        for session in sorted({sample.session for sample in method_samples}):
            session_samples = sorted(
                [sample for sample in method_samples if sample.session == session],
                key=lambda sample: sample.row_index,
            )
            summary = (
                build_rich_initial_summary(session_samples, model.summary_source_columns)
                if rich_summary
                else build_initial_summary(session_samples)
            )
            if summary is None:
                continue
            baseline_map_raw = predict(model.baseline_map, summary.summary_features)
            baseline_pp_raw = predict(model.baseline_pp, summary.summary_features)
            baseline_map = model.population_map_anchor + model.baseline_shrinkage * (
                baseline_map_raw - model.population_map_anchor
            )
            baseline_pp = model.population_pp_anchor + model.baseline_shrinkage * (
                baseline_pp_raw - model.population_pp_anchor
            )

            raw_map_pp: list[tuple[float, float]] = []
            for sample in session_samples:
                centered = sample.x - summary.anchor_features
                raw_map_pp.append(
                    (
                        baseline_map + predict(model.delta_map, centered),
                        baseline_pp + predict(model.delta_pp, centered),
                    )
                )
            smoothed_map_pp = smooth_map_pp(raw_map_pp)
            for sample, (map_raw, pp_raw), (map_smoothed, pp_smoothed) in zip(session_samples, raw_map_pp, smoothed_map_pp):
                pred_dbp = map_smoothed - pp_smoothed / 3.0
                pred_sbp = pred_dbp + pp_smoothed
                prediction_rows.append(
                    {
                        "session": sample.session,
                        "row_index": sample.row_index,
                        "method": method,
                        "series": series,
                        "pred_MAP_raw": map_raw,
                        "pred_PP_raw": pp_raw,
                        "pred_MAP": map_smoothed,
                        "pred_PP": pp_smoothed,
                        "pred_SBP": pred_sbp,
                        "pred_DBP": pred_dbp,
                        "baseline_MAP_raw": baseline_map_raw,
                        "baseline_PP_raw": baseline_pp_raw,
                        "baseline_MAP": baseline_map,
                        "baseline_PP": baseline_pp,
                        "initial_baseline_beats": summary.n,
                        "baseline_shrinkage": model.baseline_shrinkage,
                        "ref_SBP": _to_float(sample.row.get("ref_SBP")),
                        "ref_DBP": _to_float(sample.row.get("ref_DBP")),
                        "ref_MAP": sample.ref_map,
                        "ref_PP": sample.ref_pp,
                        "elapsed_s": _to_float(sample.row.get("経過時間_秒")),
                    }
                )
    return prediction_rows


def replay_baseline_dynamic_blend(
    baseline_models: AdaptiveModels,
    dynamic_models: FittedModels,
    samples_by_method: dict[str, list[Sample]],
    eval_sessions: set[str],
    series: str,
    dynamic_gain_map: float,
    dynamic_gain_pp: float,
    rich_summary: bool = False,
) -> list[dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []
    for method in ("RTBP", "SinBP_D", "SinBP_M"):
        baseline_model = baseline_models.by_method[method]
        method_samples = [sample for sample in samples_by_method[method] if sample.session in eval_sessions]
        for session in sorted({sample.session for sample in method_samples}):
            session_samples = sorted(
                [sample for sample in method_samples if sample.session == session],
                key=lambda sample: sample.row_index,
            )
            summary = (
                build_rich_initial_summary(session_samples, baseline_model.summary_source_columns)
                if rich_summary
                else build_initial_summary(session_samples)
            )
            if summary is None:
                continue

            baseline_map_raw = predict(baseline_model.baseline_map, summary.summary_features)
            baseline_pp_raw = predict(baseline_model.baseline_pp, summary.summary_features)
            baseline_map = baseline_model.population_map_anchor + baseline_model.baseline_shrinkage * (
                baseline_map_raw - baseline_model.population_map_anchor
            )
            baseline_pp = baseline_model.population_pp_anchor + baseline_model.baseline_shrinkage * (
                baseline_pp_raw - baseline_model.population_pp_anchor
            )

            rich_raw_map_pp: list[tuple[float, float]] = []
            for sample in session_samples:
                centered = sample.x - summary.anchor_features
                rich_raw_map_pp.append(
                    (
                        baseline_map + predict(baseline_model.delta_map, centered),
                        baseline_pp + predict(baseline_model.delta_pp, centered),
                    )
                )
            rich_smoothed_map_pp = smooth_map_pp(rich_raw_map_pp)
            dynamic_raw_map_pp = [
                predict_method_map_pp(dynamic_models, method, sample.x)
                for sample in session_samples
            ]
            dynamic_smoothed_map_pp = smooth_map_pp(dynamic_raw_map_pp)
            anchor_count = min(INITIAL_BASELINE_BEATS, len(dynamic_smoothed_map_pp))
            if anchor_count == 0:
                continue
            dynamic_raw_anchor_map = float(np.median([value[0] for value in dynamic_raw_map_pp[:anchor_count]]))
            dynamic_raw_anchor_pp = float(np.median([value[1] for value in dynamic_raw_map_pp[:anchor_count]]))
            dynamic_anchor_map = float(np.median([value[0] for value in dynamic_smoothed_map_pp[:anchor_count]]))
            dynamic_anchor_pp = float(np.median([value[1] for value in dynamic_smoothed_map_pp[:anchor_count]]))

            for sample, (rich_map_raw, rich_pp_raw), (rich_map, rich_pp), (map_raw, pp_raw), (map_dynamic, pp_dynamic) in zip(
                session_samples,
                rich_raw_map_pp,
                rich_smoothed_map_pp,
                dynamic_raw_map_pp,
                dynamic_smoothed_map_pp,
            ):
                map_raw_blended = rich_map_raw + dynamic_gain_map * (map_raw - dynamic_raw_anchor_map)
                pp_raw_blended = rich_pp_raw + dynamic_gain_pp * (pp_raw - dynamic_raw_anchor_pp)
                map_blended = rich_map + dynamic_gain_map * (map_dynamic - dynamic_anchor_map)
                pp_blended = rich_pp + dynamic_gain_pp * (pp_dynamic - dynamic_anchor_pp)
                pred_dbp = map_blended - pp_blended / 3.0
                pred_sbp = pred_dbp + pp_blended
                prediction_rows.append(
                    {
                        "session": sample.session,
                        "row_index": sample.row_index,
                        "method": method,
                        "series": series,
                        "pred_MAP_raw": map_raw_blended,
                        "pred_PP_raw": pp_raw_blended,
                        "pred_MAP": map_blended,
                        "pred_PP": pp_blended,
                        "pred_SBP": pred_sbp,
                        "pred_DBP": pred_dbp,
                        "baseline_MAP_raw": baseline_map_raw,
                        "baseline_PP_raw": baseline_pp_raw,
                        "baseline_MAP": baseline_map,
                        "baseline_PP": baseline_pp,
                        "rich_MAP": rich_map,
                        "rich_PP": rich_pp,
                        "dynamic_MAP": map_dynamic,
                        "dynamic_PP": pp_dynamic,
                        "dynamic_delta_MAP": map_dynamic - dynamic_anchor_map,
                        "dynamic_delta_PP": pp_dynamic - dynamic_anchor_pp,
                        "dynamic_gain_MAP": dynamic_gain_map,
                        "dynamic_gain_PP": dynamic_gain_pp,
                        "initial_baseline_beats": summary.n,
                        "baseline_shrinkage": baseline_model.baseline_shrinkage,
                        "ref_SBP": _to_float(sample.row.get("ref_SBP")),
                        "ref_DBP": _to_float(sample.row.get("ref_DBP")),
                        "ref_MAP": sample.ref_map,
                        "ref_PP": sample.ref_pp,
                        "elapsed_s": _to_float(sample.row.get("経過時間_秒")),
                    }
                )
    return prediction_rows


def baseline_smoothed_predictions(reference_rows: list[dict[str, str]], eval_sessions: set[str]) -> list[dict[str, object]]:
    prediction_rows: list[dict[str, object]] = []
    for method, (sbp_col, dbp_col, valid_col, reject_col) in BASELINE_METHOD_SPECS.items():
        for row in reference_rows:
            session = str(row["_session"])
            if session not in eval_sessions:
                continue
            if sbp_col not in row or dbp_col not in row or valid_col not in row or reject_col not in row:
                continue
            if _to_int(row.get(valid_col)) != 1:
                continue
            if str(row.get(reject_col, "missing")).strip() != "ok":
                continue
            pred_sbp = _to_float(row.get(sbp_col))
            pred_dbp = _to_float(row.get(dbp_col))
            if not math.isfinite(pred_sbp) or not math.isfinite(pred_dbp):
                continue
            ref_sbp = _to_float(row.get("ref_SBP"))
            ref_dbp = _to_float(row.get("ref_DBP"))
            pred_map = (pred_sbp + 2.0 * pred_dbp) / 3.0
            pred_pp = pred_sbp - pred_dbp
            prediction_rows.append(
                {
                    "session": session,
                    "row_index": int(row["_row_index"]),
                    "method": method,
                    "series": "current_app_smoothed",
                    "pred_MAP_raw": pred_map,
                    "pred_PP_raw": pred_pp,
                    "pred_MAP": pred_map,
                    "pred_PP": pred_pp,
                    "pred_SBP": pred_sbp,
                    "pred_DBP": pred_dbp,
                    "ref_SBP": ref_sbp,
                    "ref_DBP": ref_dbp,
                    "ref_MAP": (ref_sbp + 2.0 * ref_dbp) / 3.0,
                    "ref_PP": ref_sbp - ref_dbp,
                    "elapsed_s": _to_float(row.get("経過時間_秒")),
                }
            )
    return prediction_rows


def build_summary(prediction_rows: list[dict[str, object]], label: str) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    sessions = sorted({str(row["session"]) for row in prediction_rows})
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in prediction_rows)]
    for session in [*sessions, "ALL"]:
        for method in methods:
            rows = [
                row
                for row in prediction_rows
                if row["method"] == method and (session == "ALL" or row["session"] == session)
            ]
            summary.append({"fit": label, "session": session, "method": method, **evaluate_predictions(rows)})
    return summary


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def build_session_style_summary(prediction_groups: list[tuple[str, list[dict[str, object]]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sessions = sorted({str(row["session"]) for _, predictions in prediction_groups for row in predictions})
    for series, predictions in prediction_groups:
        methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in predictions)]
        for session in [*sessions, "ALL"]:
            for method in methods:
                subset = [
                    row
                    for row in predictions
                    if row["method"] == method and (session == "ALL" or row["session"] == session)
                ]
                pred_pp = np.array([float(row["pred_PP"]) for row in subset], dtype=float)
                ref_pp = np.array([float(row["ref_PP"]) for row in subset], dtype=float)
                pp_error = pred_pp - ref_pp if subset else np.array([], dtype=float)
                pp_mae = float(np.mean(np.abs(pp_error))) if subset else float("nan")
                pp_bias = float(np.mean(pp_error)) if subset else float("nan")
                for target_label, pred_key, ref_key in (
                    ("SBP", "pred_SBP", "ref_SBP"),
                    ("DBP", "pred_DBP", "ref_DBP"),
                ):
                    if not subset:
                        rows.append(
                            {
                                "session": session,
                                "method": method,
                                "series": series,
                                "target": target_label,
                                "filters": EVALUATION_FILTERS,
                                "n": 0,
                                "mae": float("nan"),
                                "rmse": float("nan"),
                                "corr": float("nan"),
                                "signed_bias": float("nan"),
                                "pp_mae": float("nan"),
                                "pp_signed_bias": float("nan"),
                            }
                        )
                        continue
                    pred = np.array([float(row[pred_key]) for row in subset], dtype=float)
                    ref = np.array([float(row[ref_key]) for row in subset], dtype=float)
                    error = pred - ref
                    rows.append(
                        {
                            "session": session,
                            "method": method,
                            "series": series,
                            "target": target_label,
                            "filters": EVALUATION_FILTERS,
                            "n": int(len(subset)),
                            "mae": float(np.mean(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "corr": _corr(pred, ref),
                            "signed_bias": float(np.mean(error)),
                            "pp_mae": pp_mae,
                            "pp_signed_bias": pp_bias,
                        }
                    )
    return rows


def _load_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "realtime_ibi_bp_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "realtime_ibi_bp_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _x_value(row: dict[str, object]) -> float:
    value = row.get("elapsed_s")
    try:
        x = float(value)
        return x if math.isfinite(x) else float(row["row_index"])
    except (TypeError, ValueError):
        return float(row["row_index"])


def generate_timeseries_plots(
    prediction_groups: list[tuple[str, list[dict[str, object]]]],
    output_dir: Path,
) -> list[Path]:
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    plot_groups = [
        (series, rows)
        for series, rows in prediction_groups
        if series
        in (
            "current_app_smoothed",
            "refit_map_pp_smoothed",
            "smartphone_initial_baseline",
            "smartphone_rich_baseline",
            "smartphone_rich_dynamic_blend",
            "smartphone_shared_sinbpd_baseline",
        )
    ]
    sessions = sorted({str(row["session"]) for _, rows in plot_groups for row in rows})
    colors = {"RTBP": "#1f77b4", "SinBP_D": "#d62728", "SinBP_M": "#2ca02c"}
    line_styles = {
        "current_app_smoothed": "--",
        "refit_map_pp_smoothed": "-",
        "smartphone_initial_baseline": ":",
        "smartphone_rich_baseline": "-.",
        "smartphone_rich_dynamic_blend": (0, (5, 1)),
        "smartphone_shared_sinbpd_baseline": (0, (3, 1, 1, 1)),
    }

    for session in sessions:
        session_rows = [row for _, rows in plot_groups for row in rows if row["session"] == session]
        ref_rows_by_index: dict[int, dict[str, object]] = {}
        for row in session_rows:
            ref_rows_by_index[int(row["row_index"])] = row
        ref_rows = [ref_rows_by_index[key] for key in sorted(ref_rows_by_index)]
        if not ref_rows:
            continue
        x_ref = [_x_value(row) for row in ref_rows]
        for target, ref_key, pred_key, suffix in (
            ("SBP", "ref_SBP", "pred_SBP", "sbp"),
            ("DBP", "ref_DBP", "pred_DBP", "dbp"),
        ):
            fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
            ax.plot(x_ref, [float(row[ref_key]) for row in ref_rows], label=f"CNAP {target}", color="#111111", linewidth=2.4)
            for series, rows in plot_groups:
                for method in PLOT_METHOD_ORDER:
                    method_rows = sorted(
                        [row for row in rows if row["session"] == session and row["method"] == method],
                        key=lambda row: int(row["row_index"]),
                    )
                    if not method_rows:
                        continue
                    if series == "refit_map_pp_smoothed":
                        series_label = "new"
                    elif series == "smartphone_initial_baseline":
                        series_label = "baseline-adaptive"
                    elif series == "smartphone_rich_baseline":
                        series_label = "rich-baseline"
                    elif series == "smartphone_rich_dynamic_blend":
                        series_label = "rich+dynamic"
                    elif series == "smartphone_shared_sinbpd_baseline":
                        series_label = "shared-D-baseline"
                    else:
                        series_label = "current"
                    label = f"{method} {series_label}"
                    ax.plot(
                        [_x_value(row) for row in method_rows],
                        [float(row[pred_key]) for row in method_rows],
                        label=label,
                        color=colors[method],
                        linestyle=line_styles[series],
                        linewidth=1.3,
                        alpha=0.82 if series == "refit_map_pp_smoothed" else 0.50,
                    )
            ax.set_xlabel("Elapsed Time (s)")
            ax.set_ylabel(f"{target} (mmHg)")
            ax.set_title(f"{session} {target} Time Series")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            out = output_dir / f"{session}_{suffix}_timeseries.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            outputs.append(out)
    return outputs


def generate_all_scatter_plots(prediction_rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    plt = _load_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    colors = {"RTBP": "#1f77b4", "SinBP_D": "#d62728", "SinBP_M": "#2ca02c"}
    for target, ref_key, pred_key, suffix in (
        ("SBP", "ref_SBP", "pred_SBP", "sbp"),
        ("DBP", "ref_DBP", "pred_DBP", "dbp"),
    ):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        all_ref: list[float] = []
        all_pred: list[float] = []
        for method in REFIT_METHOD_ORDER:
            rows = [row for row in prediction_rows if row["method"] == method]
            if not rows:
                continue
            ref = [float(row[ref_key]) for row in rows]
            pred = [float(row[pred_key]) for row in rows]
            all_ref.extend(ref)
            all_pred.extend(pred)
            ax.scatter(ref, pred, label=method, s=18, alpha=0.65, color=colors[method])
        if all_ref and all_pred:
            lower = min(min(all_ref), min(all_pred)) - 3.0
            upper = max(max(all_ref), max(all_pred)) + 3.0
            ax.plot([lower, upper], [lower, upper], color="#111111", linestyle="--", linewidth=1.0, label="y=x")
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        ax.set_xlabel(f"CNAP {target} (mmHg)")
        ax.set_ylabel(f"Predicted {target} (mmHg)")
        ax.set_title(f"ALL {target} Scatter (refit)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = output_dir / f"all_{suffix}_scatter.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def coefficients_payload(models: FittedModels, sessions_root: Path, ridge_alpha: float) -> dict[str, object]:
    rtbp_sbp, rtbp_dbp = map_pp_to_sbp_dbp(models.rtbp_map, models.rtbp_pp)
    sinbpd_gamma, sinbpd_delta = map_pp_to_sbp_dbp(models.sinbpd_residual_map, models.sinbpd_residual_pp)
    sinbpm_sbp, sinbpm_dbp = map_pp_to_sbp_dbp(models.sinbpm_map, models.sinbpm_pp)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sessions_root": str(sessions_root),
        "ridge_alpha": ridge_alpha,
        "note": "CNAP is used only as offline training/evaluation labels. Runtime app uses these fixed coefficients and does not read CNAP values.",
        "alpha_map": ALPHA_MAP,
        "alpha_pp": ALPHA_PP,
        "models": {
            "RTBP": {
                "features": list(RTBP.feature_cols),
                "MAP": models.rtbp_map.tolist(),
                "PP": models.rtbp_pp.tolist(),
                "SBP": rtbp_sbp.tolist(),
                "DBP": rtbp_dbp.tolist(),
            },
            "SinBP_D": {
                "architecture": "RTBP base with M2 A/HR/relTTP features plus residual [intercept, Stiffness_sin, E]",
                "residual_features": ["M2_Stiffness_used", "M2_E_used"],
                "residual_MAP": models.sinbpd_residual_map.tolist(),
                "residual_PP": models.sinbpd_residual_pp.tolist(),
                "GAMMA_SBP_correction": sinbpd_gamma.tolist(),
                "DELTA_DBP_correction": sinbpd_delta.tolist(),
            },
            "SinBP_M": {
                "features": list(SINBP_M.feature_cols),
                "MAP": models.sinbpm_map.tolist(),
                "PP": models.sinbpm_pp.tolist(),
                "SBP": sinbpm_sbp.tolist(),
                "DBP": sinbpm_dbp.tolist(),
            },
        },
    }


def adaptive_coefficients_payload(models: AdaptiveModels) -> dict[str, object]:
    return {
        method: {
            "note": (
                "Experimental smartphone-only baseline adaptation. Runtime inputs are the first "
                f"{INITIAL_BASELINE_BEATS} good beats' smartphone features plus beat-local deltas; "
                "CNAP is used only as offline training/evaluation labels. Do not ship without LOSO validation."
            ),
            "feature_names": list(model.feature_names),
            "summary_source_columns": list(model.summary_source_columns),
            "summary_feature_names": list(model.summary_feature_names),
            "initial_baseline_beats": INITIAL_BASELINE_BEATS,
            "baseline_shrinkage": model.baseline_shrinkage,
            "population_MAP_anchor": model.population_map_anchor,
            "population_PP_anchor": model.population_pp_anchor,
            "baseline_MAP": model.baseline_map.tolist(),
            "baseline_PP": model.baseline_pp.tolist(),
            "delta_MAP": model.delta_map.tolist(),
            "delta_PP": model.delta_pp.tolist(),
        }
        for method, model in models.by_method.items()
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit fixed realtime MAP/PP BP coefficients from existing sessions.")
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Data" / "realtime_sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Data" / "realtime_coefficient",
    )
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--baseline-ridge-alpha", type=float, default=DEFAULT_BASELINE_RIDGE_ALPHA)
    parser.add_argument("--rich-baseline-ridge-alpha", type=float, default=DEFAULT_RICH_BASELINE_RIDGE_ALPHA)
    parser.add_argument("--baseline-shrinkage", type=float, default=DEFAULT_BASELINE_SHRINKAGE)
    parser.add_argument("--dynamic-blend-gain-map", type=float, default=DEFAULT_DYNAMIC_BLEND_GAIN)
    parser.add_argument("--dynamic-blend-gain-pp", type=float, default=DEFAULT_DYNAMIC_BLEND_GAIN)
    args = parser.parse_args()

    rows = load_reference_rows(args.sessions_root)
    if not rows:
        raise SystemExit("no usable reference rows found")
    samples_by_method = {
        "RTBP": build_samples(rows, RTBP),
        "SinBP_D": build_samples(rows, SINBP_D),
        "SinBP_M": build_samples(rows, SINBP_M),
    }
    sessions = sorted({str(row["_session"]) for row in rows})
    if len(sessions) < 2:
        raise SystemExit("need at least two sessions for this fit")

    output_dir = args.output_dir / f"realtime_map_pp_fit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    evaluation_dir = output_dir / "evaluation"
    plots_dir = evaluation_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    all_sessions = set(sessions)
    models = train_models(samples_by_method, all_sessions, args.ridge_alpha)
    adaptive_models = train_adaptive_models(
        samples_by_method,
        all_sessions,
        baseline_ridge_alpha=args.baseline_ridge_alpha,
        delta_ridge_alpha=args.ridge_alpha,
        baseline_shrinkage=args.baseline_shrinkage,
    )
    rich_baseline_models = train_rich_baseline_models(
        samples_by_method,
        all_sessions,
        baseline_ridge_alpha=args.rich_baseline_ridge_alpha,
        delta_ridge_alpha=args.ridge_alpha,
        baseline_shrinkage=args.baseline_shrinkage,
    )
    shared_sinbpd_baseline_models = train_shared_sinbpd_baseline_models(
        samples_by_method,
        all_sessions,
        baseline_ridge_alpha=args.rich_baseline_ridge_alpha,
        delta_ridge_alpha=args.ridge_alpha,
        baseline_shrinkage=args.baseline_shrinkage,
    )
    baseline_predictions = baseline_smoothed_predictions(rows, all_sessions)
    predictions = replay_model(models, samples_by_method, all_sessions)
    adaptive_predictions = replay_adaptive_model(
        adaptive_models,
        samples_by_method,
        all_sessions,
        series="smartphone_initial_baseline",
    )
    rich_baseline_predictions = replay_adaptive_model(
        rich_baseline_models,
        samples_by_method,
        all_sessions,
        series="smartphone_rich_baseline",
        rich_summary=True,
    )
    rich_dynamic_blend_predictions = replay_baseline_dynamic_blend(
        rich_baseline_models,
        models,
        samples_by_method,
        all_sessions,
        series="smartphone_rich_dynamic_blend",
        dynamic_gain_map=args.dynamic_blend_gain_map,
        dynamic_gain_pp=args.dynamic_blend_gain_pp,
        rich_summary=True,
    )
    shared_sinbpd_baseline_predictions = replay_adaptive_model(
        shared_sinbpd_baseline_models,
        samples_by_method,
        all_sessions,
        series="smartphone_shared_sinbpd_baseline",
        rich_summary=True,
    )
    baseline_summary = build_summary(baseline_predictions, "current_app_smoothed")
    summary = build_summary(predictions, "all_sessions_fit")
    adaptive_summary = build_summary(adaptive_predictions, "smartphone_initial_baseline")
    rich_baseline_summary = build_summary(rich_baseline_predictions, "smartphone_rich_baseline")
    rich_dynamic_blend_summary = build_summary(rich_dynamic_blend_predictions, "smartphone_rich_dynamic_blend")
    shared_sinbpd_baseline_summary = build_summary(
        shared_sinbpd_baseline_predictions,
        "smartphone_shared_sinbpd_baseline",
    )

    loso_predictions: list[dict[str, object]] = []
    adaptive_loso_predictions: list[dict[str, object]] = []
    rich_baseline_loso_predictions: list[dict[str, object]] = []
    rich_dynamic_blend_loso_predictions: list[dict[str, object]] = []
    shared_sinbpd_baseline_loso_predictions: list[dict[str, object]] = []
    for held_out in sessions:
        loso_models = train_models(samples_by_method, all_sessions - {held_out}, args.ridge_alpha)
        for row in replay_model(loso_models, samples_by_method, {held_out}):
            row = dict(row)
            row["held_out_session"] = held_out
            loso_predictions.append(row)
        adaptive_loso_models = train_adaptive_models(
            samples_by_method,
            all_sessions - {held_out},
            baseline_ridge_alpha=args.baseline_ridge_alpha,
            delta_ridge_alpha=args.ridge_alpha,
            baseline_shrinkage=args.baseline_shrinkage,
        )
        for row in replay_adaptive_model(
            adaptive_loso_models,
            samples_by_method,
            {held_out},
            series="smartphone_initial_baseline_loso",
        ):
            row = dict(row)
            row["held_out_session"] = held_out
            adaptive_loso_predictions.append(row)
        rich_loso_models = train_rich_baseline_models(
            samples_by_method,
            all_sessions - {held_out},
            baseline_ridge_alpha=args.rich_baseline_ridge_alpha,
            delta_ridge_alpha=args.ridge_alpha,
            baseline_shrinkage=args.baseline_shrinkage,
        )
        for row in replay_adaptive_model(
            rich_loso_models,
            samples_by_method,
            {held_out},
            series="smartphone_rich_baseline_loso",
            rich_summary=True,
        ):
            row = dict(row)
            row["held_out_session"] = held_out
            rich_baseline_loso_predictions.append(row)
        for row in replay_baseline_dynamic_blend(
            rich_loso_models,
            loso_models,
            samples_by_method,
            {held_out},
            series="smartphone_rich_dynamic_blend_loso",
            dynamic_gain_map=args.dynamic_blend_gain_map,
            dynamic_gain_pp=args.dynamic_blend_gain_pp,
            rich_summary=True,
        ):
            row = dict(row)
            row["held_out_session"] = held_out
            rich_dynamic_blend_loso_predictions.append(row)
        shared_sinbpd_loso_models = train_shared_sinbpd_baseline_models(
            samples_by_method,
            all_sessions - {held_out},
            baseline_ridge_alpha=args.rich_baseline_ridge_alpha,
            delta_ridge_alpha=args.ridge_alpha,
            baseline_shrinkage=args.baseline_shrinkage,
        )
        for row in replay_adaptive_model(
            shared_sinbpd_loso_models,
            samples_by_method,
            {held_out},
            series="smartphone_shared_sinbpd_baseline_loso",
            rich_summary=True,
        ):
            row = dict(row)
            row["held_out_session"] = held_out
            shared_sinbpd_baseline_loso_predictions.append(row)
    loso_summary = build_summary(loso_predictions, "leave_one_session_out")
    adaptive_loso_summary = build_summary(adaptive_loso_predictions, "smartphone_initial_baseline_loso")
    rich_baseline_loso_summary = build_summary(rich_baseline_loso_predictions, "smartphone_rich_baseline_loso")
    rich_dynamic_blend_loso_summary = build_summary(
        rich_dynamic_blend_loso_predictions,
        "smartphone_rich_dynamic_blend_loso",
    )
    shared_sinbpd_baseline_loso_summary = build_summary(
        shared_sinbpd_baseline_loso_predictions,
        "smartphone_shared_sinbpd_baseline_loso",
    )
    session_style_summary = build_session_style_summary(
        [
            ("current_app_smoothed", baseline_predictions),
            ("refit_map_pp_smoothed", predictions),
            ("smartphone_initial_baseline", adaptive_predictions),
            ("smartphone_rich_baseline", rich_baseline_predictions),
            ("smartphone_rich_dynamic_blend", rich_dynamic_blend_predictions),
            ("smartphone_shared_sinbpd_baseline", shared_sinbpd_baseline_predictions),
            ("smartphone_initial_baseline_loso", adaptive_loso_predictions),
            ("smartphone_rich_baseline_loso", rich_baseline_loso_predictions),
            ("smartphone_rich_dynamic_blend_loso", rich_dynamic_blend_loso_predictions),
            ("smartphone_shared_sinbpd_baseline_loso", shared_sinbpd_baseline_loso_predictions),
            ("leave_one_session_out", loso_predictions),
        ]
    )

    payload = coefficients_payload(models, args.sessions_root, args.ridge_alpha)
    payload["sessions"] = sessions
    payload["sample_counts"] = {method: len(samples) for method, samples in samples_by_method.items()}
    payload["experimental_smartphone_initial_baseline"] = adaptive_coefficients_payload(adaptive_models)
    payload["experimental_smartphone_rich_baseline"] = adaptive_coefficients_payload(rich_baseline_models)
    payload["experimental_smartphone_rich_dynamic_blend"] = {
        "note": (
            "Smartphone-only best-of-both candidate: absolute MAP/PP baseline comes from "
            "experimental_smartphone_rich_baseline, while short-term dynamics are the "
            "refit_map_pp_smoothed model delta from the first good beats. CNAP is only used "
            "offline to fit the fixed baseline and dynamic coefficients."
        ),
        "baseline_model": "experimental_smartphone_rich_baseline",
        "dynamic_model": "models",
        "dynamic_anchor_beats": INITIAL_BASELINE_BEATS,
        "dynamic_gain_MAP": args.dynamic_blend_gain_map,
        "dynamic_gain_PP": args.dynamic_blend_gain_pp,
    }
    payload["experimental_smartphone_shared_sinbpd_baseline"] = adaptive_coefficients_payload(
        shared_sinbpd_baseline_models
    )

    plot_paths = [
        *generate_timeseries_plots(
            [
                ("current_app_smoothed", baseline_predictions),
                ("refit_map_pp_smoothed", predictions),
                ("smartphone_initial_baseline", adaptive_predictions),
                ("smartphone_rich_baseline", rich_baseline_predictions),
                ("smartphone_rich_dynamic_blend", rich_dynamic_blend_predictions),
                ("smartphone_shared_sinbpd_baseline", shared_sinbpd_baseline_predictions),
            ],
            plots_dir,
        ),
        *generate_all_scatter_plots(predictions, plots_dir),
    ]

    write_csv(evaluation_dir / "current_app_predictions.csv", baseline_predictions)
    write_csv(evaluation_dir / "current_app_evaluation_summary.csv", baseline_summary)
    write_csv(evaluation_dir / "refit_predictions.csv", predictions)
    write_csv(evaluation_dir / "refit_evaluation_summary.csv", summary)
    write_csv(evaluation_dir / "smartphone_initial_baseline_predictions.csv", adaptive_predictions)
    write_csv(evaluation_dir / "smartphone_initial_baseline_evaluation_summary.csv", adaptive_summary)
    write_csv(evaluation_dir / "smartphone_rich_baseline_predictions.csv", rich_baseline_predictions)
    write_csv(evaluation_dir / "smartphone_rich_baseline_evaluation_summary.csv", rich_baseline_summary)
    write_csv(evaluation_dir / "smartphone_rich_dynamic_blend_predictions.csv", rich_dynamic_blend_predictions)
    write_csv(evaluation_dir / "smartphone_rich_dynamic_blend_evaluation_summary.csv", rich_dynamic_blend_summary)
    write_csv(evaluation_dir / "smartphone_shared_sinbpd_baseline_predictions.csv", shared_sinbpd_baseline_predictions)
    write_csv(
        evaluation_dir / "smartphone_shared_sinbpd_baseline_evaluation_summary.csv",
        shared_sinbpd_baseline_summary,
    )
    write_csv(evaluation_dir / "smartphone_initial_baseline_loso_predictions.csv", adaptive_loso_predictions)
    write_csv(evaluation_dir / "smartphone_initial_baseline_loso_evaluation_summary.csv", adaptive_loso_summary)
    write_csv(evaluation_dir / "smartphone_rich_baseline_loso_predictions.csv", rich_baseline_loso_predictions)
    write_csv(evaluation_dir / "smartphone_rich_baseline_loso_evaluation_summary.csv", rich_baseline_loso_summary)
    write_csv(evaluation_dir / "smartphone_rich_dynamic_blend_loso_predictions.csv", rich_dynamic_blend_loso_predictions)
    write_csv(
        evaluation_dir / "smartphone_rich_dynamic_blend_loso_evaluation_summary.csv",
        rich_dynamic_blend_loso_summary,
    )
    write_csv(
        evaluation_dir / "smartphone_shared_sinbpd_baseline_loso_predictions.csv",
        shared_sinbpd_baseline_loso_predictions,
    )
    write_csv(
        evaluation_dir / "smartphone_shared_sinbpd_baseline_loso_evaluation_summary.csv",
        shared_sinbpd_baseline_loso_summary,
    )
    write_csv(evaluation_dir / "loso_predictions.csv", loso_predictions)
    write_csv(evaluation_dir / "loso_evaluation_summary.csv", loso_summary)
    write_csv(evaluation_dir / "session_evaluation_summary.csv", session_style_summary)
    (evaluation_dir / "session_evaluation_summary.json").write_text(
        json.dumps(session_style_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "fixed_coefficients.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (evaluation_dir / "coefficient_report.md").write_text(
        "\n".join(
            [
                "# Realtime Coefficient Report",
                "",
                f"- sessions_root: {args.sessions_root}",
                f"- ridge_alpha: {args.ridge_alpha}",
                f"- baseline_ridge_alpha: {args.baseline_ridge_alpha}",
                f"- rich_baseline_ridge_alpha: {args.rich_baseline_ridge_alpha}",
                f"- baseline_shrinkage: {args.baseline_shrinkage}",
                f"- dynamic_blend_gain_map: {args.dynamic_blend_gain_map}",
                f"- dynamic_blend_gain_pp: {args.dynamic_blend_gain_pp}",
                f"- initial_baseline_beats: {INITIAL_BASELINE_BEATS}",
                f"- filters: {EVALUATION_FILTERS}",
                "- CNAP usage: offline training/evaluation labels only; not used by the runtime app.",
                "- smartphone_initial_baseline: experimental smartphone-only baseline adaptation from initial beat features; validate LOSO before shipping.",
                "- smartphone_rich_baseline: richer smartphone-only baseline adaptation using initial raw/smoothed estimates, coefficient terms, beat stats, and method-specific morphology features; validate LOSO before shipping.",
                "- smartphone_rich_dynamic_blend: rich baseline anchor plus refit MAP/PP deltas from the initial dynamic anchor; intended to keep per-subject baseline while restoring real-time up/down movement.",
                "- smartphone_shared_sinbpd_baseline: shared baseline from SinBP_D rich features, with method-specific deltas; this is the most physiologically explainable smartphone-only baseline candidate.",
                "",
                "## Outputs",
                "",
                "- session_evaluation_summary.csv",
                "- session_evaluation_summary.json",
                "- current_app_evaluation_summary.csv",
                "- refit_evaluation_summary.csv",
                "- smartphone_initial_baseline_evaluation_summary.csv",
                "- smartphone_rich_baseline_evaluation_summary.csv",
                "- smartphone_rich_dynamic_blend_evaluation_summary.csv",
                "- smartphone_shared_sinbpd_baseline_evaluation_summary.csv",
                "- smartphone_initial_baseline_loso_evaluation_summary.csv",
                "- smartphone_rich_baseline_loso_evaluation_summary.csv",
                "- smartphone_rich_dynamic_blend_loso_evaluation_summary.csv",
                "- smartphone_shared_sinbpd_baseline_loso_evaluation_summary.csv",
                "- loso_evaluation_summary.csv",
                "- fixed_coefficients.json",
                "",
                "## Plots",
                "",
                *[f"- plots/{path.name}" for path in plot_paths],
                "",
            ]
        ),
        encoding="utf-8",
    )

    for row in session_style_summary:
        if row["session"] == "ALL" and row["target"] == "SBP":
            print(
                f"{row['series']} {row['method']} SBP "
                f"MAE={float(row['mae']):.3f} RMSE={float(row['rmse']):.3f} Corr={float(row['corr']):.3f}"
            )
    print(f"saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
