from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COEFFICIENTS_PATH = (
    REPO_ROOT / "RealTime-IBI-BP" / "app" / "src" / "main" / "res" / "raw" / "realtime_bp_coefficients.json"
)

RTBP_TERM_LABELS = ("intercept", "A", "HR", "V2P_relTTP", "P2V_relTTP")
SINBPD_TERM_LABELS = ("intercept", "A", "HR", "V2P_relTTP", "P2V_relTTP", "Stiffness", "E")
SINBPM_TERM_LABELS = ("intercept", "A", "HR", "Mean", "sinPhi", "cosPhi")
SINBPD_EONLY_TERM_LABELS = ("intercept", "A", "HR", "V2P_relTTP", "P2V_relTTP", "E")
SINBPD_E2_TERM_LABELS = ("intercept", "A", "HR", "V2P_relTTP", "P2V_relTTP", "E", "E2")
SINBPD_LOCALA_TERM_LABELS = ("intercept", "A_local", "HR", "V2P_relTTP", "P2V_relTTP", "Stiffness", "E")

# PP inversion diagnostics showed that the direct SBP/DBP regressions were mainly broken by
# 1) the HR sign in pulse-pressure-related motion and
# 2) over-large residual / phase terms in SinBP(D) / SinBP(M).
# Moving to MAP/PP-first already flips the HR sign via refit coefficients; the remaining
# suppression is intentionally mild and explainable.
DEFAULT_PP_TERM_SCALES = {
    "RTBP": {"HR": 1.0},
    "SinBP_D": {"HR": 1.0, "E": 0.65},
    "SinBP_M": {"HR": 1.0, "sinPhi": 0.80},
    "SinBP_D_EOnly": {"HR": 1.0, "E": 0.65},
    "SinBP_D_E2": {"HR": 1.0, "E": 0.65, "E2": 0.65},
    "SinBP_D_LocalA": {"HR": 1.0, "E": 0.65, "Stiffness": 1.0},
}
CORE_APP_PREFIXES = ("M1_", "M2_", "M3_")
DEFAULT_PRESERVE_EXISTING_CORE_COLUMNS = True
DEFAULT_ENABLE_TRACKING_BLEND_OVERRIDES = False

TRACKING_BLEND_SERIES_KEY = "experimental_smartphone_rich_dynamic_blend"
TRACKING_BLEND_METHOD_SPECS = (
    ("RTBP", "M1", "M1_output_valid", "M1_reject_reason", RTBP_TERM_LABELS),
    ("SinBP_D", "M2", "M2_output_valid", "M2_reject_reason", SINBPD_TERM_LABELS),
    ("SinBP_M", "M3", "M3_output_valid", "M3_reject_reason", SINBPM_TERM_LABELS),
)
BASELINE_MAP_MIN = 40.0
BASELINE_MAP_MAX = 180.0
BASELINE_PP_MIN = 10.0
BASELINE_PP_MAX = 120.0

_LABEL_TO_INDEX = {
    "RTBP": {label: index for index, label in enumerate(RTBP_TERM_LABELS)},
    "SinBP_D": {label: index for index, label in enumerate(SINBPD_TERM_LABELS)},
    "SinBP_M": {label: index for index, label in enumerate(SINBPM_TERM_LABELS)},
    "SinBP_D_EOnly": {label: index for index, label in enumerate(SINBPD_EONLY_TERM_LABELS)},
    "SinBP_D_E2": {label: index for index, label in enumerate(SINBPD_E2_TERM_LABELS)},
    "SinBP_D_LocalA": {label: index for index, label in enumerate(SINBPD_LOCALA_TERM_LABELS)},
}


def _is_core_app_column(column: str) -> bool:
    return column.startswith(CORE_APP_PREFIXES)


def load_runtime_coefficients(path: Path | None = None) -> dict[str, object]:
    coefficients_path = path or DEFAULT_COEFFICIENTS_PATH
    return json.loads(coefficients_path.read_text(encoding="utf-8"))


def scale_pp_coefficients(
    method: str,
    coefficients: np.ndarray,
    term_scales: dict[str, float] | None = None,
) -> np.ndarray:
    scaled = np.array(coefficients, dtype=float, copy=True)
    for label, gain in (term_scales or DEFAULT_PP_TERM_SCALES.get(method, {})).items():
        feature_index = _LABEL_TO_INDEX[method].get(label)
        if feature_index is None or feature_index == 0:
            continue
        coefficients_index = feature_index
        if coefficients_index < len(scaled):
            scaled[coefficients_index] *= gain
    return scaled


def scale_rtbp_pp_coefficients(coefficients: np.ndarray) -> np.ndarray:
    return scale_pp_coefficients("RTBP", coefficients)


def scale_sinbpm_pp_coefficients(coefficients: np.ndarray) -> np.ndarray:
    return scale_pp_coefficients("SinBP_M", coefficients)


def scale_sinbpd_residual_pp_coefficients(coefficients: np.ndarray) -> np.ndarray:
    scaled = np.array(coefficients, dtype=float, copy=True)
    if len(scaled) > 2:
        scaled[2] *= DEFAULT_PP_TERM_SCALES["SinBP_D"]["E"]
    return scaled


def map_pp_to_sbp_dbp_terms(map_terms: np.ndarray, pp_terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return map_terms + (2.0 / 3.0) * pp_terms, map_terms - (1.0 / 3.0) * pp_terms


def _linear_terms(coefficients: np.ndarray, features: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    terms = np.zeros(len(labels), dtype=float)
    if len(coefficients) == 0:
        return terms
    terms[0] = float(coefficients[0])
    usable = min(len(features), len(labels) - 1, len(coefficients) - 1)
    if usable > 0:
        terms[1 : usable + 1] = coefficients[1 : usable + 1] * features[:usable]
    return terms


def _to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _feature_vector_for_prefix(prefix: str, row: pd.Series) -> np.ndarray | None:
    if prefix == "M1":
        features = np.array(
            [
                _to_float(row.get("M1_A_used")),
                _to_float(row.get("M1_HR_used")),
                _to_float(row.get("M1_V2P_relTTP_used")),
                _to_float(row.get("M1_P2V_relTTP_used")),
            ],
            dtype=float,
        )
    elif prefix == "M2":
        features = np.array(
            [
                _to_float(row.get("M2_A_used")),
                _to_float(row.get("M2_HR_used")),
                _to_float(row.get("M2_V2P_relTTP_used")),
                _to_float(row.get("M2_P2V_relTTP_used")),
                _to_float(row.get("M2_Stiffness_used")),
                _to_float(row.get("M2_E_used")),
            ],
            dtype=float,
        )
    elif prefix == "M3":
        features = np.array(
            [
                _to_float(row.get("M3_A_used")),
                _to_float(row.get("M3_HR_used")),
                _to_float(row.get("M3_Mean_used")),
                _to_float(row.get("M3_sinPhi_used")),
                _to_float(row.get("M3_cosPhi_used")),
            ],
            dtype=float,
        )
    else:
        return None
    return features if np.isfinite(features).all() else None


def _is_valid_row(row: pd.Series, valid_col: str, reject_col: str) -> bool:
    return int(_to_float(row.get(valid_col, 0.0))) == 1 and str(row.get(reject_col, "missing")).strip() == "ok"


def _clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


def predict(coefficients: np.ndarray, features: np.ndarray) -> float:
    return float(coefficients[0] + np.dot(coefficients[1:], features))


def _clamp_bp(method: str, sbp: float, dbp: float) -> tuple[float, float]:
    constrained_sbp = sbp
    constrained_dbp = dbp
    if method in {"SinBP_D", "SinBP_D_EOnly", "SinBP_D_E2", "SinBP_D_LocalA", "SinBP_M"} and constrained_sbp < constrained_dbp + 20.0:
        constrained_sbp = constrained_dbp + 20.0
    constrained_sbp = _clamp(constrained_sbp, 60.0, 200.0)
    constrained_dbp = _clamp(constrained_dbp, 40.0, 150.0)
    return constrained_sbp, constrained_dbp


def _sanitize_baseline_anchor(value: float, population_anchor: float, minimum: float, maximum: float) -> float:
    if not math.isfinite(value) or value < minimum or value > maximum:
        return population_anchor
    return value


def _smooth_map_pp(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    smoothed: list[tuple[float, float]] = []
    last_map = float("nan")
    last_pp = float("nan")
    for map_value, pp_value in values:
        if not math.isfinite(last_map):
            map_smoothed = map_value
            pp_smoothed = pp_value
        else:
            map_smoothed = 0.30 * map_value + (1.0 - 0.30) * last_map
            pp_smoothed = 0.50 * pp_value + (1.0 - 0.50) * last_pp
        last_map = map_smoothed
        last_pp = pp_smoothed
        smoothed.append((map_smoothed, pp_smoothed))
    return smoothed


def _build_rich_summary(
    samples: list[dict[str, object]],
    source_columns: tuple[str, ...],
    initial_beats: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    initial_samples = samples[:initial_beats]
    if not initial_samples:
        return None
    anchor_features = np.median(np.vstack([sample["features"] for sample in initial_samples]), axis=0)
    summary_values: list[float] = []
    for column in source_columns:
        values = np.array([_to_float(sample["row"].get(column)) for sample in initial_samples], dtype=float)
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
    return np.array(summary_values, dtype=float), anchor_features


def _tracking_blend_overrides(df: pd.DataFrame, runtime_coefficients: dict[str, object]) -> dict[str, list[float]]:
    blend_spec = runtime_coefficients.get(TRACKING_BLEND_SERIES_KEY)
    if not isinstance(blend_spec, dict):
        return {}
    baseline_key = str(blend_spec.get("baseline_model", "")).strip()
    baseline_models = runtime_coefficients.get(baseline_key)
    if not isinstance(baseline_models, dict):
        return {}

    session_col = "session_id" if "session_id" in df.columns else None
    session_values = (
        df[session_col].fillna("__single_session__").astype(str)
        if session_col
        else pd.Series(["__single_session__"] * len(df), index=df.index)
    )
    index_positions = {index: position for position, index in enumerate(df.index)}
    dynamic_gain_map = _to_float(blend_spec.get("dynamic_gain_MAP"))
    dynamic_gain_pp = _to_float(blend_spec.get("dynamic_gain_PP"))
    if not math.isfinite(dynamic_gain_map):
        dynamic_gain_map = 0.25
    if not math.isfinite(dynamic_gain_pp):
        dynamic_gain_pp = 0.25
    dynamic_gain_map_by_method = blend_spec.get("dynamic_gain_MAP_by_method")
    dynamic_gain_pp_by_method = blend_spec.get("dynamic_gain_PP_by_method")

    def _resolve_method_gain(global_gain: float, by_method: object, method_name: str) -> float:
        if isinstance(by_method, dict):
            method_gain = _to_float(by_method.get(method_name))
            if math.isfinite(method_gain):
                return method_gain
        return global_gain

    overrides: dict[str, list[float]] = {}
    for method_name, prefix, valid_col, reject_col, labels in TRACKING_BLEND_METHOD_SPECS:
        method_dynamic_gain_map = _resolve_method_gain(dynamic_gain_map, dynamic_gain_map_by_method, method_name)
        method_dynamic_gain_pp = _resolve_method_gain(dynamic_gain_pp, dynamic_gain_pp_by_method, method_name)
        baseline_model = baseline_models.get(method_name)
        if not isinstance(baseline_model, dict):
            continue
        source_columns = tuple(str(column) for column in baseline_model.get("summary_source_columns", []))
        baseline_map_coefficients = np.array(baseline_model.get("baseline_MAP", []), dtype=float)
        baseline_pp_coefficients = np.array(baseline_model.get("baseline_PP", []), dtype=float)
        delta_map_coefficients = np.array(baseline_model.get("delta_MAP", []), dtype=float)
        delta_pp_coefficients = np.array(baseline_model.get("delta_PP", []), dtype=float)
        if (
            len(source_columns) == 0
            or len(baseline_map_coefficients) == 0
            or len(baseline_pp_coefficients) == 0
            or len(delta_map_coefficients) == 0
            or len(delta_pp_coefficients) == 0
        ):
            continue

        overrides[f"{prefix}_MAP_model_raw"] = [np.nan] * len(df)
        overrides[f"{prefix}_PP_model_raw"] = [np.nan] * len(df)
        overrides[f"{prefix}_SBP"] = [np.nan] * len(df)
        overrides[f"{prefix}_DBP"] = [np.nan] * len(df)
        overrides[f"{prefix}_SBP_raw"] = [np.nan] * len(df)
        overrides[f"{prefix}_DBP_raw"] = [np.nan] * len(df)
        overrides[f"{prefix}_MAP_raw"] = [np.nan] * len(df)
        overrides[f"{prefix}_PP_raw"] = [np.nan] * len(df)
        overrides[f"{prefix}_MAP_smoothed"] = [np.nan] * len(df)
        overrides[f"{prefix}_PP_smoothed"] = [np.nan] * len(df)
        overrides[f"{prefix}_MAP_calibrated"] = [np.nan] * len(df)
        overrides[f"{prefix}_PP_calibrated"] = [np.nan] * len(df)
        overrides[f"{prefix}_SBP_smoothed"] = [np.nan] * len(df)
        overrides[f"{prefix}_DBP_smoothed"] = [np.nan] * len(df)
        overrides[f"{prefix}_SBP_calibrated"] = [np.nan] * len(df)
        overrides[f"{prefix}_DBP_calibrated"] = [np.nan] * len(df)
        overrides[f"{prefix}_postprocess_applied"] = [0] * len(df)
        for label in labels:
            overrides[f"{prefix}_MAP_coef_{label}"] = [np.nan] * len(df)
            overrides[f"{prefix}_PP_coef_{label}"] = [np.nan] * len(df)
            overrides[f"{prefix}_MAP_term_{label}"] = [np.nan] * len(df)
            overrides[f"{prefix}_PP_term_{label}"] = [np.nan] * len(df)
            overrides[f"{prefix}_SBP_term_{label}"] = [np.nan] * len(df)
            overrides[f"{prefix}_DBP_term_{label}"] = [np.nan] * len(df)
        if prefix == "M2":
            overrides["M2_SBP_base"] = [np.nan] * len(df)
            overrides["M2_DBP_base"] = [np.nan] * len(df)
            overrides["M2_SBP_correction"] = [np.nan] * len(df)
            overrides["M2_DBP_correction"] = [np.nan] * len(df)

        for session_id in pd.unique(session_values):
            session_indices = list(df.index[session_values == session_id])
            samples: list[dict[str, object]] = []
            for index in session_indices:
                row = df.loc[index]
                if not _is_valid_row(row, valid_col, reject_col):
                    continue
                features = _feature_vector_for_prefix(prefix, row)
                if features is None:
                    continue
                prediction = _predict_method(prefix, row, runtime_coefficients)
                if prediction is None:
                    continue
                samples.append(
                    {
                        "index": index,
                        "row": row,
                        "features": features,
                        "map_raw": float(prediction["map_model_raw"]),
                        "pp_raw": float(prediction["pp_model_raw"]),
                    }
                )
            if not samples:
                continue

            baseline_beats = int(_to_float(baseline_model.get("initial_baseline_beats")))
            dynamic_anchor_beats = int(_to_float(blend_spec.get("dynamic_anchor_beats")))
            initial_beats = baseline_beats if baseline_beats > 0 else dynamic_anchor_beats
            initial_beats = initial_beats if initial_beats > 0 else 30
            summary = _build_rich_summary(samples, source_columns, initial_beats)
            if summary is None:
                continue
            summary_features, anchor_features = summary

            population_map_anchor = _to_float(baseline_model.get("population_MAP_anchor"))
            population_pp_anchor = _to_float(baseline_model.get("population_PP_anchor"))
            baseline_shrinkage = _to_float(baseline_model.get("baseline_shrinkage"))
            if not math.isfinite(baseline_shrinkage):
                baseline_shrinkage = 1.0

            baseline_map_raw = predict(baseline_map_coefficients, summary_features)
            baseline_pp_raw = predict(baseline_pp_coefficients, summary_features)
            baseline_map = population_map_anchor + baseline_shrinkage * (baseline_map_raw - population_map_anchor)
            baseline_pp = population_pp_anchor + baseline_shrinkage * (baseline_pp_raw - population_pp_anchor)
            baseline_map = _sanitize_baseline_anchor(
                baseline_map,
                population_map_anchor,
                BASELINE_MAP_MIN,
                BASELINE_MAP_MAX,
            )
            baseline_pp = _sanitize_baseline_anchor(
                baseline_pp,
                population_pp_anchor,
                BASELINE_PP_MIN,
                BASELINE_PP_MAX,
            )

            anchor_count = min(dynamic_anchor_beats if dynamic_anchor_beats > 0 else initial_beats, len(samples))
            rich_raw_map_pp: list[tuple[float, float]] = []
            for sample in samples:
                centered = sample["features"] - anchor_features
                rich_raw_map_pp.append(
                    (
                        baseline_map + predict(delta_map_coefficients, centered),
                        baseline_pp + predict(delta_pp_coefficients, centered),
                    )
                )
            rich_smoothed_map_pp = _smooth_map_pp(rich_raw_map_pp)
            dynamic_raw_map_pp = [(float(sample["map_raw"]), float(sample["pp_raw"])) for sample in samples]
            dynamic_smoothed_map_pp = _smooth_map_pp(dynamic_raw_map_pp)
            dynamic_raw_anchor_map = float(np.median([value[0] for value in dynamic_raw_map_pp[:anchor_count]]))
            dynamic_raw_anchor_pp = float(np.median([value[1] for value in dynamic_raw_map_pp[:anchor_count]]))
            dynamic_anchor_map = float(np.median([value[0] for value in dynamic_smoothed_map_pp[:anchor_count]]))
            dynamic_anchor_pp = float(np.median([value[1] for value in dynamic_smoothed_map_pp[:anchor_count]]))

            for sample, (rich_map_raw, rich_pp_raw), (rich_map, rich_pp), (map_raw, pp_raw), (map_dynamic, pp_dynamic) in zip(
                samples,
                rich_raw_map_pp,
                rich_smoothed_map_pp,
                dynamic_raw_map_pp,
                dynamic_smoothed_map_pp,
            ):
                map_smoothed_blended = rich_map + method_dynamic_gain_map * (map_dynamic - dynamic_anchor_map)
                pp_smoothed_blended = rich_pp + method_dynamic_gain_pp * (pp_dynamic - dynamic_anchor_pp)
                dbp_smoothed = map_smoothed_blended - pp_smoothed_blended / 3.0
                sbp_smoothed = dbp_smoothed + pp_smoothed_blended
                map_raw_blended = rich_map_raw + method_dynamic_gain_map * (map_raw - dynamic_raw_anchor_map)
                pp_raw_blended = rich_pp_raw + method_dynamic_gain_pp * (pp_raw - dynamic_raw_anchor_pp)
                dbp_raw = map_raw_blended - pp_raw_blended / 3.0
                sbp_raw = dbp_raw + pp_raw_blended
                position = index_positions[sample["index"]]
                overrides[f"{prefix}_MAP_model_raw"][position] = map_raw_blended
                overrides[f"{prefix}_PP_model_raw"][position] = pp_raw_blended
                overrides[f"{prefix}_SBP"][position] = sbp_raw
                overrides[f"{prefix}_DBP"][position] = dbp_raw
                overrides[f"{prefix}_SBP_raw"][position] = sbp_raw
                overrides[f"{prefix}_DBP_raw"][position] = dbp_raw
                overrides[f"{prefix}_MAP_raw"][position] = map_raw_blended
                overrides[f"{prefix}_PP_raw"][position] = pp_raw_blended
                overrides[f"{prefix}_MAP_smoothed"][position] = map_smoothed_blended
                overrides[f"{prefix}_PP_smoothed"][position] = pp_smoothed_blended
                overrides[f"{prefix}_MAP_calibrated"][position] = map_smoothed_blended
                overrides[f"{prefix}_PP_calibrated"][position] = pp_smoothed_blended
                overrides[f"{prefix}_SBP_smoothed"][position] = sbp_smoothed
                overrides[f"{prefix}_DBP_smoothed"][position] = dbp_smoothed
                overrides[f"{prefix}_SBP_calibrated"][position] = sbp_smoothed
                overrides[f"{prefix}_DBP_calibrated"][position] = dbp_smoothed
                overrides[f"{prefix}_postprocess_applied"][position] = 1
    return overrides


def _predict_direct_model(
    method: str,
    labels: tuple[str, ...],
    features: np.ndarray,
    model: dict[str, object],
) -> dict[str, object]:
    map_coefficients = np.array(model["MAP"], dtype=float)
    pp_coefficients = scale_pp_coefficients(
        method,
        np.array(model["PP"], dtype=float),
        model.get("pp_term_scales"),
    )
    map_terms = _linear_terms(map_coefficients, features, labels)
    pp_terms = _linear_terms(pp_coefficients, features, labels)
    return {
        "labels": labels,
        "map_coefficients": map_coefficients,
        "pp_coefficients": pp_coefficients,
        "map_terms": map_terms,
        "pp_terms": pp_terms,
        "map_model_raw": float(map_terms.sum()),
        "pp_model_raw": float(pp_terms.sum()),
    }


def _predict_rtbp(row: pd.Series, coefficients: dict[str, object]) -> dict[str, object] | None:
    features = np.array(
        [
            _to_float(row.get("M1_A_used")),
            _to_float(row.get("M1_HR_used")),
            _to_float(row.get("M1_V2P_relTTP_used")),
            _to_float(row.get("M1_P2V_relTTP_used")),
        ],
        dtype=float,
    )
    if not np.isfinite(features).all():
        return None

    model = coefficients.get("models", {}).get("RTBP")
    if model is None:
        return None
    return _predict_direct_model("RTBP", RTBP_TERM_LABELS, features, model)


def _predict_sinbpd(row: pd.Series, coefficients: dict[str, object]) -> dict[str, object] | None:
    base = _predict_rtbp(row, coefficients)
    if base is None:
        return None

    features = np.array(
        [
            _to_float(row.get("M2_A_used")),
            _to_float(row.get("M2_HR_used")),
            _to_float(row.get("M2_V2P_relTTP_used")),
            _to_float(row.get("M2_P2V_relTTP_used")),
            _to_float(row.get("M2_Stiffness_used")),
            _to_float(row.get("M2_E_used")),
        ],
        dtype=float,
    )
    if not np.isfinite(features).all():
        return None

    model = coefficients.get("models", {}).get("SinBP_D")
    if model is None:
        return None
    direct_model = {
        "MAP": model.get("combined_MAP", model.get("MAP")),
        "PP": model.get("combined_PP", model.get("PP")),
        "pp_term_scales": model.get("pp_term_scales"),
    }
    prediction = _predict_direct_model("SinBP_D", SINBPD_TERM_LABELS, features, direct_model)
    prediction["base_map_model_raw"] = float(base["map_terms"].sum())
    prediction["base_pp_model_raw"] = float(base["pp_terms"].sum())
    return prediction


def _predict_sinbpm(row: pd.Series, coefficients: dict[str, object]) -> dict[str, object] | None:
    features = np.array(
        [
            _to_float(row.get("M3_A_used")),
            _to_float(row.get("M3_HR_used")),
            _to_float(row.get("M3_Mean_used")),
            _to_float(row.get("M3_sinPhi_used")),
            _to_float(row.get("M3_cosPhi_used")),
        ],
        dtype=float,
    )
    if not np.isfinite(features).all():
        return None

    model = coefficients.get("models", {}).get("SinBP_M")
    if model is None:
        return None
    return _predict_direct_model("SinBP_M", SINBPM_TERM_LABELS, features, model)


def _predict_sinbpd_eonly(row: pd.Series, coefficients: dict[str, object]) -> dict[str, object] | None:
    features = np.array(
        [
            _to_float(row.get("SinBP_D_EOnly_A_used")),
            _to_float(row.get("M2_HR_used")),
            _to_float(row.get("M2_V2P_relTTP_used")),
            _to_float(row.get("M2_P2V_relTTP_used")),
            _to_float(row.get("SinBP_D_EOnly_E_used")),
        ],
        dtype=float,
    )
    if not np.isfinite(features).all():
        return None
    model = coefficients.get("models", {}).get("SinBP_D_EOnly")
    if model is None:
        return None
    return _predict_direct_model("SinBP_D_EOnly", SINBPD_EONLY_TERM_LABELS, features, model)


def _predict_sinbpd_e2(row: pd.Series, coefficients: dict[str, object]) -> dict[str, object] | None:
    e_value = _to_float(row.get("SinBP_D_E2_E_used"))
    features = np.array(
        [
            _to_float(row.get("SinBP_D_E2_A_used")),
            _to_float(row.get("M2_HR_used")),
            _to_float(row.get("M2_V2P_relTTP_used")),
            _to_float(row.get("M2_P2V_relTTP_used")),
            e_value,
            e_value * e_value if math.isfinite(e_value) else float("nan"),
        ],
        dtype=float,
    )
    if not np.isfinite(features).all():
        return None
    model = coefficients.get("models", {}).get("SinBP_D_E2")
    if model is None:
        return None
    return _predict_direct_model("SinBP_D_E2", SINBPD_E2_TERM_LABELS, features, model)


def _predict_sinbpd_locala(row: pd.Series, coefficients: dict[str, object]) -> dict[str, object] | None:
    features = np.array(
        [
            _to_float(row.get("SinBP_D_LocalA_A_used")),
            _to_float(row.get("M2_HR_used")),
            _to_float(row.get("M2_V2P_relTTP_used")),
            _to_float(row.get("M2_P2V_relTTP_used")),
            _to_float(row.get("SinBP_D_LocalA_Stiffness_used")),
            _to_float(row.get("SinBP_D_LocalA_E_used")),
        ],
        dtype=float,
    )
    if not np.isfinite(features).all():
        return None
    model = coefficients.get("models", {}).get("SinBP_D_LocalA")
    if model is None:
        return None
    return _predict_direct_model("SinBP_D_LocalA", SINBPD_LOCALA_TERM_LABELS, features, model)


def _predict_method(prefix: str, row: pd.Series, coefficients: dict[str, object]) -> dict[str, object] | None:
    if prefix == "M1":
        return _predict_rtbp(row, coefficients)
    if prefix == "M2":
        return _predict_sinbpd(row, coefficients)
    if prefix == "M3":
        return _predict_sinbpm(row, coefficients)
    if prefix == "SinBP_D_EOnly":
        return _predict_sinbpd_eonly(row, coefficients)
    if prefix == "SinBP_D_E2":
        return _predict_sinbpd_e2(row, coefficients)
    if prefix == "SinBP_D_LocalA":
        return _predict_sinbpd_locala(row, coefficients)
    raise ValueError(f"Unsupported method prefix: {prefix}")


def append_runtime_map_pp_columns(
    df: pd.DataFrame,
    coefficients: dict[str, object] | None = None,
    preserve_existing_core_columns: bool = DEFAULT_PRESERVE_EXISTING_CORE_COLUMNS,
    enable_tracking_blend_overrides: bool = DEFAULT_ENABLE_TRACKING_BLEND_OVERRIDES,
) -> pd.DataFrame:
    runtime_coefficients = coefficients or load_runtime_coefficients()
    enriched = df.copy()
    all_columns: dict[str, list[float | str]] = {}
    method_specs = (
        ("RTBP", "M1", "M1_output_valid", "M1_reject_reason"),
        ("SinBP_D", "M2", "M2_output_valid", "M2_reject_reason"),
        ("SinBP_D_EOnly", "SinBP_D_EOnly", "SinBP_D_EOnly_output_valid", "SinBP_D_EOnly_reject_reason"),
        ("SinBP_D_E2", "SinBP_D_E2", "SinBP_D_E2_output_valid", "SinBP_D_E2_reject_reason"),
        ("SinBP_D_LocalA", "SinBP_D_LocalA", "SinBP_D_LocalA_output_valid", "SinBP_D_LocalA_reject_reason"),
        ("SinBP_M", "M3", "M3_output_valid", "M3_reject_reason"),
    )

    for method_name, prefix, valid_col, reject_col in method_specs:
        labels = {
            "M1": RTBP_TERM_LABELS,
            "M2": SINBPD_TERM_LABELS,
            "SinBP_D_EOnly": SINBPD_EONLY_TERM_LABELS,
            "SinBP_D_E2": SINBPD_E2_TERM_LABELS,
            "SinBP_D_LocalA": SINBPD_LOCALA_TERM_LABELS,
            "M3": SINBPM_TERM_LABELS,
        }[prefix]
        columns: dict[str, list[float | str]] = {
            f"{prefix}_MAP_model_raw": [],
            f"{prefix}_PP_model_raw": [],
            f"{prefix}_SBP": [],
            f"{prefix}_DBP": [],
            f"{prefix}_SBP_raw": [],
            f"{prefix}_DBP_raw": [],
        }
        for label in labels:
            columns[f"{prefix}_MAP_coef_{label}"] = []
            columns[f"{prefix}_PP_coef_{label}"] = []
            columns[f"{prefix}_MAP_term_{label}"] = []
            columns[f"{prefix}_PP_term_{label}"] = []
            columns[f"{prefix}_SBP_term_{label}"] = []
            columns[f"{prefix}_DBP_term_{label}"] = []
        if prefix == "M2":
            columns["M2_SBP_base"] = []
            columns["M2_DBP_base"] = []
            columns["M2_SBP_correction"] = []
            columns["M2_DBP_correction"] = []

        for _, row in enriched.iterrows():
            if not _is_valid_row(row, valid_col, reject_col):
                for key in columns:
                    columns[key].append(np.nan)
                continue

            prediction = _predict_method(prefix, row, runtime_coefficients)
            if prediction is None:
                for key in columns:
                    columns[key].append(np.nan)
                continue

            map_model_raw = float(prediction["map_model_raw"])
            pp_model_raw = float(prediction["pp_model_raw"])
            dbp_model_raw = map_model_raw - pp_model_raw / 3.0
            sbp_model_raw = dbp_model_raw + pp_model_raw
            sbp_clamped, dbp_clamped = _clamp_bp(method_name, sbp_model_raw, dbp_model_raw)

            columns[f"{prefix}_MAP_model_raw"].append(map_model_raw)
            columns[f"{prefix}_PP_model_raw"].append(pp_model_raw)
            columns[f"{prefix}_SBP"].append(sbp_clamped)
            columns[f"{prefix}_DBP"].append(dbp_clamped)
            # For consistency with app/runtime exports, *_raw columns are clamp-applied values.
            columns[f"{prefix}_SBP_raw"].append(sbp_clamped)
            columns[f"{prefix}_DBP_raw"].append(dbp_clamped)

            sbp_terms, dbp_terms = map_pp_to_sbp_dbp_terms(prediction["map_terms"], prediction["pp_terms"])
            sbp_coefficients, dbp_coefficients = map_pp_to_sbp_dbp_terms(
                prediction["map_coefficients"],
                prediction["pp_coefficients"],
            )
            for index, label in enumerate(labels):
                columns[f"{prefix}_MAP_coef_{label}"].append(float(prediction["map_coefficients"][index]))
                columns[f"{prefix}_PP_coef_{label}"].append(float(prediction["pp_coefficients"][index]))
                columns[f"{prefix}_MAP_term_{label}"].append(float(prediction["map_terms"][index]))
                columns[f"{prefix}_PP_term_{label}"].append(float(prediction["pp_terms"][index]))
                columns[f"{prefix}_SBP_term_{label}"].append(float(sbp_terms[index]))
                columns[f"{prefix}_DBP_term_{label}"].append(float(dbp_terms[index]))

            if prefix == "M2":
                base_map = float(prediction.get("base_map_model_raw", 0.0))
                base_pp = float(prediction.get("base_pp_model_raw", 0.0))
                base_dbp = base_map - base_pp / 3.0
                base_sbp = base_dbp + base_pp
                columns["M2_SBP_base"].append(base_sbp)
                columns["M2_DBP_base"].append(base_dbp)
                columns["M2_SBP_correction"].append(sbp_model_raw - base_sbp)
                columns["M2_DBP_correction"].append(dbp_model_raw - base_dbp)
        all_columns.update(columns)
    if enable_tracking_blend_overrides:
        tracking_overrides = _tracking_blend_overrides(enriched, runtime_coefficients)
        if tracking_overrides:
            all_columns.update(tracking_overrides)
    if not all_columns:
        return enriched
    columns_to_apply: dict[str, list[float | str]] = {}
    for column, values in all_columns.items():
        if preserve_existing_core_columns and column in enriched.columns and _is_core_app_column(column):
            continue
        columns_to_apply[column] = values
    if not columns_to_apply:
        return enriched
    overwrite = [column for column in columns_to_apply if column in enriched.columns]
    if overwrite:
        enriched = enriched.drop(columns=overwrite)
    return pd.concat([enriched, pd.DataFrame(columns_to_apply, index=enriched.index)], axis=1)
