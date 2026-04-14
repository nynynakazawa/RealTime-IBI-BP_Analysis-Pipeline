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

_LABEL_TO_INDEX = {
    "RTBP": {label: index for index, label in enumerate(RTBP_TERM_LABELS)},
    "SinBP_D": {label: index for index, label in enumerate(SINBPD_TERM_LABELS)},
    "SinBP_M": {label: index for index, label in enumerate(SINBPM_TERM_LABELS)},
    "SinBP_D_EOnly": {label: index for index, label in enumerate(SINBPD_EONLY_TERM_LABELS)},
    "SinBP_D_E2": {label: index for index, label in enumerate(SINBPD_E2_TERM_LABELS)},
    "SinBP_D_LocalA": {label: index for index, label in enumerate(SINBPD_LOCALA_TERM_LABELS)},
}


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


def _is_valid_row(row: pd.Series, valid_col: str, reject_col: str) -> bool:
    return int(_to_float(row.get(valid_col, 0.0))) == 1 and str(row.get(reject_col, "missing")).strip() == "ok"


def _clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


def _clamp_bp(method: str, sbp: float, dbp: float) -> tuple[float, float]:
    constrained_sbp = sbp
    constrained_dbp = dbp
    if method in {"SinBP_D", "SinBP_D_EOnly", "SinBP_D_E2", "SinBP_D_LocalA", "SinBP_M"} and constrained_sbp < constrained_dbp + 20.0:
        constrained_sbp = constrained_dbp + 20.0
    constrained_sbp = _clamp(constrained_sbp, 60.0, 200.0)
    constrained_dbp = _clamp(constrained_dbp, 40.0, 150.0)
    return constrained_sbp, constrained_dbp


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
            columns[f"{prefix}_SBP_raw"].append(sbp_model_raw)
            columns[f"{prefix}_DBP_raw"].append(dbp_model_raw)

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
    if not all_columns:
        return enriched
    existing = [column for column in all_columns if column in enriched.columns]
    if existing:
        enriched = enriched.drop(columns=existing)
    return pd.concat([enriched, pd.DataFrame(all_columns, index=enriched.index)], axis=1)
