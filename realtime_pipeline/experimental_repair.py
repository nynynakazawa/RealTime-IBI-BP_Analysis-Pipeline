from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from BP_Analysis.fit_realtime_map_pp_coefficients import (
    DEFAULT_BASELINE_RIDGE_ALPHA,
    DEFAULT_BASELINE_SHRINKAGE,
    DEFAULT_DYNAMIC_BLEND_GAIN,
    DEFAULT_RICH_BASELINE_RIDGE_ALPHA,
    DEFAULT_RIDGE_ALPHA,
    EVALUATION_FILTERS,
    RTBP,
    SINBP_D,
    SINBP_D_E2,
    SINBP_D_EONLY,
    SINBP_D_LOCALA,
    SINBP_M,
    build_samples,
    load_reference_rows,
    replay_adaptive_model,
    replay_baseline_dynamic_blend,
    train_adaptive_models,
    train_models,
    train_rich_baseline_models,
    train_shared_sinbpd_baseline_models,
)


SERIES_LABELS = {
    "INITIAL_BASELINE": "smartphone_initial_baseline",
    "RICH_BASELINE": "smartphone_rich_baseline",
    "SHARED_D_BASELINE": "smartphone_shared_sinbpd_baseline",
    "RICH_DYNAMIC": "smartphone_rich_dynamic_blend",
}


@dataclass(frozen=True)
class ExperimentalRepairArtifacts:
    predictions_csv: Path
    summary_csv: Path
    summary_json: Path


def _prediction_groups_for_all_sessions(
    sessions_root: Path,
) -> list[tuple[str, list[dict[str, object]]]]:
    rows = load_reference_rows(sessions_root)
    if not rows:
        raise ValueError(f"no usable reference rows under {sessions_root}")

    samples_by_method = {
        "RTBP": build_samples(rows, RTBP),
        "SinBP_D": build_samples(rows, SINBP_D),
        "SinBP_D_EOnly": build_samples(rows, SINBP_D_EONLY),
        "SinBP_D_E2": build_samples(rows, SINBP_D_E2),
        "SinBP_D_LocalA": build_samples(rows, SINBP_D_LOCALA),
        "SinBP_M": build_samples(rows, SINBP_M),
    }
    all_sessions = {str(row["_session"]) for row in rows}
    if len(all_sessions) < 2:
        raise ValueError("need at least two sessions to repair experimental outputs")

    models = train_models(samples_by_method, all_sessions, DEFAULT_RIDGE_ALPHA)
    adaptive_models = train_adaptive_models(
        samples_by_method,
        all_sessions,
        baseline_ridge_alpha=DEFAULT_BASELINE_RIDGE_ALPHA,
        delta_ridge_alpha=DEFAULT_RIDGE_ALPHA,
        baseline_shrinkage=DEFAULT_BASELINE_SHRINKAGE,
    )
    rich_baseline_models = train_rich_baseline_models(
        samples_by_method,
        all_sessions,
        baseline_ridge_alpha=DEFAULT_RICH_BASELINE_RIDGE_ALPHA,
        delta_ridge_alpha=DEFAULT_RIDGE_ALPHA,
        baseline_shrinkage=DEFAULT_BASELINE_SHRINKAGE,
    )
    shared_baseline_models = train_shared_sinbpd_baseline_models(
        samples_by_method,
        all_sessions,
        baseline_ridge_alpha=DEFAULT_RICH_BASELINE_RIDGE_ALPHA,
        delta_ridge_alpha=DEFAULT_RIDGE_ALPHA,
        baseline_shrinkage=DEFAULT_BASELINE_SHRINKAGE,
    )

    return [
        (
            "INITIAL_BASELINE",
            replay_adaptive_model(
                adaptive_models,
                samples_by_method,
                all_sessions,
                series=SERIES_LABELS["INITIAL_BASELINE"],
            ),
        ),
        (
            "RICH_BASELINE",
            replay_adaptive_model(
                rich_baseline_models,
                samples_by_method,
                all_sessions,
                series=SERIES_LABELS["RICH_BASELINE"],
                rich_summary=True,
            ),
        ),
        (
            "SHARED_D_BASELINE",
            replay_adaptive_model(
                shared_baseline_models,
                samples_by_method,
                all_sessions,
                series=SERIES_LABELS["SHARED_D_BASELINE"],
                rich_summary=True,
            ),
        ),
        (
            "RICH_DYNAMIC",
            replay_baseline_dynamic_blend(
                rich_baseline_models,
                models,
                samples_by_method,
                all_sessions,
                series=SERIES_LABELS["RICH_DYNAMIC"],
                dynamic_gain_map=DEFAULT_DYNAMIC_BLEND_GAIN,
                dynamic_gain_pp=DEFAULT_DYNAMIC_BLEND_GAIN,
                rich_summary=True,
            ),
        ),
    ]


def _corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    local = np.column_stack((lhs, rhs)) if len(lhs) and len(rhs) else np.empty((0, 2))
    if local.size == 0:
        return float("nan")
    local = local[np.isfinite(local).all(axis=1)]
    if len(local) < 2:
        return float("nan")
    left = local[:, 0]
    right = local[:, 1]
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denom = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denom)


def _raw_sbp_dbp(row: dict[str, object]) -> tuple[float, float]:
    pred_map_raw = float(row["pred_MAP_raw"])
    pred_pp_raw = float(row["pred_PP_raw"])
    pred_dbp_raw = pred_map_raw - pred_pp_raw / 3.0
    pred_sbp_raw = pred_dbp_raw + pred_pp_raw
    return pred_sbp_raw, pred_dbp_raw


def _summary_rows_for_session(session: str, prediction_groups: list[tuple[str, list[dict[str, object]]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series_suffix, predictions in prediction_groups:
        for base_method in ("RTBP", "SinBP_D", "SinBP_M"):
            subset = [
                row
                for row in predictions
                if str(row["session"]) == session and str(row["method"]) == base_method
            ]
            method_name = f"{base_method}_{series_suffix}"
            for series_kind in ("raw", "smoothed", "calibrated"):
                if subset:
                    if series_kind == "raw":
                        pred_sbp = np.array([_raw_sbp_dbp(row)[0] for row in subset], dtype=float)
                        pred_dbp = np.array([_raw_sbp_dbp(row)[1] for row in subset], dtype=float)
                    else:
                        pred_sbp = np.array([float(row["pred_SBP"]) for row in subset], dtype=float)
                        pred_dbp = np.array([float(row["pred_DBP"]) for row in subset], dtype=float)
                    ref_sbp = np.array([float(row["ref_SBP"]) for row in subset], dtype=float)
                    ref_dbp = np.array([float(row["ref_DBP"]) for row in subset], dtype=float)
                    pred_pp = pred_sbp - pred_dbp
                    ref_pp = ref_sbp - ref_dbp
                    pp_error = pred_pp - ref_pp
                else:
                    pred_sbp = np.array([], dtype=float)
                    pred_dbp = np.array([], dtype=float)
                    ref_sbp = np.array([], dtype=float)
                    ref_dbp = np.array([], dtype=float)
                    pp_error = np.array([], dtype=float)

                for target, pred, ref in (
                    ("SBP", pred_sbp, ref_sbp),
                    ("DBP", pred_dbp, ref_dbp),
                ):
                    if len(pred) == 0:
                        rows.append(
                            {
                                "method": method_name,
                                "series": series_kind,
                                "target": target,
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
                    error = pred - ref
                    rows.append(
                        {
                            "method": method_name,
                            "series": series_kind,
                            "target": target,
                            "filters": EVALUATION_FILTERS,
                            "n": int(len(pred)),
                            "mae": float(np.mean(np.abs(error))),
                            "rmse": float(np.sqrt(np.mean(error**2))),
                            "corr": _corr(ref, pred),
                            "signed_bias": float(np.mean(error)),
                            "pp_mae": float(np.mean(np.abs(pp_error))),
                            "pp_signed_bias": float(np.mean(pp_error)),
                        }
                    )
    return rows


def _prediction_rows_for_session(session: str, prediction_groups: list[tuple[str, list[dict[str, object]]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series_suffix, predictions in prediction_groups:
        for row in predictions:
            if str(row["session"]) != session:
                continue
            raw_sbp, raw_dbp = _raw_sbp_dbp(row)
            rows.append(
                {
                    "session": session,
                    "row_index": int(row["row_index"]),
                    "method": f"{row['method']}_{series_suffix}",
                    "source_series": str(row["series"]),
                    "raw_SBP": raw_sbp,
                    "raw_DBP": raw_dbp,
                    "smoothed_SBP": float(row["pred_SBP"]),
                    "smoothed_DBP": float(row["pred_DBP"]),
                    "calibrated_SBP": float(row["pred_SBP"]),
                    "calibrated_DBP": float(row["pred_DBP"]),
                    "pred_MAP_raw": float(row["pred_MAP_raw"]),
                    "pred_PP_raw": float(row["pred_PP_raw"]),
                    "pred_MAP": float(row["pred_MAP"]),
                    "pred_PP": float(row["pred_PP"]),
                    "baseline_MAP_raw": row.get("baseline_MAP_raw"),
                    "baseline_PP_raw": row.get("baseline_PP_raw"),
                    "baseline_MAP": row.get("baseline_MAP"),
                    "baseline_PP": row.get("baseline_PP"),
                    "initial_baseline_beats": row.get("initial_baseline_beats"),
                    "baseline_shrinkage": row.get("baseline_shrinkage"),
                    "baseline_map_fallback_applied": row.get("baseline_map_fallback_applied", 0),
                    "baseline_pp_fallback_applied": row.get("baseline_pp_fallback_applied", 0),
                    "ref_SBP": float(row["ref_SBP"]),
                    "ref_DBP": float(row["ref_DBP"]),
                    "ref_MAP": float(row["ref_MAP"]),
                    "ref_PP": float(row["ref_PP"]),
                    "elapsed_s": row.get("elapsed_s"),
                }
            )
    return sorted(rows, key=lambda item: (item["row_index"], item["method"]))


def _backup_if_exists(path: Path) -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.stem}_app_export_backup{path.suffix}")
    if backup.exists():
        return
    shutil.copy2(path, backup)


def repair_session_experimental_outputs(
    sessions_root: Path,
    target_sessions: set[str] | None = None,
) -> list[ExperimentalRepairArtifacts]:
    prediction_groups = _prediction_groups_for_all_sessions(sessions_root)
    all_sessions = sorted({str(row["session"]) for _, rows in prediction_groups for row in rows})
    if target_sessions is not None:
        sessions = [session for session in all_sessions if session in target_sessions]
    else:
        sessions = all_sessions

    artifacts: list[ExperimentalRepairArtifacts] = []
    for session in sessions:
        session_root = sessions_root / session
        evaluation_dir = session_root / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = _summary_rows_for_session(session, prediction_groups)
        predictions_rows = _prediction_rows_for_session(session, prediction_groups)

        summary_csv = evaluation_dir / "session_evaluation_summary_experimental.csv"
        summary_json = evaluation_dir / "session_evaluation_summary_experimental.json"
        predictions_csv = evaluation_dir / "session_evaluation_predictions_experimental_repaired.csv"
        meta_json = evaluation_dir / "session_evaluation_summary_experimental_meta.json"

        _backup_if_exists(summary_csv)
        _backup_if_exists(summary_json)

        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        summary_json.write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(predictions_rows).to_csv(predictions_csv, index=False)
        meta_json.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "source": "offline_repair_from_current_fit_realtime_map_pp_coefficients_logic",
                    "sessions_root": str(sessions_root),
                    "target_session": session,
                    "series": list(SERIES_LABELS.keys()),
                    "note": "Experimental session summary was regenerated offline from current Analysis coefficient replay logic and may differ from old app-exported experimental columns kept in merged.csv.",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            ExperimentalRepairArtifacts(
                predictions_csv=predictions_csv,
                summary_csv=summary_csv,
                summary_json=summary_json,
            )
        )
    return artifacts
