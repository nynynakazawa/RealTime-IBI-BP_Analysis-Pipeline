from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .config import TRACKING_TARGET_SPECS


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    if float(x.std(ddof=0)) == 0.0 or float(y.std(ddof=0)) == 0.0:
        return math.nan
    return float(x.corr(y))


def _tracking_gain(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    denom = float((ref_centered**2).sum())
    if denom == 0.0:
        return math.nan
    return float((ref_centered * pred_centered).sum() / denom)


def _direction_agreement(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    ref_delta = np.sign(np.diff(ref_centered.to_numpy(dtype=float)))
    pred_delta = np.sign(np.diff(pred_centered.to_numpy(dtype=float)))
    valid = (ref_delta != 0) & (pred_delta != 0)
    if valid.sum() == 0:
        return math.nan
    return float((ref_delta[valid] == pred_delta[valid]).mean())


def _sign_agreement(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    ref_sign = np.sign(ref_centered.to_numpy(dtype=float))
    pred_sign = np.sign(pred_centered.to_numpy(dtype=float))
    valid = (ref_sign != 0) & (pred_sign != 0)
    if valid.sum() == 0:
        return math.nan
    return float((ref_sign[valid] == pred_sign[valid]).mean())


def _amplitude_ratio(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    ref_std = float(ref_centered.std(ddof=0))
    if ref_std == 0.0:
        return math.nan
    return float(pred_centered.std(ddof=0) / ref_std)


def _ewma_detrend(series: pd.Series, alpha: float = 0.35) -> pd.Series:
    values = series.to_numpy(dtype=float)
    trend = np.full(len(values), np.nan, dtype=float)
    state = math.nan
    for index, value in enumerate(values):
        if not math.isfinite(value):
            continue
        state = value if not math.isfinite(state) else alpha * value + (1.0 - alpha) * state
        trend[index] = state
    return pd.Series(values - trend, index=series.index)


def _difference(series: pd.Series) -> pd.Series:
    numeric = series.astype(float)
    return numeric.diff()


def compute_centered_metrics(windowed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    centered_rows: list[pd.DataFrame] = []

    for (session_id, method, method_label, window_seconds), group in windowed_df.groupby(
        ["session_id", "method", "method_label", "window_seconds"], dropna=False
    ):
        group = group.sort_values("window_index").reset_index(drop=True)
        for target, ref_col, pred_col in TRACKING_TARGET_SPECS:
            local = group[
                ["session_id", "method", "method_label", "window_seconds", "window_index", "elapsed_s", "beat_count", ref_col, pred_col]
            ].dropna().copy()
            if local.empty:
                continue
            local = local.rename(columns={ref_col: "ref", pred_col: "pred"})
            local["target"] = target
            ref_median = float(local["ref"].median())
            pred_median = float(local["pred"].median())
            local["ref_centered"] = local["ref"] - ref_median
            local["pred_centered"] = local["pred"] - pred_median
            local["centered_error"] = local["pred_centered"] - local["ref_centered"]
            local["ref_delta"] = _difference(local["ref"])
            local["pred_delta"] = _difference(local["pred"])
            local["delta_error"] = local["pred_delta"] - local["ref_delta"]
            local["ref_detrended"] = _ewma_detrend(local["ref"])
            local["pred_detrended"] = _ewma_detrend(local["pred"])
            local["detrended_error"] = local["pred_detrended"] - local["ref_detrended"]
            centered_rows.append(local)

            cmae = float(local["centered_error"].abs().mean())
            crmse = float(np.sqrt((local["centered_error"] ** 2).mean()))
            corr = _safe_corr(local["pred_centered"], local["ref_centered"])
            gain = _tracking_gain(local["ref_centered"], local["pred_centered"])
            direction_agreement = _direction_agreement(local["ref_centered"], local["pred_centered"])
            sign_agreement = _sign_agreement(local["ref_centered"], local["pred_centered"])
            amplitude_ratio = _amplitude_ratio(local["ref_centered"], local["pred_centered"])
            delta_local = local[["ref_delta", "pred_delta", "delta_error"]].dropna().copy()
            detrended_local = local[["ref_detrended", "pred_detrended", "detrended_error"]].dropna().copy()
            delta_mae = float(delta_local["delta_error"].abs().mean()) if not delta_local.empty else math.nan
            delta_rmse = float(np.sqrt((delta_local["delta_error"] ** 2).mean())) if not delta_local.empty else math.nan
            delta_corr = _safe_corr(delta_local["pred_delta"], delta_local["ref_delta"]) if not delta_local.empty else math.nan
            delta_gain = _tracking_gain(delta_local["ref_delta"], delta_local["pred_delta"]) if not delta_local.empty else math.nan
            detrended_mae = (
                float(detrended_local["detrended_error"].abs().mean()) if not detrended_local.empty else math.nan
            )
            detrended_rmse = (
                float(np.sqrt((detrended_local["detrended_error"] ** 2).mean())) if not detrended_local.empty else math.nan
            )
            detrended_corr = (
                _safe_corr(detrended_local["pred_detrended"], detrended_local["ref_detrended"])
                if not detrended_local.empty
                else math.nan
            )
            detrended_gain = (
                _tracking_gain(detrended_local["ref_detrended"], detrended_local["pred_detrended"])
                if not detrended_local.empty
                else math.nan
            )
            metric_rows.append(
                {
                    "session_id": session_id,
                    "method": method,
                    "method_label": method_label,
                    "window_seconds": int(window_seconds),
                    "target": target,
                    "n_windows": int(len(local)),
                    "centered_mae": cmae,
                    "centered_rmse": crmse,
                    "centered_corr": corr,
                    "tracking_gain": gain,
                    "direction_agreement": direction_agreement,
                    "centered_sign_agreement": sign_agreement,
                    "amplitude_ratio": amplitude_ratio,
                    "delta_mae": delta_mae,
                    "delta_rmse": delta_rmse,
                    "delta_corr": delta_corr,
                    "delta_gain": delta_gain,
                    "detrended_mae": detrended_mae,
                    "detrended_rmse": detrended_rmse,
                    "detrended_corr": detrended_corr,
                    "detrended_gain": detrended_gain,
                    "pp_inversion_like": int(target == "PP" and math.isfinite(gain) and gain < 0.0),
                    "ref_centered_std": float(local["ref_centered"].std(ddof=0)),
                    "pred_centered_std": float(local["pred_centered"].std(ddof=0)),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    centered_df = pd.concat(centered_rows, ignore_index=True) if centered_rows else pd.DataFrame()
    return metrics_df, centered_df


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    summary = (
        metrics_df.groupby(["window_seconds", "method", "method_label", "target"], dropna=False)
        .agg(
            n_sessions=("session_id", "nunique"),
            median_centered_mae=("centered_mae", "median"),
            mean_centered_mae=("centered_mae", "mean"),
            sd_centered_mae=("centered_mae", "std"),
            median_centered_rmse=("centered_rmse", "median"),
            mean_centered_rmse=("centered_rmse", "mean"),
            median_centered_corr=("centered_corr", "median"),
            mean_centered_corr=("centered_corr", "mean"),
            median_tracking_gain=("tracking_gain", "median"),
            mean_tracking_gain=("tracking_gain", "mean"),
            median_direction_agreement=("direction_agreement", "median"),
            mean_direction_agreement=("direction_agreement", "mean"),
            median_centered_sign_agreement=("centered_sign_agreement", "median"),
            mean_centered_sign_agreement=("centered_sign_agreement", "mean"),
            median_amplitude_ratio=("amplitude_ratio", "median"),
            mean_amplitude_ratio=("amplitude_ratio", "mean"),
            median_delta_mae=("delta_mae", "median"),
            mean_delta_mae=("delta_mae", "mean"),
            median_delta_rmse=("delta_rmse", "median"),
            mean_delta_rmse=("delta_rmse", "mean"),
            median_delta_corr=("delta_corr", "median"),
            mean_delta_corr=("delta_corr", "mean"),
            median_delta_gain=("delta_gain", "median"),
            mean_delta_gain=("delta_gain", "mean"),
            median_detrended_mae=("detrended_mae", "median"),
            mean_detrended_mae=("detrended_mae", "mean"),
            median_detrended_rmse=("detrended_rmse", "median"),
            mean_detrended_rmse=("detrended_rmse", "mean"),
            median_detrended_corr=("detrended_corr", "median"),
            mean_detrended_corr=("detrended_corr", "mean"),
            median_detrended_gain=("detrended_gain", "median"),
            mean_detrended_gain=("detrended_gain", "mean"),
            pp_inversion_like_sessions=("pp_inversion_like", "sum"),
        )
        .reset_index()
        .sort_values(["window_seconds", "target", "mean_centered_mae", "method"])
        .reset_index(drop=True)
    )
    return summary
