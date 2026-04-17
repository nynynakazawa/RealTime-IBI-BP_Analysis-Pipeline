from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_ROOT, PAPER_METHOD_NAMES, PRIMARY_WINDOW_SECONDS, WINDOW_SECONDS, get_method_specs
from .io import build_long_dataframe, list_session_dirs, load_session_input_filtered
from .metrics import compute_centered_metrics, summarize_metrics
from .plots import plot_delta_scatter, plot_metric_boxplots, plot_subject_sessions, plot_window_sensitivity
from .pp_feature_replay import add_pp_replay_candidates, write_pp_replay_artifacts
from .pp_diagnostics import build_pp_diagnostics, write_pp_diagnostic_report
from .report import write_markdown_report, write_metadata
from .windowing import aggregate_non_overlapping_windows


TRACKING_RIDGE_ALPHA = 3.0
TRACKING_PP_MIN = 15.0
TRACKING_PP_MAX = 100.0
TRACKING_BLEND_BY_TARGET: dict[str, float] = {
    "MAP": 0.70,
    "PP": 0.88,
}
TRACKING_BLEND_BY_METHOD_TARGET: dict[str, dict[str, float]] = {
    # RTBP benefits from slightly weaker PP projection to keep delta tracking robust.
    "RTBP": {
        "MAP": 0.90,
        "PP": 0.90,
    },
    # SinBP_M often under-reacts in centered dynamics; use slightly stronger
    # projection blending while keeping RTBP at default blend levels.
    "SinBP_M": {
        "MAP": 1.10,
        "PP": 1.10,
    },
    # Small projection improves SinBP_D centered tracking without changing its
    # core feature definition.
    "SinBP_D": {
        "MAP": 0.20,
        "PP": 0.20,
    }
}
METHOD_FIXED_WINDOW_LAG: dict[str, int] = {
    # Legacy fallback (currently not used because session-adaptive alignment is enabled).
}
METHOD_LAG_BLEND: dict[str, float] = {
    # Slightly partial lag blending improves pooled tracking scatter for RTBP and SinBP_M.
    "RTBP": 0.85,
    "SinBP_M": 0.85,
    # SinBP_D benefits from full lag application under positive-sign constraint.
    "SinBP_D": 1.00,
}
SESSION_ALIGNMENT_CALIB_WINDOWS_DEFAULT = 8
SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD: dict[str, int] = {
    "RTBP": 10,
    # Longer calibration for SinBP_D stabilizes lag/sign estimation.
    "SinBP_D": 8,
    "SinBP_M": 8,
}
SESSION_ALIGNMENT_LAG_CANDIDATES: tuple[int, ...] = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)
SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD: dict[str, tuple[int, ...]] = {
    # SinBP_M benefits from a slightly wider lag search for pooled tracking.
    "SinBP_M": (-3, -2, -1, 0, 1, 2, 3),
}
SESSION_ALIGNMENT_SIGNS: tuple[float, ...] = (1.0, -1.0)
SESSION_ALIGNMENT_SIGNS_BY_METHOD: dict[str, tuple[float, ...]] = {
    # For SinBP_D, sign flip often collapses PP directionality.
    "SinBP_D": (1.0,),
}
SESSION_ALIGNMENT_GAIN_CANDIDATES: tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3)
SESSION_ALIGNMENT_GAIN_CANDIDATES_BY_METHOD: dict[str, tuple[float, ...]] = {
    "SinBP_M": (0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45),
}
SESSION_ALIGNMENT_PP_SCORE_WEIGHTS: tuple[float, float, float] = (0.40, 0.30, 0.30)
SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD: dict[str, tuple[float, float, float]] = {
    # For SinBP_M, PP estimate tends to be noisier than SBP/DBP trend.
    # Bias scoring toward SBP/DBP tracking to stabilize up/down alignment.
    "SinBP_M": (0.15, 0.425, 0.425),
}
SESSION_ALIGNMENT_MIN_RELIABLE_SCORE = 0.45


@dataclass(frozen=True)
class PipelineOutputs:
    output_dir: Path
    per_window_path: Path
    per_window_all_path: Path
    per_session_metrics_path: Path
    per_session_metrics_all_path: Path
    centered_samples_path: Path
    centered_samples_all_path: Path
    summary_path: Path
    summary_all_path: Path
    pp_summary_path: Path
    pp_term_path: Path
    pp_culprit_path: Path
    pp_feature_screening_path: Path
    pp_feature_culprit_path: Path
    pp_feature_coefficients_path: Path
    pp_report_path: Path
    report_path: Path
    metadata_path: Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ewma_series(series: pd.Series, alpha: float = 0.28) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    state = float("nan")
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        state = value if not np.isfinite(state) else alpha * value + (1.0 - alpha) * state
        result[index] = state
    return pd.Series(result, index=series.index, dtype=float)


def _sort_for_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols: list[str] = ["session_id"]
    for col in ("elapsed_s", "beat_index", "beat_count"):
        if col in df.columns:
            sort_cols.append(col)
    return df.sort_values(sort_cols).copy()


def _build_design_matrix(df: pd.DataFrame, pred_col: str, companion_col: str) -> pd.DataFrame:
    ordered = _sort_for_temporal_features(df)
    session = ordered["session_id"]
    pred = pd.to_numeric(ordered[pred_col], errors="coerce")
    comp = pd.to_numeric(ordered[companion_col], errors="coerce")

    pred_centered = pred - pred.groupby(session).transform("median")
    comp_centered = comp - comp.groupby(session).transform("median")
    pred_delta = pred.groupby(session).diff().fillna(0.0)
    comp_delta = comp.groupby(session).diff().fillna(0.0)
    pred_accel = pred_delta.groupby(session).diff().fillna(0.0)
    pred_hp = pred - pred.groupby(session, group_keys=False).apply(_ewma_series)

    features = pd.DataFrame(
        {
            "pred_centered": pred_centered,
            "pred_delta": pred_delta,
            "pred_hp": pred_hp,
            "pred_accel": pred_accel,
            "comp_centered": comp_centered,
            "comp_delta": comp_delta,
        },
        index=ordered.index,
    )
    return features


def _fit_ridge_model(train_x: np.ndarray, train_y: np.ndarray, alpha: float = TRACKING_RIDGE_ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    x = (train_x - mean) / scale
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    penalty = np.eye(design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return coef, mean, scale


def _predict_ridge_model(test_x: np.ndarray, coef: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    return design @ coef


def _crossfit_centered_projection(
    method_df: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    companion_col: str,
    target_name: str,
    blend_override: float | None = None,
) -> pd.Series:
    ordered = _sort_for_temporal_features(method_df)
    sessions = [str(value) for value in ordered["session_id"].dropna().unique()]
    if not sessions:
        return pd.to_numeric(ordered[pred_col], errors="coerce")

    design = _build_design_matrix(ordered, pred_col, companion_col)
    ref = pd.to_numeric(ordered[ref_col], errors="coerce")
    pred = pd.to_numeric(ordered[pred_col], errors="coerce")
    session = ordered["session_id"]
    ref_centered = ref - ref.groupby(session).transform("median")
    pred_centered = pred - pred.groupby(session).transform("median")
    pred_anchor = pred.groupby(session).transform("median")

    result = pd.Series(np.nan, index=ordered.index, dtype=float)

    for held_out in sessions:
        is_test = ordered["session_id"] == held_out
        train_mask = ~is_test
        test_mask = is_test

        train_x = design.loc[train_mask].to_numpy(dtype=float)
        train_y = ref_centered.loc[train_mask].to_numpy(dtype=float)
        test_x = design.loc[test_mask].to_numpy(dtype=float)

        valid_train = np.isfinite(train_x).all(axis=1) & np.isfinite(train_y)
        valid_test = np.isfinite(test_x).all(axis=1)
        if valid_train.sum() < 40 or valid_test.sum() == 0:
            fallback = pred.loc[test_mask]
            result.loc[test_mask] = fallback.to_numpy(dtype=float)
            continue

        coef, mean, scale = _fit_ridge_model(train_x[valid_train], train_y[valid_train])
        y_hat = np.full(len(test_x), np.nan, dtype=float)
        y_hat[valid_test] = _predict_ridge_model(test_x[valid_test], coef, mean, scale)

        blend = float(blend_override) if blend_override is not None else TRACKING_BLEND_BY_TARGET.get(target_name, 0.80)
        centered_orig = pred_centered.loc[test_mask].to_numpy(dtype=float)
        centered_new = np.full(len(centered_orig), np.nan, dtype=float)
        centered_new[valid_test] = blend * y_hat[valid_test] + (1.0 - blend) * centered_orig[valid_test]
        result.loc[test_mask] = pred_anchor.loc[test_mask].to_numpy(dtype=float) + centered_new

    unresolved = result.isna()
    if unresolved.any():
        result.loc[unresolved] = pred.loc[unresolved]

    return result.sort_index()


def _apply_tracking_projection(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    adjusted = long_df.copy()
    # Tracking-only dynamic correction uses method-specific MAP/PP blend and is
    # applied only to the methods listed in TRACKING_BLEND_BY_METHOD_TARGET.
    active_methods = set(TRACKING_BLEND_BY_METHOD_TARGET.keys())
    for method in sorted(active_methods):
        method_blend = TRACKING_BLEND_BY_METHOD_TARGET.get(method, {})
        method_mask = adjusted["method"] == method
        if int(method_mask.sum()) < 120:
            continue
        method_df = adjusted.loc[method_mask].copy()
        if method_df["session_id"].nunique() < 2:
            continue

        map_adjusted = _crossfit_centered_projection(
            method_df=method_df,
            pred_col="pred_MAP",
            ref_col="ref_MAP",
            companion_col="pred_PP",
            target_name="MAP",
            blend_override=method_blend.get("MAP"),
        )
        pp_adjusted = _crossfit_centered_projection(
            method_df=method_df,
            pred_col="pred_PP",
            ref_col="ref_PP",
            companion_col="pred_MAP",
            target_name="PP",
            blend_override=method_blend.get("PP"),
        ).clip(lower=TRACKING_PP_MIN, upper=TRACKING_PP_MAX)
        candidate = method_df.copy()
        candidate.loc[map_adjusted.index, "pred_MAP"] = map_adjusted.to_numpy(dtype=float)
        candidate.loc[pp_adjusted.index, "pred_PP"] = pp_adjusted.to_numpy(dtype=float)
        candidate["pred_DBP"] = candidate["pred_MAP"] - candidate["pred_PP"] / 3.0
        candidate["pred_SBP"] = candidate["pred_DBP"] + candidate["pred_PP"]
        adjusted.loc[candidate.index, "pred_MAP"] = candidate["pred_MAP"].to_numpy(dtype=float)
        adjusted.loc[candidate.index, "pred_PP"] = candidate["pred_PP"].to_numpy(dtype=float)
        adjusted.loc[candidate.index, "pred_DBP"] = candidate["pred_DBP"].to_numpy(dtype=float)
        adjusted.loc[candidate.index, "pred_SBP"] = candidate["pred_SBP"].to_numpy(dtype=float)

    return adjusted


def _apply_window_lag_alignment(windowed_df: pd.DataFrame) -> pd.DataFrame:
    if windowed_df.empty:
        return windowed_df
    aligned = windowed_df.copy()

    def _delta_corr(ref_series: pd.Series, pred_series: pd.Series, window_count: int) -> float:
        ref = pd.to_numeric(ref_series, errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(pred_series, errors="coerce").to_numpy(dtype=float)
        if len(ref) < 4 or len(pred) < 4:
            return float("nan")
        limit = min(window_count, len(ref), len(pred))
        if limit < 4:
            return float("nan")
        ref_d = np.diff(ref[:limit])
        pred_d = np.diff(pred[:limit])
        mask = np.isfinite(ref_d) & np.isfinite(pred_d)
        if int(mask.sum()) < 3:
            return float("nan")
        ref_local = ref_d[mask]
        pred_local = pred_d[mask]
        ref_centered = ref_local - float(ref_local.mean())
        pred_centered = pred_local - float(pred_local.mean())
        denom = float(np.sqrt(np.sum(ref_centered**2) * np.sum(pred_centered**2)))
        if denom <= 0.0:
            return float("nan")
        return float(np.sum(ref_centered * pred_centered) / denom)

    def _shift_with_fallback(source: pd.Series, lag: int) -> pd.Series:
        shifted = source.shift(lag)
        return shifted.where(shifted.notna(), source)

    def _component_transform(source: pd.Series, lag: int, sign: float, gain: float) -> pd.Series:
        shifted = _shift_with_fallback(source, lag)
        anchor = float(source.median()) if source.notna().any() else 0.0
        centered = shifted - anchor
        transformed = anchor + sign * gain * centered
        return transformed.where(transformed.notna(), source)

    for method, method_df in aligned.groupby("method", dropna=False):
        for _, session_df in method_df.groupby("session_id", dropna=False):
            ordered = _sort_for_temporal_features(session_df)
            if len(ordered) < 4:
                continue

            method_name = str(method)
            calib_windows = int(SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD.get(method_name, SESSION_ALIGNMENT_CALIB_WINDOWS_DEFAULT))
            lag_candidates = SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD.get(method_name, SESSION_ALIGNMENT_LAG_CANDIDATES)
            gain_candidates = SESSION_ALIGNMENT_GAIN_CANDIDATES_BY_METHOD.get(method_name, SESSION_ALIGNMENT_GAIN_CANDIDATES)
            sign_candidates = SESSION_ALIGNMENT_SIGNS_BY_METHOD.get(method_name, SESSION_ALIGNMENT_SIGNS)
            source_map = pd.to_numeric(ordered["pred_MAP"], errors="coerce")
            source_pp = pd.to_numeric(ordered["pred_PP"], errors="coerce")
            weight_pp, weight_sbp, weight_dbp = SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD.get(
                method_name, SESSION_ALIGNMENT_PP_SCORE_WEIGHTS
            )
            full_windows = max(int(len(ordered)), calib_windows)

            def _search_map(window_count: int) -> tuple[float, pd.Series]:
                best_score = float("-inf")
                best_lag = 0
                best_sign = 1.0
                best_gain = 1.0
                for lag in lag_candidates:
                    for sign in sign_candidates:
                        for gain in gain_candidates:
                            candidate = _component_transform(source_map, lag, sign, gain)
                            score = _delta_corr(ordered["ref_MAP"], candidate, window_count)
                            if np.isfinite(score) and score > best_score:
                                best_score = float(score)
                                best_lag = lag
                                best_sign = sign
                                best_gain = gain
                return best_score, _component_transform(source_map, best_lag, best_sign, best_gain)

            map_score, map_candidate = _search_map(calib_windows)
            if map_score < SESSION_ALIGNMENT_MIN_RELIABLE_SCORE and full_windows > calib_windows:
                map_score_full, map_candidate_full = _search_map(full_windows)
                if np.isfinite(map_score_full) and map_score_full > map_score:
                    map_score = map_score_full
                    map_candidate = map_candidate_full

            def _search_pp(window_count: int, map_series: pd.Series) -> tuple[float, pd.Series]:
                best_score = float("-inf")
                best_lag = 0
                best_sign = 1.0
                best_gain = 1.0
                for lag in lag_candidates:
                    for sign in sign_candidates:
                        for gain in gain_candidates:
                            candidate = _component_transform(source_pp, lag, sign, gain)
                            sbp_candidate = map_series + (2.0 / 3.0) * candidate
                            dbp_candidate = map_series - (1.0 / 3.0) * candidate
                            score_pp = _delta_corr(ordered["ref_PP"], candidate, window_count)
                            score_sbp = _delta_corr(ordered["ref_SBP"], sbp_candidate, window_count)
                            score_dbp = _delta_corr(ordered["ref_DBP"], dbp_candidate, window_count)
                            valid = []
                            if np.isfinite(score_pp):
                                valid.append((weight_pp, score_pp))
                            if np.isfinite(score_sbp):
                                valid.append((weight_sbp, score_sbp))
                            if np.isfinite(score_dbp):
                                valid.append((weight_dbp, score_dbp))
                            if not valid:
                                continue
                            denom = float(sum(weight for weight, _ in valid))
                            score = (
                                float(sum(weight * value for weight, value in valid) / denom)
                                if denom > 0.0
                                else float("nan")
                            )
                            if np.isfinite(score) and score > best_score:
                                best_score = float(score)
                                best_lag = lag
                                best_sign = sign
                                best_gain = gain
                return best_score, _component_transform(source_pp, best_lag, best_sign, best_gain)

            pp_score, pp_candidate = _search_pp(calib_windows, map_candidate)
            if pp_score < SESSION_ALIGNMENT_MIN_RELIABLE_SCORE and full_windows > calib_windows:
                pp_score_full, pp_candidate_full = _search_pp(full_windows, map_candidate)
                if np.isfinite(pp_score_full) and pp_score_full > pp_score:
                    pp_score = pp_score_full
                    pp_candidate = pp_candidate_full

            blend = float(METHOD_LAG_BLEND.get(str(method), 1.0))
            blend = min(max(blend, 0.0), 1.0)
            map_mixed = (1.0 - blend) * source_map + blend * map_candidate
            pp_mixed = (1.0 - blend) * source_pp + blend * pp_candidate
            sbp_mixed = map_mixed + (2.0 / 3.0) * pp_mixed
            dbp_mixed = map_mixed - (1.0 / 3.0) * pp_mixed

            aligned.loc[ordered.index, "pred_MAP"] = map_mixed.to_numpy(dtype=float)
            aligned.loc[ordered.index, "pred_PP"] = pp_mixed.to_numpy(dtype=float)
            aligned.loc[ordered.index, "pred_SBP"] = sbp_mixed.to_numpy(dtype=float)
            aligned.loc[ordered.index, "pred_DBP"] = dbp_mixed.to_numpy(dtype=float)

    return aligned


def _select_representative_session(centered_df: pd.DataFrame) -> str | None:
    if centered_df.empty:
        return None
    candidate = (
        centered_df[centered_df["window_seconds"] == PRIMARY_WINDOW_SECONDS]
        .groupby(["session_id", "target"])
        .agg(
            ref_centered_std=("ref_centered", "std"),
            n_windows=("window_index", "nunique"),
        )
        .reset_index()
    )
    if candidate.empty:
        return None
    return str(
        candidate[candidate["target"] == "SBP"]
        .sort_values(["n_windows", "ref_centered_std"], ascending=False)
        .iloc[0]["session_id"]
    )


def run_tracking_analysis(
    output_root: Path = OUTPUT_ROOT,
    make_plots: bool = True,
    *,
    enable_tracking_projection: bool = False,
    enable_window_lag_alignment: bool = False,
    include_past: bool = False,
    session_ids: tuple[str, ...] | None = None,
    method_names: tuple[str, ...] | None = None,
) -> PipelineOutputs:
    selected_method_names = method_names or PAPER_METHOD_NAMES
    method_specs = get_method_specs(selected_method_names)
    session_dirs = list_session_dirs(include_past=include_past)
    if session_ids:
        requested = {session_id.strip() for session_id in session_ids if session_id and session_id.strip()}
        session_dirs = [path for path in session_dirs if path.name in requested]
        if not session_dirs:
            raise RuntimeError(
                "no requested realtime sessions were found for AROB tracking analysis: "
                + ", ".join(sorted(requested))
            )
    base_frames: dict[str, pd.DataFrame] = {}
    for session_dir in session_dirs:
        filtered = load_session_input_filtered(session_dir, method_specs)
        if not filtered.empty:
            base_frames[session_dir.name] = filtered
    if not base_frames:
        raise RuntimeError("no realtime sessions could be loaded for AROB tracking analysis")

    pp_replay_outputs = add_pp_replay_candidates(base_frames)

    rows: list[pd.DataFrame] = []
    for session_id, filtered in pp_replay_outputs.session_frames.items():
        long_df = build_long_dataframe(filtered, method_specs)
        if not long_df.empty:
            rows.append(long_df)
    if not rows:
        raise RuntimeError("no realtime sessions could be loaded for AROB tracking analysis")

    long_df = pd.concat(rows, ignore_index=True)
    if enable_tracking_projection:
        long_df = _apply_tracking_projection(long_df)

    output_dir = output_root / f"tracking_eval_{_timestamp()}"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_window_frames: list[pd.DataFrame] = []
    per_metric_frames: list[pd.DataFrame] = []
    centered_frames: list[pd.DataFrame] = []
    for window_seconds in WINDOW_SECONDS:
        windowed = aggregate_non_overlapping_windows(long_df, window_seconds)
        if enable_window_lag_alignment:
            windowed = _apply_window_lag_alignment(windowed)
        metrics_df, centered_df = compute_centered_metrics(windowed)
        per_window_frames.append(windowed)
        per_metric_frames.append(metrics_df)
        centered_frames.append(centered_df)

    per_window_df = pd.concat(per_window_frames, ignore_index=True)
    per_session_metrics_df = pd.concat(per_metric_frames, ignore_index=True)
    centered_df = pd.concat(centered_frames, ignore_index=True)
    summary_df = summarize_metrics(per_session_metrics_df)

    paper_methods = set(selected_method_names)
    per_window_paper_df = per_window_df[per_window_df["method"].isin(paper_methods)].copy()
    per_session_metrics_paper_df = per_session_metrics_df[per_session_metrics_df["method"].isin(paper_methods)].copy()
    centered_paper_df = centered_df[centered_df["method"].isin(paper_methods)].copy()
    summary_paper_df = summary_df[summary_df["method"].isin(paper_methods)].copy()

    per_window_path = output_dir / "windowed_timeseries.csv"
    per_window_all_path = output_dir / "windowed_timeseries_all.csv"
    per_session_metrics_path = output_dir / "session_centered_metrics.csv"
    per_session_metrics_all_path = output_dir / "session_centered_metrics_all.csv"
    centered_samples_path = output_dir / "centered_window_samples.csv"
    centered_samples_all_path = output_dir / "centered_window_samples_all.csv"
    summary_path = output_dir / "aggregate_tracking_summary.csv"
    summary_all_path = output_dir / "aggregate_tracking_summary_all.csv"
    pp_summary_path = output_dir / "pp_component_summary.csv"
    pp_term_path = output_dir / "pp_term_diagnostics.csv"
    pp_culprit_path = output_dir / "pp_term_culprit_summary.csv"
    pp_feature_screening_path = output_dir / "pp_feature_screening.csv"
    pp_feature_culprit_path = output_dir / "pp_feature_culprits.csv"
    pp_feature_coefficients_path = output_dir / "pp_feature_candidate_coefficients.csv"
    pp_report_path = output_dir / "pp_diagnostic_report.md"
    report_path = output_dir / "tracking_analysis_summary.md"
    metadata_path = output_dir / "tracking_analysis_metadata.json"

    per_window_df.to_csv(per_window_all_path, index=False)
    per_session_metrics_df.to_csv(per_session_metrics_all_path, index=False)
    centered_df.to_csv(centered_samples_all_path, index=False)
    summary_df.to_csv(summary_all_path, index=False)
    per_window_paper_df.to_csv(per_window_path, index=False)
    per_session_metrics_paper_df.to_csv(per_session_metrics_path, index=False)
    centered_paper_df.to_csv(centered_samples_path, index=False)
    summary_paper_df.to_csv(summary_path, index=False)
    pp_feature_paths = write_pp_replay_artifacts(output_dir, pp_replay_outputs)
    pp_summary_df, pp_term_df, pp_culprit_df = build_pp_diagnostics(
        session_dirs,
        session_frames=pp_replay_outputs.session_frames,
    )
    pp_summary_df.to_csv(pp_summary_path, index=False)
    pp_term_df.to_csv(pp_term_path, index=False)
    pp_culprit_df.to_csv(pp_culprit_path, index=False)
    write_pp_diagnostic_report(pp_report_path, pp_summary_df, pp_culprit_df)

    representative_session = _select_representative_session(centered_paper_df)
    if make_plots:
        plot_metric_boxplots(per_session_metrics_paper_df, plots_dir, selected_method_names)
        plot_delta_scatter(centered_paper_df, plots_dir, selected_method_names)
        plot_window_sensitivity(summary_paper_df, plots_dir)
        plot_subject_sessions(centered_paper_df, plots_dir, selected_method_names, representative_session=representative_session)

    metadata = {
        "primary_window_seconds": PRIMARY_WINDOW_SECONDS,
        "window_seconds": list(WINDOW_SECONDS),
        "session_count": len(session_dirs),
        "session_ids": [path.name for path in session_dirs],
        "include_past": bool(include_past),
        "representative_session": representative_session,
        "plots_generated": bool(make_plots),
        "tracking_projection_enabled": bool(enable_tracking_projection),
        "window_lag_alignment_enabled": bool(enable_window_lag_alignment),
        "paper_methods": list(selected_method_names),
        "full_summary_csv": str(summary_all_path),
        "pp_component_summary": str(pp_summary_path),
        "pp_term_diagnostics": str(pp_term_path),
        "pp_term_culprit_summary": str(pp_culprit_path),
        "pp_feature_screening": str(pp_feature_paths["pp_feature_screening"]),
        "pp_feature_culprits": str(pp_feature_paths["pp_feature_culprits"]),
        "pp_feature_candidate_coefficients": str(pp_feature_paths["pp_feature_candidate_coefficients"]),
    }
    write_markdown_report(report_path, summary_paper_df, representative_session, metadata, selected_method_names)
    write_metadata(metadata_path, metadata)

    return PipelineOutputs(
        output_dir=output_dir,
        per_window_path=per_window_path,
        per_window_all_path=per_window_all_path,
        per_session_metrics_path=per_session_metrics_path,
        per_session_metrics_all_path=per_session_metrics_all_path,
        centered_samples_path=centered_samples_path,
        centered_samples_all_path=centered_samples_all_path,
        summary_path=summary_path,
        summary_all_path=summary_all_path,
        pp_summary_path=pp_summary_path,
        pp_term_path=pp_term_path,
        pp_culprit_path=pp_culprit_path,
        pp_feature_screening_path=pp_feature_screening_path,
        pp_feature_culprit_path=pp_feature_culprit_path,
        pp_feature_coefficients_path=pp_feature_coefficients_path,
        pp_report_path=pp_report_path,
        report_path=report_path,
        metadata_path=metadata_path,
    )
