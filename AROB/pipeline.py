from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import OUTPUT_ROOT, PRIMARY_WINDOW_SECONDS, WINDOW_SECONDS
from .io import build_long_dataframe, list_session_dirs, load_session_input_filtered
from .metrics import compute_centered_metrics, summarize_metrics
from .plots import plot_metric_boxplots, plot_representative_session, plot_window_sensitivity
from .pp_diagnostics import build_pp_diagnostics, write_pp_diagnostic_report
from .report import write_markdown_report, write_metadata
from .windowing import aggregate_non_overlapping_windows


@dataclass(frozen=True)
class PipelineOutputs:
    output_dir: Path
    per_window_path: Path
    per_session_metrics_path: Path
    centered_samples_path: Path
    summary_path: Path
    pp_summary_path: Path
    pp_term_path: Path
    pp_culprit_path: Path
    pp_report_path: Path
    report_path: Path
    metadata_path: Path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def run_tracking_analysis(output_root: Path = OUTPUT_ROOT, make_plots: bool = True) -> PipelineOutputs:
    session_dirs = list_session_dirs()
    rows: list[pd.DataFrame] = []
    for session_dir in session_dirs:
        filtered = load_session_input_filtered(session_dir)
        long_df = build_long_dataframe(filtered)
        if not long_df.empty:
            rows.append(long_df)
    if not rows:
        raise RuntimeError("no realtime sessions could be loaded for AROB tracking analysis")

    long_df = pd.concat(rows, ignore_index=True)

    output_dir = output_root / f"tracking_eval_{_timestamp()}"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_window_frames: list[pd.DataFrame] = []
    per_metric_frames: list[pd.DataFrame] = []
    centered_frames: list[pd.DataFrame] = []
    for window_seconds in WINDOW_SECONDS:
        windowed = aggregate_non_overlapping_windows(long_df, window_seconds)
        metrics_df, centered_df = compute_centered_metrics(windowed)
        per_window_frames.append(windowed)
        per_metric_frames.append(metrics_df)
        centered_frames.append(centered_df)

    per_window_df = pd.concat(per_window_frames, ignore_index=True)
    per_session_metrics_df = pd.concat(per_metric_frames, ignore_index=True)
    centered_df = pd.concat(centered_frames, ignore_index=True)
    summary_df = summarize_metrics(per_session_metrics_df)

    per_window_path = output_dir / "windowed_timeseries.csv"
    per_session_metrics_path = output_dir / "session_centered_metrics.csv"
    centered_samples_path = output_dir / "centered_window_samples.csv"
    summary_path = output_dir / "aggregate_tracking_summary.csv"
    pp_summary_path = output_dir / "pp_component_summary.csv"
    pp_term_path = output_dir / "pp_term_diagnostics.csv"
    pp_culprit_path = output_dir / "pp_term_culprit_summary.csv"
    pp_report_path = output_dir / "pp_diagnostic_report.md"
    report_path = output_dir / "tracking_analysis_summary.md"
    metadata_path = output_dir / "tracking_analysis_metadata.json"

    per_window_df.to_csv(per_window_path, index=False)
    per_session_metrics_df.to_csv(per_session_metrics_path, index=False)
    centered_df.to_csv(centered_samples_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pp_summary_df, pp_term_df, pp_culprit_df = build_pp_diagnostics(session_dirs)
    pp_summary_df.to_csv(pp_summary_path, index=False)
    pp_term_df.to_csv(pp_term_path, index=False)
    pp_culprit_df.to_csv(pp_culprit_path, index=False)
    write_pp_diagnostic_report(pp_report_path, pp_summary_df, pp_culprit_df)

    representative_session = _select_representative_session(centered_df)
    if make_plots:
        plot_metric_boxplots(per_session_metrics_df, plots_dir)
        plot_window_sensitivity(summary_df, plots_dir)
        plot_representative_session(centered_df, plots_dir, representative_session=representative_session)

    metadata = {
        "primary_window_seconds": PRIMARY_WINDOW_SECONDS,
        "window_seconds": list(WINDOW_SECONDS),
        "session_count": len(session_dirs),
        "session_ids": [path.name for path in session_dirs],
        "representative_session": representative_session,
        "plots_generated": bool(make_plots),
        "pp_component_summary": str(pp_summary_path),
        "pp_term_diagnostics": str(pp_term_path),
        "pp_term_culprit_summary": str(pp_culprit_path),
    }
    write_markdown_report(report_path, summary_df, representative_session, metadata)
    write_metadata(metadata_path, metadata)

    return PipelineOutputs(
        output_dir=output_dir,
        per_window_path=per_window_path,
        per_session_metrics_path=per_session_metrics_path,
        centered_samples_path=centered_samples_path,
        summary_path=summary_path,
        pp_summary_path=pp_summary_path,
        pp_term_path=pp_term_path,
        pp_culprit_path=pp_culprit_path,
        pp_report_path=pp_report_path,
        report_path=report_path,
        metadata_path=metadata_path,
    )
