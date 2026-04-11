from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import PRIMARY_WINDOW_SECONDS


def write_markdown_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    representative_session: str | None,
    metadata: dict[str, object],
) -> None:
    primary = summary_df[summary_df["window_seconds"] == PRIMARY_WINDOW_SECONDS].copy()
    lines: list[str] = []
    lines.append("# AROB Tracking Analysis Summary")
    lines.append("")
    lines.append(f"- primary_window_seconds: {PRIMARY_WINDOW_SECONDS}")
    lines.append(f"- session_count: {metadata['session_count']}")
    lines.append(f"- representative_session: {representative_session or 'n/a'}")
    lines.append("")
    for target in ("SBP", "DBP", "MAP", "PP"):
        target_df = primary[primary["target"] == target].sort_values("mean_centered_mae")
        lines.append(f"## {target} at {PRIMARY_WINDOW_SECONDS} s")
        lines.append("")
        lines.append("| Method | Mean cMAE | Mean cRMSE | Mean Corr | Mean Gain | Mean Amp Ratio | Mean Direction Agreement | Inversion-Like Sessions |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in target_df.iterrows():
            lines.append(
                "| {label} | {cmae:.3f} | {crmse:.3f} | {corr:.3f} | {gain:.3f} | {amp:.3f} | {da:.3f} | {inv:.0f} |".format(
                    label=row["method_label"],
                    cmae=row["mean_centered_mae"],
                    crmse=row["mean_centered_rmse"],
                    corr=row["mean_centered_corr"],
                    gain=row["mean_tracking_gain"],
                    amp=row["mean_amplitude_ratio"],
                    da=row["mean_direction_agreement"],
                    inv=row["pp_inversion_like_sessions"],
                )
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(output_path: Path, metadata: dict[str, object]) -> None:
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
