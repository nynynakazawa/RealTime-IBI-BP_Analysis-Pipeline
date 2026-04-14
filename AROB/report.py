from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from current_direction import PAPER_CORE_METHOD_NAMES, PAPER_SUPPLEMENTAL_METHOD_NAMES

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
    lines.append(f"- paper_core_methods: {', '.join(PAPER_CORE_METHOD_NAMES)}")
    lines.append(f"- paper_supplemental_methods: {', '.join(PAPER_SUPPLEMENTAL_METHOD_NAMES)}")
    lines.append("")
    for target in ("SBP", "DBP", "PP"):
        target_df = primary[primary["target"] == target].sort_values("mean_centered_mae")
        lines.append(f"## {target} at {PRIMARY_WINDOW_SECONDS} s")
        lines.append("")
        lines.append("| Method | Mean cMAE | Mean dCorr | Mean hpCorr |")
        lines.append("| --- | ---: | ---: | ---: |")
        for _, row in target_df.iterrows():
            lines.append(
                "| {label} | {cmae:.3f} | {dcorr:.3f} | {hpcorr:.3f} |".format(
                    label=row["method_label"],
                    cmae=row["mean_centered_mae"],
                    dcorr=row["mean_delta_corr"],
                    hpcorr=row["mean_detrended_corr"],
                )
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(output_path: Path, metadata: dict[str, object]) -> None:
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
