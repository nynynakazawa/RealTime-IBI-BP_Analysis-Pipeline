from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluate_session import build_filtered_view


def session_input_filtered_path(session_dir: Path) -> Path:
    return session_dir / "evaluation" / "session_evaluation_input_filtered.csv"


def merged_csv_path(session_dir: Path) -> Path:
    return session_dir / f"{session_dir.name}_merged.csv"


def ensure_session_input_filtered(session_dir: Path, force_rebuild: bool = True) -> Path:
    filtered_path = session_input_filtered_path(session_dir)
    if not force_rebuild and filtered_path.exists() and filtered_path.stat().st_size > 0:
        return filtered_path

    merged_path = merged_csv_path(session_dir)
    if not merged_path.exists():
        raise FileNotFoundError(f"merged csv was not found: {merged_path}")

    merged_df = pd.read_csv(merged_path)
    filtered_df = build_filtered_view(merged_df)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(filtered_path, index=False)
    return filtered_path
