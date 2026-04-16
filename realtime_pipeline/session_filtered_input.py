from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluate_session import build_filtered_view


def session_input_filtered_path(session_dir: Path) -> Path:
    return session_dir / "evaluation" / "session_evaluation_input_filtered.csv"


def merged_csv_path(session_dir: Path) -> Path:
    return session_dir / f"{session_dir.name}_merged.csv"


def list_realtime_session_dirs(
    root: Path,
    *,
    include_past: bool = False,
    require_artifacts: bool = False,
) -> list[Path]:
    if not root.exists():
        return []

    current_dirs: list[Path] = []
    past_dirs: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if path.name == "past":
            if include_past:
                past_dirs = [past_path for past_path in sorted(path.iterdir()) if past_path.is_dir()]
            continue
        current_dirs.append(path)

    known_names = {path.name for path in current_dirs}
    candidates = list(current_dirs)
    for path in past_dirs:
        if path.name in known_names:
            continue
        candidates.append(path)
        known_names.add(path.name)

    if not require_artifacts:
        return candidates

    return [
        session_dir
        for session_dir in candidates
        if merged_csv_path(session_dir).exists() or session_input_filtered_path(session_dir).exists()
    ]


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
