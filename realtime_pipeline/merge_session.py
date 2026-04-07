from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SessionArtifacts:
    smartphone_training_csv: Path
    cnap_beats_csv: Path
    merged_csv: Path


def load_smartphone_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    timestamp_column = "timestamp" if "timestamp" in df.columns else "timestamp_ms" if "timestamp_ms" in df.columns else None
    if timestamp_column is None:
        raise ValueError(f"timestamp column is missing: {path}")
    df["timestamp_ms"] = pd.to_numeric(df[timestamp_column], errors="coerce").astype("float64")
    df = df.dropna(subset=["timestamp_ms"]).copy()
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def load_cnap_beats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    if "epoch_ns" not in df.columns:
        raise ValueError(f"epoch_ns column is missing: {path}")
    df["cnap_timestamp_ms"] = (pd.to_numeric(df["epoch_ns"], errors="coerce") / 1_000_000.0).astype("float64")
    numeric_columns = [
        "Beat Sys [mmHg]",
        "Beat Dia [mmHg]",
        "Beat Mean [mmHg]",
        "Beat HR [bpm]",
        "MAP [mmHg]",
        "経過時間",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["cnap_timestamp_ms"]).copy()
    df = df.sort_values("cnap_timestamp_ms").reset_index(drop=True)
    return df


def merge_session_data(
    smartphone_training_csv: Path,
    cnap_beats_csv: Path,
    merged_csv: Path,
    tolerance_ms: int = 1500,
) -> pd.DataFrame:
    phone_df = load_smartphone_training(smartphone_training_csv)
    cnap_df = load_cnap_beats(cnap_beats_csv)
    merged = pd.merge_asof(
        phone_df,
        cnap_df,
        left_on="timestamp_ms",
        right_on="cnap_timestamp_ms",
        direction="nearest",
        tolerance=tolerance_ms,
    )
    merged["time_delta_ms"] = merged["timestamp_ms"] - merged["cnap_timestamp_ms"]
    merged["abs_time_delta_ms"] = merged["time_delta_ms"].abs()
    if "計測回数" in merged.columns:
        merged["matched_cnap_beat_index"] = pd.to_numeric(merged["計測回数"], errors="coerce")
    merged["ref_SBP"] = merged["Beat Sys [mmHg]"]
    merged["ref_DBP"] = merged["Beat Dia [mmHg]"]
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_csv, index=False)
    return merged
