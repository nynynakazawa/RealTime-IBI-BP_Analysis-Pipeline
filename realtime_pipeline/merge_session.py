from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(frozen=True)
class SessionArtifacts:
    smartphone_training_csv: Path
    cnap_beats_csv: Path
    merged_csv: Path


def _load_aux_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    return df.reset_index(drop=True)


def _rename_method_columns(df: pd.DataFrame, prefix: str, keep_prefixes: tuple[str, ...]) -> pd.DataFrame:
    renamed = df.copy()
    rename_map: dict[str, str] = {}
    for column in renamed.columns:
        if column == "経過時間_秒":
            continue
        if column.startswith(keep_prefixes):
            continue
        rename_map[column] = f"{prefix}_{column}"
    return renamed.rename(columns=rename_map)


def _restrict_overlay_columns(df: pd.DataFrame, allowed_prefixes: tuple[str, ...] | None) -> pd.DataFrame:
    if allowed_prefixes is None:
        return df
    keep_columns = [
        column
        for column in df.columns
        if column == "経過時間_秒" or any(column.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if not keep_columns:
        return pd.DataFrame()
    return df.loc[:, keep_columns].copy()


def _overlay_by_row_index(base_df: pd.DataFrame, overlay_df: pd.DataFrame) -> pd.DataFrame:
    if overlay_df.empty:
        return base_df
    merged = base_df.copy()
    limit = min(len(merged), len(overlay_df))
    if limit <= 0:
        return merged
    for column in overlay_df.columns:
        if column == "経過時間_秒":
            continue
        values = overlay_df.loc[: limit - 1, column]
        if column not in merged.columns:
            merged[column] = pd.Series([pd.NA] * len(merged), dtype=values.dtype if hasattr(values, "dtype") else "object")
        elif is_numeric_dtype(values):
            if not is_numeric_dtype(merged[column]) or str(merged[column].dtype).startswith("int"):
                merged[column] = pd.to_numeric(merged[column], errors="coerce").astype("float64")
        elif merged[column].dtype != object:
            merged[column] = merged[column].astype("object")
        merged.loc[: limit - 1, column] = values.to_numpy()
    return merged


def _patch_from_method_csvs(training_csv: Path, phone_df: pd.DataFrame) -> pd.DataFrame:
    smartphone_dir = training_csv.parent
    session_id = training_csv.stem.replace("_Training_Data", "")
    overlays = (
        (smartphone_dir / f"{session_id}_RTBP.csv", "M1", (), None),
        # Training_Data.csv already carries the main M2 columns. Only overlay the comparison
        # variants from the dedicated SinBP_D CSV so malformed legacy M2 exports do not clobber
        # the primary SinBP_D series for older sessions.
        (smartphone_dir / f"{session_id}_SinBP_D.csv", "M2", ("M2_", "SinBP_D_"), ("SinBP_D_",)),
        (smartphone_dir / f"{session_id}_SinBP_M.csv", "M3", ("M3_",), None),
    )
    patched = phone_df.copy()
    for csv_path, prefix, keep_prefixes, allowed_prefixes in overlays:
        if not csv_path.exists():
            continue
        method_df = _load_aux_csv(csv_path)
        method_df = _restrict_overlay_columns(method_df, allowed_prefixes)
        if method_df.empty:
            continue
        method_df = _rename_method_columns(method_df, prefix, keep_prefixes)
        patched = _overlay_by_row_index(patched, method_df)
    return patched


def load_smartphone_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(column).strip() for column in df.columns]
    df = _patch_from_method_csvs(path, df)
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
    from realtime_pipeline.evaluate_session import ensure_postprocessed_columns

    merged = ensure_postprocessed_columns(merged)
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_csv, index=False)
    return merged
