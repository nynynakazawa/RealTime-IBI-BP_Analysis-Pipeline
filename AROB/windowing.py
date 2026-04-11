from __future__ import annotations

import pandas as pd

from .config import REF_DBP_COL, REF_MAP_COL, REF_PP_COL, REF_SBP_COL, SESSION_COL, TIME_COL


def aggregate_non_overlapping_windows(long_df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    df = long_df.copy()
    df["window_seconds"] = window_seconds
    df["window_index"] = (df[TIME_COL] // float(window_seconds)).astype(int)
    grouped = (
        df.groupby([SESSION_COL, "method", "method_label", "window_seconds", "window_index"], dropna=False)
        .agg(
            elapsed_s=(TIME_COL, "mean"),
            beat_count=("pred_SBP", "size"),
            ref_SBP=(REF_SBP_COL, "mean"),
            ref_DBP=(REF_DBP_COL, "mean"),
            ref_MAP=(REF_MAP_COL, "mean"),
            ref_PP=(REF_PP_COL, "mean"),
            pred_SBP=("pred_SBP", "mean"),
            pred_DBP=("pred_DBP", "mean"),
            pred_MAP=("pred_MAP", "mean"),
            pred_PP=("pred_PP", "mean"),
        )
        .reset_index()
        .sort_values([SESSION_COL, "method", "window_index"])
        .reset_index(drop=True)
    )
    return grouped
