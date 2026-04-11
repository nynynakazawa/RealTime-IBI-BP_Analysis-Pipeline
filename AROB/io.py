from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    ARTIFACT_COL,
    BEAT_COL,
    METHOD_SPECS,
    REF_MAP_COL,
    REF_PP_COL,
    REALTIME_SESSIONS_ROOT,
    REF_DBP_COL,
    REF_SBP_COL,
    SESSION_COL,
    TIME_COL,
)


def list_session_dirs(root: Path = REALTIME_SESSIONS_ROOT) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def session_input_filtered_path(session_dir: Path) -> Path:
    return session_dir / "evaluation" / "session_evaluation_input_filtered.csv"


def load_session_input_filtered(session_dir: Path) -> pd.DataFrame:
    usecols = {
        SESSION_COL,
        BEAT_COL,
        TIME_COL,
        REF_SBP_COL,
        REF_DBP_COL,
        REF_MAP_COL,
        REF_PP_COL,
        ARTIFACT_COL,
    }
    for spec in METHOD_SPECS:
        usecols.add(spec.sbp_col)
        usecols.add(spec.dbp_col)
        usecols.add(spec.map_col)
        usecols.add(spec.pp_col)
        if spec.output_valid_col:
            usecols.add(spec.output_valid_col)
        if spec.reject_reason_col:
            usecols.add(spec.reject_reason_col)
    path = session_input_filtered_path(session_dir)
    available = pd.read_csv(path, nrows=0).columns.tolist()
    selected = sorted(usecols.intersection(set(available)))
    df = pd.read_csv(path, usecols=selected)
    df = df.sort_values([SESSION_COL, TIME_COL, BEAT_COL]).reset_index(drop=True)
    return df


def build_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    common = df[
        [SESSION_COL, BEAT_COL, TIME_COL, REF_SBP_COL, REF_DBP_COL, REF_MAP_COL, REF_PP_COL, ARTIFACT_COL]
    ].copy()
    for spec in METHOD_SPECS:
        required = {spec.sbp_col, spec.dbp_col, spec.map_col, spec.pp_col}
        if not required.issubset(df.columns):
            continue
        method_df = common.copy()
        method_df["method"] = spec.name
        method_df["method_label"] = spec.label
        method_df["pred_SBP"] = df[spec.sbp_col]
        method_df["pred_DBP"] = df[spec.dbp_col]
        method_df["pred_MAP"] = df[spec.map_col]
        method_df["pred_PP"] = df[spec.pp_col]
        if spec.output_valid_col:
            method_df["output_valid"] = df[spec.output_valid_col]
        else:
            method_df["output_valid"] = 1
        if spec.reject_reason_col:
            method_df["reject_reason"] = (
                df[spec.reject_reason_col]
                .fillna("")
                .astype(str)
                .str.replace("\u3000", " ", regex=False)
                .str.strip()
                .str.lower()
            )
        else:
            method_df["reject_reason"] = ""
        method_df = method_df[
            method_df["pred_SBP"].notna()
            & method_df["pred_DBP"].notna()
            & method_df["pred_MAP"].notna()
            & method_df["pred_PP"].notna()
            & method_df[REF_SBP_COL].notna()
            & method_df[REF_DBP_COL].notna()
            & method_df[REF_MAP_COL].notna()
            & method_df[REF_PP_COL].notna()
        ].copy()
        method_df = method_df[method_df["output_valid"].fillna(0).astype(int) == 1].copy()
        if "reject_reason" in method_df.columns:
            method_df = method_df[
                method_df["reject_reason"].isin(["", "ok"])
            ].copy()
        method_df = method_df[method_df[ARTIFACT_COL].fillna(0).astype(int) == 0].copy()
        rows.append(method_df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
