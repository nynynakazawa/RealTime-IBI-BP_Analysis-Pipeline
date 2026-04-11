from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import METHOD_SPECS
from .io import session_input_filtered_path


def _safe_corr(lhs: pd.Series, rhs: pd.Series) -> float:
    if len(lhs) < 2 or len(rhs) < 2:
        return math.nan
    if float(lhs.std(ddof=0)) == 0.0 or float(rhs.std(ddof=0)) == 0.0:
        return math.nan
    return float(lhs.corr(rhs))


def _tracking_gain(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    denom = float((ref_centered**2).sum())
    if denom == 0.0:
        return math.nan
    return float((ref_centered * pred_centered).sum() / denom)


def _direction_agreement(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    ref_delta = np.sign(np.diff(ref_centered.to_numpy(dtype=float)))
    pred_delta = np.sign(np.diff(pred_centered.to_numpy(dtype=float)))
    valid = (ref_delta != 0) & (pred_delta != 0)
    if valid.sum() == 0:
        return math.nan
    return float((ref_delta[valid] == pred_delta[valid]).mean())


def _amplitude_ratio(ref_centered: pd.Series, pred_centered: pd.Series) -> float:
    ref_std = float(ref_centered.std(ddof=0))
    if ref_std == 0.0:
        return math.nan
    return float(pred_centered.std(ddof=0) / ref_std)


def _centered(series: pd.Series) -> pd.Series:
    return series - float(series.median())


def _term_pairs(columns: list[str], prefix: str) -> dict[str, tuple[str, str]]:
    sbp_terms = {
        column[len(f"{prefix}_SBP_term_") :]: column
        for column in columns
        if column.startswith(f"{prefix}_SBP_term_")
    }
    dbp_terms = {
        column[len(f"{prefix}_DBP_term_") :]: column
        for column in columns
        if column.startswith(f"{prefix}_DBP_term_")
    }
    return {
        term: (sbp_terms[term], dbp_terms[term])
        for term in sorted(set(sbp_terms).intersection(dbp_terms))
    }


def _pp_summary_rows(df: pd.DataFrame, session_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in METHOD_SPECS:
        required = {spec.map_col, spec.pp_col, "ref_MAP", "ref_PP"}
        if not required.issubset(df.columns):
            continue
        local = df[[spec.map_col, spec.pp_col, "ref_MAP", "ref_PP"]].dropna().copy()
        if local.empty:
            continue
        for component, pred_col, ref_col in (
            ("MAP", spec.map_col, "ref_MAP"),
            ("PP", spec.pp_col, "ref_PP"),
        ):
            ref = local[ref_col].astype(float)
            pred = local[pred_col].astype(float)
            error = pred - ref
            ref_centered = _centered(ref)
            pred_centered = _centered(pred)
            corr = _safe_corr(pred, ref)
            centered_corr = _safe_corr(pred_centered, ref_centered)
            gain = _tracking_gain(ref_centered, pred_centered)
            amplitude_ratio = _amplitude_ratio(ref_centered, pred_centered)
            rows.append(
                {
                    "session_id": session_id,
                    "method": spec.name,
                    "component": component,
                    "n": int(len(local)),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt((error**2).mean())),
                    "signed_bias": float(error.mean()),
                    "corr": corr,
                    "centered_corr": centered_corr,
                    "tracking_gain": gain,
                    "amplitude_ratio": amplitude_ratio,
                    "direction_agreement": _direction_agreement(ref_centered, pred_centered),
                    "inversion_like": int(component == "PP" and math.isfinite(gain) and gain < 0.0),
                }
            )
    return rows


def _pp_term_rows(df: pd.DataFrame, session_id: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in METHOD_SPECS:
        pairs = _term_pairs(list(df.columns), spec.prefix)
        if not pairs or "ref_PP" not in df.columns or spec.pp_col not in df.columns:
            continue
        for term, (sbp_col, dbp_col) in pairs.items():
            local = df[[sbp_col, dbp_col, spec.pp_col, "ref_PP"]].dropna().copy()
            if len(local) < 3:
                continue
            term_pp = local[sbp_col].astype(float) - local[dbp_col].astype(float)
            ref_pp = local["ref_PP"].astype(float)
            pred_pp = local[spec.pp_col].astype(float)
            term_centered = _centered(term_pp)
            ref_centered = _centered(ref_pp)
            pred_centered = _centered(pred_pp)
            gain_to_ref = _tracking_gain(ref_centered, term_centered)
            amplitude_ratio = _amplitude_ratio(ref_centered, term_centered)
            inverse_strength = 0.0
            if math.isfinite(gain_to_ref) and gain_to_ref < 0.0 and math.isfinite(amplitude_ratio):
                inverse_strength = abs(gain_to_ref) * amplitude_ratio
            rows.append(
                {
                    "session_id": session_id,
                    "method": spec.name,
                    "term": term,
                    "term_kind": "baseline" if term == "intercept" else "dynamic",
                    "n": int(len(local)),
                    "pp_term_median": float(term_pp.median()),
                    "pp_term_mean": float(term_pp.mean()),
                    "pp_term_std": float(term_pp.std(ddof=0)),
                    "corr_to_ref_pp": _safe_corr(term_pp, ref_pp),
                    "centered_corr_to_ref_pp": _safe_corr(term_centered, ref_centered),
                    "centered_corr_to_pred_pp": _safe_corr(term_centered, pred_centered),
                    "gain_to_ref_pp": gain_to_ref,
                    "amplitude_ratio_to_ref_pp": amplitude_ratio,
                    "direction_agreement_to_ref_pp": _direction_agreement(ref_centered, term_centered),
                    "inverse_strength": inverse_strength,
                }
            )
    return rows


def build_pp_diagnostics(session_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    for session_dir in session_dirs:
        path = session_input_filtered_path(session_dir)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        summary_rows.extend(_pp_summary_rows(df, session_dir.name))
        term_rows.extend(_pp_term_rows(df, session_dir.name))

    summary_df = pd.DataFrame(summary_rows)
    term_df = pd.DataFrame(term_rows)
    if term_df.empty:
        return summary_df, term_df, pd.DataFrame()

    culprit_rows: list[dict[str, object]] = []
    for (session_id, method), group in term_df.groupby(["session_id", "method"], dropna=False):
        dynamic = group[group["term_kind"] == "dynamic"].copy()
        if dynamic.empty:
            continue
        dynamic = dynamic.sort_values(["inverse_strength", "gain_to_ref_pp"], ascending=[False, True])
        top = dynamic.iloc[0]
        culprit_rows.append(
            {
                "session_id": session_id,
                "method": method,
                "top_inverse_term": top["term"],
                "top_inverse_strength": float(top["inverse_strength"]),
                "top_gain_to_ref_pp": float(top["gain_to_ref_pp"]),
                "top_centered_corr_to_ref_pp": float(top["centered_corr_to_ref_pp"]),
                "top_amplitude_ratio_to_ref_pp": float(top["amplitude_ratio_to_ref_pp"]),
            }
        )

    culprit_df = pd.DataFrame(culprit_rows)
    if culprit_df.empty:
        return summary_df, term_df, culprit_df

    culprit_counts = (
        culprit_df.groupby(["method", "top_inverse_term"], dropna=False)
        .agg(
            culprit_sessions=("session_id", "nunique"),
            mean_inverse_strength=("top_inverse_strength", "mean"),
            mean_gain_to_ref_pp=("top_gain_to_ref_pp", "mean"),
            mean_centered_corr_to_ref_pp=("top_centered_corr_to_ref_pp", "mean"),
            mean_amplitude_ratio_to_ref_pp=("top_amplitude_ratio_to_ref_pp", "mean"),
        )
        .reset_index()
        .sort_values(["method", "culprit_sessions", "mean_inverse_strength"], ascending=[True, False, False])
        .reset_index(drop=True)
        .rename(columns={"top_inverse_term": "term"})
    )
    return summary_df, term_df, culprit_counts


def write_pp_diagnostic_report(
    output_path: Path,
    summary_df: pd.DataFrame,
    culprit_df: pd.DataFrame,
) -> None:
    def render(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return df.to_csv(index=False)

    lines = [
        "# PP Diagnostic Report",
        "",
        "## Session-Level MAP/PP Summary",
        "",
    ]
    if summary_df.empty:
        lines.append("No PP diagnostics were generated.")
    else:
        lines.append(render(summary_df))
    lines.extend(["", "## Most Inversion-Like Terms", ""])
    if culprit_df.empty:
        lines.append("No term-level culprit rows were generated.")
    else:
        lines.append(render(culprit_df))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
