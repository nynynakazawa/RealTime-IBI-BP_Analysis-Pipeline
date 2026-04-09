from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLUMNS = {
    "RTBP": ("M1_SBP", "M1_DBP"),
    "SinBP_D": ("M2_SBP", "M2_DBP"),
    "SinBP_M": ("M3_SBP", "M3_DBP"),
}

METHOD_META_COLUMNS = {
    "RTBP": ("M1_output_valid", "M1_reject_reason"),
    "SinBP_D": ("M2_output_valid", "M2_reject_reason"),
    "SinBP_M": ("M3_output_valid", "M3_reject_reason"),
}

MAX_ABS_TIME_DELTA_MS = 350.0


def _markdown_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        headers = [str(column) for column in df.columns]
        rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def _corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a.to_numpy(), b.to_numpy())[0, 1])


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "yes", "y", "ok"})


def build_filtered_view(merged_df: pd.DataFrame) -> pd.DataFrame:
    filtered = merged_df.copy()
    base_mask = pd.Series(True, index=filtered.index)

    if "abs_time_delta_ms" in filtered.columns:
        abs_dt = pd.to_numeric(filtered["abs_time_delta_ms"], errors="coerce")
        base_mask &= abs_dt.notna() & (abs_dt <= MAX_ABS_TIME_DELTA_MS)

    if "artifact_flag" in filtered.columns:
        artifact = pd.to_numeric(filtered["artifact_flag"], errors="coerce").fillna(0)
        base_mask &= artifact == 0

    if "is_valid_beat" in filtered.columns:
        valid_beat = _coerce_bool_series(filtered["is_valid_beat"])
        base_mask &= valid_beat

    for method_name, (sbp_col, dbp_col) in METHOD_COLUMNS.items():
        method_mask = base_mask.copy()
        output_valid_col, reject_reason_col = METHOD_META_COLUMNS[method_name]

        if output_valid_col in filtered.columns:
            method_mask &= _coerce_bool_series(filtered[output_valid_col])

        if reject_reason_col in filtered.columns:
            reject_reason = filtered[reject_reason_col].fillna("").astype(str).str.strip().str.lower()
            method_mask &= reject_reason.isin({"", "ok"})

        filtered[f"eval_include_{method_name}"] = method_mask.astype(int)
        filtered.loc[~method_mask, [sbp_col, dbp_col]] = np.nan

    return filtered


def evaluate_merged_session(merged_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = build_filtered_view(merged_df)
    filtered_csv = output_dir / "session_evaluation_input_filtered.csv"
    filtered.to_csv(filtered_csv, index=False)
    valid = filtered.dropna(subset=["ref_SBP", "ref_DBP"]).copy()
    rows: list[dict] = []
    for method_name, (sbp_col, dbp_col) in METHOD_COLUMNS.items():
        for target_label, ref_col, pred_col in (
            ("SBP", "ref_SBP", sbp_col),
            ("DBP", "ref_DBP", dbp_col),
        ):
            subset = valid.dropna(subset=[ref_col, pred_col]).copy()
            if subset.empty:
                rows.append(
                    {
                    "method": method_name,
                    "target": target_label,
                    "filters": f"abs_time_delta_ms<={MAX_ABS_TIME_DELTA_MS}, output_valid, reject_reason=ok, artifact_flag=0",
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "corr": np.nan,
                }
                )
                continue
            error = subset[pred_col] - subset[ref_col]
            rows.append(
                {
                    "method": method_name,
                    "target": target_label,
                    "filters": f"abs_time_delta_ms<={MAX_ABS_TIME_DELTA_MS}, output_valid, reject_reason=ok, artifact_flag=0",
                    "n": int(len(subset)),
                    "mae": float(np.abs(error).mean()),
                    "rmse": float(np.sqrt((error ** 2).mean())),
                    "corr": _corr(subset[ref_col], subset[pred_col]),
                }
            )
    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "session_evaluation_summary.csv"
    summary_json = output_dir / "session_evaluation_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary_csv, summary_json


def generate_session_plots(merged_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = build_filtered_view(merged_df).dropna(subset=["ref_SBP", "ref_DBP"]).copy()
    if valid.empty:
        return []
    if "経過時間_秒" not in valid.columns:
        valid["経過時間_秒"] = np.arange(len(valid), dtype=float)

    outputs: list[Path] = []
    for target_label, ref_col, suffix in (
        ("SBP", "ref_SBP", "sbp"),
        ("DBP", "ref_DBP", "dbp"),
    ):
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
        ax.plot(valid["経過時間_秒"], valid[ref_col], label=f"CNAP {target_label}", linewidth=2.4, color="#111111")
        for method_name, (sbp_col, dbp_col) in METHOD_COLUMNS.items():
            pred_col = sbp_col if target_label == "SBP" else dbp_col
            if pred_col not in valid.columns:
                continue
            ax.plot(valid["経過時間_秒"], valid[pred_col], label=method_name, linewidth=1.4)
        ax.set_xlabel("Elapsed Time (s)")
        ax.set_ylabel(f"{target_label} (mmHg)")
        ax.set_title(f"{target_label} Time Series Comparison")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = output_dir / f"{suffix}_timeseries.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_session_report(
    merged_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    plot_paths: list[Path],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "session_report.md"
    valid = merged_df.dropna(subset=["ref_SBP", "ref_DBP"]).copy()
    lines = [
        "# Session Report",
        "",
        f"- samples_with_reference: {len(valid)}",
        f"- total_samples: {len(merged_df)}",
        f"- evaluation_filters: abs_time_delta_ms<={MAX_ABS_TIME_DELTA_MS}, output_valid=true, reject_reason=ok, artifact_flag=0",
        "",
        "## Metrics",
        "",
        _markdown_table(summary_df),
        "",
    ]
    if plot_paths:
        lines.extend(["## Plots", ""])
        for path in plot_paths:
            lines.append(f"- {path.name}")
        lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
