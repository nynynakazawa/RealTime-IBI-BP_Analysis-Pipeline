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


def evaluate_merged_session(merged_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = merged_df.dropna(subset=["ref_SBP", "ref_DBP"]).copy()
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
    valid = merged_df.dropna(subset=["ref_SBP", "ref_DBP"]).copy()
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
