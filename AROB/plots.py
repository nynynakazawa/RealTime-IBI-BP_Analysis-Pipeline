from __future__ import annotations

import os
import tempfile
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "arob_mpl_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from current_direction import PAPER_METHOD_NAMES

from .config import PRIMARY_WINDOW_SECONDS


def _method_colors() -> dict[str, str]:
    return {
        "RTBP": "#1f77b4",
        "SinBP_M": "#2ca02c",
        "SinBP_D": "#d62728",
        "SinBP_D_EOnly": "#ff7f0e",
        "SinBP_D_PPShapeA": "#9467bd",
        "SinBP_D_PPShapeB": "#8c564b",
        "SinBP_D_PPShapeC": "#17becf",
    }


def plot_metric_boxplots(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _method_colors()
    metrics = [
        ("centered_mae", "Centered MAE [mmHg]"),
        ("delta_corr", "Delta Corr"),
        ("detrended_corr", "Detrended Corr"),
    ]
    for target in ("SBP", "DBP", "PP"):
        for metric_key, metric_label in metrics:
            plt.figure(figsize=(8.0, 4.8))
            subset = metrics_df[metrics_df["target"] == target].copy()
            windows = sorted(subset["window_seconds"].unique())
            for idx, window_seconds in enumerate(windows):
                window_subset = subset[subset["window_seconds"] == window_seconds]
                methods = list(window_subset["method"].drop_duplicates())
                data = [window_subset[window_subset["method"] == method][metric_key].dropna().to_numpy() for method in methods]
                positions = [idx * (len(methods) + 1) + j for j in range(len(methods))]
                bp = plt.boxplot(data, positions=positions, widths=0.7, patch_artist=True, manage_ticks=False)
                for patch, method in zip(bp["boxes"], methods):
                    patch.set_facecolor(colors.get(method, "#999999"))
                    patch.set_alpha(0.75)
                for median in bp["medians"]:
                    median.set_color("black")
                centers = [(positions[0] + positions[-1]) / 2.0]
                plt.xticks(
                    [(i * (len(methods) + 1) + (len(methods) - 1) / 2.0) for i in range(len(windows))],
                    [f"{w}s" for w in windows],
                )
            plt.title(f"{target} {metric_label}")
            plt.xlabel("Window Length")
            plt.ylabel(metric_label)
            handles = [
                plt.Line2D([0], [0], color=colors.get(method, "#999999"), lw=8, label=method)
                for method in PAPER_METHOD_NAMES
            ]
            plt.legend(handles=handles, loc="best")
            plt.tight_layout()
            plt.savefig(output_dir / f"{target.lower()}_{metric_key}_boxplot.png", dpi=180)
            plt.close()


def plot_window_sensitivity(summary_df: pd.DataFrame, output_dir: Path) -> None:
    # Intentionally disabled for the paper-facing output.
    # Window sensitivity is useful during exploration, but it increases figure count
    # and does not belong in the minimal comparison set requested for the manuscript.
    return None


def plot_subject_sessions(
    centered_df: pd.DataFrame,
    output_dir: Path,
    representative_session: str | None = None,
) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if centered_df.empty:
        return None
    if representative_session is None:
        candidate = (
            centered_df.groupby(["session_id", "target"])
            .agg(ref_centered_std=("ref_centered", "std"))
            .reset_index()
        )
        if candidate.empty:
            return None
        representative_session = (
            candidate[candidate["target"] == "SBP"]
            .sort_values("ref_centered_std", ascending=False)
            .iloc[0]["session_id"]
        )
    colors = _method_colors()
    for session_id in sorted(centered_df["session_id"].dropna().unique()):
        for target in ("SBP", "DBP", "PP"):
            plt.figure(figsize=(9.2, 4.4))
            subset = centered_df[
                (centered_df["session_id"] == session_id)
                & (centered_df["window_seconds"] == PRIMARY_WINDOW_SECONDS)
                & (centered_df["target"] == target)
            ].copy()
            if subset.empty:
                plt.close()
                continue
            ref_trace = subset[subset["method"] == "RTBP"][["window_index", "elapsed_s", "ref_centered"]].drop_duplicates()
            plt.plot(ref_trace["elapsed_s"], ref_trace["ref_centered"], color="black", lw=2.2, label="CNAP")
            for method, group in subset.groupby("method"):
                plt.plot(group["elapsed_s"], group["pred_centered"], marker="o", color=colors.get(method, "#999999"), label=method)
            plt.title(f"{session_id} {target} centered {PRIMARY_WINDOW_SECONDS} s windows")
            plt.xlabel("Elapsed Time [s]")
            plt.ylabel("Centered Pressure [mmHg]")
            plt.grid(alpha=0.25)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(output_dir / f"{session_id}_{target.lower()}_centered_timeseries.png", dpi=180)
            plt.close()

            plt.figure(figsize=(9.2, 4.4))
            delta_ref = ref_trace.sort_values("elapsed_s").copy()
            delta_ref["ref_delta"] = delta_ref["ref_centered"].diff()
            plt.plot(delta_ref["elapsed_s"], delta_ref["ref_delta"], color="black", lw=2.2, label="CNAP-delta")
            for method, group in subset.groupby("method"):
                local = group.sort_values("elapsed_s").copy()
                local["pred_delta"] = local["pred_centered"].diff()
                plt.plot(local["elapsed_s"], local["pred_delta"], marker="o", color=colors.get(method, "#999999"), label=method)
            plt.title(f"{session_id} {target} delta {PRIMARY_WINDOW_SECONDS} s windows")
            plt.xlabel("Elapsed Time [s]")
            plt.ylabel("Delta Pressure [mmHg]")
            plt.grid(alpha=0.25)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(output_dir / f"{session_id}_{target.lower()}_delta_timeseries.png", dpi=180)
            plt.close()
    return str(representative_session)
