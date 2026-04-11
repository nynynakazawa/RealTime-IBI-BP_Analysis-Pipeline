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

from .config import PRIMARY_WINDOW_SECONDS


def _method_colors() -> dict[str, str]:
    return {
        "RTBP": "#1f77b4",
        "SinBP_M": "#2ca02c",
        "SinBP_D": "#d62728",
        "SinBP_D_EOnly": "#ff7f0e",
    }


def plot_metric_boxplots(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _method_colors()
    metrics = [
        ("centered_mae", "Centered MAE [mmHg]"),
        ("centered_corr", "Centered Corr"),
        ("tracking_gain", "Tracking Gain"),
        ("direction_agreement", "Direction Agreement"),
        ("amplitude_ratio", "Amplitude Ratio"),
    ]
    for target in sorted(metrics_df["target"].dropna().unique()):
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
                for method in ["RTBP", "SinBP_M", "SinBP_D", "SinBP_D_EOnly"]
            ]
            plt.legend(handles=handles, loc="best")
            plt.tight_layout()
            plt.savefig(output_dir / f"{target.lower()}_{metric_key}_boxplot.png", dpi=180)
            plt.close()


def plot_window_sensitivity(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _method_colors()
    for target in sorted(summary_df["target"].dropna().unique()):
        subset = summary_df[summary_df["target"] == target].copy()
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))
        metrics = [
            ("mean_centered_mae", "Mean Centered MAE"),
            ("mean_centered_corr", "Mean Centered Corr"),
            ("mean_tracking_gain", "Mean Tracking Gain"),
            ("mean_amplitude_ratio", "Mean Amplitude Ratio"),
        ]
        for ax, (metric_key, title) in zip(axes, metrics):
            for method, group in subset.groupby("method"):
                ax.plot(group["window_seconds"], group[metric_key], marker="o", color=colors.get(method, "#999999"), label=method)
            ax.set_title(title)
            ax.set_xlabel("Window [s]")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(target)
        axes[-1].legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"{target.lower()}_window_sensitivity.png", dpi=180)
        plt.close(fig)


def plot_representative_session(
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
    for target in ("SBP", "DBP", "PP", "MAP"):
        plt.figure(figsize=(9.2, 4.4))
        subset = centered_df[
            (centered_df["session_id"] == representative_session)
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
        plt.title(
            f"{representative_session} {target} centered {PRIMARY_WINDOW_SECONDS} s windows"
        )
        plt.xlabel("Elapsed Time [s]")
        plt.ylabel("Centered Pressure [mmHg]")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / f"{representative_session}_{target.lower()}_centered_timeseries.png", dpi=180)
        plt.close()
    return str(representative_session)
