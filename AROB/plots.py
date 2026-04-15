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
import numpy as np
import pandas as pd

from current_direction import PAPER_METHOD_NAMES

from .config import PRIMARY_WINDOW_SECONDS


def _method_colors() -> dict[str, str]:
    return {
        "RTBP": "#1f77b4",
        "SinBP_M": "#2ca02c",
        "SinBP_D": "#d62728",
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
            subset = metrics_df[(metrics_df["target"] == target) & (metrics_df["method"].isin(PAPER_METHOD_NAMES))].copy()
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


def plot_delta_scatter(centered_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if centered_df.empty:
        return None
    colors = _method_colors()
    methods = [method for method in PAPER_METHOD_NAMES if method in set(centered_df["method"].unique())]
    if not methods:
        return None
    for target in ("SBP", "DBP", "PP"):
        target_df = centered_df[
            (centered_df["window_seconds"] == PRIMARY_WINDOW_SECONDS)
            & (centered_df["target"] == target)
            & (centered_df["method"].isin(methods))
        ][["method", "ref_delta", "pred_delta"]].dropna()
        if target_df.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
        axes_flat = axes.flatten()
        for index, method in enumerate(methods):
            axis = axes_flat[index]
            local = target_df[target_df["method"] == method]
            if local.empty:
                axis.set_visible(False)
                continue
            axis.scatter(
                local["ref_delta"].to_numpy(dtype=float),
                local["pred_delta"].to_numpy(dtype=float),
                s=20.0,
                alpha=0.45,
                color=colors.get(method, "#999999"),
                edgecolor="none",
            )
            both = np.concatenate(
                [
                    local["ref_delta"].to_numpy(dtype=float),
                    local["pred_delta"].to_numpy(dtype=float),
                ]
            )
            finite = both[np.isfinite(both)]
            if finite.size:
                limit = float(np.max(np.abs(finite)))
                limit = max(limit, 1.0)
                axis.plot([-limit, limit], [-limit, limit], color="black", linestyle="--", linewidth=1.0)
                axis.set_xlim(-limit, limit)
                axis.set_ylim(-limit, limit)
            axis.set_title(method)
            axis.set_xlabel("CNAP Δ [mmHg]")
            axis.set_ylabel("Est Δ [mmHg]")
            axis.grid(alpha=0.2)
        for index in range(len(methods), len(axes_flat)):
            axes_flat[index].set_visible(False)
        fig.suptitle(f"{target} Up/Down Tracking Scatter ({PRIMARY_WINDOW_SECONDS}s)")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(output_dir / f"{target.lower()}_delta_scatter.png", dpi=180)
        plt.close(fig)


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
            subset = centered_df[
                (centered_df["session_id"] == session_id)
                & (centered_df["window_seconds"] == PRIMARY_WINDOW_SECONDS)
                & (centered_df["target"] == target)
                & (centered_df["method"].isin(PAPER_METHOD_NAMES))
            ].copy()
            if subset.empty:
                continue
            ref_trace = subset[subset["method"] == "RTBP"][["window_index", "elapsed_s", "ref_centered"]].drop_duplicates()
            if ref_trace.empty:
                continue
            # Tracking-only view: keep delta timeseries and hide centered-level plot.
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
