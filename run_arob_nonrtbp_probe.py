from __future__ import annotations

import csv
import itertools
import math
import shutil
import tempfile
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from AROB import pipeline


TARGET_METHODS = ("SinBP_D", "SinBP_D_PPShapeC", "SinBP_M")
TARGETS = ("SBP", "DBP")


@dataclass(frozen=True)
class ProbeConfig:
    sin_d_blend: float
    pp_shape_c_blend: float
    sin_m_blend: float
    sin_d_weights: tuple[float, float, float]
    pp_shape_c_weights: tuple[float, float, float]
    sin_d_sign_positive_only: bool
    pp_shape_c_sign_positive_only: bool
    sin_d_lag_mode: str
    pp_shape_c_lag_mode: str
    sin_m_lag_mode: str
    pp_shape_c_projection: float


LAG_MODES: dict[str, tuple[int, ...] | None] = {
    "base": None,
    "wide": (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
    "med": (-4, -3, -2, -1, 0, 1, 2, 3, 4),
    "narrow": (-3, -2, -1, 0, 1, 2, 3),
    "xnarrow": (-2, -1, 0, 1, 2),
}


def _snapshot() -> dict[str, object]:
    return {
        "lag_blend": deepcopy(pipeline.METHOD_LAG_BLEND),
        "calib": deepcopy(pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD),
        "weights": deepcopy(pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD),
        "signs": deepcopy(pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD),
        "lag_modes": deepcopy(pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD),
        "projection": deepcopy(pipeline.TRACKING_BLEND_BY_METHOD_TARGET),
    }


def _restore(snapshot: dict[str, object]) -> None:
    pipeline.METHOD_LAG_BLEND = deepcopy(snapshot["lag_blend"])
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD = deepcopy(snapshot["calib"])
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = deepcopy(snapshot["weights"])
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = deepcopy(snapshot["signs"])
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = deepcopy(snapshot["lag_modes"])
    pipeline.TRACKING_BLEND_BY_METHOD_TARGET = deepcopy(snapshot["projection"])


def _set_method_lag_mode(method: str, mode: str) -> None:
    candidates = LAG_MODES[mode]
    if candidates is None:
        pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD.pop(method, None)
    else:
        pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD[method] = candidates


def _apply(config: ProbeConfig, snapshot: dict[str, object]) -> None:
    lag_blend = deepcopy(snapshot["lag_blend"])
    weights = deepcopy(snapshot["weights"])
    signs = deepcopy(snapshot["signs"])
    lag_modes = deepcopy(snapshot["lag_modes"])
    projection = deepcopy(snapshot["projection"])

    lag_blend["SinBP_D"] = config.sin_d_blend
    lag_blend["SinBP_D_PPShapeC"] = config.pp_shape_c_blend
    lag_blend["SinBP_M"] = config.sin_m_blend

    weights["SinBP_D"] = config.sin_d_weights
    weights["SinBP_D_PPShapeC"] = config.pp_shape_c_weights

    if config.sin_d_sign_positive_only:
        signs["SinBP_D"] = (1.0,)
    else:
        signs.pop("SinBP_D", None)
    if config.pp_shape_c_sign_positive_only:
        signs["SinBP_D_PPShapeC"] = (1.0,)
    else:
        signs.pop("SinBP_D_PPShapeC", None)

    pipeline.METHOD_LAG_BLEND = lag_blend
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = weights
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = signs
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = lag_modes
    _set_method_lag_mode("SinBP_D", config.sin_d_lag_mode)
    _set_method_lag_mode("SinBP_D_PPShapeC", config.pp_shape_c_lag_mode)
    _set_method_lag_mode("SinBP_M", config.sin_m_lag_mode)

    if config.pp_shape_c_projection > 0.0:
        projection["SinBP_D_PPShapeC"] = {"MAP": config.pp_shape_c_projection, "PP": config.pp_shape_c_projection}
    else:
        projection.pop("SinBP_D_PPShapeC", None)
    pipeline.TRACKING_BLEND_BY_METHOD_TARGET = projection


def _pooled_nonrtbp_delta_corr(centered_csv: Path) -> tuple[dict[str, float], float]:
    rows = list(csv.DictReader(centered_csv.open()))
    acc: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
    )
    amp_ratios: list[float] = []
    for row in rows:
        if row.get("window_seconds") != "20":
            continue
        method = row.get("method", "")
        target = row.get("target", "")
        if method not in TARGET_METHODS or target not in TARGETS:
            continue
        pred_delta = row.get("pred_delta", "")
        ref_delta = row.get("ref_delta", "")
        if not pred_delta or not ref_delta:
            continue
        x = float(pred_delta)
        y = float(ref_delta)
        item = acc[(method, target)]
        item["n"] += 1.0
        item["sx"] += x
        item["sy"] += y
        item["sxx"] += x * x
        item["syy"] += y * y
        item["sxy"] += x * y

    method_corr: dict[str, float] = {}
    for method in TARGET_METHODS:
        corr_values: list[float] = []
        for target in TARGETS:
            item = acc[(method, target)]
            n = item["n"]
            if n <= 1:
                continue
            mx = item["sx"] / n
            my = item["sy"] / n
            vx = max(item["sxx"] / n - mx * mx, 0.0)
            vy = max(item["syy"] / n - my * my, 0.0)
            cov = item["sxy"] / n - mx * my
            if vx <= 0.0 or vy <= 0.0:
                continue
            corr = cov / math.sqrt(vx * vy)
            corr_values.append(corr)
            amp_ratios.append(min(max(math.sqrt(vx) / math.sqrt(vy), 0.0), 1.0))
        method_corr[method] = sum(corr_values) / len(corr_values) if corr_values else math.nan

    amp_score = sum(amp_ratios) / len(amp_ratios) if amp_ratios else 0.0
    return method_corr, amp_score


def _objective(method_corr: dict[str, float], amp_score: float) -> float:
    values = [v for v in method_corr.values() if math.isfinite(v)]
    if not values:
        return float("-inf")
    return 0.70 * min(values) + 0.30 * (sum(values) / len(values)) + 0.10 * amp_score


def main() -> int:
    trial_root = Path(tempfile.gettempdir()) / "arob_nonrtbp_probe"
    if trial_root.exists():
        shutil.rmtree(trial_root)
    trial_root.mkdir(parents=True, exist_ok=True)

    snapshot = _snapshot()
    best_config: ProbeConfig | None = None
    best_obj = float("-inf")
    best_corr: dict[str, float] = {}
    best_amp = 0.0
    try:
        configs = []
        lag_mode_sets = [
            ("base", "med", "xnarrow"),
            ("wide", "med", "xnarrow"),
        ]
        for sin_d_blend, pp_shape_c_blend, sin_m_blend in (
            (1.00, 1.00, 0.85),
            (0.95, 1.00, 0.85),
        ):
            for pp_shape_c_weights in ((0.4, 0.3, 0.3), (0.1, 0.45, 0.45)):
                for pp_shape_c_sign_positive_only in (False, True):
                    for sin_d_lag_mode, pp_shape_c_lag_mode, sin_m_lag_mode in lag_mode_sets:
                        for projection in (0.0, 0.15):
                            configs.append(
                                ProbeConfig(
                                    sin_d_blend=sin_d_blend,
                                    pp_shape_c_blend=pp_shape_c_blend,
                                    sin_m_blend=sin_m_blend,
                                    sin_d_weights=(0.4, 0.3, 0.3),
                                    pp_shape_c_weights=pp_shape_c_weights,
                                    sin_d_sign_positive_only=True,
                                    pp_shape_c_sign_positive_only=pp_shape_c_sign_positive_only,
                                    sin_d_lag_mode=sin_d_lag_mode,
                                    pp_shape_c_lag_mode=pp_shape_c_lag_mode,
                                    sin_m_lag_mode=sin_m_lag_mode,
                                    pp_shape_c_projection=projection,
                                )
                            )

        for idx, config in enumerate(configs, start=1):
            _apply(config, snapshot)
            outputs = pipeline.run_tracking_analysis(output_root=trial_root / f"cfg_{idx:04d}", make_plots=False)
            corr, amp = _pooled_nonrtbp_delta_corr(outputs.centered_samples_path)
            obj = _objective(corr, amp)
            print(
                f"[{idx:04d}/{len(configs)}] obj={obj:.4f} "
                f"D={corr.get('SinBP_D', math.nan):.4f} "
                f"C={corr.get('SinBP_D_PPShapeC', math.nan):.4f} "
                f"M={corr.get('SinBP_M', math.nan):.4f} amp={amp:.4f}"
            )
            if obj > best_obj:
                best_obj = obj
                best_config = config
                best_corr = corr
                best_amp = amp

        print("\nBEST CONFIG")
        print(best_config)
        print(f"best_obj={best_obj:.6f}")
        print(f"best_corr={best_corr}")
        print(f"best_amp={best_amp:.6f}")
    finally:
        _restore(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
