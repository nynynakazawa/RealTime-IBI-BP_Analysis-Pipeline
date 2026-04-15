from __future__ import annotations

import csv
import math
import shutil
import tempfile
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from AROB import pipeline


@dataclass(frozen=True)
class FineConfig:
    projection: float
    weights: tuple[float, float, float]
    lag_blend: float = 1.0
    calib_windows: int = 8
    sign_positive_only: bool = True
    lag_mode: str = "med"


def _snapshot() -> dict[str, object]:
    return {
        "lag_blend": deepcopy(pipeline.METHOD_LAG_BLEND),
        "calib": deepcopy(pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD),
        "weights": deepcopy(pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD),
        "signs": deepcopy(pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD),
        "lags": deepcopy(pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD),
        "proj": deepcopy(pipeline.TRACKING_BLEND_BY_METHOD_TARGET),
    }


def _restore(snapshot: dict[str, object]) -> None:
    pipeline.METHOD_LAG_BLEND = deepcopy(snapshot["lag_blend"])
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD = deepcopy(snapshot["calib"])
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = deepcopy(snapshot["weights"])
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = deepcopy(snapshot["signs"])
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = deepcopy(snapshot["lags"])
    pipeline.TRACKING_BLEND_BY_METHOD_TARGET = deepcopy(snapshot["proj"])


def _apply(config: FineConfig, snapshot: dict[str, object]) -> None:
    pipeline.METHOD_LAG_BLEND = deepcopy(snapshot["lag_blend"])
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD = deepcopy(snapshot["calib"])
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = deepcopy(snapshot["weights"])
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = deepcopy(snapshot["signs"])
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = deepcopy(snapshot["lags"])
    pipeline.TRACKING_BLEND_BY_METHOD_TARGET = deepcopy(snapshot["proj"])

    pipeline.METHOD_LAG_BLEND["SinBP_D_PPShapeC"] = config.lag_blend
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD["SinBP_D_PPShapeC"] = config.calib_windows
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD["SinBP_D_PPShapeC"] = config.weights
    if config.sign_positive_only:
        pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD["SinBP_D_PPShapeC"] = (1.0,)
    else:
        pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD.pop("SinBP_D_PPShapeC", None)
    if config.lag_mode == "med":
        pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD["SinBP_D_PPShapeC"] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
    elif config.lag_mode == "base":
        pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD.pop("SinBP_D_PPShapeC", None)
    else:
        pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD["SinBP_D_PPShapeC"] = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6)

    pipeline.TRACKING_BLEND_BY_METHOD_TARGET["SinBP_D_PPShapeC"] = {"MAP": config.projection, "PP": config.projection}


def _read_corr(centered_csv: Path) -> dict[tuple[str, str], float]:
    rows = list(csv.DictReader(centered_csv.open()))
    acc: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
    )
    for row in rows:
        if row.get("window_seconds") != "20":
            continue
        method = row.get("method", "")
        target = row.get("target", "")
        if method not in ("RTBP", "SinBP_D_PPShapeC") or target not in ("SBP", "DBP", "PP"):
            continue
        if not row.get("pred_delta") or not row.get("ref_delta"):
            continue
        x = float(row["pred_delta"])
        y = float(row["ref_delta"])
        local = acc[(method, target)]
        local["n"] += 1.0
        local["sx"] += x
        local["sy"] += y
        local["sxx"] += x * x
        local["syy"] += y * y
        local["sxy"] += x * y

    out: dict[tuple[str, str], float] = {}
    for key, local in acc.items():
        n = local["n"]
        if n <= 1:
            out[key] = math.nan
            continue
        mx = local["sx"] / n
        my = local["sy"] / n
        vx = max(local["sxx"] / n - mx * mx, 0.0)
        vy = max(local["syy"] / n - my * my, 0.0)
        if vx <= 0.0 or vy <= 0.0:
            out[key] = math.nan
            continue
        cov = local["sxy"] / n - mx * my
        out[key] = cov / math.sqrt(vx * vy)
    return out


def main() -> int:
    snapshot = _snapshot()
    trials_root = Path(tempfile.gettempdir()) / "arob_ppshapec_fineprobe"
    if trials_root.exists():
        shutil.rmtree(trials_root)
    trials_root.mkdir(parents=True, exist_ok=True)

    configs: list[FineConfig] = []
    for projection in (0.215, 0.220, 0.225, 0.230, 0.235, 0.240):
        for weights in (
            (0.4, 0.3, 0.3),
            (0.25, 0.375, 0.375),
            (0.2, 0.2, 0.6),
            (0.1, 0.2, 0.7),
            (0.05, 0.15, 0.8),
            (0.0, 0.1, 0.9),
        ):
            configs.append(FineConfig(projection=projection, weights=weights))

    best = None
    best_score = float("-inf")
    try:
        for idx, config in enumerate(configs, start=1):
            _apply(config, snapshot)
            outputs = pipeline.run_tracking_analysis(output_root=trials_root / f"cfg_{idx:03d}", make_plots=False)
            corr = _read_corr(outputs.centered_samples_path)
            r_sbp = corr[("RTBP", "SBP")]
            r_dbp = corr[("RTBP", "DBP")]
            r_pp = corr[("RTBP", "PP")]
            c_sbp = corr[("SinBP_D_PPShapeC", "SBP")]
            c_dbp = corr[("SinBP_D_PPShapeC", "DBP")]
            c_pp = corr[("SinBP_D_PPShapeC", "PP")]
            margins = [c_sbp - r_sbp, c_dbp - r_dbp, c_pp - r_pp]
            # DBP margin prioritization
            score = 0.55 * (c_dbp - r_dbp) + 0.25 * min(margins) + 0.20 * (sum(margins) / len(margins))
            print(
                f"[{idx:03d}/{len(configs)}] score={score:.6f} "
                f"PPShapeC=({c_sbp:.4f},{c_dbp:.4f},{c_pp:.4f}) "
                f"RTBP=({r_sbp:.4f},{r_dbp:.4f},{r_pp:.4f}) "
                f"proj={config.projection:.3f} weights={config.weights}"
            )
            if score > best_score:
                best_score = score
                best = (config, c_sbp, c_dbp, c_pp, r_sbp, r_dbp, r_pp, score)
    finally:
        _restore(snapshot)

    print("\nBEST")
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

