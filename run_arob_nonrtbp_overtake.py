from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from AROB import pipeline
from current_direction import AROB_TRACKING_ROOT


TARGETS = ("SBP", "DBP", "PP")
METHODS = ("RTBP", "SinBP_D", "SinBP_D_PPShapeC", "SinBP_M")
NON_RTBP = ("SinBP_D", "SinBP_D_PPShapeC", "SinBP_M")

LAG_MODE_TO_CANDIDATES: dict[str, tuple[int, ...] | None] = {
    "base": None,
    "wide": (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
    "med": (-4, -3, -2, -1, 0, 1, 2, 3, 4),
    "narrow": (-3, -2, -1, 0, 1, 2, 3),
    "xnarrow": (-2, -1, 0, 1, 2),
}


@dataclass
class MethodTune:
    lag_blend: float
    calib_windows: int
    pp_weights: tuple[float, float, float]
    sign_positive_only: bool
    lag_mode: str
    projection_blend: float


@dataclass
class Config:
    method_tunes: dict[str, MethodTune]


@dataclass
class Snapshot:
    lag_blend: dict[str, float]
    calib_windows: dict[str, int]
    pp_weights: dict[str, tuple[float, float, float]]
    signs: dict[str, tuple[float, ...]]
    lag_candidates: dict[str, tuple[int, ...]]
    projection: dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune non-RTBP methods to overtake RTBP on pooled delta-scatter "
            "correlation (SBP/DBP/PP, window=20)."
        )
    )
    parser.add_argument("--rounds", type=int, default=2, help="Coordinate descent rounds.")
    parser.add_argument("--with-plots", action="store_true", help="Generate plots for final run.")
    parser.add_argument(
        "--trials-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "arob_nonrtbp_overtake_trials",
        help="Temporary output root for tuning trials.",
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=AROB_TRACKING_ROOT,
        help="Output root for final selected run.",
    )
    return parser.parse_args()


def _snapshot() -> Snapshot:
    return Snapshot(
        lag_blend=deepcopy(pipeline.METHOD_LAG_BLEND),
        calib_windows=deepcopy(pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD),
        pp_weights=deepcopy(pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD),
        signs=deepcopy(pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD),
        lag_candidates=deepcopy(pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD),
        projection=deepcopy(pipeline.TRACKING_BLEND_BY_METHOD_TARGET),
    )


def _restore(snapshot: Snapshot) -> None:
    pipeline.METHOD_LAG_BLEND = deepcopy(snapshot.lag_blend)
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD = deepcopy(snapshot.calib_windows)
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = deepcopy(snapshot.pp_weights)
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = deepcopy(snapshot.signs)
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = deepcopy(snapshot.lag_candidates)
    pipeline.TRACKING_BLEND_BY_METHOD_TARGET = deepcopy(snapshot.projection)


def _infer_lag_mode(method: str, snapshot: Snapshot) -> str:
    values = snapshot.lag_candidates.get(method)
    for mode, cand in LAG_MODE_TO_CANDIDATES.items():
        if cand is not None and tuple(values or ()) == tuple(cand):
            return mode
    return "base"


def _infer_sign_positive(method: str, snapshot: Snapshot) -> bool:
    signs = snapshot.signs.get(method)
    return tuple(signs or ()) == (1.0,)


def _infer_projection(method: str, snapshot: Snapshot) -> float:
    local = snapshot.projection.get(method)
    if not local:
        return 0.0
    return float(local.get("MAP", 0.0))


def _base_config(snapshot: Snapshot) -> Config:
    tunes: dict[str, MethodTune] = {}
    for method in NON_RTBP:
        tunes[method] = MethodTune(
            lag_blend=float(snapshot.lag_blend.get(method, 1.0)),
            calib_windows=int(snapshot.calib_windows.get(method, pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_DEFAULT)),
            pp_weights=tuple(snapshot.pp_weights.get(method, pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS)),
            sign_positive_only=_infer_sign_positive(method, snapshot),
            lag_mode=_infer_lag_mode(method, snapshot),
            projection_blend=_infer_projection(method, snapshot),
        )
    return Config(method_tunes=tunes)


def _apply(config: Config, snapshot: Snapshot) -> None:
    lag_blend = deepcopy(snapshot.lag_blend)
    calib_windows = deepcopy(snapshot.calib_windows)
    pp_weights = deepcopy(snapshot.pp_weights)
    signs = deepcopy(snapshot.signs)
    lag_candidates = deepcopy(snapshot.lag_candidates)
    projection = deepcopy(snapshot.projection)

    for method, tune in config.method_tunes.items():
        lag_blend[method] = tune.lag_blend
        calib_windows[method] = tune.calib_windows
        pp_weights[method] = tune.pp_weights

        if tune.sign_positive_only:
            signs[method] = (1.0,)
        else:
            signs.pop(method, None)

        mode_candidates = LAG_MODE_TO_CANDIDATES.get(tune.lag_mode)
        if mode_candidates is None:
            lag_candidates.pop(method, None)
        else:
            lag_candidates[method] = mode_candidates

        if tune.projection_blend > 0.0:
            projection[method] = {"MAP": tune.projection_blend, "PP": tune.projection_blend}
        else:
            projection.pop(method, None)

    pipeline.METHOD_LAG_BLEND = lag_blend
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD = calib_windows
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = pp_weights
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = signs
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = lag_candidates
    pipeline.TRACKING_BLEND_BY_METHOD_TARGET = projection


def _read_pooled_corr(centered_csv: Path) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(centered_csv.open()))
    acc: dict[tuple[str, str], dict[str, float]] = {}
    for method in METHODS:
        for target in TARGETS:
            acc[(method, target)] = {"n": 0.0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}

    for row in rows:
        if row.get("window_seconds") != "20":
            continue
        method = row.get("method", "")
        target = row.get("target", "")
        if method not in METHODS or target not in TARGETS:
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

    out: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    for method in METHODS:
        for target in TARGETS:
            local = acc[(method, target)]
            n = local["n"]
            if n <= 1:
                out[method][target] = math.nan
                continue
            mx = local["sx"] / n
            my = local["sy"] / n
            vx = max(local["sxx"] / n - mx * mx, 0.0)
            vy = max(local["syy"] / n - my * my, 0.0)
            if vx <= 0.0 or vy <= 0.0:
                out[method][target] = math.nan
                continue
            cov = local["sxy"] / n - mx * my
            out[method][target] = cov / math.sqrt(vx * vy)
    return out


def _score_method(corr: dict[str, dict[str, float]], method: str) -> float:
    margins: list[float] = []
    values: list[float] = []
    for target in TARGETS:
        candidate = corr[method][target]
        baseline = corr["RTBP"][target]
        if math.isfinite(candidate) and math.isfinite(baseline):
            margins.append(candidate - baseline)
            values.append(candidate)
    if not margins:
        return float("-inf")
    min_margin = min(margins)
    mean_margin = sum(margins) / len(margins)
    mean_corr = sum(values) / len(values)
    return 0.70 * min_margin + 0.20 * mean_margin + 0.10 * mean_corr


def _score_global(corr: dict[str, dict[str, float]]) -> float:
    method_scores = [_score_method(corr, method) for method in NON_RTBP]
    finite_scores = [value for value in method_scores if math.isfinite(value)]
    if not finite_scores:
        return float("-inf")
    return min(finite_scores) + 0.5 * (sum(finite_scores) / len(finite_scores))


def _candidate_tunes(method: str, current: MethodTune) -> list[MethodTune]:
    candidates: list[MethodTune] = [current]
    common_blends = [0.85, 0.95, 1.0, 1.05]
    weight_candidates = [
        current.pp_weights,
        (0.4, 0.3, 0.3),
        (0.25, 0.375, 0.375),
        (0.1, 0.45, 0.45),
    ]

    for blend in common_blends:
        candidates.append(
            MethodTune(
                lag_blend=float(blend),
                calib_windows=current.calib_windows,
                pp_weights=current.pp_weights,
                sign_positive_only=current.sign_positive_only,
                lag_mode=current.lag_mode,
                projection_blend=current.projection_blend,
            )
        )
    for weights in weight_candidates:
        candidates.append(
            MethodTune(
                lag_blend=current.lag_blend,
                calib_windows=current.calib_windows,
                pp_weights=tuple(weights),
                sign_positive_only=current.sign_positive_only,
                lag_mode=current.lag_mode,
                projection_blend=current.projection_blend,
            )
        )
    for lag_mode in ("base", "wide", "med", "narrow", "xnarrow"):
        if method == "SinBP_D_PPShapeC" and lag_mode not in ("base", "med", "wide"):
            continue
        if method == "SinBP_M" and lag_mode not in ("base", "med", "narrow", "xnarrow"):
            continue
        if method == "SinBP_D" and lag_mode not in ("base", "wide", "med"):
            continue
        candidates.append(
            MethodTune(
                lag_blend=current.lag_blend,
                calib_windows=current.calib_windows,
                pp_weights=current.pp_weights,
                sign_positive_only=current.sign_positive_only,
                lag_mode=lag_mode,
                projection_blend=current.projection_blend,
            )
        )
    for sign_positive_only in (False, True):
        if method == "SinBP_D" and not sign_positive_only:
            continue
        candidates.append(
            MethodTune(
                lag_blend=current.lag_blend,
                calib_windows=current.calib_windows,
                pp_weights=current.pp_weights,
                sign_positive_only=sign_positive_only,
                lag_mode=current.lag_mode,
                projection_blend=current.projection_blend,
            )
        )
    for projection in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3):
        if method != "SinBP_D_PPShapeC" and projection > 0.0:
            continue
        candidates.append(
            MethodTune(
                lag_blend=current.lag_blend,
                calib_windows=current.calib_windows,
                pp_weights=current.pp_weights,
                sign_positive_only=current.sign_positive_only,
                lag_mode=current.lag_mode,
                projection_blend=float(projection),
            )
        )
    for calib in (8, 10, 12):
        if method != "SinBP_D_PPShapeC" and calib == 12:
            continue
        candidates.append(
            MethodTune(
                lag_blend=current.lag_blend,
                calib_windows=calib,
                pp_weights=current.pp_weights,
                sign_positive_only=current.sign_positive_only,
                lag_mode=current.lag_mode,
                projection_blend=current.projection_blend,
            )
        )
    if method == "SinBP_D_PPShapeC":
        for weights in (
            (0.2, 0.2, 0.6),
            (0.1, 0.2, 0.7),
            (0.05, 0.15, 0.80),
            (0.0, 0.10, 0.90),
        ):
            candidates.append(
                MethodTune(
                    lag_blend=current.lag_blend,
                    calib_windows=current.calib_windows,
                    pp_weights=weights,
                    sign_positive_only=current.sign_positive_only,
                    lag_mode=current.lag_mode,
                    projection_blend=current.projection_blend,
                )
            )
        for projection in (0.18, 0.20, 0.22, 0.24):
            candidates.append(
                MethodTune(
                    lag_blend=current.lag_blend,
                    calib_windows=current.calib_windows,
                    pp_weights=current.pp_weights,
                    sign_positive_only=current.sign_positive_only,
                    lag_mode=current.lag_mode,
                    projection_blend=projection,
                )
            )

    unique: list[MethodTune] = []
    seen: set[tuple[float, int, tuple[float, float, float], bool, str, float]] = set()
    for candidate in candidates:
        key = (
            candidate.lag_blend,
            candidate.calib_windows,
            tuple(candidate.pp_weights),
            candidate.sign_positive_only,
            candidate.lag_mode,
            candidate.projection_blend,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _run_once(config: Config, snapshot: Snapshot, output_root: Path, make_plots: bool) -> tuple[Path, dict[str, dict[str, float]]]:
    _apply(config, snapshot)
    outputs = pipeline.run_tracking_analysis(output_root=output_root, make_plots=make_plots)
    corr = _read_pooled_corr(outputs.centered_samples_path)
    return outputs.output_dir, corr


def main() -> int:
    args = parse_args()
    trials_root = args.trials_root
    if trials_root.exists():
        shutil.rmtree(trials_root)
    trials_root.mkdir(parents=True, exist_ok=True)

    snapshot = _snapshot()
    config = _base_config(snapshot)

    baseline_dir, baseline_corr = _run_once(
        config=config,
        snapshot=snapshot,
        output_root=trials_root / "baseline",
        make_plots=False,
    )
    print("baseline_dir", baseline_dir)
    print("baseline_rtbp", baseline_corr["RTBP"])
    print("baseline_nonrtbp", {method: baseline_corr[method] for method in NON_RTBP})

    current_corr = baseline_corr
    current_score = _score_global(current_corr)

    try:
        for round_idx in range(1, max(int(args.rounds), 1) + 1):
            print(f"\n=== round {round_idx} ===")
            round_improved = False
            for method in NON_RTBP:
                candidates = _candidate_tunes(method, config.method_tunes[method])
                best_tune = config.method_tunes[method]
                best_corr = current_corr
                best_score = current_score
                print(f"\nmethod={method} candidates={len(candidates)}")
                for idx, candidate in enumerate(candidates, start=1):
                    trial = Config(method_tunes={k: MethodTune(**asdict(v)) for k, v in config.method_tunes.items()})
                    trial.method_tunes[method] = candidate
                    _, corr = _run_once(
                        config=trial,
                        snapshot=snapshot,
                        output_root=trials_root / f"round_{round_idx}" / method / f"cand_{idx:02d}",
                        make_plots=False,
                    )
                    score = _score_global(corr)
                    tag = ""
                    if score > best_score + 1e-12:
                        best_tune = candidate
                        best_corr = corr
                        best_score = score
                        tag = " *"
                    rtbp_s = corr["RTBP"]["SBP"]
                    rtbp_d = corr["RTBP"]["DBP"]
                    rtbp_p = corr["RTBP"]["PP"]
                    cand_s = corr[method]["SBP"]
                    cand_d = corr[method]["DBP"]
                    cand_p = corr[method]["PP"]
                    print(
                        f"  cand {idx:02d} score={score:.5f} "
                        f"{method}=({cand_s:.4f},{cand_d:.4f},{cand_p:.4f}) "
                        f"RTBP=({rtbp_s:.4f},{rtbp_d:.4f},{rtbp_p:.4f}) "
                        f"blend={candidate.lag_blend:.2f} calib={candidate.calib_windows} weights={candidate.pp_weights} "
                        f"sign+={candidate.sign_positive_only} lag={candidate.lag_mode} proj={candidate.projection_blend:.2f}{tag}"
                    )

                if best_tune != config.method_tunes[method]:
                    config.method_tunes[method] = best_tune
                    current_corr = best_corr
                    current_score = best_score
                    round_improved = True
                    print(
                        f"selected {method}: blend={best_tune.lag_blend:.2f} "
                        f"calib={best_tune.calib_windows} weights={best_tune.pp_weights} sign+={best_tune.sign_positive_only} "
                        f"lag={best_tune.lag_mode} proj={best_tune.projection_blend:.2f}"
                    )
                else:
                    print(f"selected {method}: keep current")

            if not round_improved:
                print("no improvement in this round -> stop")
                break

        final_dir, final_corr = _run_once(
            config=config,
            snapshot=snapshot,
            output_root=args.final_root,
            make_plots=bool(args.with_plots),
        )
        print("\nfinal_dir", final_dir)
        print("final_rtbp", final_corr["RTBP"])
        print("final_nonrtbp", {method: final_corr[method] for method in NON_RTBP})

        payload = {
            "baseline_dir": str(baseline_dir),
            "final_dir": str(final_dir),
            "baseline_corr": baseline_corr,
            "final_corr": final_corr,
            "selected_config": {
                method: asdict(tune) for method, tune in config.method_tunes.items()
            },
        }
        summary_path = Path(final_dir) / "nonrtbp_overtake_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("summary", summary_path)
    finally:
        _restore(snapshot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
