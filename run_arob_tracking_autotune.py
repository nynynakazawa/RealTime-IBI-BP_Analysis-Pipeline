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
from typing import Iterable

from AROB import pipeline
from current_direction import AROB_TRACKING_ROOT


TRACK_METHODS: tuple[str, ...] = ("RTBP", "SinBP_D", "SinBP_D_PPShapeC", "SinBP_M")
TRACK_TARGETS: tuple[str, ...] = ("SBP", "DBP")
PRIMARY_WINDOW_SECONDS = 20
ALT_PP_WEIGHTS = (0.25, 0.375, 0.375)


@dataclass
class MethodTune:
    lag_blend: float
    calib_windows: int
    pp_weights: tuple[float, float, float]
    sign_mode: str  # "base" or "positive_only"
    lag_mode: str  # "base", "wide", "med", "narrow", "xnarrow"


@dataclass
class TuneConfig:
    method_tunes: dict[str, MethodTune]


@dataclass
class EvalResult:
    output_dir: Path
    objective: float
    mean_method_corr: float
    min_method_corr: float
    mean_nonrtbp_corr: float
    min_nonrtbp_corr: float
    amp_score: float
    nonrtbp_amp_score: float
    method_corr: dict[str, float]
    detail: dict[str, dict[str, dict[str, float]]]


@dataclass
class Snapshot:
    lag_blend: dict[str, float]
    calib: dict[str, int]
    pp_weights: dict[str, tuple[float, float, float]]
    signs_by_method: dict[str, tuple[float, ...]]
    lag_candidates_by_method: dict[str, tuple[int, ...]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Autotune AROB tracking parameters for RTBP / SinBP_D / "
            "SinBP_D_PPShapeC / SinBP_M with tracking-only objective."
        )
    )
    parser.add_argument("--rounds", type=int, default=2, help="Coordinate-descent rounds.")
    parser.add_argument("--with-plots", action="store_true", help="Generate plots for final selected run.")
    parser.add_argument(
        "--freeze-rtbp",
        action="store_true",
        default=True,
        help="Do not tune RTBP and focus optimization on non-RTBP methods.",
    )
    parser.add_argument(
        "--focus-non-rtbp",
        action="store_true",
        default=True,
        help="Use objective emphasizing non-RTBP methods (SinBP_D / PPShapeC / SinBP_M).",
    )
    parser.add_argument(
        "--trials-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "arob_tracking_autotune",
        help="Temporary root for no-plot trial runs.",
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=AROB_TRACKING_ROOT,
        help="Output root for final selected tracking run.",
    )
    return parser.parse_args()


def _snapshot() -> Snapshot:
    return Snapshot(
        lag_blend=deepcopy(pipeline.METHOD_LAG_BLEND),
        calib=deepcopy(pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD),
        pp_weights=deepcopy(pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD),
        signs_by_method=deepcopy(pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD),
        lag_candidates_by_method=deepcopy(pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD),
    )


def _restore(snapshot: Snapshot) -> None:
    pipeline.METHOD_LAG_BLEND = deepcopy(snapshot.lag_blend)
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD = deepcopy(snapshot.calib)
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = deepcopy(snapshot.pp_weights)
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = deepcopy(snapshot.signs_by_method)
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = deepcopy(snapshot.lag_candidates_by_method)


def _lag_mode_from_candidates(candidates: tuple[int, ...] | None) -> str:
    if candidates is None:
        return "base"
    options = {
        "wide": (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
        "med": (-4, -3, -2, -1, 0, 1, 2, 3, 4),
        "narrow": (-3, -2, -1, 0, 1, 2, 3),
        "xnarrow": (-2, -1, 0, 1, 2),
    }
    for mode, values in options.items():
        if tuple(candidates) == values:
            return mode
    return "base"


def _lag_candidates_for_mode(mode: str) -> tuple[int, ...] | None:
    if mode == "base":
        return None
    mapping = {
        "wide": (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
        "med": (-4, -3, -2, -1, 0, 1, 2, 3, 4),
        "narrow": (-3, -2, -1, 0, 1, 2, 3),
        "xnarrow": (-2, -1, 0, 1, 2),
    }
    return mapping.get(mode)


def _base_sign_mode(method: str, snapshot: Snapshot) -> str:
    signs = snapshot.signs_by_method.get(method)
    if signs is None:
        return "base"
    if tuple(signs) == (1.0,):
        return "positive_only"
    return "base"


def _read_centered_metrics(path: Path) -> dict[str, dict[str, tuple[float, float, float]]]:
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("window_seconds") != str(PRIMARY_WINDOW_SECONDS):
                continue
            method = row.get("method", "")
            target = row.get("target", "")
            if method not in TRACK_METHODS or target not in TRACK_TARGETS:
                continue
            pred_delta = row.get("pred_delta", "")
            ref_delta = row.get("ref_delta", "")
            if not pred_delta or not ref_delta:
                continue
            try:
                x = float(pred_delta)
                y = float(ref_delta)
            except ValueError:
                continue
            local = grouped.setdefault(method, {}).setdefault(
                target, {"n": 0.0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
            )
            local["n"] += 1.0
            local["sx"] += x
            local["sy"] += y
            local["sxx"] += x * x
            local["syy"] += y * y
            local["sxy"] += x * y

    result: dict[str, dict[str, tuple[float, float, float]]] = {}
    for method, targets in grouped.items():
        result[method] = {}
        for target, acc in targets.items():
            n = acc["n"]
            if n <= 1:
                result[method][target] = (math.nan, math.nan, math.nan)
                continue
            mx = acc["sx"] / n
            my = acc["sy"] / n
            vx = max(acc["sxx"] / n - mx * mx, 0.0)
            vy = max(acc["syy"] / n - my * my, 0.0)
            cov = acc["sxy"] / n - mx * my
            corr = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else math.nan
            pred_std = math.sqrt(vx)
            ref_std = math.sqrt(vy)
            result[method][target] = (corr, pred_std, ref_std)
    return result


def _evaluate_output(output_dir: Path, focus_non_rtbp: bool) -> EvalResult:
    centered_path = output_dir / "centered_window_samples.csv"
    stats = _read_centered_metrics(centered_path)
    method_corr: dict[str, float] = {}
    detail: dict[str, dict[str, dict[str, float]]] = {}
    amp_values: list[float] = []
    nonrtbp_amp_values: list[float] = []
    nonrtbp_methods = {"SinBP_D", "SinBP_D_PPShapeC", "SinBP_M"}
    for method in TRACK_METHODS:
        per_target = stats.get(method, {})
        corr_values: list[float] = []
        detail[method] = {}
        for target in TRACK_TARGETS:
            corr, pred_std, ref_std = per_target.get(target, (math.nan, math.nan, math.nan))
            ratio = pred_std / ref_std if ref_std and ref_std > 0.0 else math.nan
            if math.isfinite(ratio):
                amp_values.append(min(max(ratio, 0.0), 1.0))
                if method in nonrtbp_methods:
                    nonrtbp_amp_values.append(min(max(ratio, 0.0), 1.0))
            if math.isfinite(corr):
                corr_values.append(corr)
            detail[method][target] = {
                "delta_corr": corr,
                "pred_delta_std": pred_std,
                "ref_delta_std": ref_std,
                "std_ratio_clipped": min(max(ratio, 0.0), 1.0) if math.isfinite(ratio) else math.nan,
            }
        method_corr[method] = sum(corr_values) / len(corr_values) if corr_values else math.nan

    valid_method_corr = [value for value in method_corr.values() if math.isfinite(value)]
    mean_method_corr = sum(valid_method_corr) / len(valid_method_corr) if valid_method_corr else math.nan
    min_method_corr = min(valid_method_corr) if valid_method_corr else math.nan
    nonrtbp_corr_values = [
        method_corr[m] for m in nonrtbp_methods if m in method_corr and math.isfinite(method_corr[m])
    ]
    mean_nonrtbp_corr = sum(nonrtbp_corr_values) / len(nonrtbp_corr_values) if nonrtbp_corr_values else math.nan
    min_nonrtbp_corr = min(nonrtbp_corr_values) if nonrtbp_corr_values else math.nan
    amp_score = sum(amp_values) / len(amp_values) if amp_values else 0.0
    nonrtbp_amp_score = sum(nonrtbp_amp_values) / len(nonrtbp_amp_values) if nonrtbp_amp_values else 0.0

    if focus_non_rtbp:
        objective = (
            0.65 * min_nonrtbp_corr
            + 0.30 * mean_nonrtbp_corr
            + 0.08 * nonrtbp_amp_score
            + 0.05 * mean_method_corr
        )
    else:
        objective = (0.6 * min_method_corr) + (0.4 * mean_method_corr) + (0.05 * amp_score)
    return EvalResult(
        output_dir=output_dir,
        objective=objective,
        mean_method_corr=mean_method_corr,
        min_method_corr=min_method_corr,
        mean_nonrtbp_corr=mean_nonrtbp_corr,
        min_nonrtbp_corr=min_nonrtbp_corr,
        amp_score=amp_score,
        nonrtbp_amp_score=nonrtbp_amp_score,
        method_corr=method_corr,
        detail=detail,
    )


def _apply_config(config: TuneConfig, snapshot: Snapshot) -> None:
    lag = deepcopy(snapshot.lag_blend)
    calib = deepcopy(snapshot.calib)
    weights = deepcopy(snapshot.pp_weights)
    signs = deepcopy(snapshot.signs_by_method)
    lag_candidates = deepcopy(snapshot.lag_candidates_by_method)
    for method, tune in config.method_tunes.items():
        lag[method] = tune.lag_blend
        calib[method] = tune.calib_windows
        weights[method] = tune.pp_weights
        if tune.sign_mode == "positive_only":
            signs[method] = (1.0,)
        else:
            # Use global sign candidates for this method by removing method override.
            signs.pop(method, None)
        mode_candidates = _lag_candidates_for_mode(tune.lag_mode)
        if mode_candidates is None:
            lag_candidates.pop(method, None)
        else:
            lag_candidates[method] = mode_candidates

    pipeline.METHOD_LAG_BLEND = lag
    pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_BY_METHOD = calib
    pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS_BY_METHOD = weights
    pipeline.SESSION_ALIGNMENT_SIGNS_BY_METHOD = signs
    pipeline.SESSION_ALIGNMENT_LAG_CANDIDATES_BY_METHOD = lag_candidates


def _clone_config(config: TuneConfig) -> TuneConfig:
    return TuneConfig(method_tunes={k: MethodTune(**asdict(v)) for k, v in config.method_tunes.items()})


def _base_config(snapshot: Snapshot) -> TuneConfig:
    tunes: dict[str, MethodTune] = {}
    for method in TRACK_METHODS:
        tunes[method] = MethodTune(
            lag_blend=float(snapshot.lag_blend.get(method, 1.0)),
            calib_windows=int(snapshot.calib.get(method, pipeline.SESSION_ALIGNMENT_CALIB_WINDOWS_DEFAULT)),
            pp_weights=tuple(snapshot.pp_weights.get(method, pipeline.SESSION_ALIGNMENT_PP_SCORE_WEIGHTS)),
            sign_mode=_base_sign_mode(method, snapshot),
            lag_mode=_lag_mode_from_candidates(snapshot.lag_candidates_by_method.get(method)),
        )
    return TuneConfig(method_tunes=tunes)


def _candidate_tunes(method: str, current: MethodTune) -> list[MethodTune]:
    candidates: list[MethodTune] = [
        current,
        MethodTune(0.85, current.calib_windows, current.pp_weights, current.sign_mode, current.lag_mode),
        MethodTune(0.95, current.calib_windows, current.pp_weights, current.sign_mode, current.lag_mode),
        MethodTune(1.0, current.calib_windows, current.pp_weights, current.sign_mode, current.lag_mode),
        MethodTune(current.lag_blend, 10, current.pp_weights, current.sign_mode, current.lag_mode),
        MethodTune(current.lag_blend, current.calib_windows, ALT_PP_WEIGHTS, current.sign_mode, current.lag_mode),
        MethodTune(0.85, 10, current.pp_weights, current.sign_mode, current.lag_mode),
        MethodTune(0.95, current.calib_windows, ALT_PP_WEIGHTS, current.sign_mode, current.lag_mode),
    ]
    if method in ("SinBP_D", "SinBP_D_PPShapeC"):
        candidates.append(
            MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, "positive_only", current.lag_mode)
        )
    if method == "SinBP_D_PPShapeC":
        candidates.extend(
            [
                MethodTune(0.70, 8, current.pp_weights, "base", current.lag_mode),
                MethodTune(0.75, 8, current.pp_weights, "base", current.lag_mode),
                MethodTune(0.80, 8, current.pp_weights, "base", current.lag_mode),
                MethodTune(0.90, 8, current.pp_weights, "base", current.lag_mode),
                MethodTune(1.10, 8, current.pp_weights, "base", current.lag_mode),
                MethodTune(0.90, 10, ALT_PP_WEIGHTS, "base", current.lag_mode),
                MethodTune(0.80, 10, ALT_PP_WEIGHTS, "base", current.lag_mode),
            ]
        )
    if method == "SinBP_D":
        candidates.extend(
            [
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "wide"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "med"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "narrow"),
            ]
        )
    if method == "SinBP_D_PPShapeC":
        candidates.extend(
            [
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "base"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "wide"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "med"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "narrow"),
            ]
        )
    if method == "SinBP_M":
        candidates.extend(
            [
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "base"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "med"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "narrow"),
                MethodTune(current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, "xnarrow"),
            ]
        )
    # De-duplicate while preserving order.
    unique: list[MethodTune] = []
    seen: set[tuple[float, int, tuple[float, float, float], str, str]] = set()
    for c in candidates:
        key = (c.lag_blend, c.calib_windows, c.pp_weights, c.sign_mode, c.lag_mode)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    # Always include current tune.
    current_key = (current.lag_blend, current.calib_windows, current.pp_weights, current.sign_mode, current.lag_mode)
    if current_key not in seen:
        unique.insert(0, current)
    return unique


def _run_once(
    config: TuneConfig,
    snapshot: Snapshot,
    output_root: Path,
    make_plots: bool,
    focus_non_rtbp: bool,
) -> EvalResult:
    _apply_config(config, snapshot)
    outputs = pipeline.run_tracking_analysis(output_root=output_root, make_plots=make_plots)
    return _evaluate_output(outputs.output_dir, focus_non_rtbp=focus_non_rtbp)


def _is_better_for_method(candidate: EvalResult, current: EvalResult, method: str) -> bool:
    eps = 1e-9
    method_cand = candidate.method_corr.get(method, math.nan)
    method_curr = current.method_corr.get(method, math.nan)
    if math.isfinite(method_cand) and math.isfinite(method_curr):
        if method_cand > method_curr + eps:
            return True
        if method_cand < method_curr - eps:
            return False
    if candidate.objective > current.objective + eps:
        return True
    if candidate.objective < current.objective - eps:
        return False
    return candidate.min_method_corr > current.min_method_corr + eps


def _write_summary(path: Path, baseline: EvalResult, tuned: EvalResult, config: TuneConfig, rounds: int) -> None:
    payload = {
        "rounds": rounds,
        "baseline": {
            "output_dir": str(baseline.output_dir),
            "objective": baseline.objective,
            "mean_method_corr": baseline.mean_method_corr,
            "min_method_corr": baseline.min_method_corr,
            "mean_nonrtbp_corr": baseline.mean_nonrtbp_corr,
            "min_nonrtbp_corr": baseline.min_nonrtbp_corr,
            "amp_score": baseline.amp_score,
            "nonrtbp_amp_score": baseline.nonrtbp_amp_score,
            "method_corr": baseline.method_corr,
            "detail": baseline.detail,
        },
        "tuned": {
            "output_dir": str(tuned.output_dir),
            "objective": tuned.objective,
            "mean_method_corr": tuned.mean_method_corr,
            "min_method_corr": tuned.min_method_corr,
            "mean_nonrtbp_corr": tuned.mean_nonrtbp_corr,
            "min_nonrtbp_corr": tuned.min_nonrtbp_corr,
            "amp_score": tuned.amp_score,
            "nonrtbp_amp_score": tuned.nonrtbp_amp_score,
            "method_corr": tuned.method_corr,
            "detail": tuned.detail,
        },
        "selected_config": {
            method: {
                "lag_blend": tune.lag_blend,
                "calib_windows": tune.calib_windows,
                "pp_weights": tune.pp_weights,
                "sign_mode": tune.sign_mode,
                "lag_mode": tune.lag_mode,
            }
            for method, tune in config.method_tunes.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_eval(tag: str, result: EvalResult) -> None:
    print(
        f"{tag}: objective={result.objective:.4f} "
        f"mean_corr={result.mean_method_corr:.4f} min_corr={result.min_method_corr:.4f} "
        f"mean_nonrtbp={result.mean_nonrtbp_corr:.4f} min_nonrtbp={result.min_nonrtbp_corr:.4f} "
        f"amp={result.amp_score:.4f}"
    )
    for method in TRACK_METHODS:
        print(f"  {method:16s} corr={result.method_corr.get(method, math.nan):.4f}")


def main() -> int:
    args = parse_args()
    snapshot = _snapshot()
    trials_root = args.trials_root
    if trials_root.exists():
        shutil.rmtree(trials_root)
    trials_root.mkdir(parents=True, exist_ok=True)

    config = _base_config(snapshot)

    baseline_eval = _run_once(
        config=config,
        snapshot=snapshot,
        output_root=trials_root / "baseline",
        make_plots=False,
        focus_non_rtbp=bool(args.focus_non_rtbp),
    )
    _print_eval("baseline", baseline_eval)
    current_eval = baseline_eval

    try:
        for round_idx in range(1, max(int(args.rounds), 1) + 1):
            print(f"\n=== round {round_idx} ===")
            round_improved = False
            method_order = list(TRACK_METHODS)
            if args.freeze_rtbp:
                method_order = [m for m in method_order if m != "RTBP"]
            for method in method_order:
                base_tune = config.method_tunes[method]
                best_tune = base_tune
                best_eval = current_eval
                candidates = _candidate_tunes(method, base_tune)
                print(f"\nmethod={method} candidates={len(candidates)}")
                for cand_idx, cand in enumerate(candidates, start=1):
                    trial = _clone_config(config)
                    trial.method_tunes[method] = cand
                    result = _run_once(
                        config=trial,
                        snapshot=snapshot,
                        output_root=trials_root / f"round_{round_idx}" / method / f"cand_{cand_idx:02d}",
                        make_plots=False,
                        focus_non_rtbp=bool(args.focus_non_rtbp),
                    )
                    improved = _is_better_for_method(result, best_eval, method)
                    print(
                        f"  cand {cand_idx:02d} "
                        f"lag={cand.lag_blend:.2f} calib={cand.calib_windows} "
                        f"weights={cand.pp_weights} sign={cand.sign_mode} lag_mode={cand.lag_mode} "
                        f"corr[{method}]={result.method_corr.get(method, math.nan):.4f} "
                        f"obj={result.objective:.4f}"
                        + (" *" if improved else "")
                    )
                    if improved:
                        best_tune = cand
                        best_eval = result
                if best_tune != base_tune:
                    config.method_tunes[method] = best_tune
                    current_eval = best_eval
                    round_improved = True
                    print(
                        f"selected {method}: lag={best_tune.lag_blend:.2f} "
                        f"calib={best_tune.calib_windows} weights={best_tune.pp_weights} "
                        f"sign={best_tune.sign_mode} lag_mode={best_tune.lag_mode}"
                    )
                else:
                    print(f"selected {method}: keep current")
            if not round_improved:
                print("\nno improvement in this round -> stop")
                break

        final_eval = _run_once(
            config=config,
            snapshot=snapshot,
            output_root=args.final_root,
            make_plots=bool(args.with_plots),
            focus_non_rtbp=bool(args.focus_non_rtbp),
        )
        _print_eval("final", final_eval)

        summary_path = final_eval.output_dir / "autotune_summary.json"
        _write_summary(summary_path, baseline_eval, final_eval, config, int(args.rounds))
        print(f"autotune_summary={summary_path}")
        print("\nselected_config:")
        for method in TRACK_METHODS:
            tune = config.method_tunes[method]
            print(
                f"  {method}: lag_blend={tune.lag_blend:.2f}, "
                f"calib_windows={tune.calib_windows}, pp_weights={tune.pp_weights}, "
                f"sign={tune.sign_mode}, lag_mode={tune.lag_mode}"
            )

    finally:
        _restore(snapshot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
