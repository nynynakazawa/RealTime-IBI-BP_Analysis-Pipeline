from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


METHODS = (
    ("RTBP", "M1_SBP", "M1_DBP", "M1_output_valid", "M1_reject_reason"),
    ("SinBP_D", "M2_SBP", "M2_DBP", "M2_output_valid", "M2_reject_reason"),
    ("SinBP_M", "M3_SBP", "M3_DBP", "M3_output_valid", "M3_reject_reason"),
)
MAX_ABS_TIME_DELTA_MS = 350.0


def _collect_merged_csvs(root: Path) -> list[Path]:
    return sorted(root.glob("*/**/*_merged.csv"))


def _to_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _to_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _row_passes_global_filters(row: dict[str, str]) -> bool:
    ref_sbp = _to_float(row.get("ref_SBP"))
    ref_dbp = _to_float(row.get("ref_DBP"))
    if np.isnan(ref_sbp) or np.isnan(ref_dbp):
        return False
    abs_dt = _to_float(row.get("abs_time_delta_ms"))
    if np.isfinite(abs_dt) and abs_dt > MAX_ABS_TIME_DELTA_MS:
        return False
    artifact = _to_int(row.get("artifact_flag"))
    return artifact == 0


def _fit_method(rows: list[dict[str, str]], sbp_col: str, dbp_col: str, valid_col: str, reject_col: str) -> dict[str, float]:
    map_errors: list[float] = []
    pp_errors: list[float] = []
    for row in rows:
        if _to_int(row.get(valid_col)) != 1:
            continue
        if row.get(reject_col, "missing").strip() != "ok":
            continue
        sbp = _to_float(row.get(sbp_col))
        dbp = _to_float(row.get(dbp_col))
        ref_sbp = _to_float(row.get("ref_SBP"))
        ref_dbp = _to_float(row.get("ref_DBP"))
        if any(np.isnan(v) for v in (sbp, dbp, ref_sbp, ref_dbp)):
            continue
        map_est = (sbp + 2.0 * dbp) / 3.0
        pp_est = sbp - dbp
        map_ref = (ref_sbp + 2.0 * ref_dbp) / 3.0
        pp_ref = ref_sbp - ref_dbp
        map_errors.append(map_ref - map_est)
        pp_errors.append(pp_ref - pp_est)
    if not map_errors:
        return {}
    return {
        "n": len(map_errors),
        "map_a": float(np.mean(map_errors)),
        "map_b": 1.0,
        "pp_a": float(np.mean(pp_errors)),
        "pp_b": 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit shared BP postprocess coefficients from realtime sessions.")
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Data" / "realtime_sessions",
        help="Directory containing realtime session folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory to save the fitted coefficient JSON",
    )
    args = parser.parse_args()

    merged_paths = _collect_merged_csvs(args.sessions_root)
    if not merged_paths:
        raise SystemExit("no merged csv files found")

    rows: list[dict[str, str]] = []
    for path in merged_paths:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if _row_passes_global_filters(row):
                    rows.append(row)
    if not rows:
        raise SystemExit("no usable rows with reference values found")

    fitted = {}
    for method_name, sbp_col, dbp_col, valid_col, reject_col in METHODS:
        result = _fit_method(rows, sbp_col, dbp_col, valid_col, reject_col)
        if result:
            fitted[method_name] = result

    payload = {
        "generated_at": datetime.now().isoformat(),
        "sessions_root": str(args.sessions_root),
        "max_abs_time_delta_ms": MAX_ABS_TIME_DELTA_MS,
        "methods": fitted,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"bp_postprocess_coefficients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
