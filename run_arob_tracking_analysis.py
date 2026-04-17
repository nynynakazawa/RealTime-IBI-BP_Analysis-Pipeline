from __future__ import annotations

import argparse
from pathlib import Path

from AROB.pipeline import run_tracking_analysis
from current_direction import AROB_TRACKING_ROOT, CURRENT_AROB_TRACKING_RUN, PAPER_METHOD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AROB session-centered BP tracking analysis "
            f"for {', '.join(PAPER_METHOD_NAMES)}. "
            f"Current reference run: {CURRENT_AROB_TRACKING_RUN.name}"
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=AROB_TRACKING_ROOT,
        help="Root directory where tracking_eval_<timestamp> will be created.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation and compute metrics only.")
    parser.add_argument(
        "--enable-tracking-projection",
        action="store_true",
        help="Enable AROB-only centered projection adjustment before window aggregation.",
    )
    parser.add_argument(
        "--enable-window-lag-alignment",
        action="store_true",
        help="Enable AROB-only window lag/sign/gain alignment after aggregation.",
    )
    parser.add_argument(
        "--session-id",
        dest="session_ids",
        action="append",
        default=[],
        help="Run only specific realtime session id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--past",
        action="store_true",
        help="Include sessions under Analysis/Data/realtime_sessions/past/ as well.",
    )
    parser.add_argument(
        "--include-eonly",
        action="store_true",
        help="Add SinBP_D_EOnly as an ablation series alongside the default paper methods.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method_names = PAPER_METHOD_NAMES + (("SinBP_D_EOnly",) if args.include_eonly else ())
    outputs = run_tracking_analysis(
        output_root=args.output_root,
        make_plots=not args.no_plots,
        enable_tracking_projection=args.enable_tracking_projection,
        enable_window_lag_alignment=args.enable_window_lag_alignment,
        include_past=args.past,
        session_ids=tuple(args.session_ids) if args.session_ids else None,
        method_names=method_names,
    )
    print(f"output_dir={outputs.output_dir}")
    print(f"summary_csv={outputs.summary_path}")
    print(f"summary_all_csv={outputs.summary_all_path}")
    print(f"pp_summary_csv={outputs.pp_summary_path}")
    print(f"pp_term_csv={outputs.pp_term_path}")
    print(f"pp_culprit_csv={outputs.pp_culprit_path}")
    print(f"pp_feature_screening_csv={outputs.pp_feature_screening_path}")
    print(f"pp_feature_culprit_csv={outputs.pp_feature_culprit_path}")
    print(f"pp_feature_coefficients_csv={outputs.pp_feature_coefficients_path}")
    print(f"pp_report_md={outputs.pp_report_path}")
    print(f"report_md={outputs.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
