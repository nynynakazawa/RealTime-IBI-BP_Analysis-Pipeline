from __future__ import annotations

import argparse

from AROB.pipeline import run_tracking_analysis
from current_direction import CURRENT_AROB_TRACKING_RUN, PAPER_METHOD_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run AROB session-centered BP tracking analysis "
            f"for {', '.join(PAPER_METHOD_NAMES)}. "
            f"Current reference run: {CURRENT_AROB_TRACKING_RUN.name}"
        )
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation and compute metrics only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_tracking_analysis(make_plots=not args.no_plots)
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
