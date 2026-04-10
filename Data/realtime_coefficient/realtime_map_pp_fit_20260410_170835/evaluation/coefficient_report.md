# Realtime Coefficient Report

- sessions_root: /Users/nakazawa/Desktop/Nozawa Lab/RealTime-IBI&BP/Analysis/Data/realtime_sessions
- ridge_alpha: 1.0
- baseline_ridge_alpha: 10.0
- rich_baseline_ridge_alpha: 10.0
- baseline_shrinkage: 1.0
- dynamic_blend_gain_map: 0.5
- dynamic_blend_gain_pp: 0.5
- initial_baseline_beats: 30
- filters: abs_time_delta_ms<=350.0, output_valid, reject_reason=ok, artifact_flag=0, ref_pp_inlier
- CNAP usage: offline training/evaluation labels only; not used by the runtime app.
- smartphone_initial_baseline: experimental smartphone-only baseline adaptation from initial beat features; validate LOSO before shipping.
- smartphone_rich_baseline: richer smartphone-only baseline adaptation using initial raw/smoothed estimates, coefficient terms, beat stats, and method-specific morphology features; validate LOSO before shipping.
- smartphone_rich_dynamic_blend: rich baseline anchor plus refit MAP/PP deltas from the initial dynamic anchor; intended to keep per-subject baseline while restoring real-time up/down movement.
- smartphone_shared_sinbpd_baseline: shared baseline from SinBP_D rich features, with method-specific deltas; this is the most physiologically explainable smartphone-only baseline candidate.

## Outputs

- session_evaluation_summary.csv
- session_evaluation_summary.json
- current_app_evaluation_summary.csv
- refit_evaluation_summary.csv
- smartphone_initial_baseline_evaluation_summary.csv
- smartphone_rich_baseline_evaluation_summary.csv
- smartphone_rich_dynamic_blend_evaluation_summary.csv
- smartphone_shared_sinbpd_baseline_evaluation_summary.csv
- smartphone_initial_baseline_loso_evaluation_summary.csv
- smartphone_rich_baseline_loso_evaluation_summary.csv
- smartphone_rich_dynamic_blend_loso_evaluation_summary.csv
- smartphone_shared_sinbpd_baseline_loso_evaluation_summary.csv
- loso_evaluation_summary.csv
- fixed_coefficients.json

## Plots

- plots/eitaro_20260410_155023_sbp_timeseries.png
- plots/eitaro_20260410_155023_dbp_timeseries.png
- plots/jin_20260409_182834_sbp_timeseries.png
- plots/jin_20260409_182834_dbp_timeseries.png
- plots/yu_20260409_182115_sbp_timeseries.png
- plots/yu_20260409_182115_dbp_timeseries.png
- plots/zawa_20260409_181339_sbp_timeseries.png
- plots/zawa_20260409_181339_dbp_timeseries.png
- plots/all_sbp_scatter.png
- plots/all_dbp_scatter.png
